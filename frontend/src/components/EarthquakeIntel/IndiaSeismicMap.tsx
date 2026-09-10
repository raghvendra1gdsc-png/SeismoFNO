import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { IndianEarthquake } from "../../api/digitalTwinApi";
import { Compass } from "lucide-react";

interface IndiaSeismicMapProps {
  events: IndianEarthquake[];
  selectedEventId: string;
  onSelectEvent: (id: string) => void;
  selectedZone: string;
}

// Himalayan Frontal Thrust & Main Boundary Thrust fault coordinates
const HIMALAYAN_FAULT_COORDS: [number, number][] = [
  [34.5, 73.5], // Kashmir syntaxis
  [33.8, 74.8],
  [32.8, 76.0], // Himachal
  [31.2, 77.8],
  [30.4, 79.2], // Uttarakhand / Garhwal
  [29.3, 81.0], // Western Nepal border
  [28.4, 83.5], // Central Nepal
  [27.6, 86.2], // Eastern Nepal
  [27.3, 88.6], // Sikkim
  [27.2, 91.5], // Bhutan
  [27.8, 93.8], // Arunachal Pradesh
  [28.2, 95.5],
  [28.0, 96.8], // Mishmi Hills / Assam syntaxis
];

export const IndiaSeismicMap: React.FC<IndiaSeismicMapProps> = ({
  events,
  selectedEventId,
  onSelectEvent,
  selectedZone,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);
  const faultLayerRef = useRef<L.Polyline | null>(null);

  const [provider, setProvider] = useState<"maptiler" | "carto">("maptiler");

  // Providers tile URL definitions
  const TILE_URLS = {
    maptiler: "https://api.maptiler.com/maps/dataviz-dark/{z}/{x}/{y}.png?key=JAI9tztyvk89wmyqsDWx",
    carto: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?api_key=cb1_2yru_1_be975c21c3c99af922bcf25c",
  };

  const ATTRIBUTIONS = {
    maptiler: '&copy; <a href="https://www.maptiler.com/copyright/" target="_blank" class="text-[#73E6B5]">MapTiler</a> &copy; OpenStreetMap',
    carto: '&copy; <a href="https://carto.com/attributions" target="_blank" class="text-[#73E6B5]">CARTO</a> &copy; OpenStreetMap',
  };

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapInstanceRef.current) return;

    // Center on India [22.8, 82.0]
    const map = L.map(mapContainerRef.current, {
      center: [22.8, 82.0],
      zoom: 4.8,
      minZoom: 3.5,
      maxZoom: 10,
      zoomControl: false,
      attributionControl: false,
    });

    // Custom dark zoom control in top right
    L.control.zoom({ position: "topright" }).addTo(map);

    // Initial tile layer
    const initialTileLayer = L.tileLayer(TILE_URLS.maptiler, {
      attribution: ATTRIBUTIONS.maptiler,
      maxZoom: 18,
    }).addTo(map);
    tileLayerRef.current = initialTileLayer;

    // Markers layer group
    const markersGroup = L.layerGroup().addTo(map);
    markersLayerRef.current = markersGroup;

    // Fault line polyline
    const faultLine = L.polyline(HIMALAYAN_FAULT_COORDS, {
      color: "#E35D5D",
      weight: 2.2,
      dashArray: "6, 6",
      opacity: 0.85,
    }).addTo(map);
    faultLine.bindTooltip("Himalayan Frontal Thrust (HFT) / Main Boundary Thrust (MBT)", {
      className: "seismo-map-tooltip",
      direction: "top",
    });
    faultLayerRef.current = faultLine;

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update tile provider when user switches
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (tileLayerRef.current) {
      map.removeLayer(tileLayerRef.current);
    }

    const newTileLayer = L.tileLayer(TILE_URLS[provider], {
      attribution: ATTRIBUTIONS[provider],
      maxZoom: 18,
      subdomains: provider === "carto" ? "abcd" : "a",
    }).addTo(map);

    tileLayerRef.current = newTileLayer;
  }, [provider]);

  // Update Markers when events, selectedZone, or selectedEventId changes
  useEffect(() => {
    const markersGroup = markersLayerRef.current;
    if (!markersGroup) return;

    markersGroup.clearLayers();

    const filteredEvents =
      selectedZone && selectedZone !== "ALL"
        ? events.filter((e) => e.is1893_zone === selectedZone)
        : events;

    filteredEvents.forEach((evt) => {
      const isSelected = evt.id === selectedEventId;

      // Color coding based on IS 1893 Zone
      const color =
        evt.is1893_zone === "Zone V"
          ? "#E35D5D" // Very Severe (red)
          : evt.is1893_zone === "Zone IV"
          ? "#D6B56D" // Severe (amber)
          : "#73E6B5"; // Moderate (mint)

      const radius = Math.max(7, (evt.magnitude - 5.5) * 5.5);

      // Circle Marker
      const circleMarker = L.circleMarker([evt.latitude, evt.longitude], {
        radius: isSelected ? radius + 4 : radius,
        fillColor: color,
        fillOpacity: isSelected ? 0.95 : 0.75,
        color: isSelected ? "#FFFFFF" : color,
        weight: isSelected ? 2.5 : 1.2,
      });

      // HTML Tooltip on hover
      circleMarker.bindTooltip(
        `<div style="font-family: monospace; font-size: 11px; padding: 2px 4px;">
          <strong style="color: #73E6B5;">${evt.name} (${evt.year})</strong><br/>
          <span>Magnitude: <strong style="color: #E8E8DE;">Mw ${evt.magnitude.toFixed(1)}</strong></span><br/>
          <span>Zone: <strong style="color: ${color};">${evt.is1893_zone}</strong></span>
        </div>`,
        {
          direction: "top",
          className: "seismo-map-tooltip",
          offset: [0, -radius],
        }
      );

      // Click to select
      circleMarker.on("click", () => {
        onSelectEvent(evt.id);
        const map = mapInstanceRef.current;
        if (map) {
          map.panTo([evt.latitude, evt.longitude], { animate: true, duration: 0.5 });
        }
      });

      markersGroup.addLayer(circleMarker);
    });
  }, [events, selectedEventId, onSelectEvent, selectedZone]);

  // Recenter map to whole India view
  const handleRecenter = () => {
    const map = mapInstanceRef.current;
    if (map) {
      map.setView([22.8, 82.0], 4.8, { animate: true });
    }
  };

  return (
    <div className="relative w-full h-full min-h-[500px] lg:min-h-[560px] rounded-lg overflow-hidden bg-[#07110F] border border-white/[0.08] select-none">
      {/* Map container */}
      <div ref={mapContainerRef} className="w-full h-full min-h-[500px] lg:min-h-[560px]" />

      {/* TOP-LEFT HUD: Title & Provider Badge */}
      <div className="absolute top-3 left-3 z-[1000] pointer-events-none space-y-1 font-sans">
        <div className="flex items-center gap-2 px-2.5 py-1 bg-[#07110F]/90 border border-white/[0.08] rounded backdrop-blur-sm">
          <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5]" />
          <span className="text-xs font-semibold text-[#E8E8DE]">
            India Seismotectonic & Epicenter Map
          </span>
          <span className="text-[10px] font-mono text-[#82928B]">
            IS 1893:2016
          </span>
        </div>
      </div>

      {/* TOP-RIGHT CONTROLS: Provider Switcher & Recenter */}
      <div className="absolute top-3 right-12 z-[1000] flex items-center gap-1.5 font-mono text-[10px]">
        <div className="flex items-center bg-[#07110F]/90 border border-white/[0.08] p-0.5 rounded backdrop-blur-sm">
          <button
            onClick={() => setProvider("maptiler")}
            className={`px-2 py-0.5 rounded transition cursor-pointer ${
              provider === "maptiler"
                ? "bg-[#17483A] text-[#73E6B5] font-bold"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
            title="MapTiler Vector Dark Tiles"
          >
            MapTiler Dark
          </button>
          <button
            onClick={() => setProvider("carto")}
            className={`px-2 py-0.5 rounded transition cursor-pointer ${
              provider === "carto"
                ? "bg-[#17483A] text-[#73E6B5] font-bold"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
            title="Carto Dark Matter Tiles"
          >
            Carto Dark
          </button>
        </div>

        <button
          onClick={handleRecenter}
          className="p-1.5 bg-[#07110F]/90 hover:bg-[#101D19] border border-white/[0.08] rounded text-[#82928B] hover:text-[#E8E8DE] transition cursor-pointer backdrop-blur-sm"
          title="Recenter Map of India"
        >
          <Compass size={12} />
        </button>
      </div>

      {/* BOTTOM-LEFT: IS 1893 ZONATION LEGEND */}
      <div className="absolute bottom-3 left-3 z-[1000] bg-[#07110F]/90 border border-white/[0.08] p-2.5 rounded text-[10px] font-mono text-[#82928B] space-y-1.5 backdrop-blur-sm">
        <div className="font-semibold text-[#E8E8DE] uppercase text-[9px] tracking-wider">
          IS 1893 Seismic Severity
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#E35D5D] border border-white/20" />
          <span>Zone V (PGA &gt; 0.36g)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#D6B56D] border border-white/20" />
          <span>Zone IV (PGA 0.24g)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#73E6B5] border border-white/20" />
          <span>Zone III (PGA 0.16g)</span>
        </div>
        <div className="flex items-center gap-2 pt-1 border-t border-white/[0.06]">
          <span className="w-3.5 h-0.5 bg-[#E35D5D] inline-block" />
          <span className="text-[9px]">Himalayan Frontal Thrust</span>
        </div>
      </div>

      {/* BOTTOM-RIGHT: Tile Attribution Tag */}
      <div className="absolute bottom-2 right-2 z-[1000] text-[9px] font-mono text-[#82928B]/70 bg-[#07110F]/80 px-1.5 py-0.5 rounded pointer-events-none">
        {provider === "maptiler" ? "MapTiler Dark" : "CARTO Dark"} · OpenStreetMap
      </div>
    </div>
  );
};
