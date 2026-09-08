import React, { useState, useMemo } from "react";
import { Activity } from "lucide-react";
import type { IndianEarthquake, PeerRecord } from "../../api/digitalTwinApi";

interface EarthquakeIntelViewProps {
  indianCatalog: IndianEarthquake[];
  peerRecords: PeerRecord[];
  selectedEarthquakeId: string;
  onSelectEarthquake: (id: string, name: string) => void;
}

// Verified Indian coastline and boundary polyline points (lat/lon)
// Extends from 68°E to 98°E, 6°N to 37°N
const INDIA_BOUNDARY_POLYGON = [
  // Northwest (Kashmir / Ladakh)
  { lat: 37.05, lon: 74.5 },
  { lat: 35.5, lon: 77.0 },
  { lat: 34.0, lon: 79.0 },
  { lat: 32.5, lon: 79.0 },
  // Himalayas & North
  { lat: 31.0, lon: 81.0 },
  { lat: 30.0, lon: 80.5 },
  { lat: 28.5, lon: 83.5 },
  { lat: 27.5, lon: 88.0 }, // Sikkim
  { lat: 28.0, lon: 89.0 },
  { lat: 27.5, lon: 92.0 }, // Arunachal
  { lat: 29.0, lon: 96.5 },
  { lat: 28.0, lon: 97.0 },
  // East / Northeast
  { lat: 26.5, lon: 95.0 },
  { lat: 24.5, lon: 93.5 },
  { lat: 23.0, lon: 93.0 },
  { lat: 22.0, lon: 92.0 }, // Bangladesh border
  { lat: 21.5, lon: 89.0 }, // Sunderbans
  // East Coast
  { lat: 20.0, lon: 86.5 }, // Odisha
  { lat: 17.5, lon: 83.0 }, // Andhra Pradesh
  { lat: 15.5, lon: 80.0 },
  { lat: 13.0, lon: 80.3 }, // Chennai
  { lat: 10.0, lon: 79.8 },
  // Southern Tip
  { lat: 8.08, lon: 77.55 }, // Kanyakumari
  // West Coast
  { lat: 8.5, lon: 76.9 }, // Kerala
  { lat: 10.0, lon: 76.2 },
  { lat: 13.0, lon: 74.8 }, // Karnataka
  { lat: 15.5, lon: 73.8 }, // Goa
  { lat: 19.0, lon: 72.8 }, // Mumbai
  { lat: 21.0, lon: 72.6 }, // Gujarat Gulf of Khambhat
  { lat: 20.8, lon: 70.8 }, // Saurashtra
  { lat: 22.3, lon: 69.0 },
  { lat: 23.5, lon: 68.2 }, // Kachchh westernmost
  { lat: 24.5, lon: 71.0 }, // Rann of Kutch
  // Rajasthan / Punjab border
  { lat: 26.0, lon: 70.5 },
  { lat: 28.0, lon: 72.0 },
  { lat: 30.5, lon: 74.0 },
  { lat: 32.5, lon: 75.0 },
  { lat: 35.0, lon: 74.0 },
  { lat: 37.05, lon: 74.5 },
];

export const EarthquakeIntelView: React.FC<EarthquakeIntelViewProps> = ({
  indianCatalog,
  peerRecords,
  selectedEarthquakeId,
  onSelectEarthquake,
}) => {
  const [activeSubTab, setActiveSubTab] = useState<"india_map" | "peer_library">("india_map");
  const [selectedEventId, setSelectedEventId] = useState<string>(
    indianCatalog.length > 0 ? indianCatalog[0].id : "IND-2001-BHUJ"
  );
  const minMagnitude = 6.0;
  const [selectedZone, setSelectedZone] = useState<string>("ALL");

  // Coordinate projection from (lon, lat) to SVG canvas (600 x 600)
  const project = (lon: number, lat: number) => {
    const minLon = 67.0;
    const maxLon = 98.0;
    const minLat = 7.0;
    const maxLat = 38.0;

    const x = ((lon - minLon) / (maxLon - minLon)) * 560 + 20;
    const y = (1.0 - (lat - minLat) / (maxLat - minLat)) * 560 + 20;
    return { x, y };
  };

  const boundaryPath = useMemo(() => {
    return (
      INDIA_BOUNDARY_POLYGON.map((pt, idx) => {
        const { x, y } = project(pt.lon, pt.lat);
        return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      }).join(" ") + " Z"
    );
  }, []);

  const filteredIndianEvents = useMemo(() => {
    return indianCatalog.filter((e) => {
      const matchMag = e.magnitude >= minMagnitude;
      const matchZone = selectedZone === "ALL" || e.is1893_zone === selectedZone;
      return matchMag && matchZone;
    });
  }, [indianCatalog, minMagnitude, selectedZone]);

  const activeIndianEvent = useMemo(() => {
    return (
      indianCatalog.find((e) => e.id === selectedEventId) ||
      (indianCatalog.length > 0 ? indianCatalog[0] : null)
    );
  }, [indianCatalog, selectedEventId]);

  return (
    <div className="p-6 space-y-6 font-sans">
      {/* Workspace Header */}
      <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-4 bg-white p-5 rounded border">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#161616]">
              EARTHQUAKE INTELLIGENCE & INDIA SEISMIC MAP
            </h1>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] font-semibold">
              IS 1893:2016 COMPLIANT
            </span>
          </div>
          <p className="text-xs text-[#525252] mt-1 font-sans">
            Verified Indian historical seismicity catalog and global PEER NGA-West2 acceleration ground motions.
          </p>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex bg-[#F4F4F4] p-1 rounded border border-[#E0E0E0] text-xs font-mono">
          <button
            onClick={() => setActiveSubTab("india_map")}
            className={`px-3 py-1.5 rounded cursor-pointer transition font-medium ${
              activeSubTab === "india_map"
                ? "bg-white text-[#0F62FE] font-bold shadow-sm"
                : "text-[#525252] hover:text-[#161616]"
            }`}
          >
            India Earthquake Map
          </button>
          <button
            onClick={() => setActiveSubTab("peer_library")}
            className={`px-3 py-1.5 rounded cursor-pointer transition font-medium ${
              activeSubTab === "peer_library"
                ? "bg-white text-[#0F62FE] font-bold shadow-sm"
                : "text-[#525252] hover:text-[#161616]"
            }`}
          >
            PEER NGA-West2 Records ({peerRecords.length})
          </button>
        </div>
      </div>

      {activeSubTab === "india_map" ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Map Canvas Column (7 Cols) */}
          <div className="lg:col-span-7 bg-white border border-[#E0E0E0] rounded p-5 flex flex-col">
            <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-2 mb-3 text-xs font-mono">
              <span className="text-[#161616] font-bold">HISTORICAL EPICENTERS & HAZARD ZONATION</span>
              <div className="flex items-center space-x-2 text-[#525252]">
                <span>Filter:</span>
                <select
                  value={selectedZone}
                  onChange={(e) => setSelectedZone(e.target.value)}
                  className="bg-[#F4F4F4] border border-[#E0E0E0] rounded px-2 py-1 text-[11px] text-[#161616] cursor-pointer"
                >
                  <option value="ALL">All Zones</option>
                  <option value="Zone V">Zone V (Very Severe)</option>
                  <option value="Zone IV">Zone IV (Severe)</option>
                  <option value="Zone III">Zone III (Moderate)</option>
                </select>
              </div>
            </div>

            {/* Interactive SVG Geographic Map: Topographic/Hazard map styling */}
            <div className="relative w-full aspect-square max-h-[560px] bg-[#FFFFFF] rounded border border-[#E0E0E0] overflow-hidden flex items-center justify-center p-2">
              <svg className="w-full h-full" viewBox="0 0 600 600">
                {/* Background Grid Pattern */}
                {[10, 20, 30].map((lat) => {
                  const { y } = project(67, lat);
                  return (
                    <line
                      key={`lat-${lat}`}
                      x1="20"
                      y1={y}
                      x2="580"
                      y2={y}
                      stroke="#E0E0E0"
                      strokeDasharray="4 4"
                    />
                  );
                })}
                {[70, 80, 90].map((lon) => {
                  const { x } = project(lon, 7);
                  return (
                    <line
                      key={`lon-${lon}`}
                      x1={x}
                      y1="20"
                      x2={x}
                      y2="580"
                      stroke="#E0E0E0"
                      strokeDasharray="4 4"
                    />
                  );
                })}

                {/* India Outline Polygon — #F4F4F4 land with #161616 outline */}
                <path
                  d={boundaryPath}
                  fill="#F4F4F4"
                  stroke="#161616"
                  strokeWidth="1.5"
                  className="transition-colors hover:fill-[#EBEBEB]"
                />

                {/* Major Fault Lines / Tectonic Features (Himalayan Frontal Thrust) */}
                <path
                  d={`M ${project(74, 34).x} ${project(74, 34).y} Q ${project(82, 29).x} ${project(82, 29).y} ${project(96, 28).x} ${project(96, 28).y}`}
                  fill="none"
                  stroke="#DA1E28"
                  strokeWidth="2"
                  strokeDasharray="4 3"
                />

                {/* Epicenter Markers: red -> amber -> yellow severity scale */}
                {filteredIndianEvents.map((evt) => {
                  const { x, y } = project(evt.longitude, evt.latitude);
                  const isSelected = evt.id === activeIndianEvent?.id;
                  const radius = Math.max(5, (evt.magnitude - 5.5) * 6);

                  // Severity color mapping
                  const dotColor =
                    evt.is1893_zone === "Zone V"
                      ? "#DA1E28" // red-60
                      : evt.is1893_zone === "Zone IV"
                      ? "#B28600" // amber-60
                      : "#F1C21B"; // yellow-30

                  return (
                    <g
                      key={evt.id}
                      className="cursor-pointer group"
                      onClick={() => setSelectedEventId(evt.id)}
                    >
                      {/* Pulse ring for selected event */}
                      {isSelected && (
                        <circle
                          cx={x}
                          cy={y}
                          r={radius + 6}
                          fill="none"
                          stroke="#0F62FE"
                          strokeWidth="2"
                          opacity="0.75"
                        />
                      )}

                      <circle
                        cx={x}
                        cy={y}
                        r={radius}
                        fill={dotColor}
                        stroke="#161616"
                        strokeWidth="1"
                        opacity={isSelected ? 1.0 : 0.85}
                        className="transition-transform group-hover:scale-125"
                      />

                      {/* City/Name Label for large events */}
                      {(evt.magnitude >= 7.5 || isSelected) && (
                        <text
                          x={x + radius + 3}
                          y={y + 3}
                          fontSize="10"
                          fontFamily="IBM Plex Mono"
                          fontWeight={isSelected ? "bold" : "500"}
                          fill="#161616"
                          className="select-none pointer-events-none"
                        >
                          {evt.name.split(" ")[0]} ({evt.magnitude.toFixed(1)})
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>

              {/* Topographic Map Legend */}
              <div className="absolute bottom-3 left-3 bg-white/95 border border-[#E0E0E0] p-2 rounded text-[10px] font-mono text-[#525252] space-y-1 backdrop-blur-sm shadow-sm">
                <div className="font-bold text-[#161616] uppercase">IS 1893:2016 Severity</div>
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#DA1E28] border border-[#161616]" />
                  <span>Zone V (PGA &gt; 0.36g)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#B28600] border border-[#161616]" />
                  <span>Zone IV (PGA 0.24g)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#F1C21B] border border-[#161616]" />
                  <span>Zone III (PGA 0.16g)</span>
                </div>
                <div className="flex items-center space-x-2 pt-0.5 border-t border-[#E0E0E0]">
                  <span className="w-3 h-0.5 bg-[#DA1E28] inline-block" />
                  <span>Himalayan Frontal Thrust</span>
                </div>
              </div>
            </div>
          </div>

          {/* Selected Event Details & Simulation Pairing (5 Cols) */}
          <div className="lg:col-span-5 space-y-4">
            {activeIndianEvent ? (
              <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
                <div className="border-b border-[#E0E0E0] pb-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono text-[#0F62FE] font-bold tracking-widest uppercase">
                      {activeIndianEvent.region_state}
                    </span>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded border font-semibold ${
                        activeIndianEvent.is1893_zone === "Zone V"
                          ? "bg-[#FFD7D9] text-[#DA1E28] border-[#FF8389]"
                          : activeIndianEvent.is1893_zone === "Zone IV"
                          ? "bg-[#FFF8E1] text-[#B28600] border-[#F1C21B]"
                          : "bg-[#F4F4F4] text-[#525252] border-[#E0E0E0]"
                      }`}
                    >
                      {activeIndianEvent.is1893_zone}
                    </span>
                  </div>
                  <h2 className="text-lg font-bold font-mono text-[#161616] mt-1">
                    {activeIndianEvent.name} ({activeIndianEvent.year})
                  </h2>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                  <div className="bg-[#F4F4F4] p-2.5 rounded border border-[#E0E0E0]">
                    <span className="text-[#525252] text-[10px] uppercase block">Moment Magnitude</span>
                    <span className="text-base font-bold text-[#161616] font-mono">
                      Mw {activeIndianEvent.magnitude.toFixed(1)}
                    </span>
                  </div>
                  <div className="bg-[#F4F4F4] p-2.5 rounded border border-[#E0E0E0]">
                    <span className="text-[#525252] text-[10px] uppercase block">Focal Depth</span>
                    <span className="text-base font-bold text-[#161616] font-mono">
                      {activeIndianEvent.depth_km} km
                    </span>
                  </div>
                  <div className="bg-[#F4F4F4] p-2.5 rounded border border-[#E0E0E0]">
                    <span className="text-[#525252] text-[10px] uppercase block">Epicenter Coordinates</span>
                    <span className="text-[#161616] font-semibold font-mono">
                      {activeIndianEvent.latitude.toFixed(3)}°N, {activeIndianEvent.longitude.toFixed(3)}°E
                    </span>
                  </div>
                  <div className="bg-[#F4F4F4] p-2.5 rounded border border-[#E0E0E0]">
                    <span className="text-[#525252] text-[10px] uppercase block">Seismic Zone Factor</span>
                    <span className="text-[#161616] font-semibold font-mono">
                      {activeIndianEvent.is1893_zone === "Zone V"
                        ? "Z = 0.36"
                        : activeIndianEvent.is1893_zone === "Zone IV"
                        ? "Z = 0.24"
                        : "Z = 0.16"}
                    </span>
                  </div>
                </div>

                <div className="space-y-2 text-xs font-sans">
                  <div>
                    <span className="text-[#525252] font-bold text-[10px] font-mono uppercase">Tectonic Regime:</span>
                    <p className="text-[#525252] mt-0.5 leading-relaxed">{activeIndianEvent.tectonic_regime}</p>
                  </div>
                  <div>
                    <span className="text-[#525252] font-bold text-[10px] font-mono uppercase">Historical Impact:</span>
                    <p className="text-[#525252] mt-0.5 leading-relaxed">{activeIndianEvent.historical_notes}</p>
                  </div>
                </div>

                {/* Digital-Twin Load Action */}
                <div className="pt-3 border-t border-[#E0E0E0]">
                  <button
                    onClick={() =>
                      onSelectEarthquake(
                        activeIndianEvent.id,
                        `${activeIndianEvent.name} (${activeIndianEvent.year})`
                      )
                    }
                    className="w-full py-2.5 bg-[#0F62FE] hover:bg-[#0353E9] text-white font-mono text-xs font-bold rounded flex items-center justify-center space-x-2 cursor-pointer transition shadow-sm"
                  >
                    <Activity size={14} />
                    <span>SET AS ACTIVE EXCITATION FOR DIGITAL TWIN</span>
                  </button>
                  <p className="text-[10px] font-mono text-[#525252] text-center mt-2">
                    Applies proxy kinematic acceleration record scaled to scenario PGA.
                  </p>
                </div>
              </div>
            ) : null}

            {/* List of All Indian Events */}
            <div className="bg-white border border-[#E0E0E0] rounded p-4 space-y-2 max-h-60 overflow-y-auto">
              <div className="text-xs font-mono font-bold text-[#525252] uppercase mb-2">
                All Cataloged Indian Events ({filteredIndianEvents.length})
              </div>
              {filteredIndianEvents.map((evt) => (
                <div
                  key={evt.id}
                  onClick={() => setSelectedEventId(evt.id)}
                  className={`p-2 rounded border text-xs font-mono flex items-center justify-between cursor-pointer transition ${
                    evt.id === selectedEventId
                      ? "bg-[#EDF5FF] border-[#0F62FE] text-[#0F62FE] font-bold"
                      : "bg-[#F4F4F4] border-[#E0E0E0] text-[#161616] hover:bg-[#EBEBEB]"
                  }`}
                >
                  <div>
                    <span className="font-semibold">{evt.name}</span>
                    <span className="text-[10px] text-[#525252] ml-1.5 font-normal">({evt.year})</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-[#525252] text-[11px]">Mw {evt.magnitude.toFixed(1)}</span>
                    <span className="text-[10px] text-[#525252]">{evt.is1893_zone}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* PEER Library Tab */
        <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3">
            <div>
              <h3 className="text-sm font-bold font-mono text-[#161616]">
                PEER NGA-WEST2 ACCELEROGRAM LIBRARY ({peerRecords.length} RECORDS)
              </h3>
              <p className="text-xs text-[#525252] mt-0.5">
                Authoritative Pacific Earthquake Engineering Research Center database ground motions.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {peerRecords.map((rec) => (
              <div
                key={rec.filename}
                onClick={() => onSelectEarthquake(rec.filename, rec.earthquake_name || rec.filename)}
                className={`p-3 rounded border text-xs font-mono cursor-pointer transition ${
                  rec.filename === selectedEarthquakeId
                    ? "bg-[#EDF5FF] border-[#0F62FE] text-[#0F62FE]"
                    : "bg-[#F4F4F4] border-[#E0E0E0] text-[#161616] hover:bg-[#EBEBEB]"
                }`}
              >
                <div className="flex items-center justify-between font-bold">
                  <span className="truncate">{rec.earthquake_name || rec.filename}</span>
                  <span className="text-[#525252] text-[10px] shrink-0 ml-1 font-mono">{rec.year}</span>
                </div>
                <div className="mt-1 flex justify-between text-[11px] text-[#525252]">
                  <span>PGA: {rec.raw_pga_g?.toFixed(3) ?? "—"}g</span>
                  <span>dt: {rec.raw_dt ?? "—"}s</span>
                </div>
                <div className="mt-1 text-[10px] text-[#525252] truncate">
                  Station: {rec.station}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
