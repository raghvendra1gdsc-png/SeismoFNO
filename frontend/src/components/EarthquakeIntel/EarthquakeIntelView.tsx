import React, { useState, useMemo } from "react";
import { Activity } from "lucide-react";
import type { IndianEarthquake, PeerRecord } from "../../api/digitalTwinApi";
import { IndiaSeismicMap } from "./IndiaSeismicMap";

interface EarthquakeIntelViewProps {
  indianCatalog: IndianEarthquake[];
  peerRecords: PeerRecord[];
  selectedEarthquakeId: string;
  onSelectEarthquake: (id: string, name: string) => void;
  onNavigateTab?: (tab: string) => void;
}

export const EarthquakeIntelView: React.FC<EarthquakeIntelViewProps> = ({
  indianCatalog,
  peerRecords,
  selectedEarthquakeId,
  onSelectEarthquake,
  onNavigateTab,
}) => {
  const [activeSubTab, setActiveSubTab] = useState<"india_map" | "peer_library">("india_map");
  const [appliedNotification, setAppliedNotification] = useState<string | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string>(
    indianCatalog.length > 0 ? indianCatalog[0].id : "IND-2001-BHUJ"
  );
  const minMagnitude = 6.0;
  const [selectedZone, setSelectedZone] = useState<string>("ALL");

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
    <div className="p-4 md:p-8 space-y-6 font-sans text-[#E8E8DE] bg-research-desk min-h-screen select-none">
      {/* Workspace Header */}
      <header className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4 border-b border-white/[0.08] pb-4">
        <div>
          <div className="flex items-center gap-3 text-[11px] text-[#82928B] font-mono mb-1">
            <span className="flex items-center gap-1.5 text-[#73E6B5]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5]" />
              SEISMOFNO HAZARD INTEL
            </span>
            <span>·</span>
            <span>IS 1893:2016 COMPLIANT</span>
            <span>·</span>
            <span>PEER NGA-WEST2</span>
          </div>
          <h1 className="text-xl md:text-2xl font-light text-[#E8E8DE] tracking-tight">
            Earthquake Intelligence & India Seismic Zonation
          </h1>
          <p className="text-xs text-[#82928B] mt-0.5">
            Verified Indian historical seismicity catalog and Pacific Earthquake Engineering Research ground motions.
          </p>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex bg-[#0E1B17] p-0.5 rounded border border-white/[0.08] text-xs font-mono shrink-0">
          <button
            onClick={() => setActiveSubTab("india_map")}
            className={`px-3 py-1.5 rounded transition cursor-pointer font-medium ${
              activeSubTab === "india_map"
                ? "bg-[#17483A] text-[#73E6B5] font-semibold"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            India Earthquake Map
          </button>
          <button
            onClick={() => setActiveSubTab("peer_library")}
            className={`px-3 py-1.5 rounded transition cursor-pointer font-medium ${
              activeSubTab === "peer_library"
                ? "bg-[#17483A] text-[#73E6B5] font-semibold"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            PEER Records ({peerRecords.length})
          </button>
        </div>
      </header>

      {activeSubTab === "india_map" ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Map Column (7 Cols) - Powered by Leaflet + MapTiler / Carto */}
          <div className="lg:col-span-7 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#82928B] uppercase tracking-wider text-[10px]">
                Historical Epicenters & Fault Zonation
              </span>
              <div className="flex items-center space-x-2 text-[#82928B]">
                <span className="text-[10px]">ZONATION:</span>
                <select
                  value={selectedZone}
                  onChange={(e) => setSelectedZone(e.target.value)}
                  className="bg-[#0E1B17] border border-white/[0.08] rounded px-2 py-1 text-[11px] text-[#E8E8DE] outline-none cursor-pointer"
                >
                  <option value="ALL">All Zones (M6.0+)</option>
                  <option value="Zone V">Zone V (Very Severe)</option>
                  <option value="Zone IV">Zone IV (Severe)</option>
                  <option value="Zone III">Zone III (Moderate)</option>
                </select>
              </div>
            </div>

            {/* Interactive Leaflet Map of India */}
            <div className="h-[520px] lg:h-[580px] rounded-lg overflow-hidden shadow-2xl">
              <IndiaSeismicMap
                events={filteredIndianEvents}
                selectedEventId={selectedEventId}
                onSelectEvent={setSelectedEventId}
                selectedZone={selectedZone}
              />
            </div>
          </div>

          {/* Selected Event Details & Simulation Pairing (5 Cols) */}
          <div className="lg:col-span-5 space-y-4">
            {activeIndianEvent ? (
              <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-5 space-y-4">
                <div className="border-b border-white/[0.06] pb-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono text-[#73E6B5] font-semibold tracking-wider uppercase">
                      {activeIndianEvent.region_state}
                    </span>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded border font-semibold ${
                        activeIndianEvent.is1893_zone === "Zone V"
                          ? "bg-[#E35D5D]/15 text-[#E35D5D] border-[#E35D5D]/30"
                          : activeIndianEvent.is1893_zone === "Zone IV"
                          ? "bg-[#D6B56D]/15 text-[#D6B56D] border-[#D6B56D]/30"
                          : "bg-[#73E6B5]/15 text-[#73E6B5] border-[#73E6B5]/30"
                      }`}
                    >
                      {activeIndianEvent.is1893_zone}
                    </span>
                  </div>
                  <h2 className="text-xl font-light text-[#E8E8DE] mt-1">
                    {activeIndianEvent.name} ({activeIndianEvent.year})
                  </h2>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div className="bg-[#07110F] p-2.5 rounded border border-white/[0.06]">
                    <span className="text-[#82928B] text-[10px] uppercase block">Moment Magnitude</span>
                    <span className="text-base font-bold text-[#E8E8DE] font-mono">
                      Mw {activeIndianEvent.magnitude.toFixed(1)}
                    </span>
                  </div>
                  <div className="bg-[#07110F] p-2.5 rounded border border-white/[0.06]">
                    <span className="text-[#82928B] text-[10px] uppercase block">Focal Depth</span>
                    <span className="text-base font-bold text-[#E8E8DE] font-mono">
                      {activeIndianEvent.depth_km} km
                    </span>
                  </div>
                  <div className="bg-[#07110F] p-2.5 rounded border border-white/[0.06]">
                    <span className="text-[#82928B] text-[10px] uppercase block">Epicenter Coordinates</span>
                    <span className="text-[#73E6B5] font-semibold font-mono">
                      {activeIndianEvent.latitude.toFixed(3)}°N, {activeIndianEvent.longitude.toFixed(3)}°E
                    </span>
                  </div>
                  <div className="bg-[#07110F] p-2.5 rounded border border-white/[0.06]">
                    <span className="text-[#82928B] text-[10px] uppercase block">Seismic Zone Factor</span>
                    <span className="text-[#E8E8DE] font-semibold font-mono">
                      {activeIndianEvent.is1893_zone === "Zone V"
                        ? "Z = 0.36 (Very Severe)"
                        : activeIndianEvent.is1893_zone === "Zone IV"
                        ? "Z = 0.24 (Severe)"
                        : "Z = 0.16 (Moderate)"}
                    </span>
                  </div>
                </div>

                <div className="space-y-2 text-xs font-sans">
                  <div>
                    <span className="text-[#82928B] font-mono text-[10px] uppercase tracking-wider block">
                      Tectonic Regime
                    </span>
                    <p className="text-[#A4B3AC] mt-0.5 leading-relaxed text-xs">
                      {activeIndianEvent.tectonic_regime}
                    </p>
                  </div>
                  <div>
                    <span className="text-[#82928B] font-mono text-[10px] uppercase tracking-wider block">
                      Historical Impact & Ground Motion Notes
                    </span>
                    <p className="text-[#A4B3AC] mt-0.5 leading-relaxed text-xs">
                      {activeIndianEvent.historical_notes}
                    </p>
                  </div>
                </div>

                {/* Digital-Twin Load Action */}
                <div className="pt-3 border-t border-white/[0.06] space-y-2">
                  <button
                    onClick={() => {
                      onSelectEarthquake(
                        activeIndianEvent.id,
                        `${activeIndianEvent.name} (${activeIndianEvent.year})`
                      );
                      setAppliedNotification(activeIndianEvent.name);
                    }}
                    className="w-full py-2.5 bg-[#17483A] hover:bg-[#1C5746] text-[#73E6B5] border border-[#73E6B5]/30 font-mono text-xs font-semibold rounded flex items-center justify-center space-x-2 cursor-pointer transition shadow-md"
                  >
                    <Activity size={14} />
                    <span>SET AS ACTIVE EXCITATION FOR DIGITAL TWIN</span>
                  </button>

                  {appliedNotification === activeIndianEvent.name && (
                    <div className="p-3 bg-[#73E6B5]/10 border border-[#73E6B5]/30 rounded flex items-center justify-between text-xs font-mono">
                      <span className="text-[#73E6B5] flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-[#73E6B5] animate-ping" />
                        Active excitation loaded & recomputed
                      </span>
                      {onNavigateTab && (
                        <button
                          onClick={() => onNavigateTab("structural_twin")}
                          className="px-2.5 py-1 bg-[#73E6B5] text-[#07110F] font-bold rounded cursor-pointer hover:bg-[#5cd4a2] transition text-[11px]"
                        >
                          View Structural Twin →
                        </button>
                      )}
                    </div>
                  )}

                  <p className="text-[10px] font-mono text-[#82928B] text-center">
                    Applies proxy kinematic acceleration record scaled to scenario PGA.
                  </p>
                </div>
              </div>
            ) : null}

            {/* List of All Indian Events */}
            <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-4 space-y-2 max-h-64 overflow-y-auto">
              <div className="text-xs font-mono text-[#82928B] uppercase tracking-wider mb-2">
                Cataloged Indian Events ({filteredIndianEvents.length})
              </div>
              <div className="space-y-1">
                {filteredIndianEvents.map((evt) => (
                  <div
                    key={evt.id}
                    onClick={() => setSelectedEventId(evt.id)}
                    className={`p-2 rounded border text-xs font-mono flex items-center justify-between cursor-pointer transition ${
                      evt.id === selectedEventId
                        ? "bg-[#17483A]/40 border-[#73E6B5]/50 text-[#73E6B5] font-semibold"
                        : "bg-[#07110F] border-white/[0.05] text-[#82928B] hover:text-[#E8E8DE] hover:bg-[#101D19]"
                    }`}
                  >
                    <div>
                      <span className="font-semibold">{evt.name}</span>
                      <span className="text-[10px] opacity-70 ml-1.5 font-normal">({evt.year})</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-[#E8E8DE] text-[11px]">Mw {evt.magnitude.toFixed(1)}</span>
                      <span
                        className={
                          evt.is1893_zone === "Zone V"
                            ? "text-[#E35D5D]"
                            : evt.is1893_zone === "Zone IV"
                            ? "text-[#D6B56D]"
                            : "text-[#73E6B5]"
                        }
                      >
                        {evt.is1893_zone}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* PEER Library Tab */
        <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/[0.06] pb-3">
            <div>
              <h3 className="text-sm font-semibold font-mono text-[#E8E8DE]">
                PEER NGA-WEST2 ACCELEROGRAM LIBRARY ({peerRecords.length} RECORDS)
              </h3>
              <p className="text-xs text-[#82928B] mt-0.5 font-sans">
                Click any record to set it as active seismic excitation for instant surrogate simulation.
              </p>
            </div>

            {onNavigateTab && (
              <button
                onClick={() => onNavigateTab("structural_twin")}
                className="px-3 py-1.5 bg-[#17483A] hover:bg-[#1C5746] text-[#73E6B5] border border-[#73E6B5]/30 font-mono text-xs font-semibold rounded cursor-pointer transition shrink-0"
              >
                Go to Structural Simulator →
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {peerRecords.map((rec) => {
              const isSelected = rec.filename === selectedEarthquakeId;
              return (
                <div
                  key={rec.filename}
                  onClick={() => {
                    onSelectEarthquake(rec.filename, rec.earthquake_name || rec.filename);
                    setAppliedNotification(rec.earthquake_name || rec.filename);
                  }}
                  className={`p-3 rounded border text-xs font-mono cursor-pointer transition ${
                    isSelected
                      ? "bg-[#17483A]/50 border-[#73E6B5] text-[#73E6B5] shadow-md"
                      : "bg-[#07110F] border-white/[0.06] text-[#82928B] hover:text-[#E8E8DE] hover:bg-[#101D19]"
                  }`}
                >
                  <div className="flex items-center justify-between font-semibold">
                    <span className="truncate text-[#E8E8DE]">{rec.earthquake_name || rec.filename}</span>
                    <span className="text-[#82928B] text-[10px] shrink-0 ml-1 font-mono">{rec.year}</span>
                  </div>
                  <div className="mt-1 flex justify-between text-[11px]">
                    <span>PGA: <strong className="text-[#73E6B5]">{rec.raw_pga_g?.toFixed(3) ?? "—"}g</strong></span>
                    <span>dt: {rec.raw_dt ?? "—"}s</span>
                  </div>
                  <div className="mt-1 text-[10px] text-[#82928B] truncate">
                    Station: {rec.station}
                  </div>
                  {isSelected && (
                    <div className="mt-2 text-[10px] font-mono text-[#73E6B5] font-bold flex items-center justify-between pt-1 border-t border-[#73E6B5]/30">
                      <span>✓ ACTIVE EXCITATION</span>
                      {onNavigateTab && (
                        <span
                          onClick={(e) => {
                            e.stopPropagation();
                            onNavigateTab("structural_twin");
                          }}
                          className="underline cursor-pointer hover:text-white"
                        >
                          Simulate →
                        </span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default EarthquakeIntelView;
