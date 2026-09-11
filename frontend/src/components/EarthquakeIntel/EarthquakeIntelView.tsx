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
    <div className="p-4 md:p-8 space-y-6 font-sans text-[#0F172A] min-h-screen select-none max-w-7xl mx-auto">
      {/* Workspace Header */}
      <header className="panel-workstation p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 text-[11px] text-[#64748B] font-mono mb-1">
            <span className="flex items-center gap-1.5 text-[#047857] font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-[#047857]" />
              SEISMOFNO HAZARD INTEL
            </span>
            <span>·</span>
            <span>IS 1893:2016 COMPLIANT</span>
            <span>·</span>
            <span>PEER NGA-WEST2</span>
          </div>
          <h1 className="text-xl md:text-2xl font-bold font-mono tracking-tight text-[#0F172A]">
            Earthquake Intelligence & India Seismic Zonation
          </h1>
          <p className="text-xs text-[#475569] mt-1 font-sans">
            Verified Indian historical seismicity catalog and Pacific Earthquake Engineering Research ground motions.
          </p>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex bg-[#F1F5F9] p-1 rounded border border-[#E2E8F0] text-xs font-mono shrink-0">
          <button
            onClick={() => setActiveSubTab("india_map")}
            className={`px-3 py-1.5 rounded transition cursor-pointer font-medium ${
              activeSubTab === "india_map"
                ? "bg-white text-[#0F172A] font-bold shadow-xs border border-[#E2E8F0]"
                : "text-[#64748B] hover:text-[#0F172A]"
            }`}
          >
            India Earthquake Map
          </button>
          <button
            onClick={() => setActiveSubTab("peer_library")}
            className={`px-3 py-1.5 rounded transition cursor-pointer font-medium ${
              activeSubTab === "peer_library"
                ? "bg-white text-[#0F172A] font-bold shadow-xs border border-[#E2E8F0]"
                : "text-[#64748B] hover:text-[#0F172A]"
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
              <span className="text-[#64748B] uppercase tracking-wider text-[10px] font-semibold">
                Historical Epicenters & Fault Zonation
              </span>
              <div className="flex items-center space-x-2 text-[#64748B]">
                <span className="text-[10px] font-semibold">ZONATION:</span>
                <select
                  value={selectedZone}
                  onChange={(e) => setSelectedZone(e.target.value)}
                  className="bg-white border border-[#CBD5E1] rounded px-2 py-1 text-[11px] text-[#0F172A] outline-none cursor-pointer focus:border-[#047857]"
                >
                  <option value="ALL">All Zones (M6.0+)</option>
                  <option value="Zone V">Zone V (Very Severe)</option>
                  <option value="Zone IV">Zone IV (Severe)</option>
                  <option value="Zone III">Zone III (Moderate)</option>
                </select>
              </div>
            </div>

            {/* Interactive Leaflet Map of India */}
            <div className="h-[520px] lg:h-[580px] rounded-lg overflow-hidden shadow-xs border border-[#E2E8F0]">
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
              <div className="panel-workstation p-6 space-y-4">
                <div className="border-b border-[#E2E8F0] pb-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono text-[#047857] font-bold tracking-wider uppercase">
                      {activeIndianEvent.region_state}
                    </span>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded border font-semibold ${
                        activeIndianEvent.is1893_zone === "Zone V"
                          ? "bg-[#FEF2F2] text-[#DC2626] border-[#FECACA]"
                          : activeIndianEvent.is1893_zone === "Zone IV"
                          ? "bg-[#FFFBEB] text-[#B45309] border-[#FDE68A]"
                          : "bg-[#ECFDF5] text-[#047857] border-[#A7F3D0]"
                      }`}
                    >
                      {activeIndianEvent.is1893_zone}
                    </span>
                  </div>
                  <h2 className="text-xl font-semibold text-[#0F172A] mt-1 font-sans">
                    {activeIndianEvent.name} ({activeIndianEvent.year})
                  </h2>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div className="bg-[#F8FAFC] p-3 rounded border border-[#E2E8F0]">
                    <span className="text-[#64748B] text-[10px] uppercase block font-medium">Moment Magnitude</span>
                    <span className="text-base font-bold text-[#0F172A] font-mono">
                      Mw {activeIndianEvent.magnitude.toFixed(1)}
                    </span>
                  </div>
                  <div className="bg-[#F8FAFC] p-3 rounded border border-[#E2E8F0]">
                    <span className="text-[#64748B] text-[10px] uppercase block font-medium">Focal Depth</span>
                    <span className="text-base font-bold text-[#0F172A] font-mono">
                      {activeIndianEvent.depth_km} km
                    </span>
                  </div>
                  <div className="bg-[#F8FAFC] p-3 rounded border border-[#E2E8F0]">
                    <span className="text-[#64748B] text-[10px] uppercase block font-medium">Epicenter Coordinates</span>
                    <span className="text-[#047857] font-semibold font-mono">
                      {activeIndianEvent.latitude.toFixed(3)}°N, {activeIndianEvent.longitude.toFixed(3)}°E
                    </span>
                  </div>
                  <div className="bg-[#F8FAFC] p-3 rounded border border-[#E2E8F0]">
                    <span className="text-[#64748B] text-[10px] uppercase block font-medium">Seismic Zone Factor</span>
                    <span className="text-[#0F172A] font-semibold font-mono">
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
                    <span className="text-[#64748B] font-mono text-[10px] uppercase tracking-wider block font-semibold">
                      Tectonic Regime
                    </span>
                    <p className="text-[#475569] mt-0.5 leading-relaxed text-xs">
                      {activeIndianEvent.tectonic_regime}
                    </p>
                  </div>
                  <div>
                    <span className="text-[#64748B] font-mono text-[10px] uppercase tracking-wider block font-semibold">
                      Historical Impact & Ground Motion Notes
                    </span>
                    <p className="text-[#475569] mt-0.5 leading-relaxed text-xs">
                      {activeIndianEvent.historical_notes}
                    </p>
                  </div>
                </div>

                {/* Digital-Twin Load Action */}
                <div className="pt-3 border-t border-[#E2E8F0] space-y-2">
                  <button
                    onClick={() => {
                      onSelectEarthquake(
                        activeIndianEvent.id,
                        `${activeIndianEvent.name} (${activeIndianEvent.year})`
                      );
                      setAppliedNotification(activeIndianEvent.name);
                    }}
                    className="btn-engineering w-full py-2.5 bg-[#047857] hover:bg-[#065F46] text-white font-mono text-xs font-bold rounded flex items-center justify-center space-x-2 cursor-pointer transition shadow-xs"
                  >
                    <Activity size={14} />
                    <span>SET AS ACTIVE EXCITATION FOR DIGITAL TWIN</span>
                  </button>

                  {appliedNotification === activeIndianEvent.name && (
                    <div className="p-3 bg-[#ECFDF5] border border-[#A7F3D0] rounded flex items-center justify-between text-xs font-mono">
                      <span className="text-[#047857] flex items-center gap-1.5 font-semibold">
                        <span className="w-2 h-2 rounded-full bg-[#047857]" />
                        Active excitation loaded & recomputed
                      </span>
                      {onNavigateTab && (
                        <button
                          onClick={() => onNavigateTab("structural_twin")}
                          className="px-2.5 py-1 bg-[#047857] text-white font-bold rounded cursor-pointer hover:bg-[#065F46] transition text-[11px]"
                        >
                          View Structural Twin →
                        </button>
                      )}
                    </div>
                  )}

                  <p className="text-[10px] font-mono text-[#64748B] text-center">
                    Applies proxy kinematic acceleration record scaled to scenario PGA.
                  </p>
                </div>
              </div>
            ) : null}

            {/* List of All Indian Events */}
            <div className="panel-workstation p-4 space-y-2 max-h-64 overflow-y-auto">
              <div className="text-xs font-mono text-[#64748B] uppercase tracking-wider mb-2 font-semibold">
                Cataloged Indian Events ({filteredIndianEvents.length})
              </div>
              <div className="space-y-1">
                {filteredIndianEvents.map((evt) => (
                  <div
                    key={evt.id}
                    onClick={() => setSelectedEventId(evt.id)}
                    className={`p-2 rounded border text-xs font-mono flex items-center justify-between cursor-pointer transition ${
                      evt.id === selectedEventId
                        ? "bg-[#ECFDF5] border-[#A7F3D0] text-[#047857] font-semibold"
                        : "bg-[#F8FAFC] border-[#E2E8F0] text-[#475569] hover:text-[#0F172A] hover:bg-[#F1F5F9]"
                    }`}
                  >
                    <div>
                      <span className="font-semibold text-[#0F172A]">{evt.name}</span>
                      <span className="text-[10px] text-[#64748B] ml-1.5 font-normal">({evt.year})</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-[#0F172A] text-[11px]">Mw {evt.magnitude.toFixed(1)}</span>
                      <span
                        className={
                          evt.is1893_zone === "Zone V"
                            ? "text-[#DC2626] font-semibold"
                            : evt.is1893_zone === "Zone IV"
                            ? "text-[#B45309] font-semibold"
                            : "text-[#047857] font-semibold"
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
        <div className="panel-workstation p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E2E8F0] pb-3">
            <div>
              <h3 className="text-sm font-semibold font-mono text-[#0F172A]">
                PEER NGA-WEST2 ACCELEROGRAM LIBRARY ({peerRecords.length} RECORDS)
              </h3>
              <p className="text-xs text-[#64748B] mt-0.5 font-sans">
                Click any record to set it as active seismic excitation for instant surrogate simulation.
              </p>
            </div>

            {onNavigateTab && (
              <button
                onClick={() => onNavigateTab("structural_twin")}
                className="btn-engineering px-3 py-1.5 bg-[#047857] hover:bg-[#065F46] text-white font-mono text-xs font-semibold rounded cursor-pointer transition shrink-0"
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
                      ? "bg-[#ECFDF5] border-[#047857] text-[#047857] shadow-xs"
                      : "bg-[#F8FAFC] border-[#E2E8F0] text-[#475569] hover:text-[#0F172A] hover:bg-white"
                  }`}
                >
                  <div className="flex items-center justify-between font-semibold">
                    <span className="truncate text-[#0F172A]">{rec.earthquake_name || rec.filename}</span>
                    <span className="text-[#64748B] text-[10px] shrink-0 ml-1 font-mono">{rec.year}</span>
                  </div>
                  <div className="mt-1 flex justify-between text-[11px]">
                    <span>PGA: <strong className="text-[#047857]">{rec.raw_pga_g?.toFixed(3) ?? "—"}g</strong></span>
                    <span className="text-[#64748B]">dt: {rec.raw_dt ?? "—"}s</span>
                  </div>
                  <div className="mt-1 text-[10px] text-[#64748B] truncate">
                    Station: {rec.station}
                  </div>
                  {isSelected && (
                    <div className="mt-2 text-[10px] font-mono text-[#047857] font-bold flex items-center justify-between pt-1 border-t border-[#A7F3D0]">
                      <span>✓ ACTIVE EXCITATION</span>
                      {onNavigateTab && (
                        <span
                          onClick={(e) => {
                            e.stopPropagation();
                            onNavigateTab("structural_twin");
                          }}
                          className="underline cursor-pointer hover:text-[#065F46]"
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
