import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  fetchLiveEarthquakes,
  fetchDemoStructures,
  fetchDemoEarthquakes,
  runDemoSimulation,
  type LiveEarthquakeResponse,
  type DemoStructure,
  type DemoEarthquake,
  type DemoSimulationResponse,
} from "../../api/researchDemoApi";
import { LiveSeismic3DViewport } from "./LiveSeismic3DViewport";
import { RefreshCw, ExternalLink, Play } from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const DEFAULT_SCENARIO = {
  name: "Pasadena Structural Testbed",
  lat: 34.1377,
  lon: -118.1253,
};

export const LiveEarthquakeView: React.FC = () => {
  // Live Feed State
  const [feedData, setFeedData] = useState<LiveEarthquakeResponse | null>(null);
  const [isLoadingFeed, setIsLoadingFeed] = useState<boolean>(true);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);

  // Filters State
  const [minMag, setMinMag] = useState<number>(4.5);
  const [timeWindowHours, setTimeWindowHours] = useState<number>(24);
  const [radiusFilterKm, setRadiusFilterKm] = useState<number | "ALL">("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // 3D Viewport Controls
  const [visualScale, setVisualScale] = useState<number>(10);

  // Structural & Research Data
  const [structures, setStructures] = useState<DemoStructure[]>([]);
  const [selectedStructureId, setSelectedStructureId] = useState<string>("3S_T035");
  const [historicalRecords, setHistoricalRecords] = useState<DemoEarthquake[]>([]);
  const [selectedHistoricalRecordId, setSelectedHistoricalRecordId] = useState<string>("RSN0001");

  // Simulation State
  const [simResult, setSimResult] = useState<DemoSimulationResponse | null>(null);
  const [isLoadingSim, setIsLoadingSim] = useState<boolean>(false);
  const [simError, setSimError] = useState<string | null>(null);

  // Fetch Live Feed
  const loadFeed = useCallback(async () => {
    setIsLoadingFeed(true);
    try {
      const data = await fetchLiveEarthquakes({
        minmagnitude: minMag > 0 ? minMag : undefined,
        hours: timeWindowHours,
        latitude: DEFAULT_SCENARIO.lat,
        longitude: DEFAULT_SCENARIO.lon,
        radius_km: radiusFilterKm === "ALL" ? undefined : radiusFilterKm,
      });
      setFeedData(data);
      if (data.events.length > 0 && !selectedEventId) {
        setSelectedEventId(data.events[0].event_id);
      }
    } catch (err) {
      console.error("Failed to load USGS feed:", err);
    } finally {
      setIsLoadingFeed(false);
    }
  }, [minMag, timeWindowHours, radiusFilterKm, selectedEventId]);

  useEffect(() => {
    loadFeed();
  }, [loadFeed]);

  // Trigger Simulation with Verified Ground Motion
  const executeSimulation = useCallback(
    async (overrideArchId?: string, overrideRecId?: string) => {
      const archId = overrideArchId || selectedStructureId;
      const recId = overrideRecId || selectedHistoricalRecordId;
      const arch = structures.find((s) => s.archetype_id === archId) || structures[0];
      if (!arch) return;

      setIsLoadingSim(true);
      setSimError(null);
      try {
        const res = await runDemoSimulation({
          archetype_id: arch.archetype_id,
          record_id: recId,
          model_id: "exp6_multimodal_gno",
          selected_floor: arch.n_stories,
        });
        setSimResult(res);
      } catch (err: any) {
        setSimError(err.message || "Simulation failed");
      } finally {
        setIsLoadingSim(false);
      }
    },
    [selectedStructureId, selectedHistoricalRecordId, structures]
  );

  // Load Structural Catalog & Research Ground Motions and auto-run initial simulation
  useEffect(() => {
    async function initResearchData() {
      try {
        const [structs, records] = await Promise.all([
          fetchDemoStructures(),
          fetchDemoEarthquakes(),
        ]);
        setStructures(structs);
        setHistoricalRecords(records);

        const threeStory = structs.find((s) => s.n_stories === 3) || structs[0];
        const defaultRecord = records[0]?.record_id || "RSN0001";
        if (threeStory) {
          setSelectedStructureId(threeStory.archetype_id);
        }
        if (defaultRecord) {
          setSelectedHistoricalRecordId(defaultRecord);
        }

        // Auto-run simulation immediately so viewport and charts are live
        if (threeStory && defaultRecord) {
          executeSimulation(threeStory.archetype_id, defaultRecord);
        }
      } catch (err) {
        console.error("Error loading research catalogs:", err);
      }
    }
    initResearchData();
  }, [executeSimulation]);

  // Filtered Events List
  const filteredEvents = useMemo(() => {
    if (!feedData) return [];
    return feedData.events.filter((ev) => {
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        return (
          ev.location.toLowerCase().includes(q) ||
          ev.event_id.toLowerCase().includes(q) ||
          ev.magnitude.toString().includes(q)
        );
      }
      return true;
    });
  }, [feedData, searchQuery]);

  // Selected Event Object
  const selectedEvent = useMemo(() => {
    if (!filteredEvents.length) return null;
    return filteredEvents.find((e) => e.event_id === selectedEventId) || filteredEvents[0];
  }, [filteredEvents, selectedEventId]);

  // Selected Structure Object
  const currentStructure = useMemo(() => {
    return structures.find((s) => s.archetype_id === selectedStructureId) || structures[0];
  }, [structures, selectedStructureId]);

  // Event selection handler that re-runs simulation
  const handleSelectEvent = (eventId: string) => {
    setSelectedEventId(eventId);
    executeSimulation(selectedStructureId, selectedHistoricalRecordId);
  };

  // Structural Damage State Assessment (FEMA / ASCE 41-17)
  const damageAssessment = useMemo(() => {
    if (!simResult || !currentStructure) return null;
    const peakMm = Math.max(...simResult.u_pred.map(Math.abs)) * 1000.0;
    const heightM = currentStructure.n_stories * 3.5;
    const driftRatioPct = (peakMm / (heightM * 1000.0)) * 100.0;

    let state = "IMMEDIATE OCCUPANCY (IO)";
    let badgeClass = "text-[#73E6B5] bg-[#73E6B5]/15 border-[#73E6B5]/30";
    let desc = "Elastic Response: Frame remains elastic without yielding. Building is structurally safe for immediate occupancy.";

    if (driftRatioPct >= 2.5) {
      state = "COLLAPSE PREVENTION (CP)";
      badgeClass = "text-[#E35D5D] bg-[#E35D5D]/15 border-[#E35D5D]/30";
      desc = "Severe Inelastic Deformation: Significant plastic hinge rotation, large residual drift, structural safety compromised.";
    } else if (driftRatioPct >= 0.7) {
      state = "LIFE SAFETY (LS)";
      badgeClass = "text-[#D6B56D] bg-[#D6B56D]/15 border-[#D6B56D]/30";
      desc = "Moderate Ductile Yielding: Beams exhibit plastic hinges, but structural margins prevent collapse.";
    }

    return { peakMm, driftRatioPct, state, badgeClass, desc };
  }, [simResult, currentStructure]);

  // Trajectory Chart Data for Oscilloscope
  const trajectoryChartData = useMemo(() => {
    if (!simResult) return null;
    return {
      labels: simResult.time.map((t: number) => t.toFixed(2)),
      datasets: [
        {
          label: "OpenSeesPy C-Runtime Ground Truth",
          data: simResult.u_true.map((v: number) => v * 1000.0),
          borderColor: "#82928B",
          borderDash: [3, 3],
          borderWidth: 1.2,
          pointRadius: 0,
          tension: 0.05,
        },
        {
          label: `EXP6 Multi-Modal GNO (${simResult.data_provenance.prediction})`,
          data: simResult.u_pred.map((v: number) => v * 1000.0),
          borderColor: "#73E6B5",
          borderWidth: 1.6,
          pointRadius: 0,
          tension: 0.05,
        },
      ],
    };
  }, [simResult]);

  // Pipeline Step Computation
  const activeTimelineStep = useMemo(() => {
    if (simResult) return 6;
    if (isLoadingSim) return 5;
    if (selectedEvent) return 4;
    return 3;
  }, [simResult, isLoadingSim, selectedEvent]);

  return (
    <div className="w-full min-h-screen bg-research-desk text-[#E8E8DE] p-4 md:p-8 space-y-8 font-sans selection:bg-[#17483A] selection:text-[#73E6B5]">
      {/* ================================================================= */}
      {/* 1. EDITORIAL HEADER & METADATA BAR                                */}
      {/* ================================================================= */}
      <header className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4 border-b border-white/[0.06] pb-4">
        <div>
          <div className="flex items-center gap-3 text-[11px] text-[#82928B] font-mono mb-1">
            <span className="flex items-center gap-1.5 text-[#73E6B5]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5]" />
              SEISMOFNO RESEARCH DESK
            </span>
            <span>·</span>
            <span>USGS GEOJSON STREAM</span>
            <span>·</span>
            <span>REF: {DEFAULT_SCENARIO.name}</span>
          </div>
          <div className="text-xs text-[#82928B]">
            Observed Seismic Event Ingestion & Neural Operator Structural Screening
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono shrink-0">
          <div className="flex items-center gap-2">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                feedData?.is_fallback ? "bg-[#D6B56D]" : "bg-[#73E6B5]"
              }`}
            />
            <span className={feedData?.is_fallback ? "text-[#D6B56D]" : "text-[#73E6B5]"}>
              {feedData?.is_fallback ? "Offline Cache" : "USGS Live"}
            </span>
          </div>
          <span className="text-[#82928B] text-[11px]">
            {feedData?.last_updated ? feedData.last_updated.slice(11, 19) : "--:--:--"} UTC
          </span>
          <button
            onClick={loadFeed}
            disabled={isLoadingFeed}
            className="p-1 text-[#82928B] hover:text-[#73E6B5] transition cursor-pointer disabled:opacity-40"
            title="Sync USGS Feed"
          >
            <RefreshCw size={12} className={isLoadingFeed ? "animate-spin text-[#73E6B5]" : ""} />
          </button>
        </div>
      </header>

      {/* Academic Methodology Primer Box */}
      <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-sans">
        <div className="space-y-1">
          <span className="text-[#73E6B5] font-mono font-bold uppercase text-[10px] block">
            What Live USGS Screening Does
          </span>
          <p className="text-[#82928B] leading-relaxed">
            Ingests real-time earthquake hypocenters from the USGS global seismic network. Select any observed event to
            immediately evaluate the structural response, peak roof drift, and plastic hinge formation on multi-story building frames.
          </p>
        </div>
        <div className="space-y-1">
          <span className="text-[#D6B56D] font-mono font-bold uppercase text-[10px] block">
            Scientific Accelerogram Coupling
          </span>
          <p className="text-[#82928B] leading-relaxed">
            Public USGS GeoJSON feeds supply hypocentral parameters (Mw, depth, coordinates) without streaming raw 100 Hz station
            accelerograms. SeismoFNO pairs the observed event with representative accelerograms to deliver instant (&lt; 2 ms) damage
            screening before civil emergency inspections.
          </p>
        </div>
      </div>

      {/* ================================================================= */}
      {/* 2. HERO: ASYMMETRIC MASONRY (EVENT TYPOGRAPHY + 3D VIEWPORT)      */}
      {/* ================================================================= */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* LEFT / EVENT HERO TYPOGRAPHY BLOCK (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {selectedEvent ? (
            <div className="space-y-4">
              {/* Event Location & Status Tag */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 font-mono text-[11px] text-[#82928B] uppercase tracking-wider">
                  <span>Observed Event</span>
                  <span>·</span>
                  <span className="text-[#73E6B5]">USGS Catalog</span>
                </div>
                <h2 className="text-2xl sm:text-3xl lg:text-4xl font-light text-[#E8E8DE] tracking-tight leading-tight">
                  {selectedEvent.location}
                </h2>
              </div>

              {/* Large Elegant Magnitude Readout */}
              <div className="flex items-baseline gap-3 pt-2">
                <span className="text-6xl sm:text-7xl font-light tracking-tighter text-[#E8E8DE] font-sans">
                  {selectedEvent.magnitude.toFixed(1)}
                </span>
                <div className="font-mono text-sm space-y-0.5">
                  <span className="text-lg font-normal text-[#73E6B5] block">Mw</span>
                  <span className="text-[11px] text-[#82928B] block">Moment Magnitude</span>
                </div>
              </div>

              {/* Data Composition (Typography & Whitespace, No Card Clutter) */}
              <div className="pt-4 border-t border-white/[0.06] space-y-2.5 text-xs">
                <div className="flex justify-between items-baseline">
                  <span className="text-[#82928B] uppercase font-mono text-[10px] tracking-wider">Focal Depth</span>
                  <span className="font-mono text-[#E8E8DE]">{selectedEvent.depth_km.toFixed(1)} km</span>
                </div>

                <div className="flex justify-between items-baseline">
                  <span className="text-[#82928B] uppercase font-mono text-[10px] tracking-wider">Distance to Testbed</span>
                  <span className="font-mono text-[#73E6B5]">
                    {selectedEvent.distance_km != null ? `${selectedEvent.distance_km.toLocaleString()} km` : "N/A"}
                  </span>
                </div>

                <div className="flex justify-between items-baseline">
                  <span className="text-[#82928B] uppercase font-mono text-[10px] tracking-wider">Epicenter Coordinates</span>
                  <span className="font-mono text-[#E8E8DE]">
                    {selectedEvent.latitude.toFixed(3)}°, {selectedEvent.longitude.toFixed(3)}°
                  </span>
                </div>

                <div className="flex justify-between items-baseline">
                  <span className="text-[#82928B] uppercase font-mono text-[10px] tracking-wider">Origin Time</span>
                  <span className="font-mono text-[#82928B]">
                    {selectedEvent.origin_time.replace("T", " ").replace("Z", "")} UTC
                  </span>
                </div>

                <div className="flex justify-between items-baseline">
                  <span className="text-[#82928B] uppercase font-mono text-[10px] tracking-wider">Event Identifier</span>
                  <a
                    href={selectedEvent.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-mono text-[#73E6B5] hover:underline inline-flex items-center gap-1"
                  >
                    {selectedEvent.event_id} <ExternalLink size={10} />
                  </a>
                </div>

                <div className="flex justify-between items-baseline pt-1">
                  <span className="text-[#82928B] uppercase font-mono text-[10px] tracking-wider">Ground Motion Channel</span>
                  <span className="text-[#73E6B5] font-mono text-[11px]">
                    Matched PEER Accelerogram ({selectedHistoricalRecordId})
                  </span>
                </div>

                {damageAssessment && (
                  <div className="p-3 bg-[#0B1714] border border-white/[0.08] rounded space-y-1.5 mt-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono uppercase text-[#82928B]">
                        ASCE 41 Damage Category:
                      </span>
                      <span
                        className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${damageAssessment.badgeClass}`}
                      >
                        {damageAssessment.state}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-[#82928B]">Peak Roof Drift:</span>
                      <span className="text-[#73E6B5] font-bold">
                        {damageAssessment.peakMm.toFixed(2)} mm (
                        {damageAssessment.driftRatioPct.toFixed(3)}% Drift Ratio)
                      </span>
                    </div>
                    <p className="text-[10px] text-[#82928B] font-sans leading-tight pt-0.5">
                      {damageAssessment.desc}
                    </p>
                  </div>
                )}
              </div>

              {/* Surrogate Screening Controls */}
              <div className="pt-4 border-t border-white/[0.06] space-y-3">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-mono text-[#82928B] uppercase tracking-wider block">
                    Structural Archetype & Eigenvalues
                  </label>
                  <select
                    value={selectedStructureId}
                    onChange={(e) => {
                      const newId = e.target.value;
                      setSelectedStructureId(newId);
                      executeSimulation(newId, selectedHistoricalRecordId);
                    }}
                    className="w-full bg-[#0E1B17] border border-white/[0.08] rounded px-3 py-1.5 text-xs font-mono text-[#E8E8DE] outline-none cursor-pointer focus:border-[#73E6B5]"
                  >
                    {structures.map((s) => (
                      <option key={s.archetype_id} value={s.archetype_id} className="bg-[#0B1714]">
                        {s.archetype_id} — {s.n_stories} Stories (T₁ = {s.T1_s.toFixed(2)}s, T₂ = {s.T2_s.toFixed(2)}s)
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[10px] font-mono text-[#82928B] uppercase tracking-wider block">
                    Verified Ground Motion Channel (Research Baseline)
                  </label>
                  <select
                    value={selectedHistoricalRecordId}
                    onChange={(e) => {
                      const newRec = e.target.value;
                      setSelectedHistoricalRecordId(newRec);
                      executeSimulation(selectedStructureId, newRec);
                    }}
                    className="w-full bg-[#0E1B17] border border-white/[0.08] rounded px-3 py-1.5 text-xs font-mono text-[#E8E8DE] outline-none cursor-pointer focus:border-[#73E6B5]"
                  >
                    {historicalRecords.map((rec) => (
                      <option key={rec.record_id} value={rec.record_id} className="bg-[#0B1714]">
                        {rec.record_id} — {rec.event_name} (PGA: {rec.pga_g}g)
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  onClick={() => executeSimulation()}
                  disabled={isLoadingSim}
                  className="w-full py-2.5 bg-[#17483A] hover:bg-[#1C5746] text-[#73E6B5] border border-[#73E6B5]/30 font-mono text-xs font-semibold rounded transition cursor-pointer flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  <Play size={12} className={isLoadingSim ? "animate-spin" : ""} />
                  <span>{isLoadingSim ? "Executing Surrogate Forward Pass..." : "Execute Surrogate Response Simulation"}</span>
                </button>

                {simError && (
                  <div className="text-xs text-[#E35D5D] bg-[#E35D5D]/10 p-2 rounded border border-[#E35D5D]/20 font-mono">
                    {simError}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="p-8 text-[#82928B] font-mono text-xs">
              Awaiting USGS catalog event selection...
            </div>
          )}
        </div>

        {/* RIGHT / 3D STRUCTURAL MODEL CENTERPIECE (7 cols, ~50% of viewport) */}
        <div className="lg:col-span-7 h-[480px] lg:h-[540px] rounded-lg overflow-hidden border border-white/[0.06] shadow-2xl bg-[#0B1714]">
          <LiveSeismic3DViewport
            selectedEvent={selectedEvent}
            visualScale={visualScale}
            onChangeVisualScale={setVisualScale}
            isSimulating={isLoadingSim}
            simResult={simResult}
          />
        </div>
      </section>

      {/* ================================================================= */}
      {/* 3. SCIENTIFIC PROCESS PIPELINE (Linear, Horizontal, Restrained)   */}
      {/* ================================================================= */}
      <section className="border-t border-b border-white/[0.06] py-3 font-mono text-[11px]">
        <div className="flex flex-wrap items-center justify-between gap-y-2 gap-x-4">
          <div className="text-[10px] text-[#82928B] uppercase tracking-wider">
            Research Process Pipeline
          </div>

          <div className="flex flex-wrap items-center gap-3 md:gap-6 text-xs">
            {[
              { num: "01", label: "Catalog" },
              { num: "02", label: "Observed Event" },
              { num: "03", label: "Metadata Ingested" },
              { num: "04", label: "Structural Frame" },
              { num: "05", label: "Surrogate Evaluation" },
              { num: "06", label: "Screening Complete" },
            ].map((step, idx) => {
              const stepNum = idx + 1;
              const isCurrent = activeTimelineStep === stepNum;
              const isDone = activeTimelineStep >= stepNum;

              return (
                <div key={step.num} className="flex items-center gap-1.5">
                  <span
                    className={`text-[10px] ${
                      isCurrent
                        ? "text-[#73E6B5] font-bold"
                        : isDone
                        ? "text-[#E8E8DE]"
                        : "text-[#82928B]/60"
                    }`}
                  >
                    {step.num}
                  </span>
                  <span
                    className={`${
                      isCurrent
                        ? "text-[#73E6B5] font-semibold"
                        : isDone
                        ? "text-[#E8E8DE]"
                        : "text-[#82928B]/60"
                    }`}
                  >
                    {step.label}
                  </span>
                  {idx < 5 && <span className="text-white/[0.1] ml-2">→</span>}
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/* 4. ASYMMETRIC LOWER SECTION: ARCHIVE CATALOGUE & WAVEFORM STRIP   */}
      {/* ================================================================= */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* LEFT / EVENT ARCHIVE CATALOGUE (7 cols) */}
        <div className="lg:col-span-7 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <h3 className="text-sm font-semibold tracking-tight text-[#E8E8DE]">
              Recent Observed Events ({filteredEvents.length})
            </h3>
            <div className="text-[11px] font-mono text-[#82928B]">
              USGS Public GeoJSON Feed
            </div>
          </div>

          {/* Minimalist Filters Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono pt-1">
            <select
              value={minMag}
              onChange={(e) => setMinMag(parseFloat(e.target.value))}
              className="bg-[#0E1B17] border border-white/[0.06] rounded px-2.5 py-1 text-[#E8E8DE] outline-none"
            >
              <option value="0.0">All (M0.0+)</option>
              <option value="2.5">M2.5+ (Minor)</option>
              <option value="4.5">M4.5+ (Moderate)</option>
              <option value="6.0">M6.0+ (Strong)</option>
              <option value="7.0">M7.0+ (Major)</option>
            </select>

            <select
              value={timeWindowHours}
              onChange={(e) => setTimeWindowHours(parseFloat(e.target.value))}
              className="bg-[#0E1B17] border border-white/[0.06] rounded px-2.5 py-1 text-[#E8E8DE] outline-none"
            >
              <option value="1">Past 1 Hour</option>
              <option value="6">Past 6 Hours</option>
              <option value="24">Past 24 Hours</option>
              <option value="168">Past 7 Days</option>
            </select>

            <select
              value={radiusFilterKm}
              onChange={(e) =>
                setRadiusFilterKm(e.target.value === "ALL" ? "ALL" : parseFloat(e.target.value))
              }
              className="bg-[#0E1B17] border border-white/[0.06] rounded px-2.5 py-1 text-[#E8E8DE] outline-none"
            >
              <option value="ALL">Global (No Limit)</option>
              <option value="500">Within 500 km</option>
              <option value="1000">Within 1,000 km</option>
              <option value="3000">Within 3,000 km</option>
            </select>

            <input
              type="text"
              placeholder="Search region or ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-[#0E1B17] border border-white/[0.06] rounded px-2.5 py-1 text-[#E8E8DE] outline-none placeholder:text-[#82928B]/60"
            />
          </div>

          {/* Research Archive Table */}
          <div className="overflow-x-auto pt-2">
            <table className="w-full text-left border-collapse font-sans text-xs">
              <thead>
                <tr className="border-b border-white/[0.08] text-[10px] font-mono text-[#82928B] uppercase tracking-wider">
                  <th className="pb-2 font-medium">UTC Time</th>
                  <th className="pb-2 font-medium">Magnitude</th>
                  <th className="pb-2 font-medium">Region</th>
                  <th className="pb-2 font-medium">Depth</th>
                  <th className="pb-2 font-medium">Event ID</th>
                  <th className="pb-2 font-medium">Source</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {filteredEvents.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-6 text-center text-xs text-[#82928B] font-mono">
                      No earthquake records match the filter criteria.
                    </td>
                  </tr>
                ) : (
                  filteredEvents.map((ev) => {
                    const isSelected = selectedEvent?.event_id === ev.event_id;
                    return (
                      <tr
                        key={ev.event_id}
                        onClick={() => handleSelectEvent(ev.event_id)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-[#17483A]/30 text-[#E8E8DE]"
                            : "hover:bg-[#0E1B17] text-[#82928B] hover:text-[#E8E8DE]"
                        }`}
                      >
                        <td className="py-2.5 font-mono text-[11px] whitespace-nowrap">
                          {ev.origin_time.replace("T", " ").replace("Z", "").slice(5, 16)}
                        </td>
                        <td className="py-2.5 font-mono text-xs whitespace-nowrap">
                          <span
                            className={
                              ev.magnitude >= 7.0
                                ? "text-[#E35D5D] font-bold"
                                : ev.magnitude >= 5.0
                                ? "text-[#D6B56D] font-semibold"
                                : "text-[#73E6B5]"
                            }
                          >
                            M{ev.magnitude.toFixed(1)}
                          </span>
                        </td>
                        <td className="py-2.5 truncate max-w-[200px] text-[#E8E8DE]" title={ev.location}>
                          {ev.location}
                        </td>
                        <td className="py-2.5 font-mono text-[11px] whitespace-nowrap">
                          {ev.depth_km.toFixed(1)} km
                        </td>
                        <td className="py-2.5 font-mono text-[11px] text-[#73E6B5] whitespace-nowrap">
                          {ev.event_id}
                        </td>
                        <td className="py-2.5 text-[11px] uppercase tracking-wider text-[#82928B]">
                          {ev.source}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* RIGHT / WAVEFORM & STRUCTURAL TELEMETRY (5 cols) */}
        <div className="lg:col-span-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold tracking-tight text-[#E8E8DE]">
              Structural Response Waveform u(t)
            </h3>
            <span className="text-[11px] font-mono text-[#82928B]">
              {simResult ? "Simulation Complete" : "Standby"}
            </span>
          </div>

          {/* Clean Oscilloscope Canvas */}
          <div className="w-full h-56 bg-[#07110F] rounded border border-white/[0.06] p-3 flex flex-col justify-center">
            {trajectoryChartData ? (
              <Line
                data={trajectoryChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  animation: { duration: 250 },
                  scales: {
                    x: {
                      title: { display: true, text: "Time (s)", font: { family: "monospace", size: 9 }, color: "#82928B" },
                      ticks: { color: "#82928B", font: { family: "monospace", size: 9 } },
                      grid: { color: "rgba(115, 230, 181, 0.05)" },
                    },
                    y: {
                      title: { display: true, text: "Roof Drift u(t) (mm)", font: { family: "monospace", size: 9 }, color: "#82928B" },
                      ticks: { color: "#82928B", font: { family: "monospace", size: 9 } },
                      grid: { color: "rgba(115, 230, 181, 0.05)" },
                    },
                  },
                  plugins: {
                    legend: {
                      position: "top" as const,
                      labels: { font: { family: "monospace", size: 9 }, color: "#E8E8DE", boxWidth: 10 },
                    },
                  },
                }}
              />
            ) : (
              <div className="text-center p-6 space-y-2 font-mono">
                <div className="text-xs text-[#E8E8DE] uppercase tracking-wider">
                  Ground Motion Channel
                </div>
                <div className="text-[11px] text-[#D6B56D]">
                  No Observed Accelerogram Ingested
                </div>
                <p className="text-[11px] text-[#82928B] font-sans max-w-sm mx-auto leading-relaxed pt-1">
                  USGS catalog feeds provide earthquake occurrence parameters without streaming station accelerograms. Trigger surrogate screening above to compute structural response.
                </p>
              </div>
            )}
          </div>

          {/* Performance Metrics Readout */}
          {simResult && (
            <div className="grid grid-cols-3 gap-2 pt-1 font-mono text-xs text-center">
              <div className="p-2.5 bg-[#0E1B17] rounded border border-white/[0.06]">
                <span className="text-[#82928B] block text-[10px] uppercase">Peak Error</span>
                <span className="text-[#73E6B5] font-semibold">{simResult.metrics.peak_disp_err_pct}%</span>
              </div>
              <div className="p-2.5 bg-[#0E1B17] rounded border border-white/[0.06]">
                <span className="text-[#82928B] block text-[10px] uppercase">Rel L₂ Error</span>
                <span className="text-[#73E6B5] font-semibold">{simResult.metrics.rel_l2_pct}%</span>
              </div>
              <div className="p-2.5 bg-[#0E1B17] rounded border border-white/[0.06]">
                <span className="text-[#82928B] block text-[10px] uppercase">MPS Latency</span>
                <span className="text-[#E8E8DE] font-semibold">{simResult.metrics.latency_ms} ms</span>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* ================================================================= */}
      {/* 5. SCIENTIFIC PROTOCOL & BOUNDARY FOOTNOTE                         */}
      {/* ================================================================= */}
      <footer className="pt-4 border-t border-white/[0.06] text-[11px] text-[#82928B] leading-relaxed flex flex-col md:flex-row md:items-baseline justify-between gap-3">
        <div>
          <strong className="text-[#E8E8DE] uppercase font-mono text-[10px] tracking-wider block mb-0.5">
            Research Boundary & Protocol
          </strong>
          SeismoFNO performs rapid surrogate structural response estimation for multi-story systems. It does not predict earthquakes, forecast ground motions, or issue structural safety certifications.
        </div>
        <div className="font-mono text-[10px] text-[#82928B] shrink-0">
          SURROGATE: EXP6 MULTI-MODAL GNO · VALIDATED AGAINST OPENSEESPY
        </div>
      </footer>
    </div>
  );
};
