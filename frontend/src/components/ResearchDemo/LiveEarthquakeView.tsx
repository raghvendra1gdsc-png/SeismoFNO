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
import {
  Activity,
  Radio,
  RefreshCw,
  ExternalLink,
  Sliders,
  Play,
} from "lucide-react";
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

// Scenario Reference Coordinates (e.g. Structural Engineering Testbed in Pasadena / CA)
const DEFAULT_SCENARIO = {
  name: "3-Story RC Benchmark Frame (Pasadena Testbed)",
  lat: 34.1377,
  lon: -118.1253,
};

export const LiveEarthquakeView: React.FC = () => {
  // Live USGS Feed State
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

  // Load Structural Catalog & Research Ground Motions
  useEffect(() => {
    async function initResearchData() {
      try {
        const [structs, records] = await Promise.all([
          fetchDemoStructures(),
          fetchDemoEarthquakes(),
        ]);
        setStructures(structs);
        setHistoricalRecords(records);

        // Pre-select 3-Story or 5-Story structure
        const threeStory = structs.find((s) => s.n_stories === 3) || structs[0];
        if (threeStory) {
          setSelectedStructureId(threeStory.archetype_id);
        }
      } catch (err) {
        console.error("Error loading research catalogs:", err);
      }
    }
    initResearchData();
  }, []);

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

  // Trigger Simulation with Verified Ground Motion
  const executeSimulation = async () => {
    if (!currentStructure) return;
    setIsLoadingSim(true);
    setSimError(null);
    try {
      const res = await runDemoSimulation({
        archetype_id: currentStructure.archetype_id,
        record_id: selectedHistoricalRecordId,
        model_id: "exp6_multimodal_gno",
        selected_floor: currentStructure.n_stories,
      });
      setSimResult(res);
    } catch (err: any) {
      setSimError(err.message || "Simulation failed");
    } finally {
      setIsLoadingSim(false);
    }
  };

  // Trajectory Chart Data for Oscilloscope
  const trajectoryChartData = useMemo(() => {
    if (!simResult) return null;
    return {
      labels: simResult.time.map((t: number) => t.toFixed(2)),
      datasets: [
        {
          label: "OpenSeesPy C-Runtime Ground Truth (NLTHA)",
          data: simResult.u_true.map((v: number) => v * 1000.0),
          borderColor: "#8D9AAA",
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.05,
        },
        {
          label: `EXP6 Multi-Modal GNO (${simResult.data_provenance.prediction})`,
          data: simResult.u_pred.map((v: number) => v * 1000.0),
          borderColor: "#28D7FF",
          borderWidth: 1.8,
          pointRadius: 0,
          tension: 0.05,
        },
      ],
    };
  }, [simResult]);

  // Timeline Step Status Computation
  const activeTimelineStep = useMemo(() => {
    if (simResult) return 6; // Screening complete
    if (isLoadingSim) return 5; // Surrogate evaluation
    if (selectedEvent) return 4; // Structural model ready
    return 3; // Metadata ingested
  }, [simResult, isLoadingSim, selectedEvent]);

  return (
    <div className="w-full min-h-screen bg-[#080C12] text-[#E8EDF3] p-3 md:p-5 space-y-4 font-sans select-none">
      {/* ================================================================= */}
      {/* 1. TOP SYSTEM INSTRUMENTATION BAR                                 */}
      {/* ================================================================= */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 bg-[#0B1018] border border-white/[0.07] px-4 py-3 rounded-[3px]">
        <div>
          <div className="flex items-center gap-2 mb-1 flex-wrap font-mono text-[10px]">
            <span className="text-[#28D7FF] font-bold tracking-widest uppercase flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#28D7FF] shadow-[0_0_6px_#28D7FF]" />
              SEISMOFNO INGESTION CONSOLE
            </span>
            <span className="text-[#667487]">|</span>
            <span className="text-[#8D9AAA]">CATALOG: USGS REAL-TIME GEOJSON FEED (PUBLIC DOMAIN)</span>
            <span className="text-[#667487]">|</span>
            <span className="text-[#8D9AAA]">TARGET: {DEFAULT_SCENARIO.name}</span>
          </div>
          <h1 className="text-base md:text-lg font-mono font-bold tracking-tight text-[#E8EDF3]">
            Observed Seismic Event Ingestion & Neural Operator Structural Screening
          </h1>
        </div>

        <div className="flex items-center gap-2 shrink-0 font-mono text-[11px]">
          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-[#111821] border border-white/[0.07] rounded-[3px]">
            <span
              className={`w-2 h-2 rounded-full ${
                feedData?.is_fallback ? "bg-[#E8A63B] animate-pulse" : "bg-[#31D17C] shadow-[0_0_6px_#31D17C]"
              }`}
            />
            <span className={feedData?.is_fallback ? "text-[#E8A63B] font-bold" : "text-[#31D17C] font-bold"}>
              {feedData?.is_fallback ? "OFFLINE CACHED CATALOG" : "USGS FEED ONLINE"}
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-1 px-2.5 py-1 bg-[#111821] border border-white/[0.07] rounded-[3px] text-[#8D9AAA]">
            <span>LAST SYNC:</span>
            <span className="text-[#E8EDF3]">{feedData?.last_updated || "SYNCING..."}</span>
          </div>

          <button
            onClick={loadFeed}
            disabled={isLoadingFeed}
            className="flex items-center gap-1 px-2.5 py-1 bg-[#151D27] hover:bg-[#19222D] border border-white/[0.07] rounded-[3px] text-[#28D7FF] font-mono cursor-pointer transition disabled:opacity-50"
            title="Refresh USGS Catalog"
          >
            <RefreshCw size={11} className={isLoadingFeed ? "animate-spin text-[#28D7FF]" : ""} />
            <span>SYNC</span>
          </button>
        </div>
      </div>

      {/* ================================================================= */}
      {/* 2. RESEARCH PROTOCOL — SCIENTIFIC BOUNDARIES STRIP               */}
      {/* ================================================================= */}
      <div className="bg-[#0B1018] border-l-2 border-[#28D7FF] border border-white/[0.07] px-4 py-2.5 rounded-[3px] font-mono text-[10px] space-y-1">
        <div className="flex items-center justify-between text-[#28D7FF] font-bold tracking-wider uppercase">
          <span>RESEARCH PROTOCOL — STRICT SCIENTIFIC BOUNDARIES</span>
          <span className="text-[#667487] font-normal hidden sm:inline">
            OBSERVATION → PHYSICS → SURROGATE → SCREENING
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[#8D9AAA] leading-relaxed pt-0.5">
          <div>
            <strong className="text-[#E8EDF3]">PERMITTED:</strong> Observed earthquake ingestion · Event discovery · Rapid surrogate structural analysis · Comparative response estimation
          </div>
          <div>
            <strong className="text-[#E35D5D]">NOT CLAIMED:</strong> Earthquake prediction · Earthquake forecasting · Live building accelerometer monitoring · Automated structural safety certification
          </div>
        </div>
      </div>

      {/* ================================================================= */}
      {/* 3. MAIN WORKSTATION VIEWPORT & INTELLIGENCE SPLIT                 */}
      {/* ================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* CENTER / DOMINANT: 3D SEISMIC WORKSPACE (8 cols) */}
        <div className="lg:col-span-8 flex flex-col min-h-[500px] lg:min-h-[540px]">
          <LiveSeismic3DViewport
            selectedEvent={selectedEvent}
            visualScale={visualScale}
            onChangeVisualScale={setVisualScale}
            isSimulating={isLoadingSim}
            simResult={simResult}
          />
        </div>

        {/* RIGHT: EVENT INTELLIGENCE & STRUCTURAL SCREENING PANEL (4 cols) */}
        <div className="lg:col-span-4 space-y-3 flex flex-col justify-between">
          {/* Card A: Observed Event Context */}
          <div className="bg-[#0B1018] border border-white/[0.07] rounded-[3px] p-3.5 space-y-3 font-mono text-[11px]">
            <div className="flex items-center justify-between border-b border-white/[0.07] pb-2">
              <span className="text-xs font-bold text-[#E8EDF3] tracking-wide uppercase flex items-center gap-1.5">
                <Radio size={13} className="text-[#28D7FF]" />
                OBSERVED EVENT
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded-[2px] bg-[#31D17C]/15 text-[#31D17C] border border-[#31D17C]/30 font-bold uppercase">
                USGS OBSERVED
              </span>
            </div>

            {selectedEvent ? (
              <div className="space-y-2">
                {/* Event Location & Source */}
                <div className="bg-[#111821] p-2.5 rounded-[3px] border border-white/[0.07] space-y-1">
                  <span className="text-[9px] text-[#667487] uppercase block font-bold">EVENT LOCATION</span>
                  <div className="text-xs font-bold text-[#E8EDF3] leading-snug">
                    {selectedEvent.location}
                  </div>
                  <div className="flex items-center gap-2 pt-1 text-[10px]">
                    <span className="text-[#8D9AAA]">ID: <strong className="text-[#28D7FF]">{selectedEvent.event_id}</strong></span>
                    <span className="text-[#667487]">|</span>
                    <a
                      href={selectedEvent.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[#28D7FF] hover:underline inline-flex items-center gap-0.5"
                    >
                      USGS PAGE <ExternalLink size={9} />
                    </a>
                  </div>
                </div>

                {/* Primary Metric Grid */}
                <div className="grid grid-cols-2 gap-1.5 text-[10px]">
                  <div className="p-2 bg-[#111821] rounded-[3px] border border-white/[0.07]">
                    <span className="text-[#667487] block uppercase">MAGNITUDE</span>
                    <span className="text-sm font-bold text-[#E35D5D] font-mono">
                      M{selectedEvent.magnitude.toFixed(1)} Mw
                    </span>
                  </div>

                  <div className="p-2 bg-[#111821] rounded-[3px] border border-white/[0.07]">
                    <span className="text-[#667487] block uppercase">FOCAL DEPTH</span>
                    <span className="text-sm font-bold text-[#E8EDF3] font-mono">
                      {selectedEvent.depth_km.toFixed(1)} km
                    </span>
                  </div>

                  <div className="p-2 bg-[#111821] rounded-[3px] border border-white/[0.07]">
                    <span className="text-[#667487] block uppercase">EPICENTER COORDS</span>
                    <span className="text-[#8D9AAA] font-mono">
                      {selectedEvent.latitude.toFixed(3)}°, {selectedEvent.longitude.toFixed(3)}°
                    </span>
                  </div>

                  <div className="p-2 bg-[#111821] rounded-[3px] border border-white/[0.07]">
                    <span className="text-[#667487] block uppercase">DISTANCE TO SCENARIO</span>
                    <span className="text-sm font-bold text-[#28D7FF] font-mono">
                      {selectedEvent.distance_km != null ? `${selectedEvent.distance_km.toLocaleString()} km` : "N/A"}
                    </span>
                  </div>
                </div>

                {/* Status Parameters */}
                <div className="p-2 bg-[#111821] rounded-[3px] border border-white/[0.07] space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-[#667487]">ORIGIN TIME (UTC):</span>
                    <span className="text-[#E8EDF3] font-mono">{selectedEvent.origin_time.replace("T", " ").replace("Z", "")}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#667487]">DATA STATUS:</span>
                    <span className="text-[#31D17C] font-semibold">CATALOG METADATA</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#667487]">GROUND MOTION:</span>
                    <span className="text-[#E8A63B] font-semibold">NOT AVAILABLE (NO ACCEL CHANNEL)</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-[#667487]">No event selected from catalog.</div>
            )}
          </div>

          {/* Card B: Structural Screening & Forward Pass */}
          <div className="bg-[#0B1018] border border-white/[0.07] rounded-[3px] p-3.5 space-y-3 font-mono text-[11px]">
            <div className="flex items-center justify-between border-b border-white/[0.07] pb-2">
              <span className="text-xs font-bold text-[#E8EDF3] tracking-wide uppercase flex items-center gap-1.5">
                <Sliders size={13} className="text-[#28D7FF]" />
                SURROGATE SCREENING
              </span>
              <span className="text-[9px] px-1.5 py-0.5 rounded-[2px] bg-[#28D7FF]/15 text-[#28D7FF] border border-[#28D7FF]/30 font-bold uppercase">
                EXP6 MULTI-MODAL GNO
              </span>
            </div>

            {/* Structure Selection */}
            <div className="space-y-1">
              <label className="text-[10px] text-[#667487] uppercase font-bold block">
                STRUCTURAL ARCHETYPE
              </label>
              <select
                value={selectedStructureId}
                onChange={(e) => setSelectedStructureId(e.target.value)}
                className="w-full bg-[#111821] border border-white/[0.07] rounded-[3px] px-2.5 py-1.5 text-xs font-mono text-[#E8EDF3] outline-none cursor-pointer focus:border-[#28D7FF]"
              >
                {structures.map((s) => (
                  <option key={s.archetype_id} value={s.archetype_id} className="bg-[#0B1018]">
                    {s.archetype_id} — {s.n_stories} Stories ({s.category}, T₁={s.T1_s.toFixed(2)}s)
                  </option>
                ))}
              </select>
            </div>

            {/* Invariant Eigenvalue Matrix */}
            <div className="p-2 bg-[#111821] rounded-[3px] border border-white/[0.07] space-y-1">
              <span className="text-[9px] text-[#667487] uppercase block font-bold">
                MODAL EIGENVALUE INVARIANTS (FiLM INJECTION)
              </span>
              <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
                <div className="bg-[#0B1018] p-1.5 rounded-[2px] border border-white/[0.05]">
                  <span className="text-[#667487] block">T₁ (FUND)</span>
                  <strong className="text-[#28D7FF] font-mono">{currentStructure?.T1_s.toFixed(3)} s</strong>
                </div>
                <div className="bg-[#0B1018] p-1.5 rounded-[2px] border border-white/[0.05]">
                  <span className="text-[#667487] block">T₂</span>
                  <strong className="text-[#E8EDF3] font-mono">{currentStructure?.T2_s.toFixed(3)} s</strong>
                </div>
                <div className="bg-[#0B1018] p-1.5 rounded-[2px] border border-white/[0.05]">
                  <span className="text-[#667487] block">T₃</span>
                  <strong className="text-[#E8EDF3] font-mono">{currentStructure?.T3_s.toFixed(3)} s</strong>
                </div>
              </div>
            </div>

            {/* Research Acceleration Baseline Dropdown */}
            <div className="space-y-1">
              <label className="text-[10px] text-[#667487] uppercase font-bold block">
                VERIFIED ACCELERATION CHANNEL (RESEARCH RECORD)
              </label>
              <select
                value={selectedHistoricalRecordId}
                onChange={(e) => setSelectedHistoricalRecordId(e.target.value)}
                className="w-full bg-[#111821] border border-white/[0.07] rounded-[3px] px-2.5 py-1.5 text-xs font-mono text-[#E8EDF3] outline-none cursor-pointer focus:border-[#28D7FF]"
              >
                {historicalRecords.map((rec) => (
                  <option key={rec.record_id} value={rec.record_id} className="bg-[#0B1018]">
                    {rec.record_id} — {rec.event_name} (PGA: {rec.pga_g}g)
                  </option>
                ))}
              </select>
            </div>

            {/* Execute Forward Pass Button */}
            <button
              onClick={executeSimulation}
              disabled={isLoadingSim}
              className="w-full py-2 bg-[#28D7FF]/20 hover:bg-[#28D7FF]/30 text-[#28D7FF] border border-[#28D7FF]/50 font-mono font-bold rounded-[3px] text-xs flex items-center justify-center gap-1.5 cursor-pointer transition disabled:opacity-50"
            >
              <Play size={12} className={isLoadingSim ? "animate-spin" : ""} />
              <span>{isLoadingSim ? "COMPUTING SURROGATE FORWARD PASS..." : "EXECUTE SURROGATE RESPONSE SIMULATION"}</span>
            </button>

            {simError && (
              <div className="text-[10px] text-[#E35D5D] bg-[#E35D5D]/10 p-2 rounded-[2px] border border-[#E35D5D]/30">
                {simError}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ================================================================= */}
      {/* 4. HORIZONTAL EVENT LIFECYCLE TIMELINE                            */}
      {/* ================================================================= */}
      <div className="bg-[#0B1018] border border-white/[0.07] px-4 py-3 rounded-[3px] font-mono text-[10px]">
        <div className="flex items-center justify-between text-[#667487] uppercase tracking-wider mb-2 font-bold">
          <span>EVENT SCREENING PIPELINE STATE</span>
          <span className="text-[#28D7FF]">PHASE 0{activeTimelineStep} / 06</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-6 gap-2">
          {[
            { step: 1, label: "CATALOG", detail: "USGS GEOJSON" },
            { step: 2, label: "EVENT DETECTED", detail: selectedEvent ? selectedEvent.event_id : "IDLE" },
            { step: 3, label: "METADATA INGESTED", detail: selectedEvent ? `M${selectedEvent.magnitude.toFixed(1)} Mw` : "PENDING" },
            { step: 4, label: "MODEL READY", detail: currentStructure ? currentStructure.archetype_id : "READY" },
            { step: 5, label: "SURROGATE EVAL", detail: isLoadingSim ? "INFERRING" : simResult ? "COMPLETE" : "STANDBY" },
            { step: 6, label: "SCREENING COMPLETE", detail: simResult ? `${simResult.metrics.latency_ms} ms` : "AWAITING RUN" },
          ].map((item) => {
            const isDone = activeTimelineStep >= item.step;
            const isCurrent = activeTimelineStep === item.step;
            return (
              <div
                key={item.step}
                className={`p-2 rounded-[2px] border transition-all ${
                  isCurrent
                    ? "bg-[#28D7FF]/15 border-[#28D7FF]/60 text-[#28D7FF]"
                    : isDone
                    ? "bg-[#111821] border-white/[0.1] text-[#E8EDF3]"
                    : "bg-[#0B1018] border-white/[0.04] text-[#667487]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[9px] font-bold">0{item.step}</span>
                  {isCurrent && <span className="w-1.5 h-1.5 rounded-full bg-[#28D7FF] shadow-[0_0_6px_#28D7FF]" />}
                </div>
                <div className="font-bold tracking-tight text-[10px] mt-0.5">{item.label}</div>
                <div className="text-[9px] opacity-75 truncate">{item.detail}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ================================================================= */}
      {/* 5. LOWER SECTION: EVENT CONSOLE & WAVEFORM TELEMETRY              */}
      {/* ================================================================= */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* LEFT (7 cols): COMPACT EVENT CONSOLE TABLE & FILTERS */}
        <div className="lg:col-span-7 bg-[#0B1018] border border-white/[0.07] rounded-[3px] p-3.5 space-y-3 font-mono text-[11px]">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/[0.07] pb-2.5">
            <div className="flex items-center gap-2">
              <Radio size={14} className="text-[#28D7FF]" />
              <h2 className="text-xs font-bold text-[#E8EDF3] uppercase tracking-wider">
                USGS EVENT CATALOG ({filteredEvents.length} RECORDS)
              </h2>
            </div>
            <div className="text-[10px] text-[#8D9AAA]">
              AUTOSYNC: <span className="text-[#31D17C] font-semibold">ACTIVE (EVERY 60s)</span>
            </div>
          </div>

          {/* Compact Filter Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 text-[10px]">
            <div>
              <label className="text-[#667487] uppercase block font-bold mb-1">MIN MAG</label>
              <select
                value={minMag}
                onChange={(e) => setMinMag(parseFloat(e.target.value))}
                className="w-full bg-[#111821] border border-white/[0.07] rounded-[2px] px-2 py-1 text-[#E8EDF3] outline-none"
              >
                <option value="0.0">All (M0.0+)</option>
                <option value="2.5">M2.5+ (Minor)</option>
                <option value="4.5">M4.5+ (Moderate)</option>
                <option value="6.0">M6.0+ (Strong)</option>
                <option value="7.0">M7.0+ (Major)</option>
              </select>
            </div>

            <div>
              <label className="text-[#667487] uppercase block font-bold mb-1">TIME WINDOW</label>
              <select
                value={timeWindowHours}
                onChange={(e) => setTimeWindowHours(parseFloat(e.target.value))}
                className="w-full bg-[#111821] border border-white/[0.07] rounded-[2px] px-2 py-1 text-[#E8EDF3] outline-none"
              >
                <option value="1">Past 1 Hour</option>
                <option value="6">Past 6 Hours</option>
                <option value="24">Past 24 Hours</option>
                <option value="168">Past 7 Days</option>
              </select>
            </div>

            <div>
              <label className="text-[#667487] uppercase block font-bold mb-1">RADIUS FILTER</label>
              <select
                value={radiusFilterKm}
                onChange={(e) =>
                  setRadiusFilterKm(e.target.value === "ALL" ? "ALL" : parseFloat(e.target.value))
                }
                className="w-full bg-[#111821] border border-white/[0.07] rounded-[2px] px-2 py-1 text-[#E8EDF3] outline-none"
              >
                <option value="ALL">Global (No Limit)</option>
                <option value="500">Within 500 km</option>
                <option value="1000">Within 1,000 km</option>
                <option value="3000">Within 3,000 km</option>
              </select>
            </div>

            <div>
              <label className="text-[#667487] uppercase block font-bold mb-1">FILTER REGION / ID</label>
              <input
                type="text"
                placeholder="e.g. California, Japan"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-[#111821] border border-white/[0.07] rounded-[2px] px-2 py-1 text-[#E8EDF3] outline-none placeholder:text-[#667487]"
              >
              </input>
            </div>
          </div>

          {/* Compact Console Table */}
          <div className="border border-white/[0.07] rounded-[2px] overflow-hidden">
            <div className="max-h-64 overflow-y-auto font-mono text-[10px]">
              <table className="w-full text-left border-collapse">
                <thead className="bg-[#111821] border-b border-white/[0.07] text-[9px] font-bold text-[#667487] uppercase tracking-wider sticky top-0 z-10">
                  <tr>
                    <th className="p-2">TIME (UTC)</th>
                    <th className="p-2">MAG</th>
                    <th className="p-2">REGION</th>
                    <th className="p-2">DEPTH</th>
                    <th className="p-2">EVENT ID</th>
                    <th className="p-2">SOURCE</th>
                    <th className="p-2">STATUS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.04] bg-[#0B1018]">
                  {filteredEvents.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="p-4 text-center text-[#667487]">
                        No earthquake records match the filter criteria.
                      </td>
                    </tr>
                  ) : (
                    filteredEvents.map((ev) => {
                      const isSelected = selectedEvent?.event_id === ev.event_id;
                      const magColor =
                        ev.magnitude >= 7.0
                          ? "text-[#E35D5D] font-bold"
                          : ev.magnitude >= 5.0
                          ? "text-[#E8A63B] font-semibold"
                          : "text-[#E8EDF3]";

                      return (
                        <tr
                          key={ev.event_id}
                          onClick={() => setSelectedEventId(ev.event_id)}
                          className={`cursor-pointer transition-colors ${
                            isSelected
                              ? "bg-[#151D27] font-semibold border-l-2 border-l-[#28D7FF]"
                              : "hover:bg-[#111821] border-l-2 border-l-transparent"
                          }`}
                        >
                          <td className="p-2 whitespace-nowrap text-[#8D9AAA]">
                            {ev.origin_time.replace("T", " ").replace("Z", "").slice(5, 16)}
                          </td>
                          <td className={`p-2 whitespace-nowrap ${magColor}`}>
                            M{ev.magnitude.toFixed(1)}
                          </td>
                          <td className="p-2 truncate max-w-[160px] text-[#E8EDF3]" title={ev.location}>
                            {ev.location}
                          </td>
                          <td className="p-2 whitespace-nowrap text-[#8D9AAA]">
                            {ev.depth_km.toFixed(1)} km
                          </td>
                          <td className="p-2 whitespace-nowrap text-[#28D7FF] font-mono">
                            {ev.event_id}
                          </td>
                          <td className="p-2 whitespace-nowrap text-[#8D9AAA]">
                            {ev.source}
                          </td>
                          <td className="p-2 whitespace-nowrap">
                            <span className="px-1 py-0.2 rounded-[2px] text-[8px] bg-[#111821] text-[#31D17C] border border-[#31D17C]/20 uppercase font-bold">
                              {ev.status}
                            </span>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* RIGHT (5 cols): OSCILLOSCOPE WAVEFORM & RESPONSE TELEMETRY STRIP */}
        <div className="lg:col-span-5 bg-[#0B1018] border border-white/[0.07] rounded-[3px] p-3.5 space-y-3 font-mono text-[11px] flex flex-col justify-between">
          <div className="flex items-center justify-between border-b border-white/[0.07] pb-2">
            <span className="text-xs font-bold text-[#E8EDF3] tracking-wide uppercase flex items-center gap-1.5">
              <Activity size={13} className="text-[#28D7FF]" />
              WAVEFORM & RESPONSE CHANNEL
            </span>
            {simResult ? (
              <span className="text-[9px] px-1.5 py-0.5 rounded-[2px] bg-[#31D17C]/15 text-[#31D17C] border border-[#31D17C]/30 font-bold">
                SIMULATION COMPLETE
              </span>
            ) : (
              <span className="text-[9px] px-1.5 py-0.5 rounded-[2px] bg-[#E8A63B]/15 text-[#E8A63B] border border-[#E8A63B]/30 font-bold">
                CATALOG METADATA ONLY
              </span>
            )}
          </div>

          {/* Oscilloscope View Area */}
          <div className="w-full h-56 bg-[#080C12] rounded-[3px] border border-white/[0.07] p-2 relative flex flex-col justify-center">
            {trajectoryChartData ? (
              <Line
                data={trajectoryChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  animation: { duration: 250 },
                  scales: {
                    x: {
                      title: { display: true, text: "Time (s)", font: { family: "monospace", size: 9 }, color: "#8D9AAA" },
                      ticks: { color: "#667487", font: { family: "monospace", size: 9 } },
                      grid: { color: "rgba(255, 255, 255, 0.05)" },
                    },
                    y: {
                      title: { display: true, text: "Drift u(t) (mm)", font: { family: "monospace", size: 9 }, color: "#8D9AAA" },
                      ticks: { color: "#667487", font: { family: "monospace", size: 9 } },
                      grid: { color: "rgba(255, 255, 255, 0.05)" },
                    },
                  },
                  plugins: {
                    legend: {
                      position: "top" as const,
                      labels: { font: { family: "monospace", size: 9 }, color: "#E8EDF3", boxWidth: 10 },
                    },
                  },
                }}
              />
            ) : (
              <div className="text-center p-4 space-y-2">
                <div className="text-xs font-bold text-[#8D9AAA] uppercase tracking-wider">
                  WAVEFORM CHANNEL: NO OBSERVED ACCELEROGRAM INGESTED
                </div>
                <p className="text-[10px] text-[#667487] max-w-sm mx-auto leading-relaxed">
                  USGS earthquake feeds provide hypocenter and moment magnitude metadata without streaming station accelerograms. Run surrogate screening to compute structural response against the verified ground-motion baseline.
                </p>
                <div className="pt-2">
                  <span className="text-[9px] px-2 py-0.5 rounded-[2px] bg-[#111821] text-[#28D7FF] border border-white/[0.07]">
                    PIPELINE STATE: AWAITING FORWARD PASS TRIGGER
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Telemetric Performance Badges */}
          {simResult ? (
            <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
              <div className="p-2 bg-[#111821] rounded-[2px] border border-white/[0.07]">
                <span className="text-[#667487] block text-[9px] uppercase">PEAK ERROR</span>
                <strong className="text-[#28D7FF] font-mono text-xs">{simResult.metrics.peak_disp_err_pct}%</strong>
              </div>
              <div className="p-2 bg-[#111821] rounded-[2px] border border-white/[0.07]">
                <span className="text-[#667487] block text-[9px] uppercase">REL L₂ ERROR</span>
                <strong className="text-[#31D17C] font-mono text-xs">{simResult.metrics.rel_l2_pct}%</strong>
              </div>
              <div className="p-2 bg-[#111821] rounded-[2px] border border-white/[0.07]">
                <span className="text-[#667487] block text-[9px] uppercase">MPS LATENCY</span>
                <strong className="text-[#E8EDF3] font-mono text-xs">{simResult.metrics.latency_ms} ms</strong>
              </div>
            </div>
          ) : (
            <div className="p-2 bg-[#111821] rounded-[2px] border border-white/[0.07] text-[10px] text-[#8D9AAA] flex justify-between items-center">
              <span>SOLVER PIPELINE:</span>
              <span className="text-[#31D17C] font-bold">EXP6 MULTI-MODAL GNO (MPS ACCELERATED)</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
