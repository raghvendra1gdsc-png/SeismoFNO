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
import {
  Activity,
  Globe2,
  Building2,
  AlertTriangle,
  ArrowRight,
  Radio,
  RefreshCw,
  FileText,
  Info,
  ExternalLink,
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

// Scenario Default Coordinates (e.g., Structural Lab in California or New Delhi)
const DEFAULT_SCENARIO = {
  name: "5-Story Benchmark Frame (Caltech/Pasadena Reference)",
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

  // Analysis / Research State
  const [isEventLoadedIntoAnalysis, setIsEventLoadedIntoAnalysis] = useState<boolean>(false);
  const [structures, setStructures] = useState<DemoStructure[]>([]);
  const [selectedStructureId, setSelectedStructureId] = useState<string>("5S_T055");
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

        // Pre-select 5-Story structure
        const fiveStory = structs.find((s) => s.n_stories === 5);
        if (fiveStory) {
          setSelectedStructureId(fiveStory.archetype_id);
        }
      } catch (err) {
        console.error("Error loading research catalogs:", err);
      }
    }
    initResearchData();
  }, []);

  // Filtered Events
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

  // Active Selected Event
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

  // Trajectory Chart Data
  const trajectoryChartData = useMemo(() => {
    if (!simResult) return null;
    return {
      labels: simResult.time.map((t: number) => t.toFixed(2)),
      datasets: [
        {
          label: "OpenSeesPy Ground Truth (NLTHA)",
          data: simResult.u_true.map((v: number) => v * 1000.0),
          borderColor: "#161616",
          borderDash: [5, 4],
          borderWidth: 2.0,
          pointRadius: 0,
          tension: 0.1,
        },
        {
          label: `EXP6 Multi-Modal GNO (${simResult.data_provenance.prediction})`,
          data: simResult.u_pred.map((v: number) => v * 1000.0),
          borderColor: "#0F62FE",
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.1,
        },
      ],
    };
  }, [simResult]);

  // Scientific SVG Map Projection (Equirectangular / Plate Carrée)
  const mapWidth = 800;
  const mapHeight = 400;

  const projectToMap = (lon: number, lat: number) => {
    // Standard Plate Carrée: lon [-180, 180] -> [0, mapWidth], lat [90, -90] -> [0, mapHeight]
    const x = ((lon + 180) / 360) * mapWidth;
    const y = ((90 - lat) / 180) * mapHeight;
    return { x, y };
  };

  const scenarioPos = projectToMap(DEFAULT_SCENARIO.lon, DEFAULT_SCENARIO.lat);
  const selectedEventPos = selectedEvent
    ? projectToMap(selectedEvent.longitude, selectedEvent.latitude)
    : null;

  return (
    <div className="w-full min-h-screen bg-[#F4F4F4] text-[#161616] p-4 md:p-8 space-y-6 font-sans">
      {/* ----------------------------------------------------------------- */}
      {/* 1. TOP HEADER & PROVENANCE BAR                                    */}
      {/* ----------------------------------------------------------------- */}
      <header className="bg-white border border-[#E0E0E0] p-6 rounded flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="text-xs font-mono tracking-widest text-[#0F62FE] font-bold uppercase">
              SEISMOFNO RESEARCH DEMO WORKSPACE
            </span>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] font-semibold">
              00 LIVE EARTHQUAKE
            </span>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#F4F4F4] text-[#525252] border border-[#E0E0E0]">
              SOURCE: USGS PUBLIC GEOJSON FEED
            </span>
          </div>
          <h1 className="text-2xl font-bold font-mono tracking-tight text-[#161616]">
            Observed Earthquake Event Ingestion & Engineering Screening
          </h1>
          <p className="text-xs text-[#525252] mt-1 max-w-4xl leading-relaxed">
            Connecting real-world seismic observations from the USGS earthquake catalog to the frozen
            SeismoFNO multi-story surrogate. Explores rapid structural-response screening while maintaining
            strict scientific boundaries between catalog event metadata and dynamic acceleration waveforms.
          </p>
        </div>

        <div className="flex flex-col items-start md:items-end gap-1.5 shrink-0 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                feedData?.is_fallback ? "bg-[#B28600] animate-pulse" : "bg-[#198038]"
              }`}
            ></span>
            <span className={feedData?.is_fallback ? "text-[#B28600] font-bold" : "text-[#198038] font-bold"}>
              {feedData?.is_fallback ? "OFFLINE CACHED EVENT DATA" : "LIVE USGS FEED ACTIVE"}
            </span>
          </div>
          <div className="text-[#525252] text-[11px]">
            API Key: <span className="text-[#161616] font-semibold">None (Public Service)</span>
          </div>
          <div className="text-[#525252] text-[11px]">
            Last Sync: <span className="text-[#161616] font-mono">{feedData?.last_updated || "Syncing..."}</span>
          </div>
        </div>
      </header>

      {/* ----------------------------------------------------------------- */}
      {/* 2. SCIENTIFIC HONESTY & ETHICAL BOUNDARY ALERT                    */}
      {/* ----------------------------------------------------------------- */}
      <div className="bg-[#EDF5FF] border-l-4 border-[#0F62FE] p-4 rounded-r border border-t-[#D0E2FF] border-r-[#D0E2FF] border-b-[#D0E2FF] text-xs font-sans text-[#161616] space-y-1.5">
        <div className="flex items-center gap-2 font-mono font-bold text-[#0F62FE]">
          <Info size={16} className="text-[#0F62FE] shrink-0" />
          <span>RESEARCH PROTOCOL — STRICT SCIENTIFIC BOUNDARIES</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px] leading-relaxed pt-1">
          <div>
            <span className="font-bold text-[#161616] block mb-0.5 font-mono uppercase text-[10px]">
              Permitted Scientific Scope:
            </span>
            Observed earthquake ingestion, event discovery, rapid surrogate structural analysis, engineering screening, and comparative response estimation.
          </div>
          <div>
            <span className="font-bold text-[#DA1E28] block mb-0.5 font-mono uppercase text-[10px]">
              Strict System Disclaimers:
            </span>
            SeismoFNO does <strong>NOT</strong> predict earthquakes, does <strong>NOT</strong> perform earthquake forecasting, does <strong>NOT</strong> stream live building accelerometer telemetry, and does <strong>NOT</strong> issue automated structural safety certifications.
          </div>
        </div>
      </div>

      {/* ----------------------------------------------------------------- */}
      {/* 3. OFFLINE FALLBACK BANNER (WHEN USGS REMOTE IS UNAVAILABLE)       */}
      {/* ----------------------------------------------------------------- */}
      {feedData?.is_fallback && (
        <div className="bg-[#FFF8E1] border-l-4 border-[#B28600] p-4 rounded-r border border-[#F1C21B] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2.5">
            <AlertTriangle size={18} className="text-[#B28600] shrink-0" />
            <div>
              <span className="font-mono font-bold text-[#B28600]">
                USGS LIVE FEED UNAVAILABLE — SERVING VERIFIED CACHED EVENT DATA
              </span>
              <p className="text-[11px] text-[#525252] font-sans mt-0.5">
                The research demonstration is operating in zero-dependency offline mode. Displaying verified real-world historical earthquake events.
              </p>
            </div>
          </div>
          <button
            onClick={loadFeed}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-[#B28600] text-[#B28600] rounded text-xs font-mono font-bold hover:bg-[#F4F4F4] cursor-pointer transition shrink-0"
          >
            <RefreshCw size={12} />
            RETRY LIVE CONNECTION
          </button>
        </div>
      )}

      {/* ----------------------------------------------------------------- */}
      {/* 4. MAIN TECHNICAL WORKSTATION GRID                                */}
      {/* ----------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT / TOP: TECHNICAL EVENT TABLE & FILTERS (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
            {/* Table Controls & Filter Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E0E0E0] pb-3">
              <div className="flex items-center gap-2">
                <Radio size={16} className="text-[#0F62FE]" />
                <h2 className="text-sm font-bold font-mono text-[#161616] uppercase">
                  Live Earthquake Events ({filteredEvents.length})
                </h2>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={loadFeed}
                  disabled={isLoadingFeed}
                  className="flex items-center gap-1 px-2.5 py-1 text-xs font-mono rounded bg-[#F4F4F4] border border-[#E0E0E0] hover:bg-[#EAEAEA] cursor-pointer disabled:opacity-50"
                  title="Refresh USGS Feed"
                >
                  <RefreshCw size={12} className={isLoadingFeed ? "animate-spin text-[#0F62FE]" : ""} />
                  <span>SYNC</span>
                </button>
              </div>
            </div>

            {/* Filter Inputs Strip */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-xs font-mono">
              {/* Min Magnitude */}
              <div>
                <label className="block text-[10px] text-[#525252] font-bold uppercase mb-1">
                  MIN MAGNITUDE
                </label>
                <select
                  value={minMag}
                  onChange={(e) => setMinMag(parseFloat(e.target.value))}
                  className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded px-2 py-1.5 text-xs font-mono"
                >
                  <option value="0.0">All (M0.0+)</option>
                  <option value="2.5">M2.5+ (Minor)</option>
                  <option value="4.5">M4.5+ (Moderate)</option>
                  <option value="6.0">M6.0+ (Strong)</option>
                  <option value="7.0">M7.0+ (Major)</option>
                </select>
              </div>

              {/* Time Window */}
              <div>
                <label className="block text-[10px] text-[#525252] font-bold uppercase mb-1">
                  TIME WINDOW
                </label>
                <select
                  value={timeWindowHours}
                  onChange={(e) => setTimeWindowHours(parseFloat(e.target.value))}
                  className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded px-2 py-1.5 text-xs font-mono"
                >
                  <option value="1">Past 1 Hour</option>
                  <option value="6">Past 6 Hours</option>
                  <option value="24">Past 24 Hours</option>
                  <option value="168">Past 7 Days</option>
                </select>
              </div>

              {/* Radius from Scenario */}
              <div>
                <label className="block text-[10px] text-[#525252] font-bold uppercase mb-1">
                  RADIUS FILTER
                </label>
                <select
                  value={radiusFilterKm}
                  onChange={(e) =>
                    setRadiusFilterKm(e.target.value === "ALL" ? "ALL" : parseFloat(e.target.value))
                  }
                  className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded px-2 py-1.5 text-xs font-mono"
                >
                  <option value="ALL">Global (No Limit)</option>
                  <option value="500">Within 500 km</option>
                  <option value="1000">Within 1,000 km</option>
                  <option value="3000">Within 3,000 km</option>
                </select>
              </div>

              {/* Search Box */}
              <div>
                <label className="block text-[10px] text-[#525252] font-bold uppercase mb-1">
                  FILTER BY PLACE / ID
                </label>
                <input
                  type="text"
                  placeholder="e.g. California, Japan"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded px-2 py-1.5 text-xs font-mono"
                />
              </div>
            </div>

            {/* Technical Event Table */}
            <div className="border border-[#E0E0E0] rounded overflow-hidden">
              <div className="max-h-80 overflow-y-auto font-mono text-[11px]">
                <table className="w-full text-left border-collapse">
                  <thead className="bg-[#F4F4F4] border-b border-[#E0E0E0] text-[10px] font-bold text-[#525252] uppercase tracking-wider sticky top-0 z-10">
                    <tr>
                      <th className="p-2.5">TIME (UTC)</th>
                      <th className="p-2.5">MAG</th>
                      <th className="p-2.5">LOCATION</th>
                      <th className="p-2.5">DEPTH</th>
                      <th className="p-2.5">EVENT ID</th>
                      <th className="p-2.5">SOURCE</th>
                      <th className="p-2.5">STATUS</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E0E0E0] bg-white">
                    {filteredEvents.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="p-6 text-center text-xs text-[#525252]">
                          No earthquake events match the selected filter criteria.
                        </td>
                      </tr>
                    ) : (
                      filteredEvents.map((ev) => {
                        const isSelected = selectedEvent?.event_id === ev.event_id;
                        const magColor =
                          ev.magnitude >= 7.0
                            ? "text-[#DA1E28] font-bold"
                            : ev.magnitude >= 5.0
                            ? "text-[#B28600] font-semibold"
                            : "text-[#161616]";

                        return (
                          <tr
                            key={ev.event_id}
                            onClick={() => setSelectedEventId(ev.event_id)}
                            className={`cursor-pointer transition-colors ${
                              isSelected
                                ? "bg-[#EDF5FF] font-semibold border-l-4 border-l-[#0F62FE]"
                                : "hover:bg-[#F9F9F9] border-l-4 border-l-transparent"
                            }`}
                          >
                            <td className="p-2.5 whitespace-nowrap text-[#525252]">
                              {ev.origin_time.replace("T", " ").replace("Z", "")}
                            </td>
                            <td className={`p-2.5 whitespace-nowrap ${magColor}`}>
                              M{ev.magnitude.toFixed(1)}
                            </td>
                            <td className="p-2.5 truncate max-w-xs text-[#161616]" title={ev.location}>
                              {ev.location}
                            </td>
                            <td className="p-2.5 whitespace-nowrap text-[#525252]">
                              {ev.depth_km.toFixed(1)} km
                            </td>
                            <td className="p-2.5 whitespace-nowrap text-[#0F62FE] font-mono">
                              {ev.event_id}
                            </td>
                            <td className="p-2.5 whitespace-nowrap text-[#525252]">
                              {ev.source}
                            </td>
                            <td className="p-2.5 whitespace-nowrap">
                              <span className="px-1.5 py-0.5 rounded text-[9px] bg-[#F4F4F4] text-[#525252] border border-[#E0E0E0] uppercase font-bold">
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

            {/* Table Footer Provenance Notice */}
            <div className="flex items-center justify-between text-[11px] font-mono text-[#525252] pt-1">
              <span>
                Catalog: <strong>USGS Real-Time GeoJSON Feed</strong> (Open Public Domain)
              </span>
              <span>
                Selection: <strong className="text-[#161616]">{selectedEvent?.event_id || "None"}</strong>
              </span>
            </div>
          </div>

          {/* RESTRAINED SCIENTIFIC MAP (Vector SVG) */}
          <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-3">
            <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-2">
              <div className="flex items-center gap-2">
                <Globe2 size={16} className="text-[#0F62FE]" />
                <h3 className="text-xs font-bold font-mono text-[#161616] uppercase">
                  Global Epicenter Distribution & Scenario Proximity
                </h3>
              </div>
              <div className="text-[10px] font-mono text-[#525252]">
                Plate Carrée Projection · Graticule 30°
              </div>
            </div>

            {/* Technical SVG Canvas */}
            <div className="w-full bg-[#161616] rounded border border-[#393939] overflow-hidden p-2 relative">
              <svg
                viewBox={`0 0 ${mapWidth} ${mapHeight}`}
                className="w-full h-auto block select-none"
                style={{ maxHeight: "360px" }}
              >
                {/* 30-degree Graticules */}
                {[-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150].map((lon) => {
                  const x = ((lon + 180) / 360) * mapWidth;
                  return (
                    <line
                      key={`lon-${lon}`}
                      x1={x}
                      y1={0}
                      x2={x}
                      y2={mapHeight}
                      stroke="#262626"
                      strokeWidth="1"
                      strokeDasharray="2,4"
                    />
                  );
                })}
                {[-60, -30, 0, 30, 60].map((lat) => {
                  const y = ((90 - lat) / 180) * mapHeight;
                  return (
                    <line
                      key={`lat-${lat}`}
                      x1={0}
                      y1={y}
                      x2={mapWidth}
                      y2={y}
                      stroke="#262626"
                      strokeWidth="1"
                      strokeDasharray="2,4"
                    />
                  );
                })}

                {/* Equator & Prime Meridian Axis */}
                <line
                  x1={0}
                  y1={mapHeight / 2}
                  x2={mapWidth}
                  y2={mapHeight / 2}
                  stroke="#393939"
                  strokeWidth="1.2"
                />
                <line
                  x1={mapWidth / 2}
                  y1={0}
                  x2={mapWidth / 2}
                  y2={mapHeight}
                  stroke="#393939"
                  strokeWidth="1.2"
                />

                {/* Geodesic Connection Line to Scenario if Event Selected */}
                {selectedEventPos && (
                  <line
                    x1={scenarioPos.x}
                    y1={scenarioPos.y}
                    x2={selectedEventPos.x}
                    y2={selectedEventPos.y}
                    stroke="#0F62FE"
                    strokeWidth="1.8"
                    strokeDasharray="4,4"
                    className="animate-pulse"
                  />
                )}

                {/* Scenario Marker */}
                <g transform={`translate(${scenarioPos.x}, ${scenarioPos.y})`}>
                  <circle r="6" fill="#0F62FE" fillOpacity="0.4" stroke="#78A9FF" strokeWidth="1.5" />
                  <circle r="2.5" fill="#FFFFFF" />
                  <text
                    x="8"
                    y="4"
                    fill="#F4F4F4"
                    fontSize="9"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    Scenario Target
                  </text>
                </g>

                {/* Event Markers */}
                {filteredEvents.map((ev) => {
                  const pos = projectToMap(ev.longitude, ev.latitude);
                  const isSelected = selectedEvent?.event_id === ev.event_id;
                  const radius = Math.max(3.5, (ev.magnitude - 2.0) * 2.5);

                  // Color by focal depth: shallow (<30km: red/amber, intermediate: amber, deep: blue)
                  const markerColor =
                    ev.depth_km < 30.0 ? "#DA1E28" : ev.depth_km < 70.0 ? "#F1C21B" : "#4589FF";

                  return (
                    <g
                      key={ev.event_id}
                      transform={`translate(${pos.x}, ${pos.y})`}
                      className="cursor-pointer"
                      onClick={() => setSelectedEventId(ev.event_id)}
                    >
                      {isSelected && (
                        <circle
                          r={radius + 5}
                          fill="none"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeDasharray="2,2"
                        />
                      )}
                      <circle
                        r={radius}
                        fill={markerColor}
                        fillOpacity={isSelected ? 0.9 : 0.65}
                        stroke="#FFFFFF"
                        strokeWidth={isSelected ? 1.5 : 0.8}
                      />
                    </g>
                  );
                })}
              </svg>

              {/* Map Legend Overlay */}
              <div className="absolute bottom-3 left-3 bg-[#161616]/90 border border-[#393939] p-2 rounded text-[10px] font-mono text-[#C6C6C6] flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#DA1E28] inline-block"></span>
                  <span>Shallow (&lt;30 km)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#F1C21B] inline-block"></span>
                  <span>30–70 km</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#4589FF] inline-block"></span>
                  <span>Deep (&gt;70 km)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT: EVENT CONTEXT & STRUCTURAL ANALYSIS (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* A. EVENT CONTEXT CARD */}
          <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3">
              <h3 className="text-sm font-bold font-mono text-[#161616] flex items-center gap-2 uppercase">
                <FileText size={16} className="text-[#0F62FE]" />
                Event Context
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] font-bold">
                USGS OBSERVED
              </span>
            </div>

            {selectedEvent ? (
              <div className="space-y-3 font-mono text-xs">
                {/* Event Title & Source Badges */}
                <div className="bg-[#F4F4F4] p-3 rounded border border-[#E0E0E0] space-y-1.5">
                  <div className="text-[#525252] text-[10px] uppercase font-bold">EVENT LOCATION</div>
                  <div className="text-sm font-bold text-[#161616]">{selectedEvent.location}</div>
                  <div className="flex flex-wrap gap-1.5 pt-1 text-[10px]">
                    <span className="px-1.5 py-0.5 rounded bg-white text-[#161616] border border-[#E0E0E0]">
                      SOURCE: <strong>USGS</strong>
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-[#DEFBE6] text-[#198038] border border-[#6FDC8C]">
                      STATUS: <strong>OBSERVED EVENT</strong>
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-white text-[#525252] border border-[#E0E0E0]">
                      DATA TYPE: <strong>CATALOG / REAL-TIME FEED</strong>
                    </span>
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="p-2.5 bg-[#F4F4F4] rounded border border-[#E0E0E0]">
                    <span className="text-[#525252] block text-[10px] uppercase">MAGNITUDE</span>
                    <span className="text-base font-bold text-[#DA1E28]">M{selectedEvent.magnitude.toFixed(1)}</span>
                  </div>

                  <div className="p-2.5 bg-[#F4F4F4] rounded border border-[#E0E0E0]">
                    <span className="text-[#525252] block text-[10px] uppercase">FOCAL DEPTH</span>
                    <span className="text-base font-bold text-[#161616]">{selectedEvent.depth_km.toFixed(1)} km</span>
                  </div>

                  <div className="p-2.5 bg-[#F4F4F4] rounded border border-[#E0E0E0]">
                    <span className="text-[#525252] block text-[10px] uppercase">EPICENTER COORDS</span>
                    <span className="text-[#161616] font-semibold">
                      {selectedEvent.latitude.toFixed(3)}°, {selectedEvent.longitude.toFixed(3)}°
                    </span>
                  </div>

                  <div className="p-2.5 bg-[#F4F4F4] rounded border border-[#E0E0E0]">
                    <span className="text-[#525252] block text-[10px] uppercase">DISTANCE TO SCENARIO</span>
                    <span className="text-[#0F62FE] font-bold">
                      {selectedEvent.distance_km != null ? `${selectedEvent.distance_km.toLocaleString()} km` : "N/A"}
                    </span>
                  </div>
                </div>

                <div className="text-[11px] text-[#525252] space-y-1 pt-1">
                  <div>
                    Origin Time: <strong className="text-[#161616]">{selectedEvent.origin_time}</strong>
                  </div>
                  <div>
                    USGS Identifier:{" "}
                    <a
                      href={selectedEvent.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[#0F62FE] hover:underline font-mono inline-flex items-center gap-1"
                    >
                      {selectedEvent.event_id} <ExternalLink size={10} />
                    </a>
                  </div>
                </div>

                {/* Primary Action Button */}
                <button
                  onClick={() => setIsEventLoadedIntoAnalysis(true)}
                  className="w-full py-2.5 bg-[#0F62FE] hover:bg-[#0353E9] text-white font-mono font-bold rounded text-xs flex items-center justify-center gap-2 cursor-pointer transition shadow"
                >
                  <ArrowRight size={14} />
                  [ LOAD EVENT INTO ANALYSIS ]
                </button>
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-[#525252]">No event selected.</div>
            )}
          </div>

          {/* B. STRUCTURAL ANALYSIS CONTEXT (AFTER LOADING) */}
          {isEventLoadedIntoAnalysis && (
            <div className="bg-white border-2 border-[#0F62FE] rounded p-5 space-y-4 shadow-sm animate-fadeIn">
              <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3">
                <h3 className="text-sm font-bold font-mono text-[#161616] flex items-center gap-2 uppercase">
                  <Building2 size={16} className="text-[#0F62FE]" />
                  Structural Analysis Context
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#DEFBE6] text-[#198038] border border-[#6FDC8C] font-bold">
                  ACTIVE SCENARIO
                </span>
              </div>

              {/* Structure Selection & Modal Properties */}
              <div className="space-y-3 font-mono text-xs">
                <div>
                  <label className="block text-[10px] text-[#525252] font-bold uppercase mb-1">
                    SELECTED STRUCTURE (5-STORY EXAMPLE)
                  </label>
                  <select
                    value={selectedStructureId}
                    onChange={(e) => setSelectedStructureId(e.target.value)}
                    className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded px-2.5 py-1.5 text-xs font-mono"
                  >
                    {structures.map((s) => (
                      <option key={s.archetype_id} value={s.archetype_id}>
                        {s.archetype_id} — {s.n_stories} Stories ({s.category}, T1={s.T1_s.toFixed(2)}s)
                      </option>
                    ))}
                  </select>
                </div>

                {/* Modal Properties Callout */}
                <div className="p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] space-y-1.5">
                  <span className="text-[10px] text-[#525252] font-bold uppercase block">
                    THEORETICAL MODAL PROPERTIES (PRE-EARTHQUAKE INVARIANTS)
                  </span>
                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="bg-white p-1.5 rounded border border-[#E0E0E0]">
                      <span className="text-[10px] text-[#525252] block">T₁</span>
                      <strong className="text-[#161616]">{currentStructure?.T1_s.toFixed(3)} s</strong>
                    </div>
                    <div className="bg-white p-1.5 rounded border border-[#E0E0E0]">
                      <span className="text-[10px] text-[#525252] block">T₂</span>
                      <strong className="text-[#161616]">{currentStructure?.T2_s.toFixed(3)} s</strong>
                    </div>
                    <div className="bg-white p-1.5 rounded border border-[#E0E0E0]">
                      <span className="text-[10px] text-[#525252] block">T₃</span>
                      <strong className="text-[#161616]">{currentStructure?.T3_s.toFixed(3)} s</strong>
                    </div>
                  </div>
                </div>

                {/* GROUND-MOTION INPUT AVAILABILITY CHECK */}
                <div className="p-3.5 bg-[#FFF8E1] border border-[#F1C21B] rounded space-y-2 text-xs font-sans">
                  <div className="flex items-center gap-2 font-mono font-bold text-[#B28600]">
                    <AlertTriangle size={15} />
                    <span>GROUND-MOTION AVAILABILITY CHECK</span>
                  </div>
                  <p className="text-[11px] text-[#525252] leading-relaxed">
                    <strong>Event metadata loaded.</strong> Compatible acceleration waveform unavailable for direct SeismoFNO inference.
                  </p>
                  <p className="text-[11px] text-[#525252] leading-relaxed">
                    USGS catalog feeds provide earthquake occurrence parameters ($M_w$, hypocenter, origin time) but do not stream the verified baseline-corrected time-series $a_g(t)$ required by physics-conditioned neural operators.
                  </p>
                </div>

                {/* HISTORICAL ACCELERATION RECORD SELECTION */}
                <div className="space-y-1.5 pt-1">
                  <label className="block text-[10px] text-[#525252] font-bold font-mono uppercase">
                    CONTINUE WITH VERIFIED ACCELERATION RECORD (RESEARCH DATASET)
                  </label>
                  <select
                    value={selectedHistoricalRecordId}
                    onChange={(e) => setSelectedHistoricalRecordId(e.target.value)}
                    className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded px-2.5 py-1.5 text-xs font-mono"
                  >
                    {historicalRecords.map((rec) => (
                      <option key={rec.record_id} value={rec.record_id}>
                        {rec.record_id} — {rec.event_name} (PGA: {rec.pga_g}g, {rec.station})
                      </option>
                    ))}
                  </select>
                </div>

                {/* VISUAL PROVENANCE CHAIN */}
                <div className="p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] space-y-1.5 font-mono text-[10px]">
                  <span className="text-[#525252] font-bold uppercase block">SCIENTIFIC PROVENANCE CHAIN</span>
                  <div className="flex flex-wrap items-center gap-1.5 text-[#161616]">
                    <span className="px-1.5 py-0.5 bg-white border border-[#E0E0E0] rounded">USGS EVENT</span>
                    <span className="text-[#0F62FE]">↓</span>
                    <span className="px-1.5 py-0.5 bg-white border border-[#E0E0E0] rounded">METADATA</span>
                    <span className="text-[#0F62FE]">↓</span>
                    <span className="px-1.5 py-0.5 bg-white border border-[#E0E0E0] rounded">WAVEFORM CHECK</span>
                    <span className="text-[#0F62FE]">↓</span>
                    <span className="px-1.5 py-0.5 bg-[#EDF5FF] border border-[#A6C8FF] text-[#0F62FE] font-bold rounded">
                      VERIFIED ACCEL
                    </span>
                    <span className="text-[#0F62FE]">↓</span>
                    <span className="px-1.5 py-0.5 bg-white border border-[#E0E0E0] rounded">STRUCTURAL MODEL</span>
                    <span className="text-[#0F62FE]">↓</span>
                    <span className="px-1.5 py-0.5 bg-[#DEFBE6] border border-[#6FDC8C] text-[#198038] font-bold rounded">
                      SEISMOFNO
                    </span>
                    <span className="text-[#0F62FE]">↓</span>
                    <span className="px-1.5 py-0.5 bg-white border border-[#E0E0E0] rounded">RESPONSE</span>
                  </div>
                </div>

                {/* RESEARCH MODE INDICATOR */}
                <div className="p-3 bg-white rounded border border-[#0F62FE] space-y-1 font-mono text-[11px]">
                  <div className="flex items-center justify-between text-[#0F62FE] font-bold text-[10px] uppercase">
                    <span>RESEARCH MODE ACTIVE</span>
                    <span>EXP6 FROZEN CHECKPOINT</span>
                  </div>
                  <div className="text-[#525252] text-[10px] leading-relaxed">
                    <strong>REAL-WORLD EVENT:</strong> Observed USGS metadata ({selectedEvent?.location || "Observed Event"})<br />
                    + <strong>RESEARCH GROUND MOTION:</strong> Verified PEER {selectedHistoricalRecordId}<br />
                    + <strong>FROZEN SEISMOFNO:</strong> Multi-Modal GNO (Apple Silicon MPS)
                  </div>
                </div>

                {/* TRIGGER INFERENCE BUTTON */}
                <button
                  onClick={executeSimulation}
                  disabled={isLoadingSim}
                  className="w-full py-2.5 bg-[#198038] hover:bg-[#0E6027] text-white font-mono font-bold rounded text-xs flex items-center justify-center gap-2 cursor-pointer transition shadow"
                >
                  <Activity size={14} className={isLoadingSim ? "animate-spin" : ""} />
                  {isLoadingSim ? "COMPUTING SEISMOFNO FORWARD PASS..." : "EXECUTE SURROGATE RESPONSE SIMULATION"}
                </button>

                {simError && (
                  <div className="text-xs text-[#DA1E28] bg-[#FFD7D9] p-2.5 rounded border border-[#FF8389]">
                    {simError}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ----------------------------------------------------------------- */}
      {/* 5. RESPONSE RESULTS & SCIENTIFIC COMPARISON                       */}
      {/* ----------------------------------------------------------------- */}
      {simResult && (
        <section className="bg-white border border-[#E0E0E0] rounded p-6 space-y-4 animate-fadeIn">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-[#E0E0E0] pb-3">
            <div>
              <div className="flex items-center gap-2">
                <Activity size={18} className="text-[#0F62FE]" />
                <h3 className="text-base font-bold font-mono text-[#161616]">
                  Simulated Structural Displacement u(t) — Roof Floor {currentStructure.n_stories}
                </h3>
              </div>
              <p className="text-xs text-[#525252] font-sans mt-0.5">
                Surrogate forward pass conditioned on {currentStructure.archetype_id} modal properties ($T_1={currentStructure.T1_s.toFixed(2)}$s) and verified record {selectedHistoricalRecordId}.
              </p>
            </div>

            {/* Metrics Callout Badges */}
            <div className="flex items-center gap-3 text-xs font-mono">
              <div className="bg-[#F4F4F4] px-3 py-1.5 rounded border border-[#E0E0E0]">
                <span className="text-[#525252] text-[10px] block">PEAK ERROR</span>
                <strong className="text-[#0F62FE] font-mono">{simResult.metrics.peak_disp_err_pct}%</strong>
              </div>
              <div className="bg-[#F4F4F4] px-3 py-1.5 rounded border border-[#E0E0E0]">
                <span className="text-[#525252] text-[10px] block">REL L₂ ERROR</span>
                <strong className="text-[#198038] font-mono">{simResult.metrics.rel_l2_pct}%</strong>
              </div>
              <div className="bg-[#F4F4F4] px-3 py-1.5 rounded border border-[#E0E0E0]">
                <span className="text-[#525252] text-[10px] block">LATENCY</span>
                <strong className="text-[#161616] font-mono">{simResult.metrics.latency_ms} ms</strong>
              </div>
            </div>
          </div>

          {/* Trajectory Comparison Chart */}
          <div className="h-72 w-full bg-[#F4F4F4] p-3 rounded border border-[#E0E0E0]">
            {trajectoryChartData && (
              <Line
                data={trajectoryChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  animation: { duration: 300 },
                  scales: {
                    x: {
                      title: { display: true, text: "Time (seconds)", font: { family: "monospace", size: 11 } },
                      grid: { color: "#E0E0E0" },
                    },
                    y: {
                      title: { display: true, text: "Roof Displacement u(t) (mm)", font: { family: "monospace", size: 11 } },
                      grid: { color: "#E0E0E0" },
                    },
                  },
                  plugins: {
                    legend: { position: "top" as const, labels: { font: { family: "monospace", size: 11 } } },
                  },
                }}
              />
            )}
          </div>

          {/* Operational Honesty Notice */}
          <div className="p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] text-[11px] font-mono text-[#525252] flex items-center justify-between">
            <span>
              DATA PROVENANCE: <strong>{simResult.data_provenance.prediction}</strong> (Local MPS Kernel) vs{" "}
              <strong>{simResult.data_provenance.ground_truth}</strong>
            </span>
            <span className="text-[#198038] font-bold">RESEARCH VERIFICATION ONLY</span>
          </div>
        </section>
      )}
    </div>
  );
};
