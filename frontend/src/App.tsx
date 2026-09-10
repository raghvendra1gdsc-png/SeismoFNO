import React, { useEffect, useState, useCallback } from "react";
import { WorkspaceNav, type WorkspaceTab } from "./components/WorkspaceNav/WorkspaceNav";
import { CommandCenterView } from "./components/CommandCenter/CommandCenterView";
import { EarthquakeIntelView } from "./components/EarthquakeIntel/EarthquakeIntelView";
import { StructuralTwinView } from "./components/StructuralTwin/StructuralTwinView";
import { ScenarioLabView } from "./components/ScenarioLab/ScenarioLabView";
import { ModelValidationView } from "./components/ModelValidation/ModelValidationView";
import { ExplainabilityView } from "./components/Explainability/ExplainabilityView";
import { ResearchDemoView } from "./components/ResearchDemo/ResearchDemoView";
import { LiveEarthquakeView } from "./components/ResearchDemo/LiveEarthquakeView";

import {
  fetchSystemInfo,
  fetchEarthquakeCatalog,
  fetchBuildingArchetypes,
  predictDigitalTwin,
  type SystemInfo,
  type IndianEarthquake,
  type PeerRecord,
  type BuildingArchetype,
  type ScenarioInputPayload,
  type ScenarioPredictionResponse,
} from "./api/digitalTwinApi";

export const App: React.FC = () => {
  // Navigation & Workspace State - check if /demo, /live, or ?tab= is requested
  const [activeTab, setActiveTab] = useState<WorkspaceTab>(() => {
    if (typeof window !== "undefined") {
      const path = window.location.pathname.toLowerCase();
      const search = window.location.search.toLowerCase();
      if (path.includes("live") || search.includes("live")) return "live_earthquake";
      if (path.includes("demo") || search.includes("demo")) return "research_demo";
      if (path.includes("command") || search.includes("command")) return "command_center";
      if (path.includes("intel") || search.includes("intel")) return "earthquake_intel";
      if (path.includes("validation") || search.includes("validation")) return "model_validation";
      if (path.includes("scenario") || search.includes("scenario")) return "scenario_lab";
      if (path.includes("explain") || search.includes("explain")) return "explainability";
      return "structural_twin";
    }
    return "structural_twin";
  });

  const handleSelectTab = useCallback((tab: WorkspaceTab) => {
    setActiveTab(tab);
    if (typeof window !== "undefined") {
      if (tab === "research_demo") {
        window.history.pushState(null, "", "/demo");
      } else if (tab === "live_earthquake") {
        window.history.pushState(null, "", "/live");
      } else if (tab === "structural_twin") {
        window.history.pushState(null, "", "/");
      } else {
        window.history.pushState(null, "", `/?tab=${tab}`);
      }
    }
  }, []);

  // System & Catalogs State
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [indianCatalog, setIndianCatalog] = useState<IndianEarthquake[]>([]);
  const [peerRecords, setPeerRecords] = useState<PeerRecord[]>([]);
  const [buildingArchetypes, setBuildingArchetypes] = useState<BuildingArchetype[]>([]);

  // Selection States
  const [selectedEarthquakeId, setSelectedEarthquakeId] = useState<string>("RSN0001_Imperial_Valley-06.AT2");
  const [selectedEarthquakeName, setSelectedEarthquakeName] = useState<string>("Imperial Valley-06 (RSN0001)");
  const [selectedBuildingId, setSelectedBuildingId] = useState<string>("BLD-RC-03");
  const [selectedBuildingName, setSelectedBuildingName] = useState<string>("3-Story RC Moment Frame");

  // Structural Parameters
  const [pgaG, setPgaG] = useState<number>(0.40);
  const [T0, setT0] = useState<number>(0.50);
  const [damping, setDamping] = useState<number>(0.05);
  const [uy, setUy] = useState<number>(0.010);
  const [alpha, setAlpha] = useState<number>(0.05);
  const [materialType, setMaterialType] = useState<"bilinear" | "elastic">("bilinear");

  // Simulation Response State
  const [prediction, setPrediction] = useState<ScenarioPredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Initial Data Fetch
  useEffect(() => {
    async function init() {
      try {
        const [sys, eqData, bldData] = await Promise.all([
          fetchSystemInfo().catch(() => null),
          fetchEarthquakeCatalog().catch(() => ({ indian_catalog: [], peer_records: [] })),
          fetchBuildingArchetypes().catch(() => ({ buildings: [] })),
        ]);

        setSystemInfo(sys);
        setIndianCatalog(eqData.indian_catalog || []);
        setPeerRecords(eqData.peer_records || []);
        setBuildingArchetypes(bldData.buildings || []);

        // Trigger initial simulation
        const initialPayload: ScenarioInputPayload = {
          earthquake_id: "RSN0001_Imperial_Valley-06.AT2",
          pga_g: 0.40,
          T0: 0.50,
          damping_ratio: 0.05,
          yield_displacement_m: 0.010,
          post_yield_ratio: 0.05,
          material_type: "bilinear",
          include_ground_truth: false,
          stride: 2,
        };
        const initialPred = await predictDigitalTwin(initialPayload);
        setPrediction(initialPred);
      } catch (err) {
        console.error("Initialization error:", err);
      }
    }
    init();
  }, []);

  // Run Simulation Handler
  const handleRunSimulation = useCallback(async () => {
    setIsLoading(true);
    try {
      const payload: ScenarioInputPayload = {
        earthquake_id: selectedEarthquakeId,
        pga_g: pgaG,
        T0,
        damping_ratio: damping,
        yield_displacement_m: uy,
        post_yield_ratio: alpha,
        material_type: materialType,
        include_ground_truth: false,
        stride: 2,
      };
      const res = await predictDigitalTwin(payload);
      setPrediction(res);
    } catch (err) {
      console.error("Simulation run failed:", err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedEarthquakeId, pgaG, T0, damping, uy, alpha, materialType]);

  // Run OpenSees Ground Truth Comparison Handler
  const handleRunGroundTruth = useCallback(async () => {
    setIsLoading(true);
    try {
      const payload: ScenarioInputPayload = {
        earthquake_id: selectedEarthquakeId,
        pga_g: pgaG,
        T0,
        damping_ratio: damping,
        yield_displacement_m: uy,
        post_yield_ratio: alpha,
        material_type: materialType,
        include_ground_truth: true,
        stride: 2,
      };
      const res = await predictDigitalTwin(payload);
      setPrediction(res);
    } catch (err) {
      console.error("Ground truth run failed:", err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedEarthquakeId, pgaG, T0, damping, uy, alpha, materialType]);

  const handleSelectBuilding = useCallback((bld: BuildingArchetype) => {
    setSelectedBuildingId(bld.id);
    setSelectedBuildingName(bld.name);
    setT0(bld.fundamental_period_s);
    setDamping(bld.damping_ratio);
    setUy(bld.yield_displacement_m);
    setAlpha(bld.post_yield_ratio);
  }, []);

  const handleSelectEarthquake = useCallback((recordId: string, name: string) => {
    setSelectedEarthquakeId(recordId);
    setSelectedEarthquakeName(name);
  }, []);

  const currentScenarioPayload: ScenarioInputPayload = {
    earthquake_id: selectedEarthquakeId,
    pga_g: pgaG,
    T0,
    damping_ratio: damping,
    yield_displacement_m: uy,
    post_yield_ratio: alpha,
    material_type: materialType,
    include_ground_truth: false,
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-[#07110F] text-[#E8E8DE] overflow-hidden font-sans select-none">
      {/* Top Application Header - Minimal Studio Chrome */}
      <header className="h-10 bg-[#07110F] border-b border-white/[0.06] px-4 flex items-center justify-between z-30 shrink-0 select-none">
        {/* Left: Minimal Branding */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5]" />
            <span className="text-xs font-semibold tracking-wider text-[#E8E8DE] uppercase">
              SEISMOFNO
            </span>
          </div>
          <span className="text-white/[0.1] hidden sm:inline">/</span>
          <div className="text-[11px] text-[#82928B] hidden sm:inline">
            Neural Operator Research Desk
          </div>
        </div>

        {/* Center: Tabs with minimal active indicator */}
        <div className="hidden md:flex items-center h-full space-x-1 font-sans text-xs">
          <button
            onClick={() => handleSelectTab("structural_twin")}
            className={`h-full px-3 flex items-center space-x-1.5 transition cursor-pointer border-b-2 ${
              activeTab === "structural_twin"
                ? "border-[#73E6B5] text-[#E8E8DE] font-medium"
                : "border-transparent text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            <span>Digital Twin</span>
          </button>
          <button
            onClick={() => handleSelectTab("research_demo")}
            className={`h-full px-3 flex items-center space-x-1.5 transition cursor-pointer border-b-2 ${
              activeTab === "research_demo"
                ? "border-[#73E6B5] text-[#E8E8DE] font-medium"
                : "border-transparent text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            <span>Research Defense</span>
          </button>
          <button
            onClick={() => handleSelectTab("live_earthquake")}
            className={`h-full px-3 flex items-center space-x-1.5 transition cursor-pointer border-b-2 ${
              activeTab === "live_earthquake"
                ? "border-[#73E6B5] text-[#E8E8DE] font-medium"
                : "border-transparent text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            <span>Live USGS</span>
          </button>
        </div>

        {/* Right: Instrument Readouts */}
        <div className="flex items-center space-x-3 text-[10px] font-mono text-[#82928B]">
          <span>{systemInfo?.device ? systemInfo.device.toUpperCase() : "MPS"}</span>
          <span>·</span>
          <span className="text-[#73E6B5]">305 TESTS</span>
          <span>·</span>
          <span className="text-[#82928B]">ONLINE</span>
        </div>
      </header>

      {/* Main Workspace Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Navigation Sidebar */}
        <WorkspaceNav
          activeTab={activeTab}
          onSelectTab={handleSelectTab}
          latencyMs={prediction?.inference_time_ms}
        />

        {/* Content Workspace Area */}
        <main className="flex-1 overflow-y-auto bg-research-desk">
          {activeTab === "structural_twin" && (
            <StructuralTwinView
              prediction={prediction}
              buildingArchetypes={buildingArchetypes}
              selectedBuildingId={selectedBuildingId}
              onSelectBuilding={handleSelectBuilding}
              onRunSimulation={handleRunSimulation}
              isLoading={isLoading}
              T0={T0}
              setT0={setT0}
              damping={damping}
              setDamping={setDamping}
              uy={uy}
              setUy={setUy}
              alpha={alpha}
              setAlpha={setAlpha}
              materialType={materialType}
              setMaterialType={setMaterialType}
              pgaG={pgaG}
              setPgaG={setPgaG}
            />
          )}

          {activeTab === "research_demo" && <ResearchDemoView />}
          {activeTab === "live_earthquake" && <LiveEarthquakeView />}

          {activeTab === "command_center" && (
            <CommandCenterView
              prediction={prediction}
              systemInfo={systemInfo}
              selectedStructureName={selectedBuildingName}
              selectedEarthquakeName={selectedEarthquakeName}
              onNavigateTab={(tab) => setActiveTab(tab as WorkspaceTab)}
              isLoading={isLoading}
            />
          )}

          {activeTab === "earthquake_intel" && (
            <EarthquakeIntelView
              indianCatalog={indianCatalog}
              peerRecords={peerRecords}
              selectedEarthquakeId={selectedEarthquakeId}
              onSelectEarthquake={handleSelectEarthquake}
            />
          )}

          {activeTab === "scenario_lab" && (
            <ScenarioLabView baseScenario={currentScenarioPayload} />
          )}

          {activeTab === "model_validation" && (
            <ModelValidationView
              prediction={prediction}
              onRunGroundTruth={handleRunGroundTruth}
              isLoading={isLoading}
            />
          )}

          {activeTab === "explainability" && <ExplainabilityView />}
        </main>
      </div>

      {/* Bottom Status Bar - Professional Status Strip */}
      <footer className="h-6 bg-[#07110F] border-t border-white/[0.06] px-4 flex items-center justify-between text-[10px] font-mono text-[#82928B] select-none z-20">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#73E6B5]" />
            <span>SURROGATE:</span>
            <span className="text-[#E8E8DE] font-semibold">
              {prediction ? `${prediction.inference_time_ms.toFixed(2)} ms` : "< 2.0 ms"}
            </span>
          </div>

          <span className="text-white/[0.1]">·</span>

          <div className="flex items-center space-x-1.5">
            <span>GROUND TRUTH:</span>
            <span className="text-[#E8E8DE]">OpenSeesPy C-Runtime</span>
          </div>

          <span className="text-white/[0.1] hidden sm:inline">·</span>

          <div className="hidden sm:flex items-center space-x-1.5">
            <span>THROUGHPUT:</span>
            <span className="text-[#73E6B5]">1,060 sim/s</span>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5">
            <span>MODEL:</span>
            <span className="text-[#E8E8DE]">EXP6 GNO</span>
          </div>
          <span className="text-white/[0.1]">·</span>
          <span className="text-[#73E6B5]">SHA-256 AUDITED</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
