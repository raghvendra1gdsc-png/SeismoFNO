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
      if (path.includes("command") || search.includes("command")) return "command_center";
      if (path.includes("twin") || search.includes("twin")) return "structural_twin";
      if (path.includes("intel") || search.includes("intel")) return "earthquake_intel";
      if (path.includes("validation") || search.includes("validation")) return "model_validation";
      if (path.includes("scenario") || search.includes("scenario")) return "scenario_lab";
      if (path.includes("explain") || search.includes("explain")) return "explainability";
      return "research_demo";
    }
    return "research_demo";
  });

  const handleSelectTab = useCallback((tab: WorkspaceTab) => {
    setActiveTab(tab);
    if (typeof window !== "undefined") {
      if (tab === "research_demo") {
        window.history.pushState(null, "", "/demo");
      } else if (tab === "live_earthquake") {
        window.history.pushState(null, "", "/live");
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
      console.error("Ground truth simulation failed:", err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedEarthquakeId, pgaG, T0, damping, uy, alpha, materialType]);

  // Handle Selection Updates
  const handleSelectEarthquake = (id: string, name: string) => {
    setSelectedEarthquakeId(id);
    setSelectedEarthquakeName(name);
  };

  const handleSelectBuilding = (bld: BuildingArchetype) => {
    setSelectedBuildingId(bld.id);
    setSelectedBuildingName(bld.name);
    setT0(bld.fundamental_period_s);
    setDamping(bld.damping_ratio);
    setUy(bld.yield_displacement_m);
    setAlpha(bld.post_yield_ratio);
    setMaterialType(bld.material_type === "elastic" ? "elastic" : "bilinear");
  };

  // Current Scenario Payload for Scenario Lab
  const currentScenarioPayload: ScenarioInputPayload = {
    earthquake_id: selectedEarthquakeId,
    pga_g: pgaG,
    T0,
    damping_ratio: damping,
    yield_displacement_m: uy,
    post_yield_ratio: alpha,
    material_type: materialType,
    stride: 2,
  };

  return (
    <div className="min-h-screen bg-[#F4F4F4] text-[#161616] flex flex-col font-sans">
      {/* Top Application Header Bar */}
      <header className="bg-white border-b border-[#E0E0E0] px-4 py-2.5 flex items-center justify-between sticky top-0 z-30 select-none">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-[#0F62FE]"></span>
            <span className="text-xs font-mono font-bold tracking-widest text-[#161616] uppercase">
              SEISMOFNO
            </span>
          </div>
          <div className="h-3 w-px bg-[#E0E0E0]"></div>
          <div className="text-[11px] font-sans text-[#525252] tracking-wide hidden sm:inline">
            Physics-Grounded Seismic Neural Operator Workstation
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono">
          <span className="px-2 py-0.5 rounded bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] text-[10px] font-semibold">
            {systemInfo?.device ? `DEVICE: ${systemInfo.device.toUpperCase()}` : "DEVICE: ACCELERATED"}
          </span>
          <span className="px-2 py-0.5 rounded bg-[#DEFBE6] text-[#198038] border border-[#6FDC8C] text-[10px] font-semibold">
            SURROGATE READY
          </span>
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
        <main className="flex-1 overflow-y-auto bg-[#F4F4F4]">
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

      {/* Bottom Status Bar */}
      <footer className="h-7 bg-white border-t border-[#E0E0E0] px-4 flex items-center justify-between text-[11px] font-mono text-[#525252] select-none z-20">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#198038]"></span>
            <span className="text-[#525252]">SYSTEM:</span>
            <span className="text-[#161616] font-semibold uppercase">OPERATIONAL</span>
          </div>

          <div className="h-3 w-px bg-[#E0E0E0]"></div>

          <div className="flex items-center space-x-1.5">
            <span className="text-[#525252]">SURROGATE:</span>
            <span className="text-[#0F62FE] font-semibold">
              {prediction ? `${prediction.inference_time_ms.toFixed(2)} ms` : "< 2.0 ms"}
            </span>
          </div>

          <div className="h-3 w-px bg-[#E0E0E0]"></div>

          <div className="flex items-center space-x-1.5">
            <span className="text-[#525252]">SOLVER:</span>
            <span className="text-[#161616]">OpenSeesPy SDOF NLTHA Available</span>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1.5">
            <span className="text-[#525252]">CATALOG:</span>
            <span className="text-[#161616]">{selectedEarthquakeName}</span>
          </div>
          <div className="h-3 w-px bg-[#E0E0E0]"></div>
          <span className="text-[#198038] font-bold">RESEARCH CORE LOCKED</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
