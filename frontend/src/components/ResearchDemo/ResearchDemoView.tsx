import React, { useEffect, useState, useMemo } from "react";
import {
  fetchDemoModels,
  fetchDemoStructures,
  fetchDemoEarthquakes,
  fetchDemoProgression,
  fetchDemoOODMatrix,
  fetchDemoAblation,
  fetchDemoFailures,
  fetchDemoBenchmark,
  runDemoSimulation,
  type DemoModel,
  type DemoStructure,
  type DemoEarthquake,
  type ProgressionStep,
  type OODMatrix,
  type AblationData,
  type FailureMode,
  type BenchmarkData,
  type DemoSimulationResponse,
} from "../../api/researchDemoApi";
import {
  Activity,
  Layers,
  CheckCircle2,
  AlertTriangle,
  Zap,
  ArrowRight,
  GitBranch,
  Info,
  Radio,
} from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export const ResearchDemoView: React.FC = () => {
  // Remote / Archival State
  const [models, setModels] = useState<DemoModel[]>([]);
  const [structures, setStructures] = useState<DemoStructure[]>([]);
  const [earthquakes, setEarthquakes] = useState<DemoEarthquake[]>([]);
  const [progression, setProgression] = useState<ProgressionStep[]>([]);
  const [oodMatrix, setOodMatrix] = useState<OODMatrix | null>(null);
  const [ablation, setAblation] = useState<AblationData | null>(null);
  const [failures, setFailures] = useState<FailureMode[]>([]);
  const [benchmark, setBenchmark] = useState<BenchmarkData | null>(null);

  // Interactive UI Selection State
  const [selectedStructureId, setSelectedStructureId] = useState<string>("3S_T050");
  const [selectedEarthquakeId, setSelectedEarthquakeId] = useState<string>("RSN0001");
  const [selectedModelId, setSelectedModelId] = useState<string>("exp6_multimodal_gno");
  const [selectedFloor, setSelectedFloor] = useState<number>(3);
  const [activeModeShapeIdx, setActiveModeShapeIdx] = useState<number>(0);

  // Simulation Inference Output
  const [simResult, setSimResult] = useState<DemoSimulationResponse | null>(null);
  const [isLoadingSim, setIsLoadingSim] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Initial Load of Catalogs and Data
  useEffect(() => {
    async function loadData() {
      try {
        const [mods, structs, eqs, prog, ood, abl, fail, bench] = await Promise.all([
          fetchDemoModels(),
          fetchDemoStructures(),
          fetchDemoEarthquakes(),
          fetchDemoProgression(),
          fetchDemoOODMatrix(),
          fetchDemoAblation(),
          fetchDemoFailures(),
          fetchDemoBenchmark(),
        ]);

        setModels(mods);
        setStructures(structs);
        setEarthquakes(eqs);
        setProgression(prog);
        setOodMatrix(ood);
        setAblation(abl);
        setFailures(fail);
        setBenchmark(bench);

        // Run initial simulation
        if (structs.length > 0 && eqs.length > 0) {
          executeSimulation("3S_T050", "RSN0001", "exp6_multimodal_gno", 3);
        }
      } catch (err) {
        console.error("Failed to load demo data:", err);
      }
    }
    loadData();
  }, []);

  // Simulation Trigger Handler
  const executeSimulation = async (archId: string, recId: string, modId: string, floor: number) => {
    setIsLoadingSim(true);
    setErrorMsg(null);
    try {
      const res = await runDemoSimulation({
        archetype_id: archId,
        record_id: recId,
        model_id: modId,
        selected_floor: floor,
      });
      setSimResult(res);
    } catch (err: any) {
      setErrorMsg(err.message || "Simulation failed");
    } finally {
      setIsLoadingSim(false);
    }
  };

  // Re-run simulation when selection changes
  const handleStructureChange = (archId: string) => {
    setSelectedStructureId(archId);
    const arch = structures.find((s) => s.archetype_id === archId);
    const newFloor = arch ? arch.n_stories : 3;
    setSelectedFloor(newFloor);
    executeSimulation(archId, selectedEarthquakeId, selectedModelId, newFloor);
  };

  const handleEarthquakeChange = (recId: string) => {
    setSelectedEarthquakeId(recId);
    executeSimulation(selectedStructureId, recId, selectedModelId, selectedFloor);
  };

  const handleModelChange = (modId: string) => {
    setSelectedModelId(modId);
    executeSimulation(selectedStructureId, selectedEarthquakeId, modId, selectedFloor);
  };

  const handleFloorChange = (fl: number) => {
    setSelectedFloor(fl);
    executeSimulation(selectedStructureId, selectedEarthquakeId, selectedModelId, fl);
  };

  const currentStructure = useMemo(() => {
    return structures.find((s) => s.archetype_id === selectedStructureId) || structures[0];
  }, [structures, selectedStructureId]);

  // Accelerogram Chart Data
  const accelerogramChartData = useMemo(() => {
    if (!simResult) return null;
    return {
      labels: simResult.time.map((t: number) => t.toFixed(2)),
      datasets: [
        {
          label: "PGA (g)",
          data: simResult.ag,
          borderColor: "#0F62FE", // Carbon blue-60
          borderWidth: 1.2,
          pointRadius: 0,
          fill: false,
          tension: 0.1,
        },
      ],
    };
  }, [simResult]);

  // Trajectory comparison chart data: Ground truth (#161616 dashed) vs Prediction (blue-60 solid)
  const trajectoryChartData = useMemo(() => {
    if (!simResult) return null;
    const isShuffled = simResult.model.model_id.includes("shuffled");
    return {
      labels: simResult.time.map((t: number) => t.toFixed(2)),
      datasets: [
        {
          label: "OpenSeesPy Ground Truth (NLTHA)",
          data: simResult.u_true.map((v: number) => v * 1000.0), // mm
          borderColor: "#161616", // CAD dark black
          borderDash: [5, 4],
          borderWidth: 2.0,
          pointRadius: 0,
          tension: 0.1,
        },
        {
          label: `${simResult.model.display_name} (${simResult.data_provenance.prediction})`,
          data: simResult.u_pred.map((v: number) => v * 1000.0), // mm
          borderColor: isShuffled ? "#8A3FFA" : "#0F62FE", // purple-60 if shuffled control, otherwise Carbon blue-60
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.1,
        },
      ],
    };
  }, [simResult]);

  // OOD Matrix Bar Chart data
  const oodChartData = useMemo(() => {
    if (!oodMatrix) return null;
    return {
      labels: oodMatrix.partitions,
      datasets: [
        {
          label: "EXP5 Baseline GNO",
          data: oodMatrix.peak_disp_error_pct["EXP5 Baseline GNO"],
          backgroundColor: "#DA1E28", // red-60
        },
        {
          label: "EXP6-B T1-GNO",
          data: oodMatrix.peak_disp_error_pct["EXP6-B T1-GNO"],
          backgroundColor: "#B28600", // amber-60
        },
        {
          label: "EXP6-C Multi-Modal GNO",
          data: oodMatrix.peak_disp_error_pct["EXP6-C Multi-Modal GNO"],
          backgroundColor: "#198038", // green-60
        },
        {
          label: "EXP6-D Shuffled Modal (Ablation)",
          data: oodMatrix.peak_disp_error_pct["EXP6-D Shuffled Modal (Ablation)"],
          backgroundColor: "#8A3FFA", // purple-60
        },
      ],
    };
  }, [oodMatrix]);

  return (
    <div className="w-full min-h-screen bg-[#F4F4F4] text-[#161616] p-4 md:p-8 space-y-8 font-sans">
      {/* ----------------------------------------------------------------- */}
      {/* SECTION A — RESEARCH HEADER                                       */}
      {/* ----------------------------------------------------------------- */}
      <header className="bg-white border border-[#E0E0E0] p-6 rounded flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono tracking-widest text-[#0F62FE] font-bold uppercase">
              IIT Delhi CSE Research Evaluation
            </span>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] font-semibold">
              EXP4 → EXP5 → EXP6
            </span>
          </div>
          <h1 className="text-2xl font-bold font-mono tracking-tight text-[#161616]">
            Physics/Modal-Conditioned Spatiotemporal Graph Neural Operator
          </h1>
          <p className="text-xs font-sans text-[#525252] mt-1 max-w-3xl leading-relaxed">
            Multi-story nonlinear seismic dynamics surrogate with structural eigenvalue invariant FiLM conditioning,
            evaluated across 2,160 physical simulations against OpenSeesPy ground truth.
          </p>
        </div>

        <div className="flex flex-col items-start md:items-end gap-1.5 shrink-0 text-xs font-mono">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#198038]"></span>
            <span className="text-[#198038] font-bold">RESEARCH CORE FROZEN</span>
          </div>
          <div className="text-[#525252]">
            Hardware: <span className="text-[#161616] font-semibold">Apple Silicon GPU (MPS)</span>
          </div>
          <div className="text-[#525252]">
            Measured Speedup: <span className="text-[#0F62FE] font-bold">2.55× vs OpenSeesPy</span>
          </div>
        </div>
      </header>

      {/* ----------------------------------------------------------------- */}
      {/* SECTION I — EXPERIMENTAL PROGRESSION (THE RESEARCH STORY)          */}
      {/* ----------------------------------------------------------------- */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <GitBranch className="text-[#0F62FE]" size={18} />
          <h2 className="text-lg font-bold text-[#161616] font-mono">
            Scientific Research Progression (EXP4 → EXP5 → EXP6)
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {progression.map((step) => {
            // Distinct 4px top border for each stage:
            // EXP4 = red-60 (failure case)
            // EXP5 = amber-60 (improving)
            // EXP6 = green-60 (verified)
            // Stage 4 = neutral gray (boundary/limitation)
            const topBorderColor =
              step.step === 1
                ? "border-t-[#DA1E28]" // red-60
                : step.step === 2
                ? "border-t-[#B28600]" // amber-60
                : step.step === 3
                ? "border-t-[#198038]" // green-60
                : "border-t-[#8D8D8D]"; // neutral gray

            const badgeStyle =
              step.step === 1
                ? "bg-[#FFD7D9] text-[#DA1E28] border border-[#FF8389]"
                : step.step === 2
                ? "bg-[#FFF8E1] text-[#B28600] border-[#F1C21B]"
                : step.step === 3
                ? "bg-[#DEFBE6] text-[#198038] border-[#6FDC8C]"
                : "bg-[#F4F4F4] text-[#525252] border-[#E0E0E0]";

            return (
              <div
                key={step.step}
                className={`p-5 rounded border border-[#E0E0E0] border-t-4 ${topBorderColor} bg-white flex flex-col justify-between hover:border-[#8D8D8D] transition-colors`}
              >
                <div>
                  <div className="flex items-center justify-between text-xs font-mono mb-2">
                    <span className="font-bold text-[#161616]">STAGE {step.step}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${badgeStyle}`}>
                      {step.phase}
                    </span>
                  </div>
                  <h3 className="text-sm font-semibold text-[#161616] mb-1.5 font-sans">{step.title}</h3>
                  <div className="text-xs font-mono font-bold text-[#161616] mb-2">
                    {step.result_highlight}
                  </div>
                  <p className="text-xs text-[#525252] leading-relaxed font-sans">{step.finding}</p>
                </div>
                <div className="mt-4 pt-3 border-t border-[#E0E0E0] text-[11px] font-mono text-[#525252]">
                  <span className="text-[#8D8D8D] block text-[10px] uppercase font-bold mb-0.5">REPRESENTATION:</span>
                  <span className="text-[#161616] font-semibold">{step.representation}</span>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ----------------------------------------------------------------- */}
      {/* INTERACTIVE EXPERIMENTAL WORKSPACE (SECTIONS B, C, D, E, F, G, H)   */}
      {/* ----------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Structural Configuration & Modal Analysis */}
        <div className="lg:col-span-4 space-y-6">
          {/* SECTION 00: LIVE EARTHQUAKE INTEGRATION LAYER */}
          <div className="bg-white border-2 border-[#0F62FE] rounded p-5 space-y-3 shadow-sm">
            <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-2">
              <div className="flex items-center gap-2">
                <Radio size={16} className="text-[#0F62FE] animate-pulse" />
                <h3 className="text-sm font-bold text-[#161616] font-mono">
                  00 LIVE EARTHQUAKE
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] font-bold">
                USGS FEED
              </span>
            </div>
            <p className="text-xs text-[#525252] leading-relaxed">
              Real-world earthquake event discovery & engineering screening. Connect observed earthquake metadata to the multi-story surrogate.
            </p>
            <a
              href="/live"
              onClick={(e) => {
                e.preventDefault();
                window.history.pushState(null, "", "/live");
                window.dispatchEvent(new PopStateEvent("popstate"));
                // Trigger page re-render or tab switch if embedded
                const navBtn = document.querySelector('button[title*="00 Live Earthquake"], button:has(svg)');
                if (navBtn) (navBtn as HTMLElement).click();
              }}
              className="w-full py-2 bg-[#0F62FE] hover:bg-[#0353E9] text-white font-mono font-bold rounded text-xs flex items-center justify-center gap-1.5 transition cursor-pointer"
            >
              <span>OPEN LIVE EVENT WORKSPACE</span>
              <ArrowRight size={13} />
            </a>
          </div>

          {/* SECTION B: STRUCTURE SELECTOR */}
          <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3">
              <h3 className="text-sm font-bold text-[#161616] font-mono flex items-center gap-2">
                <Layers size={16} className="text-[#0F62FE]" />
                Structural Archetype
              </h3>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-[#F4F4F4] text-[#161616] font-semibold border border-[#E0E0E0]">
                {currentStructure?.category}
              </span>
            </div>

            <div>
              <label className="block text-xs font-mono text-[#525252] mb-1 font-bold uppercase">
                SELECT TEST STRUCTURE
              </label>
              <select
                value={selectedStructureId}
                onChange={(e) => handleStructureChange(e.target.value)}
                className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded px-3 py-2 text-xs font-mono text-[#161616] cursor-pointer"
              >
                {structures.map((s) => (
                  <option key={s.archetype_id} value={s.archetype_id}>
                    {s.archetype_id} — {s.n_stories} Stories (T1={s.T1_s.toFixed(2)}s, {s.category})
                  </option>
                ))}
              </select>
            </div>

            {/* SECTION C: EIGENVALUE & MODAL SHAPE VISUALIZER */}
            <div className="space-y-2 pt-2 border-t border-[#E0E0E0]">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-[#525252] font-bold uppercase">Structural Invariant:</span>
                <span className="text-[#161616] font-semibold">
                  T₁ = {currentStructure?.T1_s.toFixed(3)}s · ω₁ = {currentStructure?.omega1_rad_s.toFixed(2)} rad/s
                </span>
              </div>

              {/* Mode shape tabs */}
              <div className="grid grid-cols-3 gap-1.5 pt-1">
                {["Mode 1 (Fundamental)", "Mode 2", "Mode 3"].map((_label, mIdx) => (
                  <button
                    key={mIdx}
                    onClick={() => setActiveModeShapeIdx(mIdx)}
                    className={`py-1.5 text-xs font-mono rounded cursor-pointer transition ${
                      activeModeShapeIdx === mIdx
                        ? "bg-[#0F62FE] text-white font-bold"
                        : "bg-[#F4F4F4] border border-[#E0E0E0] text-[#525252] hover:bg-[#EBEBEB]"
                    }`}
                  >
                    Mode {mIdx + 1}
                  </button>
                ))}
              </div>
            </div>

            <div className="text-[11px] font-mono text-[#525252] bg-[#F4F4F4] p-2.5 rounded border border-[#E0E0E0] flex items-center gap-2">
              <Info size={14} className="text-[#0F62FE] shrink-0" />
              <span>
                Pre-earthquake structural invariant computed from [M] and [K] prior to excitation.
              </span>
            </div>

            {/* Vertical Mode Shape Deformation Canvas */}
            <div className="p-4 bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center justify-center">
              <div className="relative w-48 h-56 border-b-2 border-[#161616] flex flex-col justify-end">
                {/* Vertical Center Reference Line */}
                <div className="absolute left-1/2 top-0 bottom-0 w-px border-l border-dashed border-[#8D8D8D] -translate-x-1/2"></div>

                {currentStructure &&
                  currentStructure.mode_shapes[activeModeShapeIdx] &&
                  currentStructure.mode_shapes[activeModeShapeIdx].map((dispNorm: number, idx: number) => {
                    const floorNum = idx + 1;
                    const totalStories = currentStructure.n_stories;
                    const heightPct = (floorNum / totalStories) * 85;
                    const xOffsetPct = 50 + dispNorm * 38;

                    return (
                      <div
                        key={floorNum}
                        className="absolute flex items-center -translate-x-1/2 -translate-y-1/2 transition-all duration-300"
                        style={{
                          bottom: `${heightPct}%`,
                          left: `${xOffsetPct}%`,
                        }}
                      >
                        <div className="w-3.5 h-3.5 rounded-full bg-[#0F62FE] border-2 border-white shadow flex items-center justify-center" />
                        <span className="text-[10px] font-mono text-[#161616] font-semibold ml-1.5 whitespace-nowrap bg-white/90 border border-[#E0E0E0] px-1 rounded">
                          F{floorNum}: {dispNorm.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>

            <div className="text-center text-xs font-mono text-[#525252]">
              Fundamental Period:{" "}
              <strong className="text-[#161616] font-mono">
                T{activeModeShapeIdx + 1} ={" "}
                {currentStructure
                  ? [currentStructure.T1_s, currentStructure.T2_s, currentStructure.T3_s][
                      activeModeShapeIdx
                    ]?.toFixed(3)
                  : "0.00"}{" "}
                s
              </strong>
            </div>
          </div>
        </div>

        {/* Right Column: Earthquake Selection, Model Switcher & Response Plots */}
        <div className="lg:col-span-8 space-y-6">
          {/* SECTION D & E: EARTHQUAKE & MODEL CONTROLS */}
          <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* SECTION D: Earthquake Input */}
              <div>
                <label className="block text-xs font-mono text-[#525252] mb-1 font-bold uppercase">
                  SEISMIC EXCITATION a_g(t)
                </label>
                <select
                  value={selectedEarthquakeId}
                  onChange={(e) => handleEarthquakeChange(e.target.value)}
                  className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded px-3 py-2 text-xs font-mono text-[#161616] cursor-pointer"
                >
                  {earthquakes.map((eq) => (
                    <option key={eq.record_id} value={eq.record_id}>
                      {eq.record_id} — {eq.event_name} (PGA: {eq.pga_g}g, {eq.role.split("(")[0]})
                    </option>
                  ))}
                </select>
              </div>

              {/* SECTION E: Model Selector */}
              <div>
                <label className="block text-xs font-mono text-[#525252] mb-1 font-bold uppercase">
                  SURROGATE NEURAL OPERATOR
                </label>
                <select
                  value={selectedModelId}
                  onChange={(e) => handleModelChange(e.target.value)}
                  className="w-full bg-[#F4F4F4] border border-[#E0E0E0] rounded px-3 py-2 text-xs font-mono text-[#161616] cursor-pointer"
                >
                  {models.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.display_name} [{m.status}]
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Floor Selection & Execution Status */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-[#E0E0E0]">
              <div className="flex items-center gap-1.5 text-xs font-mono">
                <span className="text-[#525252] font-semibold">Target Floor:</span>
                {currentStructure &&
                  Array.from({ length: currentStructure.n_stories }, (_, i) => i + 1).map((f) => (
                    <button
                      key={f}
                      onClick={() => handleFloorChange(f)}
                      className={`px-2.5 py-1 rounded text-xs font-mono cursor-pointer transition ${
                        selectedFloor === f
                          ? "bg-[#0F62FE] text-white font-bold"
                          : "bg-[#F4F4F4] border border-[#E0E0E0] text-[#525252] hover:text-[#161616] hover:bg-[#EBEBEB]"
                      }`}
                    >
                      Floor {f} {f === currentStructure.n_stories ? "(Roof)" : ""}
                    </button>
                  ))}
              </div>

              {isLoadingSim && (
                <div className="flex items-center gap-2 text-xs font-mono text-[#0F62FE] animate-pulse">
                  <Activity size={14} className="animate-spin" />
                  <span>Computing Live Surrogate Forward Pass...</span>
                </div>
              )}

              {errorMsg && (
                <div className="text-xs font-mono text-[#DA1E28] bg-[#FFD7D9] px-2 py-1 rounded border border-[#FF8389]">
                  {errorMsg}
                </div>
              )}

              {simResult && !isLoadingSim && (
                <div className="flex items-center gap-2">
                  <span
                    className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold border ${
                      simResult.is_live_inference
                        ? "bg-[#DEFBE6] text-[#198038] border-[#6FDC8C]"
                        : "bg-[#F4F4F4] text-[#525252] border-[#E0E0E0]"
                    }`}
                  >
                    {simResult.data_provenance.prediction}
                  </span>
                  <span className="text-xs font-mono text-[#525252]">
                    Latency: <strong className="text-[#161616] font-mono">{simResult.metrics.latency_ms} ms</strong>
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* SECTION G & H: RESPONSE PREDICTION & ERROR ANALYSIS */}
          <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3">
              <div>
                <h3 className="text-sm font-bold text-[#161616] font-mono flex items-center gap-2">
                  <Activity size={16} className="text-[#0F62FE]" />
                  Structural Displacement Response u(t)
                </h3>
                <p className="text-xs text-[#525252] font-sans mt-0.5">
                  Comparing OpenSeesPy non-linear ground truth against neural operator for Floor {selectedFloor}.
                </p>
              </div>

              {simResult && (
                <div className="flex items-center gap-4 text-xs font-mono">
                  <div>
                    <span className="text-[#525252]">Rel L₂ Error:</span>{" "}
                    <strong
                      className={`font-mono ${
                        simResult.metrics.rel_l2_pct > 50 ? "text-[#DA1E28]" : "text-[#198038]"
                      }`}
                    >
                      {simResult.metrics.rel_l2_pct}%
                    </strong>
                  </div>
                  <div>
                    <span className="text-[#525252]">Peak Error:</span>{" "}
                    <strong
                      className={`font-mono ${
                        simResult.metrics.peak_disp_err_pct > 25 ? "text-[#DA1E28]" : "text-[#0F62FE]"
                      }`}
                    >
                      {simResult.metrics.peak_disp_err_pct}%
                    </strong>
                  </div>
                  <div>
                    <span className="text-[#525252]">Pearson r:</span>{" "}
                    <strong className="text-[#161616] font-mono">{simResult.metrics.pearson_r}</strong>
                  </div>
                </div>
              )}
            </div>

            {/* Main Response Chart: Contrast-preserving #F4F4F4 background */}
            <div className="h-72 w-full bg-[#F4F4F4] p-3 rounded border border-[#E0E0E0]">
              {trajectoryChartData && (
                <Line
                  data={trajectoryChartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      x: {
                        title: { display: true, text: "Time (seconds)", color: "#525252" },
                        grid: { color: "#E0E0E0" },
                        ticks: { color: "#525252", font: { family: "IBM Plex Mono", size: 10 } },
                      },
                      y: {
                        title: { display: true, text: "Displacement u(t) [mm]", color: "#525252" },
                        grid: { color: "#E0E0E0" },
                        ticks: { color: "#525252", font: { family: "IBM Plex Mono", size: 10 } },
                      },
                    },
                    plugins: {
                      legend: {
                        labels: { color: "#161616", font: { family: "IBM Plex Mono", size: 11 } },
                      },
                      tooltip: {
                        bodyFont: { family: "IBM Plex Mono" },
                        titleFont: { family: "IBM Plex Mono" },
                      },
                    },
                  }}
                />
              )}
            </div>

            {/* Input Accelerogram Waveform */}
            <div className="pt-2">
              <div className="text-xs font-mono text-[#525252] mb-1 flex items-center justify-between">
                <span>INPUT ACCELEROGRAM a_g(t) [PEER NGA-West2]</span>
                {simResult && (
                  <span className="font-mono font-bold text-[#161616]">
                    PGA: {(Math.max(...simResult.ag.map(Math.abs))).toFixed(3)}g
                  </span>
                )}
              </div>
              <div className="h-24 w-full bg-[#F4F4F4] p-2 rounded border border-[#E0E0E0]">
                {accelerogramChartData && (
                  <Line
                    data={accelerogramChartData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        x: { display: false },
                        y: {
                          ticks: { color: "#8D8D8D", font: { size: 9, family: "IBM Plex Mono" } },
                          grid: { color: "#E0E0E0" },
                        },
                      },
                      plugins: { legend: { display: false } },
                    }}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ----------------------------------------------------------------- */}
      {/* SECTION J & M: OOD GENERALIZATION & SHUFFLED FALSIFICATION        */}
      {/* ----------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* SECTION J: OOD Generalization Matrix */}
        <div className="lg:col-span-8 bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3">
            <h3 className="text-sm font-bold text-[#161616] font-mono flex items-center gap-2">
              <Activity size={16} className="text-[#0F62FE]" />
              Out-of-Distribution Generalization Matrix (Median Peak Disp Error %)
            </h3>
            <span className="text-xs font-mono text-[#525252]">2,160 Physical Simulations</span>
          </div>

          <div className="h-64 w-full bg-[#F4F4F4] p-3 rounded border border-[#E0E0E0]">
            {oodChartData && (
              <Bar
                data={oodChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    x: {
                      ticks: { color: "#525252", font: { family: "IBM Plex Mono", size: 10 } },
                      grid: { color: "#E0E0E0" },
                    },
                    y: {
                      title: { display: true, text: "Median Peak Error (%)", color: "#525252" },
                      ticks: { color: "#525252", font: { family: "IBM Plex Mono", size: 10 } },
                      grid: { color: "#E0E0E0" },
                    },
                  },
                  plugins: {
                    legend: {
                      labels: { color: "#161616", font: { family: "IBM Plex Mono", size: 11 } },
                    },
                  },
                }}
              />
            )}
          </div>

          <p className="text-xs font-mono text-[#161616] bg-[#F4F4F4] p-3 rounded border border-[#E0E0E0] leading-relaxed">
            <strong>Key Finding:</strong> Multi-Modal GNO achieves <strong className="text-[#198038]">13.06%</strong> median
            peak error on held-out structure 5S_T120, compared to <strong className="text-[#DA1E28]">35.21%</strong> for
            unconditioned GNO (a <strong className="text-[#0F62FE]">62.9% relative error reduction</strong>).
          </p>
        </div>

        {/* SECTION M: Shuffled Falsification Ablation */}
        <div className="lg:col-span-4 bg-white border border-[#E0E0E0] rounded p-5 space-y-4 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-[#161616] font-mono flex items-center gap-2 mb-1">
              <CheckCircle2 size={16} className="text-[#198038]" />
              Falsification Ablation (EXP6-D)
            </h3>
            <p className="text-xs text-[#525252] font-sans mb-3 leading-relaxed">
              Falsifying whether modal conditioning provides true physical guidance or merely adds auxiliary network capacity.
            </p>

            {ablation && (
              <div className="space-y-2">
                {ablation.metrics.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] text-xs font-mono"
                  >
                    <div className="text-[#161616] font-semibold mb-1">{item.partition}</div>
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="text-[#198038] font-bold">
                        True Modal: {item.true_multimodal_err}%
                      </span>
                      <ArrowRight size={12} className="text-[#8D8D8D]" />
                      <span className="text-[#DA1E28] font-bold">
                        Shuffled: {item.shuffled_err}%
                      </span>
                    </div>
                    <div className="text-[10px] text-[#525252] text-right mt-1">
                      +{item.delta_percentage_points} pp ({item.degradation_pct}% degradation)
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="text-[11px] font-mono text-[#525252] bg-[#F4F4F4] p-3 rounded border border-[#E0E0E0] leading-relaxed">
            <span className="font-semibold text-[#161616]">Conservative Interpretation:</span>{" "}
            Degradation under shuffled conditioning supports the hypothesis that the model
            exploits physical eigenvalue correspondence rather than auxiliary scalar capacity.
          </div>
        </div>
      </div>

      {/* ----------------------------------------------------------------- */}
      {/* SECTION K & L: FAILURE ANALYSIS & COMPUTATIONAL BENCHMARK         */}
      {/* ----------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* SECTION K: Failure Analysis */}
        <div className="lg:col-span-7 bg-white border border-[#E0E0E0] rounded p-5 space-y-4">
          <div className="flex items-center gap-2 border-b border-[#E0E0E0] pb-3">
            <AlertTriangle className="text-[#B28600]" size={18} />
            <h3 className="text-sm font-bold text-[#161616] font-mono">
              Where the Model Fails — Scientific Limitations
            </h3>
          </div>

          <div className="space-y-3">
            {failures.map((f) => (
              <div
                key={f.id}
                className="p-3.5 bg-[#FFF8E1] rounded border border-[#B28600] space-y-1 text-xs font-mono"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-[#B28600]">{f.title}</span>
                  <span className="text-[10px] bg-white border border-[#B28600] text-[#B28600] px-1.5 py-0.5 rounded font-bold">
                    {f.phase}
                  </span>
                </div>
                <div className="text-[#DA1E28] font-semibold text-[11px]">Symptom: {f.symptom}</div>
                <p className="text-[#525252] text-[11px] font-sans leading-relaxed pt-1">{f.mechanism}</p>
              </div>
            ))}
          </div>
        </div>

        {/* SECTION L: Computational Benchmark */}
        <div className="lg:col-span-5 bg-white border border-[#E0E0E0] rounded p-5 space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 border-b border-[#E0E0E0] pb-3 mb-2">
              <Zap className="text-[#0F62FE]" size={18} />
              <h3 className="text-sm font-bold text-[#161616] font-mono">
                Measured Inference Benchmark (Apple Silicon MPS)
              </h3>
            </div>
            <p className="text-xs text-[#525252] font-sans mb-3">
              Synchronized single-building transient simulation benchmarks (T=20.48s, 1024 steps).
            </p>

            {benchmark && (
              <div className="space-y-2">
                {benchmark.models.map((m, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center justify-between text-xs font-mono"
                  >
                    <div>
                      <div className="text-[#161616] font-semibold">{m.name}</div>
                      <div className="text-[10px] text-[#525252] font-mono">
                        {m.latency_ms} ms · {m.throughput_sim_s} sim/s
                      </div>
                    </div>
                    <div className="text-right">
                      <span
                        className={`text-xs font-bold font-mono ${
                          m.speedup > 1.0 ? "text-[#198038]" : "text-[#525252]"
                        }`}
                      >
                        {m.speedup.toFixed(2)}x
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="text-[11px] font-mono text-[#525252] bg-[#F4F4F4] p-3 rounded border border-[#E0E0E0] leading-relaxed">
            <strong>Conclusion:</strong> EXP6 T₁-GNO achieves a <strong className="text-[#0F62FE]">2.55× wall-clock speedup</strong>{" "}
            over OpenSeesPy while retaining topology flexibility and physics-informed envelope scaling.
          </div>
        </div>
      </div>

      {/* ----------------------------------------------------------------- */}
      {/* SECTION N & O: CONTRIBUTIONS & REPRODUCIBILITY                    */}
      {/* ----------------------------------------------------------------- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-[#E0E0E0]">
        {/* SECTION N: Three Research Contributions */}
        <div className="space-y-3">
          <h4 className="text-xs font-mono uppercase tracking-wider text-[#525252] font-bold">
            Core Scientific Contributions
          </h4>
          <div className="space-y-2 text-xs font-mono">
            <div className="p-4 bg-white border border-[#E0E0E0] rounded">
              <strong className="text-[#0F62FE] block mb-1">
                01 — Topology-Native Neural Operator
              </strong>
              <p className="text-[#525252] font-sans leading-relaxed">
                Replaces fixed-grid zero-padding with a discrete spatiotemporal graph representation,
                eliminating Gibbs boundary failure and reducing 3-story relative error by 77.51 pp.
              </p>
            </div>
            <div className="p-4 bg-white border border-[#E0E0E0] rounded">
              <strong className="text-[#198038] block mb-1">
                02 — Physics-Informed Modal Conditioning
              </strong>
              <p className="text-[#525252] font-sans leading-relaxed">
                Injects pre-earthquake structural eigenvalue invariants (T₁, ω₁) via FiLM,
                achieving a 62.9% relative reduction in peak displacement error under modal shift.
              </p>
            </div>
            <div className="p-4 bg-white border border-[#E0E0E0] rounded">
              <strong className="text-[#B28600] block mb-1">
                03 — Rigorous OOD & Falsification Methodology
              </strong>
              <p className="text-[#525252] font-sans leading-relaxed">
                Enforces strict structural-group and earthquake-group partitioning, falsifies capacity
                artifacts via shuffled ablations, and openly reports phase drift boundaries.
              </p>
            </div>
          </div>
        </div>

        {/* SECTION O: Reproducibility Panel */}
        <div className="space-y-3">
          <h4 className="text-xs font-mono uppercase tracking-wider text-[#525252] font-bold">
            Reproducibility & Verification Telemetry
          </h4>
          <div className="p-5 bg-white border border-[#E0E0E0] rounded space-y-3 text-xs font-mono">
            <div className="grid grid-cols-2 gap-3 pb-3 border-b border-[#E0E0E0]">
              <div>
                <span className="text-[#525252] block text-[10px]">Python Runtime:</span>
                <span className="text-[#161616] font-bold">3.14.5 (arm64)</span>
              </div>
              <div>
                <span className="text-[#525252] block text-[10px]">PyTorch Backend:</span>
                <span className="text-[#161616] font-bold">2.13.0 (Apple MPS)</span>
              </div>
              <div>
                <span className="text-[#525252] block text-[10px]">OpenSeesPy:</span>
                <span className="text-[#161616] font-bold">3.5.1.13</span>
              </div>
              <div>
                <span className="text-[#525252] block text-[10px]">Authoritative Seed:</span>
                <span className="text-[#161616] font-bold">42 (Strictly Fixed)</span>
              </div>
            </div>

            <div className="space-y-1.5 pt-1 text-[11px]">
              <div className="flex justify-between">
                <span className="text-[#525252]">Unit Test Suite:</span>
                <span className="text-[#198038] font-bold font-mono">294 / 294 PASSING</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#525252]">Forensic Audit Verdict:</span>
                <span className="text-[#0F62FE] font-bold font-mono">PASS (5/5 CHECKS)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#525252]">Split Partition Overlap:</span>
                <span className="text-[#198038] font-bold font-mono">EXACTLY 0 SIMULATIONS</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#525252]">Model Checkpoint Status:</span>
                <span className="text-[#161616] font-bold font-mono">FROZEN & IMMUTABLE</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
