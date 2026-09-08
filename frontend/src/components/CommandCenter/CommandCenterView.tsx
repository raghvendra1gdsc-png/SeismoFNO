import React, { useState, useEffect } from "react";
import {
  Activity,
  Shield,
  Zap,
  Gauge,
  Building2,
  Radio,
  CheckCircle2,
  Clock,
  Layers,
  GitBranch,
} from "lucide-react";
import type { ScenarioPredictionResponse, SystemInfo } from "../../api/digitalTwinApi";
import { fetchDemoProgression, type ProgressionStep } from "../../api/researchDemoApi";

interface CommandCenterViewProps {
  prediction: ScenarioPredictionResponse | null;
  systemInfo: SystemInfo | null;
  selectedStructureName: string;
  selectedEarthquakeName: string;
  onNavigateTab: (tab: string) => void;
  isLoading?: boolean;
}

const DEFAULT_PROGRESSION: ProgressionStep[] = [
  {
    step: 1,
    phase: "EXP4",
    title: "Fixed-Grid Fourier Neural Operator",
    representation: "Fixed 5x2048 2D Tensor (Zero-Padded Stories 4-5)",
    result_highlight: "3-Story Rel L2 = 99.60%",
    status: "TOPOLOGY BOUNDARY FAILURE",
    finding: "Standard FNO assumes a regular Euclidean lattice. Zero-padding smaller structures forces non-physical spatial step discontinuities, causing high-frequency Gibbs ringing that destroys physical response prediction.",
    color: "rose",
  },
  {
    step: 2,
    phase: "EXP5",
    title: "Spatiotemporal Graph Neural Operator",
    representation: "Topology-Native Graph G=(V, E) (0 Padding Nodes)",
    result_highlight: "3-Story Rel L2 = 22.09% (77.51 pp drop)",
    status: "TOPOLOGY LIMITATION RESOLVED",
    finding: "Treating building floors as graph nodes and columns as edges eliminates artificial boundary padding. 3-story relative error drops from 99.60% to 22.09% (77.82% relative reduction).",
    color: "emerald",
  },
  {
    step: 3,
    phase: "EXP6",
    title: "Physics/Modal-Conditioned GNO",
    representation: "Native Graph + Dual-Branch FiLM on [T1-3, omega1-3]",
    result_highlight: "OOD-B Peak Error: 35.21% -> 13.06% (62.9% reduction)",
    status: "MODAL ENVELOPE GENERALIZATION",
    finding: "Conditioning spatial message-passing and temporal spectral kernels on pre-earthquake modal eigenvalue invariants rescales the response envelope, yielding a 62.9% relative reduction in peak displacement error on unseen flexible structure 5S_T120.",
    color: "indigo",
  },
  {
    step: 4,
    phase: "ANALYSIS",
    title: "Documented Scientific Boundary",
    representation: "Global 1D Fourier Kernel Over Long Horizons (20.48s)",
    result_highlight: "OOD-B Waveform Rel L2 > 100%, Pearson r ~ 0.05-0.09",
    status: "PHASE EXTRAPOLATION LIMITATION",
    finding: "While peak displacement envelopes are accurately bounded, trajectory Relative L2 remains elevated due to cumulative phase drift in static 1D Fourier bases when vibration periods extrapolate far outside training support.",
    color: "amber",
  },
];

export const CommandCenterView: React.FC<CommandCenterViewProps> = ({
  prediction,
  systemInfo,
  selectedStructureName,
  selectedEarthquakeName,
  onNavigateTab,
}) => {
  const [progression, setProgression] = useState<ProgressionStep[]>(DEFAULT_PROGRESSION);

  useEffect(() => {
    fetchDemoProgression()
      .then((data) => {
        if (data && data.length > 0) setProgression(data);
      })
      .catch(() => {});
  }, []);

  const metrics = prediction?.metrics;
  const ductility = metrics?.ductility_demand_mu ?? 1.0;
  const isYielded = ductility > 1.0;

  return (
    <div className="space-y-6 p-6 font-sans">
      {/* Executive Header Banner */}
      <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-4 bg-white p-5 rounded border">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#161616]">
              SEISMIC DIGITAL-TWIN COMMAND CENTER
            </h1>
            <span className="flex items-center space-x-1.5 px-2.5 py-0.5 rounded text-[11px] font-mono font-medium bg-[#DEFBE6] text-[#198038] border border-[#6FDC8C]">
              <span className="h-1.5 w-1.5 rounded-full bg-[#198038]" />
              <span>SURROGATE OPERATIONAL</span>
            </span>
          </div>
          <p className="text-xs text-[#525252] mt-1 font-sans">
            High-throughput Fourier Neural Operator surrogate for nonlinear seismic structural response.
          </p>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="bg-[#F4F4F4] px-3 py-1.5 rounded border border-[#E0E0E0] flex items-center space-x-2">
            <Radio size={14} className="text-[#0F62FE]" />
            <span className="text-[#525252]">Catalog:</span>
            <span className="text-[#161616] font-semibold">PEER NGA-West2 / India IS 1893</span>
          </div>
          <div className="bg-[#F4F4F4] px-3 py-1.5 rounded border border-[#E0E0E0] flex items-center space-x-2">
            <Zap size={14} className="text-[#0F62FE]" />
            <span className="text-[#525252]">Inference Latency:</span>
            <span className="text-[#0F62FE] font-bold font-mono">
              {prediction ? `${prediction.inference_time_ms.toFixed(2)} ms` : "< 2.0 ms"}
            </span>
          </div>
        </div>
      </div>

      {/* Primary KPI Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Metric 1: Peak Displacement */}
        <div className="bg-white border border-[#E0E0E0] hover:border-[#8D8D8D] rounded p-4 relative transition-colors">
          <div className="flex items-center justify-between text-[#525252] text-xs font-sans mb-1 font-medium">
            <span>PEAK DISPLACEMENT</span>
            <Gauge size={14} className="text-[#0F62FE]" />
          </div>
          <div className="text-2xl font-mono font-bold text-[#161616]">
            {metrics?.peak_displacement_mm !== undefined
              ? `${metrics.peak_displacement_mm.toFixed(1)} mm`
              : "—"}
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-[#525252]">
            <span>SI Metric:</span>
            <span className="text-[#161616]">
              {metrics?.peak_displacement_m !== undefined
                ? `${metrics.peak_displacement_m.toFixed(4)} m`
                : "—"}
            </span>
          </div>
        </div>

        {/* Metric 2: Drift Demand & Damage State */}
        <div className="bg-white border border-[#E0E0E0] hover:border-[#8D8D8D] rounded p-4 relative transition-colors">
          <div className="flex items-center justify-between text-[#525252] text-xs font-sans mb-1 font-medium">
            <span>INTER-STORY DRIFT</span>
            <Activity size={14} className="text-[#B28600]" />
          </div>
          <div className="text-2xl font-mono font-bold text-[#161616]">
            {metrics?.drift_ratio_percent !== undefined
              ? `${metrics.drift_ratio_percent.toFixed(2)} %`
              : "—"}
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px] font-mono">
            <span className="text-[#525252]">Performance State:</span>
            <span
              className={`font-semibold ${
                (metrics?.drift_ratio_percent ?? 0) < 1.0
                  ? "text-[#198038]"
                  : (metrics?.drift_ratio_percent ?? 0) < 2.0
                  ? "text-[#B28600]"
                  : "text-[#DA1E28]"
              }`}
            >
              {(metrics?.drift_ratio_percent ?? 0) < 1.0
                ? "Immediate Occupancy"
                : (metrics?.drift_ratio_percent ?? 0) < 2.0
                ? "Life Safety"
                : "Collapse Prevention"}
            </span>
          </div>
        </div>

        {/* Metric 3: Ductility & Plasticity */}
        <div className="bg-white border border-[#E0E0E0] hover:border-[#8D8D8D] rounded p-4 relative transition-colors">
          <div className="flex items-center justify-between text-[#525252] text-xs font-sans mb-1 font-medium">
            <span>DUCTILITY DEMAND (μ)</span>
            <Layers size={14} className="text-[#0F62FE]" />
          </div>
          <div className="text-2xl font-mono font-bold text-[#161616]">
            {metrics?.ductility_demand_mu !== undefined
              ? `${metrics.ductility_demand_mu.toFixed(2)}`
              : "—"}
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px] font-mono">
            <span className="text-[#525252]">Yield Status:</span>
            <span className={`font-semibold ${isYielded ? "text-[#B28600]" : "text-[#198038]"}`}>
              {isYielded ? `Yielded at t = ${metrics?.yield_time_sec?.toFixed(2)}s` : "Linear Elastic"}
            </span>
          </div>
        </div>

        {/* Metric 4: Dissipated Hysteretic Energy */}
        <div className="bg-white border border-[#E0E0E0] hover:border-[#8D8D8D] rounded p-4 relative transition-colors">
          <div className="flex items-center justify-between text-[#525252] text-xs font-sans mb-1 font-medium">
            <span>HYSTERETIC DISSIPATION</span>
            <Zap size={14} className="text-[#198038]" />
          </div>
          <div className="text-2xl font-mono font-bold text-[#161616]">
            {metrics?.total_hysteretic_energy_J !== undefined
              ? `${metrics.total_hysteretic_energy_J.toFixed(1)} J`
              : "—"}
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-[#525252]">
            <span>Base Shear Ratio:</span>
            <span className="text-[#161616]">
              {metrics?.base_shear_ratio !== undefined ? `${metrics.base_shear_ratio.toFixed(2)} W` : "—"}
            </span>
          </div>
        </div>
      </div>

      {/* Stage 1-4 Scientific Research Progression Cards (Carbon Treatment) */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <GitBranch size={16} className="text-[#0F62FE]" />
            <h2 className="text-sm font-bold font-mono text-[#161616] uppercase">
              Scientific Research Progression (EXP4 → EXP5 → EXP6)
            </h2>
          </div>
          <button
            onClick={() => onNavigateTab("research_demo")}
            className="text-xs font-mono text-[#0F62FE] hover:underline cursor-pointer font-semibold"
          >
            Open Full Research Defense →
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {progression.map((step) => {
            // EXP4 = red-60, EXP5 = amber-60, EXP6 = green-60, Stage 4 = neutral gray
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
                className={`p-4 rounded border border-[#E0E0E0] border-t-4 ${topBorderColor} bg-white flex flex-col justify-between hover:border-[#8D8D8D] transition-colors`}
              >
                <div>
                  <div className="flex items-center justify-between text-xs font-mono mb-2">
                    <span className="font-bold text-[#161616]">STAGE {step.step}</span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${badgeStyle}`}>
                      {step.phase}
                    </span>
                  </div>
                  <h3 className="text-xs font-bold text-[#161616] mb-1.5 font-sans leading-snug">{step.title}</h3>
                  <div className="text-xs font-mono font-bold text-[#161616] mb-2">
                    {step.result_highlight}
                  </div>
                  <p className="text-xs text-[#525252] leading-relaxed font-sans">{step.finding}</p>
                </div>
                <div className="mt-3 pt-2.5 border-t border-[#E0E0E0] text-[10px] font-mono text-[#525252]">
                  <span className="text-[#8D8D8D] block uppercase font-bold mb-0.5">REPRESENTATION:</span>
                  <span className="text-[#161616] font-semibold">{step.representation}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Middle Section: Active Scenario + Model Health Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Active Structure & Excitation Card */}
        <div className="md:col-span-2 bg-white border border-[#E0E0E0] rounded p-5">
          <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3 mb-4">
            <div className="flex items-center space-x-2">
              <Building2 size={16} className="text-[#0F62FE]" />
              <h3 className="text-sm font-bold font-mono text-[#161616] uppercase">
                Active Scenario Configuration
              </h3>
            </div>
            <button
              onClick={() => onNavigateTab("scenario_lab")}
              className="text-xs font-mono text-[#0F62FE] hover:underline cursor-pointer font-semibold"
            >
              Open Scenario Lab →
            </button>
          </div>

          <div className="grid grid-cols-2 gap-4 text-xs font-mono">
            <div className="space-y-2 bg-[#F4F4F4] p-3 rounded border border-[#E0E0E0]">
              <div className="text-[#525252] font-semibold uppercase text-[10px]">Excitation Ground Motion</div>
              <div className="text-[#161616] font-bold text-sm truncate">{selectedEarthquakeName}</div>
              <div className="flex justify-between text-[#525252]">
                <span>Target PGA:</span>
                <span className="text-[#161616] font-semibold">
                  {prediction?.scenario?.pga_g ?? 0.40} g
                </span>
              </div>
              <div className="flex justify-between text-[#525252]">
                <span>Dataset Origin:</span>
                <span className="text-[#198038] font-semibold">Historical Ground Motion</span>
              </div>
            </div>

            <div className="space-y-2 bg-[#F4F4F4] p-3 rounded border border-[#E0E0E0]">
              <div className="text-[#525252] font-semibold uppercase text-[10px]">Structural Twin</div>
              <div className="text-[#161616] font-bold text-sm truncate">{selectedStructureName}</div>
              <div className="flex justify-between text-[#525252]">
                <span>Period / Damping:</span>
                <span className="text-[#161616] font-semibold">
                  T0 = {prediction?.scenario?.T0 ?? 0.50}s | ζ = {((prediction?.scenario?.damping_ratio ?? 0.05) * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex justify-between text-[#525252]">
                <span>Yield Disp / Alpha:</span>
                <span className="text-[#161616] font-semibold">
                  uy = {((prediction?.scenario?.yield_displacement_m ?? 0.01) * 1000).toFixed(0)} mm | α = {prediction?.scenario?.post_yield_ratio ?? 0.05}
                </span>
              </div>
            </div>
          </div>

          {/* Quick Trajectory Mini-Preview if available */}
          {prediction && prediction.trajectories.u.length > 0 && (
            <div className="mt-4 pt-3 border-t border-[#E0E0E0]">
              <div className="flex justify-between items-center text-[11px] font-mono text-[#525252] mb-2">
                <span>Relative Displacement Time History u(t) Preview</span>
                <button
                  onClick={() => onNavigateTab("structural_twin")}
                  className="text-[#0F62FE] hover:underline cursor-pointer font-semibold"
                >
                  Full Dynamic Twin →
                </button>
              </div>
              <div className="h-16 w-full bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center px-2">
                <svg className="w-full h-12 stroke-[#0F62FE] fill-none" viewBox="0 0 500 50" preserveAspectRatio="none">
                  <path
                    d={prediction.trajectories.u
                      .map((val, idx, arr) => {
                        const x = (idx / (arr.length - 1)) * 500;
                        const maxVal = Math.max(...arr.map(Math.abs)) || 1e-4;
                        const y = 25 - (val / maxVal) * 22;
                        return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
                      })
                      .join(" ")}
                    strokeWidth="1.5"
                  />
                </svg>
              </div>
            </div>
          )}
        </div>

        {/* Model Provenance & Integrity Status Card */}
        <div className="bg-white border border-[#E0E0E0] rounded p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 border-b border-[#E0E0E0] pb-3 mb-4">
              <Shield size={16} className="text-[#198038]" />
              <h3 className="text-sm font-bold font-mono text-[#161616] uppercase">
                Model Provenance
              </h3>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-[#525252]">Architecture:</span>
                <span className="text-[#161616] font-semibold">1D Fourier Neural Operator</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#525252]">Fourier Modes:</span>
                <span className="text-[#161616]">{systemInfo?.modes ?? 128} modes</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#525252]">Hidden Width:</span>
                <span className="text-[#161616]">{systemInfo?.width ?? 48} channels</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#525252]">Parameters:</span>
                <span className="text-[#161616]">
                  {systemInfo?.model_parameters ? systemInfo.model_parameters.toLocaleString() : "1,196,931"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#525252]">Compute Device:</span>
                <span className="text-[#0F62FE] font-semibold">{systemInfo?.device ?? "cpu"}</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-[#E0E0E0] space-y-2 text-[11px] font-mono">
            <div className="flex items-center space-x-1.5 text-[#198038] font-semibold">
              <CheckCircle2 size={13} />
              <span>FROZEN SCIENTIFIC CHECKPOINT</span>
            </div>
            <p className="text-[#525252] text-[10px] leading-tight font-sans">
              Model weights verified against OpenSeesPy non-linear time history benchmarks.
            </p>
          </div>
        </div>
      </div>

      {/* Quick Access Action Bar */}
      <div className="bg-white border border-[#E0E0E0] rounded p-4 flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs font-mono text-[#525252] flex items-center space-x-2">
          <Clock size={14} className="text-[#0F62FE]" />
          <span>Quick Launch Specialized Engineering Workspaces:</span>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => onNavigateTab("earthquake_intel")}
            className="px-3 py-1.5 text-xs font-mono bg-[#F4F4F4] hover:bg-[#E0E0E0] border border-[#E0E0E0] rounded text-[#161616] cursor-pointer transition font-medium"
          >
            India Hazard Map
          </button>
          <button
            onClick={() => onNavigateTab("structural_twin")}
            className="px-3 py-1.5 text-xs font-mono bg-[#F4F4F4] hover:bg-[#E0E0E0] border border-[#E0E0E0] rounded text-[#161616] cursor-pointer transition font-medium"
          >
            Structural Twin
          </button>
          <button
            onClick={() => onNavigateTab("scenario_lab")}
            className="px-3 py-1.5 text-xs font-mono bg-[#EDF5FF] hover:bg-[#D0E2FF] border border-[#0F62FE] text-[#0F62FE] font-bold rounded cursor-pointer transition"
          >
            Scenario Lab (Compare)
          </button>
          <button
            onClick={() => onNavigateTab("model_validation")}
            className="px-3 py-1.5 text-xs font-mono bg-[#F4F4F4] hover:bg-[#E0E0E0] border border-[#E0E0E0] rounded text-[#161616] cursor-pointer transition font-medium"
          >
            Benchmark Validation
          </button>
        </div>
      </div>
    </div>
  );
};
