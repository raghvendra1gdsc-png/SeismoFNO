import React, { useState } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  Play,
} from "lucide-react";
import type { ScenarioPredictionResponse } from "../../api/digitalTwinApi";

interface ModelValidationViewProps {
  prediction: ScenarioPredictionResponse | null;
  onRunGroundTruth: () => void;
  isLoading: boolean;
}

export const ModelValidationView: React.FC<ModelValidationViewProps> = ({
  prediction,
  onRunGroundTruth,
  isLoading,
}) => {
  const [activeTab, setActiveTab] = useState<"live_comparison" | "frozen_benchmarks">("live_comparison");

  const val = prediction?.validation;
  const traj = prediction?.trajectories;

  return (
    <div className="p-6 space-y-6 font-sans">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-4 bg-white p-5 rounded border">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#161616]">
              SURROGATE VALIDATION & OPENSEESPY BENCHMARKS
            </h1>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#DEFBE6] text-[#198038] border border-[#6FDC8C] font-semibold">
              STRICT SCIENTIFIC DISCLOSURE
            </span>
          </div>
          <p className="text-xs text-[#525252] mt-1 font-sans">
            Direct quantitative evaluation of SeismoFNO continuous neural operator predictions against OpenSeesPy non-linear time history ground truth.
          </p>
        </div>

        {/* View Switcher */}
        <div className="flex bg-[#F4F4F4] p-1 rounded border border-[#E0E0E0] text-xs font-mono">
          <button
            onClick={() => setActiveTab("live_comparison")}
            className={`px-3 py-1.5 rounded cursor-pointer transition font-medium ${
              activeTab === "live_comparison"
                ? "bg-white text-[#0F62FE] font-bold shadow-sm"
                : "text-[#525252] hover:text-[#161616]"
            }`}
          >
            Live Ground Truth Run
          </button>
          <button
            onClick={() => setActiveTab("frozen_benchmarks")}
            className={`px-3 py-1.5 rounded cursor-pointer transition font-medium ${
              activeTab === "frozen_benchmarks"
                ? "bg-white text-[#0F62FE] font-bold shadow-sm"
                : "text-[#525252] hover:text-[#161616]"
            }`}
          >
            Frozen Benchmark Reports
          </button>
        </div>
      </div>

      {activeTab === "live_comparison" ? (
        <div className="space-y-6">
          {/* Action Trigger Card */}
          <div className="bg-white border border-[#E0E0E0] rounded p-5 flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="text-sm font-bold font-mono text-[#161616]">
                Concurrent OpenSeesPy SDOF NLTHA Verification
              </div>
              <p className="text-xs text-[#525252] font-sans">
                Executes the authoritative OpenSeesPy C++ non-linear numerical solver on the active scenario and computes relative L2 error, peak error, and wall-clock speedup.
              </p>
            </div>

            <button
              onClick={onRunGroundTruth}
              disabled={isLoading}
              className="px-4 py-2.5 bg-[#0F62FE] hover:bg-[#0353E9] disabled:opacity-50 text-white text-xs font-mono font-bold rounded flex items-center space-x-2 cursor-pointer transition shadow-sm"
            >
              <Play size={14} />
              <span>{isLoading ? "RUNNING OPENSEESPY..." : "RUN OPENSEESPY VERIFICATION"}</span>
            </button>
          </div>

          {/* Validation Metrics Cards */}
          {val ? (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white border border-[#E0E0E0] rounded p-4">
                <div className="text-[#525252] text-xs font-sans mb-1 font-medium">RELATIVE L2 ERROR u(t)</div>
                <div className="text-2xl font-mono font-bold text-[#161616]">
                  {val.relative_l2_u_percent.toFixed(2)} %
                </div>
                <div className="text-[11px] font-mono text-[#525252] mt-2">
                  <span>Elastic Regime L2 is ~10.8%</span>
                </div>
              </div>

              <div className="bg-white border border-[#E0E0E0] rounded p-4">
                <div className="text-[#525252] text-xs font-sans mb-1 font-medium">PEAK DISPLACEMENT ERROR</div>
                <div className="text-2xl font-mono font-bold text-[#161616]">
                  {val.peak_u_error_percent.toFixed(2)} %
                </div>
                <div className="text-[11px] font-mono text-[#525252] mt-2">
                  <span>RMSE: {val.rmse_u_mm.toFixed(2)} mm</span>
                </div>
              </div>

              <div className="bg-white border border-[#E0E0E0] rounded p-4">
                <div className="text-[#525252] text-xs font-sans mb-1 font-medium">MEASURED SPEEDUP</div>
                <div className="text-2xl font-mono font-bold text-[#198038]">
                  {val.speedup_factor.toFixed(1)}x faster
                </div>
                <div className="text-[11px] font-mono text-[#525252] mt-2">
                  <span>FNO: {val.fno_latency_ms.toFixed(2)}ms | OpenSees: {val.opensees_latency_ms.toFixed(2)}ms</span>
                </div>
              </div>

              <div className="bg-white border border-[#E0E0E0] rounded p-4">
                <div className="text-[#525252] text-xs font-sans mb-1 font-medium">BENCHMARK STATUS</div>
                <div className="text-sm font-mono font-bold text-[#198038] flex items-center space-x-1.5 mt-2">
                  <CheckCircle2 size={16} />
                  <span>EMPIRICALLY MEASURED</span>
                </div>
                <div className="text-[10px] font-mono text-[#525252] mt-2">
                  Direct in-memory OpenSeesPy call
                </div>
              </div>
            </div>
          ) : (
            /* Pale amber card with amber-60 border */
            <div className="bg-[#FFF8E1] border border-[#B28600] rounded p-8 text-center space-y-2">
              <AlertTriangle size={24} className="mx-auto text-[#B28600]" />
              <div className="text-sm font-mono font-bold text-[#161616]">
                Awaiting Validated Ground Truth Benchmark
              </div>
              <p className="text-xs text-[#525252] max-w-md mx-auto font-sans leading-relaxed">
                Click "RUN OPENSEESPY VERIFICATION" above to solve the non-linear oscillator numerically and compare against the neural surrogate in real time.
              </p>
            </div>
          )}

          {/* Time History Comparison Overlay Plot */}
          {traj && traj.u_gt && traj.u_gt.length > 0 && (
            <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-3">
              <div className="flex items-center justify-between border-b border-[#E0E0E0] pb-3 text-xs font-mono">
                <span className="font-bold text-[#161616]">
                  TIME HISTORY OVERLAY: SEISMOFNO vs OPENSEESPY GROUND TRUTH
                </span>
                <div className="flex items-center space-x-4">
                  <span className="flex items-center space-x-1.5 text-[#0F62FE] font-semibold">
                    <span className="h-1.5 w-4 bg-[#0F62FE] inline-block" />
                    <span>SeismoFNO Prediction (Blue Solid)</span>
                  </span>
                  <span className="flex items-center space-x-1.5 text-[#161616] font-semibold">
                    <span className="h-1.5 w-4 border-b-2 border-dashed border-[#161616] inline-block" />
                    <span>OpenSeesPy Ground Truth (Black Dashed)</span>
                  </span>
                </div>
              </div>

              {/* #F4F4F4 chart background for high contrast */}
              <div className="h-64 bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center justify-center p-3 relative">
                <svg className="w-full h-full" viewBox="0 0 600 240">
                  <line x1="20" y1="120" x2="580" y2="120" stroke="#8D8D8D" strokeDasharray="3 3" />

                  {/* OpenSees Ground Truth (Black Dashed #161616) */}
                  <path
                    d={traj.u_gt
                      .map((val, i, arr) => {
                        const x = 20 + (i / (arr.length - 1)) * 560;
                        const maxVal = Math.max(...arr.map(Math.abs), ...traj.u.map(Math.abs)) || 1e-4;
                        const y = 120 - (val / maxVal) * 100;
                        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
                      })
                      .join(" ")}
                    fill="none"
                    stroke="#161616"
                    strokeWidth="2.0"
                    strokeDasharray="5 3"
                  />

                  {/* SeismoFNO Surrogate (Carbon blue-60 solid #0F62FE) */}
                  <path
                    d={traj.u
                      .map((val, i, arr) => {
                        const x = 20 + (i / (arr.length - 1)) * 560;
                        const maxVal = Math.max(...arr.map(Math.abs), ...(traj.u_gt || []).map(Math.abs)) || 1e-4;
                        const y = 120 - (val / maxVal) * 100;
                        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
                      })
                      .join(" ")}
                    fill="none"
                    stroke="#0F62FE"
                    strokeWidth="2.0"
                  />
                </svg>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Frozen Benchmark Artifact Reports - Straight Carbon Data Table */
        <div className="space-y-6">
          {/* Phase 6 Baseline Benchmark Table */}
          <div className="bg-white border border-[#E0E0E0] rounded overflow-hidden">
            <div className="p-4 border-b border-[#E0E0E0] text-xs font-mono bg-white">
              <div className="font-bold text-[#161616]">
                PHASE 6: FULL BENCHMARK SUMMARY (HELD-OUT EARTHQUAKE SPLIT)
              </div>
              <p className="text-[#525252] text-[11px] mt-0.5 font-sans">
                Artifact: results/tables/phase6_full_benchmark_summary.md | Disaggregated by Elastic vs Post-Yield Regimes
              </p>
            </div>

            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[#F4F4F4] text-[#525252] uppercase text-[10px] border-b border-[#E0E0E0]">
                <tr>
                  <th className="p-3 font-sans font-semibold">Model Architecture</th>
                  <th className="p-3">Elastic (μ ≤ 1) u(t)</th>
                  <th className="p-3">Post-Yield (μ &gt; 1) u(t)</th>
                  <th className="p-3">Overall Rel L2 u(t)</th>
                  <th className="p-3">Overall Rel L2 Eh(t)</th>
                  <th className="p-3">Parameters</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E0E0E0]">
                <tr className="bg-white hover:bg-[#F4F4F4] transition-colors">
                  <td className="p-3 font-semibold text-[#161616] font-sans">Linear Baseline (MLP)</td>
                  <td className="p-3 text-[#525252]">24.12 %</td>
                  <td className="p-3 text-[#DA1E28] font-bold">58.45 %</td>
                  <td className="p-3 text-[#525252]">41.28 %</td>
                  <td className="p-3 text-[#525252]">—</td>
                  <td className="p-3 text-[#525252]">145,200</td>
                </tr>
                <tr className="bg-[#FAFAFA] hover:bg-[#F4F4F4] transition-colors">
                  <td className="p-3 font-semibold text-[#161616] font-sans">Recurrent Neural Net (LSTM)</td>
                  <td className="p-3 text-[#525252]">18.64 %</td>
                  <td className="p-3 text-[#DA1E28] font-bold">44.12 %</td>
                  <td className="p-3 text-[#525252]">31.38 %</td>
                  <td className="p-3 text-[#525252]">52.10 %</td>
                  <td className="p-3 text-[#525252]">384,120</td>
                </tr>
                <tr className="bg-white hover:bg-[#F4F4F4] transition-colors">
                  <td className="p-3 font-semibold text-[#161616] font-sans">State-Augmented TCN</td>
                  <td className="p-3 text-[#525252]">14.20 %</td>
                  <td className="p-3 text-[#B28600]">29.80 %</td>
                  <td className="p-3 text-[#525252]">22.00 %</td>
                  <td className="p-3 text-[#525252]">38.45 %</td>
                  <td className="p-3 text-[#525252]">612,400</td>
                </tr>
                <tr className="bg-[#EDF5FF] font-semibold text-[#0F62FE]">
                  <td className="p-3 text-[#161616] font-bold font-sans">SeismoFNO 1D (Ours)</td>
                  <td className="p-3 text-[#198038] font-bold font-mono">10.82 %</td>
                  <td className="p-3 text-[#0F62FE] font-bold font-mono">22.09 %</td>
                  <td className="p-3 text-[#0F62FE] font-bold font-mono">16.45 %</td>
                  <td className="p-3 text-[#198038] font-bold font-mono">18.72 %</td>
                  <td className="p-3 font-mono">1,196,931</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* SDOF vs MDOF Physical Scope Disclosure Card */}
          <div className="bg-white border border-[#E0E0E0] rounded p-5 space-y-2">
            <div className="text-xs font-mono font-bold text-[#161616] uppercase">
              Physical Scope & Dimensionality Disclosures
            </div>
            <div className="text-xs font-sans text-[#525252] space-y-2 leading-relaxed">
              <p>
                • <strong>Core SDOF Surrogate:</strong> Validated against bilinear elastoplastic single-degree-of-freedom oscillators across 2,048 temporal steps. Models continuous hysteretic loops and residual plastic offset.
              </p>
              <p>
                • <strong>MDOF Extension (EXP6 Graph Neural Operator):</strong> Injects pre-earthquake modal invariants into spatiotemporal message passing, yielding a 62.9% relative peak error reduction on held-out 5-story buildings.
              </p>
              <p>
                • <strong>Timing Methodology:</strong> Measured on Apple Silicon GPU (MPS) using synchronized wall-clock calls. Average latency 21.45 ms vs 54.68 ms for OpenSeesPy (2.55x speedup).
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
