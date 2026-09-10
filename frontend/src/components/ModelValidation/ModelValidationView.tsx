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
  isLoading?: boolean;
}

export const ModelValidationView: React.FC<ModelValidationViewProps> = ({
  prediction,
  onRunGroundTruth,
  isLoading = false,
}) => {
  const [activeTab, setActiveTab] = useState<"live_comparison" | "frozen_benchmarks">("live_comparison");

  const val = prediction?.validation;
  const traj = prediction?.trajectories;

  return (
    <div className="p-6 space-y-6 font-sans text-[#E8E8DE]">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border border-white/[0.08] bg-[#0E1B17] p-5 rounded-lg shadow-lg">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#E8E8DE]">
              SURROGATE VALIDATION & OPENSEESPY BENCHMARKS
            </h1>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#73E6B5]/10 text-[#73E6B5] border border-[#73E6B5]/30 font-semibold">
              STRICT SCIENTIFIC DISCLOSURE
            </span>
          </div>
          <p className="text-xs text-[#82928B] mt-1 font-sans">
            Direct quantitative evaluation of SeismoFNO continuous neural operator predictions against OpenSeesPy non-linear time history ground truth.
          </p>
        </div>

        {/* View Switcher */}
        <div className="flex bg-[#07110F] p-1 rounded border border-white/[0.08] text-xs font-mono">
          <button
            onClick={() => setActiveTab("live_comparison")}
            className={`px-3 py-1.5 rounded cursor-pointer transition font-medium ${
              activeTab === "live_comparison"
                ? "bg-[#73E6B5] text-[#07110F] font-bold shadow-sm"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            Live Ground Truth Run
          </button>
          <button
            onClick={() => setActiveTab("frozen_benchmarks")}
            className={`px-3 py-1.5 rounded cursor-pointer transition font-medium ${
              activeTab === "frozen_benchmarks"
                ? "bg-[#73E6B5] text-[#07110F] font-bold shadow-sm"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            Frozen Benchmark Reports
          </button>
        </div>
      </div>

      {activeTab === "live_comparison" ? (
        <div className="space-y-6">
          {/* Action Trigger Card */}
          <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-5 flex flex-wrap items-center justify-between gap-4 shadow-lg">
            <div className="space-y-1">
              <div className="text-sm font-bold font-mono text-[#E8E8DE]">
                Concurrent OpenSeesPy SDOF NLTHA Verification
              </div>
              <p className="text-xs text-[#82928B] font-sans">
                Executes the authoritative OpenSeesPy C++ non-linear numerical solver on the active scenario and computes relative L2 error, peak error, and wall-clock speedup.
              </p>
            </div>

            <button
              onClick={onRunGroundTruth}
              disabled={isLoading}
              className="px-4 py-2.5 bg-[#73E6B5] hover:bg-[#5cd4a2] disabled:opacity-50 text-[#07110F] text-xs font-mono font-bold rounded flex items-center space-x-2 cursor-pointer transition shadow-md"
            >
              <Play size={14} className="fill-[#07110F]" />
              <span>{isLoading ? "RUNNING OPENSEESPY..." : "RUN OPENSEESPY VERIFICATION"}</span>
            </button>
          </div>

          {/* Validation Metrics Cards */}
          {val ? (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-4">
                <div className="text-[#82928B] text-xs font-sans mb-1 font-medium">RELATIVE L2 ERROR u(t)</div>
                <div className="text-2xl font-mono font-bold text-[#E8E8DE]">
                  {val.relative_l2_u_percent.toFixed(2)} %
                </div>
                <div className="text-[11px] font-mono text-[#82928B] mt-2">
                  <span>Elastic Regime L2 is ~10.8%</span>
                </div>
              </div>

              <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-4">
                <div className="text-[#82928B] text-xs font-sans mb-1 font-medium">PEAK DISPLACEMENT ERROR</div>
                <div className="text-2xl font-mono font-bold text-[#E8E8DE]">
                  {val.peak_u_error_percent.toFixed(2)} %
                </div>
                <div className="text-[11px] font-mono text-[#82928B] mt-2">
                  <span>RMSE: {val.rmse_u_mm.toFixed(2)} mm</span>
                </div>
              </div>

              <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-4">
                <div className="text-[#82928B] text-xs font-sans mb-1 font-medium">MEASURED SPEEDUP</div>
                <div className="text-2xl font-mono font-bold text-[#73E6B5]">
                  {val.speedup_factor.toFixed(1)}x faster
                </div>
                <div className="text-[11px] font-mono text-[#82928B] mt-2">
                  <span>FNO: {val.fno_latency_ms.toFixed(2)}ms | OpenSees: {val.opensees_latency_ms.toFixed(2)}ms</span>
                </div>
              </div>

              <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-4">
                <div className="text-[#82928B] text-xs font-sans mb-1 font-medium">BENCHMARK STATUS</div>
                <div className="text-sm font-mono font-bold text-[#73E6B5] flex items-center space-x-1.5 mt-2">
                  <CheckCircle2 size={16} />
                  <span>EMPIRICALLY MEASURED</span>
                </div>
                <div className="text-[10px] font-mono text-[#82928B] mt-2">
                  Direct in-memory OpenSeesPy call
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-[#D6B56D]/10 border border-[#D6B56D]/30 rounded-lg p-8 text-center space-y-2">
              <AlertTriangle size={24} className="mx-auto text-[#D6B56D]" />
              <div className="text-sm font-mono font-bold text-[#D6B56D]">
                Awaiting Validated Ground Truth Benchmark
              </div>
              <p className="text-xs text-[#82928B] max-w-md mx-auto font-sans leading-relaxed">
                Click "RUN OPENSEESPY VERIFICATION" above to solve the non-linear oscillator numerically and compare against the neural surrogate in real time.
              </p>
            </div>
          )}

          {/* Time History Comparison Overlay Plot */}
          {traj && traj.u_gt && traj.u_gt.length > 0 && (
            <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-5 space-y-3 shadow-lg">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.08] pb-3 text-xs font-mono">
                <span className="font-bold text-[#E8E8DE]">
                  TIME HISTORY OVERLAY: SEISMOFNO vs OPENSEESPY GROUND TRUTH
                </span>
                <div className="flex items-center space-x-4">
                  <span className="flex items-center space-x-1.5 text-[#73E6B5] font-semibold">
                    <span className="h-1.5 w-4 bg-[#73E6B5] inline-block" />
                    <span>SeismoFNO Prediction (Mint Solid)</span>
                  </span>
                  <span className="flex items-center space-x-1.5 text-[#D6B56D] font-semibold">
                    <span className="h-1.5 w-4 border-b-2 border-dashed border-[#D6B56D] inline-block" />
                    <span>OpenSeesPy Ground Truth (Amber Dashed)</span>
                  </span>
                </div>
              </div>

              {/* Chart background for high contrast */}
              <div className="h-64 bg-[#07110F] rounded-lg border border-white/[0.08] flex items-center justify-center p-3 relative">
                <svg className="w-full h-full" viewBox="0 0 600 240">
                  <line x1="20" y1="120" x2="580" y2="120" stroke="rgba(255,255,255,0.12)" strokeDasharray="3 3" />

                  {/* OpenSees Ground Truth (Amber Dashed) */}
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
                    stroke="#D6B56D"
                    strokeWidth="2.0"
                    strokeDasharray="5 3"
                  />

                  {/* SeismoFNO Surrogate (Mint solid #73E6B5) */}
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
                    stroke="#73E6B5"
                    strokeWidth="2.0"
                  />
                </svg>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Frozen Benchmark Artifact Reports */
        <div className="space-y-6">
          {/* Phase 6 Baseline Benchmark Table */}
          <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg overflow-hidden shadow-lg">
            <div className="p-4 border-b border-white/[0.08] text-xs font-mono bg-[#0B1714]">
              <div className="font-bold text-[#E8E8DE]">
                PHASE 6: FULL BENCHMARK SUMMARY (HELD-OUT EARTHQUAKE SPLIT)
              </div>
              <p className="text-[#82928B] text-[11px] mt-0.5 font-sans">
                Artifact: results/tables/phase6_full_benchmark_summary.md | Disaggregated by Elastic vs Post-Yield Regimes
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-[#07110F] text-[#82928B] uppercase text-[10px] border-b border-white/[0.08]">
                  <tr>
                    <th className="p-3 font-sans font-semibold">Model Architecture</th>
                    <th className="p-3">Elastic (μ ≤ 1) u(t)</th>
                    <th className="p-3">Post-Yield (μ &gt; 1) u(t)</th>
                    <th className="p-3">Overall Rel L2 u(t)</th>
                    <th className="p-3">Overall Rel L2 Eh(t)</th>
                    <th className="p-3">Parameters</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.06]">
                  <tr className="bg-[#0E1B17] hover:bg-[#101D19] transition-colors">
                    <td className="p-3 font-semibold text-[#E8E8DE] font-sans">Linear Baseline (MLP)</td>
                    <td className="p-3 text-[#82928B]">24.12 %</td>
                    <td className="p-3 text-[#E35D5D] font-bold">58.45 %</td>
                    <td className="p-3 text-[#82928B]">41.28 %</td>
                    <td className="p-3 text-[#82928B]">—</td>
                    <td className="p-3 text-[#82928B]">145,200</td>
                  </tr>
                  <tr className="bg-[#0B1714] hover:bg-[#101D19] transition-colors">
                    <td className="p-3 font-semibold text-[#E8E8DE] font-sans">Recurrent Neural Net (LSTM)</td>
                    <td className="p-3 text-[#82928B]">18.64 %</td>
                    <td className="p-3 text-[#E35D5D] font-bold">44.12 %</td>
                    <td className="p-3 text-[#82928B]">31.38 %</td>
                    <td className="p-3 text-[#82928B]">52.10 %</td>
                    <td className="p-3 text-[#82928B]">384,120</td>
                  </tr>
                  <tr className="bg-[#0E1B17] hover:bg-[#101D19] transition-colors">
                    <td className="p-3 font-semibold text-[#E8E8DE] font-sans">State-Augmented TCN</td>
                    <td className="p-3 text-[#82928B]">14.20 %</td>
                    <td className="p-3 text-[#D6B56D]">29.80 %</td>
                    <td className="p-3 text-[#82928B]">22.00 %</td>
                    <td className="p-3 text-[#82928B]">38.45 %</td>
                    <td className="p-3 text-[#82928B]">612,400</td>
                  </tr>
                  <tr className="bg-[#73E6B5]/10 border-t-2 border-[#73E6B5]/40 font-semibold text-[#73E6B5]">
                    <td className="p-3 text-[#E8E8DE] font-bold font-sans">SeismoFNO 1D (Ours)</td>
                    <td className="p-3 text-[#73E6B5] font-bold font-mono">10.82 %</td>
                    <td className="p-3 text-[#73E6B5] font-bold font-mono">22.09 %</td>
                    <td className="p-3 text-[#73E6B5] font-bold font-mono">16.45 %</td>
                    <td className="p-3 text-[#73E6B5] font-bold font-mono">18.72 %</td>
                    <td className="p-3 font-mono text-[#E8E8DE]">1,196,931</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* SDOF vs MDOF Physical Scope Disclosure Card */}
          <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-5 space-y-2 shadow-lg">
            <div className="text-xs font-mono font-bold text-[#E8E8DE] uppercase">
              Physical Scope & Dimensionality Disclosures
            </div>
            <div className="text-xs font-sans text-[#82928B] space-y-2 leading-relaxed">
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

export default ModelValidationView;
