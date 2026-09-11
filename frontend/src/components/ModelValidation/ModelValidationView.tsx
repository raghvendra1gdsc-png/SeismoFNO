import React, { useState } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  Play,
  Activity,
} from "lucide-react";
import type {
  ScenarioPredictionResponse,
  ScenarioInputPayload,
  PeerRecord,
} from "../../api/digitalTwinApi";

interface ModelValidationViewProps {
  prediction: ScenarioPredictionResponse | null;
  onRunGroundTruth: (overrideParams?: Partial<ScenarioInputPayload>) => void;
  isLoading?: boolean;
  peerRecords?: PeerRecord[];
  currentEarthquakeId?: string;
  currentPgaG?: number;
  currentT0?: number;
  currentDamping?: number;
  currentUy?: number;
  currentAlpha?: number;
  currentMaterialType?: "bilinear" | "elastic";
}

export const ModelValidationView: React.FC<ModelValidationViewProps> = ({
  prediction,
  onRunGroundTruth,
  isLoading = false,
  peerRecords = [],
  currentEarthquakeId = "RSN0001_Imperial_Valley-06.AT2",
  currentPgaG = 0.40,
  currentT0 = 0.50,
  currentDamping = 0.05,
  currentUy = 0.010,
  currentAlpha = 0.05,
  currentMaterialType = "bilinear",
}) => {
  const [activeTab, setActiveTab] = useState<"live_comparison" | "frozen_benchmarks">("live_comparison");

  // Local interactive parameter state for custom OpenSees verification runs
  const [selectedRecordId, setSelectedRecordId] = useState<string>(currentEarthquakeId);
  const [pga, setPga] = useState<number>(currentPgaG);
  const [T1, setT1] = useState<number>(currentT0);
  const [damping, setDamping] = useState<number>(currentDamping);
  const [uyMm, setUyMm] = useState<number>(currentUy * 1000.0);
  const [alphaPct, setAlphaPct] = useState<number>(currentAlpha * 100.0);
  const [material, setMaterial] = useState<"bilinear" | "elastic">(currentMaterialType);

  const val = prediction?.validation;
  const traj = prediction?.trajectories;

  // Preset handlers
  const handleApplyPreset = (
    presetPga: number,
    presetT1: number,
    presetUyMm: number,
    presetAlphaPct: number,
    presetMat: "bilinear" | "elastic",
    recordId?: string
  ) => {
    setPga(presetPga);
    setT1(presetT1);
    setUyMm(presetUyMm);
    setAlphaPct(presetAlphaPct);
    setMaterial(presetMat);
    if (recordId) setSelectedRecordId(recordId);

    onRunGroundTruth({
      earthquake_id: recordId || selectedRecordId,
      pga_g: presetPga,
      T0: presetT1,
      damping_ratio: damping,
      yield_displacement_m: presetUyMm / 1000.0,
      post_yield_ratio: presetAlphaPct / 100.0,
      material_type: presetMat,
    });
  };

  const handleExecuteVerification = () => {
    onRunGroundTruth({
      earthquake_id: selectedRecordId,
      pga_g: pga,
      T0: T1,
      damping_ratio: damping,
      yield_displacement_m: uyMm / 1000.0,
      post_yield_ratio: alphaPct / 100.0,
      material_type: material,
    });
  };

  return (
    <div className="p-4 md:p-6 space-y-6 font-sans text-[#E8E8DE] max-w-7xl mx-auto">
      {/* 1. Academic Header with Clear Explanation */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border border-white/[0.08] bg-[#0E1B17] p-5 rounded-lg shadow-lg">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#E8E8DE] flex items-center gap-2">
              <Activity size={18} className="text-[#73E6B5]" />
              OPENSEESPY GROUND TRUTH NUMERICAL BENCHMARK
            </h1>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#73E6B5]/10 text-[#73E6B5] border border-[#73E6B5]/30 font-semibold">
              C++ NEWMARK-β INTEGRATOR
            </span>
          </div>
          <p className="text-xs text-[#82928B] mt-1.5 font-sans leading-relaxed max-w-4xl">
            Solves the nonlinear dynamic equation of motion{" "}
            <span className="font-mono text-[#E8E8DE]">mü(t) + c u̇(t) + Fs(u, u̇) = -müg(t)</span> using step-by-step
            implicit Newmark-β integration in OpenSeesPy C-runtime, directly comparing against the instantaneous SeismoFNO
            surrogate trajectory.
          </p>
        </div>

        {/* View Switcher */}
        <div className="flex bg-[#07110F] p-1 rounded border border-white/[0.08] text-xs font-mono shrink-0">
          <button
            onClick={() => setActiveTab("live_comparison")}
            className={`px-3 py-1.5 rounded cursor-pointer transition font-medium ${
              activeTab === "live_comparison"
                ? "bg-[#73E6B5] text-[#07110F] font-bold shadow-sm"
                : "text-[#82928B] hover:text-[#E8E8DE]"
            }`}
          >
            Live Numerical Run
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
          {/* 2. Educational Explainer: What This Verification Does */}
          <div className="bg-[#0B1714] border border-white/[0.08] rounded-lg p-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-sans">
            <div className="space-y-1">
              <span className="text-[#73E6B5] font-mono font-bold uppercase text-[10px] block">
                1. The Physical Problem
              </span>
              <p className="text-[#82928B] leading-relaxed">
                Nonlinear hysteretic response of a structural oscillator under seismic ground shaking. Restoring force{" "}
                <span className="text-[#E8E8DE] font-mono">Fs(u)</span> follows kinematic bilinear elastoplasticity with yield
                displacement <span className="text-[#E8E8DE] font-mono">uy</span> and strain hardening{" "}
                <span className="text-[#E8E8DE] font-mono">α</span>.
              </p>
            </div>

            <div className="space-y-1">
              <span className="text-[#D6B56D] font-mono font-bold uppercase text-[10px] block">
                2. OpenSeesPy C++ Solver
              </span>
              <p className="text-[#82928B] leading-relaxed">
                The reference solution is integrated time-step by time-step via Newton-Raphson tangent iterations. Wall-clock
                execution is measured live in Python memory to determine empirical computational speedup.
              </p>
            </div>

            <div className="space-y-1">
              <span className="text-[#73E6B5] font-mono font-bold uppercase text-[10px] block">
                3. Continuous Neural Operator
              </span>
              <p className="text-[#82928B] leading-relaxed">
                SeismoFNO evaluates the continuous trajectory in a single forward pass without numerical iteration. Error is
                quantified as Pointwise Relative L₂ Norm:{" "}
                <span className="text-[#E8E8DE] font-mono">||u_pred - u_true||₂ / ||u_true||₂ × 100%</span>.
              </p>
            </div>
          </div>

          {/* 3. Physical Benchmark Presets & Custom Verification Runner */}
          <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-5 space-y-4 shadow-lg">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-white/[0.06] pb-3">
              <div>
                <span className="text-xs font-mono font-bold text-[#E8E8DE] uppercase">
                  Select Physical Benchmark Regime or Customize Parameters
                </span>
                <p className="text-[11px] text-[#82928B] font-sans">
                  Choose a regime to demonstrate how the neural operator generalizes across linear-elastic, yielding, and severe hysteretic degradation.
                </p>
              </div>

              {/* Quick Preset Buttons */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => handleApplyPreset(0.15, 0.60, 25.0, 5.0, "bilinear")}
                  className="px-2.5 py-1 text-[11px] font-mono bg-[#07110F] hover:bg-[#17483A] text-[#73E6B5] border border-[#73E6B5]/30 rounded cursor-pointer transition"
                >
                  Regime A: Elastic (μ ≤ 1.0)
                </button>
                <button
                  onClick={() => handleApplyPreset(0.40, 0.50, 12.0, 5.0, "bilinear")}
                  className="px-2.5 py-1 text-[11px] font-mono bg-[#07110F] hover:bg-[#17483A] text-[#D6B56D] border border-[#D6B56D]/30 rounded cursor-pointer transition"
                >
                  Regime B: Moderate Yielding (μ ≈ 2.2)
                </button>
                <button
                  onClick={() => handleApplyPreset(0.85, 0.35, 6.0, 2.0, "bilinear")}
                  className="px-2.5 py-1 text-[11px] font-mono bg-[#07110F] hover:bg-[#17483A] text-[#E35D5D] border border-[#E35D5D]/30 rounded cursor-pointer transition"
                >
                  Regime C: Severe Inelastic (μ &gt; 4.5)
                </button>
              </div>
            </div>

            {/* Interactive Sliders Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 text-xs font-mono">
              {/* Earthquake Selection */}
              <div className="space-y-1">
                <label className="text-[10px] text-[#82928B] uppercase block font-semibold">
                  Seismic Excitation (PEER)
                </label>
                <select
                  value={selectedRecordId}
                  onChange={(e) => setSelectedRecordId(e.target.value)}
                  className="w-full bg-[#07110F] border border-white/[0.08] rounded px-2.5 py-1.5 text-xs text-[#E8E8DE] outline-none cursor-pointer focus:border-[#73E6B5]"
                >
                  {peerRecords.length > 0 ? (
                    peerRecords.map((r) => (
                      <option key={r.filename} value={r.filename}>
                        {r.earthquake_name || r.filename} ({r.raw_pga_g?.toFixed(2) ?? "—"}g)
                      </option>
                    ))
                  ) : (
                    <option value={currentEarthquakeId}>Imperial Valley-06 (RSN0001)</option>
                  )}
                </select>
              </div>

              {/* PGA Slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#82928B]">Target PGA:</span>
                  <span className="text-[#73E6B5] font-bold">{pga.toFixed(2)} g</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="1.20"
                  step="0.05"
                  value={pga}
                  onChange={(e) => setPga(parseFloat(e.target.value))}
                  className="w-full accent-[#73E6B5] cursor-pointer"
                />
                <div className="flex justify-between text-[9px] text-[#82928B]">
                  <span>0.05g (Minor)</span>
                  <span>1.20g (Severe MCE)</span>
                </div>
              </div>

              {/* Period T1 Slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#82928B]">Period T₁:</span>
                  <span className="text-[#73E6B5] font-bold">{T1.toFixed(2)} s</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="2.50"
                  step="0.05"
                  value={T1}
                  onChange={(e) => setT1(parseFloat(e.target.value))}
                  className="w-full accent-[#73E6B5] cursor-pointer"
                />
                <div className="flex justify-between text-[9px] text-[#82928B]">
                  <span>0.10s (Stiff)</span>
                  <span>2.50s (Flexible)</span>
                </div>
              </div>

              {/* Damping Slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#82928B]">Damping ζ:</span>
                  <span className="text-[#73E6B5] font-bold">{(damping * 100).toFixed(1)} %</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.15"
                  step="0.01"
                  value={damping}
                  onChange={(e) => setDamping(parseFloat(e.target.value))}
                  className="w-full accent-[#73E6B5] cursor-pointer"
                />
                <div className="flex justify-between text-[9px] text-[#82928B]">
                  <span>1% (Light)</span>
                  <span>15% (Heavy)</span>
                </div>
              </div>

              {/* Yield Displacement uy Slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#82928B]">Yield Limit uy:</span>
                  <span className="text-[#73E6B5] font-bold">{uyMm.toFixed(1)} mm</span>
                </div>
                <input
                  type="range"
                  min="2.0"
                  max="40.0"
                  step="1.0"
                  value={uyMm}
                  onChange={(e) => setUyMm(parseFloat(e.target.value))}
                  className="w-full accent-[#73E6B5] cursor-pointer"
                />
                <div className="flex justify-between text-[9px] text-[#82928B]">
                  <span>2.0mm (Brittle)</span>
                  <span>40.0mm (Ductile)</span>
                </div>
              </div>
            </div>

            {/* Execute Button Bar */}
            <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-white/[0.06]">
              <div className="text-[11px] font-mono text-[#82928B]">
                Active Config: <strong className="text-[#E8E8DE]">PGA={pga.toFixed(2)}g</strong> ·{" "}
                <strong className="text-[#E8E8DE]">T₁={T1.toFixed(2)}s</strong> ·{" "}
                <strong className="text-[#E8E8DE]">uy={uyMm.toFixed(1)}mm</strong> ·{" "}
                <strong className="text-[#E8E8DE]">α={alphaPct.toFixed(0)}%</strong>
              </div>

              <button
                onClick={handleExecuteVerification}
                disabled={isLoading}
                className="px-5 py-2.5 bg-[#73E6B5] hover:bg-[#5cd4a2] disabled:opacity-50 text-[#07110F] text-xs font-mono font-bold rounded flex items-center justify-center space-x-2 cursor-pointer transition shadow-md shrink-0"
              >
                <Play size={14} className={isLoading ? "animate-spin fill-[#07110F]" : "fill-[#07110F]"} />
                <span>
                  {isLoading
                    ? "SOLVING OPENSEESPY C++ NLTHA..."
                    : "EXECUTE OPENSEESPY VERIFICATION"}
                </span>
              </button>
            </div>
          </div>

          {/* 4. Validation Metrics Cards */}
          {val ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-4">
                <div className="text-[#82928B] text-xs font-sans mb-1 font-medium">RELATIVE L2 ERROR u(t)</div>
                <div
                  className={`text-2xl font-mono font-bold ${
                    val.relative_l2_u_percent > 20
                      ? "text-[#E35D5D]"
                      : val.relative_l2_u_percent > 10
                      ? "text-[#D6B56D]"
                      : "text-[#73E6B5]"
                  }`}
                >
                  {val.relative_l2_u_percent.toFixed(2)} %
                </div>
                <div className="text-[11px] font-mono text-[#82928B] mt-2">
                  <span>||u_FNO - u_GT||₂ / ||u_GT||₂</span>
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
                  In-memory OpenSeesPy C-runtime
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-[#D6B56D]/10 border border-[#D6B56D]/30 rounded-lg p-8 text-center space-y-2">
              <AlertTriangle size={24} className="mx-auto text-[#D6B56D]" />
              <div className="text-sm font-mono font-bold text-[#D6B56D]">
                Awaiting Validated Ground Truth Benchmark Execution
              </div>
              <p className="text-xs text-[#82928B] max-w-md mx-auto font-sans leading-relaxed">
                Click "EXECUTE OPENSEESPY VERIFICATION" or select a physical regime preset above to solve the non-linear oscillator in OpenSeesPy and compare against the neural surrogate in real time.
              </p>
            </div>
          )}

          {/* 5. High-Resolution Time History Comparison Overlay Plot */}
          {traj && traj.u_gt && traj.u_gt.length > 0 && (
            <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-5 space-y-3 shadow-lg">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.08] pb-3 text-xs font-mono">
                <div>
                  <span className="font-bold text-[#E8E8DE] block">
                    TIME HISTORY OVERLAY: SEISMOFNO vs OPENSEESPY GROUND TRUTH
                  </span>
                  <span className="text-[10px] text-[#82928B] font-sans">
                    Displacement response u(t) over 2,048 timesteps under active excitation.
                  </span>
                </div>
                <div className="flex items-center space-x-4">
                  <span className="flex items-center space-x-1.5 text-[#73E6B5] font-semibold">
                    <span className="h-1.5 w-4 bg-[#73E6B5] inline-block rounded-xs" />
                    <span>SeismoFNO Neural Operator (Mint Solid)</span>
                  </span>
                  <span className="flex items-center space-x-1.5 text-[#D6B56D] font-semibold">
                    <span className="h-1.5 w-4 border-b-2 border-dashed border-[#D6B56D] inline-block" />
                    <span>OpenSeesPy C++ NLTHA (Amber Dashed)</span>
                  </span>
                </div>
              </div>

              {/* Chart canvas */}
              <div className="h-72 bg-[#07110F] rounded-lg border border-white/[0.08] flex items-center justify-center p-3 relative">
                <svg className="w-full h-full" viewBox="0 0 700 240">
                  <line x1="20" y1="120" x2="680" y2="120" stroke="rgba(255,255,255,0.12)" strokeDasharray="3 3" />

                  {/* OpenSees Ground Truth (Amber Dashed) */}
                  <path
                    d={traj.u_gt
                      .map((val, i, arr) => {
                        const x = 20 + (i / (arr.length - 1)) * 660;
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
                        const x = 20 + (i / (arr.length - 1)) * 660;
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

              {/* Pointwise Residual Discrepancy Strip */}
              <div className="p-3 bg-[#07110F] rounded border border-white/[0.06] flex items-center justify-between text-[11px] font-mono text-[#82928B]">
                <span>
                  Max Absolute Deviation:{" "}
                  <strong className="text-[#E8E8DE]">
                    {(
                      Math.max(...traj.u.map((u, i) => Math.abs(u - (traj.u_gt?.[i] ?? 0)))) * 1000.0
                    ).toFixed(2)}{" "}
                    mm
                  </strong>
                </span>
                <span>
                  Peak Ground Truth:{" "}
                  <strong className="text-[#D6B56D]">
                    {(Math.max(...(traj.u_gt || []).map(Math.abs)) * 1000.0).toFixed(2)} mm
                  </strong>
                </span>
                <span>
                  Peak Surrogate:{" "}
                  <strong className="text-[#73E6B5]">
                    {(Math.max(...traj.u.map(Math.abs)) * 1000.0).toFixed(2)} mm
                  </strong>
                </span>
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
                Source: results/tables/phase6_full_benchmark_summary.md | Disaggregated by Elastic vs Post-Yield Regimes
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
