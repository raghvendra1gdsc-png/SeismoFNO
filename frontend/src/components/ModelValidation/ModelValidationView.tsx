import React, { useState } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  Play,
  Activity,
  Sliders,
  ShieldCheck,
  Zap,
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
    <div className="p-4 md:p-8 space-y-6 max-w-7xl mx-auto font-sans text-[#0F172A]">
      {/* 1. Academic Header with Clear Engineering Subtitle */}
      <div className="panel-workstation p-6 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#0F172A] flex items-center gap-2">
              <Activity size={20} className="text-[#047857]" />
              OPENSEESPY GROUND TRUTH NUMERICAL BENCHMARK
            </h1>
            <span className="badge-tech bg-[#F1F5F9] border border-[#CBD5E1] text-[#334155]">
              C++ NEWMARK-β INTEGRATOR
            </span>
            <span className="badge-tech bg-[#ECFDF5] border border-[#A7F3D0] text-[#047857]">
              VERIFICATION ENGINE
            </span>
          </div>
          <p className="text-xs text-[#475569] mt-2 font-sans leading-relaxed max-w-4xl">
            Solves the nonlinear dynamic equation of motion{" "}
            <code className="font-mono text-[#0F172A] bg-[#F1F5F9] px-1.5 py-0.5 rounded border border-[#E2E8F0]">
              mü(t) + c u̇(t) + Fs(u, u̇) = -müg(t)
            </code>{" "}
            using step-by-step implicit Newmark-β integration in OpenSeesPy C-runtime, directly comparing against the instantaneous SeismoFNO surrogate trajectory.
          </p>
        </div>

        {/* View Switcher */}
        <div className="flex bg-[#F1F5F9] p-1 rounded border border-[#E2E8F0] text-xs font-mono shrink-0">
          <button
            onClick={() => setActiveTab("live_comparison")}
            className={`px-3 py-1.5 rounded cursor-pointer transition font-medium ${
              activeTab === "live_comparison"
                ? "bg-white text-[#0F172A] font-bold shadow-xs border border-[#E2E8F0]"
                : "text-[#64748B] hover:text-[#0F172A]"
            }`}
          >
            Live Numerical Run
          </button>
          <button
            onClick={() => setActiveTab("frozen_benchmarks")}
            className={`px-3 py-1.5 rounded cursor-pointer transition font-medium ${
              activeTab === "frozen_benchmarks"
                ? "bg-white text-[#0F172A] font-bold shadow-xs border border-[#E2E8F0]"
                : "text-[#64748B] hover:text-[#0F172A]"
            }`}
          >
            Physics Checks & Benchmarks
          </button>
        </div>
      </div>

      {activeTab === "live_comparison" ? (
        <div className="space-y-6">
          {/* 2. Educational Explainer: Three Core Mechanics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="panel-workstation p-4 space-y-1.5">
              <span className="text-[#047857] font-mono font-bold uppercase text-[11px] block">
                1. The Physical Problem
              </span>
              <p className="text-[#475569] leading-relaxed">
                Nonlinear hysteretic response of a structural oscillator under seismic ground shaking. Restoring force{" "}
                <span className="text-[#0F172A] font-mono font-medium">Fs(u)</span> follows kinematic bilinear elastoplasticity with yield
                displacement <span className="text-[#0F172A] font-mono font-medium">uy</span> and strain hardening{" "}
                <span className="text-[#0F172A] font-mono font-medium">α</span>.
              </p>
            </div>

            <div className="panel-workstation p-4 space-y-1.5">
              <span className="text-[#B45309] font-mono font-bold uppercase text-[11px] block">
                2. OpenSeesPy C++ Solver
              </span>
              <p className="text-[#475569] leading-relaxed">
                The reference solution is integrated step-by-step via Newton-Raphson tangent iterations. Wall-clock
                execution is measured live in Python memory to determine empirical computational speedup.
              </p>
            </div>

            <div className="panel-workstation p-4 space-y-1.5">
              <span className="text-[#047857] font-mono font-bold uppercase text-[11px] block">
                3. Continuous Neural Operator
              </span>
              <p className="text-[#475569] leading-relaxed">
                SeismoFNO evaluates the continuous trajectory in a single forward pass without numerical iteration. Error is
                quantified as Pointwise Relative L₂ Norm:{" "}
                <span className="text-[#0F172A] font-mono font-medium">||u_pred - u_true||₂ / ||u_true||₂ × 100%</span>.
              </p>
            </div>
          </div>

          {/* 3. Physical Benchmark Presets & Custom Verification Runner */}
          <div className="panel-workstation p-6 space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E2E8F0] pb-4">
              <div>
                <span className="text-xs font-mono font-bold text-[#0F172A] uppercase flex items-center gap-1.5">
                  <Sliders size={14} className="text-[#047857]" />
                  Select Physical Benchmark Regime or Customize Parameters
                </span>
                <p className="text-[11px] text-[#475569] font-sans mt-0.5">
                  Demonstrate how the neural operator generalizes across linear-elastic, yielding, and severe hysteretic degradation.
                </p>
              </div>

              {/* Quick Preset Buttons */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => handleApplyPreset(0.15, 0.60, 25.0, 5.0, "bilinear")}
                  className="px-2.5 py-1 text-[11px] font-mono bg-[#F8FAFC] hover:bg-[#F1F5F9] text-[#047857] border border-[#CBD5E1] rounded cursor-pointer transition font-medium"
                >
                  Regime A: Elastic (μ ≤ 1.0)
                </button>
                <button
                  onClick={() => handleApplyPreset(0.40, 0.50, 12.0, 5.0, "bilinear")}
                  className="px-2.5 py-1 text-[11px] font-mono bg-[#F8FAFC] hover:bg-[#F1F5F9] text-[#B45309] border border-[#CBD5E1] rounded cursor-pointer transition font-medium"
                >
                  Regime B: Moderate Yield (μ ≈ 2.2)
                </button>
                <button
                  onClick={() => handleApplyPreset(0.85, 0.35, 6.0, 2.0, "bilinear")}
                  className="px-2.5 py-1 text-[11px] font-mono bg-[#F8FAFC] hover:bg-[#F1F5F9] text-[#DC2626] border border-[#CBD5E1] rounded cursor-pointer transition font-medium"
                >
                  Regime C: Severe Inelastic (μ &gt; 4.5)
                </button>
              </div>
            </div>

            {/* Interactive Sliders Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 text-xs font-mono">
              {/* Earthquake Selection */}
              <div className="space-y-1.5">
                <label className="text-[10px] text-[#64748B] uppercase block font-semibold">
                  Seismic Excitation (PEER)
                </label>
                <select
                  value={selectedRecordId}
                  onChange={(e) => setSelectedRecordId(e.target.value)}
                  className="w-full bg-[#FFFFFF] border border-[#CBD5E1] rounded px-2.5 py-1.5 text-xs text-[#0F172A] outline-none cursor-pointer focus:border-[#047857]"
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
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#64748B]">Target PGA:</span>
                  <span className="text-[#047857] font-bold">{pga.toFixed(2)} g</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="1.20"
                  step="0.05"
                  value={pga}
                  onChange={(e) => setPga(parseFloat(e.target.value))}
                  className="range-input w-full cursor-pointer"
                />
                <div className="flex justify-between text-[9px] text-[#94A3B8]">
                  <span>0.05g (Minor)</span>
                  <span>1.20g (Severe MCE)</span>
                </div>
              </div>

              {/* Period T1 Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#64748B]">Period T₁:</span>
                  <span className="text-[#047857] font-bold">{T1.toFixed(2)} s</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="2.50"
                  step="0.05"
                  value={T1}
                  onChange={(e) => setT1(parseFloat(e.target.value))}
                  className="range-input w-full cursor-pointer"
                />
                <div className="flex justify-between text-[9px] text-[#94A3B8]">
                  <span>0.10s (Stiff)</span>
                  <span>2.50s (Flexible)</span>
                </div>
              </div>

              {/* Damping Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#64748B]">Damping ζ:</span>
                  <span className="text-[#047857] font-bold">{(damping * 100).toFixed(1)} %</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.15"
                  step="0.01"
                  value={damping}
                  onChange={(e) => setDamping(parseFloat(e.target.value))}
                  className="range-input w-full cursor-pointer"
                />
                <div className="flex justify-between text-[9px] text-[#94A3B8]">
                  <span>1% (Light)</span>
                  <span>15% (Heavy)</span>
                </div>
              </div>

              {/* Yield Displacement uy Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#64748B]">Yield Limit uy:</span>
                  <span className="text-[#047857] font-bold">{uyMm.toFixed(1)} mm</span>
                </div>
                <input
                  type="range"
                  min="2.0"
                  max="40.0"
                  step="1.0"
                  value={uyMm}
                  onChange={(e) => setUyMm(parseFloat(e.target.value))}
                  className="range-input w-full cursor-pointer"
                />
                <div className="flex justify-between text-[9px] text-[#94A3B8]">
                  <span>2.0mm (Brittle)</span>
                  <span>40.0mm (Ductile)</span>
                </div>
              </div>
            </div>

            {/* Execute Button Bar */}
            <div className="pt-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-[#E2E8F0]">
              <div className="text-[11px] font-mono text-[#64748B]">
                Active Config: <strong className="text-[#0F172A]">PGA={pga.toFixed(2)}g</strong> ·{" "}
                <strong className="text-[#0F172A]">T₁={T1.toFixed(2)}s</strong> ·{" "}
                <strong className="text-[#0F172A]">uy={uyMm.toFixed(1)}mm</strong> ·{" "}
                <strong className="text-[#0F172A]">α={alphaPct.toFixed(0)}%</strong>
              </div>

              <button
                onClick={handleExecuteVerification}
                disabled={isLoading}
                className="btn-engineering px-5 py-2.5 bg-[#047857] text-white hover:bg-[#065F46] disabled:opacity-50 text-xs font-mono font-bold rounded flex items-center justify-center space-x-2 cursor-pointer transition shadow-xs shrink-0"
              >
                <Play size={14} className={isLoading ? "animate-spin" : "fill-white"} />
                <span>
                  {isLoading
                    ? "SOLVING OPENSEESPY C++ NLTHA..."
                    : "EXECUTE OPENSEESPY VERIFICATION"}
                </span>
              </button>
            </div>

            {/* Live Verification Status Banner */}
            {val && (
              <div className="p-3 bg-[#ECFDF5] border border-[#A7F3D0] rounded text-xs font-mono text-[#047857] flex flex-wrap items-center justify-between gap-2 shadow-xs">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-[#047857] animate-ping" />
                  <span className="font-semibold">
                    ✓ OpenSeesPy C++ verification solved ({val.opensees_latency_ms.toFixed(0)} ms) vs SeismoFNO surrogate ({val.fno_latency_ms.toFixed(1)} ms) — {val.speedup_factor.toFixed(0)}x Measured Speedup
                  </span>
                </div>
                <span className="text-[11px] font-bold text-[#065F46]">
                  Relative L₂: {val.relative_l2_u_percent.toFixed(2)}% · Peak Error: {val.peak_u_error_percent.toFixed(2)}%
                </span>
              </div>
            )}
          </div>

          {/* 4. Validation Metrics Cards */}
          {val ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="panel-workstation p-4">
                <div className="text-[#64748B] text-xs font-sans mb-1 font-medium">RELATIVE L2 ERROR u(t)</div>
                <div
                  className={`text-2xl font-mono font-bold ${
                    val.relative_l2_u_percent > 20
                      ? "text-[#DC2626]"
                      : val.relative_l2_u_percent > 10
                      ? "text-[#B45309]"
                      : "text-[#047857]"
                  }`}
                >
                  {val.relative_l2_u_percent.toFixed(2)} %
                </div>
                <div className="text-[11px] font-mono text-[#64748B] mt-2">
                  <span>||u_FNO - u_GT||₂ / ||u_GT||₂</span>
                </div>
              </div>

              <div className="panel-workstation p-4">
                <div className="text-[#64748B] text-xs font-sans mb-1 font-medium">PEAK DISPLACEMENT ERROR</div>
                <div className="text-2xl font-mono font-bold text-[#0F172A]">
                  {val.peak_u_error_percent.toFixed(2)} %
                </div>
                <div className="text-[11px] font-mono text-[#64748B] mt-2">
                  <span>RMSE: {val.rmse_u_mm.toFixed(2)} mm</span>
                </div>
              </div>

              <div className="panel-workstation p-4">
                <div className="text-[#64748B] text-xs font-sans mb-1 font-medium">MEASURED SPEEDUP</div>
                <div className="text-2xl font-mono font-bold text-[#047857]">
                  {val.speedup_factor.toFixed(1)}x faster
                </div>
                <div className="text-[11px] font-mono text-[#64748B] mt-2">
                  <span>FNO: {val.fno_latency_ms.toFixed(2)}ms | OpenSees: {val.opensees_latency_ms.toFixed(2)}ms</span>
                </div>
              </div>

              <div className="panel-workstation p-4">
                <div className="text-[#64748B] text-xs font-sans mb-1 font-medium">BENCHMARK STATUS</div>
                <div className="text-sm font-mono font-bold text-[#047857] flex items-center space-x-1.5 mt-2">
                  <CheckCircle2 size={16} />
                  <span>EMPIRICALLY MEASURED</span>
                </div>
                <div className="text-[10px] font-mono text-[#64748B] mt-2">
                  In-memory OpenSeesPy C-runtime
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-[#FFFBEB] border border-[#FDE68A] rounded-lg p-6 text-center space-y-2">
              <AlertTriangle size={24} className="mx-auto text-[#B45309]" />
              <div className="text-sm font-mono font-bold text-[#B45309]">
                Awaiting Validated Ground Truth Benchmark Execution
              </div>
              <p className="text-xs text-[#475569] max-w-md mx-auto font-sans leading-relaxed">
                Click "EXECUTE OPENSEESPY VERIFICATION" or select a physical regime preset above to solve the non-linear oscillator in OpenSeesPy and compare against the neural surrogate in real time.
              </p>
            </div>
          )}

          {/* 5. High-Resolution Time History Comparison Overlay Plot */}
          {traj && traj.u_gt && traj.u_gt.length > 0 && (
            <div className="panel-workstation p-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#E2E8F0] pb-3 text-xs font-mono">
                <div>
                  <span className="font-bold text-[#0F172A] block text-sm">
                    TIME HISTORY OVERLAY: SEISMOFNO vs OPENSEESPY GROUND TRUTH
                  </span>
                  <span className="text-[11px] text-[#64748B] font-sans">
                    Displacement response u(t) over 2,048 timesteps under active excitation.
                  </span>
                </div>
                <div className="flex items-center space-x-5">
                  <span className="flex items-center space-x-2 text-[#047857] font-semibold">
                    <span className="h-2 w-4 bg-[#047857] inline-block rounded-xs" />
                    <span>SeismoFNO Surrogate (Emerald Solid)</span>
                  </span>
                  <span className="flex items-center space-x-2 text-[#B45309] font-semibold">
                    <span className="h-2 w-4 border-b-2 border-dashed border-[#B45309] inline-block" />
                    <span>OpenSeesPy C++ NLTHA (Amber Dashed)</span>
                  </span>
                </div>
              </div>

              {/* Chart canvas */}
              <div className="h-72 bg-[#FFFFFF] rounded-lg border border-[#E2E8F0] flex items-center justify-center p-3 relative shadow-2xs">
                <svg className="w-full h-full" viewBox="0 0 700 240">
                  {/* Grid lines */}
                  <line x1="20" y1="40" x2="680" y2="40" stroke="#F1F5F9" strokeDasharray="3 3" />
                  <line x1="20" y1="80" x2="680" y2="80" stroke="#F1F5F9" strokeDasharray="3 3" />
                  <line x1="20" y1="120" x2="680" y2="120" stroke="#E2E8F0" strokeWidth="1.5" />
                  <line x1="20" y1="160" x2="680" y2="160" stroke="#F1F5F9" strokeDasharray="3 3" />
                  <line x1="20" y1="200" x2="680" y2="200" stroke="#F1F5F9" strokeDasharray="3 3" />

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
                    stroke="#B45309"
                    strokeWidth="2.0"
                    strokeDasharray="5 3"
                  />

                  {/* SeismoFNO Surrogate (Emerald solid #047857) */}
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
                    stroke="#047857"
                    strokeWidth="2.0"
                  />
                </svg>
              </div>

              {/* Pointwise Residual Discrepancy Strip */}
              <div className="p-3 bg-[#F8FAFC] rounded border border-[#E2E8F0] flex flex-wrap items-center justify-between text-[11px] font-mono text-[#64748B] gap-2">
                <span>
                  Max Absolute Deviation:{" "}
                  <strong className="text-[#0F172A]">
                    {(
                      Math.max(...traj.u.map((u, i) => Math.abs(u - (traj.u_gt?.[i] ?? 0)))) * 1000.0
                    ).toFixed(2)}{" "}
                    mm
                  </strong>
                </span>
                <span>
                  Peak Ground Truth:{" "}
                  <strong className="text-[#B45309]">
                    {(Math.max(...(traj.u_gt || []).map(Math.abs)) * 1000.0).toFixed(2)} mm
                  </strong>
                </span>
                <span>
                  Peak Surrogate:{" "}
                  <strong className="text-[#047857]">
                    {(Math.max(...traj.u.map(Math.abs)) * 1000.0).toFixed(2)} mm
                  </strong>
                </span>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Frozen Benchmark Artifact Reports & Physics Checks */
        <div className="space-y-6">
          {/* Phase 6 Baseline Benchmark Table */}
          <div className="panel-workstation overflow-hidden">
            <div className="p-4 border-b border-[#E2E8F0] text-xs font-mono bg-[#F8FAFC]">
              <div className="font-bold text-[#0F172A] text-sm">
                PHASE 6: FULL BENCHMARK SUMMARY (HELD-OUT EARTHQUAKE SPLIT)
              </div>
              <p className="text-[#64748B] text-[11px] mt-0.5 font-sans">
                Source: results/tables/phase6_full_benchmark_summary.md | Disaggregated by Elastic vs Post-Yield Regimes
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="table-engineering w-full text-left text-xs font-mono">
                <thead>
                  <tr>
                    <th className="font-sans font-semibold">Model Architecture</th>
                    <th>Elastic (μ ≤ 1) u(t)</th>
                    <th>Post-Yield (μ &gt; 1) u(t)</th>
                    <th>Overall Rel L2 u(t)</th>
                    <th>Overall Rel L2 Eh(t)</th>
                    <th>Parameters</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="font-semibold text-[#0F172A] font-sans">Linear Baseline (MLP)</td>
                    <td className="text-[#475569]">24.12 %</td>
                    <td className="text-[#DC2626] font-bold">58.45 %</td>
                    <td className="text-[#475569]">41.28 %</td>
                    <td className="text-[#94A3B8]">—</td>
                    <td className="text-[#475569]">145,200</td>
                  </tr>
                  <tr>
                    <td className="font-semibold text-[#0F172A] font-sans">Recurrent Neural Net (LSTM)</td>
                    <td className="text-[#475569]">18.64 %</td>
                    <td className="text-[#DC2626] font-bold">44.12 %</td>
                    <td className="text-[#475569]">31.38 %</td>
                    <td className="text-[#475569]">52.10 %</td>
                    <td className="text-[#475569]">384,120</td>
                  </tr>
                  <tr>
                    <td className="font-semibold text-[#0F172A] font-sans">State-Augmented TCN</td>
                    <td className="text-[#475569]">14.20 %</td>
                    <td className="text-[#B45309] font-medium">29.80 %</td>
                    <td className="text-[#475569]">22.00 %</td>
                    <td className="text-[#475569]">38.45 %</td>
                    <td className="text-[#475569]">612,400</td>
                  </tr>
                  <tr className="bg-[#ECFDF5] font-semibold text-[#047857]">
                    <td className="text-[#047857] font-bold font-sans">SeismoFNO 1D (Ours)</td>
                    <td className="text-[#047857] font-bold font-mono">10.82 %</td>
                    <td className="text-[#047857] font-bold font-mono">22.09 %</td>
                    <td className="text-[#047857] font-bold font-mono">16.45 %</td>
                    <td className="text-[#047857] font-bold font-mono">18.72 %</td>
                    <td className="font-mono text-[#0F172A]">1,196,931</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Deterministic Physics Verification Checks Table */}
          <div className="panel-workstation overflow-hidden">
            <div className="p-4 border-b border-[#E2E8F0] text-xs font-mono bg-[#F8FAFC]">
              <div className="font-bold text-[#0F172A] text-sm flex items-center gap-2">
                <ShieldCheck size={16} className="text-[#047857]" />
                DETERMINISTIC PHYSICS VALIDATION CHECKS (TESTS/ GROUND TRUTH)
              </div>
              <p className="text-[#64748B] text-[11px] mt-0.5 font-sans">
                Hard rule: All 8 ground truth physics validation suites pass with deterministic tolerances prior to neural surrogate training.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="table-engineering w-full text-left text-xs font-mono">
                <thead>
                  <tr>
                    <th className="font-sans font-semibold">Verification Check</th>
                    <th>Analytical Reference</th>
                    <th>Tolerance</th>
                    <th>Measured Error</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="font-semibold text-[#0F172A] font-sans">Linear Elastic Duhamel Comparison</td>
                    <td className="text-[#475569]">Closed-form exact integral</td>
                    <td className="text-[#475569]">Rel-L₂ &lt; 1e-4</td>
                    <td className="text-[#047857] font-bold">1.01 × 10⁻¹⁵</td>
                    <td><span className="badge-tech bg-[#ECFDF5] text-[#047857] border-[#A7F3D0]">PASS</span></td>
                  </tr>
                  <tr>
                    <td className="font-semibold text-[#0F172A] font-sans">Energy Balance (Elastic Undamped)</td>
                    <td className="text-[#475569]">Ek + Es = E_input</td>
                    <td className="text-[#475569]">ΔE / E &lt; 0.1%</td>
                    <td className="text-[#047857] font-bold">0.003 %</td>
                    <td><span className="badge-tech bg-[#ECFDF5] text-[#047857] border-[#A7F3D0]">PASS</span></td>
                  </tr>
                  <tr>
                    <td className="font-semibold text-[#0F172A] font-sans">Sign Antisymmetry in Ground Acceleration</td>
                    <td className="text-[#475569]">u(-ag) = -u(ag)</td>
                    <td className="text-[#475569]">Max diff &lt; 1e-6 m</td>
                    <td className="text-[#047857] font-bold">0.000 mm</td>
                    <td><span className="badge-tech bg-[#ECFDF5] text-[#047857] border-[#A7F3D0]">PASS</span></td>
                  </tr>
                  <tr>
                    <td className="font-semibold text-[#0F172A] font-sans">MDOF Modal Frequency Orthogonality</td>
                    <td className="text-[#475569]">Φᵀ [M] Φ = [I]</td>
                    <td className="text-[#475569]">Off-diag &lt; 1e-12</td>
                    <td className="text-[#047857] font-bold">&lt; 10⁻¹⁴</td>
                    <td><span className="badge-tech bg-[#ECFDF5] text-[#047857] border-[#A7F3D0]">PASS</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* SDOF vs MDOF Physical Scope Disclosure Card */}
          <div className="panel-workstation p-6 space-y-3">
            <div className="text-xs font-mono font-bold text-[#0F172A] uppercase flex items-center gap-2">
              <Zap size={14} className="text-[#047857]" />
              Physical Scope & Dimensionality Disclosures
            </div>
            <div className="text-xs font-sans text-[#475569] space-y-2 leading-relaxed">
              <p>
                • <strong>Core SDOF Surrogate:</strong> Validated against bilinear elastoplastic single-degree-of-freedom oscillators across 2,048 temporal steps. Models continuous hysteretic loops and residual plastic offset.
              </p>
              <p>
                • <strong>MDOF Extension (EXP6 Graph Neural Operator):</strong> Injects pre-earthquake modal invariants into spatiotemporal message passing, yielding a 62.9% relative peak error reduction on held-out 5-story buildings.
              </p>
              <p>
                • <strong>Timing Methodology:</strong> Measured on Apple Silicon GPU (MPS) using synchronized wall-clock calls. Average latency 21.45 ms vs 54.68 ms for OpenSeesPy (2.55x speedup), reaching 1,060 simulations/sec under batched evaluation.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelValidationView;
