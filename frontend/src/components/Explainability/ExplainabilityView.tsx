import React from "react";
import {
  AlertCircle,
  ArrowDown,
  CheckCircle2,
  XCircle,
  Clock,
  Lock,
  Info,
} from "lucide-react";

export const ExplainabilityView: React.FC = () => {
  return (
    <div className="p-4 md:p-8 space-y-6 font-sans text-[#0F172A] max-w-7xl mx-auto">
      {/* Header */}
      <div className="panel-workstation p-6 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#0F172A]">
              MODEL EXPLAINABILITY & ARCHITECTURE
            </h1>
            <span className="badge-tech bg-[#ECFDF5] border border-[#A7F3D0] text-[#047857]">
              PHYSICS MAPPING
            </span>
          </div>
          <p className="text-xs text-[#475569] mt-1 font-sans">
            Continuous operator pipeline architecture, spectral projection flow, and experimental hypothesis ledger.
          </p>
        </div>
      </div>

      {/* Uncertainty Interface (Alert Card) */}
      <div className="bg-[#FFFBEB] border border-[#FDE68A] rounded-lg p-5 flex items-start space-x-4 shadow-xs">
        <AlertCircle size={20} className="text-[#B45309] shrink-0 mt-0.5" />
        <div className="space-y-1 text-xs">
          <div className="font-bold text-[#B45309] uppercase tracking-wider font-mono">
            UNCERTAINTY ESTIMATION: RESEARCH MODULE PENDING
          </div>
          <p className="text-[#475569] leading-relaxed font-sans">
            No synthetic Bayesian or heuristic confidence percentages are displayed. Formal epistemic and aleatoric uncertainty quantification (conformal prediction / deep ensembles) is currently an active research module. Decisions in post-yield regimes must rely on direct OpenSeesPy ground-truth verification.
          </p>
        </div>
      </div>

      {/* Pipeline Concept Flowchart */}
      <div className="panel-workstation p-6 space-y-4">
        <div className="text-xs font-mono font-bold text-[#0F172A] uppercase tracking-wider">
          CONTINUOUS NEURAL OPERATOR DYNAMICS PIPELINE
        </div>

        <div className="space-y-3 text-xs">
          {/* Box 1 */}
          <div className="bg-[#F8FAFC] p-4 rounded-lg border border-[#E2E8F0] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-1">
              <span className="text-[10px] text-[#047857] font-bold uppercase font-mono">Input Layer</span>
              <div className="text-[#0F172A] font-bold text-sm">Earthquake Kinematics & Structural Encoding</div>
              <div className="text-[11px] text-[#475569] leading-relaxed">
                <span className="font-mono text-[#0F172A] font-semibold">ag(t)</span> [m/s²], period <span className="font-mono text-[#0F172A] font-semibold">T0</span>, natural frequency <span className="font-mono text-[#0F172A] font-semibold">ωn</span>, initial stiffness <span className="font-mono text-[#0F172A] font-semibold">k0</span>, damping <span className="font-mono text-[#0F172A] font-semibold">ζ</span>, material law, yield disp <span className="font-mono text-[#0F172A] font-semibold">uy</span>, alpha <span className="font-mono text-[#0F172A] font-semibold">α</span>, target PGA, time grid
              </div>
            </div>
            <span className="badge-tech bg-white border border-[#CBD5E1] text-[#047857] font-bold">10 Channels</span>
          </div>

          <div className="flex justify-center text-[#94A3B8]">
            <ArrowDown size={18} />
          </div>

          {/* Box 2 */}
          <div className="bg-[#F8FAFC] p-4 rounded-lg border border-[#E2E8F0] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-1">
              <span className="text-[10px] text-[#047857] font-bold uppercase font-mono">Fourier Space Mapping</span>
              <div className="text-[#0F172A] font-bold text-sm">Continuous 1D Spectral Convolutions</div>
              <div className="text-[11px] text-[#475569] leading-relaxed">
                Lifting channel projection to width <span className="font-mono text-[#0F172A] font-semibold">d=48</span>, followed by 4 Fourier layers: <span className="font-mono text-[#0F172A] font-semibold">R · F(v)</span> in frequency domain using <span className="font-mono text-[#0F172A] font-semibold">128</span> modes, skip connections & GeLU
              </div>
            </div>
            <span className="badge-tech bg-white border border-[#CBD5E1] text-[#047857] font-bold">Resolution-Invariant</span>
          </div>

          <div className="flex justify-center text-[#94A3B8]">
            <ArrowDown size={18} />
          </div>

          {/* Box 3 */}
          <div className="bg-[#F8FAFC] p-4 rounded-lg border border-[#E2E8F0] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-1">
              <span className="text-[10px] text-[#B45309] font-bold uppercase font-mono">Nonlinear Dynamics State</span>
              <div className="text-[#0F172A] font-bold text-sm">Latent State Representation</div>
              <div className="text-[11px] text-[#475569] leading-relaxed">
                Captures dynamic oscillator response in continuous frequency space; path-dependent hysteretic transitions mapped to trajectory manifolds
              </div>
            </div>
            <span className="badge-tech bg-white border border-[#CBD5E1] text-[#B45309] font-bold">Latent Trajectory</span>
          </div>

          <div className="flex justify-center text-[#94A3B8]">
            <ArrowDown size={18} />
          </div>

          {/* Box 4 */}
          <div className="bg-[#F8FAFC] p-4 rounded-lg border border-[#E2E8F0] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-1">
              <span className="text-[10px] text-[#047857] font-bold uppercase font-mono">Output Projection</span>
              <div className="text-[#0F172A] font-bold text-sm">Decoded Physical Quantities & Energy Indicators</div>
              <div className="text-[11px] text-[#475569] leading-relaxed">
                Relative displacement <span className="font-mono text-[#0F172A] font-semibold">u(t)</span> [m], restoring force <span className="font-mono text-[#0F172A] font-semibold">FR(t)</span> [N/kg], and cumulative dissipated hysteretic energy <span className="font-mono text-[#0F172A] font-semibold">Eh(t)</span> [J/kg]
              </div>
            </div>
            <span className="badge-tech bg-white border border-[#CBD5E1] text-[#047857] font-bold">3 Physical Trajectories</span>
          </div>
        </div>
      </div>

      {/* Experiment Status Dashboard */}
      <div className="panel-workstation p-6 space-y-4">
        <div className="text-xs font-mono font-bold text-[#0F172A] uppercase tracking-wider">
          SCIENTIFIC RESEARCH PHASES & HYPOTHESIS STATUS
        </div>

        <div className="space-y-3 text-xs">
          {/* EXP1 */}
          <div className="p-3.5 bg-[#F8FAFC] rounded-lg border border-[#E2E8F0] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <CheckCircle2 size={15} className="text-[#047857]" />
                <span className="font-bold text-[#0F172A] font-mono">EXP1 — Causal TCN Baseline</span>
              </div>
              <p className="text-[11px] text-[#475569]">
                Temporal convolutional network with causal dilated convolutions for seismic time series.
              </p>
            </div>
            <span className="badge-tech bg-[#ECFDF5] text-[#047857] border-[#A7F3D0]">
              COMPLETED
            </span>
          </div>

          {/* EXP2 */}
          <div className="p-3.5 bg-[#F8FAFC] rounded-lg border border-[#E2E8F0] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <CheckCircle2 size={15} className="text-[#047857]" />
                <span className="font-bold text-[#0F172A] font-mono">EXP2 — State-Space Memory (S4/SSM)</span>
              </div>
              <p className="text-[11px] text-[#475569]">
                Continuous state-space models for long-range seismic memory retention.
              </p>
            </div>
            <span className="badge-tech bg-[#ECFDF5] text-[#047857] border-[#A7F3D0]">
              COMPLETED
            </span>
          </div>

          {/* EXP3 */}
          <div className="p-3.5 bg-[#F8FAFC] rounded-lg border border-[#E2E8F0] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <XCircle size={15} className="text-[#DC2626]" />
                <span className="font-bold text-[#0F172A] font-mono">EXP3 — Physics-Guided State Causal Hypothesis</span>
              </div>
              <p className="text-[11px] text-[#475569]">
                Hypothesis that latent causal state vectors alone eliminate post-yield error was rigorously tested and rejected.
              </p>
            </div>
            <span className="badge-tech bg-[#FEF2F2] text-[#DC2626] border-[#FECACA]">
              HYPOTHESIS REJECTED / AUDITED
            </span>
          </div>

          {/* EXP3-R */}
          <div className="p-3.5 bg-[#F8FAFC] rounded-lg border border-[#FDE68A] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <Clock size={15} className="text-[#B45309]" />
                <span className="font-bold text-[#0F172A] font-mono">EXP3-R — Recurrent State Operator / Structured SSM</span>
              </div>
              <p className="text-[11px] text-[#475569]">
                Research hypothesis under experimental validation. Investigating path-dependent memory mechanisms.
              </p>
            </div>
            <span className="badge-tech bg-[#FFFBEB] text-[#B45309] border-[#FDE68A]">
              EXPERIMENTAL VALIDATION
            </span>
          </div>

          {/* EXP4 */}
          <div className="p-3.5 bg-[#F8FAFC] rounded-lg border border-[#E2E8F0] opacity-80 flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <Lock size={15} className="text-[#64748B]" />
                <span className="font-bold text-[#64748B] font-mono">EXP4 — Multi-Degree-of-Freedom (MDOF) Extension</span>
              </div>
              <p className="text-[11px] text-[#64748B]">
                Investigated in Research Layer (EXP4 zero-padding boundary collapse vs EXP6 Modal-GNO).
              </p>
            </div>
            <span className="badge-tech bg-[#F1F5F9] text-[#64748B] border-[#CBD5E1]">
              INTEGRATED (EXP6)
            </span>
          </div>
        </div>
      </div>

      {/* Honest Scientific Limitations Section */}
      <div className="panel-workstation p-6 space-y-3 text-xs">
        <div className="text-[#0F172A] font-bold uppercase tracking-wider flex items-center space-x-2 font-mono">
          <Info size={16} className="text-[#047857]" />
          <span>Honest Scientific Disclosure: Error vs Ductility Demand</span>
        </div>
        <p className="text-[#475569] leading-relaxed font-sans">
          In linear elastic regimes (ductility <span className="font-mono text-[#0F172A] font-semibold">μ ≤ 1.0</span>), SeismoFNO achieves high accuracy with mean relative L2 error of <strong className="font-mono text-[#047857]">10.81%</strong> and restoring force error of <strong className="font-mono text-[#047857]">6.75%</strong>.
          However, in high-ductility hysteretic regimes (<span className="font-mono text-[#0F172A] font-semibold">μ &gt; 4.0</span>), path-dependent plastic yielding introduces phase lag accumulation, resulting in ~<strong className="font-mono text-[#DC2626]">55–60%</strong> L2 error.
          Civil engineering workflows should adopt a tiered screening methodology: use ultra-fast SeismoFNO inference for broad parameter exploration and prioritize high-risk structures for targeted OpenSeesPy non-linear time history analysis.
        </p>
      </div>
    </div>
  );
};

export default ExplainabilityView;
