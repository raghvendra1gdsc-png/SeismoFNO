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
    <div className="p-6 space-y-6 font-sans text-[#E8E8DE]">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border border-white/[0.08] bg-[#0E1B17] p-5 rounded-lg shadow-lg">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-bold font-mono tracking-tight text-[#E8E8DE]">
              MODEL EXPLAINABILITY & ARCHITECTURE
            </h1>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#73E6B5]/10 text-[#73E6B5] border border-[#73E6B5]/30 font-semibold">
              PHYSICS MAPPING
            </span>
          </div>
          <p className="text-xs text-[#82928B] mt-1 font-sans">
            Continuous operator pipeline architecture, spectral projection flow, and experimental hypothesis ledger.
          </p>
        </div>
      </div>

      {/* Uncertainty Interface (Alert Card) */}
      <div className="bg-[#D6B56D]/10 border border-[#D6B56D]/30 rounded-lg p-5 flex items-start space-x-4 shadow-md">
        <AlertCircle size={20} className="text-[#D6B56D] shrink-0 mt-0.5" />
        <div className="space-y-1 text-xs">
          <div className="font-bold text-[#D6B56D] uppercase tracking-wider font-mono">
            UNCERTAINTY ESTIMATION: RESEARCH MODULE PENDING
          </div>
          <p className="text-[#82928B] leading-relaxed">
            No synthetic Bayesian or heuristic confidence percentages are displayed. Formal epistemic and aleatoric uncertainty quantification (conformal prediction / deep ensembles) is currently an active research module. Decisions in post-yield regimes must rely on direct OpenSeesPy ground-truth verification.
          </p>
        </div>
      </div>

      {/* Pipeline Concept Flowchart */}
      <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-6 space-y-4 shadow-lg">
        <div className="text-xs font-mono font-bold text-[#E8E8DE] uppercase tracking-wider">
          CONTINUOUS NEURAL OPERATOR DYNAMICS PIPELINE
        </div>

        <div className="space-y-3 text-xs">
          {/* Box 1 */}
          <div className="bg-[#07110F] p-3.5 rounded-lg border border-white/[0.08] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <span className="text-[10px] text-[#73E6B5] font-bold uppercase font-mono">Input Layer</span>
              <div className="text-[#E8E8DE] font-bold">Earthquake Kinematics & Structural Encoding</div>
              <div className="text-[11px] text-[#82928B]">
                <span className="font-mono text-[#E8E8DE]">ag(t)</span> [m/s²], period <span className="font-mono text-[#E8E8DE]">T0</span>, natural frequency <span className="font-mono text-[#E8E8DE]">ωn</span>, initial stiffness <span className="font-mono text-[#E8E8DE]">k0</span>, damping <span className="font-mono text-[#E8E8DE]">ζ</span>, material law, yield disp <span className="font-mono text-[#E8E8DE]">uy</span>, alpha <span className="font-mono text-[#E8E8DE]">α</span>, target PGA, time grid
              </div>
            </div>
            <span className="px-2 py-1 rounded bg-[#0E1B17] text-[10px] text-[#73E6B5] border border-[#73E6B5]/30 font-mono font-semibold">10 Channels</span>
          </div>

          <div className="flex justify-center text-[#73E6B5]/60">
            <ArrowDown size={18} />
          </div>

          {/* Box 2 */}
          <div className="bg-[#07110F] p-3.5 rounded-lg border border-white/[0.08] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <span className="text-[10px] text-[#73E6B5] font-bold uppercase font-mono">Fourier Space Mapping</span>
              <div className="text-[#E8E8DE] font-bold">Continuous 1D Spectral Convolutions</div>
              <div className="text-[11px] text-[#82928B]">
                Lifting channel projection to width <span className="font-mono text-[#E8E8DE]">d=48</span>, followed by 4 Fourier layers: <span className="font-mono text-[#E8E8DE]">R · F(v)</span> in frequency domain using <span className="font-mono text-[#E8E8DE]">128</span> modes, skip connections & GeLU
              </div>
            </div>
            <span className="px-2 py-1 rounded bg-[#0E1B17] text-[10px] text-[#73E6B5] border border-[#73E6B5]/30 font-mono font-semibold">Resolution-Invariant</span>
          </div>

          <div className="flex justify-center text-[#73E6B5]/60">
            <ArrowDown size={18} />
          </div>

          {/* Box 3 */}
          <div className="bg-[#07110F] p-3.5 rounded-lg border border-white/[0.08] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <span className="text-[10px] text-[#D6B56D] font-bold uppercase font-mono">Nonlinear Dynamics State</span>
              <div className="text-[#E8E8DE] font-bold">Latent State Representation</div>
              <div className="text-[11px] text-[#82928B]">
                Captures dynamic oscillator response in continuous frequency space; path-dependent hysteretic transitions mapped to trajectory manifolds
              </div>
            </div>
            <span className="px-2 py-1 rounded bg-[#0E1B17] text-[10px] text-[#D6B56D] border border-[#D6B56D]/30 font-mono font-semibold">Latent Trajectory</span>
          </div>

          <div className="flex justify-center text-[#73E6B5]/60">
            <ArrowDown size={18} />
          </div>

          {/* Box 4 */}
          <div className="bg-[#07110F] p-3.5 rounded-lg border border-white/[0.08] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <span className="text-[10px] text-[#73E6B5] font-bold uppercase font-mono">Output Projection</span>
              <div className="text-[#E8E8DE] font-bold">Decoded Physical Quantities & Energy Indicators</div>
              <div className="text-[11px] text-[#82928B]">
                Relative displacement <span className="font-mono text-[#E8E8DE]">u(t)</span> [m], restoring force <span className="font-mono text-[#E8E8DE]">FR(t)</span> [N/kg], and cumulative dissipated hysteretic energy <span className="font-mono text-[#E8E8DE]">Eh(t)</span> [J/kg]
              </div>
            </div>
            <span className="px-2 py-1 rounded bg-[#0E1B17] text-[10px] text-[#73E6B5] border border-[#73E6B5]/30 font-mono font-semibold">3 Physical Trajectories</span>
          </div>
        </div>
      </div>

      {/* Experiment Status Dashboard */}
      <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-6 space-y-4 shadow-lg">
        <div className="text-xs font-mono font-bold text-[#E8E8DE] uppercase tracking-wider">
          SCIENTIFIC RESEARCH PHASES & HYPOTHESIS STATUS
        </div>

        <div className="space-y-3 text-xs">
          {/* EXP1 */}
          <div className="p-3.5 bg-[#07110F] rounded-lg border border-white/[0.08] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <CheckCircle2 size={15} className="text-[#73E6B5]" />
                <span className="font-bold text-[#E8E8DE] font-mono">EXP1 — Causal TCN Baseline</span>
              </div>
              <p className="text-[11px] text-[#82928B]">
                Temporal convolutional network with causal dilated convolutions for seismic time series.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-[#73E6B5]/10 text-[#73E6B5] border border-[#73E6B5]/30 font-mono font-semibold">
              COMPLETED
            </span>
          </div>

          {/* EXP2 */}
          <div className="p-3.5 bg-[#07110F] rounded-lg border border-white/[0.08] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <CheckCircle2 size={15} className="text-[#73E6B5]" />
                <span className="font-bold text-[#E8E8DE] font-mono">EXP2 — State-Space Memory (S4/SSM)</span>
              </div>
              <p className="text-[11px] text-[#82928B]">
                Continuous state-space models for long-range seismic memory retention.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-[#73E6B5]/10 text-[#73E6B5] border border-[#73E6B5]/30 font-mono font-semibold">
              COMPLETED
            </span>
          </div>

          {/* EXP3 */}
          <div className="p-3.5 bg-[#07110F] rounded-lg border border-white/[0.08] flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <XCircle size={15} className="text-[#E35D5D]" />
                <span className="font-bold text-[#E8E8DE] font-mono">EXP3 — Physics-Guided State Causal Hypothesis</span>
              </div>
              <p className="text-[11px] text-[#82928B]">
                Hypothesis that latent causal state vectors alone eliminate post-yield error was rigorously tested and rejected.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-[#E35D5D]/15 text-[#E35D5D] border border-[#E35D5D]/30 font-mono font-semibold">
              HYPOTHESIS REJECTED / AUDITED
            </span>
          </div>

          {/* EXP3-R */}
          <div className="p-3.5 bg-[#07110F] rounded-lg border border-[#D6B56D]/30 flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <Clock size={15} className="text-[#D6B56D]" />
                <span className="font-bold text-[#E8E8DE] font-mono">EXP3-R — Recurrent State Operator / Structured SSM</span>
              </div>
              <p className="text-[11px] text-[#82928B]">
                Research hypothesis under experimental validation. Investigating path-dependent memory mechanisms.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-[#D6B56D]/15 text-[#D6B56D] border border-[#D6B56D]/30 font-mono font-semibold">
              EXPERIMENTAL VALIDATION
            </span>
          </div>

          {/* EXP4 */}
          <div className="p-3.5 bg-[#07110F] rounded-lg border border-white/[0.08] opacity-70 flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <Lock size={15} className="text-[#82928B]" />
                <span className="font-bold text-[#82928B] font-mono">EXP4 — Multi-Degree-of-Freedom (MDOF) Nonlinear Extension</span>
              </div>
              <p className="text-[11px] text-[#82928B]">
                Locked per AGENTS.md protocol until complete closure of EXP3-R.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-white/10 text-[#82928B] border border-white/10 font-mono font-semibold">
              LOCKED
            </span>
          </div>
        </div>
      </div>

      {/* Honest Scientific Limitations Section */}
      <div className="bg-[#0E1B17] border border-white/[0.08] rounded-lg p-6 space-y-3 text-xs shadow-lg">
        <div className="text-[#E8E8DE] font-bold uppercase tracking-wider flex items-center space-x-2 font-mono">
          <Info size={16} className="text-[#73E6B5]" />
          <span>Honest Scientific Disclosure: Error vs Ductility Demand</span>
        </div>
        <p className="text-[#82928B] leading-relaxed">
          In linear elastic regimes (ductility <span className="font-mono text-[#E8E8DE]">μ ≤ 1.0</span>), SeismoFNO achieves high accuracy with mean relative L2 error of <strong className="font-mono text-[#73E6B5]">10.81%</strong> and restoring force error of <strong className="font-mono text-[#73E6B5]">6.75%</strong>.
          However, in high-ductility hysteretic regimes (<span className="font-mono text-[#E8E8DE]">μ &gt; 4.0</span>), path-dependent plastic yielding introduces phase lag accumulation, resulting in ~<strong className="font-mono text-[#E35D5D]">55–60%</strong> L2 error.
          Civil engineering workflows should adopt a tiered screening methodology: use ultra-fast SeismoFNO inference for broad parameter exploration and prioritize high-risk structures for targeted OpenSeesPy non-linear time history analysis.
        </p>
      </div>
    </div>
  );
};

export default ExplainabilityView;
