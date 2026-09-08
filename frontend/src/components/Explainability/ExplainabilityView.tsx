import React from "react";
import {
  ArrowDown,
  CheckCircle2,
  XCircle,
  Clock,
  Lock,
  AlertCircle,
  Info,
} from "lucide-react";

export const ExplainabilityView: React.FC = () => {
  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto font-sans">
      {/* Header */}
      <div className="border-b border-[#E0E0E0] pb-4">
        <div className="flex items-center space-x-3">
          <h1 className="text-xl font-bold font-mono tracking-tight text-[#161616]">
            SCIENTIFIC FOUNDATION & EXPLAINABILITY
          </h1>
          <span className="px-2.5 py-0.5 rounded text-[10px] font-mono bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] font-semibold">
            TRANSPARENT SCIENTIFIC REPORTING
          </span>
        </div>
        <p className="text-xs text-[#525252] mt-1 font-sans">
          Mathematical architecture, continuous operator mapping, research experiment status, and empirical limitations.
        </p>
      </div>

      {/* Uncertainty Interface (Carbon Notification Card) */}
      <div className="bg-[#FFF8E1] border border-[#B28600] rounded p-5 flex items-start space-x-4">
        <AlertCircle size={20} className="text-[#B28600] shrink-0 mt-0.5" />
        <div className="space-y-1 text-xs">
          <div className="font-bold text-[#B28600] uppercase tracking-wider font-mono">
            UNCERTAINTY ESTIMATION: RESEARCH MODULE PENDING
          </div>
          <p className="text-[#525252] leading-relaxed">
            No synthetic Bayesian or heuristic confidence percentages are displayed. Formal epistemic and aleatoric uncertainty quantification (conformal prediction / deep ensembles) is currently an active research module. Decisions in post-yield regimes must rely on direct OpenSeesPy ground-truth verification.
          </p>
        </div>
      </div>

      {/* Pipeline Concept Flowchart */}
      <div className="bg-white border border-[#E0E0E0] rounded p-6 space-y-4">
        <div className="text-xs font-mono font-bold text-[#161616] uppercase tracking-wider">
          CONTINUOUS NEURAL OPERATOR DYNAMICS PIPELINE
        </div>

        <div className="space-y-3 text-xs">
          {/* Box 1 */}
          <div className="bg-[#F4F4F4] p-3.5 rounded border border-[#E0E0E0] flex items-center justify-between">
            <div className="space-y-0.5">
              <span className="text-[10px] text-[#0F62FE] font-bold uppercase font-mono">Input Layer</span>
              <div className="text-[#161616] font-bold">Earthquake Kinematics & Structural Encoding</div>
              <div className="text-[11px] text-[#525252]">
                <span className="font-mono">ag(t)</span> [m/s²], period <span className="font-mono">T0</span>, natural frequency <span className="font-mono">ωn</span>, initial stiffness <span className="font-mono">k0</span>, damping <span className="font-mono">ζ</span>, material law, yield disp <span className="font-mono">uy</span>, alpha <span className="font-mono">α</span>, target PGA, time grid
              </div>
            </div>
            <span className="px-2 py-1 rounded bg-white text-[10px] text-[#161616] border border-[#E0E0E0] font-mono font-semibold">10 Channels</span>
          </div>

          <div className="flex justify-center text-[#8D8D8D]">
            <ArrowDown size={18} />
          </div>

          {/* Box 2 */}
          <div className="bg-[#F4F4F4] p-3.5 rounded border border-[#E0E0E0] flex items-center justify-between">
            <div className="space-y-0.5">
              <span className="text-[10px] text-[#0F62FE] font-bold uppercase font-mono">Fourier Space Mapping</span>
              <div className="text-[#161616] font-bold">Continuous 1D Spectral Convolutions</div>
              <div className="text-[11px] text-[#525252]">
                Lifting channel projection to width <span className="font-mono">d=48</span>, followed by 4 Fourier layers: <span className="font-mono">R · F(v)</span> in frequency domain using <span className="font-mono">128</span> modes, skip connections & GeLU
              </div>
            </div>
            <span className="px-2 py-1 rounded bg-white text-[10px] text-[#161616] border border-[#E0E0E0] font-mono font-semibold">Resolution-Invariant</span>
          </div>

          <div className="flex justify-center text-[#8D8D8D]">
            <ArrowDown size={18} />
          </div>

          {/* Box 3 */}
          <div className="bg-[#F4F4F4] p-3.5 rounded border border-[#E0E0E0] flex items-center justify-between">
            <div className="space-y-0.5">
              <span className="text-[10px] text-[#B28600] font-bold uppercase font-mono">Nonlinear Dynamics State</span>
              <div className="text-[#161616] font-bold">Latent State Representation</div>
              <div className="text-[11px] text-[#525252]">
                Captures dynamic oscillator response in continuous frequency space; path-dependent hysteretic transitions mapped to trajectory manifolds
              </div>
            </div>
            <span className="px-2 py-1 rounded bg-white text-[10px] text-[#161616] border border-[#E0E0E0] font-mono font-semibold">Latent Trajectory</span>
          </div>

          <div className="flex justify-center text-[#8D8D8D]">
            <ArrowDown size={18} />
          </div>

          {/* Box 4 */}
          <div className="bg-[#F4F4F4] p-3.5 rounded border border-[#E0E0E0] flex items-center justify-between">
            <div className="space-y-0.5">
              <span className="text-[10px] text-[#198038] font-bold uppercase font-mono">Output Projection</span>
              <div className="text-[#161616] font-bold">Decoded Physical Quantities & Energy Indicators</div>
              <div className="text-[11px] text-[#525252]">
                Relative displacement <span className="font-mono">u(t)</span> [m], restoring force <span className="font-mono">FR(t)</span> [N/kg], and cumulative dissipated hysteretic energy <span className="font-mono">Eh(t)</span> [J/kg]
              </div>
            </div>
            <span className="px-2 py-1 rounded bg-white text-[10px] text-[#161616] border border-[#E0E0E0] font-mono font-semibold">3 Physical Trajectories</span>
          </div>
        </div>
      </div>

      {/* Experiment Status Dashboard */}
      <div className="bg-white border border-[#E0E0E0] rounded p-6 space-y-4">
        <div className="text-xs font-mono font-bold text-[#161616] uppercase tracking-wider">
          SCIENTIFIC RESEARCH PHASES & HYPOTHESIS STATUS
        </div>

        <div className="space-y-3 text-xs">
          {/* EXP1 */}
          <div className="p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <CheckCircle2 size={15} className="text-[#198038]" />
                <span className="font-bold text-[#161616] font-mono">EXP1 — Causal TCN Baseline</span>
              </div>
              <p className="text-[11px] text-[#525252]">
                Temporal convolutional network with causal dilated convolutions for seismic time series.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-[#DEFBE6] text-[#198038] border border-[#6FDC8C] font-mono font-semibold">
              COMPLETED
            </span>
          </div>

          {/* EXP2 */}
          <div className="p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <CheckCircle2 size={15} className="text-[#198038]" />
                <span className="font-bold text-[#161616] font-mono">EXP2 — State-Space Memory (S4/SSM)</span>
              </div>
              <p className="text-[11px] text-[#525252]">
                Continuous state-space models for long-range seismic memory retention.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-[#DEFBE6] text-[#198038] border border-[#6FDC8C] font-mono font-semibold">
              COMPLETED
            </span>
          </div>

          {/* EXP3 */}
          <div className="p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <XCircle size={15} className="text-[#DA1E28]" />
                <span className="font-bold text-[#161616] font-mono">EXP3 — Physics-Guided State Causal Hypothesis</span>
              </div>
              <p className="text-[11px] text-[#525252]">
                Hypothesis that latent causal state vectors alone eliminate post-yield error was rigorously tested and rejected.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-[#FFF1F1] text-[#DA1E28] border border-[#FFB3B8] font-mono font-semibold">
              HYPOTHESIS REJECTED / AUDITED
            </span>
          </div>

          {/* EXP3-R */}
          <div className="p-3 bg-[#F4F4F4] rounded border border-[#A6C8FF] flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <Clock size={15} className="text-[#0F62FE]" />
                <span className="font-bold text-[#161616] font-mono">EXP3-R — Recurrent State Operator / Structured SSM</span>
              </div>
              <p className="text-[11px] text-[#525252]">
                Research hypothesis under experimental validation. Investigating path-dependent memory mechanisms.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-[#EDF5FF] text-[#0F62FE] border border-[#A6C8FF] font-mono font-semibold">
              EXPERIMENTAL VALIDATION
            </span>
          </div>

          {/* EXP4 */}
          <div className="p-3 bg-[#F4F4F4] rounded border border-[#E0E0E0] opacity-70 flex items-center justify-between">
            <div className="space-y-0.5">
              <div className="flex items-center space-x-2">
                <Lock size={15} className="text-[#8D8D8D]" />
                <span className="font-bold text-[#525252] font-mono">EXP4 — Multi-Degree-of-Freedom (MDOF) Nonlinear Extension</span>
              </div>
              <p className="text-[11px] text-[#8D8D8D]">
                Locked per AGENTS.md protocol until complete closure of EXP3-R.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] bg-white text-[#525252] border border-[#E0E0E0] font-mono font-semibold">
              LOCKED
            </span>
          </div>
        </div>
      </div>

      {/* Honest Scientific Limitations Section */}
      <div className="bg-white border border-[#E0E0E0] rounded p-6 space-y-3 text-xs">
        <div className="text-[#161616] font-bold uppercase tracking-wider flex items-center space-x-2 font-mono">
          <Info size={16} className="text-[#0F62FE]" />
          <span>Honest Scientific Disclosure: Error vs Ductility Demand</span>
        </div>
        <p className="text-[#525252] leading-relaxed">
          In linear elastic regimes (ductility <span className="font-mono">μ ≤ 1.0</span>), SeismoFNO achieves high accuracy with mean relative L2 error of <strong className="font-mono text-[#161616]">10.81%</strong> and restoring force error of <strong className="font-mono text-[#161616]">6.75%</strong>.
          However, in high-ductility hysteretic regimes (<span className="font-mono">μ &gt; 4.0</span>), path-dependent plastic yielding introduces phase lag accumulation, resulting in ~<strong className="font-mono text-[#161616]">55–60%</strong> L2 error.
          Civil engineering workflows should adopt a tiered screening methodology: use ultra-fast SeismoFNO inference for broad parameter exploration and prioritize high-risk structures for targeted OpenSeesPy non-linear time history analysis.
        </p>
      </div>
    </div>
  );
};

export default ExplainabilityView;
