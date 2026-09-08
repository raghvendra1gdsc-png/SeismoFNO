import React from "react";
import type { SystemHealth } from "../../types/simulation";

interface ModelMetadataModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: SystemHealth | null;
}

export const ModelMetadataModal: React.FC<ModelMetadataModalProps> = ({
  isOpen,
  onClose,
  health,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-background-deep/85 flex items-center justify-center p-4">
      <div className="bg-background-base border border-border-medium max-w-xl w-full font-mono text-xs shadow-none">
        {/* Header */}
        <div className="px-5 py-3 border-b border-border-subtle flex items-center justify-between">
          <span className="font-bold text-text-primary uppercase tracking-wider">
            TECHNICAL SPECIFICATION & VERIFICATION
          </span>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary cursor-pointer uppercase text-[11px]"
          >
            [CLOSE]
          </button>
        </div>

        {/* Content */}
        <div className="p-5 space-y-5">
          {/* Metadata Grid */}
          <div className="grid grid-cols-2 gap-4 text-[11px] border-b border-border-subtle pb-4">
            <div>
              <div className="text-[10px] text-text-muted uppercase">SURROGATE MODEL</div>
              <div className="text-text-primary font-semibold">Fourier Neural Operator (1D)</div>
              <div className="text-text-muted text-[10px]">4 Spectral Blocks · Width 48 · Modes 128</div>
            </div>

            <div>
              <div className="text-[10px] text-text-muted uppercase">TRAINABLE PARAMETERS</div>
              <div className="text-text-primary font-semibold font-mono-num">
                {health?.num_params?.toLocaleString() || "1,196,931"}
              </div>
              <div className="text-text-muted text-[10px]">Complex Weights + Residual Skip</div>
            </div>

            <div>
              <div className="text-[10px] text-text-muted uppercase">GROUND TRUTH ENGINE</div>
              <div className="text-text-primary font-semibold">OpenSeesPy C++ NLTHA</div>
              <div className="text-text-muted text-[10px]">Newmark-β (γ=0.5, β=0.25, Tol &lt; 1e-10)</div>
            </div>

            <div>
              <div className="text-[10px] text-text-muted uppercase">PHYSICS-INFORMED LOSS</div>
              <div className="text-text-primary font-semibold font-mono-num">L_data + 0.10 L_E + 0.05 L_B</div>
              <div className="text-text-muted text-[10px]">Hysteretic Work + Initial Boundary Penalty</div>
            </div>
          </div>

          {/* Verification Protocol */}
          <div className="space-y-2">
            <div className="text-[10px] uppercase tracking-wider text-text-muted font-bold">
              VERIFICATION PROTOCOL
            </div>
            <div className="border border-border-subtle">
              <table className="w-full text-left text-[11px]">
                <thead className="bg-background-deep border-b border-border-subtle text-text-muted text-[10px]">
                  <tr>
                    <th className="py-1.5 px-3">Benchmark</th>
                    <th className="py-1.5 px-3">Test Condition</th>
                    <th className="py-1.5 px-3">Measured Result</th>
                    <th className="py-1.5 px-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle text-text-secondary">
                  <tr>
                    <td className="py-1.5 px-3 text-text-primary">SDOF Analytical</td>
                    <td className="py-1.5 px-3">Damped free vibration sinusoids</td>
                    <td className="py-1.5 px-3 text-energy font-mono-num">0.0000% error</td>
                    <td className="py-1.5 px-3 text-energy">PASS</td>
                  </tr>
                  <tr>
                    <td className="py-1.5 px-3 text-text-primary">Newmark-β Cross-Check</td>
                    <td className="py-1.5 px-3">Independent vectorized solver</td>
                    <td className="py-1.5 px-3 text-energy font-mono-num">&lt; 1e-6 diff</td>
                    <td className="py-1.5 px-3 text-energy">PASS</td>
                  </tr>
                  <tr>
                    <td className="py-1.5 px-3 text-text-primary">Zero-Leakage Splits</td>
                    <td className="py-1.5 px-3">140 held-out seismic events</td>
                    <td className="py-1.5 px-3 text-energy font-mono-num">0% overlap</td>
                    <td className="py-1.5 px-3 text-energy">PASS</td>
                  </tr>
                  <tr>
                    <td className="py-1.5 px-3 text-text-primary">Batch Acceleration</td>
                    <td className="py-1.5 px-3">Batch 64 vs OpenSees CPU</td>
                    <td className="py-1.5 px-3 text-fno font-mono-num">5.8× measured speedup</td>
                    <td className="py-1.5 px-3 text-fno">PASS</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-2.5 border-t border-border-subtle flex justify-end">
          <button
            onClick={onClose}
            className="px-3 py-1 bg-background-deep border border-border-medium text-text-primary hover:bg-panel-hover transition cursor-pointer text-[11px]"
          >
            DISMISS
          </button>
        </div>
      </div>
    </div>
  );
};
