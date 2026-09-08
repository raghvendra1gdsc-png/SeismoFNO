import React from "react";
import type { SimulationResponse } from "../../types/simulation";
import { formatNumber } from "../../utils/formatting";

interface EngineeringInterpretationProps {
  data: SimulationResponse | null;
  uy: number;
}

export const EngineeringInterpretation: React.FC<EngineeringInterpretationProps> = ({
  data,
  uy,
}) => {
  if (!data) return null;

  const m = data.metrics;
  const mu = m.ductility_mu;
  const isYielded = mu > 1.0;
  const uMaxGT = m.u_max_gt_m;

  let regimeTitle = "LINEAR ELASTIC";
  let regimeNarrative = "";
  let agreementTitle = "HIGH PRECISION";
  let agreementNarrative = "";

  if (!isYielded) {
    regimeTitle = "LINEAR ELASTIC (μ ≤ 1.0)";
    regimeNarrative = `Peak displacement (${(uMaxGT * 1000).toFixed(2)} mm) remains below the yield threshold (${(uy * 1000).toFixed(2)} mm). No plastic yielding occurred.`;
    agreementTitle = m.err_u_rel_l2 < 20 ? "EXCELLENT AGREEMENT" : "SATISFACTORY";
    agreementNarrative = `The Fourier Neural Operator maps the linear-elastic transfer function with ${m.err_u_rel_l2}% relative L₂ displacement error.`;
  } else if (mu <= 2.0) {
    regimeTitle = "LOW INELASTIC (1.0 < μ ≤ 2.0)";
    regimeNarrative = `Peak displacement exceeds yield threshold by a ductility factor of μ = ${formatNumber(mu, 2)}. Minor plastic excursions dissipated ${formatNumber(data.eh_gt[data.eh_gt.length - 1], 1)} J of energy.`;
    agreementTitle = m.err_u_rel_l2 < 30 ? "STRONG AGREEMENT" : "MODERATE";
    agreementNarrative = `The surrogate tracks the primary nonlinear excursions with ${m.err_fr_rel_l2}% force error and ${m.err_umax_rel}% peak demand error.`;
  } else if (mu <= 4.0) {
    regimeTitle = "MODERATE INELASTIC (2.0 < μ ≤ 4.0)";
    regimeNarrative = `Significant plastic deformation across multiple load reversals (ductility μ = ${formatNumber(mu, 2)}). Stiffness degradation leads to effective period elongation.`;
    agreementTitle = "MODERATE AGREEMENT";
    agreementNarrative = `The surrogate accurately bounds peak demand (|Δu_max| = ${m.err_umax_rel}%) while accumulating minor phase shifts during cyclic decay.`;
  } else {
    regimeTitle = "SEVERE INELASTIC (μ > 4.0)";
    regimeNarrative = `Extensive nonlinear yielding and large plastic excursions dominate the response (ductility μ = ${formatNumber(mu, 2)}). Hysteretic dissipation reached ${formatNumber(data.eh_gt[data.eh_gt.length - 1], 1)} J.`;
    agreementTitle = "GLOBAL ENVELOPE CAPTURED";
    agreementNarrative = `Global Fourier modes accurately capture the hysteretic work (${m.err_eh_rel_l2}% energy error); path-dependent memory loss in the FNO causes baseline drift in the free-vibration coda.`;
  }

  return (
    <div className="border-t border-border-subtle bg-background-base font-mono text-xs select-none">
      <div className="px-6 py-2.5 border-b border-border-subtle text-text-primary uppercase tracking-wider font-semibold">
        ENGINEERING INTERPRETATION
      </div>

      <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 text-text-secondary leading-relaxed">
        <div className="space-y-1.5">
          <div className="text-[10px] text-text-muted uppercase tracking-wider">
            RESPONSE REGIME
          </div>
          <div className="text-text-primary font-bold">
            {regimeTitle}
          </div>
          <p className="text-[11px] text-text-secondary">
            {regimeNarrative}
          </p>
        </div>

        <div className="space-y-1.5">
          <div className="text-[10px] text-text-muted uppercase tracking-wider">
            SURROGATE AGREEMENT
          </div>
          <div className="text-text-primary font-bold">
            {agreementTitle}
          </div>
          <p className="text-[11px] text-text-secondary">
            {agreementNarrative}
          </p>
        </div>
      </div>
    </div>
  );
};
