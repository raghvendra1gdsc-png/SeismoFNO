import React from "react";
import type { SimulationMetrics } from "../../types/simulation";
import { formatNumber } from "../../utils/formatting";

interface ResultSummaryProps {
  metrics: SimulationMetrics | null;
  isLoading: boolean;
}

export const ResultSummary: React.FC<ResultSummaryProps> = ({
  metrics,
  isLoading,
}) => {
  if (isLoading) {
    return (
      <div className="border-b border-border-subtle bg-background-base px-6 py-3 font-mono text-xs text-text-muted">
        COMPUTING FOURIER SPECTRAL OPERATORS & OPEN_SEES_PY NLTHA...
      </div>
    );
  }

  if (!metrics) return null;

  const mu = metrics.ductility_mu;
  const isYielded = mu > 1.0;
  const speedup = metrics.speedup;

  return (
    <div className="border-b border-border-subtle bg-background-base grid grid-cols-2 md:grid-cols-4 divide-x divide-border-subtle font-mono select-none">
      {/* 1. SURROGATE SPEEDUP */}
      <div className="px-5 py-3 space-y-0.5">
        <div className="text-[10px] text-text-muted uppercase tracking-wider">
          SURROGATE SPEEDUP
        </div>
        <div className="text-xl font-bold text-text-primary tracking-tight font-mono-num">
          {formatNumber(speedup, 1)}×
        </div>
        <div className="text-[10px] text-text-secondary font-mono-num">
          {metrics.t_fno_ms.toFixed(2)} ms / {metrics.t_gt_ms.toFixed(2)} ms
        </div>
      </div>

      {/* 2. RELATIVE DISPLACEMENT ERROR */}
      <div className="px-5 py-3 space-y-0.5">
        <div className="text-[10px] text-text-muted uppercase tracking-wider">
          DISP. ERROR (u)
        </div>
        <div className="text-xl font-bold text-text-primary tracking-tight font-mono-num">
          {formatNumber(metrics.err_u_rel_l2, 1)}%
        </div>
        <div className="text-[10px] text-text-secondary">
          RELATIVE L₂ NORM
        </div>
      </div>

      {/* 3. DUCTILITY DEMAND */}
      <div className="px-5 py-3 space-y-0.5">
        <div className="text-[10px] text-text-muted uppercase tracking-wider">
          DUCTILITY DEMAND
        </div>
        <div className="text-xl font-bold text-text-primary tracking-tight font-mono-num">
          μ = {formatNumber(mu, 2)}
        </div>
        <div className={`text-[10px] uppercase font-semibold ${isYielded ? "text-opensees" : "text-energy"}`}>
          {isYielded ? (mu > 4.0 ? "SEVERE INELASTIC" : "INELASTIC YIELD") : "LINEAR ELASTIC"}
        </div>
      </div>

      {/* 4. DISSIPATED ENERGY WORK */}
      <div className="px-5 py-3 space-y-0.5">
        <div className="text-[10px] text-text-muted uppercase tracking-wider">
          HYSTERETIC ENERGY ERROR
        </div>
        <div className="text-xl font-bold text-text-primary tracking-tight font-mono-num">
          {formatNumber(metrics.err_eh_rel_l2, 1)}%
        </div>
        <div className="text-[10px] text-text-secondary">
          ∫ F<sub>R</sub> du WORK
        </div>
      </div>
    </div>
  );
};
