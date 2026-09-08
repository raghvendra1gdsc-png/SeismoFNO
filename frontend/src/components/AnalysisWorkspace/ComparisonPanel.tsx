import React from "react";
import { GitCompare } from "lucide-react";
import type { SimulationMetrics } from "../../types/agent";

interface ComparisonPanelProps {
  metrics?: SimulationMetrics | null;
  isLoading?: boolean;
}

export const ComparisonPanel: React.FC<ComparisonPanelProps> = ({
  metrics,
  isLoading: _isLoading = false,
}) => {
  if (!metrics) {
    return (
      <div className="p-8 text-center text-text-muted font-mono text-xs">
        No comparison data available. Execute an analysis to cross-validate SeismoFNO against OpenSeesPy.
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4 font-mono text-xs select-none">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-3">
        <div className="flex items-center space-x-2">
          <GitCompare size={16} className="text-fno" />
          <span className="font-bold text-text-primary uppercase tracking-wider text-sm">
            SURROGATE FIDELITY & CROSS-VALIDATION MATRIX
          </span>
        </div>
        <div className="text-[11px] text-text-muted">
          ComparisonTool · Deterministic Error Assessment
        </div>
      </div>

      {/* Side-by-Side Model Contrast Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* SeismoFNO Card */}
        <div className="bg-panel-base border border-fno/30 rounded-md p-3.5 space-y-2.5 shadow-[0_0_12px_rgba(0,210,255,0.05)]">
          <div className="flex items-center justify-between border-b border-border-subtle pb-2">
            <span className="text-fno font-bold text-xs tracking-wider">SEISMOFNO SURROGATE</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-fno/10 text-fno font-semibold">PREDICTION</span>
          </div>
          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between">
              <span className="text-text-muted">Peak Displacement:</span>
              <span className="text-text-primary font-bold">
                {(metrics.u_max_pred_m * 1000).toFixed(2)} mm
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Compute Device:</span>
              <span className="text-fno uppercase font-semibold">{metrics.device}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Inference Wall-Clock:</span>
              <span className="text-energy font-bold">{metrics.t_fno_ms.toFixed(2)} ms</span>
            </div>
          </div>
        </div>

        {/* Physics Reference Card */}
        <div className="bg-panel-base border border-opensees/30 rounded-md p-3.5 space-y-2.5 shadow-[0_0_12px_rgba(245,158,11,0.05)]">
          <div className="flex items-center justify-between border-b border-border-subtle pb-2">
            <span className="text-opensees font-bold text-xs tracking-wider">OPENSEESPY NLTHA</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-opensees/10 text-opensees font-semibold">
              PHYSICS REFERENCE
            </span>
          </div>
          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between">
              <span className="text-text-muted">Peak Displacement:</span>
              <span className="text-text-primary font-bold">
                {(metrics.u_max_gt_m * 1000).toFixed(2)} mm
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Ductility Demand (μ):</span>
              <span className="text-opensees font-bold">{metrics.ductility_mu.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">Simulation Wall-Clock:</span>
              <span className="text-text-primary font-bold">{metrics.t_gt_ms.toFixed(2)} ms</span>
            </div>
          </div>
        </div>
      </div>

      {/* Discrepancy & Fidelity Metrics Table */}
      <div className="bg-panel-base border border-border-subtle rounded-md overflow-hidden">
        <table className="w-full text-left text-xs">
          <thead className="bg-background-base text-[10px] uppercase text-text-muted border-b border-border-subtle">
            <tr>
              <th className="px-3 py-2">Engineering Discrepancy Metric</th>
              <th className="px-3 py-2">Measured Value</th>
              <th className="px-3 py-2">Physical Unit</th>
              <th className="px-3 py-2">Status</th>
              <th className="px-3 py-2">Authoritative Source</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-subtle/50 text-text-secondary">
            <tr>
              <td className="px-3 py-2 font-semibold text-text-primary">Trajectory Relative L2 Error</td>
              <td className="px-3 py-2 text-fno font-bold">{metrics.err_u_rel_l2.toFixed(2)}%</td>
              <td className="px-3 py-2 text-text-muted">%</td>
              <td className="px-3 py-2">
                {metrics.err_u_rel_l2 < 10.0 ? (
                  <span className="text-energy font-semibold">EXCELLENT</span>
                ) : metrics.err_u_rel_l2 < 25.0 ? (
                  <span className="text-opensees font-semibold">ACCEPTABLE</span>
                ) : (
                  <span className="text-danger font-semibold">HIGH DISCREPANCY</span>
                )}
              </td>
              <td className="px-3 py-2 text-text-muted">ComparisonTool</td>
            </tr>

            <tr>
              <td className="px-3 py-2 font-semibold text-text-primary">Peak Displacement Error (|u_max|)</td>
              <td className="px-3 py-2 text-text-primary font-bold">{metrics.err_umax_rel.toFixed(2)}%</td>
              <td className="px-3 py-2 text-text-muted">%</td>
              <td className="px-3 py-2 text-energy">MEASURED</td>
              <td className="px-3 py-2 text-text-muted">ComparisonTool</td>
            </tr>

            <tr>
              <td className="px-3 py-2 font-semibold text-text-primary">Restoring Force L2 Error</td>
              <td className="px-3 py-2 text-text-primary font-bold">
                {metrics.err_fr_rel_l2 ? `${metrics.err_fr_rel_l2.toFixed(2)}%` : "N/A"}
              </td>
              <td className="px-3 py-2 text-text-muted">%</td>
              <td className="px-3 py-2 text-text-muted">MEASURED</td>
              <td className="px-3 py-2 text-text-muted">ComparisonTool</td>
            </tr>

            <tr>
              <td className="px-3 py-2 font-semibold text-text-primary">Dissipated Energy L2 Error</td>
              <td className="px-3 py-2 text-text-primary font-bold">
                {metrics.err_eh_rel_l2 ? `${metrics.err_eh_rel_l2.toFixed(2)}%` : "N/A"}
              </td>
              <td className="px-3 py-2 text-text-muted">%</td>
              <td className="px-3 py-2 text-text-muted">MEASURED</td>
              <td className="px-3 py-2 text-text-muted">ComparisonTool</td>
            </tr>

            <tr>
              <td className="px-3 py-2 font-semibold text-text-primary">Instantaneous Phase Error (Hilbert)</td>
              <td className="px-3 py-2 text-text-primary font-bold">
                {metrics.phase_error_rad != null ? `${metrics.phase_error_rad.toFixed(3)}` : "N/A"}
              </td>
              <td className="px-3 py-2 text-text-muted">rad</td>
              <td className="px-3 py-2 text-text-muted">MEASURED</td>
              <td className="px-3 py-2 text-text-muted">ComparisonTool</td>
            </tr>

            <tr>
              <td className="px-3 py-2 font-semibold text-text-primary">Measured Acceleration Speedup</td>
              <td className="px-3 py-2 text-energy font-bold text-sm">
                {metrics.speedup.toFixed(1)}x
              </td>
              <td className="px-3 py-2 text-text-muted">factor</td>
              <td className="px-3 py-2 text-energy font-bold">BENCHMARKED</td>
              <td className="px-3 py-2 text-text-muted">Wall-Clock Timer</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};
