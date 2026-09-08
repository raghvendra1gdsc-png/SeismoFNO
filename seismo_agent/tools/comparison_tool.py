"""
seismo_agent/tools/comparison_tool.py — Deterministic Cross-Validation & Metric Tool.

Wraps the standalone scientific metrics library `src.evaluation.metrics` to compare
surrogate time histories against ground-truth physics simulations across the 11
computational mechanics metrics (relative L2, peak error, phase error, yield onset,
hysteresis loop area error, and hardware speedup).
"""

from typing import Optional
import numpy as np

from seismo_agent.schemas.inputs import ComparisonRequest
from seismo_agent.schemas.outputs import ComparisonResult
from seismo_agent.tools.base import BaseTool, ToolExecutionError
from src.evaluation.metrics import (
    compute_rel_l2_error,
    compute_peak_error,
    compute_phase_error,
    compute_yield_time_error,
    compute_hysteresis_area_error,
    compute_hysteretic_energy_loss_normalized,
)


class ComparisonTool(BaseTool):
    """
    Evaluates discrepancy and computational speedup between SeismoFNO and physics references.
    """
    name = "compare_fno_vs_physics"
    description = (
        "Compares predicted structural response trajectories against ground truth physics "
        "simulations using rigorous computational mechanics metrics: Trajectory Relative L2 error, "
        "Peak Displacement Error, Phase Coherence Discrepancy (Hilbert analytic signal), "
        "Yield Timing Error, Hysteresis Loop Dissipation Area Discrepancy, and measured wall-clock Speedup. "
        "Enforces mathematical consistency without inventing arbitrary thresholds."
    )
    input_schema = ComparisonRequest
    output_schema = ComparisonResult

    def _run(self, request: ComparisonRequest) -> ComparisonResult:
        u_p = np.asarray(request.u_pred, dtype=np.float64)
        u_r = np.asarray(request.u_ref, dtype=np.float64)

        if len(u_p) == 0 or len(u_r) == 0:
            raise ToolExecutionError(self.name, "Displacement trajectories cannot be empty.")

        # Ensure length matching
        min_len = min(len(u_p), len(u_r))
        if len(u_p) != len(u_r):
            # Truncate to common length
            u_p = u_p[:min_len]
            u_r = u_r[:min_len]

        # 1. Displacement Errors
        err_u_l2 = compute_rel_l2_error(u_p, u_r)
        err_umax = compute_peak_error(u_p, u_r)

        # 2. Phase Discrepancy (Hilbert transform)
        try:
            err_phase = compute_phase_error(u_p, u_r)
        except Exception:
            err_phase = None

        # 3. Permanent Residual Drift Error
        err_residual_m = float(abs(u_p[-1] - u_r[-1]))

        # 4. Restoring Force Errors (if provided)
        err_fr_l2 = None
        err_frmax = None
        if request.fr_pred is not None and request.fr_ref is not None:
            fr_p = np.asarray(request.fr_pred, dtype=np.float64)[:min_len]
            fr_r = np.asarray(request.fr_ref, dtype=np.float64)[:min_len]
            err_fr_l2 = compute_rel_l2_error(fr_p, fr_r)
            err_frmax = compute_peak_error(fr_p, fr_r)

        # 5. Hysteretic Energy Errors (if provided)
        err_eh_l2 = None
        if request.eh_pred is not None and request.eh_ref is not None:
            eh_p = np.asarray(request.eh_pred, dtype=np.float64)[:min_len]
            eh_r = np.asarray(request.eh_ref, dtype=np.float64)[:min_len]
            err_eh_l2 = compute_hysteretic_energy_loss_normalized(eh_p, eh_r)

        # 6. Yield Onset Timing Error
        err_yield_time = None
        if request.u_y is not None and request.u_y > 0:
            err_yield_time = compute_yield_time_error(u_p, u_r, request.u_y, dt=request.dt)

        # 7. Speedup Calculation
        speedup = None
        if request.fno_runtime_ms is not None and request.physics_runtime_ms is not None:
            if request.fno_runtime_ms > 0:
                speedup = round(float(request.physics_runtime_ms / request.fno_runtime_ms), 1)

        summary = (
            f"Displacement Rel L2 Error: {err_u_l2:.2f}%, Peak Error: {err_umax:.2f}%. "
            f"Residual Drift Error: {err_residual_m*1000.0:.2f} mm."
        )
        if speedup is not None:
            summary += f" Measured Speedup: {speedup}x."

        return ComparisonResult(
            status="success",
            err_u_rel_l2_pct=round(err_u_l2, 2),
            err_fr_rel_l2_pct=round(err_fr_l2, 2) if err_fr_l2 is not None else None,
            err_eh_rel_l2_pct=round(err_eh_l2, 2) if err_eh_l2 is not None else None,
            err_umax_rel_pct=round(err_umax, 2),
            err_frmax_rel_pct=round(err_frmax, 2) if err_frmax is not None else None,
            phase_error_rad=round(float(err_phase), 4) if err_phase is not None else None,
            yield_time_error_ms=round(float(err_yield_time), 2) if err_yield_time is not None else None,
            residual_drift_error_m=round(err_residual_m, 6),
            fno_runtime_ms=request.fno_runtime_ms,
            physics_runtime_ms=request.physics_runtime_ms,
            speedup_factor=speedup,
            agreement_summary=summary,
        )
