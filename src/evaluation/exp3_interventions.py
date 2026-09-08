"""
exp3_interventions.py — Physical State Reconstruction and Counterfactual Intervention Utilities.

Constitutive Identities for 1D Bilinear Kinematic Hardening:
    u_p(t) = (u(t) - F_R(t) / k0) / (1 - alpha)
    alpha_b(t) = (alpha / (1 - alpha)) * (k0 * u(t) - F_R(t))
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch


def compute_exact_physical_state(
    u: Union[np.ndarray, torch.Tensor],
    f_r: Union[np.ndarray, torch.Tensor],
    k0: Union[float, np.ndarray, torch.Tensor],
    alpha: Union[float, np.ndarray, torch.Tensor],
    eps_alpha: float = 1e-4,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Compute exact closed-form plastic displacement u_p(t) and kinematic back-stress alpha_b(t)
    for a 1D bilinear kinematic hardening oscillator (Steel01 constitutive model).

    Args:
        u: Total displacement trajectory [..., Length] (meters)
        f_r: Restoring force trajectory [..., Length] (Newtons)
        k0: Initial elastic stiffness (N/m)
        alpha: Post-yield stiffness ratio k_post / k0 (dimensionless, e.g. 0.02)
        eps_alpha: Epsilon denominator floor to prevent division by zero if alpha=1.0

    Returns:
        u_p: Exact plastic displacement trajectory [..., Length] (meters)
        alpha_b: Exact kinematic back-stress trajectory [..., Length] (Newtons)
    """
    if isinstance(u, torch.Tensor):
        if not isinstance(k0, torch.Tensor):
            k0 = torch.tensor(k0, device=u.device, dtype=u.dtype)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, device=u.device, dtype=u.dtype)

        denom = torch.clamp(1.0 - alpha, min=eps_alpha)
        k0_safe = torch.clamp(k0, min=1e-4)

        u_p = (u - (f_r / k0_safe)) / denom
        alpha_b = (alpha / denom) * (k0 * u - f_r)
        return u_p, alpha_b
    else:
        u_np = np.asarray(u, dtype=np.float64)
        f_np = np.asarray(f_r, dtype=np.float64)
        k0_np = np.asarray(k0, dtype=np.float64)
        alpha_np = np.asarray(alpha, dtype=np.float64)

        denom = np.maximum(1.0 - alpha_np, eps_alpha)
        k0_safe = np.maximum(k0_np, 1e-4)

        u_p = (u_np - (f_np / k0_safe)) / denom
        alpha_b = (alpha_np / denom) * (k0_np * u_np - f_np)
        return u_p, alpha_b


def identify_yield_onset_time(
    u: Union[np.ndarray, torch.Tensor],
    u_y: float,
    default_step: Optional[int] = None,
) -> int:
    """
    Identify the earliest discrete time index where displacement reaches or exceeds yield displacement:
        t_y = min { t | |u(t)| >= u_y }

    Args:
        u: Displacement trajectory [Length]
        u_y: Yield displacement threshold [m]
        default_step: Fallback index if sequence never yields (default: mid-sequence L // 2)

    Returns:
        t_y: Discrete time step index
    """
    if default_step is None:
        default_step = len(u) // 2

    if isinstance(u, torch.Tensor):
        u_abs = torch.abs(u)
        yield_mask = u_abs >= u_y
        indices = torch.nonzero(yield_mask, as_tuple=False)
        if len(indices) > 0:
            return int(indices[0].item())
        return default_step
    else:
        u_abs = np.abs(u)
        yield_indices = np.where(u_abs >= u_y)[0]
        if len(yield_indices) > 0:
            return int(yield_indices[0])
        return default_step


def run_counterfactual_dose_sweep(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_y: int,
    k0: float,
    alpha: float,
    doses_mm: List[float] = [-10.0, -5.0, 0.0, 5.0, 10.0],
) -> Dict[str, Any]:
    """
    Execute a causal counterfactual dose sweep across predefined perturbation magnitudes.

    Args:
        model: PG-TCN model implementing forward(x, intervention=...)
        x: Input tensor [1, in_channels, Length]
        t_y: Yield onset time index
        k0: Initial elastic stiffness
        alpha: Post-yield stiffness ratio
        doses_mm: List of plastic displacement intervention doses in millimeters

    Returns:
        results: Dictionary containing trajectory offsets, residual drift changes, and invariance telemetry
    """
    model.eval()
    results = {
        "doses_mm": doses_mm,
        "delta_u_residual_mm": [],
        "max_past_error_mm": [],
        "trajectories_cf": [],
        "baseline_u": None,
    }

    with torch.no_grad():
        # Baseline run (zero intervention)
        y_base, s_base = model(x, intervention=None, return_state=True)
        u_base = y_base[0, 0, :].cpu().numpy()
        results["baseline_u"] = u_base

        for dose_mm in doses_mm:
            delta_u_p_m = dose_mm / 1000.0  # Convert mm to m
            delta_alpha_b = alpha * k0 * delta_u_p_m

            intervention = {
                "t_y": t_y,
                "delta_u_p": delta_u_p_m,
                "delta_alpha_b": delta_alpha_b,
            }

            y_cf, s_cf = model(x, intervention=intervention, return_state=True)
            u_cf = y_cf[0, 0, :].cpu().numpy()

            # Measure past trajectory discrepancy (t < t_y)
            if t_y > 0:
                past_err_mm = np.max(np.abs(u_cf[:t_y] - u_base[:t_y])) * 1000.0
            else:
                past_err_mm = 0.0

            # Measure final residual displacement change
            drift_shift_mm = (u_cf[-1] - u_base[-1]) * 1000.0

            results["delta_u_residual_mm"].append(drift_shift_mm)
            results["max_past_error_mm"].append(past_err_mm)
            results["trajectories_cf"].append(u_cf)

    return results
