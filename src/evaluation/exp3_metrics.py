"""
exp3_metrics.py — Dedicated Evaluation Metrics for EXP 3 Causal Interventions and Physics Supervision.
"""

from typing import Dict, List, Tuple, Union
import numpy as np


def compute_dose_response_linearity(
    doses_mm: Union[List[float], np.ndarray],
    delta_residual_mm: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """
    Compute linear dose-response statistics (slope, intercept, R^2, and Pearson correlation)
    between injected physical state perturbation and observed residual displacement offset.

    Args:
        doses_mm: Injected plastic state shifts Delta u_p in mm (e.g. [-10, -5, 0, 5, 10])
        delta_residual_mm: Resulting change in residual displacement [u_cf(Tend) - u_base(Tend)] in mm

    Returns:
        metrics: Dictionary containing slope, intercept, r2_score, pearson_corr, and monotonicity_flag
    """
    x = np.asarray(doses_mm, dtype=np.float64)
    y = np.asarray(delta_residual_mm, dtype=np.float64)

    # 1. Least-squares linear regression: y = m * x + c
    n = len(x)
    if n < 2:
        return {"slope": 0.0, "intercept": 0.0, "r2_score": 0.0, "pearson_corr": 0.0, "is_monotonic": False}

    cov_xy = np.cov(x, y)[0, 1]
    var_x = np.var(x, ddof=1)
    if var_x > 1e-12:
        slope = cov_xy / var_x
        intercept = np.mean(y) - slope * np.mean(x)
    else:
        slope = 0.0
        intercept = np.mean(y)

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 1.0

    # Pearson correlation
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x > 1e-12 and std_y > 1e-12:
        pearson = float(np.corrcoef(x, y)[0, 1])
    else:
        pearson = 1.0 if np.allclose(x, y) else 0.0

    # Monotonicity check (strictly non-decreasing)
    is_monotonic = bool(np.all(np.diff(y) >= -1e-6))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2_score": float(r2),
        "pearson_corr": float(pearson),
        "is_monotonic": is_monotonic,
    }


def compute_state_tracking_metrics(
    s_pred: np.ndarray,
    s_true: np.ndarray,
) -> Dict[str, float]:
    """
    Compute state tracking error metrics across plastic displacement u_p and kinematic back-stress alpha_b.

    Args:
        s_pred: Predicted physical state [2, Length] -> [u_p, alpha_b]
        s_true: Ground truth physical state [2, Length]

    Returns:
        metrics: Dictionary containing Rel L2 and RMSE for both physical state variables
    """
    up_pred = s_pred[0, :]
    up_true = s_true[0, :]
    ab_pred = s_pred[1, :]
    ab_true = s_true[1, :]

    # Plastic displacement metrics (meters -> mm for RMSE)
    rel_l2_up = np.linalg.norm(up_pred - up_true) / (np.linalg.norm(up_true) + 1e-6) * 100.0
    rmse_up_mm = np.sqrt(np.mean((up_pred - up_true) ** 2)) * 1000.0

    # Back-stress metrics (Newtons)
    rel_l2_ab = np.linalg.norm(ab_pred - ab_true) / (np.linalg.norm(ab_true) + 1e-4) * 100.0
    rmse_ab_N = np.sqrt(np.mean((ab_pred - ab_true) ** 2))

    return {
        "rel_l2_up_pct": float(rel_l2_up),
        "rmse_up_mm": float(rmse_up_mm),
        "rel_l2_ab_pct": float(rel_l2_ab),
        "rmse_ab_N": float(rmse_ab_N),
    }
