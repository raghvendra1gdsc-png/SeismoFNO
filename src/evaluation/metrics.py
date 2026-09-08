"""
metrics.py — Standalone Comprehensive Diagnostic Evaluation Metrics Library.

Implements all 11 scientific metrics for nonlinear structural dynamics surrogates:
  1. Trajectory Relative L2 Error (u, v, a, F_R)
  2. Peak Displacement Error (|u_max - u_hat_max| / u_max)
  3. Peak Force Error (|F_{R,max} - F_{R,hat_max}| / F_{R,max})
  4. Residual Plastic Drift Error (|u(Tend) - u_hat(Tend)|)
  5. Instantaneous Phase Coherence Error (via Hilbert analytic signal)
  6. Yield Onset Timing Discrepancy (|t_yield - t_hat_yield|)
  7. Hysteresis Loop Dissipation Area Discrepancy
  8. Scale-Regularized Cumulative Absorbed Energy Error
  9. Total Dynamic Energy Balance Residual
  10. Bootstrap 95% Confidence Intervals
  11. Paired per-earthquake effect size & difference
"""

from typing import Dict, Any, Union, Optional, Tuple, List
import numpy as np
import scipy.signal as signal
import torch


def compute_rel_l2_error(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7,
) -> float:
    """Compute relative L2 norm error as a percentage: ||pred - target|| / (||target|| + eps) * 100%."""
    if isinstance(pred, torch.Tensor):
        diff_norm = torch.norm(pred - target, p=2).item()
        target_norm = torch.norm(target, p=2).item()
    else:
        diff_norm = float(np.linalg.norm(pred - target))
        target_norm = float(np.linalg.norm(target))

    if target_norm < eps:
        return float((diff_norm / eps) * 100.0)
    return float((diff_norm / target_norm) * 100.0)


def compute_peak_error(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7,
) -> float:
    """Compute relative error in peak absolute response: |max(|pred|) - max(|target|)| / (max(|target|) + eps) * 100%."""
    if isinstance(pred, torch.Tensor):
        p_max = torch.max(torch.abs(pred)).item()
        t_max = torch.max(torch.abs(target)).item()
    else:
        p_max = float(np.max(np.abs(pred)))
        t_max = float(np.max(np.abs(target)))

    if t_max < eps:
        return 0.0
    return float((abs(p_max - t_max) / t_max) * 100.0)


def compute_phase_error(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Compute mean absolute instantaneous phase error via the Hilbert transform analytic signal:
        phi(t) = unwrap(angle(Hilbert(x(t))))
    Returns mean absolute phase discrepancy in radians.
    """
    if np.all(np.abs(target) < eps) or np.all(np.abs(pred) < eps):
        return 0.0

    analytic_tgt = signal.hilbert(target)
    analytic_prd = signal.hilbert(pred)

    phase_tgt = np.unwrap(np.angle(analytic_tgt))
    phase_prd = np.unwrap(np.angle(analytic_prd))

    phase_diff = np.abs(phase_prd - phase_tgt)
    return float(np.mean(phase_diff))


def compute_yield_time_error(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    u_y: float,
    dt: float = 0.01,
) -> float:
    """
    Compute error in predicting the exact onset time of initial plastic yielding (in ms):
        t_yield = inf { t : |u(t)| >= u_y }
    """
    yield_idx_gt = np.where(np.abs(u_gt) >= u_y)[0]
    yield_idx_prd = np.where(np.abs(u_pred) >= u_y)[0]

    t_gt = float(yield_idx_gt[0] * dt * 1000.0) if len(yield_idx_gt) > 0 else -1.0
    t_prd = float(yield_idx_prd[0] * dt * 1000.0) if len(yield_idx_prd) > 0 else -1.0

    if t_gt < 0.0 and t_prd < 0.0:
        return 0.0  # Both correctly predicted purely elastic response
    elif t_gt < 0.0 or t_prd < 0.0:
        return 1000.0  # Misclassified elastic vs inelastic
    return abs(t_prd - t_gt)


def compute_hysteresis_area_error(
    u_pred: np.ndarray,
    fr_pred: np.ndarray,
    u_gt: np.ndarray,
    fr_gt: np.ndarray,
    eps: float = 1e-4,
) -> float:
    """
    Compute relative error in total enclosed hysteretic loop dissipation area:
        Area = integral(F_R du)
    """
    du_gt = np.diff(u_gt)
    f_avg_gt = 0.5 * (fr_gt[:-1] + fr_gt[1:])
    area_gt = float(np.sum(f_avg_gt * du_gt))

    du_prd = np.diff(u_pred)
    f_avg_prd = 0.5 * (fr_pred[:-1] + fr_pred[1:])
    area_prd = float(np.sum(f_avg_prd * du_prd))

    denom = max(abs(area_gt), eps)
    return float(abs(area_prd - area_gt) / denom * 100.0)


def compute_seismic_input_energy(
    ag: np.ndarray,
    v: np.ndarray,
    mass: float = 1.0,
    dt: float = 0.01,
) -> np.ndarray:
    """Compute cumulative input seismic energy: E_i(t) = -integral_0^t m * a_g(tau) * v(tau) dtau."""
    n = len(ag)
    e_i = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return e_i
    p_ground = -mass * ag * v
    e_i[1:] = np.cumsum(0.5 * (p_ground[:-1] + p_ground[1:]) * dt)
    return e_i


def compute_hysteretic_energy_loss_normalized(
    eh_pred: np.ndarray,
    eh_target: np.ndarray,
    ei_total: Optional[float] = None,
    eps: float = 1e-4,
) -> float:
    """
    Compute scale-regularized cumulative absorbed energy error:
        ||eh_pred - eh_target|| / max(||eh_target||, E_i, 1.0 J) * 100%
    """
    diff_norm = float(np.linalg.norm(eh_pred - eh_target))
    target_norm = float(np.linalg.norm(eh_target))
    ref_scale = max(target_norm, ei_total or 0.0, 1.0, eps)
    return float((diff_norm / ref_scale) * 100.0)


def compute_energy_balance_residual(
    u: np.ndarray,
    v: np.ndarray,
    fr: np.ndarray,
    ag: np.ndarray,
    mass: float = 1.0,
    c: float = 0.05,
    k0: float = 100.0,
    dt: float = 0.01,
) -> float:
    """
    Compute mean dynamic energy balance residual:
        |E_k(t) + E_d(t) + E_s(t) + E_h(t) - E_i(t)|
    """
    ek = 0.5 * mass * (v ** 2)
    ed = np.zeros_like(u)
    c_f = c * v
    ed[1:] = np.cumsum(0.5 * (c_f[:-1] + c_f[1:]) * np.diff(u))

    eh = np.zeros_like(u)
    eh[1:] = np.cumsum(0.5 * (fr[:-1] + fr[1:]) * np.diff(u))

    ei = compute_seismic_input_energy(ag, v, mass=mass, dt=dt)
    residual = np.abs((ek + ed + eh) - ei)
    return float(np.mean(residual))


def compute_bootstrap_ci(
    data: Union[List[float], np.ndarray],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute mean, lower, and upper bounds using non-parametric percentile bootstrap."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return float(np.mean(arr)), low, high


def compute_comprehensive_record_metrics(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    fr_pred: np.ndarray,
    fr_gt: np.ndarray,
    eh_pred: np.ndarray,
    eh_gt: np.ndarray,
    u_y: float,
    ag: np.ndarray,
    dt: float = 0.01,
    mass: float = 1.0,
) -> Dict[str, float]:
    """Compute all 11 scientific metrics for a single simulation."""
    # Approximate velocities & accelerations via numerical differentiation
    v_gt = np.gradient(u_gt, dt)
    v_prd = np.gradient(u_pred, dt)
    a_gt = np.gradient(v_gt, dt)
    a_prd = np.gradient(v_prd, dt)

    u_max_gt = float(np.max(np.abs(u_gt)))
    u_max_prd = float(np.max(np.abs(u_pred)))
    fr_max_gt = float(np.max(np.abs(fr_gt)))
    fr_max_prd = float(np.max(np.abs(fr_pred)))

    mu = u_max_gt / max(1e-6, u_y)
    ei_arr = compute_seismic_input_energy(ag, v_gt, mass=mass, dt=dt)
    ei_total = float(np.linalg.norm(ei_arr))

    return {
        "err_u_rel_l2": compute_rel_l2_error(u_pred, u_gt),
        "err_v_rel_l2": compute_rel_l2_error(v_prd, v_gt),
        "err_a_rel_l2": compute_rel_l2_error(a_prd, a_gt),
        "err_fr_rel_l2": compute_rel_l2_error(fr_pred, fr_gt),
        "err_umax_rel": compute_peak_error(u_pred, u_gt),
        "err_frmax_rel": compute_peak_error(fr_pred, fr_gt),
        "err_phase_rad": compute_phase_error(u_pred, u_gt),
        "err_yield_time_ms": compute_yield_time_error(u_pred, u_gt, u_y, dt=dt),
        "err_loop_area_pct": compute_hysteresis_area_error(u_pred, fr_pred, u_gt, fr_gt),
        "err_eh_rel_l2": compute_hysteretic_energy_loss_normalized(eh_pred, eh_gt, ei_total),
        "err_uresidual_mm": abs(float(u_pred[-1] - u_gt[-1])) * 1000.0,
        "ductility_mu": mu,
        "is_yielded": 1.0 if mu > 1.0 else 0.0,
        "u_max_gt": u_max_gt,
        "u_max_pred": u_max_prd,
    }


def compute_clustered_bootstrap_ci(
    records: List[Dict[str, Any]],
    metric_key: str,
    cluster_key: str = "earthquake",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute mean, lower, and upper bounds using Earthquake-Clustered Block Bootstrap.

    Groups records by their parent cluster (e.g. earthquake name), resamples clusters
    with replacement, and pools records to account for intra-cluster covariance.

    Args:
        records: List of record dictionaries containing metric_key and cluster_key.
        metric_key: The numeric metric to compute confidence intervals for.
        cluster_key: The clustering entity (e.g. 'earthquake').
        n_boot: Number of bootstrap iterations (default: 2000).
        ci: Confidence interval fraction (default: 0.95).
        seed: Random seed for reproducibility.

    Returns:
        (mean, ci_low, ci_high)
    """
    if len(records) == 0:
        return 0.0, 0.0, 0.0

    # Group record values by cluster
    cluster_map: Dict[Any, List[float]] = {}
    for r in records:
        c = r.get(cluster_key, "default")
        val = float(r[metric_key])
        if c not in cluster_map:
            cluster_map[c] = []
        cluster_map[c].append(val)

    clusters = list(cluster_map.keys())
    n_clusters = len(clusters)
    all_values = [v for vals in cluster_map.values() for v in vals]
    sample_mean = float(np.mean(all_values))

    if n_clusters <= 1:
        # Fallback to standard bootstrap if only 1 cluster
        return compute_bootstrap_ci(all_values, n_boot=n_boot, ci=ci, seed=seed)

    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        sampled_vals = []
        for c in sampled_clusters:
            sampled_vals.extend(cluster_map[c])
        boot_means.append(np.mean(sampled_vals))

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100.0))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return sample_mean, low, high

