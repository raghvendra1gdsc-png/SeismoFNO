"""
zero_shot_tests.py — Comprehensive Zero-Shot Evaluation Suite for SeismoFNO.

Implements three separately reported evaluations per AGENTS.md Hard Rules 2 & 6:
  1. Held-out-earthquake generalization (unseen seismic events)
  2. Held-out-structure-parameters generalization (unseen period T and ductility mu combinations)
  3. Zero-shot resolution invariance (evaluating at finer/coarser dt without retraining)

Never averages these three distinct evaluations into a single headline number.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.losses.data_loss import RelativeL2Loss
from src.data_pipeline.dataset_builder import SeismicSDOFDataset, UnitGaussianNormalizer


def evaluate_split_disaggregated(
    model: nn.Module,
    test_loader: DataLoader,
    y_normalizer: Optional[UnitGaussianNormalizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluate model predictions disaggregated by elastic (mu <= 1.0) vs. post-yield (mu > 1.0) regime.

    Args:
        model: PyTorch model module
        test_loader: DataLoader yielding (x, y, meta) or (x, y)
        y_normalizer: Normalizer for decoding output predictions to physical units
        device: Evaluation device (CPU/CUDA/MPS)

    Returns:
        Dictionary of measured relative L2 errors for u(t), F_R(t), and E_h(t)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    criterion = RelativeL2Loss()

    all_u_errs = []
    all_f_errs = []
    all_e_errs = []
    all_ductilities = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, y, meta = batch
            else:
                x, y = batch
                meta = {}

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            if y_normalizer is not None:
                pred = y_normalizer.decode(pred)
                y = y_normalizer.decode(y)

            # Extract ductilities if present in meta
            mu_batch = meta.get("mu_target", torch.ones(len(x)) * 2.0).cpu().numpy()

            # Compute sample-wise errors
            batch_sz = x.shape[0]
            for b in range(batch_sz):
                p_u = pred[b : b + 1, 0:1, :]
                y_u = y[b : b + 1, 0:1, :]
                p_f = pred[b : b + 1, 1:2, :]
                y_f = y[b : b + 1, 1:2, :]
                p_e = pred[b : b + 1, 2:3, :]
                y_e = y[b : b + 1, 2:3, :]

                u_err = criterion(p_u, y_u).item()
                f_err = criterion(p_f, y_f).item()
                e_err = criterion(p_e, y_e).item()

                all_u_errs.append(u_err)
                all_f_errs.append(f_err)
                all_e_errs.append(e_err)
                all_ductilities.append(float(mu_batch[b]))

    all_u_errs = np.array(all_u_errs)
    all_f_errs = np.array(all_f_errs)
    all_e_errs = np.array(all_e_errs)
    all_ductilities = np.array(all_ductilities)

    elastic_mask = all_ductilities <= 1.0001
    post_yield_mask = all_ductilities > 1.0001

    results = {
        "n_total": len(all_u_errs),
        "overall_rel_l2_u": float(np.mean(all_u_errs)),
        "overall_rel_l2_f_r": float(np.mean(all_f_errs)),
        "overall_rel_l2_e_h": float(np.mean(all_e_errs)),
        "n_elastic": int(np.sum(elastic_mask)),
        "elastic_rel_l2_u": float(np.mean(all_u_errs[elastic_mask])) if np.any(elastic_mask) else 0.0,
        "elastic_rel_l2_f_r": float(np.mean(all_f_errs[elastic_mask])) if np.any(elastic_mask) else 0.0,
        "elastic_rel_l2_e_h": float(np.mean(all_e_errs[elastic_mask])) if np.any(elastic_mask) else 0.0,
        "n_post_yield": int(np.sum(post_yield_mask)),
        "post_yield_rel_l2_u": float(np.mean(all_u_errs[post_yield_mask])) if np.any(post_yield_mask) else 0.0,
        "post_yield_rel_l2_f_r": float(np.mean(all_f_errs[post_yield_mask])) if np.any(post_yield_mask) else 0.0,
        "post_yield_rel_l2_e_h": float(np.mean(all_e_errs[post_yield_mask])) if np.any(post_yield_mask) else 0.0,
    }

    return results


def evaluate_resolution_invariance(
    model: nn.Module,
    test_loader: DataLoader,
    y_normalizer: Optional[UnitGaussianNormalizer] = None,
    resolutions: List[int] = [256, 512, 1024, 2048, 4096, 8192],
    base_duration: float = 20.48,
    device: Optional[torch.device] = None,
    max_samples: int = 200,
) -> pd.DataFrame:
    """
    Evaluate zero-shot resolution invariance by passing variable sequence lengths into FNO.

    Args:
        model: FNO model trained at base resolution (e.g. 2048 time steps)
        test_loader: DataLoader yielding unnormalized or normalized test batches
        y_normalizer: Normalizer for physical unit decoding
        resolutions: List of grid sizes L to evaluate
        base_duration: Total simulation duration in seconds (T_total = 20.48 s)
        device: Computation device
        max_samples: Number of test samples to evaluate

    Returns:
        DataFrame containing resolution L, dt (s), sampling frequency (Hz), and relative L2 error (%)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    criterion = RelativeL2Loss()

    # Collect sample subset
    collected_x = []
    collected_y = []

    count = 0
    for batch in test_loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        collected_x.append(x)
        collected_y.append(y)
        count += x.shape[0]
        if count >= max_samples:
            break

    x_all = torch.cat(collected_x, dim=0)[:max_samples]
    y_all = torch.cat(collected_y, dim=0)[:max_samples]

    rows = []

    with torch.no_grad():
        for res in resolutions:
            dt = base_duration / res
            sampling_rate_hz = 1.0 / dt

            # Resample x along time dimension (dim 2)
            # x shape: (B, C_in, L_orig)
            x_resampled = F.interpolate(x_all, size=res, mode="linear", align_corners=True)

            # If time_grid is the last channel, rebuild normalized time grid on [0, 1]
            time_grid = torch.linspace(0, 1, res, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            x_resampled[:, -1:, :] = time_grid

            x_resampled = x_resampled.to(device)
            pred_resampled = model(x_resampled)

            # Resample ground truth y to the same resolution
            y_resampled = F.interpolate(y_all, size=res, mode="linear", align_corners=True).to(device)

            if y_normalizer is not None:
                # Normalizer decode requires standard spatial stats, applied per time point
                pred_resampled = y_normalizer.decode(pred_resampled)
                y_resampled = y_normalizer.decode(y_resampled)

            pred_u = pred_resampled[:, 0:1, :]
            y_u = y_resampled[:, 0:1, :]

            pred_f = pred_resampled[:, 1:2, :]
            y_f = y_resampled[:, 1:2, :]

            pred_e = pred_resampled[:, 2:3, :]
            y_e = y_resampled[:, 2:3, :]

            rel_l2_u = criterion(pred_u, y_u).item() * 100.0
            rel_l2_f = criterion(pred_f, y_f).item() * 100.0
            rel_l2_e = criterion(pred_e, y_e).item() * 100.0

            rows.append({
                "Resolution (Steps)": res,
                "dt (s)": f"{dt:.5f}",
                "Sampling Freq (Hz)": f"{sampling_rate_hz:.1f}",
                "Displacement Rel L2 Error (%)": f"{rel_l2_u:.2f}%",
                "Force Rel L2 Error (%)": f"{rel_l2_f:.2f}%",
                "Hysteretic Energy Rel L2 Error (%)": f"{rel_l2_e:.2f}%",
                "rel_l2_u_val": rel_l2_u,
            })

    df_res = pd.DataFrame(rows)
    return df_res


def plot_resolution_invariance_curve(df: pd.DataFrame, output_path: str = "results/figures/resolution_invariance_curve.png"):
    """Plot resolution invariance error curve."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    res_values = df["Resolution (Steps)"].values
    errors = df["rel_l2_u_val"].values

    ax.plot(res_values, errors, marker="o", linewidth=2.5, color="#1f77b4", label="SeismoFNO (Zero-Shot)")
    ax.axvline(x=2048, color="red", linestyle="--", alpha=0.7, label="Training Resolution (N=2048, dt=0.01s)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Temporal Discretization Resolution $N$ (Grid Steps)", fontsize=12)
    ax.set_ylabel("Relative $L_2$ Displacement Error (%)", fontsize=12)
    ax.set_title("Zero-Shot Resolution Invariance of SeismoFNO", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved resolution invariance plot to: {output_path}")
