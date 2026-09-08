"""
viz.py — Publication-Quality Scientific Plotting Utilities for SeismoFNO.
"""

from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def set_scientific_style() -> None:
    """Configure matplotlib with publication-grade font and aesthetic parameters."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "figure.autolayout": True,
        "figure.dpi": 300,
    })


def plot_displacement_comparison(
    time: np.ndarray,
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    title: str = "Displacement Response Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot high-contrast comparison of predicted vs ground truth displacement."""
    set_scientific_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, u_gt * 1000.0, color="#F59E0B", linestyle="--", linewidth=1.5, label="OpenSeesPy Ground Truth")
    ax.plot(time, u_pred * 1000.0, color="#00D2FF", linewidth=1.8, label="SeismoFNO Prediction")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Displacement $u(t)$ (mm)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_hysteresis_loop(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    fr_pred: np.ndarray,
    fr_gt: np.ndarray,
    u_y: float,
    title: str = "Force–Displacement Hysteresis Loop",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot nonlinear force-displacement hysteresis loops with yield lines."""
    set_scientific_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(u_gt * 1000.0, fr_gt, color="#F59E0B", linestyle="--", linewidth=1.25, label="OpenSeesPy NLTHA")
    ax.plot(u_pred * 1000.0, fr_pred, color="#00D2FF", linewidth=1.5, label="SeismoFNO Prediction")
    ax.axvline(x=u_y * 1000.0, color="#EF4444", linestyle=":", label="$\\pm u_y$ Yield Boundary")
    ax.axvline(x=-u_y * 1000.0, color="#EF4444", linestyle=":")
    ax.set_xlabel("Displacement $u$ (mm)", fontsize=11)
    ax.set_ylabel("Restoring Force $F_R$ (N)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
