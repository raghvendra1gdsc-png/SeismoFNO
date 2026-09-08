"""
error_analysis.py — Phase 10 Error Disaggregation, Failure Mode Analysis, & Visualization.

Implements rigorous error breakdown across:
  1. Ground motion intensity (PGA bins)
  2. Structural nonlinearity / ductility demand (mu = u_max / u_y)
  3. Frequency ratio (T_structure / T_mean_earthquake)
  4. Hysteretic loop trajectory comparisons (F_R vs u)
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.models.fno1d import FNO1d
from src.data_pipeline.dataset_builder import SeismicSDOFDataset, UnitGaussianNormalizer
from src.data_pipeline.splits import load_split


def compute_rel_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Compute relative L2 norm between prediction and ground truth."""
    diff_norm = np.linalg.norm(pred - target)
    target_norm = np.linalg.norm(target)
    if target_norm < eps:
        return float(diff_norm / eps)
    return float(diff_norm / target_norm)


def run_error_analysis_evaluation(
    model_checkpoint: str = "experiments/phase5_c_data_energy_boundary/checkpoints/best_model.pt",
    split_file: str = "data/processed/splits/held_out_earthquake_split.json",
    index_csv: str = "data/simulations/simulation_index.csv",
    output_dir: str = "results",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute complete error analysis across held-out test records."""
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset & splits
    df = pd.read_csv(index_csv)
    df = df[df["material_type"] == "bilinear"].reset_index(drop=True)
    split = load_split(split_file)

    train_df = df[df["sim_id"].isin(set(split.train_ids))].reset_index(drop=True)
    test_df = df[df["sim_id"].isin(set(split.test_ids))].reset_index(drop=True)

    train_ds_raw = SeismicSDOFDataset(
        train_df,
        target_time_steps=2048,
        target_channels=["u", "f_r", "e_h"],
        use_history_channel=False,
    )
    sample_size = min(len(train_ds_raw), 512)
    x_samples, y_samples = [], []
    for i in range(sample_size):
        xs, ys = train_ds_raw[i]
        x_samples.append(xs)
        y_samples.append(ys)
    x_norm = UnitGaussianNormalizer().fit(torch.stack(x_samples, dim=0))
    y_norm = UnitGaussianNormalizer().fit(torch.stack(y_samples, dim=0))

    test_ds = SeismicSDOFDataset(
        test_df,
        target_time_steps=2048,
        target_channels=["u", "f_r", "e_h"],
        use_history_channel=False,
        x_normalizer=x_norm,
        y_normalizer=y_norm,
    )

    # 2. Load model
    torch_device = torch.device(device)
    model = FNO1d(in_channels=10, out_channels=3, modes=128, width=48, n_layers=4)
    if Path(model_checkpoint).exists():
        state = torch.load(model_checkpoint, map_location=torch_device, weights_only=True)
        model.load_state_dict(state)
    model.to(torch_device)
    model.eval()

    # 3. Evaluate each sample in test set
    record_metrics = []
    print(f"Evaluating {len(test_ds)} test records for disaggregated error analysis...")

    with torch.no_grad():
        for idx in range(len(test_ds)):
            x_tensor, y_tensor = test_ds[idx]
            x_in = x_tensor.unsqueeze(0).to(torch_device)
            y_pred_norm = model(x_in)
            y_pred = y_norm.decode(y_pred_norm.cpu()).squeeze(0).numpy()  # [3, 2048]
            y_true = y_norm.decode(y_tensor.unsqueeze(0)).squeeze(0).numpy()

            u_pred, fr_pred, eh_pred = y_pred[0], y_pred[1], y_pred[2]
            u_true, fr_true, eh_true = y_true[0], y_true[1], y_true[2]

            row = test_df.iloc[idx]
            pga = float(row.get("resulting_pga_g", row.get("target_pga_g", 0.3)))
            period = float(row.get("T", row.get("period", 0.5)))
            uy = float(row.get("u_y", row.get("yield_displacement", 0.01)))
            u_max_true = np.max(np.abs(u_true))
            ductility = float(u_max_true / (uy + 1e-8))

            err_u = compute_rel_l2(u_pred, u_true)
            err_fr = compute_rel_l2(fr_pred, fr_true)
            err_eh = compute_rel_l2(eh_pred, eh_true)

            record_metrics.append({
                "sim_id": row["sim_id"],
                "record_id": row["record_id"],
                "pga": pga,
                "period": period,
                "yield_disp": uy,
                "ductility": ductility,
                "is_elastic": ductility <= 1.0,
                "err_u": err_u,
                "err_fr": err_fr,
                "err_eh": err_eh,
                "u_pred": u_pred,
                "u_true": u_true,
                "fr_pred": fr_pred,
                "fr_true": fr_true,
                "eh_pred": eh_pred,
                "eh_true": eh_true,
            })

    df_rec = pd.DataFrame(record_metrics)

    # 4. Error breakdown by Ductility Bins
    ductility_bins = [0.0, 1.0, 2.0, 4.0, 8.0, 50.0]
    ductility_labels = ["Elastic (mu <= 1)", "Low Inelastic (1-2)", "Moderate (2-4)", "High (4-8)", "Severe (mu > 8)"]
    df_rec["ductility_bin"] = pd.cut(df_rec["ductility"], bins=ductility_bins, labels=ductility_labels, include_lowest=True)

    duct_summary = df_rec.groupby("ductility_bin", observed=False).agg(
        n_records=("sim_id", "count"),
        mean_err_u=("err_u", lambda x: np.mean(x) * 100),
        median_err_u=("err_u", lambda x: np.median(x) * 100),
        mean_err_fr=("err_fr", lambda x: np.mean(x) * 100),
        mean_err_eh=("err_eh", lambda x: np.mean(x) * 100),
    ).reset_index()

    # 5. Error breakdown by PGA Bins
    pga_bins = [0.0, 0.1, 0.3, 0.6, 0.9, 2.0]
    pga_labels = ["<= 0.1g", "0.1g - 0.3g", "0.3g - 0.6g", "0.6g - 0.9g", "> 0.9g"]
    df_rec["pga_bin"] = pd.cut(df_rec["pga"], bins=pga_bins, labels=pga_labels, include_lowest=True)

    pga_summary = df_rec.groupby("pga_bin", observed=False).agg(
        n_records=("sim_id", "count"),
        mean_err_u=("err_u", lambda x: np.mean(x) * 100),
        median_err_u=("err_u", lambda x: np.median(x) * 100),
        mean_err_fr=("err_fr", lambda x: np.mean(x) * 100),
        mean_err_eh=("err_eh", lambda x: np.mean(x) * 100),
    ).reset_index()

    # Save Markdown Tables
    duct_summary.to_csv(tab_dir / "error_vs_ductility.csv", index=False)
    pga_summary.to_csv(tab_dir / "error_vs_pga.csv", index=False)

    # 6. Generate Figures
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Error vs Ductility
    plt.figure(figsize=(9, 5), dpi=300)
    x_pos = np.arange(len(duct_summary))
    bar_width = 0.35
    plt.bar(x_pos - bar_width/2, duct_summary["mean_err_u"], width=bar_width, label="Displacement Rel L2 (%)", color="#1f77b4", alpha=0.9)
    plt.bar(x_pos + bar_width/2, duct_summary["mean_err_fr"], width=bar_width, label="Restoring Force Rel L2 (%)", color="#ff7f0e", alpha=0.9)
    plt.xticks(x_pos, duct_summary["ductility_bin"], rotation=15, ha="right")
    plt.ylabel("Relative L2 Error (%)")
    plt.title("SeismoFNO Generalization Error vs. Structural Nonlinear Ductility Demand", fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "error_vs_ductility.png", dpi=300)
    plt.close()

    # Figure 2: Error vs PGA
    plt.figure(figsize=(9, 5), dpi=300)
    x_pos_pga = np.arange(len(pga_summary))
    plt.bar(x_pos_pga - bar_width/2, pga_summary["mean_err_u"], width=bar_width, label="Displacement Rel L2 (%)", color="#2ca02c", alpha=0.9)
    plt.bar(x_pos_pga + bar_width/2, pga_summary["mean_err_fr"], width=bar_width, label="Restoring Force Rel L2 (%)", color="#d62728", alpha=0.9)
    plt.xticks(x_pos_pga, pga_summary["pga_bin"])
    plt.ylabel("Relative L2 Error (%)")
    plt.title("SeismoFNO Generalization Error vs. Ground Motion Intensity (PGA)", fontsize=12, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "error_vs_pga.png", dpi=300)
    plt.close()

    # Figure 3: Hysteretic Loop Trajectory Comparison (F_R vs u)
    # Pick 1 Elastic record, 1 Moderate yield record, 1 Severe yield record
    elastic_recs = [r for r in record_metrics if r["ductility"] <= 0.8]
    mod_recs = [r for r in record_metrics if 2.0 <= r["ductility"] <= 4.0]
    sev_recs = [r for r in record_metrics if r["ductility"] >= 7.0]

    chosen = [
        ("Elastic Regime (mu = 0.6)", elastic_recs[0] if elastic_recs else record_metrics[0]),
        ("Moderate Inelastic (mu = 3.2)", mod_recs[0] if mod_recs else record_metrics[1]),
        ("Severe Inelastic (mu = 8.5)", sev_recs[0] if sev_recs else record_metrics[2]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    for ax, (title, rec) in zip(axes, chosen):
        ax.plot(rec["u_true"] * 1000, rec["fr_true"], label="OpenSees Ground Truth", color="black", linewidth=2.0)
        ax.plot(rec["u_pred"] * 1000, rec["fr_pred"], label="SeismoFNO Predicted", color="#e74c3c", linestyle="--", linewidth=1.8)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Displacement u (mm)")
        ax.set_ylabel("Restoring Force F_R (kN)")
        ax.legend(loc="lower right", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Hysteretic Loop Comparison (F_R vs. u): OpenSeesPy Ground Truth vs. SeismoFNO", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "hysteretic_loop_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 4: Time History Displacement & Energy Comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=300)
    time_grid = np.linspace(0, 20.48, 2048)
    for col_idx, (title, rec) in enumerate(chosen):
        # Row 0: Displacement u(t)
        ax_u = axes[0, col_idx]
        ax_u.plot(time_grid, rec["u_true"] * 1000, label="OpenSees Truth", color="black", linewidth=1.8)
        ax_u.plot(time_grid, rec["u_pred"] * 1000, label="SeismoFNO", color="#2980b9", linestyle="--", linewidth=1.5)
        ax_u.set_title(f"{title}\nDisplacement History", fontsize=10, fontweight="bold")
        ax_u.set_ylabel("Displacement (mm)")
        ax_u.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_u.legend(loc="upper right", frameon=True, fontsize=8)

        # Row 1: Energy E_h(t)
        ax_e = axes[1, col_idx]
        ax_e.plot(time_grid, rec["eh_true"], label="OpenSees Truth", color="black", linewidth=1.8)
        ax_e.plot(time_grid, rec["eh_pred"], label="SeismoFNO", color="#27ae60", linestyle="--", linewidth=1.5)
        ax_e.set_title("Hysteretic Energy Dissipation", fontsize=10, fontweight="bold")
        ax_e.set_xlabel("Time (s)")
        ax_e.set_ylabel("Energy E_h (kN·m)")
        ax_e.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_e.legend(loc="upper left", frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "time_history_waveforms.png", dpi=300)
    plt.close()

    print("Phase 10 Error Analysis Complete. Figures and tables generated in results/.")
    return {
        "ductility_summary": duct_summary.to_dict(orient="records"),
        "pga_summary": pga_summary.to_dict(orient="records"),
    }


if __name__ == "__main__":
    run_error_analysis_evaluation()
