"""
run_phase8_mdof_suite.py — Master Orchestration for Phase 8 MDOF Operator Learning Suite.

Executes:
  1. Full training & evaluation of MDOF models (Linear FNO, Nonlinear FNO, LSTM, MLP, Held-Out Structure FNO)
  2. Compilation of consolidated comparison table in results/tables/phase8_mdof_benchmark.md (.csv)
  3. Spatiotemporal Zero-Shot Resolution Invariance test across sampling rates from 12.5 Hz to 400 Hz
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.train_mdof import train_mdof, load_config
from src.data_pipeline.splits import load_split
from src.data_pipeline.mdof_dataset import SeismicMDOFDataset, UnitGaussianNormalizer2D
from src.models.fno2d import FNO2d


def df_to_markdown_str(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def run_mdof_experiments():
    configs = [
        "configs/phase8_mdof_fno_linear.yaml",
        "configs/phase8_mdof_fno_nonlinear.yaml",
        "configs/phase8_mdof_lstm_baseline.yaml",
        "configs/phase8_mdof_mlp_baseline.yaml",
        "configs/phase8_mdof_held_out_structure.yaml",
    ]

    for cfg_file in configs:
        cfg = load_config(cfg_file)
        exp_name = cfg["experiment_name"]
        metrics_file = Path(f"experiments/{exp_name}/results/test_metrics.json")
        if not metrics_file.exists():
            print("\n" + "=" * 80)
            print(f"Executing MDOF Experiment: {exp_name} ({cfg_file})")
            print("=" * 80)
            train_mdof(cfg)
        else:
            print(f"Experiment '{exp_name}' already completed. Loaded metrics from: {metrics_file}")


def build_mdof_comparison_table():
    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("MDOF FNO Linear MVP (Random Split)", "experiments/phase8_mdof_fno_linear/results/test_metrics.json", "Random"),
        ("MDOF FNO + Physics Loss (Held-Out-Earthquake)", "experiments/phase8_mdof_fno_nonlinear/results/test_metrics.json", "Held-Out EQ"),
        ("MDOF LSTM Baseline (Held-Out-Earthquake)", "experiments/phase8_mdof_lstm_baseline/results/test_metrics.json", "Held-Out EQ"),
        ("MDOF MLP Baseline (Held-Out-Earthquake)", "experiments/phase8_mdof_mlp_baseline/results/test_metrics.json", "Held-Out EQ"),
        ("MDOF FNO (Held-Out Structural Archetypes)", "experiments/phase8_mdof_held_out_structure/results/test_metrics.json", "Held-Out Structure"),
    ]

    rows = []
    for model_name, path_str, split_type in experiments:
        p = Path(path_str)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        tm = data["test_metrics"]

        row = {
            "Model / System": model_name,
            "Split Strategy": split_type,
            "Overall Rel L2 u(t)": f"{tm.get('overall_rel_l2_u', 0)*100:.2f}%",
            "Overall Rel L2 Force": f"{tm.get('overall_rel_l2_f_r', 0)*100:.2f}%",
            "Overall Rel L2 E_h(t)": f"{tm.get('overall_rel_l2_e_h', 0)*100:.2f}%",
            "Elastic u(t)": f"{tm.get('elastic_rel_l2_u', 0)*100:.2f}%" if "elastic_rel_l2_u" in tm else "N/A",
            "Post-Yield u(t)": f"{tm.get('post_yield_rel_l2_u', 0)*100:.2f}%" if "post_yield_rel_l2_u" in tm else "N/A",
            "Post-Yield E_h(t)": f"{tm.get('post_yield_rel_l2_e_h', 0)*100:.2f}%" if "post_yield_rel_l2_e_h" in tm else "N/A",
            "Train Time (s)": f"{data.get('total_train_time_s', 0):.1f}",
        }
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    csv_file = table_dir / "phase8_mdof_benchmark.csv"
    md_file = table_dir / "phase8_mdof_benchmark.md"

    df_summary.to_csv(csv_file, index=False)
    md_str = df_to_markdown_str(df_summary)

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 8: Multi-Degree-of-Freedom (MDOF) Building Surrogate Benchmark\n\n")
        f.write("Evaluation across 3-story and 5-story building archetypes disaggregated by split and regime:\n\n")
        f.write(md_str)
        f.write("\n")

    print(f"\nSaved consolidated MDOF benchmark table to: {md_file}")
    print("\n" + md_str)
    return df_summary


def evaluate_mdof_resolution_invariance():
    print("\n" + "=" * 80)
    print("Evaluating MDOF Zero-Shot Spatiotemporal Resolution Invariance")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("experiments/phase8_mdof_fno_nonlinear/checkpoints/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("experiments/phase8_mdof_fno_linear/checkpoints/best_model.pt")

    model = FNO2d(in_channels=10, out_channels=3, modes1=4, modes2=64, width=48, n_layers=4)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    df = pd.read_csv("data/simulations/mdof/simulation_index.csv")
    split = load_split("data/processed/splits/mdof_held_out_earthquake_split.json")
    train_df = df[df["sim_id"].isin(set(split.train_ids))].reset_index(drop=True)
    test_df = df[df["sim_id"].isin(set(split.test_ids))].reset_index(drop=True)

    train_ds_raw = SeismicMDOFDataset(train_df, target_time_steps=2048)
    sample_n = min(len(train_ds_raw), 256)
    x_s, y_s = [], []
    for i in range(sample_n):
        xs, ys = train_ds_raw[i]
        x_s.append(xs)
        y_s.append(ys)
    x_norm = UnitGaussianNormalizer2D().fit(torch.stack(x_s, dim=0))
    y_norm = UnitGaussianNormalizer2D().fit(torch.stack(y_s, dim=0))

    test_ds = SeismicMDOFDataset(test_df, target_time_steps=2048, x_normalizer=x_norm, y_normalizer=y_norm)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    collected_x, collected_y = [], []
    count = 0
    for x, y in loader:
        collected_x.append(x)
        collected_y.append(y)
        count += x.shape[0]
        if count >= 100:
            break

    x_all = torch.cat(collected_x, dim=0)[:100]
    y_all = torch.cat(collected_y, dim=0)[:100]

    resolutions = [256, 512, 1024, 2048, 4096, 8192]
    base_duration = 20.48
    rows = []

    with torch.no_grad():
        for res in resolutions:
            dt = base_duration / res
            freq_hz = 1.0 / dt

            # Resample x along time dimension (dim -1)
            b, c, s, t_orig = x_all.shape
            x_res = F.interpolate(x_all.reshape(b * c, s, t_orig), size=res, mode="linear", align_corners=True).reshape(b, c, s, res)
            y_res = F.interpolate(y_all.reshape(b * 3, s, t_orig), size=res, mode="linear", align_corners=True).reshape(b, 3, s, res)

            # Rebuild time grid channel
            time_grid = torch.linspace(0.0, 1.0, res).unsqueeze(0).unsqueeze(0).repeat(s, 1)
            x_res[:, 9, :, :] = time_grid.unsqueeze(0)

            x_res = x_res.to(device)
            pred = model(x_res)

            pred_phys = y_norm.decode(pred)
            y_phys = y_norm.decode(y_res.to(device))

            u_err = (torch.norm(pred_phys[:, 0, :, :] - y_phys[:, 0, :, :]) / (torch.norm(y_phys[:, 0, :, :]) + 1e-6)).item()
            f_err = (torch.norm(pred_phys[:, 1, :, :] - y_phys[:, 1, :, :]) / (torch.norm(y_phys[:, 1, :, :]) + 1e-6)).item()
            e_err = (torch.norm(pred_phys[:, 2, :, :] - y_phys[:, 2, :, :]) / (torch.norm(y_phys[:, 2, :, :]) + 1e-6)).item()

            rows.append({
                "Resolution (Steps)": res,
                "dt (s)": f"{dt:.5f}",
                "Sampling Freq (Hz)": f"{freq_hz:.1f}",
                "Displacement Rel L2 (%)": f"{u_err*100:.2f}%",
                "Force Rel L2 (%)": f"{f_err*100:.2f}%",
                "Energy Rel L2 (%)": f"{e_err*100:.2f}%",
                "_u_val": u_err * 100,
            })

    df_res = pd.DataFrame(rows)
    table_dir = Path("results/tables")
    csv_file = table_dir / "phase8_mdof_resolution_invariance.csv"
    md_file = table_dir / "phase8_mdof_resolution_invariance.md"

    df_display = df_res.drop(columns=["_u_val"])
    df_display.to_csv(csv_file, index=False)

    md_str = df_to_markdown_str(df_display)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 8: Spatiotemporal Zero-Shot Resolution Invariance of MDOF FNO\n\n")
        f.write("Model trained at base resolution $N = 2048$ ($\\Delta t = 0.01$ s) and evaluated across sampling rates from 12.5 Hz to 400 Hz **without retraining**:\n\n")
        f.write(md_str)
        f.write("\n")

    print(f"Saved: {md_file}")
    print("\n" + md_str)

    # Plot resolution invariance curve
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "mdof_resolution_invariance_curve.png"

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(
        df_res["Sampling Freq (Hz)"].astype(float),
        df_res["_u_val"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="#1f77b4",
        label="FNO2d Displacement Rel L2 Error (%)",
    )
    plt.axvline(x=100.0, color="crimson", linestyle="--", alpha=0.7, label="Training Frequency (100 Hz)")
    plt.title("MDOF Zero-Shot Temporal Super-Resolution Invariance", fontsize=13, fontweight="bold")
    plt.xlabel("Sampling Frequency (Hz)", fontsize=11)
    plt.ylabel("Relative L2 Displacement Error (%)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved MDOF resolution invariance figure to: {fig_path}")


def main():
    print("=" * 80)
    print("STARTING SEISMOFNO PHASE 8 MDOF OPERATOR LEARNING SUITE")
    print("=" * 80)

    run_mdof_experiments()
    build_mdof_comparison_table()
    evaluate_mdof_resolution_invariance()

    print("\n" + "=" * 80)
    print("PHASE 8 MDOF SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
