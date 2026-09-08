"""
run_phase7_zero_shot_suite.py — Automated Execution of Phase 7 Zero-Shot Generalization Suite.

Executes and reports three distinct, un-averaged evaluations:
  1. Held-Out-Earthquake Generalization Table
  2. Held-Out-Structure-Parameters Generalization Run & Table
  3. Zero-Shot Resolution Invariance Curve & Table
"""

from typing import Dict, Any
from pathlib import Path
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.fno1d import FNO1d
from src.training.train import train_fno, load_config
from src.data_pipeline.dataset_builder import SeismicSDOFDataset, UnitGaussianNormalizer
from src.data_pipeline.splits import load_split
from src.evaluation.zero_shot_tests import evaluate_resolution_invariance, plot_resolution_invariance_curve


def df_to_markdown_str(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def run_held_out_earthquake_report():
    """Extract and format held-out earthquake generalization results."""
    print("\n" + "=" * 80)
    print("[Phase 7: Task 1] Compiling Held-Out-Earthquake Generalization Report")
    print("=" * 80)

    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("FNO (+ Energy + Boundary)", "experiments/phase5_c_data_energy_boundary/results/test_metrics.json"),
        ("FNO (+ History Channel)", "experiments/phase5_d_history_augmented/results/test_metrics.json"),
        ("Baseline: LSTM", "experiments/phase6_lstm_baseline/results/test_metrics.json"),
        ("Baseline: MLP", "experiments/phase6_mlp_baseline/results/test_metrics.json"),
    ]

    rows = []
    for model_name, path_str in candidates:
        p = Path(path_str)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        tm = data["test_metrics"]
        rows.append({
            "Model": model_name,
            "Overall Rel L2 u(t)": f"{tm['overall_rel_l2_u'] * 100:.2f}%",
            "Overall Rel L2 E_h(t)": f"{tm['overall_rel_l2_e_h'] * 100:.2f}%",
            "Elastic (mu <= 1.0) u(t)": f"{tm['elastic_rel_l2_u'] * 100:.2f}%",
            "Elastic (mu <= 1.0) E_h(t)": f"{tm['elastic_rel_l2_e_h'] * 100:.2f}%",
            "Post-Yield (mu > 1.0) u(t)": f"{tm['post_yield_rel_l2_u'] * 100:.2f}%",
            "Post-Yield (mu > 1.0) E_h(t)": f"{tm['post_yield_rel_l2_e_h'] * 100:.2f}%",
        })

    if not rows:
        print("Held-out earthquake candidate metrics not found yet.")
        return

    df_eq = pd.DataFrame(rows)
    csv_file = table_dir / "phase7_held_out_earthquake.csv"
    md_file = table_dir / "phase7_held_out_earthquake.md"
    df_eq.to_csv(csv_file, index=False)

    md_str = df_to_markdown_str(df_eq)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 7.1: Zero-Shot Generalization on Held-Out Earthquakes\n\n")
        f.write("Evaluated on completely unseen ground motion time series:\n\n")
        f.write(md_str)
        f.write("\n")

    print(f"Saved: {md_file}")
    print("\n" + md_str)


def run_held_out_structure_experiment():
    """Train and evaluate model on held-out structural parameters split."""
    print("\n" + "=" * 80)
    print("[Phase 7: Task 2] Training & Evaluating on Held-Out-Structure Split")
    print("=" * 80)

    cfg_file = "configs/phase7_held_out_structure.yaml"
    metrics_path = Path("experiments/phase7_held_out_structure/results/test_metrics.json")

    if not metrics_path.exists():
        cfg = load_config(cfg_file)
        train_fno(cfg)

    # Load metrics
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tm = data["test_metrics"]
    rows = [{
        "Evaluation": "Held-Out Structure Parameters (Zero-Shot)",
        "Overall Rel L2 u(t)": f"{tm['overall_rel_l2_u'] * 100:.2f}%",
        "Overall Rel L2 Force": f"{tm.get('overall_rel_l2_f_r', 0) * 100:.2f}%",
        "Overall Rel L2 E_h(t)": f"{tm['overall_rel_l2_e_h'] * 100:.2f}%",
        "Elastic (mu <= 1.0) u(t)": f"{tm['elastic_rel_l2_u'] * 100:.2f}%",
        "Post-Yield (mu > 1.0) u(t)": f"{tm['post_yield_rel_l2_u'] * 100:.2f}%",
        "Post-Yield (mu > 1.0) E_h(t)": f"{tm['post_yield_rel_l2_e_h'] * 100:.2f}%",
        "Train Time (s)": f"{data.get('total_train_time_s', 0):.1f}",
    }]

    df_struct = pd.DataFrame(rows)
    table_dir = Path("results/tables")
    csv_file = table_dir / "phase7_held_out_structure.csv"
    md_file = table_dir / "phase7_held_out_structure.md"

    df_struct.to_csv(csv_file, index=False)
    md_str = df_to_markdown_str(df_struct)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 7.2: Zero-Shot Generalization on Held-Out Structural Parameters\n\n")
        f.write("Evaluated on structures with natural periods and ductility ratios held out during training:\n\n")
        f.write(md_str)
        f.write("\n")

    print(f"Saved: {md_file}")
    print("\n" + md_str)


def run_resolution_invariance_test():
    """Evaluate trained FNO model across fine and coarse dt without retraining."""
    print("\n" + "=" * 80)
    print("[Phase 7: Task 3] Evaluating Zero-Shot Resolution Invariance across Temporal Discretizations")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load trained FNO model checkpoint (from Phase 5c)
    ckpt_path = Path("experiments/phase5_c_data_energy_boundary/checkpoints/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("experiments/fno_sdof_linear_mvp/checkpoints/best_model.pt")

    model = FNO1d(in_channels=10, out_channels=3, modes=128, width=48, n_layers=4)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Load train and test datasets
    df = pd.read_csv("data/simulations/simulation_index.csv")
    df = df[df["material_type"] == "bilinear"].reset_index(drop=True)
    split = load_split("data/processed/splits/held_out_earthquake_split.json")
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
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    resolutions = [256, 512, 1024, 2048, 4096, 8192]
    df_res = evaluate_resolution_invariance(
        model=model,
        test_loader=test_loader,
        y_normalizer=y_norm,
        resolutions=resolutions,
        base_duration=20.48,
        device=device,
        max_samples=200,
    )

    table_dir = Path("results/tables")
    csv_file = table_dir / "phase7_resolution_invariance.csv"
    md_file = table_dir / "phase7_resolution_invariance.md"

    # Drop internal column for display
    df_display = df_res.drop(columns=["rel_l2_u_val"])
    df_display.to_csv(csv_file, index=False)

    md_str = df_to_markdown_str(df_display)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 7.3: Zero-Shot Resolution Invariance of SeismoFNO\n\n")
        f.write("Model trained strictly at base resolution $N = 2048$ ($\\Delta t = 0.01$ s) and evaluated across sampling frequencies from 12.5 Hz to 400 Hz **without retraining**:\n\n")
        f.write(md_str)
        f.write("\n")

    print(f"Saved: {md_file}")
    print("\n" + md_str)

    plot_resolution_invariance_curve(df_res, output_path="results/figures/resolution_invariance_curve.png")


def main():
    print("=" * 80)
    print("STARTING SEISMOFNO PHASE 7 ZERO-SHOT GENERALIZATION & RESOLUTION INVARIANCE SUITE")
    print("=" * 80)

    run_held_out_earthquake_report()
    run_held_out_structure_experiment()
    run_resolution_invariance_test()

    print("\n" + "=" * 80)
    print("PHASE 7 ZERO-SHOT EVALUATION SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
