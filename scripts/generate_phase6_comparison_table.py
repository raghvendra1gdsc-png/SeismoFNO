"""
generate_phase6_comparison_table.py — Consolidate and Tabulate Phase 5 FNO Ablations & Phase 6 Baselines.

Compiles test metrics across:
  (a) FNO: Data Loss Only
  (b) FNO: Data + Energy Consistency
  (c) FNO: Data + Energy + Boundary
  (d) FNO: Data + Energy + Boundary + Auxiliary History Channel
  (e) Baseline: Recurrent LSTM Sequence Model
  (f) Baseline: Deep Residual MLP

Generates Markdown and CSV summary tables in results/tables/
disaggregated by elastic regime (mu <= 1.0) vs. post-yield regime (mu > 1.0).
"""

from typing import Dict, List, Any
from pathlib import Path
import json
import pandas as pd


EXPERIMENTS = [
    ("a", "phase5_a_data_only", "FNO (Data Only)"),
    ("b", "phase5_b_data_energy", "FNO (+ Energy Consistency)"),
    ("c", "phase5_c_data_energy_boundary", "FNO (+ Energy + Boundary)"),
    ("d", "phase5_d_history_augmented", "FNO (+ Physics + History Channel)"),
    ("e", "phase6_lstm_baseline", "Baseline: LSTM Sequence Model"),
    ("f", "phase6_mlp_baseline", "Baseline: Deep Residual MLP"),
]


def load_experiment_metrics(exp_name: str) -> Dict[str, Any]:
    metrics_path = Path(f"experiments/{exp_name}/results/test_metrics.json")
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_table() -> pd.DataFrame:
    rows = []

    for label, exp_id, description in EXPERIMENTS:
        data = load_experiment_metrics(exp_id)
        if data is None:
            rows.append({
                "ID": f"({label})",
                "Model Architecture & Loss Formulation": description,
                "Overall Rel L2 u(t)": "Pending",
                "Overall Rel L2 E_h(t)": "Pending",
                "Elastic (mu <= 1) u(t)": "Pending",
                "Elastic (mu <= 1) E_h(t)": "Pending",
                "Post-Yield (mu > 1) u(t)": "Pending",
                "Post-Yield (mu > 1) E_h(t)": "Pending",
                "Train Time (s)": "-",
                "Parameters": "-",
            })
            continue

        tm = data["test_metrics"]
        row = {
            "ID": f"({label})",
            "Model Architecture & Loss Formulation": description,
            "Overall Rel L2 u(t)": f"{tm['overall_rel_l2_u'] * 100:.2f}%",
            "Overall Rel L2 E_h(t)": f"{tm['overall_rel_l2_e_h'] * 100:.2f}%",
            "Elastic (mu <= 1) u(t)": f"{tm['elastic_rel_l2_u'] * 100:.2f}%",
            "Elastic (mu <= 1) E_h(t)": f"{tm['elastic_rel_l2_e_h'] * 100:.2f}%",
            "Post-Yield (mu > 1) u(t)": f"{tm['post_yield_rel_l2_u'] * 100:.2f}%",
            "Post-Yield (mu > 1) E_h(t)": f"{tm['post_yield_rel_l2_e_h'] * 100:.2f}%",
            "Train Time (s)": f"{data.get('total_train_time_s', 0):.1f}",
            "Parameters": f"{data.get('n_parameters', 0):,}",
        }
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    return df_summary


def df_to_markdown_str(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main():
    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    df = build_comparison_table()

    # Save CSV
    csv_file = table_dir / "phase6_full_benchmark_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved full benchmark CSV to: {csv_file}")

    # Also update phase5 table if relevant
    p5_csv = table_dir / "phase5_ablation_summary.csv"
    df.to_csv(p5_csv, index=False)

    # Save Markdown
    md_str = df_to_markdown_str(df)
    md_file = table_dir / "phase6_full_benchmark_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# SeismoFNO Benchmark: FNO Core vs. Physics Losses vs. LSTM & MLP Baselines\n\n")
        f.write("Evaluation across unseen earthquake events (Held-Out-Earthquake Split) disaggregated by elastic regime ($\\mu \\le 1.0$) vs. post-yield regime ($\\mu > 1.0$):\n\n")
        f.write(md_str)
        f.write("\n")
    print(f"Saved full benchmark Markdown table to: {md_file}")

    p5_md = table_dir / "phase5_ablation_summary.md"
    with open(p5_md, "w", encoding="utf-8") as f:
        f.write("# Phase 5 & 6: Bilinear-Hysteretic SDOF Benchmark (Held-Out-Earthquake Split)\n\n")
        f.write("Evaluation across unseen earthquake events disaggregated by elastic regime ($\\mu \\le 1.0$) vs. post-yield regime ($\\mu > 1.0$):\n\n")
        f.write(md_str)
        f.write("\n")

    print("\n" + md_str)


if __name__ == "__main__":
    main()
