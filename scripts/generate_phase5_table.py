"""
generate_phase5_table.py — Consolidate and Tabulate Phase 5 Ablation Experiments.

Extracts disaggregated metrics across:
  (a) Data loss only
  (b) Data + Energy consistency
  (c) Data + Energy + Boundary
  (d) Data + Energy + Boundary + Auxiliary History Channel

Produces Markdown and CSV summary tables in results/tables/
broken down by elastic-regime (mu <= 1.0) vs. post-yield regime (mu > 1.0).
"""

from typing import Dict, List, Any
from pathlib import Path
import json
import pandas as pd


EXPERIMENTS = [
    ("a", "phase5_a_data_only", "Data Loss Only"),
    ("b", "phase5_b_data_energy", "Data + Energy Consistency"),
    ("c", "phase5_c_data_energy_boundary", "Data + Energy + Boundary"),
    ("d", "phase5_d_history_augmented", "Data + Energy + Boundary + History Channel"),
]


def load_experiment_metrics(exp_name: str) -> Dict[str, Any]:
    metrics_path = Path(f"experiments/{exp_name}/results/test_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_summary_table() -> pd.DataFrame:
    rows = []

    for label, exp_id, description in EXPERIMENTS:
        data = load_experiment_metrics(exp_id)
        tm = data["test_metrics"]

        row = {
            "Exp": f"({label})",
            "Model Variant / Loss Formulation": description,
            "Overall Rel L2 u(t)": f"{tm['overall_rel_l2_u'] * 100:.2f}%",
            "Overall Rel L2 E_h(t)": f"{tm['overall_rel_l2_e_h'] * 100:.2f}%",
            "Elastic (mu <= 1) Rel L2 u(t)": f"{tm['elastic_rel_l2_u'] * 100:.2f}%",
            "Elastic (mu <= 1) Rel L2 E_h(t)": f"{tm['elastic_rel_l2_e_h'] * 100:.2f}%",
            "Post-Yield (mu > 1) Rel L2 u(t)": f"{tm['post_yield_rel_l2_u'] * 100:.2f}%",
            "Post-Yield (mu > 1) Rel L2 E_h(t)": f"{tm['post_yield_rel_l2_e_h'] * 100:.2f}%",
            "Train Time (s)": f"{data['total_train_time_s']:.1f}",
            "Params": f"{data['n_parameters']:,}",
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

    df = build_summary_table()

    # Save CSV
    csv_file = table_dir / "phase5_ablation_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved summary CSV to: {csv_file}")

    # Save Markdown
    md_str = df_to_markdown_str(df)
    md_file = table_dir / "phase5_ablation_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 5: Bilinear-Hysteretic SDOF Ablation Summary (Held-Out-Earthquake Split)\n\n")
        f.write("Evaluation across unseen earthquake events disaggregated by elastic regime ($\\mu \\le 1.0$) vs. post-yield regime ($\\mu > 1.0$):\n\n")
        f.write(md_str)
        f.write("\n")
    print(f"Saved summary Markdown table to: {md_file}")
    print("\n" + md_str)


if __name__ == "__main__":
    main()
