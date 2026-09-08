"""
run_phase6_experiments.py — Automated Execution of Phase 5 Final Run and Phase 6 Baselines.

Runs:
  - (d) FNO with Data + Energy + Boundary + History Channel (if not completed)
  - (e) LSTM Sequence Baseline Model
  - (f) Deep Residual MLP Baseline Model

Then generates the consolidated benchmark comparison table in results/tables/
"""

import sys
import time
from pathlib import Path

from src.training.train import train_fno, load_config
from scripts.generate_phase6_comparison_table import main as generate_table_main


EXPERIMENT_CONFIGS = [
    ("configs/phase5_d_history_augmented.yaml", "experiments/phase5_d_history_augmented/results/test_metrics.json"),
    ("configs/phase6_lstm_baseline.yaml", "experiments/phase6_lstm_baseline/results/test_metrics.json"),
    ("configs/phase6_mlp_baseline.yaml", "experiments/phase6_mlp_baseline/results/test_metrics.json"),
]


def main():
    print("=" * 80)
    print("STARTING SEISMOFNO PHASE 5 (FINAL) & PHASE 6 BASELINE SUITE")
    print("Bilinear-Hysteretic SDOF on Held-Out-Earthquake Split")
    print("=" * 80)

    t_suite_start = time.time()

    for i, (cfg_file, expected_metrics) in enumerate(EXPERIMENT_CONFIGS, 1):
        if Path(expected_metrics).exists():
            print(f"\n[{i}/{len(EXPERIMENT_CONFIGS)}] Metrics already exist for {cfg_file} ({expected_metrics}). Skipping.")
            continue

        print(f"\n[{i}/{len(EXPERIMENT_CONFIGS)}] Running experiment configuration: {cfg_file}")
        cfg = load_config(cfg_file)
        train_fno(cfg)

    total_time = time.time() - t_suite_start
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETED IN {total_time / 60:.2f} MINUTES")
    print("Compiling consolidated benchmark table across FNO ablations and baselines...")
    print("=" * 80)

    generate_table_main()


if __name__ == "__main__":
    main()
