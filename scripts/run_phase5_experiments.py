"""
run_phase5_experiments.py — Automated Execution of Phase 5 Ablation Suite.

Runs:
  (a) Data Loss Only
  (b) Data + Energy Consistency Loss
  (c) Data + Energy + Boundary Loss
  (d) Data + Energy + Boundary + Auxiliary History Channel

Then aggregates all results into results/tables/
"""

import sys
import time
from pathlib import Path
import subprocess

from src.training.train import train_fno, load_config
from scripts.generate_phase5_table import main as generate_table_main


CONFIGS = [
    "configs/phase5_a_data_only.yaml",
    "configs/phase5_b_data_energy.yaml",
    "configs/phase5_c_data_energy_boundary.yaml",
    "configs/phase5_d_history_augmented.yaml",
]


def main():
    print("=" * 80)
    print("STARTING SEISMOFNO PHASE 5 EXPERIMENTAL ABLATION SUITE")
    print("Bilinear-Hysteretic SDOF on Held-Out-Earthquake Split")
    print("=" * 80)

    t_suite_start = time.time()

    for i, cfg_file in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Running experiment configuration: {cfg_file}")
        cfg = load_config(cfg_file)
        train_fno(cfg)

    total_time = time.time() - t_suite_start
    print("\n" + "=" * 80)
    print(f"ALL 4 PHASE 5 EXPERIMENTS COMPLETED IN {total_time / 60:.2f} MINUTES")
    print("Generating consolidated ablation table...")
    print("=" * 80)

    generate_table_main()


if __name__ == "__main__":
    main()
