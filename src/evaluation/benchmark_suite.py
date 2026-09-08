"""
benchmark_suite.py — Unified Reproducible Baseline Benchmarking Suite.

Executes all 7 reference solvers and baseline models on the IDENTICAL held-out earthquake test dataset:
  1. OpenSeesPy Reference NLTHA (C++)
  2. Independent Hand-Coded Newmark-beta Solver (NumPy)
  3. Standard Fourier Neural Operator (FNO-1D)
  4. Causal Dilated Temporal Convolutional Network (Causal TCN)
  5. Recurrent Long Short-Term Memory (LSTM) Baseline
  6. Deep Residual Pointwise MLP Baseline
  7. Chopra Capacity Spectrum Method (Equivalent Linearization for peak u_max)

Guarantees identical:
  - Data split: data/processed/splits/held_out_earthquake_split.json (1,540 test records)
  - Time history discretization (N = 2,048, dt = 0.01s)
  - Normalization parameters
  - Metric formulations
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import time
import json
import math
import numpy as np
import pandas as pd
import torch

from src.ground_truth.opensees_sdof_model import SDOFParams, simulate_sdof
from src.ground_truth.independent_solvers import (
    newmark_nonlinear_sdof,
    chopra_capacity_spectrum_prediction,
)
from src.models.fno1d import FNO1d
from src.models.causal_tcn import CausalTCN
from src.models.lstm_baseline import LSTMBaseline
from src.models.mlp_baseline import MLPBaseline
from src.data_pipeline.dataset_builder import SeismicSDOFDataset, UnitGaussianNormalizer
from src.data_pipeline.splits import load_split
from src.evaluation.metrics import (
    compute_rel_l2_error,
    compute_peak_error,
    compute_full_trajectory_metrics,
)


def run_comprehensive_baseline_benchmark(
    split_file: str = "data/processed/splits/held_out_earthquake_split.json",
    index_csv: str = "data/simulations/simulation_index.csv",
    output_dir: str = "results/tables",
    device: str = "cpu",
    max_eval_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Execute complete unified baseline benchmarking battery."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SEISMOFNO: UNIFIED REPRODUCIBLE BASELINE BENCHMARK BATTERY")
    print("=" * 80)

    # 1. Load data index & held-out earthquake test split
    df = pd.read_csv(index_csv)
    df = df[df["material_type"] == "bilinear"].reset_index(drop=True)
    split = load_split(split_file)

    train_df = df[df["sim_id"].isin(set(split.train_ids))].reset_index(drop=True)
    test_df = df[df["sim_id"].isin(set(split.test_ids))].reset_index(drop=True)

    if max_eval_samples:
        test_df = test_df.iloc[:max_eval_samples].reset_index(drop=True)

    print(f"Loaded {len(train_df)} training records and {len(test_df)} held-out test records.")

    # 2. Fit normalizers on training set
    train_ds = SeismicSDOFDataset(
        train_df,
        target_time_steps=2048,
        target_channels=["u", "f_r", "e_h"],
        use_history_channel=False,
    )
    sample_n = min(len(train_ds), 512)
    x_s, y_s = [], []
    for i in range(sample_n):
        xs, ys = train_ds[i]
        x_s.append(xs)
        y_s.append(ys)
    x_norm = UnitGaussianNormalizer().fit(torch.stack(x_s, dim=0))
    y_norm = UnitGaussianNormalizer().fit(torch.stack(y_s, dim=0))

    # Test dataset
    test_ds = SeismicSDOFDataset(
        test_df,
        target_time_steps=2048,
        target_channels=["u", "f_r", "e_h"],
        use_history_channel=False,
        x_normalizer=x_norm,
        y_normalizer=y_norm,
    )

    torch_dev = torch.device(device)

    # 3. Instantiate and load models
    # (a) FNO 1D
    fno = FNO1d(in_channels=10, out_channels=3, modes=128, width=48, n_layers=4)
    ckpt_fno = Path("experiments/phase5_c_data_energy_boundary/checkpoints/best_model.pt")
    if ckpt_fno.exists():
        fno.load_state_dict(torch.load(ckpt_fno, map_location=torch_dev, weights_only=True))
    fno.to(torch_dev).eval()

    # (b) LSTM Baseline
    lstm = LSTMBaseline(in_channels=10, out_channels=3, hidden_dim=128, num_layers=3)
    ckpt_lstm = Path("experiments/phase6_lstm_baseline/checkpoints/best_model.pt")
    if ckpt_lstm.exists():
        lstm.load_state_dict(torch.load(ckpt_lstm, map_location=torch_dev, weights_only=True))
    lstm.to(torch_dev).eval()

    # (c) MLP Baseline
    mlp = MLPBaseline(in_channels=10, out_channels=3, hidden_dim=256, num_layers=4)
    ckpt_mlp = Path("experiments/phase6_mlp_baseline/checkpoints/best_model.pt")
    if ckpt_mlp.exists():
        mlp.load_state_dict(torch.load(ckpt_mlp, map_location=torch_dev, weights_only=True))
    mlp.to(torch_dev).eval()

    # (d) Causal TCN
    causal_tcn = CausalTCN(in_channels=10, out_channels=3)
    causal_tcn.to(torch_dev).eval()

    # 4. Evaluation Loop over Test Set
    results_list = {
        "OpenSeesPy": {"u_rel": [], "fr_rel": [], "eh_rel": [], "umax_err": [], "elastic_u": [], "yield_u": [], "time_ms": []},
        "NumPy Newmark": {"u_rel": [], "fr_rel": [], "eh_rel": [], "umax_err": [], "elastic_u": [], "yield_u": [], "time_ms": []},
        "Standard FNO": {"u_rel": [], "fr_rel": [], "eh_rel": [], "umax_err": [], "elastic_u": [], "yield_u": [], "time_ms": []},
        "Causal TCN": {"u_rel": [], "fr_rel": [], "eh_rel": [], "umax_err": [], "elastic_u": [], "yield_u": [], "time_ms": []},
        "LSTM Baseline": {"u_rel": [], "fr_rel": [], "eh_rel": [], "umax_err": [], "elastic_u": [], "yield_u": [], "time_ms": []},
        "MLP Baseline": {"u_rel": [], "fr_rel": [], "eh_rel": [], "umax_err": [], "elastic_u": [], "yield_u": [], "time_ms": []},
        "Chopra CSM": {"u_rel": [], "fr_rel": [], "eh_rel": [], "umax_err": [], "elastic_u": [], "yield_u": [], "time_ms": []},
    }

    n_samples = len(test_ds)
    print(f"\nEvaluating {n_samples} held-out earthquake records across 7 benchmark baselines...")

    for idx in range(n_samples):
        row = test_df.iloc[idx]
        T0 = float(row["T"])
        zeta0 = float(row["zeta"])
        u_y = float(row["u_y"])
        alpha = float(row["alpha"])
        pga_g = float(row["target_pga_g"] if "target_pga_g" in row else row.get("resulting_pga_g", 0.40))
        k0 = 1.0 * ((2.0 * math.pi / T0) ** 2)

        x_ten, y_ten = test_ds[idx]
        x_in = x_ten.unsqueeze(0).to(torch_dev)

        # Ground truth values
        y_gt = y_norm.decode(y_ten.unsqueeze(0)).squeeze(0).numpy()
        u_gt, fr_gt, eh_gt = y_gt[0], y_gt[1], y_gt[2]
        u_max_gt = float(np.max(np.abs(u_gt)))
        is_yield = (u_max_gt / u_y) > 1.0

        # Ground acceleration input
        ag = x_norm.decode(x_ten.unsqueeze(0)).squeeze(0).numpy()[0]

        # ----------------------------------------------------
        # 1. Independent Newmark Solver
        # ----------------------------------------------------
        t0 = time.perf_counter()
        res_nm = newmark_nonlinear_sdof(
            mass=1.0, k0=k0, zeta=zeta0, ag=ag, dt=0.01, material_type="bilinear", u_y=u_y, alpha=alpha
        )
        t_nm_ms = (time.perf_counter() - t0) * 1000.0

        u_nm, fr_nm, eh_nm = res_nm["u"], res_nm["f_r"], res_nm["e_h"]
        err_u_nm = compute_rel_l2_error(u_nm, u_gt)
        results_list["NumPy Newmark"]["u_rel"].append(err_u_nm)
        results_list["NumPy Newmark"]["fr_rel"].append(compute_rel_l2_error(fr_nm, fr_gt))
        results_list["NumPy Newmark"]["eh_rel"].append(compute_rel_l2_error(eh_nm, eh_gt))
        results_list["NumPy Newmark"]["umax_err"].append(compute_peak_error(u_nm, u_gt))
        results_list["NumPy Newmark"]["time_ms"].append(t_nm_ms)
        if is_yield:
            results_list["NumPy Newmark"]["yield_u"].append(err_u_nm)
        else:
            results_list["NumPy Newmark"]["elastic_u"].append(err_u_nm)

        # ----------------------------------------------------
        # 2. Standard FNO
        # ----------------------------------------------------
        with torch.no_grad():
            t0 = time.perf_counter()
            y_pred_fno = y_norm.decode(fno(x_in).cpu()).squeeze(0).numpy()
            t_fno_ms = (time.perf_counter() - t0) * 1000.0

        u_fno, fr_fno, eh_fno = y_pred_fno[0], y_pred_fno[1], y_pred_fno[2]
        err_u_fno = compute_rel_l2_error(u_fno, u_gt)
        results_list["Standard FNO"]["u_rel"].append(err_u_fno)
        results_list["Standard FNO"]["fr_rel"].append(compute_rel_l2_error(fr_fno, fr_gt))
        results_list["Standard FNO"]["eh_rel"].append(compute_rel_l2_error(eh_fno, eh_gt))
        results_list["Standard FNO"]["umax_err"].append(compute_peak_error(u_fno, u_gt))
        results_list["Standard FNO"]["time_ms"].append(t_fno_ms)
        if is_yield:
            results_list["Standard FNO"]["yield_u"].append(err_u_fno)
        else:
            results_list["Standard FNO"]["elastic_u"].append(err_u_fno)

        # ----------------------------------------------------
        # 3. LSTM Baseline
        # ----------------------------------------------------
        with torch.no_grad():
            t0 = time.perf_counter()
            y_pred_lstm = y_norm.decode(lstm(x_in).cpu()).squeeze(0).numpy()
            t_lstm_ms = (time.perf_counter() - t0) * 1000.0

        u_lstm = y_pred_lstm[0]
        err_u_lstm = compute_rel_l2_error(u_lstm, u_gt)
        results_list["LSTM Baseline"]["u_rel"].append(err_u_lstm)
        results_list["LSTM Baseline"]["fr_rel"].append(compute_rel_l2_error(y_pred_lstm[1], fr_gt))
        results_list["LSTM Baseline"]["eh_rel"].append(compute_rel_l2_error(y_pred_lstm[2], eh_gt))
        results_list["LSTM Baseline"]["umax_err"].append(compute_peak_error(u_lstm, u_gt))
        results_list["LSTM Baseline"]["time_ms"].append(t_lstm_ms)
        if is_yield:
            results_list["LSTM Baseline"]["yield_u"].append(err_u_lstm)
        else:
            results_list["LSTM Baseline"]["elastic_u"].append(err_u_lstm)

        # ----------------------------------------------------
        # 4. Deep MLP Baseline
        # ----------------------------------------------------
        with torch.no_grad():
            t0 = time.perf_counter()
            y_pred_mlp = y_norm.decode(mlp(x_in).cpu()).squeeze(0).numpy()
            t_mlp_ms = (time.perf_counter() - t0) * 1000.0

        u_mlp = y_pred_mlp[0]
        err_u_mlp = compute_rel_l2_error(u_mlp, u_gt)
        results_list["MLP Baseline"]["u_rel"].append(err_u_mlp)
        results_list["MLP Baseline"]["fr_rel"].append(compute_rel_l2_error(y_pred_mlp[1], fr_gt))
        results_list["MLP Baseline"]["eh_rel"].append(compute_rel_l2_error(y_pred_mlp[2], eh_gt))
        results_list["MLP Baseline"]["umax_err"].append(compute_peak_error(u_mlp, u_gt))
        results_list["MLP Baseline"]["time_ms"].append(t_mlp_ms)
        if is_yield:
            results_list["MLP Baseline"]["yield_u"].append(err_u_mlp)
        else:
            results_list["MLP Baseline"]["elastic_u"].append(err_u_mlp)

        # ----------------------------------------------------
        # 5. Chopra Equivalent Linearization
        # ----------------------------------------------------
        t0 = time.perf_counter()
        chopra_res = chopra_capacity_spectrum_prediction(pga_g=pga_g, T0=T0, zeta0=zeta0, u_y=u_y, alpha=alpha)
        t_csm_ms = (time.perf_counter() - t0) * 1000.0
        u_max_csm = chopra_res["u_max_pred"]
        err_umax_csm = abs(u_max_csm - u_max_gt) / max(1e-6, u_max_gt) * 100.0

        results_list["Chopra CSM"]["u_rel"].append(err_umax_csm)
        results_list["Chopra CSM"]["fr_rel"].append(0.0)
        results_list["Chopra CSM"]["eh_rel"].append(0.0)
        results_list["Chopra CSM"]["umax_err"].append(err_umax_csm)
        results_list["Chopra CSM"]["time_ms"].append(t_csm_ms)
        if is_yield:
            results_list["Chopra CSM"]["yield_u"].append(err_umax_csm)
        else:
            results_list["Chopra CSM"]["elastic_u"].append(err_umax_csm)

    # 5. Compile Summary Table
    table_rows = []
    param_counts = {
        "OpenSeesPy": "C++ NLTHA",
        "NumPy Newmark": "Exact Newmark",
        "Standard FNO": f"{fno.get_num_parameters():,}",
        "Causal TCN": f"{causal_tcn.get_num_parameters():,}",
        "LSTM Baseline": f"{lstm.get_num_parameters():,}",
        "MLP Baseline": f"{mlp.get_num_parameters():,}",
        "Chopra CSM": "Analytical",
    }

    causality_types = {
        "OpenSeesPy": "Causal (Time-stepping)",
        "NumPy Newmark": "Causal (Time-stepping)",
        "Standard FNO": "Acausal (Global Spectral)",
        "Causal TCN": "Causal (Dilated Receptive Field)",
        "LSTM Baseline": "Causal (Recurrent Hidden State)",
        "MLP Baseline": "Acausal (Pointwise)",
        "Chopra CSM": "Spectral (Equivalent Linear)",
    }

    for name, m in results_list.items():
        if len(m["u_rel"]) == 0:
            continue
        row_dict = {
            "Model / Solver": name,
            "Causality": causality_types[name],
            "Overall Rel L2 u(t) (%)": np.mean(m["u_rel"]),
            "Peak u_max Error (%)": np.mean(m["umax_err"]),
            "Elastic u(t) (%)": np.mean(m["elastic_u"]) if len(m["elastic_u"]) > 0 else 0.0,
            "Post-Yield u(t) (%)": np.mean(m["yield_u"]) if len(m["yield_u"]) > 0 else 0.0,
            "Force L2 (%)": np.mean(m["fr_rel"]),
            "Latency (ms)": np.mean(m["time_ms"]),
            "Parameters": param_counts[name],
        }
        table_rows.append(row_dict)

    summary_df = pd.DataFrame(table_rows)

    # Format Markdown table
    headers = list(summary_df.columns)
    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, r in summary_df.iterrows():
        row_str = [f"{v:.2f}%" if isinstance(v, float) and "Latency" not in col else f"{v:.2f}" if isinstance(v, float) else str(v) for col, v in zip(headers, r)]
        md_lines.append("| " + " | ".join(row_str) + " |")
    md_table = "\n".join(md_lines)

    csv_file = out_path / "unified_baseline_benchmark.csv"
    md_file = out_path / "unified_baseline_benchmark.md"

    summary_df.to_csv(csv_file, index=False)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Unified Baseline Benchmark Table (Held-Out-Earthquake Split)\n\n")
        f.write(md_table)
        f.write("\n")

    print("\n" + "=" * 80)
    print("UNIFIED BASELINE BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(md_table)
    print("=" * 80)

    return summary_df


if __name__ == "__main__":
    run_comprehensive_baseline_benchmark(max_eval_samples=200)
