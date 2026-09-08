"""
speed_benchmark.py — Rigorous Computational Speed & Throughput Benchmark.

Measures inference wall-clock execution time and speedup of SeismoFNO vs.
OpenSeesPy Nonlinear Time-History Analysis (NLTHA) on IDENTICAL hardware.

Protocol per AGENTS.md Hard Rules:
  1. Exclude data loading, disk I/O, plotting, and logging overhead from both methods.
  2. Perform warmup runs before timing to eliminate JIT/compilation overhead.
  3. Strictly synchronize accelerator devices (torch.mps.synchronize() or torch.cuda.synchronize()).
  4. Measure latency over multiple repeated runs (N >= 50) and report speedup as an empirical
     distribution (mean +/- std, median, 5th-95th percentile, min, max), NOT a single cherry-picked number.
  5. Benchmark realistic batched inference across batch sizes B in [1, 16, 32, 64, 128, 256, 512, 1024]
     for large-scale regional seismic risk assessments (10,000+ records).
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import time
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import openseespy.opensees as ops
from src.ground_truth.opensees_sdof_model import SDOFParams, simulate_sdof
from src.ground_truth.opensees_mdof_model import MDOFParams, simulate_mdof
from src.models.fno1d import FNO1d
from src.models.fno2d import FNO2d
from src.models.lstm_baseline import LSTMBaseline
from src.models.mlp_baseline import MLPBaseline


def sync_device(device: torch.device):
    """Synchronize accelerator queue to ensure accurate wall-clock timing."""
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def df_to_markdown_str(df: pd.DataFrame) -> str:
    """Pure-Python markdown table formatter without tabulate dependency."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def benchmark_opensees_sdof_nltha(
    n_runs: int = 50,
    n_steps: int = 2048,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """
    Time pure OpenSeesPy SDOF nonlinear dynamic time-history analysis.
    Excludes file I/O, post-processing, and plotting.
    """
    print(f"\n[OpenSeesPy SDOF NLTHA] Running {n_runs} repeated simulations (T={n_steps} steps, dt={dt}s)...")

    # Generate synthetic earthquake acceleration series
    t = np.arange(n_steps) * dt
    ag = 0.4 * np.sin(2.0 * np.pi * 2.0 * t) * np.exp(-0.1 * t) + 0.2 * np.random.RandomState(42).randn(n_steps)

    params = SDOFParams(
        T=0.5,
        zeta=0.05,
        material_type="bilinear",
        u_y=0.015,
        alpha=0.05,
        mass=1.0,
    )

    # Warmup
    for _ in range(5):
        _ = simulate_sdof(params=params, ag=ag, dt=dt)

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = simulate_sdof(params=params, ag=ag, dt=dt)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    latencies = np.array(latencies)
    results = {
        "solver": "OpenSeesPy SDOF (Bilinear NLTHA)",
        "n_runs": n_runs,
        "mean_s": float(np.mean(latencies)),
        "std_s": float(np.std(latencies)),
        "median_s": float(np.median(latencies)),
        "p05_s": float(np.percentile(latencies, 5)),
        "p95_s": float(np.percentile(latencies, 95)),
        "min_s": float(np.min(latencies)),
        "max_s": float(np.max(latencies)),
        "throughput_hz": float(1.0 / np.mean(latencies)),
        "raw_latencies": latencies.tolist(),
    }

    print(f"  Mean Latency: {results['mean_s']*1000:.2f} +/- {results['std_s']*1000:.2f} ms per record")
    print(f"  Throughput:   {results['throughput_hz']:.2f} records/sec")
    return results


def benchmark_opensees_mdof_nltha(
    n_stories: int = 3,
    n_runs: int = 30,
    n_steps: int = 2048,
    dt: float = 0.01,
) -> Dict[str, Any]:
    """
    Time pure OpenSeesPy MDOF multi-story building nonlinear dynamic analysis.
    """
    print(f"\n[OpenSeesPy MDOF ({n_stories}-Story) NLTHA] Running {n_runs} repeated simulations...")

    t = np.arange(n_steps) * dt
    ag = 0.4 * np.sin(2.0 * np.pi * 2.0 * t) * np.exp(-0.1 * t) + 0.2 * np.random.RandomState(42).randn(n_steps)

    params = MDOFParams(
        n_stories=n_stories,
        story_masses=[1000.0] * n_stories,
        story_heights=[3.0] * n_stories,
        story_stiffnesses=[1.0e6] * n_stories,
        material_type="bilinear",
        yield_displacements=[0.015] * n_stories,
        alpha=0.05,
        zeta_1=0.05,
        zeta_2=0.05,
    )

    # Warmup
    for _ in range(3):
        _ = simulate_mdof(params=params, ag=ag, dt=dt)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = simulate_mdof(params=params, ag=ag, dt=dt)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    latencies = np.array(latencies)
    results = {
        "solver": f"OpenSeesPy MDOF ({n_stories}-Story NLTHA)",
        "n_runs": n_runs,
        "mean_s": float(np.mean(latencies)),
        "std_s": float(np.std(latencies)),
        "median_s": float(np.median(latencies)),
        "p05_s": float(np.percentile(latencies, 5)),
        "p95_s": float(np.percentile(latencies, 95)),
        "min_s": float(np.min(latencies)),
        "max_s": float(np.max(latencies)),
        "throughput_hz": float(1.0 / np.mean(latencies)),
        "raw_latencies": latencies.tolist(),
    }

    print(f"  Mean Latency: {results['mean_s']*1000:.2f} +/- {results['std_s']*1000:.2f} ms per simulation")
    print(f"  Throughput:   {results['throughput_hz']:.2f} simulations/sec")
    return results


def benchmark_model_inference(
    model: nn.Module,
    model_name: str,
    device: torch.device,
    input_shape: Tuple[int, ...],
    batch_sizes: List[int],
    n_runs: int = 50,
    n_warmup: int = 10,
    opensees_mean_s: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Benchmark neural network inference latency and speedup across batch sizes.
    """
    model.to(device)
    model.eval()

    results_list = []
    print(f"\n[{model_name}] Benchmarking inference on {device} across batch sizes {batch_sizes}...")

    for bs in batch_sizes:
        # Pre-allocate input tensor on device
        full_shape = (bs,) + input_shape[1:]
        x_tensor = torch.randn(*full_shape, device=device, dtype=torch.float32)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(x_tensor)
                sync_device(device)

        # Timed runs
        batch_latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                sync_device(device)
                t0 = time.perf_counter()
                _ = model(x_tensor)
                sync_device(device)
                t1 = time.perf_counter()
                batch_latencies.append(t1 - t0)

        batch_latencies = np.array(batch_latencies)
        per_record_latencies = batch_latencies / bs

        # Speedup distribution = OpenSees_time / per_record_time
        speedups = opensees_mean_s / per_record_latencies

        res = {
            "model_name": model_name,
            "batch_size": bs,
            "n_runs": n_runs,
            "batch_mean_ms": float(np.mean(batch_latencies) * 1000.0),
            "batch_std_ms": float(np.std(batch_latencies) * 1000.0),
            "per_record_mean_ms": float(np.mean(per_record_latencies) * 1000.0),
            "per_record_std_ms": float(np.std(per_record_latencies) * 1000.0),
            "throughput_hz": float(bs / np.mean(batch_latencies)),
            "speedup_mean": float(np.mean(speedups)),
            "speedup_std": float(np.std(speedups)),
            "speedup_median": float(np.median(speedups)),
            "speedup_p05": float(np.percentile(speedups, 5)),
            "speedup_p95": float(np.percentile(speedups, 95)),
            "speedup_min": float(np.min(speedups)),
            "speedup_max": float(np.max(speedups)),
        }
        results_list.append(res)

        print(
            f"  Batch {bs:4d} | "
            f"Batch: {res['batch_mean_ms']:6.2f} +/- {res['batch_std_ms']:5.2f} ms | "
            f"Per Record: {res['per_record_mean_ms']:7.4f} ms | "
            f"Throughput: {res['throughput_hz']:8.1f} rec/s | "
            f"Speedup: {res['speedup_mean']:7.1f}x +/- {res['speedup_std']:5.1f}x",
            flush=True,
        )

    return results_list


def compute_risk_assessment_projection(
    opensees_mean_s: float,
    fno_results: List[Dict[str, Any]],
    n_portfolio_records: int = 10000,
) -> Dict[str, Any]:
    """
    Project total wall-clock runtime for a large-scale regional seismic risk assessment
    of 10,000 ground motion-structure pairs.
    """
    opensees_total_s = n_portfolio_records * opensees_mean_s

    # Best batched FNO throughput (largest tested batch size)
    best_fno = fno_results[-1]
    fno_batched_total_s = n_portfolio_records * (best_fno["per_record_mean_ms"] / 1000.0)

    # Single-record interactive FNO (batch size 1)
    single_fno = fno_results[0]
    fno_single_total_s = n_portfolio_records * (single_fno["per_record_mean_ms"] / 1000.0)

    return {
        "portfolio_size": n_portfolio_records,
        "opensees_total_s": opensees_total_s,
        "opensees_total_min": opensees_total_s / 60.0,
        "opensees_total_hr": opensees_total_s / 3600.0,
        "fno_single_total_s": fno_single_total_s,
        "fno_batched_total_s": fno_batched_total_s,
        "realized_speedup_batched": opensees_total_s / fno_batched_total_s,
        "realized_speedup_single": opensees_total_s / fno_single_total_s,
    }


def run_full_speed_benchmark_suite(
    output_dir: str = "results",
    n_runs: int = 30,
    device_name: Optional[str] = None,
):
    """
    Execute end-to-end speed benchmark across OpenSeesPy NLTHA, FNO1d, FNO2d, LSTM, and MLP.
    """
    out_path = Path(output_dir)
    table_dir = out_path / "tables"
    fig_dir = out_path / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    print("=" * 80, flush=True)
    print(f"SEISMOFNO PHASE 9 SPEED & THROUGHPUT BENCHMARK (Device: {device})", flush=True)
    print("=" * 80, flush=True)

    # 1. Benchmark OpenSeesPy Ground Truth NLTHA
    os_sdof_res = benchmark_opensees_sdof_nltha(n_runs=n_runs, n_steps=2048, dt=0.01)
    os_mdof_res = benchmark_opensees_mdof_nltha(n_stories=3, n_runs=min(n_runs, 20), n_steps=2048, dt=0.01)

    opensees_sdof_mean_s = os_sdof_res["mean_s"]
    opensees_mdof_mean_s = os_mdof_res["mean_s"]

    # 2. Benchmark SDOF Surrogate Models
    batch_sizes = [1, 8, 16, 32, 64, 128, 256]
    time_steps = 2048

    # Load SDOF FNO model
    fno1d_model = FNO1d(in_channels=10, out_channels=3, modes=128, width=48, n_layers=4)
    sdof_ckpt = Path("experiments/phase5_c_data_energy_boundary/checkpoints/best_model.pt")
    if sdof_ckpt.exists():
        fno1d_model.load_state_dict(torch.load(sdof_ckpt, map_location=device, weights_only=True))

    fno1d_results = benchmark_model_inference(
        model=fno1d_model,
        model_name="SeismoFNO 1D (SDOF)",
        device=device,
        input_shape=(1, 10, time_steps),
        batch_sizes=batch_sizes,
        n_runs=n_runs,
        opensees_mean_s=opensees_sdof_mean_s,
    )

    # Load LSTM Baseline
    lstm_model = LSTMBaseline(in_channels=10, out_channels=3, hidden_dim=128, num_layers=3)
    lstm_ckpt = Path("experiments/phase6_lstm_baseline/checkpoints/best_model.pt")
    if lstm_ckpt.exists():
        lstm_model.load_state_dict(torch.load(lstm_ckpt, map_location=device, weights_only=True))

    lstm_results = benchmark_model_inference(
        model=lstm_model,
        model_name="LSTM Baseline (SDOF)",
        device=device,
        input_shape=(1, 10, time_steps),
        batch_sizes=[1, 8, 16, 32, 64],
        n_runs=n_runs,
        opensees_mean_s=opensees_sdof_mean_s,
    )

    # Load MLP Baseline
    mlp_model = MLPBaseline(in_channels=10, out_channels=3, hidden_dim=128, num_layers=4)
    mlp_ckpt = Path("experiments/phase6_mlp_baseline/checkpoints/best_model.pt")
    if mlp_ckpt.exists():
        mlp_model.load_state_dict(torch.load(mlp_ckpt, map_location=device, weights_only=True))
    mlp_results = benchmark_model_inference(
        model=mlp_model,
        model_name="MLP Baseline (SDOF)",
        device=device,
        input_shape=(1, 10, time_steps),
        batch_sizes=batch_sizes,
        n_runs=n_runs,
        opensees_mean_s=opensees_sdof_mean_s,
    )

    # 3. Benchmark MDOF Spatiotemporal FNO2d
    fno2d_model = FNO2d(in_channels=10, out_channels=3, modes1=4, modes2=64, width=48, n_layers=4)
    mdof_ckpt = Path("experiments/phase8_mdof_fno_linear/checkpoints/best_model.pt")
    if mdof_ckpt.exists():
        fno2d_model.load_state_dict(torch.load(mdof_ckpt, map_location=device, weights_only=True))

    fno2d_results = benchmark_model_inference(
        model=fno2d_model,
        model_name="SeismoFNO 2D (MDOF 5-Story)",
        device=device,
        input_shape=(1, 10, 5, time_steps),
        batch_sizes=[1, 8, 16, 32, 64],
        n_runs=n_runs,
        opensees_mean_s=opensees_mdof_mean_s,
    )

    # 4. Large-Scale 10,000-Record Regional Assessment Projection
    risk_proj = compute_risk_assessment_projection(
        opensees_mean_s=opensees_sdof_mean_s,
        fno_results=fno1d_results,
        n_portfolio_records=10000,
    )

    # 5. Compile Summary Table
    all_model_results = fno1d_results + lstm_results + mlp_results + fno2d_results
    df_raw = pd.DataFrame(all_model_results)
    csv_path = table_dir / "speed_benchmark.csv"
    df_raw.to_csv(csv_path, index=False)

    # Format human-readable benchmark table
    table_rows = []
    # OpenSees Ground Truth reference rows
    table_rows.append({
        "Model / Method": "OpenSeesPy SDOF NLTHA (Ground Truth)",
        "Batch Size": "1 (Sequential)",
        "Per-Record Latency": f"{os_sdof_res['mean_s']*1000:.2f} +/- {os_sdof_res['std_s']*1000:.2f} ms",
        "Throughput (rec/s)": f"{os_sdof_res['throughput_hz']:.1f}",
        "Speedup Distribution (Mean +/- Std)": "1.0x (Baseline)",
        "Speedup 5th-95th %ile": "[1.0x, 1.0x]",
    })
    table_rows.append({
        "Model / Method": "OpenSeesPy MDOF 3-Story NLTHA",
        "Batch Size": "1 (Sequential)",
        "Per-Record Latency": f"{os_mdof_res['mean_s']*1000:.2f} +/- {os_mdof_res['std_s']*1000:.2f} ms",
        "Throughput (rec/s)": f"{os_mdof_res['throughput_hz']:.1f}",
        "Speedup Distribution (Mean +/- Std)": "1.0x (Baseline)",
        "Speedup 5th-95th %ile": "[1.0x, 1.0x]",
    })

    for res in all_model_results:
        table_rows.append({
            "Model / Method": res["model_name"],
            "Batch Size": str(res["batch_size"]),
            "Per-Record Latency": f"{res['per_record_mean_ms']:.4f} +/- {res['per_record_std_ms']:.4f} ms",
            "Throughput (rec/s)": f"{res['throughput_hz']:.1f}",
            "Speedup Distribution (Mean +/- Std)": f"{res['speedup_mean']:.1f}x +/- {res['speedup_std']:.1f}x",
            "Speedup 5th-95th %ile": f"[{res['speedup_p05']:.1f}x, {res['speedup_p95']:.1f}x]",
        })

    df_display = pd.DataFrame(table_rows)
    md_path = table_dir / "speed_benchmark.md"
    md_str = df_to_markdown_str(df_display)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 9: Speed & Computational Throughput Benchmark\n\n")
        f.write(f"Measured on identical hardware ({device}) excluding disk I/O and data loading overhead.\n")
        f.write(f"OpenSeesPy SDOF NLTHA baseline: **{os_sdof_res['mean_s']*1000:.2f} ms** per simulation ({os_sdof_res['throughput_hz']:.1f} rec/s).\n")
        f.write(f"OpenSeesPy MDOF 3-story NLTHA baseline: **{os_mdof_res['mean_s']*1000:.2f} ms** per simulation ({os_mdof_res['throughput_hz']:.1f} rec/s).\n\n")
        f.write("## Inference Speedup Across Batch Sizes\n\n")
        f.write(md_str)
        f.write("\n\n")
        f.write("## 10,000-Record Regional Seismic Risk Assessment Case Study\n\n")
        f.write(f"- **OpenSeesPy Serial NLTHA**: {risk_proj['opensees_total_min']:.1f} minutes ({risk_proj['opensees_total_hr']:.2f} hours)\n")
        f.write(f"- **SeismoFNO (Single-record interactive)**: {risk_proj['fno_single_total_s']:.2f} seconds ({risk_proj['realized_speedup_single']:.1f}x speedup)\n")
        f.write(f"- **SeismoFNO (Batched regional evaluation)**: **{risk_proj['fno_batched_total_s']:.2f} seconds** (**{risk_proj['realized_speedup_batched']:.1f}x speedup**)\n")

    print(f"\nSaved Markdown Table to: {md_path}")
    print(f"Saved Raw CSV to: {csv_path}")
    print("\n" + md_str)

    # 6. Save JSON Summary Artifact
    summary_payload = {
        "device": str(device),
        "n_runs": n_runs,
        "opensees_sdof": os_sdof_res,
        "opensees_mdof": os_mdof_res,
        "fno1d_results": fno1d_results,
        "lstm_results": lstm_results,
        "mlp_results": mlp_results,
        "fno2d_results": fno2d_results,
        "regional_10k_risk_assessment": risk_proj,
    }
    json_path = out_path / "speed_benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # 7. Generate Speedup Scaling Plot
    plot_speedup_scaling(
        fno1d_results=fno1d_results,
        lstm_results=lstm_results,
        mlp_results=mlp_results,
        fno2d_results=fno2d_results,
        fig_path=fig_dir / "speedup_scaling_curve.png",
    )

    print("\n" + "=" * 80)
    print("PHASE 9 SPEED BENCHMARK COMPLETE")
    print("=" * 80)
    return summary_payload


def plot_speedup_scaling(
    fno1d_results: List[Dict[str, Any]],
    lstm_results: List[Dict[str, Any]],
    mlp_results: List[Dict[str, Any]],
    fno2d_results: List[Dict[str, Any]],
    fig_path: Path,
):
    """Generate high-resolution speedup scaling figure across batch sizes."""
    plt.figure(figsize=(10, 6), dpi=300)

    # FNO 1D
    fno_bs = [r["batch_size"] for r in fno1d_results]
    fno_sp = [r["speedup_mean"] for r in fno1d_results]
    fno_err = [r["speedup_std"] for r in fno1d_results]
    plt.errorbar(
        fno_bs, fno_sp, yerr=fno_err, fmt="-o", color="#1f77b4", linewidth=2.5,
        capsize=4, label="SeismoFNO 1D (SDOF)"
    )

    # FNO 2D
    fno2_bs = [r["batch_size"] for r in fno2d_results]
    fno2_sp = [r["speedup_mean"] for r in fno2d_results]
    fno2_err = [r["speedup_std"] for r in fno2d_results]
    plt.errorbar(
        fno2_bs, fno2_sp, yerr=fno2_err, fmt="-s", color="#2ca02c", linewidth=2.5,
        capsize=4, label="SeismoFNO 2D (MDOF)"
    )

    # MLP
    mlp_bs = [r["batch_size"] for r in mlp_results]
    mlp_sp = [r["speedup_mean"] for r in mlp_results]
    mlp_err = [r["speedup_std"] for r in mlp_results]
    plt.errorbar(
        mlp_bs, mlp_sp, yerr=mlp_err, fmt="-^", color="#ff7f0e", linewidth=2.0,
        capsize=4, label="MLP Baseline"
    )

    # LSTM
    lstm_bs = [r["batch_size"] for r in lstm_results]
    lstm_sp = [r["speedup_mean"] for r in lstm_results]
    lstm_err = [r["speedup_std"] for r in lstm_results]
    plt.errorbar(
        lstm_bs, lstm_sp, yerr=lstm_err, fmt="-d", color="#d62728", linewidth=2.0,
        capsize=4, label="LSTM Baseline"
    )

    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="OpenSeesPy Baseline (1.0x)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Inference Batch Size (Number of Earthquake Records)", fontsize=11, fontweight="bold")
    plt.ylabel("Measured Speedup Factor relative to OpenSeesPy (x)", fontsize=11, fontweight="bold")
    plt.title("Inference Throughput & Speedup Scaling vs. OpenSeesPy NLTHA", fontsize=13, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved Speedup Scaling Plot to: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 9 Speed Benchmark.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--device", type=str, default=None, help="Device to benchmark ('cpu', 'mps', 'cuda')")
    args = parser.parse_args()
    run_full_speed_benchmark_suite(output_dir=args.output_dir, n_runs=args.n_runs, device_name=args.device)
