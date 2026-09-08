"""
test_speed_benchmark.py — Unit Tests for Speed Benchmarking Module.

Verifies:
  1. OpenSeesPy SDOF and MDOF pure timing functions return valid statistics.
  2. Neural operator inference benchmark measures latencies and computes speedup distributions.
  3. Regional 10k risk assessment projection computes consistent mathematical scalings.
"""

import pytest
import torch

from src.evaluation.speed_benchmark import (
    benchmark_opensees_sdof_nltha,
    benchmark_opensees_mdof_nltha,
    benchmark_model_inference,
    compute_risk_assessment_projection,
)
from src.models.fno1d import FNO1d


def test_opensees_sdof_benchmark_quick():
    res = benchmark_opensees_sdof_nltha(n_runs=3, n_steps=512, dt=0.01)
    assert res["mean_s"] > 0
    assert res["throughput_hz"] > 0
    assert len(res["raw_latencies"]) == 3


def test_opensees_mdof_benchmark_quick():
    res = benchmark_opensees_mdof_nltha(n_stories=3, n_runs=2, n_steps=512, dt=0.01)
    assert res["mean_s"] > 0
    assert res["throughput_hz"] > 0
    assert len(res["raw_latencies"]) == 2


def test_fno_inference_benchmark_quick():
    device = torch.device("cpu")
    model = FNO1d(in_channels=10, out_channels=3, modes=16, width=16, n_layers=2)
    res = benchmark_model_inference(
        model=model,
        model_name="TestFNO",
        device=device,
        input_shape=(1, 10, 512),
        batch_sizes=[1, 4],
        n_runs=3,
        n_warmup=1,
        opensees_mean_s=0.05,
    )
    assert res[0]["speedup_mean"] > 0
    # NOTE: Batch scaling is non-monotonic on CPU/MPS — only verify throughput is positive
    assert res[1]["throughput_hz"] > 0
    assert res[1]["speedup_mean"] > 0


def test_risk_projection_calculation():
    fno_results = [
        {"per_record_mean_ms": 10.0},
        {"per_record_mean_ms": 0.1},
    ]
    proj = compute_risk_assessment_projection(
        opensees_mean_s=0.05,
        fno_results=fno_results,
        n_portfolio_records=10000,
    )
    assert proj["opensees_total_s"] == 500.0
    assert proj["fno_batched_total_s"] == 1.0
    assert proj["realized_speedup_batched"] == 500.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
