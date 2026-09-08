"""
api/tests/test_inference_adapter.py — Unit Tests for SeismoFNOInferenceAdapter.
"""

import numpy as np
import pytest

from inference.adapter import SeismoFNOInferenceAdapter, InferenceOutput


def test_adapter_metadata_and_loading():
    """Verify adapter initializes and reports accurate metadata."""
    adapter = SeismoFNOInferenceAdapter()
    meta = adapter.metadata()
    assert "loaded" in meta
    assert "device" in meta
    assert "checkpoint_path" in meta


def test_adapter_predict_outputs_and_shapes():
    """Verify forward inference returns all required physical channels and derived quantities."""
    adapter = SeismoFNOInferenceAdapter()
    scenario = {
        "T0": 0.50,
        "damping": 0.05,
        "yield_displacement": 0.010,
        "alpha": 0.05,
        "material_type": "bilinear",
        "mass": 1.0,
        "pga_g": 0.40,
    }

    n_steps = 2048
    dt = 0.01
    time_arr = np.linspace(0, (n_steps - 1) * dt, n_steps)
    ag = np.sin(2 * np.pi * 1.5 * time_arr) * 9.80665 * 0.4  # m/s^2

    out: InferenceOutput = adapter.predict(scenario=scenario, ag=ag, dt=dt, n_steps=n_steps)

    # 1. Shape verifications
    assert len(out.time) == n_steps
    assert len(out.ag) == n_steps
    assert len(out.u) == n_steps
    assert len(out.v) == n_steps
    assert len(out.fr) == n_steps
    assert len(out.up) == n_steps
    assert len(out.eh) == n_steps

    # 2. Latency measurement must be positive
    assert out.latency_ms > 0.0

    # 3. Deterministically derived velocity: v ≈ du/dt
    du_dt = np.gradient(out.u, dt)
    assert np.allclose(out.v, du_dt, atol=1e-4)

    # 4. Plastic displacement: up = u - FR/k0
    omega_n = 2.0 * np.pi / 0.50
    k0 = 1.0 * (omega_n ** 2)
    expected_up = out.u - (out.fr / k0)
    assert np.allclose(out.up, expected_up, atol=1e-3)


def test_adapter_elastic_plastic_displacement_zero():
    """Verify plastic displacement is zero in linear elastic mode."""
    adapter = SeismoFNOInferenceAdapter()
    scenario = {
        "T0": 0.50,
        "damping": 0.05,
        "yield_displacement": 0.050,
        "alpha": 0.05,
        "material_type": "elastic",
        "mass": 1.0,
        "pga_g": 0.10,
    }
    n_steps = 1024
    dt = 0.01
    ag = np.zeros(n_steps, dtype=np.float32)

    out = adapter.predict(scenario=scenario, ag=ag, dt=dt, n_steps=n_steps)
    assert np.all(out.up == 0.0)


def test_adapter_invalid_input_raises():
    """Verify input validation catches negative period or invalid damping."""
    adapter = SeismoFNOInferenceAdapter()
    with pytest.raises(ValueError):
        adapter.predict(scenario={"T0": -1.0}, ag=np.zeros(100))
