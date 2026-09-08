"""
api/tests/test_engineering_metrics.py — Unit Tests for Engineering Analysis Service.

Verifies mathematical formulas, SI and engineering unit consistency, boundary handling,
and exact separation between meters vs millimeters, m/s^2 vs g, and seconds vs samples.
"""

import numpy as np
import pytest

from api.services.engineering_metrics import compute_engineering_metrics, EngineeringMetricsSummary


def test_engineering_metrics_units_and_peaks():
    """Verify displacement, velocity, and acceleration unit conversions."""
    dt = 0.01
    n_pts = 1000
    time = np.linspace(0, (n_pts - 1) * dt, n_pts)

    # Known harmonic displacement: u(t) = 0.05 * sin(2*pi*f*t) [meters]
    f = 2.0
    u = 0.05 * np.sin(2 * np.pi * f * time)
    # v(t) = 0.05 * 2*pi*f * cos(2*pi*f*t) -> max v ≈ 0.6283 m/s
    v = np.gradient(u, dt)
    ag = 0.2 * np.cos(2 * np.pi * 1.0 * time) * 9.80665  # 0.2g ground motion
    fr = 10000.0 * u  # k0 = 10,000 N/m
    eh = np.linspace(0, 50.0, n_pts)  # 50 J dissipation

    summary = compute_engineering_metrics(
        time=time,
        ag=ag,
        u=u,
        fr=fr,
        eh=eh,
        v=v,
        dt=dt,
        uy=0.02,  # Yield at 20 mm -> mu = 50 / 20 = 2.5
        mass=100.0,
        height=3.0,
        T0=0.5,
    )

    # 1. Displacement consistency: mm = 1000 * m
    assert pytest.approx(summary.peak_displacement_m, abs=1e-4) == 0.05
    assert pytest.approx(summary.peak_displacement_mm, abs=0.1) == 50.0

    # 2. Velocity
    assert pytest.approx(summary.peak_velocity_mps, abs=0.05) == (0.05 * 2 * np.pi * f)

    # 3. Accelerations: g = mps2 / 9.80665
    assert pytest.approx(summary.peak_ground_accel_g, abs=1e-3) == 0.20
    assert summary.peak_total_accel_g > 0.20

    # 4. Drift: (peak_disp_m / 3.0) * 100
    assert pytest.approx(summary.drift_ratio_percent, abs=1e-3) == (summary.peak_displacement_m / 3.0 * 100.0)

    # 5. Ductility demand: peak_disp_m / 0.02
    assert pytest.approx(summary.ductility_demand_mu, abs=1e-3) == (summary.peak_displacement_m / 0.02)

    # 6. Yield time should be detected since mu > 1.0
    assert summary.yield_time_sec is not None
    assert summary.yield_time_sec > 0.0

    # 7. Energy conversions: kJ = J / 1000
    assert pytest.approx(summary.total_hysteretic_energy_J, abs=1e-3) == 50.0
    assert pytest.approx(summary.total_hysteretic_energy_kJ, abs=1e-4) == 0.05


def test_engineering_metrics_elastic_regime():
    """Verify that when response is elastic (mu <= 1.0), yield_time_sec is None."""
    dt = 0.01
    n_pts = 500
    time = np.linspace(0, (n_pts - 1) * dt, n_pts)
    u = 0.005 * np.sin(2 * np.pi * 1.0 * time)  # 5 mm peak
    ag = np.zeros(n_pts)
    fr = 1000.0 * u
    eh = np.zeros(n_pts)

    summary = compute_engineering_metrics(
        time=time,
        ag=ag,
        u=u,
        fr=fr,
        eh=eh,
        dt=dt,
        uy=0.010,  # 10 mm yield displacement -> mu = 0.5 <= 1.0
        mass=10.0,
    )

    assert summary.ductility_demand_mu <= 1.0
    assert summary.yield_time_sec is None
    assert summary.total_hysteretic_energy_J == 0.0


def test_engineering_metrics_empty_input_raises():
    """Verify error on empty array input."""
    with pytest.raises(ValueError):
        compute_engineering_metrics(
            time=np.array([]),
            ag=np.array([]),
            u=np.array([]),
            fr=np.array([]),
            eh=np.array([]),
        )
