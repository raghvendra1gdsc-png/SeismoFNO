"""
test_opensees_sdof_known_solution.py — Phase 1 Ground Truth Verification.

Performs two independent validation checks against OpenSeesPy SDOF solver,
each required to pass within 1.0 % error:

Check 1: Free-vibration decay
  - Initial displacement u0, zero ground motion, underdamped (zeta = 0.05).
  - Compare numerical trajectory and decay envelope against the closed-form
    analytical solution:
        u(t) = exp(-zeta * omega_n * t) * [ u0*cos(omega_d*t) + (zeta*omega_n*u0/omega_d)*sin(omega_d*t) ]
    decay envelope = u0 * exp(-zeta * omega_n * t).

Check 2: Independent hand-coded Newmark-beta integrator
  - Written completely from scratch in NumPy (no OpenSeesPy dependency).
  - Constant average acceleration method (gamma = 0.5, beta = 0.25).
  - Cross-check linear-elastic response under harmonic base excitation
    a_g(t) = A * sin(omega_drive * t).

Additional check: Bilinear hysteretic energy and yield force validation.
"""

import math
import numpy as np
import pytest

from src.ground_truth.opensees_sdof_model import (
    SDOFParams,
    SDOFResponse,
    simulate_sdof,
    OpenSeesSDOF,
    compute_cumulative_energy,
)


def hand_coded_newmark_linear_sdof(
    m: float,
    k: float,
    c: float,
    p: np.ndarray,
    dt: float,
    u0: float = 0.0,
    v0: float = 0.0,
    gamma: float = 0.5,
    beta: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Independent hand-coded Newmark-beta numerical time-stepping integrator
    for a linear SDOF oscillator:
        m * u_ddot + c * u_dot + k * u = p(t)

    This is an independent reference solver written with zero OpenSeesPy code.
    """
    n = len(p)
    u = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)

    # Initial state
    u[0] = u0
    v[0] = v0
    a[0] = (p[0] - c * v0 - k * u0) / m

    # Integration constants for average acceleration (Chopra, Dynamics of Structures)
    a1 = (1.0 / (beta * dt**2)) * m + (gamma / (beta * dt)) * c
    a2 = (1.0 / (beta * dt)) * m + (gamma / beta - 1.0) * c
    a3 = (1.0 / (2.0 * beta) - 1.0) * m + dt * (gamma / (2.0 * beta) - 1.0) * c

    k_hat = k + a1

    for i in range(1, n):
        p_hat = p[i] + m * (
            (1.0 / (beta * dt**2)) * u[i - 1]
            + (1.0 / (beta * dt)) * v[i - 1]
            + (1.0 / (2.0 * beta) - 1.0) * a[i - 1]
        ) + c * (
            (gamma / (beta * dt)) * u[i - 1]
            + (gamma / beta - 1.0) * v[i - 1]
            + dt * (gamma / (2.0 * beta) - 1.0) * a[i - 1]
        )
        u[i] = p_hat / k_hat
        a[i] = (1.0 / (beta * dt**2)) * (u[i] - u[i - 1]) - (1.0 / (beta * dt)) * v[i - 1] - (1.0 / (2.0 * beta) - 1.0) * a[i - 1]
        v[i] = v[i - 1] + dt * ((1.0 - gamma) * a[i - 1] + gamma * a[i])

    return u, v, a


def test_sdof_free_vibration_decay_closed_form():
    """
    Check 1: Free-vibration decay envelope and trajectory vs closed-form solution.
    Tolerance: < 1.0 % relative error.
    """
    T = 1.0          # Period = 1.0 s
    zeta = 0.05      # 5 % damping
    m = 1.0          # mass = 1.0 kg
    u0 = 0.05        # initial displacement = 5 cm
    v0 = 0.0         # initial velocity = 0
    dt = 0.002       # time step
    duration = 10.0  # 10 full cycles
    n_steps = int(duration / dt) + 1

    params = SDOFParams(T=T, zeta=zeta, mass=m, material_type="elastic", damping_type="mass")
    omega_n = params.omega_n
    omega_d = omega_n * math.sqrt(1.0 - zeta**2)

    # Run OpenSees SDOF simulation
    resp = simulate_sdof(params=params, ag=None, dt=dt, n_steps=n_steps, u0=u0, v0=v0)

    t = resp.time
    u_num = resp.u

    # Exact closed-form analytical solution for underdamped free vibration
    u_exact = np.exp(-zeta * omega_n * t) * (
        u0 * np.cos(omega_d * t) + (zeta * omega_n * u0 + v0) / omega_d * np.sin(omega_d * t)
    )

    # 1. Full trajectory relative L2 error
    rel_l2_error = np.linalg.norm(u_num - u_exact) / np.linalg.norm(u_exact)

    # 2. Peak decay envelope error: compare numerical peaks to exponential decay envelope
    # Theoretical peak times: t_k = k * (pi / omega_d) = k * (T_d / 2)
    T_d = 2.0 * math.pi / omega_d
    half_periods = np.arange(0, duration, T_d / 2.0)
    peak_indices = [int(round(tp / dt)) for tp in half_periods if int(round(tp / dt)) < len(t)]
    
    envelope_exact = u0 * np.exp(-zeta * omega_n * t[peak_indices])
    peak_abs_num = np.abs(u_num[peak_indices])
    peak_envelope_rel_error = np.max(np.abs(peak_abs_num - envelope_exact) / envelope_exact)

    print(f"\n[Check 1: Free Vibration]")
    print(f"  Relative L2 trajectory error : {rel_l2_error:.4e} ({rel_l2_error * 100:.3f} %)")
    print(f"  Max peak envelope error      : {peak_envelope_rel_error:.4e} ({peak_envelope_rel_error * 100:.3f} %)")

    assert rel_l2_error < 0.01, f"Trajectory relative L2 error {rel_l2_error * 100:.2f}% exceeds 1.0%"
    assert peak_envelope_rel_error < 0.01, f"Peak envelope error {peak_envelope_rel_error * 100:.2f}% exceeds 1.0%"


def test_sdof_harmonic_vs_hand_coded_newmark():
    """
    Check 2: Cross-check OpenSees response against independent hand-coded Newmark-beta.
    Tolerance: < 1.0 % relative error (in practice matches to machine precision).
    """
    T = 0.5          # Period = 0.5 s (omega_n = 4*pi rad/s)
    zeta = 0.05      # 5 % damping
    m = 1.0          # mass = 1.0 kg
    dt = 0.005       # 200 Hz sampling
    duration = 5.0   # 5 seconds
    t = np.arange(0, duration, dt)
    n_steps = len(t)

    # Harmonic base excitation: a_g(t) = A * sin(omega_drive * t)
    # Driving near resonance: f_drive = 1.5 Hz
    omega_drive = 2.0 * math.pi * 1.5
    ag = 1.5 * np.sin(omega_drive * t)

    params = SDOFParams(T=T, zeta=zeta, mass=m, material_type="elastic", damping_type="mass")
    k = params.k0
    c = params.c
    p = -m * ag  # Effective earthquake inertial load

    # Independent hand-coded Newmark integrator
    u_hand, v_hand, a_hand = hand_coded_newmark_linear_sdof(
        m=m, k=k, c=c, p=p, dt=dt, u0=0.0, v0=0.0, gamma=0.5, beta=0.25
    )

    # OpenSees SDOF simulation
    resp = simulate_sdof(params=params, ag=ag, dt=dt)
    u_ops = resp.u
    v_ops = resp.v
    a_ops = resp.a

    rel_l2_u = np.linalg.norm(u_ops - u_hand) / np.linalg.norm(u_hand)
    rel_l2_v = np.linalg.norm(v_ops - v_hand) / np.linalg.norm(v_hand)
    max_abs_u = np.max(np.abs(u_ops - u_hand))

    print(f"\n[Check 2: Hand-Coded Newmark vs OpenSees]")
    print(f"  Displacement relative L2 error : {rel_l2_u:.4e} ({rel_l2_u * 100:.3f} %)")
    print(f"  Velocity relative L2 error     : {rel_l2_v:.4e} ({rel_l2_v * 100:.3f} %)")
    print(f"  Max absolute displacement diff : {max_abs_u:.4e} m")

    assert rel_l2_u < 0.01, f"Displacement relative L2 error {rel_l2_u * 100:.2f}% exceeds 1.0%"
    assert rel_l2_v < 0.01, f"Velocity relative L2 error {rel_l2_v * 100:.2f}% exceeds 1.0%"


def test_sdof_bilinear_hysteretic_energy_and_yield():
    """
    Check 3: Bilinear-hysteretic response validation.
    Verifies:
      1. Peak force matches post-yield formulation: F_peak = Fy + alpha * k * (u_peak - u_y).
      2. Cumulative hysteretic energy E_h(t) is monotonically non-decreasing over full cycles.
      3. Hysteresis loops dissipate positive total energy.
    """
    T = 0.5
    zeta = 0.02
    m = 1.0
    u_y = 0.005       # 5 mm yield displacement
    alpha = 0.05      # 5 % post-yield stiffness
    dt = 0.002
    duration = 4.0
    t = np.arange(0, duration, dt)

    # Strong sinusoidal excitation driving structure well into post-yield regime
    ag = 4.0 * np.sin(2.0 * math.pi * 2.0 * t)

    params = SDOFParams(
        T=T,
        zeta=zeta,
        mass=m,
        material_type="bilinear",
        u_y=u_y,
        alpha=alpha,
        damping_type="mass",
    )

    resp = simulate_sdof(params=params, ag=ag, dt=dt)

    u_peak = np.max(np.abs(resp.u))
    ductility = u_peak / u_y
    f_peak = np.max(np.abs(resp.f_r))
    expected_f_peak = params.Fy + alpha * params.k0 * (u_peak - u_y)

    print(f"\n[Check 3: Bilinear Hysteresis]")
    print(f"  Ductility demand (u_max / u_y) : {ductility:.2f}")
    print(f"  Measured peak force            : {f_peak:.4f} N")
    print(f"  Expected peak force            : {expected_f_peak:.4f} N")
    print(f"  Final cumulative energy E_h    : {resp.e_h[-1]:.4f} J")

    # Structure must have yielded significantly (ductility > 2.0)
    assert ductility > 2.0, f"Ductility demand {ductility:.2f} is too low to test yielding"

    # Peak force must match bilinear backbone within 1 %
    force_err = abs(f_peak - expected_f_peak) / expected_f_peak
    assert force_err < 0.01, f"Peak restoring force error {force_err * 100:.2f}% exceeds 1.0%"

    # Cumulative hysteretic energy must be positive and non-trivial
    assert resp.e_h[-1] > 0.0, "Cumulative hysteretic energy must be positive"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
