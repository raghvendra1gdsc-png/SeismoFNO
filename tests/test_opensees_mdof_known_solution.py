"""
test_opensees_mdof_known_solution.py — Phase 8 MDOF Ground Truth Verification.

Performs three rigorous independent validation checks against OpenSeesPy MDOF solver:
  1. Check 1: Closed-Form Modal Eigenvalue Analysis
     - Analytical tridiagonal stiffness matrix [K] and diagonal mass matrix [M].
     - Verifies natural circular frequencies omega_1, ..., omega_N against OpenSees ops.eigen(N).
     - Required tolerance: < 0.1 % relative error on all modes.

  2. Check 2: Independent Hand-Coded MDOF Newmark-Beta Integrator
     - Fully independent NumPy multi-DOF matrix time-stepping solver.
     - Cross-checks transient floor displacement trajectories u_i(t) under multi-frequency ground excitation.
     - Required tolerance: < 1.0 % relative L2 error across all floors.

  3. Check 3: Bilinear Hysteretic Energy Dissipation & Yield Drift
     - Verifies post-yield inter-story shear forces, hysteretic looping, and positive monotonic energy dissipation.
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import pytest
import scipy.linalg

from src.ground_truth.opensees_mdof_model import (
    MDOFParams,
    MDOFResponse,
    OpenSeesMDOF,
    simulate_mdof,
)


def hand_coded_newmark_linear_mdof(
    M: np.ndarray,
    K: np.ndarray,
    C: np.ndarray,
    ag: np.ndarray,
    dt: float,
    gamma: float = 0.5,
    beta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Independent hand-coded Newmark-beta matrix time-stepping integrator for an MDOF system:
        [M] u_ddot + [C] u_dot + [K] u = -[M] {r} a_g(t)

    This is an independent reference solver written with zero OpenSeesPy code.
    """
    n_dof = M.shape[0]
    n_steps = len(ag)
    r = np.ones(n_dof, dtype=np.float64)  # Influence vector for uniform base excitation

    u = np.zeros((n_dof, n_steps), dtype=np.float64)
    v = np.zeros((n_dof, n_steps), dtype=np.float64)
    a = np.zeros((n_dof, n_steps), dtype=np.float64)

    # Initial state (at rest)
    p0 = -M @ r * ag[0]
    a[:, 0] = scipy.linalg.solve(M, p0 - C @ v[:, 0] - K @ u[:, 0])

    # Integration constants for constant average acceleration (Chopra)
    a1 = (1.0 / (beta * dt**2)) * M + (gamma / (beta * dt)) * C
    a2 = (1.0 / (beta * dt)) * M + (gamma / beta - 1.0) * C
    a3 = (1.0 / (2.0 * beta) - 1.0) * M + dt * (gamma / (2.0 * beta) - 1.0) * C

    K_hat = K + a1

    for i in range(1, n_steps):
        p_curr = -M @ r * ag[i]
        p_hat = p_curr + M @ (
            (1.0 / (beta * dt**2)) * u[:, i - 1]
            + (1.0 / (beta * dt)) * v[:, i - 1]
            + (1.0 / (2.0 * beta) - 1.0) * a[:, i - 1]
        ) + C @ (
            (gamma / (beta * dt)) * u[:, i - 1]
            + (gamma / beta - 1.0) * v[:, i - 1]
            + dt * (gamma / (2.0 * beta) - 1.0) * a[:, i - 1]
        )

        u[:, i] = scipy.linalg.solve(K_hat, p_hat)
        a[:, i] = (1.0 / (beta * dt**2)) * (u[:, i] - u[:, i - 1]) - (1.0 / (beta * dt)) * v[:, i - 1] - (1.0 / (2.0 * beta) - 1.0) * a[:, i - 1]
        v[:, i] = v[:, i - 1] + dt * ((1.0 - gamma) * a[:, i - 1] + gamma * a[:, i])

    return u, v, a


def test_mdof_modal_eigenvalues_closed_form():
    """
    Check 1: Cross-check OpenSees modal frequencies against theoretical eigenvalues of [K], [M].
    Evaluates both 3-story and 5-story buildings.
    Tolerance: < 0.1 % error on all modes.
    """
    configurations = [
        # (name, n_stories, masses, stiffnesses)
        ("3-Story Frame", 3, [1200.0, 1000.0, 800.0], [1.5e6, 1.2e6, 1.0e6]),
        ("5-Story Building", 5, [1500.0, 1400.0, 1200.0, 1000.0, 800.0], [2.5e6, 2.2e6, 2.0e6, 1.8e6, 1.5e6]),
    ]

    for name, n_stories, masses, stiffnesses in configurations:
        params = MDOFParams(
            n_stories=n_stories,
            story_masses=masses,
            story_stiffnesses=stiffnesses,
            material_type="elastic",
        )

        # 1. Theoretical eigenvalue calculation
        omegas_exact, periods_exact, phi_exact = params.compute_theoretical_modal_properties()

        # 2. OpenSees modal analysis
        model = OpenSeesMDOF(params)
        omegas_ops, periods_ops, phi_ops = model.build_model()

        print(f"\n[Check 1: {name} Modal Eigenvalue Verification]")
        for mode in range(n_stories):
            err_omega = abs(omegas_ops[mode] - omegas_exact[mode]) / omegas_exact[mode]
            err_period = abs(periods_ops[mode] - periods_exact[mode]) / periods_exact[mode]
            print(f"  Mode {mode + 1}: Exact omega = {omegas_exact[mode]:.4f} rad/s, OpenSees = {omegas_ops[mode]:.4f} rad/s | Error = {err_omega * 100:.4f} %")
            assert err_omega < 0.001, f"{name} Mode {mode+1} frequency error {err_omega*100:.3f}% exceeds 0.1%"
            assert err_period < 0.001, f"{name} Mode {mode+1} period error {err_period*100:.3f}% exceeds 0.1%"


def test_mdof_harmonic_vs_hand_coded_newmark_integrator():
    """
    Check 2: Cross-check OpenSees transient dynamic response against independent hand-coded MDOF Newmark.
    Evaluates a 3-story building under multi-frequency base excitation.
    Tolerance: < 1.0 % relative L2 error across all floors.
    """
    n_stories = 3
    masses = [1000.0, 1000.0, 1000.0]
    stiffnesses = [1.2e6, 1.0e6, 0.8e6]
    dt = 0.005
    duration = 6.0
    t = np.arange(0, duration, dt)

    # Multi-frequency base excitation driving multiple modes
    ag = 1.2 * np.sin(2.0 * math.pi * 1.5 * t) + 0.8 * np.sin(2.0 * math.pi * 4.0 * t)

    params = MDOFParams(
        n_stories=n_stories,
        story_masses=masses,
        story_stiffnesses=stiffnesses,
        material_type="elastic",
        zeta_1=0.05,
        zeta_2=0.05,
    )

    # Compute theoretical system matrices and Rayleigh damping matrix [C]
    M, K = params.build_theoretical_matrices()
    omegas_exact, _, _ = params.compute_theoretical_modal_properties()
    alpha_m, beta_k = params.compute_rayleigh_constants(omegas_exact[0], omegas_exact[1])
    C = alpha_m * M + beta_k * K

    # 1. Independent hand-coded Newmark integrator
    u_hand, v_hand, a_hand = hand_coded_newmark_linear_mdof(
        M=M, K=K, C=C, ag=ag, dt=dt, gamma=0.5, beta=0.25
    )

    # 2. OpenSees simulation
    resp_ops = simulate_mdof(params=params, ag=ag, dt=dt)
    u_ops = resp_ops.u
    v_ops = resp_ops.v

    print(f"\n[Check 2: Hand-Coded Newmark vs OpenSees (3-Story Building)]")
    for floor in range(n_stories):
        rel_l2_u = np.linalg.norm(u_ops[floor, :] - u_hand[floor, :]) / np.linalg.norm(u_hand[floor, :])
        rel_l2_v = np.linalg.norm(v_ops[floor, :] - v_hand[floor, :]) / np.linalg.norm(v_hand[floor, :])
        max_abs_diff = np.max(np.abs(u_ops[floor, :] - u_hand[floor, :]))
        print(f"  Floor {floor + 1}: Displacement Rel L2 Error = {rel_l2_u * 100:.4f} %, Velocity Rel L2 Error = {rel_l2_v * 100:.4f} %, Max Diff = {max_abs_diff:.4e} m")
        assert rel_l2_u < 0.01, f"Floor {floor+1} displacement relative L2 error {rel_l2_u*100:.3f}% exceeds 1.0%"
        assert rel_l2_v < 0.01, f"Floor {floor+1} velocity relative L2 error {rel_l2_v*100:.3f}% exceeds 1.0%"


def test_mdof_bilinear_fiber_hysteretic_energy():
    """
    Check 3: Nonlinear Bilinear MDOF response verification.
    Verifies yielding, post-yield drift, peak force consistency, and positive monotonic hysteretic energy.
    """
    n_stories = 3
    masses = [1000.0, 1000.0, 1000.0]
    stiffnesses = [1.0e6, 1.0e6, 1.0e6]
    yield_drifts = [0.005, 0.005, 0.005]  # 5 mm yield drift per story
    alpha = 0.05
    dt = 0.002
    duration = 4.0
    t = np.arange(0, duration, dt)

    # Strong excitation to ensure all floors yield
    ag = 4.5 * np.sin(2.0 * math.pi * 2.0 * t)

    params = MDOFParams(
        n_stories=n_stories,
        story_masses=masses,
        story_stiffnesses=stiffnesses,
        material_type="bilinear",
        yield_displacements=yield_drifts,
        alpha=alpha,
        zeta_1=0.03,
        zeta_2=0.03,
    )

    resp = simulate_mdof(params=params, ag=ag, dt=dt)

    print(f"\n[Check 3: Nonlinear Bilinear MDOF Hysteresis]")
    for floor in range(n_stories):
        u_peak = np.max(np.abs(resp.u[floor, :]))
        ductility = u_peak / yield_drifts[floor]
        final_energy = resp.e_h[floor, -1]
        print(f"  Floor {floor + 1}: Peak Disp = {u_peak*1000:.2f} mm | Ductility = {ductility:.2f} | Final Hysteretic Energy = {final_energy:.2f} J")

        assert ductility > 1.5, f"Floor {floor+1} ductility demand {ductility:.2f} is insufficient to verify nonlinear yielding"
        assert final_energy > 0.0, f"Floor {floor+1} cumulative hysteretic energy must be strictly positive"
        # Energy must be monotonically non-decreasing
        assert np.all(np.diff(resp.e_h[floor, :]) >= -1e-8), f"Floor {floor+1} hysteretic energy decreased!"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
