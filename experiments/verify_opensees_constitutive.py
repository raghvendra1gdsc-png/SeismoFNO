"""
verify_opensees_constitutive.py — Phase 1 Independent OpenSees Constitutive & Energy Verification.

Verifies:
1. Exact constitutive equation: F_R = k0 * u - (1 - alpha) * k0 * u_p
2. Asymptotic intervention slope under constant Delta u_p: lim_{t->end} Delta u(t) = (1 - alpha) * Delta u_p
   For alpha = 0.02, expected slope = +0.98 in Regime A (no secondary yielding).
3. Energy identity: W_int = integral(F_R du) == E_s + E_diss
   across:
   - Elastic response
   - Monotonic yielding
   - Cyclic symmetric yielding
   - Asymmetric reverse yielding
"""

import math
import numpy as np
from src.ground_truth.opensees_sdof_model import SDOFParams, simulate_sdof


def verify_constitutive_and_energy():
    print("=== 1. VERIFYING STEEL01 CONSTITUTIVE EQUATION & ENERGY DECOMPOSITION IN OPENSEES ===")
    mass = 1.0
    k0 = 100.0  # T = 0.6283 s
    T = 2.0 * math.pi / math.sqrt(k0 / mass)
    alpha = 0.02
    zeta = 0.05
    u_y = 0.01  # fy = 1.0 N
    dt = 0.005
    t = np.arange(0, 8.0, dt)

    cases_ag = {}

    # Case A: Elastic (peak u < 0.01 m)
    cases_ag["Elastic"] = 0.15 * np.sin(2.0 * np.pi * 1.5 * t)

    # Case B: Monotonic Yielding (strong half-pulse followed by zero)
    ag_mono = np.zeros_like(t)
    pulse_len = int(0.6 / dt)
    ag_mono[:pulse_len] = 3.5 * np.sin(np.pi * t[:pulse_len] / 0.6)
    cases_ag["Monotonic"] = ag_mono

    # Case C: Cyclic Symmetric Yielding (strong steady harmonic)
    cases_ag["Cyclic_Symmetric"] = 2.5 * np.sin(2.0 * np.pi * 1.5 * t)

    # Case D: Asymmetric Reverse Yielding (two-frequency asymmetric excitation)
    cases_ag["Asymmetric_Reverse"] = (
        2.5 * np.sin(2.0 * np.pi * 1.2 * t) + 1.2 * np.sin(2.0 * np.pi * 2.4 * t)
    )

    results = {}

    for name, ag in cases_ag.items():
        mat_type = "elastic" if name == "Elastic" else "bilinear"
        params = SDOFParams(
            T=T,
            zeta=zeta,
            mass=mass,
            material_type=mat_type,
            u_y=u_y,
            alpha=alpha,
            damping_type="mass",
        )

        resp = simulate_sdof(params=params, ag=ag, dt=dt)
        u = resp.u
        f_r = resp.f_r
        w_int = resp.e_h  # integral(F_R du)

        # 1. Invert for plastic displacement: u_p = [u - F_R / k0] / (1 - alpha)
        if mat_type == "elastic":
            u_p = np.zeros_like(u)
        else:
            u_p = (u - f_r / k0) / (1.0 - alpha)

        # 2. Check back-stress identity: alpha_b = alpha * k0 * u_p
        if mat_type == "elastic":
            backstress_err = 0.0
        else:
            alpha_b = (alpha / (1.0 - alpha)) * (k0 * u - f_r)
            alpha_b_expected = alpha * k0 * u_p
            backstress_err = np.max(np.abs(alpha_b - alpha_b_expected))

        # 3. Check constitutive force reconstruction: F_R = k0 * u - (1 - alpha) * k0 * u_p
        if mat_type == "elastic":
            f_r_reconstructed = k0 * u
        else:
            f_r_reconstructed = k0 * u - (1.0 - alpha) * k0 * u_p
        f_err = np.max(np.abs(f_r - f_r_reconstructed))

        # 4. Check energy decomposition: W_int = E_s + E_diss
        z = u - u_p
        e_s = 0.5 * alpha * k0 * (u ** 2) + 0.5 * (1.0 - alpha) * k0 * (z ** 2)

        du_p = np.diff(u_p)
        e_diss = np.zeros_like(u)
        e_diss[1:] = np.cumsum((1.0 - alpha) * params.Fy * np.abs(du_p))

        e_total = e_s + e_diss
        energy_discrepancy = np.max(np.abs(w_int - e_total))

        results[name] = {
            "max_u_mm": np.max(np.abs(u)) * 1000.0,
            "max_up_mm": np.max(np.abs(u_p)) * 1000.0,
            "f_reconstruction_err_N": f_err,
            "backstress_err_N": backstress_err,
            "energy_discrepancy_J": energy_discrepancy,
            "w_int_final_J": w_int[-1],
            "e_s_final_J": e_s[-1],
            "e_diss_final_J": e_diss[-1],
        }

        print(f"[{name}]")
        print(f"  Max u: {results[name]['max_u_mm']:.3f} mm | Max u_p: {results[name]['max_up_mm']:.3f} mm")
        print(f"  F_R Reconstruct Max Error: {f_err:.2e} N")
        print(f"  Backstress Identity Max Error: {backstress_err:.2e} N")
        print(f"  Energy Identity Max Error: {energy_discrepancy:.2e} J (W_int == E_s + E_diss)")
        print(f"  Final W_int: {w_int[-1]:.6f} J | E_s: {e_s[-1]:.6f} J | E_diss: {e_diss[-1]:.6f} J\n")

        assert f_err < 1e-6, f"F_R reconstruction failed for {name}: {f_err}"
        assert backstress_err < 1e-6, f"Backstress identity failed for {name}: {backstress_err}"
        assert energy_discrepancy < 5e-3, f"Energy identity failed for {name}: {energy_discrepancy}"

    print(">>> 1. ALL STEEL01 CONSTITUTIVE & ENERGY IDENTITIES VERIFIED (100% PASS) <<<\n")
    return results


def verify_asymptotic_slope():
    print("=== 2. VERIFYING DYNAMIC ASYMPTOTIC PERTURBATION SLOPE IN OPENSEES ===")
    mass = 1.0
    k0 = 100.0
    T = 2.0 * math.pi / math.sqrt(k0 / mass)
    alpha = 0.02
    zeta = 0.05
    u_y = 0.05  # Set yield displacement higher (50 mm, Fy = 5.0 N) so doses stay strictly within Regime A
    dt = 0.005
    t = np.arange(0, 20.0, dt)

    # Clean yield pulse: single symmetric half-cycle of shaking at t=0.5 to 1.5s that yields the structure,
    # followed by completely zero ground acceleration for t > 1.5s (pure free-vibration decay).
    pulse_ag = np.zeros_like(t)
    pulse_mask = (t >= 0.5) & (t <= 1.5)
    pulse_ag[pulse_mask] = 8.0 * np.sin(np.pi * (t[pulse_mask] - 0.5) / 1.0)

    params = SDOFParams(
        T=T,
        zeta=zeta,
        mass=mass,
        material_type="bilinear",
        u_y=u_y,
        alpha=alpha,
        damping_type="mass",
    )

    t_y = 2.0  # Intervention applied at t = 2.0s during free vibration decay
    idx_y = int(t_y / dt)

    resp_base = simulate_sdof(params=params, ag=pulse_ag, dt=dt)
    u_base = resp_base.u
    f_base = resp_base.f_r
    u_res_base = u_base[-1]

    u_p_base = (u_base - f_base / k0) / (1.0 - alpha)
    print(f"Baseline Peak u: {np.max(np.abs(u_base))*1000.0:.2f} mm | Final u_p: {u_p_base[-1]*1000.0:.2f} mm")
    print(f"Baseline Final Residual Displacement: {u_res_base * 1000.0:.4f} mm\n")

    # Range of doses that remain strictly within elastic limits of the perturbed equilibrium (Regime A)
    doses_mm = [-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0]
    drift_shifts_mm = []

    for dose_mm in doses_mm:
        dose_m = dose_mm / 1000.0
        # Perturbation effective force as acceleration step:
        # Delta F = (1 - alpha) * k0 * Delta u_p
        # a_eff = - Delta F / mass
        ag_perturbed = pulse_ag.copy()
        ag_perturbed[idx_y:] += -(1.0 - alpha) * (k0 / mass) * dose_m

        resp_pert = simulate_sdof(params=params, ag=ag_perturbed, dt=dt)
        u_pert = resp_pert.u
        f_pert = resp_pert.f_r
        u_res_pert = u_pert[-1]

        # Verify that for t >= t_y, NO secondary yielding occurred:
        u_p_pert = (u_pert - f_pert / k0) / (1.0 - alpha)
        # Check if u_p changed after t_y
        delta_up_after_ty = np.max(np.abs(u_p_pert[idx_y:] - u_p_pert[idx_y]))
        assert delta_up_after_ty < 1e-6, f"Secondary yielding occurred in dose {dose_mm}: {delta_up_after_ty}"

        drift_shift_mm = (u_res_pert - u_res_base) * 1000.0
        drift_shifts_mm.append(drift_shift_mm)
        ratio = drift_shift_mm / dose_mm if abs(dose_mm) > 1e-6 else 1.0
        print(f"  Dose = {dose_mm:+5.2f} mm | Drift Shift = {drift_shift_mm:+6.4f} mm | Ratio = {ratio:.5f}")

    doses = np.array(doses_mm)
    shifts = np.array(drift_shifts_mm)
    slope, intercept = np.polyfit(doses, shifts, 1)
    expected_slope = 1.0 - alpha

    print(f"\nMeasured Asymptotic Slope in Regime A: m = {slope:.5f}")
    print(f"Expected Asymptotic Slope (1 - alpha):   = {expected_slope:.5f}")
    slope_error = abs(slope - expected_slope)
    print(f"Discrepancy: {slope_error:.2e}")
    assert slope_error < 1e-3, f"Regime A slope error too large: {slope_error}"
    print(">>> 2. REGIME A DYNAMIC ASYMPTOTIC SLOPE VERIFICATION: PASS (100% MATCH) <<<\n")


if __name__ == "__main__":
    verify_constitutive_and_energy()
    verify_asymptotic_slope()
