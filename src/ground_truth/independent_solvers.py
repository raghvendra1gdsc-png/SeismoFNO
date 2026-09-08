"""
independent_solvers.py — Pure Python & NumPy Independent Structural Dynamics Solvers.

Provides:
  1. Exact closed-form analytical solutions:
     - Underdamped free vibration decay envelope & trajectory.
     - Steady-state harmonic base excitation response.
  2. Independent hand-coded nonlinear Newmark-beta integrator:
     - Constant average acceleration method (gamma = 0.5, beta = 0.25).
     - Newton-Raphson equilibrium iteration for bilinear kinematic hardening (k0, alpha, u_y).
     - Zero OpenSeesPy dependency for independent verification.
  3. Classical Structural Engineering Baseline:
     - Chopra Capacity Spectrum Method / Equivalent Linearization for peak displacement u_max.
"""

from typing import Dict, Any, Tuple, Optional
import math
import numpy as np


# ==============================================================================
# 1. Closed-Form Analytical Reference Solvers
# ==============================================================================

def analytical_sdof_free_vibration(
    u0: float,
    v0: float,
    omega_n: float,
    zeta: float,
    time: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact analytical solution for underdamped SDOF free vibration:
        m * u_ddot + c * u_dot + k * u = 0, with u(0)=u0, v(0)=v0.

    Returns:
        u(t): Displacement array [m]
        v(t): Velocity array [m/s]
        envelope(t): Exponential decay envelope [m]
    """
    if zeta >= 1.0:
        raise ValueError("Analytical solver currently configured for underdamped systems (zeta < 1.0).")

    omega_d = omega_n * math.sqrt(1.0 - zeta**2)
    decay = np.exp(-zeta * omega_n * time)

    A = u0
    B = (v0 + zeta * omega_n * u0) / omega_d

    u = decay * (A * np.cos(omega_d * time) + B * np.sin(omega_d * time))
    
    # Velocity derivative:
    v = -zeta * omega_n * u + decay * (-A * omega_d * np.sin(omega_d * time) + B * omega_d * np.cos(omega_d * time))
    
    # Peak envelope:
    C = math.sqrt(A**2 + B**2)
    envelope = C * decay

    return u, v, envelope


def analytical_sdof_harmonic_response(
    p0: float,
    omega_drive: float,
    omega_n: float,
    zeta: float,
    mass: float,
    time: np.ndarray,
) -> np.ndarray:
    """
    Exact analytical total response (transient + steady-state) for an SDOF oscillator
    subjected to harmonic forcing p(t) = p0 * sin(omega_drive * t) starting from rest u(0)=0, v(0)=0.
    """
    k0 = mass * (omega_n ** 2)
    r = omega_drive / omega_n
    omega_d = omega_n * math.sqrt(max(1e-12, 1.0 - zeta**2))
    
    denom = (1.0 - r**2)**2 + (2.0 * zeta * r)**2
    c_sin = (1.0 - r**2) / denom
    c_cos = -(2.0 * zeta * r) / denom
    
    u_st = p0 / k0
    u_steady = u_st * (c_sin * np.sin(omega_drive * time) + c_cos * np.cos(omega_drive * time))
    
    # Transient constants to satisfy u(0) = 0, v(0) = 0
    # u_steady(0) = u_st * c_cos
    # v_steady(0) = u_st * omega_drive * c_sin
    A = -u_st * c_cos
    v_st_0 = u_st * omega_drive * c_sin
    B = (-v_st_0 - zeta * omega_n * A) / omega_d
    
    u_transient = np.exp(-zeta * omega_n * time) * (A * np.cos(omega_d * time) + B * np.sin(omega_d * time))
    
    return u_steady + u_transient


# ==============================================================================
# 2. Independent Hand-Coded Nonlinear Newmark-Beta Integrator
# ==============================================================================

def newmark_nonlinear_sdof(
    mass: float,
    k0: float,
    zeta: float,
    ag: np.ndarray,
    dt: float,
    material_type: str = "bilinear",
    u_y: float = 0.01,
    alpha: float = 0.05,
    gamma: float = 0.5,
    beta: float = 0.25,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Independent hand-coded Newmark-beta step-by-step integrator with Newton-Raphson
    equilibrium iterations for linear-elastic and bilinear kinematic hardening SDOF systems.

    Zero dependency on OpenSeesPy.
    """
    n = len(ag)
    time = np.arange(n) * dt
    omega_n = math.sqrt(k0 / mass)
    c = 2.0 * zeta * mass * omega_n

    # Response arrays
    u = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)
    f_r = np.zeros(n, dtype=np.float64)
    e_h = np.zeros(n, dtype=np.float64)

    # Initial state
    p = -mass * ag
    a[0] = (p[0] - c * v[0] - f_r[0]) / mass

    # Newmark integration constants (Average acceleration)
    a1 = mass / (beta * dt**2) + (gamma / (beta * dt)) * c
    a2 = mass / (beta * dt) + (gamma / beta - 1.0) * c
    a3 = (1.0 / (2.0 * beta) - 1.0) * mass + dt * (gamma / (2.0 * beta) - 1.0) * c

    # Internal hysteretic state (Kinematic hardening)
    # F_R = alpha * k0 * u + (1 - alpha) * k0 * z
    # where z in [-u_y, u_y] tracks the plastic yield surface
    z = 0.0
    u_yield = max(1e-6, u_y) if material_type == "bilinear" else 1e9

    for i in range(1, n):
        # Predictor displacement and velocity
        u_trial = u[i - 1]
        p_eff = p[i] + mass * (
            (1.0 / (beta * dt**2)) * u[i - 1]
            + (1.0 / (beta * dt)) * v[i - 1]
            + (1.0 / (2.0 * beta) - 1.0) * a[i - 1]
        ) + c * (
            (gamma / (beta * dt)) * u[i - 1]
            + (gamma / beta - 1.0) * v[i - 1]
            + dt * (gamma / (2.0 * beta) - 1.0) * a[i - 1]
        )

        z_curr = z
        kt_curr = k0

        # Newton-Raphson equilibrium iteration
        for _ in range(max_iter):
            # Compute internal restoring force for trial displacement
            delta_u = u_trial - u[i - 1]
            
            if material_type == "elastic":
                f_trial = k0 * u_trial
                kt_curr = k0
            else:
                # Bilinear kinematic hardening
                z_trial = z + delta_u
                if z_trial > u_yield:
                    z_curr = u_yield
                    kt_curr = alpha * k0
                elif z_trial < -u_yield:
                    z_curr = -u_yield
                    kt_curr = alpha * k0
                else:
                    z_curr = z_trial
                    kt_curr = k0
                f_trial = alpha * k0 * u_trial + (1.0 - alpha) * k0 * z_curr

            # Effective tangent stiffness
            k_hat = kt_curr + a1

            # Residual out-of-balance force
            r = p_eff - f_trial - (a1 * u_trial)

            if abs(r) < tol:
                break

            # Displacement increment
            du = r / k_hat
            u_trial += du

        # Update state at step i
        u[i] = u_trial
        z = z_curr
        f_r[i] = f_trial

        a[i] = (1.0 / (beta * dt**2)) * (u[i] - u[i - 1]) - (1.0 / (beta * dt)) * v[i - 1] - (1.0 / (2.0 * beta) - 1.0) * a[i - 1]
        v[i] = v[i - 1] + dt * ((1.0 - gamma) * a[i - 1] + gamma * a[i])

    # Compute cumulative hysteretic work: integral(F_R du)
    du_arr = np.diff(u)
    f_avg = 0.5 * (f_r[:-1] + f_r[1:])
    e_h[1:] = np.cumsum(f_avg * du_arr)

    return {
        "time": time,
        "u": u,
        "v": v,
        "a": a,
        "f_r": f_r,
        "e_h": e_h,
    }


# ==============================================================================
# 3. Classical Engineering Baseline: Chopra Equivalent Linearization
# ==============================================================================

def chopra_capacity_spectrum_prediction(
    pga_g: float,
    T0: float,
    zeta0: float,
    u_y: float,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Classical Structural Engineering Baseline (Chopra & Goel, 1999):
    Estimates inelastic peak displacement demand using equivalent linearization.

    Effective Period:
        T_eff = T0 * sqrt(mu / (1 + alpha*(mu - 1)))
    Equivalent Damping:
        zeta_eff = zeta0 + (2/pi) * ( (mu - 1)*(1 - alpha) / (mu * (1 + alpha*mu - alpha)) )
    """
    omega0 = 2.0 * math.pi / T0
    # Peak ground acceleration in m/s^2
    pga_ms2 = pga_g * 9.80665
    
    # Elastic spectral displacement estimate: S_d = S_a / omega0^2 (approximating Sa ~ 2.5 * PGA)
    sa_elastic = 2.5 * pga_ms2
    sd_elastic = sa_elastic / (omega0 ** 2)

    # Elastic estimate
    if sd_elastic <= u_y:
        return {
            "u_max_pred": sd_elastic,
            "ductility_mu": sd_elastic / u_y,
            "T_eff": T0,
            "zeta_eff": zeta0,
            "regime": "elastic",
        }

    # Iterative solution for ductility mu using Newmark-Hall / Chopra R-mu-T relation
    R = sd_elastic / u_y  # Strength reduction factor
    # For moderate/long period structures: mu ≈ R (Equal Displacement Rule)
    # For short period structures: mu ≈ 0.5 * (R^2 + 1) (Equal Energy Rule)
    if T0 >= 0.5:
        mu = R
    else:
        mu = 0.5 * (R**2 + 1.0)

    u_max_inelastic = mu * u_y
    t_eff = T0 * math.sqrt(mu / max(1e-4, 1.0 + alpha * (mu - 1.0)))
    zeta_eff = zeta0 + (2.0 / math.pi) * ((mu - 1.0) * (1.0 - alpha)) / max(1e-4, mu * (1.0 + alpha * mu - alpha))

    return {
        "u_max_pred": u_max_inelastic,
        "ductility_mu": mu,
        "T_eff": t_eff,
        "zeta_eff": zeta_eff,
        "regime": "inelastic",
    }
