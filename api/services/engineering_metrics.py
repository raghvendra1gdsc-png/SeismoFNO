"""
api/services/engineering_metrics.py — Engineering Analysis Service.

Computes standardized civil/structural engineering metrics from dynamic response time histories.
Every metric has an explicit mathematical definition, physical unit, and strict separation
between metric/SI units (e.g. m vs mm, m/s^2 vs g, seconds vs discrete sample indices).
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Union
import numpy as np


@dataclass
class EngineeringMetricsSummary:
    """
    Structured container for civil engineering response quantities.
    Units are strictly labeled in field names to prevent silent conversion errors.
    """
    # Displacements
    peak_displacement_m: float       # u_max = max_t |u(t)| [meters]
    peak_displacement_mm: float      # u_max * 1000.0 [millimeters]
    residual_displacement_m: float   # u(t_end) [meters]
    residual_displacement_mm: float  # u(t_end) * 1000.0 [millimeters]

    # Velocities
    peak_velocity_mps: float         # v_max = max_t |v(t)| [meters/second]

    # Accelerations
    peak_total_accel_mps2: float     # a_tot_max = max_t |ag(t) + u_ddot(t)| [m/s^2]
    peak_total_accel_g: float        # a_tot_max / 9.80665 [g]
    peak_ground_accel_g: float       # max_t |ag(t)| / 9.80665 [g]

    # Drift & Ductility
    drift_ratio_percent: Optional[float]  # (u_max / height) * 100 [%]
    ductility_demand_mu: float            # mu = u_max / u_y [dimensionless]
    yield_time_sec: Optional[float]       # min { t | |u(t)| >= u_y } [seconds]

    # Forces
    peak_restoring_force_N: float         # max_t |F_R(t)| [Newtons]
    peak_restoring_force_kN: float        # max_t |F_R(t)| / 1000.0 [kilonewtons]
    base_shear_ratio: float               # F_R_max / (mass * 9.80665) [dimensionless]

    # Energy Dissipation
    total_hysteretic_energy_J: float      # E_h(t_end) [Joules]
    total_hysteretic_energy_kJ: float     # E_h(t_end) / 1000.0 [kilojoules]
    energy_dissipation_ratio: float       # E_h / (0.5 * k0 * uy^2) or normalized [dimensionless]

    # Temporal & Spectral
    response_duration_sec: float          # Total signal duration [seconds]
    significant_duration_sec: float       # Duration between 5% and 95% total energy [seconds]
    pseudo_spectral_accel_g: float        # S_a = omega_n^2 * u_max / 9.80665 [g]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary with rounded numerical values."""
        raw = asdict(self)
        out = {}
        for k, v in raw.items():
            if isinstance(v, float):
                out[k] = round(v, 4) if abs(v) < 1.0 else round(v, 3)
            else:
                out[k] = v
        return out


def compute_engineering_metrics(
    time: np.ndarray,
    ag: np.ndarray,
    u: np.ndarray,
    fr: np.ndarray,
    eh: np.ndarray,
    v: Optional[np.ndarray] = None,
    dt: float = 0.01,
    uy: float = 0.01,
    mass: float = 1.0,
    height: Optional[float] = None,
    T0: Optional[float] = None,
) -> EngineeringMetricsSummary:
    """
    Compute rigorous structural engineering response metrics from time history arrays.

    Parameters
    ----------
    time : 1D np.ndarray, time vector in seconds [s]
    ag : 1D np.ndarray, ground acceleration in meters per second squared [m/s^2]
    u : 1D np.ndarray, relative displacement in meters [m]
    fr : 1D np.ndarray, restoring force in Newtons [N]
    eh : 1D np.ndarray, cumulative dissipated hysteretic energy in Joules [J]
    v : Optional 1D np.ndarray, relative velocity in m/s (numerically computed if None)
    dt : float, sampling interval in seconds [s]
    uy : float, structural yield displacement in meters [m]
    mass : float, structural mass in kilograms [kg]
    height : Optional float, building tributary height in meters [m] (default 3.0m if None)
    T0 : Optional float, fundamental structural period in seconds [s]

    Returns
    -------
    EngineeringMetricsSummary containing all derived quantities with explicit units.
    """
    if len(u) == 0:
        raise ValueError("Input time history arrays cannot be empty.")

    n_pts = len(u)
    total_duration_sec = float(time[-1] - time[0]) if len(time) > 1 else float(n_pts * dt)

    # 1. Peak & Residual Displacements (meters and millimeters)
    abs_u = np.abs(u)
    peak_disp_m = float(np.max(abs_u))
    peak_disp_mm = float(peak_disp_m * 1000.0)
    residual_disp_m = float(abs(u[-1]))
    residual_disp_mm = float(residual_disp_m * 1000.0)

    # 2. Velocity (meters per second)
    if v is None or len(v) != n_pts:
        v_arr = np.gradient(u, dt)
    else:
        v_arr = v
    peak_vel_mps = float(np.max(np.abs(v_arr)))

    # 3. Accelerations (m/s^2 and g)
    u_ddot = np.gradient(v_arr, dt)
    total_accel = ag[:n_pts] + u_ddot
    peak_tot_accel_mps2 = float(np.max(np.abs(total_accel)))
    peak_tot_accel_g = float(peak_tot_accel_mps2 / 9.80665)
    peak_ground_accel_g = float(np.max(np.abs(ag)) / 9.80665)

    # 4. Drift Ratio and Ductility
    eff_height = height if (height is not None and height > 0) else 3.0  # standard single-story height
    drift_percent = float((peak_disp_m / eff_height) * 100.0)
    safe_uy = max(1e-6, uy)
    ductility_mu = float(peak_disp_m / safe_uy)

    # 5. Yield Time: min { t | |u(t)| >= uy }
    yield_indices = np.where(abs_u >= safe_uy)[0]
    if len(yield_indices) > 0 and ductility_mu > 1.0:
        yield_time_sec = float(time[yield_indices[0]])
    else:
        yield_time_sec = None  # Elastic regime: structure did not yield

    # 6. Restoring Force & Base Shear
    peak_fr_N = float(np.max(np.abs(fr)))
    peak_fr_kN = float(peak_fr_N / 1000.0)
    weight_N = max(1e-4, mass * 9.80665)
    base_shear_ratio = float(peak_fr_N / weight_N)

    # 7. Hysteretic Energy Dissipation
    tot_eh_J = float(max(0.0, eh[-1]))
    tot_eh_kJ = float(tot_eh_J / 1000.0)

    # Energy dissipation ratio: Eh relative to elastic strain energy capacity at yield (0.5 * k0 * uy^2)
    # k0 = mass * omega_n^2
    period = T0 if (T0 is not None and T0 > 0) else 0.5
    omega_n = 2.0 * np.pi / period
    k0 = mass * (omega_n ** 2)
    elastic_yield_energy = 0.5 * k0 * (safe_uy ** 2)
    energy_dissipation_ratio = float(tot_eh_J / max(1e-6, elastic_yield_energy))

    # 8. Significant Duration (Trifunac & Brady 1975 energy duration: 5% to 95%)
    # Use cumulative total input / response energy proxy
    cum_energy = np.cumsum(v_arr ** 2 * dt)
    tot_e = cum_energy[-1]
    if tot_e > 1e-9:
        idx_05 = np.searchsorted(cum_energy, 0.05 * tot_e)
        idx_95 = np.searchsorted(cum_energy, 0.95 * tot_e)
        sig_duration_sec = float(time[min(idx_95, n_pts - 1)] - time[min(idx_05, n_pts - 1)])
    else:
        sig_duration_sec = float(total_duration_sec)

    # 9. Pseudo-spectral acceleration: Sa = omega_n^2 * u_max
    pseudo_sa_mps2 = (omega_n ** 2) * peak_disp_m
    pseudo_sa_g = float(pseudo_sa_mps2 / 9.80665)

    return EngineeringMetricsSummary(
        peak_displacement_m=peak_disp_m,
        peak_displacement_mm=peak_disp_mm,
        residual_displacement_m=residual_disp_m,
        residual_displacement_mm=residual_disp_mm,
        peak_velocity_mps=peak_vel_mps,
        peak_total_accel_mps2=peak_tot_accel_mps2,
        peak_total_accel_g=peak_tot_accel_g,
        peak_ground_accel_g=peak_ground_accel_g,
        drift_ratio_percent=drift_percent,
        ductility_demand_mu=ductility_mu,
        yield_time_sec=yield_time_sec,
        peak_restoring_force_N=peak_fr_N,
        peak_restoring_force_kN=peak_fr_kN,
        base_shear_ratio=base_shear_ratio,
        total_hysteretic_energy_J=tot_eh_J,
        total_hysteretic_energy_kJ=tot_eh_kJ,
        energy_dissipation_ratio=energy_dissipation_ratio,
        response_duration_sec=total_duration_sec,
        significant_duration_sec=sig_duration_sec,
        pseudo_spectral_accel_g=pseudo_sa_g,
    )
