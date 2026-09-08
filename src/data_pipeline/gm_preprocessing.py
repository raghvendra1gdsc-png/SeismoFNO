"""
gm_preprocessing.py — Ground Motion Record Preprocessing Pipeline.

Provides functions to:
  1. Read PEER NGA-West2 (.AT2) files and parse header metadata.
  2. Perform seismological baseline correction (polynomial and high-pass filtering).
  3. Resample acceleration time series to a uniform target dt.
  4. Compute standard seismic intensity measures (PGA, PGV, PGD, Arias Intensity, D_5_95).
  5. Apply end tapers (Tukey / cosine window) to eliminate transient startup/shutdown artifacts.
"""

from typing import Dict, Any, Tuple, Optional, Union
import math
import re
import numpy as np
from scipy import signal, integrate


# Acceleration of gravity [m/s^2]
G_ACCEL = 9.80665


def read_peer_at2(filepath: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Parse a standard PEER NGA-West2 format (.AT2) ground motion file.

    Parameters
    ----------
    filepath : str
        Path to the .AT2 file.

    Returns
    -------
    accel_g : np.ndarray
        Ground acceleration time history in units of g.
    dt : float
        Sampling time step in seconds.
    metadata : dict
        Extracted header metadata (earthquake, station, component, npts, dt).
    """
    metadata: Dict[str, Any] = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise ValueError(f"File {filepath} is too short to be a valid PEER .AT2 file.")

    # Headers typically span lines 0 to 3
    metadata["header_line_1"] = lines[0].strip()
    metadata["header_line_2"] = lines[1].strip()
    metadata["header_line_3"] = lines[2].strip()
    line4 = lines[3].strip()

    # Line 4 format: NPTS= 4000, DT= .0050 SEC or similar
    dt_match = re.search(r"DT=\s*([0-9.]+)", line4, re.IGNORECASE)
    npts_match = re.search(r"NPTS=\s*([0-9]+)", line4, re.IGNORECASE)

    if dt_match:
        dt = float(dt_match.group(1))
    else:
        # Fallback search anywhere in line 4
        floats = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line4)
        if len(floats) >= 2:
            dt = float(floats[1])
        else:
            raise ValueError(f"Could not parse dt from line 4: '{line4}'")

    if npts_match:
        npts = int(npts_match.group(1))
    else:
        npts = -1

    metadata["dt"] = dt
    metadata["npts"] = npts

    # Parse numerical acceleration data from line 4 onwards
    data_str = " ".join(lines[4:])
    accel_values = [float(val) for val in data_str.split()]
    accel_g = np.array(accel_values, dtype=np.float64)

    return accel_g, dt, metadata


def write_peer_at2(
    filepath: str,
    accel_g: np.ndarray,
    dt: float,
    earthquake_name: str = "Synthetic / Processed Record",
    station_name: str = "Station 1",
    component: str = "H1",
) -> None:
    """Write an acceleration series to standard PEER NGA-West2 .AT2 format."""
    npts = len(accel_g)
    header = [
        f"PEER NGA-West2 Record: {earthquake_name}\n",
        f"Station: {station_name}, Component: {component}\n",
        f"ACCELERATION TIME SERIES IN UNITS OF G\n",
        f"NPTS= {npts:6d}, DT= {dt:8.4f} SEC\n",
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(header)
        # Write 5 values per line formatted as 15.7e
        for i in range(0, npts, 5):
            chunk = accel_g[i : i + 5]
            line = "".join(f"{val:15.7e}" for val in chunk) + "\n"
            f.write(line)


def baseline_correct(
    accel: np.ndarray,
    dt: float,
    method: str = "polynomial",
    poly_order: int = 2,
) -> np.ndarray:
    """
    Perform seismological baseline correction on acceleration time history.

    Parameters
    ----------
    accel : np.ndarray
        Acceleration time series [g or m/s^2].
    dt : float
        Time step [s].
    method : str
        'mean' : Simple mean removal (demean).
        'polynomial' : Low-order polynomial fit on velocity to eliminate velocity drift.
        'highpass' : 4th-order zero-phase Butterworth high-pass filter.
    poly_order : int
        Polynomial order for velocity drift correction (default 2 for quadratic).

    Returns
    -------
    corrected_accel : np.ndarray
        Baseline-corrected acceleration time series.
    """
    accel = np.asarray(accel, dtype=np.float64)
    # Step 1: Always demean
    accel_zero_mean = accel - np.mean(accel)

    if method == "mean":
        return accel_zero_mean

    n = len(accel)
    t = np.arange(n, dtype=np.float64) * dt

    if method == "polynomial":
        # Integrate acceleration to velocity
        vel = integrate.cumulative_trapezoid(accel_zero_mean, t, initial=0.0)
        # Fit polynomial to velocity drift
        coeffs = np.polyfit(t, vel, deg=poly_order)
        # The derivative of the velocity trend polynomial is the acceleration correction
        accel_trend = np.zeros_like(t)
        for power, c in enumerate(reversed(coeffs[1:]), start=1):
            accel_trend += c * power * (t ** (power - 1))
        # Note: polyder gives exact derivative coefficients
        poly_vel = np.poly1d(coeffs)
        poly_acc = np.polyder(poly_vel)
        accel_trend = poly_acc(t)

        corrected = accel_zero_mean - accel_trend
        # Final demean to ensure exact zero mean
        return corrected - np.mean(corrected)

    elif method == "highpass":
        # Highpass Butterworth filter at 0.1 Hz
        cutoff_hz = 0.10
        fs = 1.0 / dt
        nyq = 0.5 * fs
        if cutoff_hz >= nyq:
            return accel_zero_mean
        sos = signal.butter(4, cutoff_hz / nyq, btype="highpass", output="sos")
        corrected = signal.sosfiltfilt(sos, accel_zero_mean)
        return corrected - np.mean(corrected)

    else:
        raise ValueError(f"Unknown baseline correction method '{method}'. Choose 'mean', 'polynomial', or 'highpass'.")


def resample_record(
    accel: np.ndarray,
    orig_dt: float,
    target_dt: float = 0.01,
) -> np.ndarray:
    """
    Resample acceleration record to a uniform target time step.

    Parameters
    ----------
    accel : np.ndarray
        Input acceleration series.
    orig_dt : float
        Original time step [s].
    target_dt : float
        Target uniform time step [s].

    Returns
    -------
    resampled_accel : np.ndarray
        Resampled acceleration time series.
    """
    accel = np.asarray(accel, dtype=np.float64)
    if abs(orig_dt - target_dt) < 1e-9:
        return accel.copy()

    n_orig = len(accel)
    duration = (n_orig - 1) * orig_dt
    t_orig = np.arange(n_orig) * orig_dt

    n_target = int(round(duration / target_dt)) + 1
    t_target = np.arange(n_target) * target_dt

    # High-quality cubic spline interpolation
    # If downsampling significantly, apply anti-aliasing lowpass filter first
    if target_dt > orig_dt * 1.05:
        fs_target = 1.0 / target_dt
        nyq_target = 0.5 * fs_target
        fs_orig = 1.0 / orig_dt
        nyq_orig = 0.5 * fs_orig
        cutoff = 0.90 * nyq_target
        if cutoff < nyq_orig:
            sos = signal.butter(4, cutoff / nyq_orig, btype="lowpass", output="sos")
            accel = signal.sosfiltfilt(sos, accel)

    resampled = np.interp(t_target, t_orig, accel)
    return resampled


def apply_cosine_taper(accel: np.ndarray, fraction: float = 0.02) -> np.ndarray:
    """
    Apply a split cosine (Tukey) taper to the first and last `fraction` of the record
    to ensure smooth zero start and end.
    """
    n = len(accel)
    taper_len = int(max(1, round(n * fraction)))
    taper = np.ones(n, dtype=np.float64)
    
    # Cosine ramp up
    ramp_up = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_len)))
    taper[:taper_len] = ramp_up
    # Cosine ramp down
    taper[-taper_len:] = ramp_up[::-1]
    
    return accel * taper


def compute_intensity_measures(accel: np.ndarray, dt: float, in_g: bool = True) -> Dict[str, float]:
    """
    Compute standard ground motion intensity measures.

    Parameters
    ----------
    accel : np.ndarray
        Acceleration time series.
    dt : float
        Sampling step [s].
    in_g : bool
        True if accel is in units of g, False if in m/s^2.

    Returns
    -------
    dict of intensity measures:
        - pga_g : Peak Ground Acceleration [g]
        - pga_ms2 : Peak Ground Acceleration [m/s^2]
        - pgv_ms : Peak Ground Velocity [m/s]
        - pgd_m : Peak Ground Displacement [m]
        - arias_intensity_ms : Arias Intensity [m/s]
        - d5_95_s : 5-95% Significant Duration [s]
    """
    accel = np.asarray(accel, dtype=np.float64)
    if in_g:
        acc_g = accel
        acc_ms2 = accel * G_ACCEL
    else:
        acc_g = accel / G_ACCEL
        acc_ms2 = accel

    pga_g = float(np.max(np.abs(acc_g)))
    pga_ms2 = float(np.max(np.abs(acc_ms2)))

    n = len(accel)
    t = np.arange(n) * dt

    # Velocity and displacement time histories
    vel = integrate.cumulative_trapezoid(acc_ms2, t, initial=0.0)
    disp = integrate.cumulative_trapezoid(vel, t, initial=0.0)

    pgv_ms = float(np.max(np.abs(vel)))
    pgd_m = float(np.max(np.abs(disp)))

    # Arias intensity: I_a = (pi / (2*g)) * int(a^2 dt)
    husk_energy = integrate.cumulative_trapezoid(acc_ms2 ** 2, t, initial=0.0)
    arias_total = float((math.pi / (2.0 * G_ACCEL)) * husk_energy[-1])

    if arias_total > 1e-12:
        norm_husid = husk_energy / husk_energy[-1]
        t5_idx = np.searchsorted(norm_husid, 0.05)
        t95_idx = np.searchsorted(norm_husid, 0.95)
        d5_95_s = float(t[min(t95_idx, n - 1)] - t[min(t5_idx, n - 1)])
    else:
        d5_95_s = 0.0

    return {
        "pga_g": pga_g,
        "pga_ms2": pga_ms2,
        "pgv_ms": pgv_ms,
        "pgd_m": pgd_m,
        "arias_intensity_ms": arias_total,
        "d5_95_s": d5_95_s,
    }


def preprocess_ground_motion(
    accel: np.ndarray,
    orig_dt: float,
    target_dt: float = 0.01,
    baseline_method: str = "polynomial",
    taper_fraction: float = 0.02,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Execute complete ground motion preprocessing workflow:
      1. Resample to target_dt
      2. Baseline correction
      3. End tapering
      4. Compute intensity measures

    Returns
    -------
    processed_accel_g : np.ndarray
        Cleaned acceleration time series in g.
    intensity_measures : dict
        Computed IMs.
    """
    # 1. Resample
    accel_resampled = resample_record(accel, orig_dt=orig_dt, target_dt=target_dt)
    
    # 2. Baseline correct
    accel_corrected = baseline_correct(accel_resampled, dt=target_dt, method=baseline_method)
    
    # 3. Taper
    accel_tapered = apply_cosine_taper(accel_corrected, fraction=taper_fraction)
    
    # 4. Final zero-mean ensure
    final_accel = accel_tapered - np.mean(accel_tapered)
    
    ims = compute_intensity_measures(final_accel, dt=target_dt, in_g=True)
    return final_accel, ims
