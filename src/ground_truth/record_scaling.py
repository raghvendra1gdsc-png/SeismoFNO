"""
record_scaling.py — Ground Motion Scaling and Manifest Generation.

Provides routines to:
  1. Scale ground motion time series to specified target Peak Ground Accelerations (PGAs)
     spanning 0.05g to 1.2g (or custom ranges).
  2. Maintain seismological record metadata (Earthquake Name, Mw, Rrup, Vs30, Mechanism).
  3. Generate and save scaled acceleration records (.npy / .csv).
  4. Build and write the comprehensive `data/simulations/dataset_manifest.csv`.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_pipeline.gm_preprocessing import G_ACCEL, compute_intensity_measures


# Standard target PGA levels spanning 0.05g to 1.20g
DEFAULT_TARGET_PGAS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20]


def scale_record_to_pga(
    accel: np.ndarray,
    target_pga_g: float,
    current_pga_g: Optional[float] = None,
    in_g: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """
    Scale an acceleration time series to a target PGA.

    Parameters
    ----------
    accel : np.ndarray
        Input acceleration series.
    target_pga_g : float
        Desired target PGA in units of g.
    current_pga_g : float, optional
        Pre-computed unscaled PGA in g.
    in_g : bool
        True if accel is in units of g, False if m/s^2.

    Returns
    -------
    scaled_accel : np.ndarray
        Scaled acceleration series in the same units as input.
    scale_factor : float
        Multiplicative scale factor applied (target_pga / unscaled_pga).
    resulting_pga_g : float
        Actual measured PGA of the scaled record in g.
    """
    accel = np.asarray(accel, dtype=np.float64)
    if current_pga_g is None:
        peak_val = float(np.max(np.abs(accel)))
        current_pga_g = peak_val if in_g else peak_val / G_ACCEL

    if current_pga_g < 1e-12:
        raise ValueError("Cannot scale a zero or near-zero acceleration record.")

    scale_factor = float(target_pga_g / current_pga_g)
    scaled_accel = accel * scale_factor
    resulting_pga_g = float(np.max(np.abs(scaled_accel))) if in_g else float(np.max(np.abs(scaled_accel)) / G_ACCEL)

    return scaled_accel, scale_factor, resulting_pga_g


def process_and_scale_suite(
    records: List[Dict[str, Any]],
    target_pgas: Optional[List[float]] = None,
    output_dir: Union[str, Path] = "data/simulations/scaled_motions",
    manifest_path: Union[str, Path] = "data/simulations/dataset_manifest.csv",
    save_format: str = "npy",
) -> pd.DataFrame:
    """
    Scale a suite of preprocessed ground motion records to target PGAs,
    save the resulting time series to disk, and write the dataset manifest.

    Parameters
    ----------
    records : list of dict
        Each dict must contain:
          - 'record_id' or 'rsn': identifier string
          - 'accel_g': preprocessed acceleration array in g
          - 'dt': time step in seconds
          - 'earthquake_name': str
          - 'magnitude' or 'Mw': float
          - 'r_rup_km': float
          - 'vs30_ms': float
          Optional: 'station_name', 'year', 'mechanism'.
    target_pgas : list of float, optional
        List of target PGAs in g (default: 0.05g to 1.20g).
    output_dir : str or Path
        Directory where scaled records will be saved.
    manifest_path : str or Path
        Path to output dataset_manifest.csv.
    save_format : str
        'npy' (fast binary) or 'csv'.

    Returns
    -------
    manifest_df : pd.DataFrame
        Complete manifest of all scaled records and metadata.
    """
    if target_pgas is None:
        target_pgas = DEFAULT_TARGET_PGAS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for rec in records:
        base_id = str(rec.get("record_id", rec.get("rsn", "REC")))
        accel_g = np.asarray(rec["accel_g"], dtype=np.float64)
        dt = float(rec["dt"])
        eq_name = str(rec.get("earthquake_name", "Unknown"))
        year = rec.get("year", np.nan)
        station = str(rec.get("station_name", "Unknown"))
        magnitude = float(rec.get("magnitude", rec.get("Mw", np.nan)))
        r_rup = float(rec.get("r_rup_km", np.nan))
        vs30 = float(rec.get("vs30_ms", np.nan))
        mechanism = str(rec.get("mechanism", "Unspecified"))

        base_pga_g = float(np.max(np.abs(accel_g)))
        n_steps = len(accel_g)
        duration_s = float((n_steps - 1) * dt)

        for target_pga in target_pgas:
            scaled_accel, scale_factor, res_pga_g = scale_record_to_pga(
                accel=accel_g, target_pga_g=target_pga, current_pga_g=base_pga_g, in_g=True
            )
            
            # Formulate unique instance ID
            pga_tag = f"{target_pga:.3f}g".replace(".", "p")
            instance_id = f"{base_id}_PGA_{pga_tag}"

            # Save file
            if save_format == "npy":
                filename = f"{instance_id}.npy"
                filepath = output_path / filename
                np.save(filepath, scaled_accel)
            elif save_format == "csv":
                filename = f"{instance_id}.csv"
                filepath = output_path / filename
                np.savetxt(filepath, scaled_accel, delimiter=",", fmt="%.8e")
            else:
                raise ValueError(f"Unsupported save_format: {save_format}")

            # Compute intensity measures for the scaled motion
            ims = compute_intensity_measures(scaled_accel, dt=dt, in_g=True)

            manifest_rows.append({
                "record_id": instance_id,
                "base_record_id": base_id,
                "earthquake_name": eq_name,
                "year": year,
                "station_name": station,
                "magnitude": magnitude,
                "r_rup_km": r_rup,
                "vs30_ms": vs30,
                "mechanism": mechanism,
                "scale_factor": scale_factor,
                "target_pga_g": target_pga,
                "resulting_pga_g": res_pga_g,
                "resulting_pga_ms2": res_pga_g * G_ACCEL,
                "pgv_ms": ims["pgv_ms"],
                "pgd_m": ims["pgd_m"],
                "arias_intensity_ms": ims["arias_intensity_ms"],
                "d5_95_s": ims["d5_95_s"],
                "dt": dt,
                "n_steps": n_steps,
                "duration_s": duration_s,
                "file_path": str(filepath.resolve()),
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_file, index=False)
    print(f"Saved {len(manifest_df)} scaled record entries to {manifest_file}")
    return manifest_df
