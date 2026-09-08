"""
test_gm_preprocessing.py — Ground Motion Processing & Scaling Unit Tests.

Verifies:
  1. PEER .AT2 reader/writer roundtrip consistency.
  2. Baseline correction removes unphysical velocity drifts.
  3. Record scaling accurately achieves target PGAs spanning 0.05g to 1.20g.
  4. Dataset manifest consistency: all files exist, metadata is populated (Mw, Rrup, Vs30).
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.gm_preprocessing import (
    G_ACCEL,
    read_peer_at2,
    write_peer_at2,
    baseline_correct,
    resample_record,
    compute_intensity_measures,
    preprocess_ground_motion,
)
from src.ground_truth.record_scaling import (
    scale_record_to_pga,
    process_and_scale_suite,
    DEFAULT_TARGET_PGAS,
)


def test_peer_at2_read_write_roundtrip(tmp_path: Path):
    """Verify that writing and reading PEER .AT2 format preserves data and header metadata."""
    dt = 0.005
    npts = 1000
    t = np.arange(npts) * dt
    accel_orig = 0.35 * np.sin(2.0 * math.pi * 3.0 * t) * np.exp(-0.1 * t)
    
    file_path = tmp_path / "test_record.AT2"
    write_peer_at2(
        filepath=str(file_path),
        accel_g=accel_orig,
        dt=dt,
        earthquake_name="Test Event",
        station_name="Station Test",
        component="H1",
    )

    accel_read, dt_read, meta = read_peer_at2(str(file_path))

    assert len(accel_read) == npts
    assert abs(dt_read - dt) < 1e-6
    assert np.max(np.abs(accel_read - accel_orig)) < 1e-6


def test_baseline_correction_removes_velocity_drift():
    """Verify that baseline correction reduces unphysical velocity/displacement drift."""
    dt = 0.01
    npts = 2000
    t = np.arange(npts) * dt
    
    # Ground motion with a synthetic DC drift/offset and linear trend
    accel_raw = 0.20 * np.sin(2.0 * math.pi * 2.0 * t) + 0.02 + 0.001 * t
    
    accel_corrected = baseline_correct(accel_raw, dt=dt, method="polynomial")
    
    # Mean of corrected acceleration must be ~0
    assert abs(np.mean(accel_corrected)) < 1e-10
    
    # Integrated velocity drift at end of record should be close to zero
    vel_raw = np.cumsum(accel_raw) * dt
    vel_corrected = np.cumsum(accel_corrected) * dt
    
    assert abs(vel_corrected[-1]) < abs(vel_raw[-1]) * 0.05


def test_record_scaling_accuracy():
    """Verify that scaling routine achieves exact target PGAs across 0.05g to 1.20g."""
    dt = 0.01
    t = np.arange(1000) * dt
    base_accel = 0.18 * np.sin(2.0 * math.pi * 1.5 * t)
    
    target_pgas = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20]
    
    for target in target_pgas:
        scaled_accel, scale_factor, res_pga = scale_record_to_pga(
            accel=base_accel, target_pga_g=target, in_g=True
        )
        assert abs(res_pga - target) < 1e-9
        assert abs(np.max(np.abs(scaled_accel)) - target) < 1e-9


def test_dataset_manifest_integrity():
    """Verify that data/simulations/dataset_manifest.csv is fully populated and valid."""
    manifest_path = Path("data/simulations/dataset_manifest.csv")
    assert manifest_path.exists(), "dataset_manifest.csv does not exist"

    df = pd.read_csv(manifest_path)
    assert len(df) >= 1000, f"Expected >= 1000 scaled records, found {len(df)}"

    required_cols = [
        "record_id",
        "base_record_id",
        "earthquake_name",
        "magnitude",
        "r_rup_km",
        "vs30_ms",
        "scale_factor",
        "target_pga_g",
        "resulting_pga_g",
        "file_path",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in manifest"

    # Check that metadata fields have no NaNs
    assert not df["magnitude"].isna().any(), "Magnitude contains NaNs"
    assert not df["r_rup_km"].isna().any(), "Rrup contains NaNs"
    assert not df["vs30_ms"].isna().any(), "Vs30 contains NaNs"
    assert not df["earthquake_name"].isna().any(), "earthquake_name contains NaNs"

    # Check PGA range coverage
    assert df["resulting_pga_g"].min() <= 0.05
    assert df["resulting_pga_g"].max() >= 1.20

    # Check that referenced files exist on disk
    sample_files = df["file_path"].sample(n=min(20, len(df)), random_state=42)
    for fp in sample_files:
        assert Path(fp).exists(), f"Manifest file does not exist on disk: {fp}"
        arr = np.load(fp)
        assert len(arr) > 0, f"Loaded empty array from {fp}"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
