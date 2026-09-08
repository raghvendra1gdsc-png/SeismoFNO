"""
seismo_agent/tools/earthquake_tool.py — Deterministic Earthquake Analysis Tool.

Wraps existing research routines in `src.data_pipeline.gm_preprocessing` and
`src.ground_truth.record_scaling` to parse, clean, baseline-correct, resample,
scale, and compute seismic intensity measures (PGA, PGV, PGD, Arias, D5-95).
"""

import csv
from pathlib import Path
import re
from typing import Tuple, Dict, Any, Optional
import numpy as np

from seismo_agent.config import REPO_ROOT, DEFAULT_PRESETS_DIR
from seismo_agent.schemas.inputs import EarthquakeRequest
from seismo_agent.schemas.outputs import EarthquakeResult
from seismo_agent.tools.base import BaseTool, ToolExecutionError
from src.data_pipeline.gm_preprocessing import (
    read_peer_at2,
    baseline_correct,
    resample_record,
    apply_cosine_taper,
    compute_intensity_measures,
    G_ACCEL,
)
from src.ground_truth.record_scaling import scale_record_to_pga


class EarthquakeTool(BaseTool):
    """
    Analyzes, processes, and extracts ground motion intensity measures from seismic time series.
    """
    name = "analyze_earthquake_record"
    description = (
        "Parses, baseline-corrects, resamples, and calculates seismic intensity measures "
        "(Peak Ground Acceleration PGA, Peak Ground Velocity PGV, Peak Ground Displacement PGD, "
        "Arias Intensity Ia, and Significant Duration D5-95) for a designated ground motion. "
        "Accepts standard PEER NGA-West2 (.AT2) files, tabular text/CSV files, or preset records. "
        "Supports optional amplitude scaling to target PGAs."
    )
    input_schema = EarthquakeRequest
    output_schema = EarthquakeResult

    def __init__(self, presets_dir: Optional[Path] = None):
        self.presets_dir = presets_dir or DEFAULT_PRESETS_DIR

    def _load_or_parse_raw_acceleration(self, request: EarthquakeRequest) -> Tuple[np.ndarray, float, str, Dict[str, Any]]:
        """Extract raw acceleration array in m/s^2, original dt, record name, and metadata."""
        # 1. Direct series provided
        if request.raw_accel_series and len(request.raw_accel_series) > 0:
            raw_ag = np.asarray(request.raw_accel_series, dtype=np.float64)
            dt = request.target_dt
            name = request.record_id or "Direct Input Acceleration"
            meta = {"source": "direct_array"}
            return raw_ag, dt, name, meta

        # 2. Preset record requested
        if request.record_id:
            preset_id = request.record_id.strip()
            if preset_id.startswith("synthetic"):
                dt = request.target_dt
                n_steps = request.n_steps
                t = np.linspace(0, (n_steps - 1) * dt, n_steps)
                if "ricker" in preset_id:
                    fp = 2.0
                    t0 = 2.0
                    tau = np.pi * fp * (t - t0)
                    ag = (1.0 - 2.0 * tau**2) * np.exp(-tau**2) * G_ACCEL * 0.35
                    name = "Synthetic Ricker Wavelet (2 Hz)"
                else:
                    ag = np.sin(2 * np.pi * 1.0 * t) * np.exp(-0.08 * t) * G_ACCEL * 0.35
                    name = "Synthetic Harmonic Damped Sine (1 Hz)"
                return ag.astype(np.float64), dt, name, {"preset": preset_id}

            preset_path = self.presets_dir / preset_id
            if not preset_path.exists():
                # Check with .AT2 extension
                preset_path_at2 = self.presets_dir / f"{preset_id}.AT2"
                if preset_path_at2.exists():
                    preset_path = preset_path_at2
                else:
                    raise ToolExecutionError(
                        self.name,
                        f"Preset earthquake record '{preset_id}' not found in {self.presets_dir}."
                    )
            raw_ag_g, raw_dt, meta = read_peer_at2(str(preset_path))
            raw_ag_ms2 = raw_ag_g * G_ACCEL
            rec_name = meta.get("header_line_1", preset_id)
            return raw_ag_ms2, raw_dt, rec_name, meta

        # 3. Custom file content uploaded
        if request.custom_content and request.custom_filename:
            content = request.custom_content
            fname = request.custom_filename.strip()
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if not lines:
                raise ToolExecutionError(self.name, "Uploaded custom file is empty.")

            # PEER AT2 format detection
            if fname.upper().endswith(".AT2") or any("DT=" in l.upper() for l in lines[:6]):
                meta_line = ""
                for l in lines[:6]:
                    if "DT=" in l.upper():
                        meta_line = l
                        break
                dt_match = re.search(r"DT=\s*([0-9.]+)", meta_line, re.IGNORECASE)
                raw_dt = float(dt_match.group(1)) if dt_match else request.target_dt

                data_lines = lines[4:] if len(lines) > 4 else lines
                values = []
                for l in data_lines:
                    values.extend([float(v) for v in l.split() if v])
                raw_ag_g = np.array(values, dtype=np.float64)
                raw_ag_ms2 = raw_ag_g * G_ACCEL
                meta = {"header": lines[0] if lines else "Custom PEER Record"}
                return raw_ag_ms2, raw_dt, fname, meta

            # CSV format
            elif fname.lower().endswith(".csv"):
                reader = csv.reader(lines)
                rows = list(reader)
                if not rows:
                    raise ToolExecutionError(self.name, "CSV file has no rows.")
                header = rows[0]
                col_idx = 0
                start_row = 0
                for i, h in enumerate(header):
                    if any(k in h.lower() for k in ["acc", "ag", "g", "val"]):
                        col_idx = i
                        start_row = 1
                        break
                values = []
                for r in rows[start_row:]:
                    if r and len(r) > col_idx:
                        try:
                            values.append(float(r[col_idx]))
                        except ValueError:
                            continue
                if not values:
                    raise ToolExecutionError(self.name, "No numerical acceleration data found in CSV.")
                raw_ag = np.array(values, dtype=np.float64)
                # If peak < 5.0, assume units of g
                if np.max(np.abs(raw_ag)) < 5.0 and np.max(np.abs(raw_ag)) > 0:
                    raw_ag = raw_ag * G_ACCEL
                return raw_ag, request.target_dt, fname, {"format": "csv"}

            # Whitespace-delimited table
            else:
                values = []
                for l in lines:
                    for v in l.split():
                        try:
                            values.append(float(v))
                        except ValueError:
                            continue
                if not values:
                    raise ToolExecutionError(self.name, "No numerical data could be parsed from text file.")
                raw_ag = np.array(values, dtype=np.float64)
                if np.max(np.abs(raw_ag)) < 5.0 and np.max(np.abs(raw_ag)) > 0:
                    raw_ag = raw_ag * G_ACCEL
                return raw_ag, request.target_dt, fname, {"format": "whitespace_text"}

        raise ToolExecutionError(self.name, "Failed to resolve earthquake acceleration source.")

    def _run(self, request: EarthquakeRequest) -> EarthquakeResult:
        raw_ag, orig_dt, name, meta = self._load_or_parse_raw_acceleration(request)

        # Validation: check for NaN/Inf
        if np.any(np.isnan(raw_ag)) or np.any(np.isinf(raw_ag)):
            raise ToolExecutionError(self.name, "Ground motion record contains NaN or Infinite numerical values.")

        if orig_dt <= 0.0:
            raise ToolExecutionError(self.name, f"Invalid sampling time step dt={orig_dt} (must be strictly positive).")

        if len(raw_ag) < 10:
            raise ToolExecutionError(self.name, f"Ground motion record too short: contains only {len(raw_ag)} points.")

        # 1. Baseline Correction
        if request.apply_baseline_correction:
            accel_clean = baseline_correct(raw_ag, orig_dt, method=request.baseline_method)
        else:
            accel_clean = raw_ag - np.mean(raw_ag)

        # 2. Resampling to target_dt
        target_dt = request.target_dt
        if abs(orig_dt - target_dt) > 1e-6:
            accel_resampled = resample_record(accel_clean, orig_dt, target_dt)
        else:
            accel_resampled = accel_clean

        # 3. End Tapering
        accel_tapered = apply_cosine_taper(accel_resampled, fraction=0.01)

        # 4. Length crop/pad to target n_steps
        n_steps = request.n_steps
        if len(accel_tapered) >= n_steps:
            accel_final = accel_tapered[:n_steps]
        else:
            accel_final = np.pad(accel_tapered, (0, n_steps - len(accel_tapered)), mode="constant")

        # 5. Amplitude Scaling (if target_pga_g is specified)
        current_pga_g = float(np.max(np.abs(accel_final)) / G_ACCEL)
        scale_factor = 1.0
        if request.target_pga_g is not None and request.target_pga_g > 0.0:
            if current_pga_g < 1e-8:
                raise ToolExecutionError(self.name, "Cannot scale a zero-amplitude ground motion record.")
            scaled_ag, scale_factor, res_pga = scale_record_to_pga(
                accel=accel_final,
                target_pga_g=request.target_pga_g,
                current_pga_g=current_pga_g,
                in_g=False,
            )
            accel_final = scaled_ag

        # 6. Intensity Measures Calculation
        ims = compute_intensity_measures(accel_final, target_dt, in_g=False)

        # Downsample preview for concise payload transmission
        stride = max(1, len(accel_final) // 100)
        preview = [round(float(v), 5) for v in accel_final[::stride][:100]]

        return EarthquakeResult(
            record_id=request.record_id or request.custom_filename or "direct_record",
            record_name=name,
            n_points=len(accel_final),
            dt=target_dt,
            duration_s=round((len(accel_final) - 1) * target_dt, 3),
            pga_g=round(ims["pga_g"], 4),
            pga_ms2=round(ims["pga_ms2"], 4),
            pgv_ms=round(ims["pgv_ms"], 4),
            pgd_m=round(ims["pgd_m"], 5),
            arias_intensity_ms=round(ims["arias_intensity_ms"], 4),
            significant_duration_d5_95_s=round(ims["d5_95_s"], 3),
            scale_factor_applied=round(scale_factor, 4),
            metadata=meta,
            accel_preview=preview,
        )

    def get_processed_acceleration_array(self, request: EarthquakeRequest) -> Tuple[np.ndarray, float, str]:
        """
        Helper method returning the raw NumPy acceleration array in m/s^2, target dt, and name.
        Allows other tools (Inference, Physics, Sweep) to consume preprocessed series directly.
        """
        raw_ag, orig_dt, name, _ = self._load_or_parse_raw_acceleration(request)
        if request.apply_baseline_correction:
            accel_clean = baseline_correct(raw_ag, orig_dt, method=request.baseline_method)
        else:
            accel_clean = raw_ag - np.mean(raw_ag)

        target_dt = request.target_dt
        if abs(orig_dt - target_dt) > 1e-6:
            accel_resampled = resample_record(accel_clean, orig_dt, target_dt)
        else:
            accel_resampled = accel_clean

        accel_tapered = apply_cosine_taper(accel_resampled, fraction=0.01)
        n_steps = request.n_steps
        if len(accel_tapered) >= n_steps:
            accel_final = accel_tapered[:n_steps]
        else:
            accel_final = np.pad(accel_tapered, (0, n_steps - len(accel_tapered)), mode="constant")

        current_pga_g = float(np.max(np.abs(accel_final)) / G_ACCEL)
        if request.target_pga_g is not None and request.target_pga_g > 0.0:
            if current_pga_g > 1e-8:
                accel_final = accel_final * (request.target_pga_g / current_pga_g)

        return accel_final.astype(np.float64), target_dt, name
