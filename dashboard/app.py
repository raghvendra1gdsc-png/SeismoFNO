"""
dashboard/app.py — Interactive Web Dashboard and Demonstration Layer for SeismoFNO.

Provides an interactive graphical and programmatic interface to:
  1. Upload custom ground motion files (.AT2, .txt, .csv, .dat) or select PEER presets.
  2. Configure structural oscillator parameters (T_n, zeta, u_y, alpha, material_type, PGA).
  3. Execute instant SeismoFNO neural operator inference (< 1 ms latency).
  4. Concurrently run OpenSeesPy NLTHA ground truth for exact comparison.
  5. Interactively visualize:
     - Relative displacement time history u(t)
     - Nonlinear hysteretic loops F_R(t) vs. u(t)
     - Cumulative dissipated hysteretic energy E_h(t)
     - Restoring force time history F_R(t)
  6. Display measured speedup metrics, latency breakdowns, and relative L2 errors.

Zero external dependencies required for the server: built on Python's standard `http.server`.
Run locally:
    python3 dashboard/app.py --port 8080
"""

import argparse
import base64
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import socketserver
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add repository root to sys.path for direct execution
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch

from src.models.fno1d import FNO1d
from src.training.train import load_config
from src.data_pipeline.dataset_builder import UnitGaussianNormalizer
from src.ground_truth.opensees_sdof_model import simulate_sdof, SDOFParams, SDOFResponse
from src.data_pipeline.gm_preprocessing import read_peer_at2, baseline_correct, resample_record


# -----------------------------------------------------------------------------
# 1. Global Model and Normalizer Manager
# -----------------------------------------------------------------------------

class DashboardEngine:
    """Manages model loading, inference, and OpenSeesPy execution."""

    def __init__(
        self,
        config_path: str = "configs/phase5_c_data_energy_boundary.yaml",
        checkpoint_path: Optional[str] = None,
        normalizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.device = torch.device(device if device else ("mps" if torch.backends.mps.is_available() else "cpu"))

        # Model Paths
        exp_dir = Path(self.cfg["output"]["checkpoint_dir"])
        self.ckpt_path = Path(checkpoint_path) if checkpoint_path else (exp_dir / "best_model.pt")
        self.norm_path = Path(normalizer_path) if normalizer_path else (exp_dir / "normalizers.pt")

        # Initialize Model Architecture
        model_cfg = self.cfg["model"]
        self.model = FNO1d(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            modes=model_cfg["modes"],
            width=model_cfg["width"],
            n_layers=model_cfg["n_layers"],
        ).to(self.device)

        # Load or initialize normalizers
        self.x_norm = UnitGaussianNormalizer()
        self.y_norm = UnitGaussianNormalizer()
        if self.norm_path.exists():
            norm_data = torch.load(self.norm_path, map_location=self.device, weights_only=False)
            self.x_norm.mean = norm_data["x_mean"].to(self.device)
            self.x_norm.std = norm_data["x_std"].to(self.device)
            self.y_norm.mean = norm_data["y_mean"].to(self.device)
            self.y_norm.std = norm_data["y_std"].to(self.device)
        else:
            # Fallback identity normalizers
            self.x_norm.mean = torch.zeros((1, 10, 1), device=self.device)
            self.x_norm.std = torch.ones((1, 10, 1), device=self.device)
            self.y_norm.mean = torch.zeros((1, 3, 1), device=self.device)
            self.y_norm.std = torch.ones((1, 3, 1), device=self.device)

        # Load weights
        if self.ckpt_path.exists():
            state_dict = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state_dict:
                self.model.load_state_dict(state_dict["model_state_dict"])
            else:
                self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_loaded = True
            # Warm up device compilation / kernel caches
            dummy_x = torch.zeros((1, model_cfg["in_channels"], 2048), device=self.device)
            with torch.no_grad():
                _ = self.model(self.x_norm.encode(dummy_x))
            if self.device.type == "mps":
                torch.mps.synchronize()
        else:
            self.model_loaded = False
            print(f"[Warning] Checkpoint not found at {self.ckpt_path}. Running with uninitialized weights.")

        # Available Preset Records
        self.presets_dir = Path("data/raw/ground_motions")
        self.available_presets = self._discover_presets()

    def _discover_presets(self) -> List[Dict[str, str]]:
        presets = []
        if self.presets_dir.exists():
            for p in sorted(self.presets_dir.glob("*.AT2"))[:20]:
                name = p.stem.replace("RSN", "Record ").replace("_", " ")
                presets.append({
                    "id": p.name,
                    "name": name,
                    "path": str(p),
                })
        if not presets:
            presets.append({"id": "synthetic_ricker", "name": "Synthetic Ricker Wavelet (2 Hz)", "path": ""})
            presets.append({"id": "synthetic_sine", "name": "Harmonic Resonant Sine (1 Hz)", "path": ""})
        return presets

    def load_preset_acceleration(self, preset_id: str, target_pga_g: Optional[float] = None) -> Tuple[np.ndarray, float, str]:
        """Load ground acceleration from preset AT2 file or generate synthetic record."""
        target_dt = 0.01
        n_steps = 2048

        if preset_id.startswith("synthetic"):
            time_arr = np.linspace(0, (n_steps - 1) * target_dt, n_steps)
            if "ricker" in preset_id:
                fp = 2.0
                t0 = 2.0
                tau = np.pi * fp * (time_arr - t0)
                ag = (1.0 - 2.0 * tau**2) * np.exp(-tau**2) * 9.81 * 0.3
            else:
                ag = np.sin(2 * np.pi * 1.0 * time_arr) * np.exp(-0.08 * time_arr) * 9.81 * 0.3
            rec_name = "Synthetic Ground Motion"
        else:
            file_path = self.presets_dir / preset_id
            if not file_path.exists():
                raise FileNotFoundError(f"Preset file not found: {file_path}")
            raw_ag_g, raw_dt, meta = read_peer_at2(str(file_path))
            bc_ag_g = baseline_correct(raw_ag_g, raw_dt)
            resampled_ag_g = resample_record(bc_ag_g, raw_dt, target_dt)

            # Crop or pad to n_steps
            if len(resampled_ag_g) >= n_steps:
                ag = resampled_ag_g[:n_steps] * 9.80665
            else:
                ag = np.pad(resampled_ag_g * 9.80665, (0, n_steps - len(resampled_ag_g)), mode="constant")
            rec_name = meta.get("header_line_1", preset_id)

        # Scale to target PGA if requested
        current_pga_g = np.max(np.abs(ag)) / 9.80665
        if target_pga_g is not None and target_pga_g > 0 and current_pga_g > 1e-6:
            ag = ag * (target_pga_g / current_pga_g)

        return ag.astype(np.float32), target_dt, rec_name

    def parse_uploaded_file(self, file_content: str, filename: str, target_dt: float = 0.01) -> Tuple[np.ndarray, float, str]:
        """Parse user-uploaded ground motion file (.AT2, .txt, .csv)."""
        lines = [line.strip() for line in file_content.splitlines() if line.strip()]
        n_steps = 2048

        if filename.upper().endswith(".AT2") or "NPTS=" in "".join(lines[:6]).upper():
            # Parse PEER format
            meta_line = ""
            for l in lines[:6]:
                if "DT=" in l.upper():
                    meta_line = l
                    break
            dt_match = re.search(r"DT=\s*([0-9.]+)", meta_line, re.IGNORECASE)
            raw_dt = float(dt_match.group(1)) if dt_match else target_dt

            data_lines = lines[4:] if len(lines) > 4 else lines
            values = []
            for l in data_lines:
                values.extend([float(v) for v in l.split() if v])
            raw_ag = np.array(values, dtype=np.float64) * 9.80665
        elif filename.endswith(".csv"):
            # Parse CSV (first column or column named 'ag'/'accel')
            reader = csv.reader(lines)
            rows = list(reader)
            values = []
            header = rows[0]
            col_idx = 0
            start_row = 0
            for i, h in enumerate(header):
                if any(k in h.lower() for k in ["acc", "ag", "g", "val"]):
                    col_idx = i
                    start_row = 1
                    break
            for r in rows[start_row:]:
                if r and len(r) > col_idx:
                    try:
                        values.append(float(r[col_idx]))
                    except ValueError:
                        continue
            raw_ag = np.array(values, dtype=np.float64)
            raw_dt = target_dt
        else:
            # General whitespace-separated text
            values = []
            for l in lines:
                for v in l.split():
                    try:
                        values.append(float(v))
                    except ValueError:
                        continue
            raw_ag = np.array(values, dtype=np.float64)
            raw_dt = target_dt

        # Ensure units in m/s^2 (if max < 5, likely in g)
        if np.max(np.abs(raw_ag)) < 5.0 and np.max(np.abs(raw_ag)) > 0:
            raw_ag = raw_ag * 9.80665

        # Resample and pad
        if abs(raw_dt - target_dt) > 1e-5 and len(raw_ag) > 10:
            raw_ag = resample_record(raw_ag, raw_dt, target_dt)

        if len(raw_ag) >= n_steps:
            ag = raw_ag[:n_steps]
        else:
            ag = np.pad(raw_ag, (0, n_steps - len(raw_ag)), mode="constant")

        return ag.astype(np.float32), target_dt, filename

    def run_simulation(
        self,
        ag: np.ndarray,
        dt: float,
        T: float,
        zeta: float,
        material_type: str,
        u_y: float,
        alpha: float,
        mass: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Execute FNO inference and OpenSeesPy ground truth simulation.
        """
        n_steps = len(ag)
        time_arr = np.linspace(0, (n_steps - 1) * dt, n_steps)
        pga_g = float(np.max(np.abs(ag)) / 9.80665)
        omega_n = 2.0 * np.pi / max(1e-4, T)
        k0 = mass * (omega_n ** 2)

        # -------------------------------------------------------------
        # 1. OpenSeesPy Ground Truth NLTHA
        # -------------------------------------------------------------
        params = SDOFParams(
            T=T,
            zeta=zeta,
            material_type="bilinear" if material_type == "bilinear" else "elastic",
            u_y=u_y if material_type == "bilinear" else None,
            alpha=alpha,
            mass=mass,
        )

        t0_gt = time.perf_counter()
        gt: SDOFResponse = simulate_sdof(params, ag=ag, dt=dt)
        t_gt_ms = (time.perf_counter() - t0_gt) * 1000.0

        # -------------------------------------------------------------
        # 2. SeismoFNO Neural Operator Inference
        # -------------------------------------------------------------
        time_grid = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
        is_bilinear = 1.0 if material_type == "bilinear" else 0.0

        channels = [
            ag.astype(np.float32),
            np.full(n_steps, T, dtype=np.float32),
            np.full(n_steps, omega_n, dtype=np.float32),
            np.full(n_steps, k0, dtype=np.float32),
            np.full(n_steps, zeta, dtype=np.float32),
            np.full(n_steps, is_bilinear, dtype=np.float32),
            np.full(n_steps, u_y if material_type == "bilinear" else 0.0, dtype=np.float32),
            np.full(n_steps, alpha, dtype=np.float32),
            np.full(n_steps, pga_g, dtype=np.float32),
            time_grid,
        ]
        x_tensor = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(self.device)  # [1, 10, N]

        # Warm-up / Sync
        if self.device.type == "mps":
            torch.mps.synchronize()

        t0_fno = time.perf_counter()
        with torch.no_grad():
            x_norm = self.x_norm.encode(x_tensor)
            y_pred_norm = self.model(x_norm)
            y_pred = self.y_norm.decode(y_pred_norm).squeeze(0).cpu().numpy()  # [3, N]

        if self.device.type == "mps":
            torch.mps.synchronize()
        t_fno_ms = (time.perf_counter() - t0_fno) * 1000.0

        u_pred = y_pred[0]
        fr_pred = y_pred[1]
        eh_pred = y_pred[2]

        # -------------------------------------------------------------
        # 3. Compute Metrics and Disaggregation
        # -------------------------------------------------------------
        u_gt = gt.u
        fr_gt = gt.f_r
        eh_gt = gt.e_h

        err_u = float(np.linalg.norm(u_gt - u_pred) / (np.linalg.norm(u_gt) + 1e-6) * 100.0)
        err_fr = float(np.linalg.norm(fr_gt - fr_pred) / (np.linalg.norm(fr_gt) + 1e-6) * 100.0)
        err_eh = float(np.linalg.norm(eh_gt - eh_pred) / (np.linalg.norm(eh_gt) + 1e-6) * 100.0)

        u_max_gt = float(np.max(np.abs(u_gt)))
        u_max_pred = float(np.max(np.abs(u_pred)))
        err_umax = float(abs(u_max_gt - u_max_pred) / (u_max_gt + 1e-6) * 100.0)
        ductility = float(u_max_gt / max(1e-6, u_y)) if material_type == "bilinear" else 1.0
        speedup = float(t_gt_ms / max(1e-4, t_fno_ms))

        # Downsample trajectories for efficient browser JSON transfer (every 2nd point -> 1024 pts)
        step_stride = 2
        return {
            "time": time_arr[::step_stride].tolist(),
            "ag": ag[::step_stride].tolist(),
            "u_gt": u_gt[::step_stride].tolist(),
            "u_pred": u_pred[::step_stride].tolist(),
            "fr_gt": fr_gt[::step_stride].tolist(),
            "fr_pred": fr_pred[::step_stride].tolist(),
            "eh_gt": eh_gt[::step_stride].tolist(),
            "eh_pred": eh_pred[::step_stride].tolist(),
            "metrics": {
                "err_u_rel_l2": round(err_u, 2),
                "err_fr_rel_l2": round(err_fr, 2),
                "err_eh_rel_l2": round(err_eh, 2),
                "err_umax_rel": round(err_umax, 2),
                "u_max_gt_m": round(u_max_gt, 5),
                "u_max_pred_m": round(u_max_pred, 5),
                "ductility_mu": round(ductility, 2),
                "pga_g": round(pga_g, 3),
                "t_fno_ms": round(t_fno_ms, 3),
                "t_gt_ms": round(t_gt_ms, 3),
                "speedup": round(speedup, 1),
                "device": str(self.device),
            }
        }


# -----------------------------------------------------------------------------
# 2. Modern HTML5 / CSS3 / JavaScript Web GUI
# -----------------------------------------------------------------------------

HTML_DASHBOARD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SeismoFNO | Nonlinear Structural Dynamics & Hysteresis Surrogate</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: rgba(18, 26, 44, 0.75);
            --bg-card-hover: rgba(28, 40, 68, 0.85);
            --border-card: rgba(80, 120, 200, 0.2);
            --border-glow: rgba(0, 210, 255, 0.4);
            --accent-cyan: #00d2ff;
            --accent-blue: #0066ff;
            --accent-orange: #ff7a00;
            --accent-emerald: #00e699;
            --accent-rose: #ff3366;
            --text-primary: #f0f4fc;
            --text-secondary: #9ab0d3;
            --text-muted: #5e7399;
            --font-sans: 'Inter', -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-sans);
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(0, 210, 255, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(0, 102, 255, 0.08) 0%, transparent 40%);
            background-attachment: fixed;
        }

        /* Top Header */
        header {
            background: rgba(10, 14, 23, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-card);
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 18px;
            color: #fff;
            box-shadow: 0 0 20px rgba(0, 210, 255, 0.4);
        }

        .brand-text h1 {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #fff, var(--text-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text p {
            font-size: 12px;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }

        .header-badges {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .badge {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-card);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-family: var(--font-mono);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-emerald);
            box-shadow: 0 0 8px var(--accent-emerald);
        }

        /* Layout Grid */
        .dashboard-container {
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 24px;
            padding: 24px 32px;
            flex: 1;
            max-width: 1800px;
            margin: 0 auto;
            width: 100%;
        }

        /* Glassmorphism Cards */
        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .card:hover {
            border-color: rgba(80, 120, 200, 0.35);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 12px;
        }

        .card-title {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Controls Panel */
        .control-group {
            margin-bottom: 18px;
        }

        .control-label {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            font-weight: 500;
        }

        .control-val {
            font-family: var(--font-mono);
            color: var(--accent-cyan);
            font-weight: 600;
        }

        select, input[type="text"] {
            width: 100%;
            background: rgba(10, 14, 23, 0.7);
            border: 1px solid var(--border-card);
            color: var(--text-primary);
            padding: 10px 14px;
            border-radius: 8px;
            font-family: var(--font-sans);
            font-size: 13px;
            outline: none;
            transition: border-color 0.2s;
        }

        select:focus, input[type="text"]:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 10px rgba(0, 210, 255, 0.2);
        }

        input[type="range"] {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-cyan);
            cursor: pointer;
            box-shadow: 0 0 8px var(--accent-cyan);
        }

        /* Drop Zone */
        .drop-zone {
            border: 2px dashed rgba(80, 120, 200, 0.3);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            cursor: pointer;
            background: rgba(0, 0, 0, 0.2);
            transition: all 0.2s;
            margin-top: 10px;
        }

        .drop-zone:hover, .drop-zone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(0, 210, 255, 0.05);
        }

        .drop-zone p {
            font-size: 12px;
            color: var(--text-muted);
        }

        /* Run Button */
        .btn-run {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: #fff;
            border: none;
            padding: 14px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 4px 20px rgba(0, 102, 255, 0.4);
            transition: all 0.2s;
            margin-top: 10px;
        }

        .btn-run:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 24px rgba(0, 210, 255, 0.5);
        }

        .btn-run:active {
            transform: translateY(0);
        }

        /* Metrics Bar */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }

        .metric-box {
            background: rgba(18, 26, 44, 0.6);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .metric-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            font-weight: 600;
        }

        .metric-val {
            font-family: var(--font-mono);
            font-size: 22px;
            font-weight: 700;
            color: #fff;
        }

        .metric-sub {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .text-cyan { color: var(--accent-cyan); }
        .text-emerald { color: var(--accent-emerald); }
        .text-orange { color: var(--accent-orange); }
        .text-rose { color: var(--accent-rose); }

        /* Charts Layout */
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .chart-box {
            background: rgba(10, 14, 23, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 16px;
            position: relative;
            height: 320px;
        }

        .chart-box.full-width {
            grid-column: span 2;
            height: 280px;
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
            .charts-container {
                grid-template-columns: 1fr;
            }
            .chart-box.full-width {
                grid-column: span 1;
            }
            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="brand-container">
            <div class="brand-logo">SF</div>
            <div class="brand-text">
                <h1>SeismoFNO Interactive Surrogate Dashboard</h1>
                <p>Physics-Informed Fourier Neural Operator vs. OpenSeesPy Ground Truth</p>
            </div>
        </div>
        <div class="header-badges">
            <div class="badge">
                <span class="badge-dot"></span>
                <span id="device-badge">MPS / GPU Active</span>
            </div>
            <div class="badge">
                <span>Phase 11 Demo Layer</span>
            </div>
        </div>
    </header>

    <!-- Main Grid -->
    <div class="dashboard-container">

        <!-- Controls Sidebar -->
        <aside class="card">
            <div class="card-header">
                <span class="card-title">⚡ Simulation Controls</span>
            </div>

            <!-- Ground Motion Preset -->
            <div class="control-group">
                <div class="control-label">
                    <span>Ground Motion Excitation</span>
                </div>
                <select id="preset-select">
                    <option value="loading">Loading Presets...</option>
                </select>
                <div class="drop-zone" id="drop-zone">
                    <p id="drop-text">📁 Drag & Drop custom .AT2 / .csv file</p>
                    <input type="file" id="file-input" style="display: none;">
                </div>
            </div>

            <!-- PGA Scale -->
            <div class="control-group">
                <div class="control-label">
                    <span>Target PGA Scaling</span>
                    <span class="control-val" id="pga-val">0.40 g</span>
                </div>
                <input type="range" id="pga-slider" min="0.05" max="1.50" step="0.05" value="0.40">
            </div>

            <div class="card-header" style="margin-top: 24px;">
                <span class="card-title">🏗️ SDOF Structure Parameters</span>
            </div>

            <!-- Natural Period T -->
            <div class="control-group">
                <div class="control-label">
                    <span>Natural Period (T_n)</span>
                    <span class="control-val" id="period-val">0.50 s</span>
                </div>
                <input type="range" id="period-slider" min="0.10" max="2.00" step="0.05" value="0.50">
            </div>

            <!-- Damping Ratio -->
            <div class="control-group">
                <div class="control-label">
                    <span>Damping Ratio (ζ)</span>
                    <span class="control-val" id="damping-val">5.0 %</span>
                </div>
                <input type="range" id="damping-slider" min="0.01" max="0.15" step="0.01" value="0.05">
            </div>

            <!-- Material Type -->
            <div class="control-group">
                <div class="control-label">
                    <span>Constitutive Model</span>
                </div>
                <select id="material-select">
                    <option value="bilinear" selected>Bilinear-Hysteretic (Kinematic Hardening)</option>
                    <option value="elastic">Linear-Elastic</option>
                </select>
            </div>

            <!-- Yield Displacement -->
            <div class="control-group" id="uy-group">
                <div class="control-label">
                    <span>Yield Displacement (u_y)</span>
                    <span class="control-val" id="uy-val">0.010 m</span>
                </div>
                <input type="range" id="uy-slider" min="0.002" max="0.050" step="0.002" value="0.010">
            </div>

            <!-- Post-Yield Ratio -->
            <div class="control-group" id="alpha-group">
                <div class="control-label">
                    <span>Post-Yield Ratio (α)</span>
                    <span class="control-val" id="alpha-val">0.05</span>
                </div>
                <input type="range" id="alpha-slider" min="0.00" max="0.20" step="0.01" value="0.05">
            </div>

            <button class="btn-run" id="btn-run">
                <span>⚡ Run SeismoFNO & OpenSeesPy</span>
            </button>
        </aside>

        <!-- Main Display & Charts -->
        <main>
            <!-- Metrics Summary -->
            <div class="metrics-grid">
                <div class="metric-box">
                    <span class="metric-title">Inference Speedup</span>
                    <span class="metric-val text-cyan" id="metric-speedup">5.8x</span>
                    <span class="metric-sub" id="metric-latency">FNO: 0.8 ms | OpenSees: 4.6 ms</span>
                </div>
                <div class="metric-box">
                    <span class="metric-title">Displacement Rel L2 Error</span>
                    <span class="metric-val text-emerald" id="metric-err-u">24.5 %</span>
                    <span class="metric-sub" id="metric-err-umax">Peak Error: 8.2 %</span>
                </div>
                <div class="metric-box">
                    <span class="metric-title">Ductility Demand (μ)</span>
                    <span class="metric-val text-orange" id="metric-ductility">3.4</span>
                    <span class="metric-sub" id="metric-regime">Post-Yield Inelastic</span>
                </div>
                <div class="metric-box">
                    <span class="metric-title">Hysteretic Energy Rel L2</span>
                    <span class="metric-val text-rose" id="metric-err-eh">18.4 %</span>
                    <span class="metric-sub" id="metric-eh-total">E_h: 124.5 J</span>
                </div>
            </div>

            <!-- Charts Container -->
            <div class="charts-container">
                <!-- Displacement Time History -->
                <div class="card chart-box full-width">
                    <div class="card-header">
                        <span class="card-title">📈 Relative Displacement u(t): SeismoFNO vs. OpenSeesPy Ground Truth</span>
                    </div>
                    <canvas id="chart-u"></canvas>
                </div>

                <!-- Hysteresis Loop -->
                <div class="card chart-box">
                    <div class="card-header">
                        <span class="card-title">🔄 Hysteresis Loop: F_R(t) vs. u(t)</span>
                    </div>
                    <canvas id="chart-hysteresis"></canvas>
                </div>

                <!-- Dissipated Energy -->
                <div class="card chart-box">
                    <div class="card-header">
                        <span class="card-title">⚡ Dissipated Hysteretic Energy E_h(t)</span>
                    </div>
                    <canvas id="chart-energy"></canvas>
                </div>
            </div>
        </main>
    </div>

    <!-- Application JavaScript Logic -->
    <script>
        let chartU = null;
        let chartHysteresis = null;
        let chartEnergy = null;
        let customFileContent = null;
        let customFileName = null;

        // Sliders live updates
        document.getElementById('pga-slider').oninput = (e) => document.getElementById('pga-val').innerText = parseFloat(e.target.value).toFixed(2) + ' g';
        document.getElementById('period-slider').oninput = (e) => document.getElementById('period-val').innerText = parseFloat(e.target.value).toFixed(2) + ' s';
        document.getElementById('damping-slider').oninput = (e) => document.getElementById('damping-val').innerText = (parseFloat(e.target.value) * 100).toFixed(1) + ' %';
        document.getElementById('uy-slider').oninput = (e) => document.getElementById('uy-val').innerText = parseFloat(e.target.value).toFixed(3) + ' m';
        document.getElementById('alpha-slider').oninput = (e) => document.getElementById('alpha-val').innerText = parseFloat(e.target.value).toFixed(2);

        // Material select toggle
        document.getElementById('material-select').onchange = (e) => {
            const isBilinear = e.target.value === 'bilinear';
            document.getElementById('uy-group').style.display = isBilinear ? 'block' : 'none';
            document.getElementById('alpha-group').style.display = isBilinear ? 'block' : 'none';
        };

        // Initialize Charts
        function initCharts() {
            const commonOptions = {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ab0d3', font: { family: 'JetBrains Mono', size: 10 } } },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ab0d3', font: { family: 'JetBrains Mono', size: 10 } } }
                },
                plugins: {
                    legend: { labels: { color: '#f0f4fc', font: { family: 'Inter', size: 11 } } }
                }
            };

            // Chart U
            chartU = new Chart(document.getElementById('chart-u'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'SeismoFNO Prediction (Cyan)', borderColor: '#00d2ff', borderWidth: 2, data: [], pointRadius: 0 },
                        { label: 'OpenSeesPy Ground Truth (Orange)', borderColor: '#ff7a00', borderWidth: 2, borderDash: [4, 4], data: [], pointRadius: 0 }
                    ]
                },
                options: commonOptions
            });

            // Chart Hysteresis
            chartHysteresis = new Chart(document.getElementById('chart-hysteresis'), {
                type: 'line',
                data: {
                    datasets: [
                        { label: 'SeismoFNO Hysteresis', borderColor: '#00d2ff', borderWidth: 1.5, data: [], pointRadius: 0 },
                        { label: 'OpenSeesPy GT', borderColor: '#ff7a00', borderWidth: 1.5, borderDash: [4, 4], data: [], pointRadius: 0 }
                    ]
                },
                options: {
                    ...commonOptions,
                    scales: {
                        x: { title: { display: true, text: 'Displacement u (m)', color: '#9ab0d3' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ab0d3' } },
                        y: { title: { display: true, text: 'Restoring Force F_R (N)', color: '#9ab0d3' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ab0d3' } }
                    }
                }
            });

            // Chart Energy
            chartEnergy = new Chart(document.getElementById('chart-energy'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'SeismoFNO Dissipated Energy E_h(t)', borderColor: '#ff3366', backgroundColor: 'rgba(255, 51, 102, 0.1)', fill: true, borderWidth: 2, data: [], pointRadius: 0 },
                        { label: 'OpenSeesPy E_h(t)', borderColor: '#00e699', borderWidth: 2, borderDash: [4, 4], data: [], pointRadius: 0 }
                    ]
                },
                options: commonOptions
            });
        }

        // Fetch Presets
        async function fetchPresets() {
            try {
                const res = await fetch('/api/presets');
                const data = await res.json();
                const sel = document.getElementById('preset-select');
                sel.innerHTML = '';
                data.presets.forEach(p => {
                    const opt = document.createElement('option');
                    opt.value = p.id;
                    opt.innerText = p.name;
                    sel.appendChild(opt);
                });
            } catch (err) {
                console.error('Failed to load presets:', err);
            }
        }

        // Run Simulation Action
        async function runSimulation() {
            const btn = document.getElementById('btn-run');
            btn.disabled = true;
            btn.innerHTML = '<span>⏳ Computing NLTHA & FNO...</span>';

            const payload = {
                preset_id: document.getElementById('preset-select').value,
                pga_g: parseFloat(document.getElementById('pga-slider').value),
                T: parseFloat(document.getElementById('period-slider').value),
                zeta: parseFloat(document.getElementById('damping-slider').value),
                material_type: document.getElementById('material-select').value,
                u_y: parseFloat(document.getElementById('uy-slider').value),
                alpha: parseFloat(document.getElementById('alpha-slider').value),
                custom_content: customFileContent,
                custom_filename: customFileName
            };

            try {
                const res = await fetch('/api/simulate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                // Update Metrics
                const m = data.metrics;
                document.getElementById('metric-speedup').innerText = m.speedup + 'x';
                document.getElementById('metric-latency').innerText = `FNO: ${m.t_fno_ms} ms | OpenSees: ${m.t_gt_ms} ms`;
                document.getElementById('metric-err-u').innerText = m.err_u_rel_l2 + ' %';
                document.getElementById('metric-err-umax').innerText = `Peak Error: ${m.err_umax_rel} % (GT: ${m.u_max_gt_m}m)`;
                document.getElementById('metric-ductility').innerText = m.ductility_mu;
                document.getElementById('metric-regime').innerText = m.ductility_mu > 1.0 ? 'Inelastic Post-Yield' : 'Linear-Elastic';
                document.getElementById('metric-err-eh').innerText = m.err_eh_rel_l2 + ' %';
                document.getElementById('device-badge').innerText = m.device.toUpperCase() + ' Active';

                // Update Chart U
                chartU.data.labels = data.time.map(t => t.toFixed(2));
                chartU.data.datasets[0].data = data.u_pred;
                chartU.data.datasets[1].data = data.u_gt;
                chartU.update();

                // Update Chart Hysteresis
                const hystPred = data.u_pred.map((u, i) => ({ x: u, y: data.fr_pred[i] }));
                const hystGT = data.u_gt.map((u, i) => ({ x: u, y: data.fr_gt[i] }));
                chartHysteresis.data.datasets[0].data = hystPred;
                chartHysteresis.data.datasets[1].data = hystGT;
                chartHysteresis.update();

                // Update Chart Energy
                chartEnergy.data.labels = data.time.map(t => t.toFixed(2));
                chartEnergy.data.datasets[0].data = data.eh_pred;
                chartEnergy.data.datasets[1].data = data.eh_gt;
                chartEnergy.update();

            } catch (err) {
                alert('Simulation error: ' + err);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>⚡ Run SeismoFNO & OpenSeesPy</span>';
            }
        }

        // File Drag and Drop Handling
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');

        dropZone.onclick = () => fileInput.click();
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
        dropZone.ondragleave = () => dropZone.classList.remove('dragover');
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        };
        fileInput.onchange = (e) => {
            if (e.target.files.length) handleFile(e.target.files[0]);
        };

        function handleFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                customFileContent = e.target.result;
                customFileName = file.name;
                document.getElementById('drop-text').innerText = '✅ ' + file.name;
                document.getElementById('preset-select').disabled = true;
            };
            reader.readAsText(file);
        }

        // Init
        document.getElementById('btn-run').onclick = runSimulation;
        window.onload = async () => {
            initCharts();
            await fetchPresets();
            runSimulation();
        };
    </script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# 3. HTTP Request Handler
# -----------------------------------------------------------------------------

class SeismoFNODashboardHandler(BaseHTTPRequestHandler):
    """HTTP server handler routing REST APIs and static dashboard page."""

    engine: DashboardEngine = None

    def _set_headers(self, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=204)

    def do_GET(self):
        dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"

        if self.path == "/api/presets":
            self._set_headers("application/json")
            presets = self.engine.available_presets
            self.wfile.write(json.dumps({"presets": presets}).encode("utf-8"))
        elif self.path == "/api/health":
            self._set_headers("application/json")
            res = {
                "status": "online",
                "model_loaded": self.engine.model_loaded,
                "device": str(self.engine.device),
                "num_params": self.engine.model.get_num_parameters(),
            }
            self.wfile.write(json.dumps(res).encode("utf-8"))
        elif dist_dir.exists():
            # Serve static files from React build
            clean_path = self.path.split("?")[0].lstrip("/")
            file_path = dist_dir / clean_path if clean_path else dist_dir / "index.html"
            if not file_path.exists() or file_path.is_dir():
                file_path = dist_dir / "index.html"

            mime_types = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".ico": "image/x-icon",
                ".json": "application/json",
            }
            content_type = mime_types.get(file_path.suffix.lower(), "application/octet-stream")
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                self._set_headers(content_type)
                self.wfile.write(content)
            except Exception as e:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
        elif self.path == "/" or self.path == "/index.html":
            self._set_headers("text/html; charset=utf-8")
            self.wfile.write(HTML_DASHBOARD_PAGE.encode("utf-8"))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Resource not found")

    def do_POST(self):
        if self.path == "/api/simulate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                pga_g = float(data.get("pga_g", 0.4))
                T = float(data.get("T", 0.5))
                zeta = float(data.get("zeta", 0.05))
                material_type = str(data.get("material_type", "bilinear"))
                u_y = float(data.get("u_y", 0.01))
                alpha = float(data.get("alpha", 0.05))

                # Load Ground Motion
                if data.get("custom_content") and data.get("custom_filename"):
                    ag, dt, _ = self.engine.parse_uploaded_file(
                        data["custom_content"], data["custom_filename"]
                    )
                else:
                    preset_id = data.get("preset_id", "RSN0001_Imperial_Valley-06.AT2")
                    ag, dt, _ = self.engine.load_preset_acceleration(preset_id, target_pga_g=pga_g)

                # Scale PGA if specified
                current_pga = np.max(np.abs(ag)) / 9.80665
                if current_pga > 1e-6 and abs(current_pga - pga_g) > 1e-3:
                    ag = ag * (pga_g / current_pga)

                # Run simulation
                res = self.engine.run_simulation(
                    ag=ag,
                    dt=dt,
                    T=T,
                    zeta=zeta,
                    material_type=material_type,
                    u_y=u_y,
                    alpha=alpha,
                )

                self._set_headers("application/json")
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self._set_headers("application/json", status=500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def log_message(self, format, *args):
        # Quiet server logs for clean stdout
        return


# -----------------------------------------------------------------------------
# 4. CLI Runner and Server Launcher
# -----------------------------------------------------------------------------

def start_server(port: int = 8080, engine: Optional[DashboardEngine] = None):
    """Launch the local HTTP server."""
    if engine is None:
        engine = DashboardEngine()
    SeismoFNODashboardHandler.engine = engine

    server_address = ("", port)
    httpd = HTTPServer(server_address, SeismoFNODashboardHandler)
    print(f"=" * 80)
    print(f"🚀 SEISMOFNO INTERACTIVE DASHBOARD RUNNING")
    print(f"🔗 Local Web URL: http://localhost:{port}")
    print(f"⚙️  Device: {engine.device} | Trainable Parameters: {engine.model.get_num_parameters():,}")
    print(f"=" * 80)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


def run_cli_demo(
    preset_id: str = "RSN0001_Imperial_Valley-06.AT2",
    T: float = 0.5,
    zeta: float = 0.05,
    pga: float = 0.40,
    u_y: float = 0.010,
    alpha: float = 0.05,
    material_type: str = "bilinear",
):
    """Run interactive simulation via command line."""
    engine = DashboardEngine()
    ag, dt, name = engine.load_preset_acceleration(preset_id, target_pga_g=pga)
    print(f"\n[SeismoFNO CLI Demo] Running Record: {name} (PGA={pga:.2f}g, T={T:.2f}s, u_y={u_y*1000:.1f}mm)")
    res = engine.run_simulation(ag=ag, dt=dt, T=T, zeta=zeta, material_type=material_type, u_y=u_y, alpha=alpha)
    m = res["metrics"]
    print("=" * 60)
    print(f"  Speedup Factor        : {m['speedup']}x (FNO: {m['t_fno_ms']} ms vs OpenSees: {m['t_gt_ms']} ms)")
    print(f"  Displacement Rel L2   : {m['err_u_rel_l2']}%")
    print(f"  Peak Displacement GT  : {m['u_max_gt_m'] * 1000:.2f} mm | FNO: {m['u_max_pred_m'] * 1000:.2f} mm")
    print(f"  Ductility Demand (mu) : {m['ductility_mu']}")
    print(f"  Hysteretic Energy L2  : {m['err_eh_rel_l2']}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeismoFNO Interactive Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve dashboard (default 8080)")
    parser.add_argument("--cli", action="store_true", help="Run quick CLI evaluation instead of starting server")
    parser.add_argument("--preset", type=str, default="RSN0001_Imperial_Valley-06.AT2", help="Preset record ID for CLI")
    parser.add_argument("--period", type=float, default=0.5, help="Period T [s] for CLI")
    parser.add_argument("--pga", type=float, default=0.4, help="PGA [g] for CLI")
    args = parser.parse_args()

    if args.cli:
        run_cli_demo(preset_id=args.preset, T=args.period, pga=args.pga)
    else:
        start_server(port=args.port)
