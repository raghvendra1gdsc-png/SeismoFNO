"""
inference/adapter.py — Isolated Inference Adapter for SeismoFNO Surrogate Models.

Provides a clean, decoupled boundary between external application layers (FastAPI, UI)
and the underlying PyTorch neural operator models. Never modifies frozen research code.
"""

from dataclasses import dataclass, field
import os
import threading
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch

from src.models.fno1d import FNO1d
from src.training.train import load_config
from src.data_pipeline.dataset_builder import UnitGaussianNormalizer
from src.utils.device import GLOBAL_DEVICE_LOCK, resolve_device, sync_device


@dataclass
class InferenceOutput:
    """Standardized output container for SeismoFNO surrogate inference."""
    time: np.ndarray
    ag: np.ndarray
    u: np.ndarray
    v: np.ndarray
    fr: np.ndarray
    up: np.ndarray
    eh: np.ndarray
    latency_ms: float
    is_mock: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, stride: int = 1) -> Dict[str, Any]:
        """Convert numpy arrays to serializable JSON-friendly lists."""
        return {
            "time": self.time[::stride].tolist(),
            "ag": self.ag[::stride].tolist(),
            "u": self.u[::stride].tolist(),
            "v": self.v[::stride].tolist(),
            "fr": self.fr[::stride].tolist(),
            "up": self.up[::stride].tolist(),
            "eh": self.eh[::stride].tolist(),
            "latency_ms": round(self.latency_ms, 3),
            "is_mock": self.is_mock,
            "metadata": self.metadata,
        }


class SeismoFNOInferenceAdapter:
    """
    Production-grade inference adapter for the SeismoFNO neural operator surrogate.
    
    Adheres strictly to the research freeze boundary:
    - Treats checkpoints and configs as read-only.
    - Encodes physical parameters into the 10-channel representation required by FNO-1D.
    - Decodes normalized neural outputs into physical engineering quantities.
    - Deterministically derives velocity v(t) and plastic displacement up(t).
    - Falls back to deterministic mock response if checkpoint is unavailable, clearly labeled.
    """

    DEFAULT_CONFIG_PATH = Path("configs/phase5_c_data_energy_boundary.yaml")
    DEFAULT_CHECKPOINT_PATH = Path("experiments/phase5_c_data_energy_boundary/checkpoints/best_model.pt")
    DEFAULT_NORMALIZER_PATH = Path("experiments/phase5_c_data_energy_boundary/checkpoints/normalizers.pt")

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        normalizer_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.DEFAULT_CHECKPOINT_PATH
        self.normalizer_path = Path(normalizer_path) if normalizer_path else self.DEFAULT_NORMALIZER_PATH

        self.device = self._resolve_device(device)
        self._lock = GLOBAL_DEVICE_LOCK
        self.model: Optional[FNO1d] = None
        self.x_norm: Optional[UnitGaussianNormalizer] = None
        self.y_norm: Optional[UnitGaussianNormalizer] = None
        self.is_loaded: bool = False
        self.model_info: Dict[str, Any] = {}

        self.load_model()

    def _resolve_device(self, preferred: Optional[str] = None) -> torch.device:
        return resolve_device(preferred)

    def _sync(self):
        """Synchronize device execution queues for precise wall-clock timing."""
        sync_device(self.device)

    def load_model(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        normalizer_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Load FNO-1D model weights and normalizers into memory."""
        with GLOBAL_DEVICE_LOCK:
            if config_path:
                self.config_path = Path(config_path)
            if checkpoint_path:
                self.checkpoint_path = Path(checkpoint_path)
            if normalizer_path:
                self.normalizer_path = Path(normalizer_path)

            if not self.config_path.exists():
                self.is_loaded = False
                self.model_info = {"status": "unloaded", "reason": f"Config not found at {self.config_path}"}
                return False

            try:
                cfg = load_config(str(self.config_path))
                model_cfg = cfg.get("model", {})
                self.model = FNO1d(
                    in_channels=model_cfg.get("in_channels", 10),
                    out_channels=model_cfg.get("out_channels", 3),
                    modes=model_cfg.get("modes", 32),
                    width=model_cfg.get("width", 64),
                    n_layers=model_cfg.get("n_layers", 4),
                ).to(self.device)

                self.x_norm = UnitGaussianNormalizer()
                self.y_norm = UnitGaussianNormalizer()

                if self.normalizer_path.exists():
                    norm_data = torch.load(self.normalizer_path, map_location=self.device, weights_only=False)
                    self.x_norm.mean = norm_data["x_mean"].to(self.device)
                    self.x_norm.std = norm_data["x_std"].to(self.device)
                    self.y_norm.mean = norm_data["y_mean"].to(self.device)
                    self.y_norm.std = norm_data["y_std"].to(self.device)
                else:
                    self.x_norm.mean = torch.zeros((1, 10, 1), device=self.device)
                    self.x_norm.std = torch.ones((1, 10, 1), device=self.device)
                    self.y_norm.mean = torch.zeros((1, 3, 1), device=self.device)
                    self.y_norm.std = torch.ones((1, 3, 1), device=self.device)

                if self.checkpoint_path.exists():
                    state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                    if isinstance(state, dict) and "model_state_dict" in state:
                        self.model.load_state_dict(state["model_state_dict"])
                    elif isinstance(state, dict):
                        self.model.load_state_dict(state)
                    self.model.eval()
                    self.is_loaded = True

                    # Warm up accelerator kernel caches
                    dummy_x = torch.zeros((1, model_cfg.get("in_channels", 10), 2048), device=self.device)
                    with torch.no_grad():
                        _ = self.model(self.x_norm.encode(dummy_x))
                    self._sync()

                    total_params = sum(p.numel() for p in self.model.parameters())
                    self.model_info = {
                        "status": "loaded",
                        "checkpoint": str(self.checkpoint_path),
                        "parameters": total_params,
                        "modes": model_cfg.get("modes", 32),
                        "width": model_cfg.get("width", 64),
                        "device": str(self.device),
                        "in_channels": model_cfg.get("in_channels", 10),
                        "out_channels": model_cfg.get("out_channels", 3),
                    }
                    return True
                else:
                    self.is_loaded = False
                    self.model_info = {
                        "status": "unloaded",
                        "reason": f"Checkpoint not found at {self.checkpoint_path}",
                        "mock_available": True,
                    }
                    return False
            except Exception as exc:
                self.is_loaded = False
                self.model_info = {"status": "error", "error": str(exc), "mock_available": True}
                return False

    def validate_input(self, scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate structural oscillator parameters against physical domain boundaries.
        """
        errors = []
        T = scenario.get("T0", scenario.get("T", 0.5))
        zeta = scenario.get("damping", scenario.get("zeta", 0.05))
        uy = scenario.get("yield_displacement", scenario.get("uy", 0.01))
        alpha = scenario.get("alpha", 0.05)
        pga = scenario.get("pga", scenario.get("pga_g", 0.4))

        if T <= 0.02 or T > 10.0:
            errors.append(f"Structural period T={T:.3f}s must be between 0.02s and 10.0s.")
        if zeta < 0.0 or zeta > 0.5:
            errors.append(f"Damping ratio zeta={zeta:.3f} must be between 0.0 and 0.5.")
        if uy <= 0.0 or uy > 5.0:
            errors.append(f"Yield displacement uy={uy:.4f}m must be positive and <= 5.0m.")
        if alpha < 0.0 or alpha > 1.0:
            errors.append(f"Post-yield stiffness ratio alpha={alpha:.3f} must be in [0.0, 1.0].")
        if pga <= 0.0 or pga > 5.0:
            errors.append(f"PGA={pga:.3f}g must be in (0.0g, 5.0g].")

        return len(errors) == 0, errors

    def predict(
        self,
        scenario: Dict[str, Any],
        ag: np.ndarray,
        dt: float = 0.01,
        n_steps: int = 2048,
    ) -> InferenceOutput:
        """
        Execute forward surrogate prediction or deterministic mock fallback.
        
        Parameters
        ----------
        scenario : Dict containing T0, zeta, uy, alpha, material_type, mass, etc.
        ag : 1D numpy array of ground acceleration in m/s^2.
        dt : time step in seconds (default 0.01s).
        n_steps : number of time points (default 2048).
        """
        is_valid, errors = self.validate_input(scenario)
        if not is_valid:
            raise ValueError(f"Invalid structural scenario parameters: {'; '.join(errors)}")

        T = float(scenario.get("T0", scenario.get("T", 0.5)))
        zeta = float(scenario.get("damping", scenario.get("zeta", 0.05)))
        uy = float(scenario.get("yield_displacement", scenario.get("uy", 0.01)))
        alpha = float(scenario.get("alpha", 0.05))
        material_type = str(scenario.get("material_type", "bilinear")).lower()
        is_bilinear = 1.0 if material_type == "bilinear" else 0.0
        mass = float(scenario.get("mass", 1.0))

        omega_n = 2.0 * np.pi / max(1e-4, T)
        k0 = mass * (omega_n ** 2)

        # Resample / Pad ag to n_steps
        if len(ag) >= n_steps:
            ag_proc = ag[:n_steps].astype(np.float32)
        else:
            ag_proc = np.pad(ag.astype(np.float32), (0, n_steps - len(ag)), mode="constant")

        pga_g = float(np.max(np.abs(ag_proc)) / 9.80665)
        time_arr = np.linspace(0, (n_steps - 1) * dt, n_steps, dtype=np.float32)

        # Check if real model is available
        if self.is_loaded and self.model is not None and self.x_norm is not None and self.y_norm is not None:
            time_grid = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
            channels = [
                ag_proc,
                np.full(n_steps, T, dtype=np.float32),
                np.full(n_steps, omega_n, dtype=np.float32),
                np.full(n_steps, k0 / mass, dtype=np.float32),
                np.full(n_steps, zeta, dtype=np.float32),
                np.full(n_steps, is_bilinear, dtype=np.float32),
                np.full(n_steps, uy if is_bilinear else 0.0, dtype=np.float32),
                np.full(n_steps, alpha, dtype=np.float32),
                np.full(n_steps, pga_g, dtype=np.float32),
                time_grid,
            ]
            with self._lock:
                x_tensor = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(self.device)
                self._sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    # Protect against zero-variance training channels (e.g. fixed zeta=0.05 where std ≈ 1e-6)
                    safe_std = torch.clamp(self.x_norm.std, min=0.02)
                    x_norm = (x_tensor - self.x_norm.mean) / safe_std
                    y_pred_norm = self.model(x_norm)
                    y_pred = self.y_norm.decode(y_pred_norm).squeeze(0).cpu().numpy()
                self._sync()
                latency_ms = (time.perf_counter() - t0) * 1000.0

            u = y_pred[0]
            fr = y_pred[1] * mass  # N
            eh = y_pred[2] * mass  # J
            is_mock = False
            meta = {
                "source": "SeismoFNO Continuous Spectral Operator",
                "checkpoint": str(self.checkpoint_path),
                "device": str(self.device),
                "modes": self.model_info.get("modes", 32),
            }
        else:
            # Deterministic linear/bilinear analytical mock
            t0 = time.perf_counter()
            u, fr, eh = self._deterministic_sdof_mock(ag_proc, dt, T, zeta, is_bilinear, uy, alpha, mass)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            is_mock = True
            meta = {
                "source": "Deterministic SDOF Numerical Mock (Model Checkpoint Unavailable)",
                "device": "cpu",
            }

        # Deterministically derive velocity v(t) via central differences
        v = np.gradient(u, dt).astype(np.float32)

        # Deterministically derive plastic displacement: up(t) = u(t) - FR(t)/k0
        k0_eff = max(1e-4, k0)
        up = (u - (fr / k0_eff)).astype(np.float32)
        if not is_bilinear:
            up = np.zeros_like(u)

        return InferenceOutput(
            time=time_arr,
            ag=ag_proc,
            u=u.astype(np.float32),
            v=v,
            fr=fr.astype(np.float32),
            up=up,
            eh=eh.astype(np.float32),
            latency_ms=latency_ms,
            is_mock=is_mock,
            metadata=meta,
        )

    def _deterministic_sdof_mock(
        self,
        ag: np.ndarray,
        dt: float,
        T: float,
        zeta: float,
        is_bilinear: float,
        uy: float,
        alpha: float,
        mass: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deterministic, lightweight numerical integration (Newmark-beta average acceleration)
        used as fallback when PyTorch checkpoint cannot be loaded.
        """
        n = len(ag)
        u = np.zeros(n, dtype=np.float32)
        v = np.zeros(n, dtype=np.float32)
        a = np.zeros(n, dtype=np.float32)
        fr = np.zeros(n, dtype=np.float32)
        eh = np.zeros(n, dtype=np.float32)

        omega_n = 2.0 * np.pi / max(1e-4, T)
        k0 = mass * (omega_n ** 2)
        c = 2.0 * mass * omega_n * zeta

        gamma = 0.5
        beta = 0.25

        fy = k0 * uy if is_bilinear else 1e9
        u_p = 0.0

        for i in range(1, n):
            p_eff = -mass * ag[i]
            # Predictor
            u_pred = u[i-1] + dt * v[i-1] + (0.5 - beta) * dt**2 * a[i-1]
            v_pred = v[i-1] + (1.0 - gamma) * dt * a[i-1]

            # Inelastic trial
            f_trial = k0 * (u_pred - u_p)
            if is_bilinear and abs(f_trial) > fy:
                sign = 1.0 if f_trial > 0 else -1.0
                fr_val = sign * fy + alpha * k0 * (u_pred - u_p - sign * uy)
                u_p += (1.0 - alpha) * (f_trial - sign * fy) / k0
            else:
                fr_val = f_trial

            k_tan = (alpha * k0) if (is_bilinear and abs(f_trial) > fy) else k0
            k_hat = k_tan + gamma / (beta * dt) * c + 1.0 / (beta * dt**2) * mass
            delta_p = p_eff - (mass * a[i-1] + c * v_pred + fr_val)
            delta_u = delta_p / max(1e-4, k_hat)

            u[i] = u_pred + delta_u
            v[i] = v_pred + (gamma / (beta * dt)) * delta_u
            a[i] = (u[i] - u[i-1] - dt * v[i-1] - (0.5 - beta) * dt**2 * a[i-1]) / (beta * dt**2)
            fr[i] = fr_val

            # Hysteretic energy incremental dissipation
            du = u[i] - u[i-1]
            eh[i] = eh[i-1] + max(0.0, float(fr[i] * du - 0.5 * (fr[i]**2 - fr[i-1]**2) / k0))

        return u, fr, eh

    def metadata(self) -> Dict[str, Any]:
        """Return runtime metadata and checkpoint configuration."""
        return {
            "loaded": self.is_loaded,
            "device": str(self.device),
            "config_path": str(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "model_info": self.model_info,
        }
