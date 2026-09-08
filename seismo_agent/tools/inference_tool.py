"""
seismo_agent/tools/inference_tool.py — Deterministic SeismoFNO Surrogate Inference Tool.

Wraps the continuous Fourier Neural Operator (FNO-1D / FNO-2D) inference pipeline.
Loads frozen research checkpoints and unit Gaussian normalizers, encodes multi-channel
physical inputs, and executes sub-millisecond forward passes.
"""

from pathlib import Path
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch

from seismo_agent.config import (
    DEFAULT_SDOF_CONFIG_PATH,
    DEFAULT_SDOF_CHECKPOINT,
    DEFAULT_SDOF_NORMALIZERS,
    get_default_device,
)
from seismo_agent.schemas.inputs import InferenceRequest
from seismo_agent.schemas.outputs import InferenceResult
from seismo_agent.tools.base import BaseTool, ToolExecutionError
from seismo_agent.tools.earthquake_tool import EarthquakeTool
from src.models.fno1d import FNO1d
from src.training.train import load_config
from src.data_pipeline.dataset_builder import UnitGaussianNormalizer
from src.ground_truth.opensees_sdof_model import SDOFParams


class InferenceTool(BaseTool):
    """
    Executes sub-millisecond Fourier Neural Operator surrogate inference for dynamic structural response.
    """
    name = "run_seismo_inference"
    description = (
        "Executes ultra-fast (< 1 ms) neural operator surrogate inference for nonlinear "
        "structural dynamics. Given an earthquake acceleration record and structural properties, "
        "evaluates continuous Fourier Neural Operators to predict full time histories of relative "
        "displacement u(t), restoring force F_R(t), and dissipated hysteretic energy E_h(t). "
        "Uses verified frozen research checkpoints without altering model weights."
    )
    input_schema = InferenceRequest
    output_schema = InferenceResult

    def __init__(self, eq_tool: Optional[EarthquakeTool] = None):
        self.eq_tool = eq_tool or EarthquakeTool()
        # In-memory model cache keyed by (config_path, checkpoint_path, device)
        self._model_cache: Dict[str, Tuple[torch.nn.Module, UnitGaussianNormalizer, UnitGaussianNormalizer]] = {}

    def _sync_device(self, device: torch.device):
        """Synchronize accelerator execution queue for accurate wall-clock timing."""
        if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        elif device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _load_model_and_normalizers(
        self,
        config_path_str: Optional[str],
        ckpt_path_str: Optional[str],
        norm_path_str: Optional[str],
        device: torch.device,
    ) -> Tuple[torch.nn.Module, UnitGaussianNormalizer, UnitGaussianNormalizer, str]:
        """Load FNO model architecture, checkpoint weights, and normalizers."""
        cfg_path = Path(config_path_str) if config_path_str else DEFAULT_SDOF_CONFIG_PATH
        if not cfg_path.exists():
            raise ToolExecutionError(self.name, f"Experiment configuration file not found at: {cfg_path}")

        cfg = load_config(str(cfg_path))
        exp_dir = Path(cfg["output"]["checkpoint_dir"])

        ckpt_path = Path(ckpt_path_str) if ckpt_path_str else (
            DEFAULT_SDOF_CHECKPOINT if DEFAULT_SDOF_CHECKPOINT.exists() else (exp_dir / "best_model.pt")
        )
        norm_path = Path(norm_path_str) if norm_path_str else (
            DEFAULT_SDOF_NORMALIZERS if DEFAULT_SDOF_NORMALIZERS.exists() else (exp_dir / "normalizers.pt")
        )

        cache_key = f"{cfg_path}:{ckpt_path}:{device}"
        if cache_key in self._model_cache:
            model, x_norm, y_norm = self._model_cache[cache_key]
            return model, x_norm, y_norm, str(ckpt_path)

        model_cfg = cfg["model"]
        model = FNO1d(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            modes=model_cfg["modes"],
            width=model_cfg["width"],
            n_layers=model_cfg["n_layers"],
        ).to(device)

        # Load normalizers
        x_norm = UnitGaussianNormalizer()
        y_norm = UnitGaussianNormalizer()
        if norm_path.exists():
            norm_data = torch.load(norm_path, map_location=device, weights_only=False)
            x_norm.mean = norm_data["x_mean"].to(device)
            x_norm.std = norm_data["x_std"].to(device)
            y_norm.mean = norm_data["y_mean"].to(device)
            y_norm.std = norm_data["y_std"].to(device)
        else:
            x_norm.mean = torch.zeros((1, model_cfg["in_channels"], 1), device=device)
            x_norm.std = torch.ones((1, model_cfg["in_channels"], 1), device=device)
            y_norm.mean = torch.zeros((1, model_cfg["out_channels"], 1), device=device)
            y_norm.std = torch.ones((1, model_cfg["out_channels"], 1), device=device)

        # Load weights
        if not ckpt_path.exists():
            raise ToolExecutionError(
                self.name,
                f"Model checkpoint not found at: {ckpt_path}. Please verify checkpoint path."
            )

        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)

        model.eval()

        # Warmup pass
        dummy_x = torch.zeros((1, model_cfg["in_channels"], 2048), device=device)
        with torch.no_grad():
            _ = model(x_norm.encode(dummy_x))
        self._sync_device(device)

        self._model_cache[cache_key] = (model, x_norm, y_norm)
        return model, x_norm, y_norm, str(ckpt_path)

    def _run(self, request: InferenceRequest) -> InferenceResult:
        device = get_default_device(request.device)

        # 1. Obtain processed acceleration time series
        ag, dt, rec_name = self.eq_tool.get_processed_acceleration_array(request.earthquake)
        n_steps = len(ag)
        time_arr = np.linspace(0, (n_steps - 1) * dt, n_steps)
        pga_g = float(np.max(np.abs(ag)) / 9.80665)

        # 2. Structural parameters
        st = request.structure
        omega_n = 2.0 * np.pi / max(1e-4, st.T)
        k0 = st.mass * (omega_n ** 2)
        is_bilinear = 1.0 if st.material_type == "bilinear" else 0.0
        u_y_val = st.u_y if (st.material_type == "bilinear" and st.u_y is not None) else 0.0

        # 3. Load model and normalizers
        model, x_norm, y_norm, ckpt_str = self._load_model_and_normalizers(
            request.config_path,
            request.checkpoint_path,
            request.normalizer_path,
            device,
        )

        # 4. Construct 10-channel input tensor [1, 10, N]
        time_grid = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
        channels = [
            ag.astype(np.float32),
            np.full(n_steps, st.T, dtype=np.float32),
            np.full(n_steps, omega_n, dtype=np.float32),
            np.full(n_steps, k0, dtype=np.float32),
            np.full(n_steps, st.zeta, dtype=np.float32),
            np.full(n_steps, is_bilinear, dtype=np.float32),
            np.full(n_steps, u_y_val, dtype=np.float32),
            np.full(n_steps, st.alpha, dtype=np.float32),
            np.full(n_steps, pga_g, dtype=np.float32),
            time_grid,
        ]
        x_tensor = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(device)

        # 5. Measure pure forward-pass inference latency
        self._sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            x_normed = x_norm.encode(x_tensor)
            y_pred_norm = model(x_normed)
            y_pred = y_norm.decode(y_pred_norm).squeeze(0).cpu().numpy()
        self._sync_device(device)
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        u_pred = y_pred[0]
        fr_pred = y_pred[1]
        eh_pred = y_pred[2]

        u_max = float(np.max(np.abs(u_pred)))
        fr_max = float(np.max(np.abs(fr_pred)))
        eh_total = float(eh_pred[-1]) if len(eh_pred) > 0 else 0.0
        ductility = float(u_max / max(1e-6, u_y_val)) if is_bilinear > 0 else 1.0

        return InferenceResult(
            status="success",
            model_architecture="FNO1d",
            checkpoint_used=ckpt_str,
            device_used=str(device),
            runtime_ms=round(runtime_ms, 3),
            n_points=n_steps,
            dt=round(dt, 5),
            u_max_m=round(u_max, 6),
            fr_max_n=round(fr_max, 4),
            eh_total_j=round(eh_total, 4),
            ductility_mu=round(ductility, 3),
            time=time_arr.tolist(),
            u_pred=u_pred.tolist(),
            fr_pred=fr_pred.tolist(),
            eh_pred=eh_pred.tolist(),
            metadata={
                "record_name": rec_name,
                "pga_g": round(pga_g, 4),
                "modes": getattr(model, "modes", None),
                "width": getattr(model, "width", None),
                "num_params": model.get_num_parameters() if hasattr(model, "get_num_parameters") else None,
            },
        )
