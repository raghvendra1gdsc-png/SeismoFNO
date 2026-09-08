"""
seismo_agent/config.py — Configuration for SeismoAgent tool harness.

Provides repository paths, device configuration, and default model checkpoints.
All paths are resolved relative to the SeismoFNO repository root.
"""

from pathlib import Path
from typing import Optional
import torch

# Base repository root (assumes seismo_agent/ is directly under repo root)
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Default research configurations
DEFAULT_SDOF_CONFIG_PATH: Path = REPO_ROOT / "configs" / "phase5_c_data_energy_boundary.yaml"
DEFAULT_MDOF_CONFIG_PATH: Path = REPO_ROOT / "configs" / "phase8_mdof_fno_nonlinear.yaml"
DEFAULT_DATA_GEN_CONFIG_PATH: Path = REPO_ROOT / "configs" / "data_generation.yaml"

# Default model checkpoints
DEFAULT_SDOF_CHECKPOINT: Path = REPO_ROOT / "experiments" / "phase5_c_data_energy_boundary" / "checkpoints" / "best_model.pt"
DEFAULT_SDOF_NORMALIZERS: Path = REPO_ROOT / "experiments" / "phase5_c_data_energy_boundary" / "checkpoints" / "normalizers.pt"

DEFAULT_MDOF_CHECKPOINT: Path = REPO_ROOT / "experiments" / "phase8_mdof_fno_linear" / "checkpoints" / "best_model.pt"

# Ground motion directories
DEFAULT_PRESETS_DIR: Path = REPO_ROOT / "data" / "raw" / "ground_motions"
DEFAULT_REPORTS_DIR: Path = REPO_ROOT / "seismo_agent" / "reports"

# Nebius & NVIDIA Nemotron Configuration
import os

NEBIUS_API_KEY: str = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_BASE_URL: str = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
NEBIUS_MODEL: str = os.environ.get("NEBIUS_MODEL", "nvidia/nemotron-4-340b-instruct")

# Orchestration safety limits
MAX_TOOL_CALLS: int = int(os.environ.get("SEISMO_MAX_TOOL_CALLS", "10"))
MAX_ORCHESTRATION_STEPS: int = int(os.environ.get("SEISMO_MAX_STEPS", "12"))


def get_default_device(preference: Optional[str] = None) -> torch.device:
    """
    Resolve target compute device with fallback order: preference -> mps -> cuda -> cpu.
    """
    if preference:
        try:
            return torch.device(preference)
        except Exception:
            pass

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
