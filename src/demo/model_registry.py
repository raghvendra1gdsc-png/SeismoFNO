"""
src/demo/model_registry.py — Model Registry for SeismoFNO Research Demonstration.

Provides an immutable, central registry describing all verified neural operator models
from EXP4, EXP5, and EXP6 without altering frozen checkpoints or training scripts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import os

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum for artifact integrity verification."""
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable metadata record for a research model checkpoint."""
    model_id: str
    display_name: str
    phase: str
    architecture: str
    representation: str
    conditioning_type: str
    cond_dim: int
    parameter_count: int
    checkpoint_relative_path: str
    status: str
    primary_role: str
    verified_key_metric: str
    supports_live_inference: bool
    notes: str

    @property
    def checkpoint_path(self) -> Path:
        return REPO_ROOT / self.checkpoint_relative_path

    @property
    def sha256(self) -> str:
        return compute_file_sha256(self.checkpoint_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "phase": self.phase,
            "architecture": self.architecture,
            "representation": self.representation,
            "conditioning_type": self.conditioning_type,
            "cond_dim": self.cond_dim,
            "parameter_count": self.parameter_count,
            "checkpoint_path": str(self.checkpoint_relative_path),
            "sha256": self.sha256[:16] + "...",
            "full_sha256": self.sha256,
            "status": self.status,
            "primary_role": self.primary_role,
            "verified_key_metric": self.verified_key_metric,
            "supports_live_inference": self.supports_live_inference,
            "notes": self.notes,
        }


# Authoritative Registry of Frozen Models
RESEARCH_MODELS: Dict[str, ModelMetadata] = {
    "exp4_fno2d": ModelMetadata(
        model_id="exp4_fno2d",
        display_name="EXP4: Fixed-Grid FNO2D (Baseline)",
        phase="EXP4",
        architecture="2D Fourier Neural Operator (FNO2d)",
        representation="Fixed 5x2048 grid with zero-padded stories",
        conditioning_type="Unconditioned",
        cond_dim=0,
        parameter_count=4735187,
        checkpoint_relative_path="results/experiments/exp4/training/best_checkpoint.pt",
        status="ARCHIVAL_FROZEN",
        primary_role="Initial MDOF surrogate baseline; uncovers zero-padding boundary failure",
        verified_key_metric="3-Story Rel L2 = 99.60% (Boundary failure mode)",
        supports_live_inference=False,
        notes="Evaluated via verified archival records to preserve historical execution environment.",
    ),
    "exp5_gno": ModelMetadata(
        model_id="exp5_gno",
        display_name="EXP5: Spatiotemporal GNO (Unconditioned)",
        phase="EXP5",
        architecture="Spatiotemporal Graph Neural Operator",
        representation="Topology-native graph G=(V, E), 0 zero-padding",
        conditioning_type="Unconditioned",
        cond_dim=0,
        parameter_count=674115,
        checkpoint_relative_path="results/experiments/exp5/training/best_checkpoint.pt",
        status="ARCHIVAL_FROZEN",
        primary_role="Resolves variable topology; uncovers modal extrapolation phase drift",
        verified_key_metric="3-Story Rel L2 = 22.09% (77.5 percentage-point reduction)",
        supports_live_inference=True,
        notes="Exact 3 nodes for 3S, 5 nodes for 5S. Checkpoint verified at val loss 0.2485.",
    ),
    "exp6_t1_gno": ModelMetadata(
        model_id="exp6_t1_gno",
        display_name="EXP6-B: T1-Conditioned GNO (FiLM)",
        phase="EXP6",
        architecture="Conditioned Spatiotemporal GNO",
        representation="Topology-native graph + FiLM modulation",
        conditioning_type="Fundamental period [T1] scalar",
        cond_dim=1,
        parameter_count=725059,
        checkpoint_relative_path="results/experiments/exp6/training/best_t1_gno.pt",
        status="ACTIVE_RESEARCH",
        primary_role="Physics-informed conditioning on fundamental modal period",
        verified_key_metric="OOD-B Peak Disp Error = 13.47% (vs 35.21% baseline)",
        supports_live_inference=True,
        notes="Pre-earthquake structural invariant T1 injected into spatial and temporal layers.",
    ),
    "exp6_multimodal_gno": ModelMetadata(
        model_id="exp6_multimodal_gno",
        display_name="EXP6-C: Multi-Modal GNO (FiLM)",
        phase="EXP6",
        architecture="Conditioned Spatiotemporal GNO",
        representation="Topology-native graph + FiLM modulation",
        conditioning_type="Multi-modal [T1-3, omega1-3] vector",
        cond_dim=6,
        parameter_count=727619,
        checkpoint_relative_path="results/experiments/exp6/training/best_multimodal_gno.pt",
        status="ACTIVE_RESEARCH",
        primary_role="Physics conditioning on 3 natural periods and 3 circular frequencies",
        verified_key_metric="OOD-B Peak Disp Error = 13.06% (62.9% relative reduction)",
        supports_live_inference=True,
        notes="Pre-earthquake structural invariants from [K],[M] matrices. Peak performance.",
    ),
    "exp6_shuffled_gno": ModelMetadata(
        model_id="exp6_shuffled_gno",
        display_name="EXP6-D: Shuffled Modal GNO (Ablation)",
        phase="EXP6",
        architecture="Conditioned Spatiotemporal GNO",
        representation="Topology-native graph + randomized FiLM",
        conditioning_type="Permuted [T1] across batch elements",
        cond_dim=1,
        parameter_count=725059,
        checkpoint_relative_path="results/experiments/exp6/training/best_shuffled_modal_gno.pt",
        status="FALSIFICATION_ABLATION",
        primary_role="Falsification control proving gains arise from physical correspondence",
        verified_key_metric="OOD-B Peak Disp Error = 24.33% (significant degradation)",
        supports_live_inference=True,
        notes="Random permutation breaks physical correspondence while preserving parameter capacity.",
    ),
}


class ModelRegistry:
    """Registry query interface for the research demonstration."""

    @staticmethod
    def get_model(model_id: str) -> Optional[ModelMetadata]:
        return RESEARCH_MODELS.get(model_id)

    @staticmethod
    def list_models() -> List[Dict[str, Any]]:
        return [m.to_dict() for m in RESEARCH_MODELS.values()]

    @staticmethod
    def verify_all_checkpoints_exist() -> Dict[str, bool]:
        return {
            m.model_id: m.checkpoint_path.exists()
            for m in RESEARCH_MODELS.values()
        }


FROZEN_MODELS = RESEARCH_MODELS


def get_model_spec(model_id: str) -> Optional[Dict[str, Any]]:
    m = ModelRegistry.get_model(model_id)
    return m.to_dict() if m else None


def list_registered_models() -> List[Dict[str, Any]]:
    return ModelRegistry.list_models()
