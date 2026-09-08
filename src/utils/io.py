"""
io.py — File I/O Utilities for SeismoFNO Configurations and Datasets.
"""

from typing import Dict, Any, Union
from pathlib import Path
import json
import yaml
import numpy as np


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Safely load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def save_npz_simulation(
    filepath: Union[str, Path],
    time: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    f_r: np.ndarray,
    e_h: np.ndarray,
    ag: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """Save simulation arrays and metadata to NPZ archive."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        time=time,
        u=u,
        v=v,
        a=a,
        f_r=f_r,
        e_h=e_h,
        ag=ag,
        metadata_json=json.dumps(metadata),
    )
