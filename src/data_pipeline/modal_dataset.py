"""
modal_dataset.py — Physics/Modal-Conditioned Graph Dataset & Batching for EXP6.

Extends the EXP5 topology-native graph dataset with explicit structural modal conditioning:
  - Zero zero-padding preserved (exact 3 nodes for 3S, 5 nodes for 5S)
  - Physical interstory shear connectivity preserved
  - Extracts eigenvalue modal properties (T1, T2, T3, omega1, omega2, omega3) derived from theoretical [K], [M]
  - ModalNormalizer fitted strictly on training data
  - Supports unconditioned, T1-conditioned, multi-modal conditioned, and shuffled modal conditioning
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data_pipeline.graph_dataset import (
    build_shear_frame_edges,
    GraphNormalizer,
)
from src.ground_truth.opensees_mdof_model import MDOFParams


# Precomputed theoretical modal catalog for all structural archetypes
MODAL_CATALOG: Dict[str, Dict[str, float]] = {
    "3S_T035": {
        "n_stories": 3, "T1": 0.3500, "T2": 0.1249, "T3": 0.0864,
        "omega1": 17.9519, "omega2": 50.2998, "omega3": 72.6934,
    },
    "3S_T060": {
        "n_stories": 3, "T1": 0.6000, "T2": 0.2141, "T3": 0.1482,
        "omega1": 10.4720, "omega2": 29.3415, "omega3": 42.4045,
    },
    "3S_T090": {
        "n_stories": 3, "T1": 0.9000, "T2": 0.3212, "T3": 0.2223,
        "omega1": 6.9813, "omega2": 19.5610, "omega3": 28.2697,
    },
    "5S_T055": {
        "n_stories": 5, "T1": 0.5500, "T2": 0.1884, "T3": 0.1195,
        "omega1": 11.4240, "omega2": 33.3465, "omega3": 52.5746,
    },
    "5S_T085": {
        "n_stories": 5, "T1": 0.8500, "T2": 0.2912, "T3": 0.1847,
        "omega1": 7.3920, "omega2": 21.5772, "omega3": 34.0189,
    },
    "5S_T105": {
        "n_stories": 5, "T1": 1.0500, "T2": 0.3597, "T3": 0.2282,
        "omega1": 5.9840, "omega2": 17.4673, "omega3": 27.5390,
    },
    "5S_T120": {
        "n_stories": 5, "T1": 1.2000, "T2": 0.4111, "T3": 0.2608,
        "omega1": 5.2360, "omega2": 15.2839, "omega3": 24.0967,
    },
    "5S_T140": {
        "n_stories": 5, "T1": 1.4000, "T2": 0.4796, "T3": 0.3042,
        "omega1": 4.4880, "omega2": 13.1005, "omega3": 20.6543,
    },
}


def get_modal_vector(struct_id: str, cond_mode: str = "t1") -> np.ndarray:
    """
    Retrieve theoretical modal conditioning vector for a structure.
    cond_mode options:
      - 'none': empty array (dim 0)
      - 't1': [T1] (dim 1)
      - 'multimodal': [T1, T2, T3, omega1, omega2, omega3] (dim 6)
    """
    if cond_mode == "none":
        return np.empty((0,), dtype=np.float32)

    entry = MODAL_CATALOG.get(struct_id)
    if entry is None:
        # Fallback to default estimation if unknown archetype
        t1 = 0.5
        t2 = 0.2
        t3 = 0.1
        w1 = 2.0 * math.pi / t1
        w2 = 2.0 * math.pi / t2
        w3 = 2.0 * math.pi / t3
    else:
        t1 = entry["T1"]
        t2 = entry["T2"]
        t3 = entry["T3"]
        w1 = entry["omega1"]
        w2 = entry["omega2"]
        w3 = entry["omega3"]

    if cond_mode == "t1":
        return np.array([t1], dtype=np.float32)
    elif cond_mode == "multimodal":
        return np.array([t1, t2, t3, w1, w2, w3], dtype=np.float32)
    else:
        raise ValueError(f"Unknown cond_mode '{cond_mode}'. Expected 'none', 't1', or 'multimodal'.")


class ModalNormalizer:
    """Standard Z-score normalizer for modal conditioning vectors."""

    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        self.mean = mean
        self.std = std

    def fit(self, vectors: List[np.ndarray], eps: float = 1e-6) -> "ModalNormalizer":
        """Fit mean and std strictly on training modal vectors."""
        if len(vectors) == 0 or vectors[0].size == 0:
            self.mean = None
            self.std = None
            return self

        arr = np.stack(vectors, axis=0)  # [N, d_cond]
        self.mean = torch.from_numpy(arr.mean(axis=0)).float()
        self.std = torch.from_numpy(arr.std(axis=0)).float() + eps
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None or x.numel() == 0:
            return x
        if x.shape[-1] != self.mean.shape[-1]:
            return x
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return (x - mean) / std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None or x.numel() == 0:
            return x
        if x.shape[-1] != self.mean.shape[-1]:
            return x
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return x * std + mean


@dataclass
class ModalStructuralGraphSample:
    """Represents a single structural dynamic simulation graph with modal conditioning."""
    sim_id: int
    struct_id: str
    n_stories: int
    earthquake_id: str
    x: torch.Tensor           # Node features: [n_stories, in_channels, time_steps]
    y: torch.Tensor           # Target features: [n_stories, out_channels, time_steps]
    edge_index: torch.Tensor  # Edge indices: [2, n_edges]
    edge_attr: torch.Tensor   # Edge features: [n_edges, edge_dim]
    modal_cond: torch.Tensor  # Modal conditioning: [cond_dim]


@dataclass
class BatchedModalStructuralGraphs:
    """Disjoint batch of variable-story structural graphs with modal conditioning."""
    x: torch.Tensor           # Concatenated node features: [N_total, in_channels, time_steps]
    y: torch.Tensor           # Concatenated target features: [N_total, out_channels, time_steps]
    edge_index: torch.Tensor  # Shifted edge indices: [2, E_total]
    edge_attr: torch.Tensor   # Concatenated edge features: [E_total, edge_dim]
    batch_idx: torch.Tensor   # Node to graph mapping: [N_total]
    modal_cond: torch.Tensor  # Graph modal conditioning: [B, cond_dim]
    node_counts: List[int]    # Number of nodes per graph: [B]
    sim_ids: List[int]        # Simulation IDs: [B]
    struct_ids: List[str]     # Structural IDs: [B]

    def to(self, device: torch.device) -> "BatchedModalStructuralGraphs":
        return BatchedModalStructuralGraphs(
            x=self.x.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            batch_idx=self.batch_idx.to(device),
            modal_cond=self.modal_cond.to(device),
            node_counts=self.node_counts,
            sim_ids=self.sim_ids,
            struct_ids=self.struct_ids,
        )


class SeismicMDOFModalGraphDataset(Dataset):
    """
    Topology-Native Graph Dataset with Modal Conditioning for MDOF Seismic Simulations.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        target_time_steps: int = 2048,
        x_normalizer: Optional[GraphNormalizer] = None,
        y_normalizer: Optional[GraphNormalizer] = None,
        modal_normalizer: Optional[ModalNormalizer] = None,
        cond_mode: str = "t1",
        shuffle_cond: bool = False,
    ):
        self.df = index_df.reset_index(drop=True)
        self.target_time_steps = target_time_steps
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.modal_normalizer = modal_normalizer
        self.cond_mode = cond_mode
        self.shuffle_cond = shuffle_cond

        # Precompute modal vectors for all rows
        self.raw_modal_vectors = [
            get_modal_vector(str(row.get("struct_id", "5S_T085")), cond_mode=cond_mode)
            for _, row in self.df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> ModalStructuralGraphSample:
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        data = np.load(file_path)
        ag = data["ag"]
        u = data["u"]
        f_r = data["f_r"]
        e_h = data["e_h"]

        n_stories = int(u.shape[0])
        t_steps = int(u.shape[1])

        # Temporal interpolation to target_time_steps if necessary
        if t_steps != self.target_time_steps:
            u_t = F.interpolate(torch.from_numpy(u).unsqueeze(0), size=self.target_time_steps, mode="linear", align_corners=True).squeeze(0)
            f_t = F.interpolate(torch.from_numpy(f_r).unsqueeze(0), size=self.target_time_steps, mode="linear", align_corners=True).squeeze(0)
            e_t = F.interpolate(torch.from_numpy(e_h).unsqueeze(0), size=self.target_time_steps, mode="linear", align_corners=True).squeeze(0)
            ag_t = F.interpolate(torch.from_numpy(ag).unsqueeze(0).unsqueeze(0), size=self.target_time_steps, mode="linear", align_corners=True).squeeze(0).squeeze(0)
        else:
            u_t = torch.from_numpy(u).float()
            f_t = torch.from_numpy(f_r).float()
            e_t = torch.from_numpy(e_h).float()
            ag_t = torch.from_numpy(ag).float()

        T = self.target_time_steps
        S = n_stories

        # Target tensor: [S, 3, T] -> [u, F_R, E_h]
        y_tensor = torch.stack([u_t, f_t, e_t], dim=1)

        # Node feature tensor: [S, in_channels, T] (10 channels as in EXP5)
        x_tensor = torch.zeros(S, 10, T, dtype=torch.float32)
        x_tensor[:, 0, :] = ag_t.unsqueeze(0).repeat(S, 1)
        x_tensor[:, 1, :] = float(row.get("T1_s", 0.5))
        x_tensor[:, 2, :] = float(row.get("T2_s", 0.2))

        for s in range(S):
            x_tensor[s, 3, :] = (s + 1.0) / float(S)

        x_tensor[:, 4, :] = 1000.0
        x_tensor[:, 5, :] = float(S)

        mat_type = str(row.get("material_type", "bilinear"))
        x_tensor[:, 6, :] = 1.0 if mat_type.lower() == "bilinear" else 0.0

        ydr = float(row.get("yield_drift_ratio", 0.005))
        story_h = 3.2
        x_tensor[:, 7, :] = ydr * story_h
        x_tensor[:, 8, :] = float(row.get("pga_g", 0.4))

        tau = torch.linspace(0.0, 1.0, T).unsqueeze(0).repeat(S, 1)
        x_tensor[:, 9, :] = tau

        # Normalization
        if self.x_normalizer is not None:
            x_tensor = self.x_normalizer.encode(x_tensor)
        if self.y_normalizer is not None:
            y_tensor = self.y_normalizer.encode(y_tensor)

        # Modal conditioning vector
        if self.shuffle_cond:
            # Pick a random modal vector from the dataset to test ablation D
            rand_idx = np.random.randint(0, len(self.raw_modal_vectors))
            modal_raw = self.raw_modal_vectors[rand_idx]
        else:
            modal_raw = self.raw_modal_vectors[idx]

        modal_tensor = torch.from_numpy(modal_raw).float()
        if self.modal_normalizer is not None:
            modal_tensor = self.modal_normalizer.encode(modal_tensor)

        # Build physical shear-frame graph topology
        edge_index, edge_attr = build_shear_frame_edges(S)

        return ModalStructuralGraphSample(
            sim_id=int(row["sim_id"]),
            struct_id=str(row.get("struct_id", f"{S}S")),
            n_stories=S,
            earthquake_id=str(row.get("earthquake_id", "RSN0000")),
            x=x_tensor,
            y=y_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            modal_cond=modal_tensor,
        )


def collate_modal_structural_graphs(batch: List[ModalStructuralGraphSample]) -> BatchedModalStructuralGraphs:
    """
    Collate a list of variable-story structural graphs with modal conditioning into a single disjoint batch.
    """
    x_list = []
    y_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_idx_list = []
    modal_cond_list = []
    node_counts = []
    sim_ids = []
    struct_ids = []

    node_offset = 0
    for graph_idx, sample in enumerate(batch):
        n_nodes = sample.n_stories
        node_counts.append(n_nodes)
        sim_ids.append(sample.sim_id)
        struct_ids.append(sample.struct_id)

        x_list.append(sample.x)
        y_list.append(sample.y)
        modal_cond_list.append(sample.modal_cond.unsqueeze(0))

        shifted_edges = sample.edge_index + node_offset
        edge_index_list.append(shifted_edges)
        edge_attr_list.append(sample.edge_attr)

        batch_idx_list.append(torch.full((n_nodes,), graph_idx, dtype=torch.long))
        node_offset += n_nodes

    modal_cond_batch = torch.cat(modal_cond_list, dim=0) if modal_cond_list[0].numel() > 0 else torch.empty((len(batch), 0))

    return BatchedModalStructuralGraphs(
        x=torch.cat(x_list, dim=0),
        y=torch.cat(y_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1),
        edge_attr=torch.cat(edge_attr_list, dim=0),
        batch_idx=torch.cat(batch_idx_list, dim=0),
        modal_cond=modal_cond_batch,
        node_counts=node_counts,
        sim_ids=sim_ids,
        struct_ids=struct_ids,
    )
