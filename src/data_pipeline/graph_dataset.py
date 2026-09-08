"""
graph_dataset.py — Topology-Native Graph Dataset & Batching for EXP5.

Implements native structural graph representation for multi-story buildings:
  - 3-story buildings have exactly 3 nodes
  - 5-story buildings have exactly 5 nodes
  - Zero padding is strictly eliminated
  - Physical interstory shear connectivity is preserved via bidirectional edges
  - Batched graph collation for efficient Apple Silicon MPS execution
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class StructuralGraphSample:
    """Represents a single structural dynamic simulation as a graph."""
    sim_id: int
    struct_id: str
    n_stories: int
    earthquake_id: str
    # Node features: [n_stories, in_channels, time_steps]
    x: torch.Tensor
    # Target features: [n_stories, out_channels, time_steps]
    y: torch.Tensor
    # Edge index: [2, n_edges]
    edge_index: torch.Tensor
    # Edge features: [n_edges, edge_dim]
    edge_attr: torch.Tensor


@dataclass
class BatchedStructuralGraphs:
    """Disjoint batch of variable-story structural graphs."""
    # Concatenated node features: [N_total, in_channels, time_steps]
    x: torch.Tensor
    # Concatenated target features: [N_total, out_channels, time_steps]
    y: torch.Tensor
    # Shifted edge indices: [2, E_total]
    edge_index: torch.Tensor
    # Concatenated edge features: [E_total, edge_dim]
    edge_attr: torch.Tensor
    # Batch assignment vector: [N_total], mapping each node to graph index in [0, B-1]
    batch_idx: torch.Tensor
    # Number of nodes per graph: list of ints of length B
    node_counts: List[int]
    # Simulation IDs in batch: list of ints of length B
    sim_ids: List[int]
    # Structural IDs in batch: list of strings of length B
    struct_ids: List[str]

    def to(self, device: torch.device) -> "BatchedStructuralGraphs":
        return BatchedStructuralGraphs(
            x=self.x.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            batch_idx=self.batch_idx.to(device),
            node_counts=self.node_counts,
            sim_ids=self.sim_ids,
            struct_ids=self.struct_ids,
        )


def build_shear_frame_edges(n_stories: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build physical shear-frame connectivity for an N-story building.

    Nodes: 0 (Story 1 / ground level) to N-1 (Roof story).
    Edges:
      - Upward shear connection: (i, i+1)
      - Downward shear connection: (i+1, i)
      - Self-loops: (i, i) for story inertia

    Returns:
      edge_index: [2, E], edge_attr: [E, 2] where:
        col 0: relative elevation delta (1/N for up, -1/N for down, 0 for self)
        col 1: direction code (+1 up, -1 down, 0 self)
    """
    src, dst = [], []
    attrs = []

    # Interstory vertical edges
    for i in range(n_stories - 1):
        # Upward: i -> i+1
        src.append(i)
        dst.append(i + 1)
        attrs.append([1.0 / n_stories, 1.0])

        # Downward: i+1 -> i
        src.append(i + 1)
        dst.append(i)
        attrs.append([-1.0 / n_stories, -1.0])

    # Self loops for local node identity
    for i in range(n_stories):
        src.append(i)
        dst.append(i)
        attrs.append([0.0, 0.0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


class GraphNormalizer:
    """Channel-wise unit Gaussian normalizer for graph node tensors."""

    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        self.mean = mean
        self.std = std

    def fit(self, tensors: List[torch.Tensor], eps: float = 1e-6) -> "GraphNormalizer":
        """
        Fit mean and standard deviation over list of node tensors [n_nodes, channels, time_steps].
        Averages across nodes and time steps for each channel.
        """
        # Concatenate along node dimension
        all_x = torch.cat(tensors, dim=0)  # [Total_nodes, channels, T]
        # Compute mean per channel: [1, channels, 1]
        self.mean = all_x.mean(dim=(0, 2), keepdim=True)
        # Compute std per channel: [1, channels, 1]
        self.std = all_x.std(dim=(0, 2), keepdim=True) + eps
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return (x - mean) / std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return x * std + mean


class SeismicMDOFGraphDataset(Dataset):
    """
    Topology-Native Graph Dataset for MDOF Seismic Simulations.

    Zero-padding is strictly eliminated:
      3-story building -> Exactly 3 nodes
      5-story building -> Exactly 5 nodes
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        target_time_steps: int = 2048,
        x_normalizer: Optional[GraphNormalizer] = None,
        y_normalizer: Optional[GraphNormalizer] = None,
    ):
        self.df = index_df.reset_index(drop=True)
        self.target_time_steps = target_time_steps
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> StructuralGraphSample:
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        data = np.load(file_path)
        ag = data["ag"]
        u = data["u"]      # [S, T]
        f_r = data["f_r"]  # [S, T]
        e_h = data["e_h"]  # [S, T]

        n_stories = int(u.shape[0])
        t_steps = int(u.shape[1])

        # Temporal alignment / interpolation to target_time_steps if necessary
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
        y_tensor = torch.stack([u_t, f_t, e_t], dim=1)  # [S, 3, T]

        # Node feature tensor: [S, in_channels, T]
        # in_channels = 10:
        #   0: Ground acceleration a_g(t)
        #   1: Fundamental modal period T_1 (s)
        #   2: Second modal period T_2 (s)
        #   3: Normalized floor elevation coordinate (s+1)/S in (0, 1]
        #   4: Floor lumped mass m_s (kg)
        #   5: Story count S (e.g. 3 or 5)
        #   6: Nonlinearity indicator (0.0=elastic, 1.0=bilinear)
        #   7: Yield drift limit u_{y, s} (m)
        #   8: Peak ground acceleration PGA (g)
        #   9: Continuous time coordinate tau = t / T_total in [0, 1]
        x_tensor = torch.zeros(S, 10, T, dtype=torch.float32)

        # 0: Ground acceleration (broadcast across all floors)
        x_tensor[:, 0, :] = ag_t.unsqueeze(0).repeat(S, 1)

        # 1 & 2: Modal periods
        x_tensor[:, 1, :] = float(row.get("T1_s", 0.5))
        x_tensor[:, 2, :] = float(row.get("T2_s", 0.2))

        # 3: Normalized floor elevation
        for s in range(S):
            x_tensor[s, 3, :] = (s + 1.0) / float(S)

        # 4: Floor lumped mass (kg)
        x_tensor[:, 4, :] = 1000.0

        # 5: Story count
        x_tensor[:, 5, :] = float(S)

        # 6: Nonlinearity flag
        mat_type = str(row.get("material_type", "bilinear"))
        x_tensor[:, 6, :] = 1.0 if mat_type.lower() == "bilinear" else 0.0

        # 7: Yield drift limit
        ydr = float(row.get("yield_drift_ratio", 0.005))
        story_h = 3.2  # Nominal story height
        x_tensor[:, 7, :] = ydr * story_h

        # 8: PGA
        x_tensor[:, 8, :] = float(row.get("pga_g", 0.4))

        # 9: Continuous time coordinate tau
        tau = torch.linspace(0.0, 1.0, T).unsqueeze(0).repeat(S, 1)
        x_tensor[:, 9, :] = tau

        # Apply normalizers if provided
        if self.x_normalizer is not None:
            x_tensor = self.x_normalizer.encode(x_tensor)
        if self.y_normalizer is not None:
            y_tensor = self.y_normalizer.encode(y_tensor)

        # Build physical shear-frame graph topology
        edge_index, edge_attr = build_shear_frame_edges(S)

        return StructuralGraphSample(
            sim_id=int(row["sim_id"]),
            struct_id=str(row.get("struct_id", f"{S}S")),
            n_stories=S,
            earthquake_id=str(row.get("earthquake_id", "RSN0000")),
            x=x_tensor,
            y=y_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )


def collate_structural_graphs(batch: List[StructuralGraphSample]) -> BatchedStructuralGraphs:
    """
    Collate a list of variable-story structural graphs into a single disjoint batch.
    Nodes and edges are concatenated with shifted indices.
    """
    x_list = []
    y_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_idx_list = []
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

        # Shift edge indices by node_offset
        shifted_edges = sample.edge_index + node_offset
        edge_index_list.append(shifted_edges)
        edge_attr_list.append(sample.edge_attr)

        # Batch assignment
        batch_idx_list.append(torch.full((n_nodes,), graph_idx, dtype=torch.long))

        node_offset += n_nodes

    return BatchedStructuralGraphs(
        x=torch.cat(x_list, dim=0),                    # [N_total, 10, T]
        y=torch.cat(y_list, dim=0),                    # [N_total, 3, T]
        edge_index=torch.cat(edge_index_list, dim=1),   # [2, E_total]
        edge_attr=torch.cat(edge_attr_list, dim=0),     # [E_total, 2]
        batch_idx=torch.cat(batch_idx_list, dim=0),     # [N_total]
        node_counts=node_counts,
        sim_ids=sim_ids,
        struct_ids=struct_ids,
    )
