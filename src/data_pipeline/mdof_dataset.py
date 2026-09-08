"""
mdof_dataset.py — PyTorch Dataset and Normalization Pipeline for MDOF Operator Learning.

Loads MDOF simulation arrays (.npz) and formats them into spatiotemporal tensors:
  x: [Batch, In_channels, Story, Time]
  y: [Batch, Out_channels, Story, Time]
"""

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data_pipeline.dataset_builder import UnitGaussianNormalizer


class UnitGaussianNormalizer2D:
    """
    Normalizes spatiotemporal tensors along channel dimensions:
        x_norm = (x - mean) / (std + eps)
    where mean and std have shape [1, Channels, 1, 1].
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "UnitGaussianNormalizer2D":
        """x shape: [Batch, Channels, Story, Time]"""
        self.mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        self.std = torch.std(x, dim=(0, 2, 3), keepdim=True) + self.eps
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        return x * self.std.to(x.device) + self.mean.to(x.device)


class SeismicMDOFDataset(Dataset):
    """
    PyTorch Dataset for Multi-Degree-of-Freedom building dynamic response.

    Input channels (10 channels per story-time grid point):
      0: Ground acceleration a_g(t)
      1: Fundamental modal period T_1 (s)
      2: Second modal period T_2 (s)
      3: Normalized floor elevation s / S in (0, 1]
      4: Story floor mass m_s (kg)
      5: Total number of stories S
      6: Material type indicator (0.0=elastic, 1.0=bilinear)
      7: Yield drift limit u_{y, s} (m)
      8: Peak Ground Acceleration PGA (g)
      9: Normalized time coordinate tau = t / T_total in [0, 1]

    Target channels (3 channels):
      0: Floor relative displacement u(s, t) (m)
      1: Story shear restoring force F_R(s, t) (N)
      2: Cumulative hysteretic energy E_h(s, t) (J)
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        target_time_steps: int = 2048,
        target_channels: Optional[List[str]] = None,
        max_stories: int = 5,
        x_normalizer: Optional[UnitGaussianNormalizer2D] = None,
        y_normalizer: Optional[UnitGaussianNormalizer2D] = None,
    ):
        self.df = index_df.reset_index(drop=True)
        self.target_time_steps = target_time_steps
        self.target_channels = target_channels or ["u", "f_r", "e_h"]
        self.max_stories = max_stories
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        data = np.load(file_path)
        ag = data["ag"]
        u = data["u"]      # (S, T)
        f_r = data["f_r"]  # (S, T)
        e_h = data["e_h"]  # (S, T)

        n_stories = u.shape[0]
        t_steps = u.shape[1]

        # Resample / interpolate to target_time_steps if necessary
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
        max_S = self.max_stories

        # Construct input grid: [10, max_S, T]
        x_tensor = torch.zeros(10, max_S, T, dtype=torch.float32)

        # 0: Ground acceleration (broadcast to active stories)
        x_tensor[0, :S, :] = ag_t.unsqueeze(0).repeat(S, 1)

        # 1: T_1
        x_tensor[1, :S, :] = float(row.get("T1_s", 0.5))

        # 2: T_2
        x_tensor[2, :S, :] = float(row.get("T2_s", 0.2))

        # 3: Story coordinate s / S
        for s in range(S):
            x_tensor[3, s, :] = (s + 1.0) / float(S)

        # 4: Floor mass
        x_tensor[4, :S, :] = 1000.0

        # 5: Story count S
        x_tensor[5, :S, :] = float(S)

        # 6: Material indicator
        x_tensor[6, :S, :] = 1.0 if row.get("material_type", "elastic") == "bilinear" else 0.0

        # 7: Yield drift limit
        x_tensor[7, :S, :] = float(row.get("yield_drift_ratio", 0.005))

        # 8: PGA
        x_tensor[8, :S, :] = float(row.get("pga_g", 0.1))

        # 9: Normalized time coordinate
        time_grid = torch.linspace(0.0, 1.0, T)
        x_tensor[9, :S, :] = time_grid.unsqueeze(0).repeat(S, 1)

        # Target tensor: [3, max_S, T]
        y_tensor = torch.zeros(3, max_S, T, dtype=torch.float32)
        y_tensor[0, :S, :] = u_t
        y_tensor[1, :S, :] = f_t
        y_tensor[2, :S, :] = e_t

        # Apply normalizers if available
        if self.x_normalizer is not None:
            x_tensor = self.x_normalizer.encode(x_tensor.unsqueeze(0)).squeeze(0)
        if self.y_normalizer is not None:
            y_tensor = self.y_normalizer.encode(y_tensor.unsqueeze(0)).squeeze(0)

        return x_tensor, y_tensor
