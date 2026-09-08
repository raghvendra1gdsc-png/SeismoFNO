"""
dataset_builder.py — PyTorch Dataset and Dataloaders for Seismic SDOF Operator Learning.

Builds PyTorch Datasets and DataLoaders supporting:
  - Multi-channel input representations (ground motion, structural parameters, time grids, auxiliary history channels)
  - Multi-target outputs (displacement, restoring force, hysteretic energy)
  - UnitGaussianNormalizer for zero-mean unit-variance scaling
  - Partitioning via saved split JSON files (random, held-out-earthquake, held-out-structure)
"""

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data_pipeline.splits import SplitResult, load_split


class UnitGaussianNormalizer:
    """
    Normalizes tensors along spatial/time dimensions to zero mean and unit variance:
        x_norm = (x - mean) / (std + eps)
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "UnitGaussianNormalizer":
        """
        Fit mean and standard deviation across batch and time dimensions.
        x shape: [Batch, Channels, Time]
        """
        # Compute mean and std per channel across batch and time
        self.mean = torch.mean(x, dim=(0, 2), keepdim=True)
        self.std = torch.std(x, dim=(0, 2), keepdim=True) + self.eps
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization: [B, C, T] -> [B, C, T]."""
        if self.mean is None or self.std is None:
            return x
        device = x.device
        return (x - self.mean.to(device)) / self.std.to(device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Invert normalization: [B, C, T] -> [B, C, T]."""
        if self.mean is None or self.std is None:
            return x
        device = x.device
        return x * self.std.to(device) + self.mean.to(device)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.mean = state["mean"]
        self.std = state["std"]


class SeismicSDOFDataset(Dataset):
    """
    PyTorch Dataset for Seismic SDOF Time Series Responses.

    Input features per sample (when include_structural_params=True):
      - Channel 0: Ground acceleration a_g(t) [m/s^2]
      - Channel 1: Period T [s]
      - Channel 2: Natural frequency omega_n = 2*pi/T [rad/s]
      - Channel 3: Initial stiffness k0 = omega_n^2 [N/m]
      - Channel 4: Damping ratio zeta
      - Channel 5: Material flag (0=elastic, 1=bilinear)
      - Channel 6: Yield displacement u_y [m]
      - Channel 7: Post-yield stiffness ratio alpha
      - Channel 8: PGA [g]
      - Channel 9: Normalized time coordinate grid tau in [0, 1]
      (Optional if use_history_channel=True):
      - Channel 10: Running cumulative displacement path int_0^t |v| dtau
      - Channel 11: Running peak displacement max_{tau <= t} |u(tau)|
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        target_time_steps: Optional[int] = 2048,
        target_channels: List[str] = ["u", "f_r", "e_h"],
        include_structural_params: bool = True,
        use_history_channel: bool = False,
        return_meta: bool = False,
        x_normalizer: Optional[UnitGaussianNormalizer] = None,
        y_normalizer: Optional[UnitGaussianNormalizer] = None,
    ):
        self.df = index_df.reset_index(drop=True)
        self.target_time_steps = target_time_steps
        self.target_channels = target_channels
        self.include_structural_params = include_structural_params
        self.use_history_channel = use_history_channel
        self.return_meta = return_meta
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        row = self.df.iloc[idx]
        file_path = row["sim_file_path"]

        data = np.load(file_path)
        ag = data["ag"]               # [N]
        u = data["u"]                 # [N]
        f_r = data["f_r"]             # [N]
        e_h = data["e_h"]             # [N]

        # Crop or pad to target_time_steps
        n = len(ag)
        if self.target_time_steps is not None:
            target_n = self.target_time_steps
            if n >= target_n:
                ag = ag[:target_n]
                u = u[:target_n]
                f_r = f_r[:target_n]
                e_h = e_h[:target_n]
            else:
                pad_width = target_n - n
                ag = np.pad(ag, (0, pad_width), mode="constant")
                u = np.pad(u, (0, pad_width), mode="edge")
                f_r = np.pad(f_r, (0, pad_width), mode="edge")
                e_h = np.pad(e_h, (0, pad_width), mode="edge")
            actual_n = target_n
        else:
            actual_n = n

        # Assemble input channels
        input_channels = [ag]

        if self.include_structural_params:
            t_period = float(row["T"])
            zeta = float(row["zeta"])
            is_bilinear = 1.0 if row["material_type"] == "bilinear" else 0.0
            u_y = float(row["u_y"]) if not pd.isna(row["u_y"]) else 0.0
            alpha = float(row["alpha"]) if not pd.isna(row["alpha"]) else 0.0
            pga = float(row["target_pga_g"]) if "target_pga_g" in row and not pd.isna(row["target_pga_g"]) else 0.0
            omega_n = 2.0 * np.pi / max(1e-4, t_period)
            k0 = omega_n ** 2
            # Normalized time grid [0, 1]
            time_grid = np.linspace(0.0, 1.0, actual_n, dtype=np.float32)

            input_channels.extend([
                np.full(actual_n, t_period, dtype=np.float32),
                np.full(actual_n, omega_n, dtype=np.float32),
                np.full(actual_n, k0, dtype=np.float32),
                np.full(actual_n, zeta, dtype=np.float32),
                np.full(actual_n, is_bilinear, dtype=np.float32),
                np.full(actual_n, u_y, dtype=np.float32),
                np.full(actual_n, alpha, dtype=np.float32),
                np.full(actual_n, pga, dtype=np.float32),
                time_grid,
            ])

        if self.use_history_channel:
            # Auxiliary loading history: cumulative excursion and running peak displacement
            cum_abs_u = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(u)))])
            running_peak_u = np.maximum.accumulate(np.abs(u))
            input_channels.extend([
                cum_abs_u.astype(np.float32),
                running_peak_u.astype(np.float32),
            ])

        x_arr = np.stack(input_channels, axis=0).astype(np.float32)

        # Assemble target channels
        target_dict = {
            "u": u,
            "f_r": f_r,
            "e_h": e_h,
            "v": data.get("v", np.zeros_like(u))[:actual_n],
            "total_accel": data.get("total_accel", np.zeros_like(u))[:actual_n],
        }
        y_list = [target_dict[ch] for ch in self.target_channels]
        y_arr = np.stack(y_list, axis=0).astype(np.float32)

        x_tensor = torch.from_numpy(x_arr)
        y_tensor = torch.from_numpy(y_arr)

        if self.x_normalizer is not None:
            x_tensor = self.x_normalizer.encode(x_tensor.unsqueeze(0)).squeeze(0)
        if self.y_normalizer is not None:
            y_tensor = self.y_normalizer.encode(y_tensor.unsqueeze(0)).squeeze(0)

        if self.return_meta:
            ductility = float(row["ductility"]) if "ductility" in row and not pd.isna(row["ductility"]) else 0.0
            meta = {
                "sim_id": str(row["sim_id"]),
                "earthquake_name": str(row.get("earthquake_name", "unknown")),
                "ductility": ductility,
                "is_post_yield": float(ductility > 1.0),
                "material_type": str(row.get("material_type", "elastic")),
            }
            return x_tensor, y_tensor, meta

        return x_tensor, y_tensor


def build_dataloaders(
    index_csv: Union[str, Path] = "data/simulations/simulation_index.csv",
    split_file: Union[str, Path] = "data/processed/splits/held_out_earthquake_split.json",
    batch_size: int = 32,
    target_time_steps: int = 2048,
    target_channels: List[str] = ["u", "f_r", "e_h"],
    use_history_channel: bool = False,
    filter_material: Optional[str] = None,
    normalize: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[UnitGaussianNormalizer], Optional[UnitGaussianNormalizer]]:
    """
    Build Train, Val, and Test DataLoaders from simulation index and a split JSON file.
    """
    df = pd.read_csv(index_csv)

    if filter_material is not None:
        df = df[df["material_type"] == filter_material].reset_index(drop=True)

    # Re-split or load split
    if Path(split_file).exists():
        split = load_split(split_file)
        # Match IDs to guarantee exact split alignment
        train_ids = set(split.train_ids)
        val_ids = set(split.val_ids)
        test_ids = set(split.test_ids)

        train_df = df[df["sim_id"].isin(train_ids)].reset_index(drop=True)
        val_df = df[df["sim_id"].isin(val_ids)].reset_index(drop=True)
        test_df = df[df["sim_id"].isin(test_ids)].reset_index(drop=True)
    else:
        from src.data_pipeline.splits import split_held_out_earthquake
        split = split_held_out_earthquake(df, seed=42)
        train_df = df.iloc[split.train_indices].reset_index(drop=True)
        val_df = df.iloc[split.val_indices].reset_index(drop=True)
        test_df = df.iloc[split.test_indices].reset_index(drop=True)

    # Initial unnormalized train dataset to compute normalization statistics
    train_dataset_raw = SeismicSDOFDataset(
        index_df=train_df,
        target_time_steps=target_time_steps,
        target_channels=target_channels,
        use_history_channel=use_history_channel,
    )

    x_norm = None
    y_norm = None

    if normalize and len(train_dataset_raw) > 0:
        sample_size = min(len(train_dataset_raw), 256)
        x_samples = []
        y_samples = []
        for i in range(sample_size):
            x_s, y_s = train_dataset_raw[i]
            x_samples.append(x_s)
            y_samples.append(y_s)

        x_stacked = torch.stack(x_samples, dim=0)  # [B, C_in, T]
        y_stacked = torch.stack(y_samples, dim=0)  # [B, C_out, T]

        x_norm = UnitGaussianNormalizer().fit(x_stacked)
        y_norm = UnitGaussianNormalizer().fit(y_stacked)

    # Instantiate datasets with normalizers
    train_ds = SeismicSDOFDataset(
        train_df,
        target_time_steps=target_time_steps,
        target_channels=target_channels,
        use_history_channel=use_history_channel,
        x_normalizer=x_norm,
        y_normalizer=y_norm,
    )
    val_ds = SeismicSDOFDataset(
        val_df,
        target_time_steps=target_time_steps,
        target_channels=target_channels,
        use_history_channel=use_history_channel,
        x_normalizer=x_norm,
        y_normalizer=y_norm,
    )
    # Test dataset configured to return metadata for regime disaggregation
    test_ds = SeismicSDOFDataset(
        test_df,
        target_time_steps=target_time_steps,
        target_channels=target_channels,
        use_history_channel=use_history_channel,
        return_meta=True,
        x_normalizer=x_norm,
        y_normalizer=y_norm,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, x_norm, y_norm
