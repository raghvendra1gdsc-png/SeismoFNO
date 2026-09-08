"""
gno.py — Spatiotemporal Graph Neural Operator (GNO) for Multi-Story Structural Dynamics.

Operates natively on physical structural graphs:
  - Preserves variable-floor graph topology (3-story, 5-story, etc.) with ZERO zero-padding
  - Spatial dimension: Physical message passing across structural columns and floors
  - Temporal dimension: Global 1D Fourier spectral convolutions over continuous time
  - Includes topology ablation mode (use_topology=False) for rigorous scientific comparison
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution along the continuous time dimension.
    Computes global Fourier kernel convolution in O(T log T).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [N, in_channels, T]
        Returns:
          out: [N, out_channels, T]
        """
        N, C, T = x.shape

        # Fast Fourier Transform along temporal dimension
        x_ft = torch.fft.rfft(x, dim=-1)  # [N, in_channels, T // 2 + 1]

        # Allocate output in frequency domain
        out_ft = torch.zeros(
            N, self.out_channels, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )

        modes_to_use = min(self.modes, x_ft.size(-1))

        # Complex matrix multiplication on lower Fourier frequencies
        # x_ft[..., :modes]: [N, in_channels, modes]
        # weights[..., :modes]: [in_channels, out_channels, modes]
        # output: [N, out_channels, modes]
        out_ft[..., :modes_to_use] = torch.einsum(
            "bct,cot->bot",
            x_ft[..., :modes_to_use],
            self.weights[..., :modes_to_use]
        )

        # Inverse Fast Fourier Transform back to time domain
        x_out = torch.fft.irfft(out_ft, n=T, dim=-1)
        return x_out


class SpatialGraphConv(nn.Module):
    """
    Message passing layer along physical structural columns and floors.
    Propagates interstory shear forces, drifts, and base accelerations.
    """

    def __init__(self, channels: int, edge_dim: int = 2, use_topology: bool = True):
        super().__init__()
        self.channels = channels
        self.edge_dim = edge_dim
        self.use_topology = use_topology

        # Node transformation
        self.w_node = nn.Conv1d(channels, channels, kernel_size=1)
        # Self transformation
        self.w_self = nn.Conv1d(channels, channels, kernel_size=1)

        # Edge gating / modulation
        if self.use_topology and edge_dim > 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_dim, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
                nn.Sigmoid(),
            )
        else:
            self.edge_mlp = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x: Node features [N_total, channels, T]
          edge_index: Graph edge connectivity [2, E_total]
          edge_attr: Edge attributes [E_total, edge_dim]
        Returns:
          out: Updated node features [N_total, channels, T]
        """
        N_total, C, T = x.shape
        out_self = self.w_self(x)

        # If topology is disabled (Ablation B), skip message passing between nodes
        if not self.use_topology or edge_index.numel() == 0:
            return out_self

        # Source and destination node indices
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # Transform source node messages
        src_feats = self.w_node(x)[src_idx]  # [E_total, C, T]

        # Modulate by edge attributes if available
        if self.edge_mlp is not None and edge_attr is not None:
            # [E_total, C, 1]
            e_mod = self.edge_mlp(edge_attr).unsqueeze(-1)
            messages = src_feats * e_mod
        else:
            messages = src_feats

        # Aggregate messages into destination nodes via index_add_
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst_idx, messages)

        return agg + out_self


class SpatiotemporalGNOBlock(nn.Module):
    """
    Composite Spatiotemporal Graph Neural Operator Block.
    Executes Spatial Graph Kernel Integration followed by 1D Temporal FNO.
    """

    def __init__(
        self,
        channels: int,
        modes: int = 64,
        edge_dim: int = 2,
        use_topology: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.use_topology = use_topology

        # Spatial operator
        self.spatial_conv = SpatialGraphConv(channels, edge_dim=edge_dim, use_topology=use_topology)

        # Temporal operator (1D FNO)
        self.spectral_conv = SpectralConv1d(channels, channels, modes=modes)
        self.w_time = nn.Conv1d(channels, channels, kernel_size=1)

        # Point-wise MLP expansion
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels * 2, channels, kernel_size=1),
        )

        # Normalization layers (channel-wise across nodes and time)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x: [N_total, channels, T]
          edge_index: [2, E_total]
          edge_attr: [E_total, edge_dim]
        """
        # 1. Spatial Graph Message Passing
        res = x
        x_spat = self.spatial_conv(x, edge_index, edge_attr)
        x = self.norm1(res + x_spat)

        # 2. Temporal Spectral Convolution
        res = x
        x_spec = self.spectral_conv(x)
        x_linear = self.w_time(x)
        x_temp = F.gelu(x_spec + x_linear)
        x = self.norm2(res + x_temp)

        # 3. Feed-forward expansion
        x = x + self.mlp(x)
        return x


class SpatiotemporalGNO(nn.Module):
    """
    Complete Spatiotemporal Graph Neural Operator (GNO).

    Maps:
      [N_total, in_channels, T] (Multi-story building graphs across time)
      -> [N_total, out_channels, T] (Predicted floor displacements, shear, energy)
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        width: int = 48,
        modes: int = 64,
        n_layers: int = 4,
        edge_dim: int = 2,
        use_topology: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.n_layers = n_layers
        self.use_topology = use_topology

        # 1. Lifting: Project input features to latent channel dimension
        self.lifting = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1),
        )

        # 2. Spatiotemporal GNO Blocks
        self.blocks = nn.ModuleList([
            SpatiotemporalGNOBlock(
                channels=width,
                modes=modes,
                edge_dim=edge_dim,
                use_topology=use_topology,
            )
            for _ in range(n_layers)
        ])

        # 3. Projection / Decoder: Project latent states to output channels
        self.projection = nn.Sequential(
            nn.Conv1d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width * 2, out_channels, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x: [N_total, in_channels, T]
          edge_index: [2, E_total]
          edge_attr: [E_total, edge_dim]
        Returns:
          out: [N_total, out_channels, T]
        """
        h = self.lifting(x)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)
        out = self.projection(h)
        return out
