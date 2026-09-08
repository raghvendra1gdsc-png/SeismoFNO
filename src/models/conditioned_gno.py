"""
conditioned_gno.py — Physics/Modal-Conditioned Spatiotemporal Graph Neural Operator for EXP6.

Extends the EXP5 Spatiotemporal GNO with principled Feature-wise Linear Modulation (FiLM):
  - Preserves variable-floor graph topology with ZERO dummy padding
  - Spatial dimension: Physical message passing across structural columns and floors
  - Temporal dimension: Global 1D Fourier spectral convolutions over continuous time
  - Modal Conditioning: Injects structural eigenvalue properties (T1, modal frequencies, mode descriptors)
    into latent spatial and spectral operator layers via FiLM modulation
  - Controlled ablation support: Unconditioned (EXP5 baseline), T1-only, Multi-Modal, and Shuffled conditioning
"""

import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gno import SpectralConv1d, SpatialGraphConv


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) generator.
    Maps conditioning vector c in R^{d_cond} to channel scale gamma and shift beta:
      h_mod = (1 + gamma) * h + beta
    """

    def __init__(self, cond_dim: int, channels: int, hidden_dim: int = 64):
        super().__init__()
        self.cond_dim = cond_dim
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * 2),
        )
        # Initialize final projection with small weights so initial modulation is near-identity
        # while allowing immediate non-zero gradient flow
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          cond: [N, cond_dim]
        Returns:
          gamma: [N, channels, 1]
          beta:  [N, channels, 1]
        """
        out = self.mlp(cond)  # [N, channels * 2]
        gamma, beta = torch.chunk(out, chunks=2, dim=-1)
        return gamma.unsqueeze(-1), beta.unsqueeze(-1)


class ConditionedSpatiotemporalGNOBlock(nn.Module):
    """
    Composite Spatiotemporal Graph Neural Operator Block with FiLM Modal Conditioning.
    Executes Spatial Graph Kernel Integration followed by 1D Temporal FNO,
    with adaptive modal modulation after each sub-layer.
    """

    def __init__(
        self,
        channels: int,
        modes: int = 64,
        edge_dim: int = 2,
        use_topology: bool = True,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.use_topology = use_topology
        self.cond_dim = cond_dim

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

        # FiLM condition generators (for spatial and spectral features)
        if cond_dim > 0:
            self.film_spat = FiLMBlock(cond_dim, channels)
            self.film_temp = FiLMBlock(cond_dim, channels)
        else:
            self.film_spat = None
            self.film_temp = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x: [N_total, channels, T]
          edge_index: [2, E_total]
          edge_attr: [E_total, edge_dim]
          cond: Optional conditioning tensor [N_total, cond_dim]
        """
        # 1. Spatial Graph Message Passing
        res = x
        x_spat = self.spatial_conv(x, edge_index, edge_attr)
        x = self.norm1(res + x_spat)

        # Apply spatial FiLM modulation if conditioned
        if self.film_spat is not None and cond is not None:
            gamma_s, beta_s = self.film_spat(cond)
            x = (1.0 + gamma_s) * x + beta_s

        # 2. Temporal Spectral Convolution
        res = x
        x_spec = self.spectral_conv(x)
        x_linear = self.w_time(x)
        x_temp = F.gelu(x_spec + x_linear)
        x = self.norm2(res + x_temp)

        # Apply temporal FiLM modulation if conditioned
        if self.film_temp is not None and cond is not None:
            gamma_t, beta_t = self.film_temp(cond)
            x = (1.0 + gamma_t) * x + beta_t

        # 3. Feed-forward expansion
        x = x + self.mlp(x)
        return x


class ConditionedSpatiotemporalGNO(nn.Module):
    """
    Physics/Modal-Conditioned Spatiotemporal Graph Neural Operator (GNO).

    Maps:
      x: [N_total, in_channels, T] (Multi-story building graphs across time)
      cond: [N_total, cond_dim] (Modal conditioning descriptors)
      -> [N_total, out_channels, T] (Predicted floor displacements, shear, energy)

    Variants:
      - cond_dim = 0: Unconditioned Baseline (equivalent to EXP5 GNO)
      - cond_dim = 1: T1-Conditioned GNO (EXP6-B)
      - cond_dim = 6: Multi-Modal Conditioned GNO (EXP6-C)
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
        cond_dim: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.n_layers = n_layers
        self.use_topology = use_topology
        self.cond_dim = cond_dim

        # 1. Lifting: Project input features to latent channel dimension
        self.lifting = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=1),
        )

        # 2. Spatiotemporal GNO Blocks with optional FiLM conditioning
        self.blocks = nn.ModuleList([
            ConditionedSpatiotemporalGNOBlock(
                channels=width,
                modes=modes,
                edge_dim=edge_dim,
                use_topology=use_topology,
                cond_dim=cond_dim,
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
        cond: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x: [N_total, in_channels, T]
          edge_index: [2, E_total]
          edge_attr: [E_total, edge_dim]
          cond: Optional conditioning tensor.
                Either [N_total, cond_dim] or [B, cond_dim] with batch_idx provided.
          batch_idx: Optional [N_total] graph index for each node.
        Returns:
          out: [N_total, out_channels, T]
        """
        # Resolve conditioning shape
        node_cond = None
        if self.cond_dim > 0 and cond is not None:
            if cond.dim() == 2 and cond.shape[0] != x.shape[0]:
                if cond.shape[0] == 1:
                    # Single-building inference: broadcast cond across all nodes
                    node_cond = cond.repeat(x.shape[0], 1)
                elif batch_idx is not None:
                    node_cond = cond[batch_idx]
                else:
                    raise ValueError(
                        f"cond shape {cond.shape} does not match node count {x.shape[0]} and batch_idx was not provided."
                    )
            else:
                node_cond = cond

        h = self.lifting(x)
        for block in self.blocks:
            h = block(h, edge_index, edge_attr, cond=node_cond)
        out = self.projection(h)
        return out
