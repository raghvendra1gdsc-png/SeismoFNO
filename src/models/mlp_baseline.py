"""
mlp_baseline.py — Multi-Layer Perceptron (MLP) Baseline for Seismic Structural Dynamics.

Maps multi-channel ground motion and structural parameter time series
    x(t) in R^{B x C_in x T} -> y(t) in R^{B x C_out x T}
using a deep feedforward network with residual blocks and layer normalization.
"""

from typing import Optional
import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """Residual feedforward block with LayerNorm and GELU activation."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        residual = x
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(residual + h)


class MLPBaseline(nn.Module):
    """Deep Multi-Layer Perceptron (MLP) baseline model."""

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input lifting layer
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Deep residual MLP backbone
        self.blocks = nn.ModuleList([
            MLPBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection head
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Output prediction tensor of shape (batch_size, out_channels, seq_len)
        """
        # (B, C_in, L) -> (B, L, C_in)
        x_seq = x.transpose(1, 2)

        # Lift features
        h = self.input_proj(x_seq)

        # Pass through residual MLP layers
        for block in self.blocks:
            h = block(h)

        # Project to target outputs
        out = self.output_proj(h)

        # (B, L, C_out) -> (B, C_out, L)
        return out.transpose(1, 2)

    def get_num_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
