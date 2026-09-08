"""
lstm_baseline.py — Recurrent LSTM Sequence Model Baseline for Seismic Structural Dynamics.

Maps multi-channel ground motion and structural parameter time series
    x(t) in R^{B x C_in x T} -> y(t) in R^{B x C_out x T}
using a multi-layer Long Short-Term Memory (LSTM) network with input lifting
and output linear projection head.
"""

from typing import Optional
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """Multi-layer LSTM sequence model baseline."""

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Input lifting layer
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Recurrent LSTM backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Output projection head
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
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

        # Lift features: (B, L, C_in) -> (B, L, hidden_dim)
        h = self.input_proj(x_seq)

        # Pass through LSTM: (B, L, hidden_dim) -> (B, L, lstm_out_dim)
        lstm_out, _ = self.lstm(h)

        # Project to target outputs: (B, L, lstm_out_dim) -> (B, L, C_out)
        out = self.output_proj(lstm_out)

        # (B, L, C_out) -> (B, C_out, L)
        return out.transpose(1, 2)

    def get_num_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
