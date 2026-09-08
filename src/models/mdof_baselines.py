"""
mdof_baselines.py — Sequence and Pointwise Baselines for MDOF Seismic Surrogate Learning.

Implements:
  1. MDOFLSTMBaseline: Multi-layer recurrent LSTM sequence model.
  2. MDOFMLPBaseline: Deep residual Multi-Layer Perceptron.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDOFLSTMBaseline(nn.Module):
    """
    Recurrent LSTM Sequence Model for MDOF Dynamic Response.

    Accepts spatiotemporal input [B, C_in, S, T], flattens spatial channels into [B, T, C_in * S],
    passes through 3 LSTM layers, and projects to [B, C_out, S, T].
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        max_stories: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_stories = max_stories
        self.hidden_dim = hidden_dim

        # Input dimension per time step: C_in * S
        self.in_proj = nn.Linear(in_channels, hidden_dim // max_stories)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels * max_stories),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, S, T]
        Returns: [B, C_out, S, T]
        """
        b, c, s, t = x.shape

        # Pointwise story projection: [B, S, T, C_in] -> [B, T, S, H//S] -> [B, T, H]
        x_perm = x.permute(0, 3, 2, 1)  # [B, T, S, C_in]
        x_story = self.in_proj(x_perm)  # [B, T, S, H//S]

        # If s < max_stories, pad along spatial dimension
        if s < self.max_stories:
            pad_s = self.max_stories - s
            pad_tensor = torch.zeros(b, t, pad_s, x_story.shape[-1], device=x.device)
            x_story = torch.cat([x_story, pad_tensor], dim=2)

        x_flat = x_story.reshape(b, t, -1)  # [B, T, H]

        lstm_out, _ = self.lstm(x_flat)     # [B, T, H]
        proj_out = self.out_proj(lstm_out)  # [B, T, C_out * max_stories]

        # Unpack back to [B, C_out, S, T]
        proj_out = proj_out.reshape(b, t, self.max_stories, self.out_channels)  # [B, T, max_S, C_out]
        proj_out = proj_out[:, :, :s, :]  # Select active stories: [B, T, S, C_out]
        out = proj_out.permute(0, 3, 2, 1)  # [B, C_out, S, T]

        return out


class MDOFMLPBaseline(nn.Module):
    """
    Pointwise Deep Residual MLP for MDOF Dynamic Response.
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, S, T]
        Returns: [B, C_out, S, T]
        """
        b, c, s, t = x.shape
        # Permute to [B, S, T, C_in]
        x_pt = x.permute(0, 2, 3, 1)
        h = self.fc0(x_pt)

        for block in self.blocks:
            h = h + block(h)

        out_pt = self.out_head(h)  # [B, S, T, C_out]
        out = out_pt.permute(0, 3, 1, 2)  # [B, C_out, S, T]
        return out
