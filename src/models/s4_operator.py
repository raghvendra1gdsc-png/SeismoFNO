"""
s4_operator.py — Continuous Structured State-Space Operator for Seismic Dynamics.

Implements the 6-block Continuous S4 State-Space Neural Operator parameter-matched
to ~1.192M parameters (+-1.5% of Standard FNO and Causal TCN).

Architecture Pipeline:
    Input x: [Batch, 10, 2048]
      │
      ├── Linear Encoder Projection: 10 -> 128
      │
      ├── 6 x S4 Residual Blocks:
      │      ┌────────────────────────────────────────────────────────┐
      │      │ LayerNorm(128)                                         │
      │      │ Continuous S4 Layer (N=64, HiPPO-LegS, Bilinear dt)     │
      │      │ GELU Activation + Dropout(0.05)                        │
      │      │ Pointwise Feedforward: Linear(128 -> 578 -> 128)       │
      │      │ Residual Connection: x = x + Block(x)                  │
      │      └────────────────────────────────────────────────────────┘
      │
      └── Linear Decoder Head: LayerNorm(128) -> Linear(128 -> 3)
            Output y: [Batch, 3, 2048] (u, F_R, E_h)
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.s4_layer import S4Layer


class S4Block(nn.Module):
    """Residual S4 Block with LayerNorm, S4 Layer, GELU, and Positionwise Feedforward."""

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_ff: int = 578,
        dropout: float = 0.05,
        memory_truncated: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s4 = S4Layer(
            d_model=d_model,
            d_state=d_state,
            memory_truncated=memory_truncated,
        )
        self.drop = nn.Dropout(dropout)

        # Positionwise feedforward
        self.ff1 = nn.Conv1d(d_model, d_ff, 1)
        self.ff2 = nn.Conv1d(d_ff, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [Batch, d_model, Length]
        """
        # S4 branch with LayerNorm across feature dimension
        # LayerNorm expects [Batch, Length, d_model]
        norm_x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        s4_out = self.s4(norm_x)
        s4_out = F.gelu(s4_out)
        s4_out = self.drop(s4_out)

        # Feedforward branch
        ff_out = self.ff1(s4_out)
        ff_out = F.gelu(ff_out)
        ff_out = self.drop(ff_out)
        ff_out = self.ff2(ff_out)

        # Residual connection
        return x + ff_out


class S4Operator(nn.Module):
    """
    Continuous Structured State-Space Operator.

    Args:
        in_channels: Input channels (default: 10).
        out_channels: Output channels (default: 3: u, F_R, E_h).
        d_model: Hidden channel width (default: 128).
        d_state: Continuous state dimension N (default: 64).
        n_blocks: Number of S4 residual blocks (default: 6).
        d_ff: Positionwise feedforward intermediate width (default: 578).
        dropout: Dropout probability (default: 0.05).
        memory_truncated: If True, imposes strong state decay truncating memory to < 10 steps.
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        d_model: int = 128,
        d_state: int = 64,
        n_blocks: int = 6,
        d_ff: int = 578,
        dropout: float = 0.05,
        memory_truncated: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.d_state = d_state
        self.n_blocks = n_blocks
        self.memory_truncated = memory_truncated

        # 1. Linear Input Encoder Projection: in_channels -> d_model
        self.encoder = nn.Conv1d(in_channels, d_model, 1)

        # 2. Stack of 6 S4 Residual Blocks
        self.blocks = nn.ModuleList([
            S4Block(
                d_model=d_model,
                d_state=d_state,
                d_ff=d_ff,
                dropout=dropout,
                memory_truncated=memory_truncated,
            )
            for _ in range(n_blocks)
        ])

        # 3. Linear Decoder Head: LayerNorm(d_model) -> Linear(d_model -> out_channels)
        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Conv1d(d_model, out_channels, 1)

    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for continuous S4 sequence processing.

        Input:
            x: [Batch, in_channels, Length]
            return_latent: If True, returns tuple (y, h_latent) where h_latent is [Batch, d_model, Length].
        Returns:
            y: [Batch, out_channels, Length]
            (optional) h_latent: [Batch, d_model, Length]
        """
        # Encoder projection: [B, 10, L] -> [B, 128, L]
        h = self.encoder(x)

        # S4 Residual Block Stack
        for block in self.blocks:
            h = block(h)

        # Penultimate latent feature representation: [B, 128, L]
        # LayerNorm across channels
        h_norm = self.final_norm(h.transpose(1, 2)).transpose(1, 2)

        # Decoder head: [B, 128, L] -> [B, 3, L]
        y = self.decoder(h_norm)

        if return_latent:
            return y, h_norm
        return y
