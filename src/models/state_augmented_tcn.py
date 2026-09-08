"""
state_augmented_tcn.py — State-Augmented Causal Temporal Convolutional Network.

Implements the within-family state control for EXP 2 (Mechanism B isolation).
Augments the strictly causal 11-layer dilated TCN backbone with an explicit
1D causal recurrent state integrator s(t) = tanh( W_s s(t-1) + W_x x(t) ).

Architecture Pipeline:
    Input x: [Batch, 10, 2048]
      │
      ├── Causal Recurrent State Integrator: x(t) -> s(t) in R^4
      │   Concatenation: [x(t), s(t)] in R^14
      │
      ├── 11 x Causal Dilated Residual Blocks (Width 136, Kernel 3, Dilation 1...1024)
      │
      └── Final Output Conv1d: 136 -> 3
            Output y: [Batch, 3, 2048] (u, F_R, E_h)
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.causal_tcn import TemporalBlock
from src.models.recurrent_state_cell import CausalRecurrentStateCell


class StateAugmentedCausalTCN(nn.Module):
    """
    State-Augmented Causal Dilated Temporal Convolutional Network.

    Args:
        in_channels: Base input channels (default: 10).
        out_channels: Output channels (default: 3: u, F_R, E_h).
        num_channels: List of channel widths for 11 dilated blocks (default: [136] * 11).
        kernel_size: Convolution kernel size (default: 3).
        dropout: Dropout probability (default: 0.05).
        state_dim: Dimension of the recurrent state channel s(t) (default: 4).
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.05,
        state_dim: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.state_dim = state_dim

        if num_channels is None:
            num_channels = [136] * 11

        # 1. Causal Recurrent State Integrator: 10 -> state_dim
        self.state_cell = CausalRecurrentStateCell(in_channels=in_channels, state_dim=state_dim)

        # 2. Causal Dilated Temporal Block Stack
        layers = []
        num_levels = len(num_channels)
        # Combined channels entering first block: in_channels + state_dim (e.g. 10 + 4 = 14)
        current_in = in_channels + state_dim

        for i in range(num_levels):
            dilation = 2 ** i
            current_out = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels=current_in,
                    out_channels=current_out,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current_in = current_out

        self.network = nn.ModuleList(layers)

        # 3. Output Projection Layer: 136 -> out_channels (3)
        self.head = nn.Conv1d(num_channels[-1], out_channels, kernel_size=1)

    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for State-Augmented Causal TCN.

        Input:
            x: [Batch, in_channels, Length]
            return_latent: If True, returns tuple (y, h_latent).
        Returns:
            y: [Batch, out_channels, Length]
            (optional) h_latent: [Batch, 136, Length]
        """
        # 1. Integrate causal state: [B, 10, L] -> [B, 4, L]
        s = self.state_cell(x)

        # 2. Concatenate state channel: [B, 10 + 4, L] -> [B, 14, L]
        x_aug = torch.cat([x, s], dim=1)

        # 3. Pass through 11 dilated residual blocks
        h = x_aug
        for layer in self.network:
            h = layer(h)

        # 4. Final output projection: [B, 136, L] -> [B, 3, L]
        y = self.head(h)

        if return_latent:
            return y, h
        return y
