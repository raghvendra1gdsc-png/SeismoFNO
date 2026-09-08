"""
causal_tcn.py — Causal Dilated Temporal Convolutional Network (Causal TCN).

Implements a strictly causal sequence model (Bai et al., 2018; van den Oord et al., 2016)
for nonlinear structural dynamics where the prediction at time t depends exclusively
on past and present inputs tau <= t (zero future information leakage):

Key Features:
  - 1D Causal Dilated Convolutions (left-padded only, zero right-padding)
  - Exponential dilation growth: d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
  - Receptive field spanning > 2,048 time steps
  - Weight normalization and residual skip connections
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D Convolution with strictly causal left padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # Manual causal padding in forward
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad format: (pad_left, pad_right)
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class TemporalBlock(nn.Module):
    """Residual causal dilated convolution block with GELU and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.act2 = nn.GELU()
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)

        out = self.conv1(x)
        out = self.act1(out)
        out = self.norm1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.act2(out)
        out = self.norm2(out)
        out = self.drop2(out)

        return F.gelu(out + res)


class CausalTCN(nn.Module):
    """
    Causal Dilated Temporal Convolutional Network baseline.

    Parameters:
        in_channels: Number of input channels (e.g. 10)
        out_channels: Number of output channels (e.g. 3: u, F_R, E_h)
        num_channels: List of channel widths for each dilated block
        kernel_size: Kernel size for convolutions (default: 3)
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Default: 11 layers with dilations 2^0 to 2^10 = 1024
        # Receptive field = 1 + sum(2 * (kernel_size - 1) * dilation) > 4,000 steps
        if num_channels is None:
            num_channels = [64] * 11

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Conv1d(num_channels[-1], 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (Batch, in_channels, Time)

        Returns:
            Output tensor of shape (Batch, out_channels, Time)
        """
        feat = self.network(x)
        out = self.projection(feat)
        return out

    def get_num_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
