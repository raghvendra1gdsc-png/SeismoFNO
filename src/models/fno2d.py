"""
fno2d.py — 2D Fourier Neural Operator Architecture for Spatiotemporal Building Dynamics.

Maps multi-channel ground motion and building inputs over the (Story, Time) continuous domain:
    v_0(s, t) = P(x(s, t))
    v_{l+1}(s, t) = sigma( K(v_l) + W v_l )
    u(s, t) = Q(v_L(s, t))

where K is the 2D Spectral Convolution operator across space (stories) and time.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spectral_conv import SpectralConv2d


class FNO2dBlock(nn.Module):
    """
    Single 2D FNO Block:
      v_{l+1} = GELU( SpectralConv2d(v_l) + Conv2d_1x1(v_l) )
    """

    def __init__(
        self,
        width: int,
        modes1: int = 4,
        modes2: int = 64,
        activation: str = "gelu",
    ):
        super().__init__()
        self.conv = SpectralConv2d(
            in_channels=width,
            out_channels=width,
            modes1=modes1,
            modes2=modes2,
        )
        self.w = nn.Conv2d(width, width, kernel_size=1)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: [Batch, Width, Story, Time]"""
        return self.act(self.conv(x) + self.w(x))


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for MDOF Seismic Response.

    Parameters
    ----------
    in_channels : int
        Number of input channels per (story, time) point.
    out_channels : int
        Number of output channels (e.g. 3 for u, F_R, E_h).
    modes1 : int
        Number of Fourier modes along story dimension.
    modes2 : int
        Number of Fourier modes along time dimension.
    width : int
        Hidden channel width.
    n_layers : int
        Number of Fourier Operator blocks.
    activation : str
        Nonlinear activation function ('gelu', 'relu', 'silu').
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 3,
        modes1: int = 4,
        modes2: int = 64,
        width: int = 48,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        # Lifting layer: [B, C_in, S, T] -> [B, Width, S, T]
        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)

        # Spectral convolution blocks
        self.blocks = nn.ModuleList([
            FNO2dBlock(
                width=width,
                modes1=modes1,
                modes2=modes2,
                activation=activation,
            )
            for _ in range(n_layers)
        ])

        # Projection head: [B, Width, S, T] -> [B, Out_channels, S, T]
        self.fc1 = nn.Conv2d(width, 128, kernel_size=1)
        self.fc2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [Batch, In_channels, Story, Time].

        Returns
        -------
        out : torch.Tensor
            Predicted response of shape [Batch, Out_channels, Story, Time].
        """
        # Lifting
        h = self.fc0(x)

        # Fourier neural operator blocks with residual skips
        for block in self.blocks:
            h = block(h)

        # Projection head
        h = self.act(self.fc1(h))
        out = self.fc2(h)
        return out
