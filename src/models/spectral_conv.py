"""
spectral_conv.py — 1D Spectral Convolution Layer for Fourier Neural Operators.

Implements discrete Fourier space convolution via Real FFT (rfft):
    K(v)(x) = F^{-1}( R(k) * F(v)(k) )(x)
where R(k) is a learnable complex parameter tensor for k in [0, modes1 - 1].
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution Layer (Li et al., 2020).

    Applies frequency-domain filtering by:
      1. Computing 1D real Fast Fourier Transform (rfft) along the spatial/time dimension.
      2. Multiplying the lowest `modes1` frequency modes by a learnable complex weight tensor.
      3. Computing the inverse real FFT (irfft) back to the physical time domain.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    modes1 : int
        Number of Fourier modes to retain (0 <= modes1 <= N/2 + 1).
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        # Parameter initialization: scale by 1 / (in_channels * out_channels)
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication of Fourier modes along channels:
            (batch, in_channel, modes), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
        """
        return torch.einsum("bix,iox->box", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [Batch, In_channels, Time_steps].

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape [Batch, Out_channels, Time_steps].
        """
        batchsize = x.shape[0]
        n_steps = x.shape[-1]

        # Compute 1D real FFT: [B, C_in, n_steps // 2 + 1]
        x_ft = torch.fft.rfft(x, dim=-1)

        # Allocate complex output tensor in frequency domain
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            n_steps // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        # Truncate to active modes
        active_modes = min(self.modes1, x_ft.shape[-1])
        out_ft[:, :, :active_modes] = self.compl_mul1d(
            x_ft[:, :, :active_modes], self.weights1[:, :, :active_modes]
        )

        # Inverse real FFT back to time domain
        x_out = torch.fft.irfft(out_ft, n=n_steps, dim=-1)
        return x_out


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer for Spatiotemporal Operator Learning (Li et al., 2020).

    Computes 2D Fourier integral operator over (Story, Time) domain:
      x_ft = rfft2(x)
      out_ft = W * x_ft (truncated to modes1 x modes2)
      x_out = irfft2(out_ft)

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    modes1 : int
        Number of Fourier modes along spatial/story dimension (dim -2).
    modes2 : int
        Number of Fourier modes along time dimension (dim -1).
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int = 4, modes2: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # Weights for top-left frequency quadrant (positive spatial, positive temporal)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        # Weights for bottom-left frequency quadrant (negative spatial, positive temporal)
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [Batch, In_channels, Story, Time].

        Returns
        -------
        out : torch.Tensor
            Output of shape [Batch, Out_channels, Story, Time].
        """
        batchsize = x.shape[0]
        size_s = x.shape[-2]
        size_t = x.shape[-1]

        # 2D real FFT: [B, C_in, S, T // 2 + 1]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        # Output Fourier representation
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            size_s,
            size_t // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        m1 = min(self.modes1, size_s // 2 + 1 if size_s > 1 else 1)
        m2 = min(self.modes2, x_ft.shape[-1])

        # Top-left quadrant (positive spatial frequencies)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bist,iost->bost", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )

        # Bottom-left quadrant (negative spatial frequencies) if size_s > 1
        if size_s > 1 and m1 > 1:
            out_ft[:, :, -m1 + 1 :, :m2] = torch.einsum(
                "bist,iost->bost", x_ft[:, :, -m1 + 1 :, :m2], self.weights2[:, :, -m1 + 1 :, :m2]
            )

        # Inverse 2D real FFT back to physical domain
        x_out = torch.fft.irfft2(out_ft, s=(size_s, size_t), dim=(-2, -1))
        return x_out
