"""
fno_block.py — 1D Fourier Neural Operator Block.

Combines:
  1. Global Fourier spectral convolution (SpectralConv1d)
  2. Local linear spatial bypass / skip connection (1x1 Conv1d)
  3. Optional normalization and non-linear activation (GELU / LeakyReLU)
"""

from typing import Optional
import torch
import torch.nn as nn

from src.models.spectral_conv import SpectralConv1d


class FNOBlock1d(nn.Module):
    """
    Standard 1D Fourier Neural Operator Layer Block:
        v_{l+1}(x) = sigma( K(v_l)(x) + W v_l(x) )

    Parameters
    ----------
    width : int
        Number of latent feature channels.
    modes : int
        Number of Fourier modes to retain in spectral convolution.
    activation : str
        Activation function ('gelu', 'leaky_relu', 'relu', 'tanh').
    use_norm : bool
        Whether to include channel-wise batch/instance normalization.
    """

    def __init__(
        self,
        width: int,
        modes: int = 16,
        activation: str = "gelu",
        use_norm: bool = False,
    ):
        super().__init__()
        self.width = width
        self.modes = modes
        self.use_norm = use_norm

        self.spectral_conv = SpectralConv1d(
            in_channels=width,
            out_channels=width,
            modes1=modes,
        )
        self.skip_conv = nn.Conv1d(width, width, kernel_size=1)

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.1)
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.norm = nn.BatchNorm1d(width) if use_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Latent tensor [Batch, Width, Time_steps].

        Returns
        -------
        out : torch.Tensor
            Updated latent tensor [Batch, Width, Time_steps].
        """
        x_spectral = self.spectral_conv(x)
        x_skip = self.skip_conv(x)
        x_out = x_spectral + x_skip
        x_out = self.norm(x_out)
        return self.act(x_out)
