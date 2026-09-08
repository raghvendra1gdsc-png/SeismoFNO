"""
fno1d.py — 1D Fourier Neural Operator (FNO-1D) for Seismic Structural Dynamics.

Architecture (Li et al., 2020):
  1. Lifting Layer P:
     Projects input channels (e.g. ground acceleration a_g(t) + structural parameters)
     into latent width d_v:
       v_0(t) = P(x(t))
  2. N Spectral Convolution Blocks:
     Applies global frequency filtering and local skip connections:
       v_{l+1}(t) = sigma( K(v_l)(t) + W v_l(t) )
  3. Projection Layer Q:
     Projects latent representations to output channels (u(t), F_R(t), E_h(t)):
       y_hat(t) = Q(v_N(t))
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn

from src.models.fno_block import FNOBlock1d


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator model.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 7: a_g(t), T, zeta, is_bilin, u_y, alpha, PGA).
    out_channels : int
        Number of output channels (e.g. 1 for displacement u(t), or 3 for [u, F_R, E_h]).
    modes : int
        Number of low-frequency Fourier modes kept in spectral convolutions (default: 16).
    width : int
        Latent channel dimension / hidden width (default: 32).
    n_layers : int
        Number of FNO blocks (default: 4).
    activation : str
        Activation function ('gelu', 'leaky_relu', 'relu').
    use_norm : bool
        Whether to use normalization in FNO blocks.
    projection_dim : Optional[int]
        Intermediate projection layer dimension (default: 2 * width).
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        modes: int = 16,
        width: int = 32,
        n_layers: int = 4,
        activation: str = "gelu",
        use_norm: bool = False,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        # 1. Lifting Layer P: in_channels -> width
        self.lifting = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1),
            nn.Conv1d(width, width, kernel_size=1),
        )

        # 2. FNO Blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(
                width=width,
                modes=modes,
                activation=activation,
                use_norm=use_norm,
            )
            for _ in range(n_layers)
        ])

        # 3. Projection Layer Q: width -> projection_dim -> out_channels
        proj_dim = projection_dim if projection_dim is not None else width * 2
        self.projection = nn.Sequential(
            nn.Conv1d(width, proj_dim, kernel_size=1),
            nn.GELU() if activation == "gelu" else nn.LeakyReLU(0.1),
            nn.Conv1d(proj_dim, out_channels, kernel_size=1),
        )

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
            Predicted response tensor [Batch, Out_channels, Time_steps].
        """
        # Lifting
        v = self.lifting(x)

        # Fourier operator layers with skip connections
        for block in self.fno_blocks:
            v = block(v)

        # Projection
        out = self.projection(v)
        return out

    def get_num_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
