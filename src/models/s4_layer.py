"""
s4_layer.py — Continuous Structured State-Space (S4) Layer for Operator Learning.

Implements the continuous-time Structured State-Space sequence model (Gu et al., 2021, 2022)
with diagonalized HiPPO-LegS initialization, Bilinear (Tustin) discretization,
and strictly causal convolutional / recurrent state transitions.

Governing Continuous-Time State Equation:
    h_dot(t) = A h(t) + B x(t)
    y(t)     = C h(t) + D x(t)

Key Properties:
    - Diagonalized HiPPO representation for optimal continuous memory projection.
    - Bilinear discretization preserving continuous operator mapping.
    - Parameter-matched to ~1.192M parameters.
    - Evaluation kernel caching for sub-millisecond inference across test sets.
    - Zero cross-sample state persistence (h_0 = 0 initialized for every sequence).
    - Memory-truncated ablation mode for isolating state memory (Mechanism B).
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_hippo_diagonal(n: int) -> torch.Tensor:
    """
    Generate the continuous Diagonal HiPPO-LegS transition eigenvalues:
        Lambda_n = -1/2 + i * pi * n   for n in [0, N-1]
    Guarantees negative real part for bounded-input bounded-output (BIBO) stability.
    """
    real = -0.5 * torch.ones(n, dtype=torch.float32)
    imag = math.pi * torch.arange(n, dtype=torch.float32)
    return torch.complex(real, imag)


class S4Layer(nn.Module):
    """
    Continuous-Time Structured State-Space Layer (S4D).

    Args:
        d_model: Feature dimension (channels).
        d_state: Latent continuous state dimension N (e.g. 64).
        dt_min: Minimum discretization step size Delta (default: 0.001).
        dt_max: Maximum discretization step size Delta (default: 0.1).
        memory_truncated: If True, imposes strong artificial dissipation (decay -> infinity)
                          truncating the state memory horizon to < 10 time steps.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        memory_truncated: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.memory_truncated = memory_truncated
        self.cached_k: Optional[torch.Tensor] = None

        # 1. State Matrix Lambda (Complex Diagonal, d_model x d_state)
        hippo = make_hippo_diagonal(d_state)  # [N]
        lambda_real = hippo.real.unsqueeze(0).repeat(d_model, 1)  # [d_model, N]
        lambda_imag = hippo.imag.unsqueeze(0).repeat(d_model, 1)  # [d_model, N]

        self.log_lambda_real = nn.Parameter(torch.log(-lambda_real))
        self.lambda_imag = nn.Parameter(lambda_imag)

        # 2. Input Projection Matrix B (Complex, d_model x d_state)
        self.b_real = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.b_imag = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))

        # 3. Output Projection Matrix C (Complex, d_model x d_state)
        self.c_real = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.c_imag = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))

        # 4. Feedthrough Direct Term D (Real, d_model)
        self.d = nn.Parameter(torch.zeros(d_model))

        # 5. Continuous Step Size log(Delta) (Real, d_model)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def get_continuous_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct complex continuous parameters: Lambda, B, C."""
        lambda_real = -torch.exp(self.log_lambda_real)
        if self.memory_truncated:
            lambda_real = lambda_real - 100.0

        Lambda = torch.complex(lambda_real, self.lambda_imag)  # [d_model, d_state]
        B = torch.complex(self.b_real, self.b_imag)           # [d_model, d_state]
        C = torch.complex(self.c_real, self.c_imag)           # [d_model, d_state]
        return Lambda, B, C

    def compute_causal_kernel(self, l_seq: int) -> torch.Tensor:
        """
        Compute the continuous-time discretized causal convolutional kernel K in R^{d_model x L}:
            K_k = 2 * Re( C * A_bar^k * B_bar )
        via Bilinear (Tustin) transformation.
        """
        Lambda, B, C = self.get_continuous_matrices()  # [d_model, N]
        dt = torch.exp(self.log_dt).unsqueeze(-1)     # [d_model, 1]

        # Bilinear / Tustin discretization:
        # A_bar: [d_model, N], B_bar: [d_model, N]
        dt_A = dt * Lambda  # [d_model, N]
        A_bar = (2.0 + dt_A) / (2.0 - dt_A)  # [d_model, N]
        B_bar = (2.0 * dt * B) / (2.0 - dt_A)  # [d_model, N]

        # C * B_bar: [d_model, N]
        cb = C * B_bar  # [d_model, N]

        # Power calculation: P_k = A_bar^k
        log_A_bar = torch.log(A_bar)  # [d_model, N]
        steps = torch.arange(l_seq, device=A_bar.device, dtype=torch.float32)  # [L]

        # P: [d_model, N, L]
        powers = torch.exp(log_A_bar.unsqueeze(-1) * steps)  # [d_model, N, L]

        # K_k = 2 * Re( sum_n cb_n * P_n,k )
        k_complex = torch.sum(cb.unsqueeze(-1) * powers, dim=1)  # [d_model, L]
        kernel = 2.0 * k_complex.real  # [d_model, L]
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward convolution with strictly causal left-padding:
            x: [Batch, d_model, Length]
        Returns:
            y: [Batch, d_model, Length]
        """
        batch_size, d_model, l_seq = x.shape

        # 1. Compute causal convolutional kernel K (with evaluation caching for blazing speed)
        if self.training or self.cached_k is None or self.cached_k.shape[-1] != l_seq:
            K = self.compute_causal_kernel(l_seq)
            if not self.training:
                self.cached_k = K
        else:
            K = self.cached_k

        # 2. FFT Convolution with linear zero padding to 2*L
        fft_len = 2 * l_seq
        x_fft = torch.fft.rfft(x, n=fft_len, dim=-1)      # [Batch, d_model, fft_len//2 + 1]
        k_fft = torch.fft.rfft(K, n=fft_len, dim=-1)      # [d_model, fft_len//2 + 1]

        # Frequency multiplication
        y_fft = x_fft * k_fft.unsqueeze(0)                # [Batch, d_model, fft_len//2 + 1]
        y_conv = torch.fft.irfft(y_fft, n=fft_len, dim=-1) # [Batch, d_model, fft_len]

        # Slice strictly the first l_seq time steps (causal response)
        y = y_conv[..., :l_seq]

        # 3. Direct Feedthrough D * x
        y = y + x * self.d.unsqueeze(0).unsqueeze(-1)
        return y
