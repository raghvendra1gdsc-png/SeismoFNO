"""
energy_consistency_loss.py — Energy Conservation and Work-Energy Consistency Loss.

Implements the physical consistency constraint:
    E_h(t) = \\int_0^t F_R(\\tau) du(\\tau)

Penalizes the discrepancy between the predicted cumulative hysteretic energy
channel \\hat{E}_h(t) and the discrete trapezoidal work integral of predicted
restoring force \\hat{F}_R(t) over displacement increments d\\hat{u}(t).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


def compute_cumulative_trapezoidal_work(
    u: torch.Tensor,
    f_r: torch.Tensor,
) -> torch.Tensor:
    """
    Computes cumulative discrete trapezoidal work:
        W(t_k) = \\sum_{i=1}^k 0.5 * (F_R(i) + F_R(i-1)) * (u(i) - u(i-1))
    with W(0) = 0.

    Parameters
    ----------
    u : torch.Tensor
        Displacement tensor of shape [Batch, Time].
    f_r : torch.Tensor
        Restoring force tensor of shape [Batch, Time].

    Returns
    -------
    work : torch.Tensor
        Cumulative work tensor of shape [Batch, Time], with work[:, 0] = 0.
    """
    # Force midpoint: [B, T-1]
    f_mid = 0.5 * (f_r[:, 1:] + f_r[:, :-1])
    # Displacement increment: [B, T-1]
    du = u[:, 1:] - u[:, :-1]
    # Step work: [B, T-1]
    dw = f_mid * du
    # Cumulative sum: [B, T-1]
    cum_w = torch.cumsum(dw, dim=-1)
    # Prepend zero for t=0 initial state: [B, T]
    zeros = torch.zeros(u.size(0), 1, device=u.device, dtype=u.dtype)
    work = torch.cat([zeros, cum_w], dim=-1)
    return work


class EnergyConsistencyLoss(nn.Module):
    """
    Energy consistency loss penalizing discrepancy between predicted hysteretic
    energy and the internal work integral of predicted force over displacement:

        L_energy = || \\hat{E}_h - \\hat{W} ||_2 / (|| \\hat{E}_h ||_2 + eps)
    """

    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-7,
        u_channel: int = 0,
        f_channel: int = 1,
        e_channel: int = 2,
    ):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.u_channel = u_channel
        self.f_channel = f_channel
        self.e_channel = e_channel

    def forward(
        self,
        pred: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        f_r: Optional[torch.Tensor] = None,
        e_h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor, optional
            Tensor of shape [Batch, Channels, Time] where channels contain (u, f_r, e_h).
        u, f_r, e_h : torch.Tensor, optional
            Individual tensors of shape [Batch, Time] if passed separately.

        Returns
        -------
        loss : torch.Tensor
            Scalar energy consistency loss.
        """
        if u is None or f_r is None or e_h is None:
            if pred is None or pred.dim() != 3 or pred.size(1) < 3:
                raise ValueError("Expected either individual (u, f_r, e_h) or pred with >= 3 channels.")
            u = pred[:, self.u_channel, :]
            f_r = pred[:, self.f_channel, :]
            e_h = pred[:, self.e_channel, :]

        # Compute internal work integral
        work = compute_cumulative_trapezoidal_work(u, f_r)

        # Discrepancy between predicted energy channel and discrete work integral
        diff = torch.norm(e_h - work, p=2, dim=-1)
        # Regularize scale to prevent division by near-zero numbers in linear-elastic regime
        scale = torch.clamp(torch.norm(e_h, p=2, dim=-1), min=1.0)
        sample_loss = diff / scale

        if self.reduction == "mean":
            return torch.mean(sample_loss)
        elif self.reduction == "sum":
            return torch.sum(sample_loss)
        elif self.reduction == "none":
            return sample_loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
