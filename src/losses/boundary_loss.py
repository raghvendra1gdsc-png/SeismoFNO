"""
boundary_loss.py — Initial Condition / Boundary Loss for Seismic Time-Series Operators.

Enforces physical initial conditions at t=0 for oscillators at rest:
    u(0) = 0
    F_R(0) = 0
    E_h(0) = 0

Penalizes deviations of the predicted initial state:
    L_boundary = || \\hat{u}(0) ||_2^2 + || \\hat{F}_R(0) ||_2^2 + || \\hat{E}_h(0) ||_2^2
"""

from typing import Optional, List
import torch
import torch.nn as nn


class BoundaryLoss(nn.Module):
    """
    Initial condition boundary loss for seismic response operators.

    Penalizes non-zero initial values at index t=0 across specified channels.
    """

    def __init__(
        self,
        channels: Optional[List[int]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted tensor of shape [Batch, Channels, Time] or [Batch, Time].

        Returns
        -------
        loss : torch.Tensor
            Boundary loss scalar.
        """
        if pred.dim() == 2:
            # [Batch, Time] -> initial state is [Batch, 0]
            init_vals = pred[:, 0]
            loss_per_sample = init_vals ** 2
        elif pred.dim() == 3:
            # [Batch, Channels, Time]
            if self.channels is not None:
                init_vals = pred[:, self.channels, 0]  # [B, C_sub]
            else:
                init_vals = pred[:, :, 0]              # [B, C]
            loss_per_sample = torch.sum(init_vals ** 2, dim=-1)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {pred.shape}")

        if self.reduction == "mean":
            return torch.mean(loss_per_sample)
        elif self.reduction == "sum":
            return torch.sum(loss_per_sample)
        elif self.reduction == "none":
            return loss_per_sample
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
