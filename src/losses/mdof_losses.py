"""
mdof_losses.py — Multi-Story Spatiotemporal Physics-Informed Loss Functions.

Implements:
  1. MDOFDataLoss: Relative L2 loss averaged across all floors.
  2. MDOFEnergyConsistencyLoss: Penalizes violation of energy conservation per story:
         | integral(F_R,s * d(drift_s)) - E_{h,s} |
  3. MDOFBoundaryLoss: Enforces u_s(0) = 0 across all floors.
  4. MDOFCompositeLoss: Weighted composite objective.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from src.losses.data_loss import RelativeL2Loss


class MDOFDataLoss(nn.Module):
    """
    Multi-Story Relative L2 Data Loss.
    Computes weighted Relative L2 error for [u, F_R, E_h] across stories.
    """

    def __init__(self, data_weights: Optional[List[float]] = None):
        super().__init__()
        self.weights = data_weights or [1.0, 1.0, 1.0]
        self.rel_l2 = RelativeL2Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        pred, target shape: [Batch, Channels=3, Story, Time]
        """
        # Reshape to [B * Story, Channels, Time] to compute per-story metrics
        b, c, s, t = pred.shape
        pred_flat = pred.permute(0, 2, 1, 3).reshape(b * s, c, t)
        target_flat = target.permute(0, 2, 1, 3).reshape(b * s, c, t)

        loss_u = self.rel_l2(pred_flat[:, 0:1, :], target_flat[:, 0:1, :])
        loss_f = self.rel_l2(pred_flat[:, 1:2, :], target_flat[:, 1:2, :])
        loss_e = self.rel_l2(pred_flat[:, 2:3, :], target_flat[:, 2:3, :])

        total_loss = (
            self.weights[0] * loss_u
            + self.weights[1] * loss_f
            + self.weights[2] * loss_e
        )

        metrics = {
            "loss_u": loss_u.item(),
            "loss_f": loss_f.item(),
            "loss_e": loss_e.item(),
            "data_loss": total_loss.item(),
        }
        return total_loss, metrics


class MDOFEnergyConsistencyLoss(nn.Module):
    """
    Multi-Story Physics Loss: Enforces work-energy consistency across all stories:
        E_{h,s}(t) = integral_0^t F_{R,s}(tau) d(drift_s(tau))
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        pred shape: [Batch, Channels=3, Story, Time]
        """
        b, _, s, t = pred.shape
        u = pred[:, 0, :, :]    # (B, S, T)
        f_r = pred[:, 1, :, :]  # (B, S, T)
        e_h = pred[:, 2, :, :]  # (B, S, T)

        # Compute inter-story drift: drift_0 = u_0, drift_s = u_s - u_{s-1}
        drift = torch.zeros_like(u)
        drift[:, 0, :] = u[:, 0, :]
        if s > 1:
            drift[:, 1:, :] = u[:, 1:, :] - u[:, :-1, :]

        # Trapezoidal numerical integration of work d_drift * F_avg
        d_drift = drift[:, :, 1:] - drift[:, :, :-1]
        f_avg = 0.5 * (f_r[:, :, 1:] + f_r[:, :, :-1])
        d_work = f_avg * d_drift

        # Cumulative work: [B, S, T-1]
        work_cumsum = torch.cumsum(d_work, dim=-1)
        work_full = torch.zeros_like(e_h)
        work_full[:, :, 1:] = work_cumsum

        # Relative L2 mismatch between predicted energy and integrated work
        diff_norm = torch.norm(work_full - e_h, p=2, dim=-1)
        ref_norm = torch.norm(e_h, p=2, dim=-1) + self.eps

        loss = torch.mean(diff_norm / ref_norm)
        return loss


class MDOFBoundaryLoss(nn.Module):
    """
    Enforces initial displacement and energy condition u_s(0) = 0 across all floors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """pred shape: [Batch, Channels=3, Story, Time]"""
        u0 = pred[:, 0, :, 0]  # [B, S] at t=0
        e0 = pred[:, 2, :, 0]  # [B, S] at t=0
        return torch.mean(u0**2) + torch.mean(e0**2)


class MDOFCompositeLoss(nn.Module):
    """
    Composite Objective: Data Loss + lambda_E * EnergyLoss + lambda_0 * BoundaryLoss.
    """

    def __init__(
        self,
        data_weights: Optional[List[float]] = None,
        lambda_energy: float = 0.1,
        lambda_boundary: float = 0.05,
    ):
        super().__init__()
        self.data_loss_fn = MDOFDataLoss(data_weights=data_weights)
        self.energy_loss_fn = MDOFEnergyConsistencyLoss()
        self.boundary_loss_fn = MDOFBoundaryLoss()
        self.lambda_energy = lambda_energy
        self.lambda_boundary = lambda_boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        data_loss, metrics = self.data_loss_fn(pred, target)
        total_loss = data_loss

        if self.lambda_energy > 0:
            energy_loss = self.energy_loss_fn(pred)
            total_loss = total_loss + self.lambda_energy * energy_loss
            metrics["energy_loss"] = energy_loss.item()

        if self.lambda_boundary > 0:
            boundary_loss = self.boundary_loss_fn(pred)
            total_loss = total_loss + self.lambda_boundary * boundary_loss
            metrics["boundary_loss"] = boundary_loss.item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics
