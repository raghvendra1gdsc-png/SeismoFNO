"""
data_loss.py — Objective, Physics-Informed, and Evaluation Losses for SeismoFNO.

Implements:
  1. RelativeL2Loss:
       L_rel = ||y - y_hat||_2 / (||y||_2 + eps)
  2. MultiChannelRelativeL2Loss:
       Weighted multi-target loss across channels (u, F_R, E_h).
  3. CompositePhysicsLoss:
       L = L_data + lambda_energy * L_energy + lambda_boundary * L_boundary
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from src.losses.energy_consistency_loss import EnergyConsistencyLoss
from src.losses.boundary_loss import BoundaryLoss


class RelativeL2Loss(nn.Module):
    """
    Relative L2 norm loss function:
        L(y, y_hat) = ||y - y_hat||_2 / (||y||_2 + eps)

    Can be aggregated via 'mean' or 'sum' across the batch.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predicted tensor of shape [Batch, Channels, Time] or [Batch, Time].
        target : torch.Tensor
            Target ground truth tensor of matching shape.

        Returns
        -------
        loss : torch.Tensor
            Relative L2 loss scalar.
        """
        b = pred.size(0)
        diff_norm = torch.norm(pred.view(b, -1) - target.view(b, -1), p=2, dim=1)
        target_norm = torch.norm(target.view(b, -1), p=2, dim=1) + self.eps

        sample_rel_l2 = diff_norm / target_norm

        if self.reduction == "mean":
            return torch.mean(sample_rel_l2)
        elif self.reduction == "sum":
            return torch.sum(sample_rel_l2)
        elif self.reduction == "none":
            return sample_rel_l2
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class MultiChannelRelativeL2Loss(nn.Module):
    """
    Weighted multi-channel relative L2 loss for multi-target operator learning:
        L = \\sum_c w_c * RelL2(pred[:, c, :], target[:, c, :])
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        reduction: str = "mean",
        eps: float = 1e-7,
    ):
        super().__init__()
        self.weights = weights
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : torch.Tensor
            [Batch, Channels, Time]
        target : torch.Tensor
            [Batch, Channels, Time]
        """
        n_channels = pred.size(1)
        weights = self.weights if self.weights is not None else [1.0] * n_channels

        total_loss = torch.tensor(0.0, device=pred.device)
        b = pred.size(0)

        for c in range(n_channels):
            p_c = pred[:, c, :].view(b, -1)
            t_c = target[:, c, :].view(b, -1)
            diff_norm = torch.norm(p_c - t_c, p=2, dim=1)
            target_norm = torch.norm(t_c, p=2, dim=1) + self.eps
            c_loss = torch.mean(diff_norm / target_norm)
            total_loss = total_loss + weights[c] * c_loss

        return total_loss


class CompositePhysicsLoss(nn.Module):
    """
    Composite physics-informed loss combining data loss with optional energy
    consistency and boundary initial-condition penalties:

        L_total = L_data + lambda_energy * L_energy + lambda_boundary * L_boundary
    """

    def __init__(
        self,
        data_weights: Optional[List[float]] = None,
        lambda_energy: float = 0.0,
        lambda_boundary: float = 0.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.data_loss = MultiChannelRelativeL2Loss(weights=data_weights, eps=eps)
        self.energy_loss = EnergyConsistencyLoss(eps=eps) if lambda_energy > 0.0 else None
        self.boundary_loss = BoundaryLoss() if lambda_boundary > 0.0 else None
        self.lambda_energy = lambda_energy
        self.lambda_boundary = lambda_boundary

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_physical: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pred : torch.Tensor
            Model prediction tensor [Batch, Channels, Time] (in training/normalized space).
        target : torch.Tensor
            Ground truth target tensor [Batch, Channels, Time] (matching pred space).
        pred_physical : torch.Tensor, optional
            Physical-unit predictions for computing work integral and boundary terms.
            If None, pred is used.

        Returns
        -------
        losses : dict
            {
                'loss': total scalar loss,
                'data_loss': data component,
                'energy_loss': energy consistency component (or 0),
                'boundary_loss': boundary component (or 0)
            }
        """
        l_data = self.data_loss(pred, target)
        total_loss = l_data

        phys_tensor = pred_physical if pred_physical is not None else pred

        l_energy = torch.tensor(0.0, device=pred.device)
        if self.energy_loss is not None and self.lambda_energy > 0.0:
            l_energy = self.energy_loss(phys_tensor)
            total_loss = total_loss + self.lambda_energy * l_energy

        l_boundary = torch.tensor(0.0, device=pred.device)
        if self.boundary_loss is not None and self.lambda_boundary > 0.0:
            l_boundary = self.boundary_loss(phys_tensor)
            total_loss = total_loss + self.lambda_boundary * l_boundary

        return {
            "loss": total_loss,
            "data_loss": l_data,
            "energy_loss": l_energy,
            "boundary_loss": l_boundary,
        }
