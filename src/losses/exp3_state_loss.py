"""
exp3_state_loss.py — Composite Loss Function for EXP 3 Physics-Guided State Supervision.

Loss Formulation:
    L_total = L_response + lambda_state * L_state + lambda_energy * L_energy

Where:
    L_response: Relative L_2 error across outputs [u, F_R, E_h]
    L_state: Mean squared relative error across physical state variables [u_p, alpha_b]
    L_energy: Dynamic energy balance residual penalty
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class EXP3CompositeLoss(nn.Module):
    """
    Composite loss function for Physics-Guided Neural Operator training.

    Args:
        lambda_state: Weight for auxiliary physical state loss (default: 0.20).
        lambda_energy: Weight for physical energy balance loss (default: 0.10).
        eps_u: Epsilon denominator floor for displacement (default: 1e-4).
        eps_f: Epsilon denominator floor for restoring force (default: 1.0).
        eps_e: Epsilon denominator floor for hysteretic energy (default: 1e-2).
        eps_state: Epsilon denominator floor for physical state (default: 1e-6).
    """

    def __init__(
        self,
        lambda_state: float = 0.20,
        lambda_energy: float = 0.10,
        eps_u: float = 1e-4,
        eps_f: float = 1.0,
        eps_e: float = 1e-2,
        eps_state: float = 1e-6,
    ):
        super().__init__()
        self.lambda_state = lambda_state
        self.lambda_energy = lambda_energy
        self.eps_u = eps_u
        self.eps_f = eps_f
        self.eps_e = eps_e
        self.eps_state = eps_state

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        s_pred: Optional[torch.Tensor] = None,
        s_true: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute composite multi-objective loss.

        Args:
            y_pred: Predicted output [Batch, 3, Length] -> [u, F_R, E_h]
            y_true: Ground truth output [Batch, 3, Length]
            s_pred: (Optional) Predicted physical state [Batch, 2, Length] -> [u_p, alpha_b]
            s_true: (Optional) Ground truth physical state [Batch, 2, Length]

        Returns:
            total_loss: Scalar composite loss tensor
            metrics: Dictionary of individual loss components
        """
        batch_size = y_pred.shape[0]

        # 1. Output Trajectory Relative L2 Loss
        diff_u = y_pred[:, 0, :] - y_true[:, 0, :]
        norm_u = torch.norm(y_true[:, 0, :], p=2, dim=-1) + self.eps_u
        loss_u = torch.mean(torch.norm(diff_u, p=2, dim=-1) / norm_u)

        diff_f = y_pred[:, 1, :] - y_true[:, 1, :]
        norm_f = torch.norm(y_true[:, 1, :], p=2, dim=-1) + self.eps_f
        loss_f = torch.mean(torch.norm(diff_f, p=2, dim=-1) / norm_f)

        diff_e = y_pred[:, 2, :] - y_true[:, 2, :]
        norm_e = torch.norm(y_true[:, 2, :], p=2, dim=-1) + self.eps_e
        loss_e = torch.mean(torch.norm(diff_e, p=2, dim=-1) / norm_e)

        l_response = (loss_u + loss_f + loss_e) / 3.0

        # 2. Physics State Supervision Loss (if state targets provided)
        if s_pred is not None and s_true is not None and self.lambda_state > 0.0:
            diff_up = s_pred[:, 0, :] - s_true[:, 0, :]
            norm_up = torch.norm(s_true[:, 0, :], p=2, dim=-1) + self.eps_state
            loss_up = torch.mean(torch.norm(diff_up, p=2, dim=-1) / norm_up)

            diff_ab = s_pred[:, 1, :] - s_true[:, 1, :]
            norm_ab = torch.norm(s_true[:, 1, :], p=2, dim=-1) + self.eps_state
            loss_ab = torch.mean(torch.norm(diff_ab, p=2, dim=-1) / norm_ab)

            l_state = 0.5 * (loss_up + loss_ab)
        else:
            l_state = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        # 3. Dynamic Energy Residual Loss
        # Trapz discrete approximation of int F_R du
        u_pred = y_pred[:, 0, :]
        f_pred = y_pred[:, 1, :]
        e_pred = y_pred[:, 2, :]

        du = u_pred[:, 1:] - u_pred[:, :-1]
        f_mid = 0.5 * (f_pred[:, 1:] + f_pred[:, :-1])
        de_h = f_mid * du
        e_h_calc = torch.cat([torch.zeros(batch_size, 1, device=y_pred.device), torch.cumsum(de_h, dim=-1)], dim=-1)

        diff_energy = torch.abs(e_pred - e_h_calc)
        norm_energy = torch.amax(torch.abs(y_true[:, 2, :]), dim=-1, keepdim=True) + self.eps_e
        l_energy = torch.mean(diff_energy / norm_energy)

        # Composite Total Loss
        total_loss = l_response + self.lambda_state * l_state + self.lambda_energy * l_energy

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_response": l_response.detach(),
            "loss_u": loss_u.detach(),
            "loss_f": loss_f.detach(),
            "loss_e": loss_e.detach(),
            "loss_state": l_state.detach() if isinstance(l_state, torch.Tensor) else torch.tensor(l_state),
            "loss_energy": l_energy.detach(),
        }

        return total_loss, metrics
