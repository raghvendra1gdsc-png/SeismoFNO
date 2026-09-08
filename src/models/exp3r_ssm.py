"""
exp3r_ssm.py — Authoritative EXP3-R Pure Recurrent State-Space Model.

Mathematical Formulation:
  Inputs:  x_t = [a_g(t), T, zeta, u_y, alpha]^T in R^5
           (NO global PGA, NO future leakage, NO encoder-decoder bypass)

  State:   h_t = [s_t^phys, q_t]^T in R^64
           where s_t^phys = [u_t, v_t, u_p,t]^T in R^3 (physically supervised)
                 q_t in R^61 (unconstrained complementary latent memory)

  Transition:
           h_{t+1} = h_t + F_theta(h_t, x_t)

  Readout:
           y_t = G_theta(h_t) = [u(t), F_R(t), E_diss(t)]^T in R^3

  Graph Properties:
           - NO direct x_t -> y_t path
           - NO separate encoder feature map
           - Intervention Delta h modifies ONLY the recurrent state h at t_y
           - Downstream effect propagates strictly through repeated F_theta applications
"""

import math
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class StateTransitionBlock(nn.Module):
    """Deep residual transition block for F_theta."""
    def __init__(self, in_dim: int = 69, hidden_dim: int = 601, state_dim: int = 64, num_layers: int = 4):
        super().__init__()
        layers = []
        # Input projection
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())

        # Deep hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())

        # State delta output projection
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # [B, 64] + [B, 5] -> [B, 69]
        hx = torch.cat([h, x], dim=-1)
        return self.net(hx)


class StateReadoutHead(nn.Module):
    """Pure state readout G_theta mapping h_t -> y_t."""
    def __init__(self, state_dim: int = 64, hidden_dim: int = 300, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class PureRecurrentSSM(nn.Module):
    """
    Authoritative EXP 3-R Pure Recurrent State-Space Neural Operator.
    Matches the ~1.192M parameter budget with zero encoder-decoder bypass.
    """
    def __init__(
        self,
        in_channels: int = 5,
        state_dim: int = 64,
        phys_dim: int = 3,
        hidden_dim: int = 601,
        num_layers: int = 4,
        out_channels: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.state_dim = state_dim
        self.phys_dim = phys_dim
        self.latent_dim = state_dim - phys_dim  # 61
        self.out_channels = out_channels

        # Transition operator: F_theta(h_t, x_t)
        self.transition = StateTransitionBlock(
            in_dim=state_dim + in_channels,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_layers=num_layers,
        )

        # Readout head: G_theta(h_t)
        self.readout = StateReadoutHead(
            state_dim=state_dim,
            hidden_dim=hidden_dim // 2,
            out_dim=out_channels,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        intervention: Optional[Dict[str, Any]] = None,
        return_state: bool = False,
        static_hold: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward simulation over time steps.
        
        Args:
            x: Input tensor [Batch, Channels=5, Time=2048]
            intervention: Dict with keys:
                - 't_idx': int, time step at which intervention is applied
                - 'delta_h': Tensor [Batch, 64] or [64], perturbation to inject at t_idx
            return_state: If True, returns full state history h_history [Batch, 64, Time]
            static_hold: If True, freezes recurrent state after t_idx at h_{t_idx}^+ (Static-Hold Diagnostic)
            
        Returns:
            y: Output trajectory [Batch, Channels=3, Time=2048]
            (Optional) s_phys: Physical state trajectory [Batch, 3, Time=2048]
            (Optional) h_full: Complete state trajectory [Batch, 64, Time=2048]
        """
        batch_size, _, time_steps = x.shape
        device = x.device

        # Exact state reset: h_0 = 0
        h_t = torch.zeros(batch_size, self.state_dim, device=device)

        h_list = []
        interv_step = intervention.get("t_idx", -1) if intervention else -1
        delta_h = intervention.get("delta_h", None) if intervention else None

        if delta_h is not None and delta_h.dim() == 1:
            delta_h = delta_h.unsqueeze(0).expand(batch_size, -1)

        for t in range(time_steps):
            # 1. Apply intervention at t == interv_step
            if t == interv_step and delta_h is not None:
                h_t = h_t + delta_h

            h_list.append(h_t)

            # 2. State transition to next step
            if t < time_steps - 1:
                if static_hold and t >= interv_step and interv_step >= 0:
                    # Static-Hold diagnostic: state remains frozen at h_{t_idx}^+
                    h_t = h_t
                else:
                    # Pure dynamic recurrent transition
                    x_t = x[:, :, t]  # [B, 5]
                    delta_h_dynamic = self.transition(h_t, x_t)
                    h_t = h_t + delta_h_dynamic

        # Stack states along time dimension: [B, 64, Time]
        h_full = torch.stack(h_list, dim=-1)

        # Pointwise readout across all time steps: [B, 64, Time] -> [B, Time, 64] -> [B, Time, 3] -> [B, 3, Time]
        h_flat = h_full.permute(0, 2, 1)
        y_flat = self.readout(h_flat)
        y = y_flat.permute(0, 2, 1)

        if return_state:
            s_phys = h_full[:, :self.phys_dim, :]  # [B, 3, Time]: u, v, u_p
            return y, s_phys, h_full

        return y
