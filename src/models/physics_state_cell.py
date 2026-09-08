"""
physics_state_cell.py — Causal Physics State Cell for Latent Plastic State Integration.

Governing State Dynamics:
    h_s(t) = tanh( W_s * h_s(t-1) + W_z * z(t) + b_s )
    s_phys(t) = W_proj * h_s(t) + b_proj = [ u_p(t), alpha_b(t) ]^T

Key Properties:
    - Strictly causal: h_s(t) and s_phys(t) depend exclusively on past/present inputs tau <= t.
    - Zero future leakage: No right-side padding or temporal lookahead.
    - Explicit state reset: h_s(0) = 0 initialized per independent sequence.
    - C-accelerated standard Elman recurrence for sub-millisecond execution.
    - Outputs physical 2D state: plastic displacement u_p(t) and kinematic back-stress alpha_b(t).
"""

from typing import Tuple, Union, Optional
import torch
import torch.nn as nn


class PhysicsStateCell(nn.Module):
    """
    Causal Physics State Cell integrating latent sequence features into a 2D physical state.

    Args:
        in_channels: Dimension of the latent input features z(t) (default: 132).
        hidden_dim: Internal state hidden dimension (default: 16).
        out_state_dim: Dimension of the physical state vector (default: 2: [u_p, alpha_b]).
    """

    def __init__(
        self,
        in_channels: int = 132,
        hidden_dim: int = 16,
        out_state_dim: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_state_dim = out_state_dim

        # Standard C-accelerated Elman RNN for strictly causal tanh state recurrence
        self.rnn = nn.RNN(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
            bias=True,
        )

        # 1x1 Conv projection from hidden state to 2D physical state [u_p, alpha_b]
        self.state_proj = nn.Conv1d(hidden_dim, out_state_dim, kernel_size=1)

        # Initialize recurrent weights close to orthogonal for long-term state retention
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.state_proj.weight)
        nn.init.zeros_(self.state_proj.bias)

    def forward(
        self,
        z: torch.Tensor,
        return_hidden: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for causal state integration.

        Args:
            z: Latent sequence features [Batch, in_channels, Length]
            return_hidden: If True, also returns the internal hidden state trajectory [Batch, hidden_dim, Length]

        Returns:
            s_phys: Predicted physical state trajectory [Batch, 2, Length]
            (optional) h_s: Internal hidden state trajectory [Batch, hidden_dim, Length]
        """
        batch_size, channels, l_seq = z.shape
        # Permute to [Batch, Length, in_channels] for PyTorch RNN
        z_seq = z.transpose(1, 2)  # [B, L, C]

        # Initial state h_0 = 0 (explicitly reset per batch to guarantee sequence independence)
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=z.device, dtype=z.dtype)

        # Strictly causal recurrence over time: [B, L, hidden_dim]
        out, _ = self.rnn(z_seq, h0)

        # Permute back to [Batch, hidden_dim, Length]
        h_s = out.transpose(1, 2)

        # Linear projection to 2D physical state [Batch, 2, Length]
        s_phys = self.state_proj(h_s)

        if return_hidden:
            return s_phys, h_s
        return s_phys
