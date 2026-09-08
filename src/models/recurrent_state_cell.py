"""
recurrent_state_cell.py — Strictly Causal 1D Recurrent State Integrator.

Implements a lightweight causal continuous-time state integrator cell that prepends
an explicit recurrent state memory variable s(t) to feedforward convolutional operators.

Governing State Recurrence:
    s(t) = tanh( W_s * s(t-1) + W_x * x(t) + b )
    with initial condition s(0) = 0 for every independent sequence.

Key Properties:
    - Strictly causal: s(t) depends strictly on past and present inputs tau <= t.
    - Zero future leakage: No right-side padding or future lookahead.
    - Explicit state reset semantics (s_0 = 0 initialized per batch).
    - C-accelerated standard Elman recurrence for sub-millisecond execution.
    - Enables within-family state ablation for Causal TCN (Mechanism B isolation).
"""

import torch
import torch.nn as nn


class CausalRecurrentStateCell(nn.Module):
    """
    1D Causal Recurrent State Integrator.

    Args:
        in_channels: Input dimension (e.g. 10 channels).
        state_dim: Dimension of the recurrent state s(t) (e.g. 4).
    """

    def __init__(self, in_channels: int = 10, state_dim: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.state_dim = state_dim

        # Standard C-accelerated Elman RNN for exact, strictly causal tanh state recurrence
        self.rnn = nn.RNN(
            input_size=in_channels,
            hidden_size=state_dim,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
            bias=True,
        )

        # Initialize recurrent weights close to orthogonal for long-term state retention
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [Batch, in_channels, Length]
        Returns:
            s: [Batch, state_dim, Length] (Integrated state trajectory)
        """
        batch_size, channels, l_seq = x.shape
        # Permute to [Batch, Length, in_channels] for PyTorch RNN
        x_seq = x.transpose(1, 2)  # [B, L, C]

        # Initial state h_0 = 0 (explicitly reset per batch)
        h0 = torch.zeros(1, batch_size, self.state_dim, device=x.device, dtype=x.dtype)

        # Strictly causal recurrence over time: [B, L, state_dim]
        out, _ = self.rnn(x_seq, h0)

        # Permute back to [Batch, state_dim, Length]
        s_traj = out.transpose(1, 2)
        return s_traj
