"""
pg_tcn.py — Physics-Guided State-Conditioned Causal Neural Operator (PG-TCN).

Architecture:
    Input x(t) [B, 10, L]
       │
       ▼
    Causal Encoder (6 dilated residual blocks, d_enc=135, dilations 1...32)
       │
       ├──► Latent Representation z(t) [B, 135, L]
       │       │
       │       ▼
       │    Physics State Cell (Causal RNN 135 -> 16 -> 2)
       │       │
       │       ▼
       │    Predicted Physical State: s_pred(t) = [u_p(t), alpha_b(t)] [B, 2, L]
       │       │
       │       ▼ [Intervention Hook: s_cf(t) = s_pred(t) + Delta_s for t >= t_y]
       │    Active Physical State: s_active(t) [B, 2, L]
       │       │
       └───────┼────────────────────────────────────────┐
               ▼                                        ▼
    Concatenation [z(t), s_active(t)] [B, 137, L]
       │
       ▼
    Causal Decoder (5 dilated residual blocks, d_dec=136, dilations 64...1024)
       │
       ▼
    Projection Head: 136 -> 3
       │
       ▼
    Output y(t) = [u(t), F_R(t), E_h(t)] [B, 3, L]

Key Invariants:
    - Strictly causal: Left-padding on all convolutions; RNN hidden state h_0 = 0 reset per batch.
    - Zero future leakage: No right-side padding or future sequence lookahead.
    - Parameter budget: Matched to 1,192,849 params (+0.03% vs 1,192,448 target).
    - Modular counterfactual intervention: Modifies only the physical state bottleneck at t >= t_y.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.causal_tcn import TemporalBlock
from src.models.physics_state_cell import PhysicsStateCell


class PhysicsSupervisedCausalTCN(nn.Module):
    """
    Physics-Guided State-Conditioned Causal Temporal Convolutional Network (PG-TCN).

    Args:
        in_channels: Input channels (default: 10).
        out_channels: Output channels (default: 3: u, F_R, E_h).
        encoder_dim: Latent dimension of the causal encoder (default: 135).
        decoder_dim: Latent dimension of the causal decoder (default: 136).
        state_hidden_dim: Hidden dimension of the internal recurrent state cell (default: 16).
        state_dim: Physical state dimension (default: 2: [u_p, alpha_b]).
        kernel_size: Convolution kernel size (default: 3).
        dropout: Dropout probability (default: 0.05).
        encoder_dilations: Dilations for encoder blocks (default: [1, 2, 4, 8, 16, 32]).
        decoder_dilations: Dilations for decoder blocks (default: [64, 128, 256, 512, 1024]).
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        encoder_dim: int = 135,
        decoder_dim: int = 136,
        state_hidden_dim: int = 16,
        state_dim: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.05,
        encoder_dilations: Optional[List[int]] = None,
        decoder_dilations: Optional[List[int]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.state_hidden_dim = state_hidden_dim
        self.state_dim = state_dim

        if encoder_dilations is None:
            encoder_dilations = [1, 2, 4, 8, 16, 32]
        if decoder_dilations is None:
            decoder_dilations = [64, 128, 256, 512, 1024]

        # 1. Causal Encoder Backbone (6 dilated residual blocks)
        enc_layers = []
        curr_in = in_channels
        for dilation in encoder_dilations:
            enc_layers.append(
                TemporalBlock(
                    in_channels=curr_in,
                    out_channels=encoder_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            curr_in = encoder_dim
        self.encoder = nn.ModuleList(enc_layers)

        # 2. Causal Physics State Cell (RNN 135 -> 16 -> 2)
        self.state_cell = PhysicsStateCell(
            in_channels=encoder_dim,
            hidden_dim=state_hidden_dim,
            out_state_dim=state_dim,
        )

        # 3. Causal Decoder Backbone (5 dilated residual blocks)
        # Input to first decoder block: encoder_dim + state_dim (135 + 2 = 137)
        dec_layers = []
        curr_in = encoder_dim + state_dim
        for dilation in decoder_dilations:
            dec_layers.append(
                TemporalBlock(
                    in_channels=curr_in,
                    out_channels=decoder_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            curr_in = decoder_dim
        self.decoder = nn.ModuleList(dec_layers)

        # 4. Final Output Projection Head (136 -> 3)
        self.head = nn.Conv1d(decoder_dim, out_channels, kernel_size=1)

    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        intervention: Optional[Dict[str, Any]] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional causal counterfactual state intervention.

        Args:
            x: Input tensor [Batch, in_channels, Length]
            intervention: Optional dictionary specifying counterfactual perturbation:
                {
                    "t_y": int, float, or torch.Tensor,  # Yield time index (0 <= t_y < Length)
                    "delta_u_p": float,                  # Clamped plastic displacement shift
                    "delta_alpha_b": float,              # Clamped kinematic back-stress shift
                }
            return_state: If True, returns tuple (y, s_active)

        Returns:
            y: Output trajectory [Batch, 3, Length]
            (optional) s_active: Predicted or intervened physical state [Batch, 2, Length]
        """
        # 1. Causal Encoder Pass: [B, 10, L] -> [B, 135, L]
        z = x
        for layer in self.encoder:
            z = layer(z)

        # 2. Causal Physics State Prediction: [B, 135, L] -> [B, 2, L]
        s_pred = self.state_cell(z)

        # 3. Counterfactual Intervention Bottleneck
        if intervention is not None:
            s_active = s_pred.clone()
            batch_size = x.shape[0]
            t_y = intervention.get("t_y", 0)
            delta_u = intervention.get("delta_u_p", 0.0)
            delta_alpha = intervention.get("delta_alpha_b", 0.0)

            # Apply intervention strictly for t >= t_y
            if isinstance(t_y, (int, float, np_int := getattr(torch, "int", int))):
                t_idx = int(t_y)
                if 0 <= t_idx < s_active.shape[-1]:
                    s_active[:, 0, t_idx:] += delta_u
                    s_active[:, 1, t_idx:] += delta_alpha
            elif isinstance(t_y, torch.Tensor):
                if t_y.ndim == 0:
                    t_idx = int(t_y.item())
                    if 0 <= t_idx < s_active.shape[-1]:
                        s_active[:, 0, t_idx:] += delta_u
                        s_active[:, 1, t_idx:] += delta_alpha
                else:
                    for i in range(batch_size):
                        t_idx = int(t_y[i].item())
                        if 0 <= t_idx < s_active.shape[-1]:
                            s_active[i, 0, t_idx:] += delta_u
                            s_active[i, 1, t_idx:] += delta_alpha
        else:
            s_active = s_pred

        # 4. State-Conditioned Concatenation: [B, 135 + 2, L] = [B, 137, L]
        h_combined = torch.cat([z, s_active], dim=1)

        # 5. Causal Decoder Pass: [B, 137, L] -> [B, 136, L]
        h_dec = h_combined
        for layer in self.decoder:
            h_dec = layer(h_dec)

        # 6. Final Output Head: [B, 136, L] -> [B, 3, L]
        y = self.head(h_dec)

        if return_state:
            return y, s_active
        return y
