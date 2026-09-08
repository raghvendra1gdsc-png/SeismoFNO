"""
high_dim_state_tcn.py — 64-Dimensional Unconstrained State Control Model for EXP 3 (H3C).

Architecture:
    Input x(t) [B, 10, L]
       │
       ▼
    Causal Encoder (6 dilated residual blocks, d_enc=132, dilations 1...32)
       │
       ├──► Latent Representation z(t) [B, 132, L]
       │       │
       │       ▼
       │    Unconstrained State Cell (Causal RNN 132 -> 64)
       │       │
       │       ▼
       │    Unconstrained 64D State: s_unconstrained(t) [B, 64, L]
       │       │
       └───────┼────────────────────────────────────────┐
               ▼                                        ▼
    Concatenation [z(t), s_unconstrained(t)] [B, 196, L]
       │
       ▼
    Causal Decoder (5 dilated residual blocks, d_dec=134, dilations 64...1024)
       │
       ▼
    Projection Head: 134 -> 3
       │
       ▼
    Output y(t) = [u(t), F_R(t), E_h(t)] [B, 3, L]

Parameter Budget:
    - 1,192,255 parameters (Delta = -0.02% vs 1,192,448 target).
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.causal_tcn import TemporalBlock


class HighDimStateCausalTCN(nn.Module):
    """
    64-Dimensional Unconstrained State Causal Temporal Convolutional Network.

    Args:
        in_channels: Input channels (default: 10).
        out_channels: Output channels (default: 3: u, F_R, E_h).
        encoder_dim: Latent dimension of the causal encoder (default: 132).
        decoder_dim: Latent dimension of the causal decoder (default: 134).
        state_dim: Unconstrained internal state dimension (default: 64).
        kernel_size: Convolution kernel size (default: 3).
        dropout: Dropout probability (default: 0.05).
        encoder_dilations: Dilations for encoder blocks (default: [1, 2, 4, 8, 16, 32]).
        decoder_dilations: Dilations for decoder blocks (default: [64, 128, 256, 512, 1024]).
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 3,
        encoder_dim: int = 132,
        decoder_dim: int = 134,
        state_dim: int = 64,
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

        # 2. Causal Unconstrained State RNN Cell (132 -> 64)
        self.rnn = nn.RNN(
            input_size=encoder_dim,
            hidden_size=state_dim,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
            bias=True,
        )

        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # 3. Causal Decoder Backbone (5 dilated residual blocks)
        # Input to first decoder block: encoder_dim + state_dim (132 + 64 = 196)
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

        # 4. Final Output Projection Head (134 -> 3)
        self.head = nn.Conv1d(decoder_dim, out_channels, kernel_size=1)

    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: torch.Tensor,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for 64D Unconstrained State Model.
        """
        batch_size = x.shape[0]

        # 1. Causal Encoder Pass: [B, 10, L] -> [B, 132, L]
        z = x
        for layer in self.encoder:
            z = layer(z)

        # 2. Causal State Integration: [B, 132, L] -> [B, 64, L]
        z_seq = z.transpose(1, 2)
        h0 = torch.zeros(1, batch_size, self.state_dim, device=x.device, dtype=x.dtype)
        out_rnn, _ = self.rnn(z_seq, h0)
        s_64d = out_rnn.transpose(1, 2)

        # 3. Concatenation: [B, 132 + 64, L] = [B, 196, L]
        h_combined = torch.cat([z, s_64d], dim=1)

        # 4. Causal Decoder Pass: [B, 196, L] -> [B, 134, L]
        h_dec = h_combined
        for layer in self.decoder:
            h_dec = layer(h_dec)

        # 5. Final Output Head: [B, 134, L] -> [B, 3, L]
        y = self.head(h_dec)

        if return_state:
            return y, s_64d
        return y
