"""
test_resolution_invariance.py — Unit Tests for FNO Zero-Shot Resolution Invariance.
"""

import pytest
import torch
from src.models.fno1d import FNO1d


def test_fno_mesh_invariance_arbitrary_sequence_lengths():
    in_channels = 10
    out_channels = 3
    modes = 16
    width = 32

    model = FNO1d(
        in_channels=in_channels,
        out_channels=out_channels,
        modes=modes,
        width=width,
        n_layers=2,
    )
    model.eval()

    # Test across multiple arbitrary sequence lengths
    resolutions = [128, 256, 512, 1024, 2048, 4096]
    batch_size = 2

    for L in resolutions:
        x = torch.randn(batch_size, in_channels, L)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, out_channels, L), f"Failed for resolution L={L}: got shape {out.shape}"
        assert not torch.isnan(out).any(), f"NaN encountered at resolution L={L}"
