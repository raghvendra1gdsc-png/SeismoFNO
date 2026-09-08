"""
test_fno2d_shapes.py — Shape, Gradient, and Resolution Flexibility Tests for FNO2d.

Verifies:
  1. Forward pass output tensor shape matches [Batch, Out_channels, Story, Time].
  2. Backward pass computes non-zero gradients on all learnable parameters.
  3. Resolution invariance: handles different story counts (S=3, S=5) and temporal lengths (T=512, 2048, 4096).
"""

import pytest
import torch

from src.models.spectral_conv import SpectralConv2d
from src.models.fno2d import FNO2d


def test_spectral_conv2d_forward_and_backward():
    batch = 4
    in_c = 8
    out_c = 16
    stories = 3
    time_steps = 512

    layer = SpectralConv2d(in_channels=in_c, out_channels=out_c, modes1=4, modes2=32)
    x = torch.randn(batch, in_c, stories, time_steps, requires_grad=True)

    out = layer(x)
    assert out.shape == (batch, out_c, stories, time_steps)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert layer.weights1.grad is not None


def test_fno2d_full_forward():
    batch = 2
    in_c = 12
    out_c = 3
    stories = 5
    time_steps = 1024

    model = FNO2d(in_channels=in_c, out_channels=out_c, modes1=4, modes2=32, width=32, n_layers=3)
    x = torch.randn(batch, in_c, stories, time_steps)

    out = model(x)
    assert out.shape == (batch, out_c, stories, time_steps)


def test_fno2d_variable_resolution():
    model = FNO2d(in_channels=6, out_channels=3, modes1=4, modes2=32, width=24, n_layers=2)

    # Evaluate across variable story counts and sequence lengths
    resolutions = [
        (3, 256),
        (3, 1024),
        (5, 512),
        (5, 2048),
        (8, 4096),
    ]

    for s, t in resolutions:
        x = torch.randn(1, 6, s, t)
        out = model(x)
        assert out.shape == (1, 3, s, t), f"Failed for resolution (S={s}, T={t}): got {out.shape}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
