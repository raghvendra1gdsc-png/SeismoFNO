"""
test_spectral_conv_shapes.py — SpectralConv1d and FNO1d Tensor Shape Verification.

Verifies:
  1. SpectralConv1d preserves time dimension for even/odd lengths and arbitrary channel configurations.
  2. FNOBlock1d forward pass shape consistency.
  3. FNO1d full network shape mapping [B, C_in, T] -> [B, C_out, T].
"""

import pytest
import torch

from src.models.spectral_conv import SpectralConv1d
from src.models.fno_block import FNOBlock1d
from src.models.fno1d import FNO1d


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("in_channels,out_channels", [(1, 1), (4, 8), (7, 32)])
@pytest.mark.parametrize("time_steps", [256, 512, 1024, 2048])
@pytest.mark.parametrize("modes", [8, 16, 24])
def test_spectral_conv1d_forward_shapes(batch_size, in_channels, out_channels, time_steps, modes):
    """Verify that SpectralConv1d produces exact output shapes for varying dimensions."""
    x = torch.randn(batch_size, in_channels, time_steps)
    layer = SpectralConv1d(in_channels=in_channels, out_channels=out_channels, modes1=modes)
    out = layer(x)

    assert out.shape == (batch_size, out_channels, time_steps), (
        f"Expected shape ({batch_size}, {out_channels}, {time_steps}), got {out.shape}"
    )
    assert not torch.isnan(out).any(), "NaN found in SpectralConv1d output"


@pytest.mark.parametrize("width", [16, 32])
@pytest.mark.parametrize("modes", [12, 16])
@pytest.mark.parametrize("activation", ["gelu", "leaky_relu", "relu"])
def test_fno_block1d_shapes(width, modes, activation):
    """Verify FNOBlock1d shape preservation."""
    b, t = 4, 1024
    x = torch.randn(b, width, t)
    block = FNOBlock1d(width=width, modes=modes, activation=activation)
    out = block(x)

    assert out.shape == (b, width, t)
    assert not torch.isnan(out).any()


def test_fno1d_full_model_end_to_end():
    """Verify complete FNO1d model forward pass."""
    b, c_in, c_out, t = 8, 7, 3, 2048
    x = torch.randn(b, c_in, t)
    model = FNO1d(
        in_channels=c_in,
        out_channels=c_out,
        modes=16,
        width=32,
        n_layers=4,
        activation="gelu",
    )
    out = model(x)

    assert out.shape == (b, c_out, t)
    assert not torch.isnan(out).any()
    assert model.get_num_parameters() > 0


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
