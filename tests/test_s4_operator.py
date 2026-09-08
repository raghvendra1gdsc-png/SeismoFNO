"""
test_s4_operator.py — Unit Tests for Continuous S4 State-Space Operator.
"""

import math
import pytest
import torch
from src.models.s4_layer import S4Layer
from src.models.s4_operator import S4Operator


def test_s4_layer_shapes_and_causality():
    """Test S4Layer output shape and causal convolution behavior."""
    layer = S4Layer(d_model=32, d_state=16)
    x = torch.randn(2, 32, 256)
    y = layer(x)
    assert y.shape == (2, 32, 256), f"Expected (2, 32, 256), got {y.shape}"

    # Causality test: perturb future at t >= 128
    x_pert = x.clone()
    x_pert[:, :, 128:] += 5.0
    y_pert = layer(x_pert)

    # Discrepancy before t = 128 must be negligible (< 1e-5)
    past_diff = torch.max(torch.abs(y[:, :, :128] - y_pert[:, :, :128])).item()
    assert past_diff < 1e-5, f"S4Layer future leakage detected: max past diff = {past_diff}"


def test_s4_operator_shapes_and_latent():
    """Test S4Operator forward pass and latent feature extraction."""
    model = S4Operator(in_channels=10, out_channels=3, d_model=128, d_state=64, n_blocks=6, d_ff=578)
    x = torch.randn(2, 10, 512)
    y, h = model(x, return_latent=True)

    assert y.shape == (2, 3, 512), f"Expected y shape (2, 3, 512), got {y.shape}"
    assert h.shape == (2, 128, 512), f"Expected latent h shape (2, 128, 512), got {h.shape}"


def test_s4_operator_parameter_count():
    """Verify S4Operator matches target parameter budget within +-1.5%."""
    target_params = 1192448
    model = S4Operator(in_channels=10, out_channels=3, d_model=128, d_state=64, n_blocks=6, d_ff=578)
    params = model.get_num_parameters()
    delta_pct = abs(params - target_params) / target_params * 100.0

    assert delta_pct < 1.5, f"S4Operator params {params} deviates by {delta_pct:.2f}% from target {target_params}"


def test_s4_operator_strict_causality():
    """Verify S4Operator output at t < t0 is strictly invariant to future perturbations at t >= t0."""
    model = S4Operator(in_channels=10, out_channels=3, d_model=128, d_state=64, n_blocks=6, d_ff=578)
    model.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 10, 512)

    with torch.no_grad():
        y_clean = model(x)

    # Inject massive perturbation in future half (t >= 256)
    x_perturbed = x.clone()
    x_perturbed[:, :, 256:] += torch.randn(1, 10, 256) * 10.0

    with torch.no_grad():
        y_perturbed = model(x_perturbed)

    past_diff = torch.max(torch.abs(y_clean[:, :, :256] - y_perturbed[:, :, :256])).item()
    assert past_diff < 1e-5, f"S4Operator violated causality: pre-t0 max diff = {past_diff}"


def test_s4_state_reset_independence():
    """Verify processing [A, B] gives identical output for B as processing B alone (no state bleeding)."""
    model = S4Operator(in_channels=10, out_channels=3, d_model=128, d_state=64, n_blocks=6, d_ff=578)
    model.eval()

    torch.manual_seed(42)
    sample_a = torch.randn(1, 10, 512)
    sample_b = torch.randn(1, 10, 512)
    batch_ab = torch.cat([sample_a, sample_b], dim=0)

    with torch.no_grad():
        y_ab = model(batch_ab)
        y_b_alone = model(sample_b)

    diff = torch.max(torch.abs(y_ab[1:2] - y_b_alone)).item()
    assert diff < 1e-5, f"State bleeding detected between batch samples: diff = {diff}"


def test_s4_memory_truncation():
    """Verify memory-truncated S4 dampens continuous state transition rapidly."""
    layer_full = S4Layer(d_model=32, d_state=16, memory_truncated=False)
    layer_trunc = S4Layer(d_model=32, d_state=16, memory_truncated=True)

    k_full = layer_full.compute_causal_kernel(256)
    k_trunc = layer_trunc.compute_causal_kernel(256)

    # In truncated mode, tail of kernel (steps 20..255) must decay to near zero
    tail_energy_full = torch.norm(k_full[:, 20:]).item()
    tail_energy_trunc = torch.norm(k_trunc[:, 20:]).item()

    assert tail_energy_trunc < 0.01 * max(1e-6, tail_energy_full), (
        f"Memory truncation failed: tail energy full={tail_energy_full}, trunc={tail_energy_trunc}"
    )
