"""
test_state_augmented_tcn.py — Unit Tests for State-Augmented Causal TCN.
"""

import pytest
import torch
from src.models.recurrent_state_cell import CausalRecurrentStateCell
from src.models.state_augmented_tcn import StateAugmentedCausalTCN


def test_recurrent_state_cell_causality():
    """Verify CausalRecurrentStateCell output at t < t0 is invariant to future input perturbations."""
    cell = CausalRecurrentStateCell(in_channels=10, state_dim=4)
    x = torch.randn(2, 10, 256)
    s = cell(x)
    assert s.shape == (2, 4, 256), f"Expected (2, 4, 256), got {s.shape}"

    # Perturb future at t >= 128
    x_pert = x.clone()
    x_pert[:, :, 128:] += 10.0
    s_pert = cell(x_pert)

    past_diff = torch.max(torch.abs(s[:, :, :128] - s_pert[:, :, :128])).item()
    assert past_diff < 1e-6, f"Recurrent state cell leaked future inputs: past diff = {past_diff}"


def test_state_augmented_tcn_shapes_and_latent():
    """Test forward pass shapes and latent extraction."""
    model = StateAugmentedCausalTCN(in_channels=10, out_channels=3, num_channels=[136]*11, state_dim=4)
    x = torch.randn(2, 10, 512)
    y, h = model(x, return_latent=True)

    assert y.shape == (2, 3, 512), f"Expected y (2, 3, 512), got {y.shape}"
    assert h.shape == (2, 136, 512), f"Expected latent h (2, 136, 512), got {h.shape}"


def test_state_augmented_tcn_parameter_count():
    """Verify State-Augmented TCN parameter matching (+-1.5%)."""
    target_params = 1192448
    model = StateAugmentedCausalTCN(in_channels=10, out_channels=3, num_channels=[136]*11, state_dim=4)
    params = model.get_num_parameters()
    delta_pct = abs(params - target_params) / target_params * 100.0

    assert delta_pct < 1.5, f"State-Augmented TCN params {params} deviates by {delta_pct:.2f}% from target {target_params}"


def test_state_augmented_tcn_strict_causality():
    """Verify strict causality under future perturbation intervention."""
    model = StateAugmentedCausalTCN(in_channels=10, out_channels=3, num_channels=[136]*11, state_dim=4)
    model.eval()

    torch.manual_seed(42)
    x = torch.randn(1, 10, 512)

    with torch.no_grad():
        y_clean = model(x)

    # Perturb future at t >= 256
    x_perturbed = x.clone()
    x_perturbed[:, :, 256:] += torch.randn(1, 10, 256) * 10.0

    with torch.no_grad():
        y_perturbed = model(x_perturbed)

    past_diff = torch.max(torch.abs(y_clean[:, :, :256] - y_perturbed[:, :, :256])).item()
    assert past_diff < 1e-5, f"State-Augmented TCN violated causality: pre-t0 max diff = {past_diff}"


def test_state_augmented_tcn_state_reset():
    """Verify sample independence (no state bleeding between batch rows)."""
    model = StateAugmentedCausalTCN(in_channels=10, out_channels=3, num_channels=[136]*11, state_dim=4)
    model.eval()

    torch.manual_seed(42)
    sample_a = torch.randn(1, 10, 512)
    sample_b = torch.randn(1, 10, 512)
    batch_ab = torch.cat([sample_a, sample_b], dim=0)

    with torch.no_grad():
        y_ab = model(batch_ab)
        y_b_alone = model(sample_b)

    diff = torch.max(torch.abs(y_ab[1:2] - y_b_alone)).item()
    assert diff < 1e-5, f"State bleeding detected in State-Augmented TCN: diff = {diff}"
