"""
test_physics_losses.py — Unit Tests for Physics-Informed and Consistency Losses.
"""

import pytest
import torch
import numpy as np

from src.losses.energy_consistency_loss import (
    EnergyConsistencyLoss,
    compute_cumulative_trapezoidal_work,
)
from src.losses.boundary_loss import BoundaryLoss
from src.losses.data_loss import (
    RelativeL2Loss,
    MultiChannelRelativeL2Loss,
    CompositePhysicsLoss,
)


def test_cumulative_trapezoidal_work_analytical_sine():
    """
    Test discrete trapezoidal work against closed-form elastic strain energy:
        u(t) = sin(omega * t),  F_R(t) = k * u(t)
        W(t) = int_0^t k * u du = 0.5 * k * u(t)^2
    """
    t = torch.linspace(0.0, 2.0 * np.pi, 2048).unsqueeze(0)  # [1, 2048]
    k = 50.0
    u = torch.sin(t)
    f_r = k * u

    work = compute_cumulative_trapezoidal_work(u, f_r)  # [1, 2048]
    exact_energy = 0.5 * k * (u ** 2)

    # Initial condition check
    assert work[0, 0].item() == 0.0

    # Max relative discrepancy should be < 0.05% for 2048 points
    max_err = torch.max(torch.abs(work - exact_energy)).item()
    max_val = torch.max(exact_energy).item()
    rel_err = max_err / max_val
    assert rel_err < 1e-3, f"Trapezoidal work rel error {rel_err:.6f} exceeded 0.1%"


def test_energy_consistency_loss_zero_when_consistent():
    """Energy consistency loss should be near 0 when E_h matches work."""
    t = torch.linspace(0.0, 4.0, 1024).unsqueeze(0).repeat(4, 1)  # [4, 1024]
    u = torch.sin(2.0 * np.pi * t)
    f_r = 10.0 * u
    e_h = compute_cumulative_trapezoidal_work(u, f_r)

    pred = torch.stack([u, f_r, e_h], dim=1)  # [4, 3, 1024]

    loss_fn = EnergyConsistencyLoss()
    loss = loss_fn(pred)

    assert loss.item() < 1e-5, f"Expected near zero loss for consistent inputs, got {loss.item()}"


def test_energy_consistency_loss_gradient_flow():
    """Verify gradients propagate backwards through energy consistency loss."""
    pred = torch.randn(2, 3, 512, requires_grad=True)
    loss_fn = EnergyConsistencyLoss()
    loss = loss_fn(pred)
    loss.backward()

    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))
    assert not torch.all(pred.grad == 0.0)


def test_boundary_loss_zero_at_origin():
    """Boundary loss should be 0 when initial states are 0."""
    pred = torch.zeros(4, 3, 256)
    pred[:, :, 1:] = torch.randn(4, 3, 255)  # non-zero later in time

    loss_fn = BoundaryLoss()
    loss = loss_fn(pred)
    assert loss.item() == 0.0


def test_boundary_loss_penalizes_nonzero_origin():
    """Boundary loss should be sum of squared initial values."""
    pred = torch.zeros(2, 3, 100)
    pred[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])  # sum = 1 + 4 + 9 = 14
    pred[1, :, 0] = torch.tensor([0.0, 1.0, 2.0])  # sum = 0 + 1 + 4 = 5

    loss_fn = BoundaryLoss(reduction="mean")
    loss = loss_fn(pred)
    expected = (14.0 + 5.0) / 2.0  # 9.5
    assert abs(loss.item() - expected) < 1e-6


def test_composite_physics_loss_ablation_combinations():
    """Verify CompositePhysicsLoss toggles terms appropriately."""
    pred = torch.randn(2, 3, 256, requires_grad=True)
    target = torch.randn(2, 3, 256)

    # (a) Data only
    loss_a = CompositePhysicsLoss(lambda_energy=0.0, lambda_boundary=0.0)
    res_a = loss_a(pred, target)
    assert res_a["energy_loss"].item() == 0.0
    assert res_a["boundary_loss"].item() == 0.0
    assert abs(res_a["loss"].item() - res_a["data_loss"].item()) < 1e-6

    # (b) Data + Energy
    loss_b = CompositePhysicsLoss(lambda_energy=0.5, lambda_boundary=0.0)
    res_b = loss_b(pred, target)
    assert res_b["energy_loss"].item() > 0.0
    assert res_b["boundary_loss"].item() == 0.0

    # (c) Data + Energy + Boundary
    loss_c = CompositePhysicsLoss(lambda_energy=0.5, lambda_boundary=0.1)
    res_c = loss_c(pred, target)
    assert res_c["energy_loss"].item() > 0.0
    assert res_c["boundary_loss"].item() > 0.0
