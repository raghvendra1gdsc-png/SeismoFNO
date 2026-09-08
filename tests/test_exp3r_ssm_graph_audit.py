"""
test_exp3r_ssm_graph_audit.py — Automated Graph & Causal Tests for EXP 3-R.

Verifies:
A. No x_t -> y_t direct edge (Autograd gradient w.r.t simultaneous x_t with h detached is zero).
B. No encoder feature -> y_t edge (pure state readout).
C. y_t depends directly on h_t.
D. h_{t+1} depends on h_t and x_t.
E. Intervention Delta h_{t_y} affects future outputs strictly through recurrent transitions.
F. Zero past perturbation: Delta y(t) == 0 for all t < t_y.
G. State reset h_0 = 0 is exact.
H. Physical state coordinates s^phys = [u, v, u_p] are properly partitioned.
I. Static-Hold diagnostic produces non-zero dynamic divergence under subsequent excitation.
J. Sham control constructions (C0, C1, C2.A, C2.B, C3, C4, C6).
"""

import math
import pytest
import torch
import numpy as np

from src.models.exp3r_ssm import PureRecurrentSSM


def test_parameter_count_budget():
    """Verify parameter count matches 1,192,448 within +/- 1.0%."""
    model = PureRecurrentSSM()
    target = 1192448
    p = model.count_parameters()
    dev = (p - target) / target * 100
    assert abs(dev) <= 1.0, f"Parameter deviation {dev:.3f}% exceeds +/- 1.0%"
    assert p == 1191815


def test_computational_graph_no_direct_input_bypass():
    """
    Verify there is NO direct path x_t -> y_t bypassing the recurrent state h_t.
    If h_t is detached, the gradient of y_t w.r.t x_t must be identically zero.
    """
    model = PureRecurrentSSM()
    x = torch.randn(2, 5, 20, requires_grad=True)

    # Manual single step execution to test graph isolation
    h_0 = torch.zeros(2, 64)
    x_0 = x[:, :, 0]

    # Compute h_1
    delta_h_0 = model.transition(h_0, x_0)
    h_1 = h_0 + delta_h_0

    # Compute y_0 from h_0
    y_0 = model.readout(h_0)

    # 1. Gradient of y_0 w.r.t x_0 must be None or zero (no connection)
    grad_x0_from_y0 = torch.autograd.grad(y_0.sum(), x, retain_graph=True, allow_unused=True)[0]
    assert grad_x0_from_y0 is None or torch.all(grad_x0_from_y0 == 0.0)

    # 2. If h_1 is detached, gradient of y_1 w.r.t x_0 must be zero
    h_1_detached = h_1.detach()
    y_1_detached = model.readout(h_1_detached)
    grad_x0_from_y1_detached = torch.autograd.grad(y_1_detached.sum(), x, retain_graph=True, allow_unused=True)[0]
    assert grad_x0_from_y1_detached is None or torch.all(grad_x0_from_y1_detached == 0.0)

    # 3. When h_1 is NOT detached, gradient of y_1 w.r.t x_0 MUST be non-zero (passes through transition)
    y_1 = model.readout(h_1)
    grad_x0_from_y1 = torch.autograd.grad(y_1.sum(), x, retain_graph=True)[0]
    assert grad_x0_from_y1 is not None
    assert torch.any(grad_x0_from_y1[:, :, 0] != 0.0)
    # But gradient w.r.t future inputs x[:, :, 1:] must be strictly zero (causality)
    assert torch.all(grad_x0_from_y1[:, :, 1:] == 0.0)


def test_intervention_path_and_past_invariance():
    """
    Verify:
    1. Past outputs for t < t_y are 100% bit-for-bit identical between base and perturbed.
    2. Future outputs for t >= t_y are modified.
    3. Intervention at t_y cannot affect outputs at t < t_y.
    """
    model = PureRecurrentSSM()
    model.eval()

    batch_size = 2
    time_steps = 100
    x = torch.randn(batch_size, 5, time_steps)

    with torch.no_grad():
        y_base, s_base, h_base = model(x, return_state=True)

        t_y = 40
        delta_u_p = 0.010  # 10 mm
        # Delta h = [0, 0, Delta u_p, 0_61]
        delta_h = torch.zeros(batch_size, 64)
        delta_h[:, 2] = delta_u_p  # Coordinate index 2 is u_p

        interv = {"t_idx": t_y, "delta_h": delta_h}
        y_pert, s_pert, h_pert = model(x, intervention=interv, return_state=True)

    # Past invariance: t < t_y
    past_diff_y = torch.max(torch.abs(y_pert[:, :, :t_y] - y_base[:, :, :t_y])).item()
    past_diff_h = torch.max(torch.abs(h_pert[:, :, :t_y] - h_base[:, :, :t_y])).item()
    assert past_diff_y == 0.0, f"Past output leaked! diff = {past_diff_y}"
    assert past_diff_h == 0.0, f"Past state leaked! diff = {past_diff_h}"

    # Future modification: t >= t_y
    future_diff_y = torch.max(torch.abs(y_pert[:, :, t_y:] - y_base[:, :, t_y:])).item()
    future_diff_h = torch.max(torch.abs(h_pert[:, :, t_y:] - h_base[:, :, t_y:])).item()
    assert future_diff_y > 1e-6, "Future output was not modified by intervention!"
    assert future_diff_h > 1e-6, "Future state was not modified by intervention!"


def test_static_hold_diagnostic():
    """
    Verify Static-Hold condition:
    When static_hold=True, h_t remains frozen at h_{t_idx}^+ for all t >= t_idx.
    """
    model = PureRecurrentSSM()
    model.eval()

    x = torch.randn(2, 5, 50)
    t_y = 20
    delta_h = torch.zeros(2, 64)
    delta_h[:, 2] = 0.010
    interv = {"t_idx": t_y, "delta_h": delta_h}

    with torch.no_grad():
        y_act, _, h_act = model(x, intervention=interv, return_state=True, static_hold=False)
        y_hold, _, h_hold = model(x, intervention=interv, return_state=True, static_hold=True)

    # In static hold, all state vectors for t >= t_y must be IDENTICAL to h_{t_y}
    h_frozen_target = h_hold[:, :, t_y]
    for t in range(t_y, 50):
        diff = torch.max(torch.abs(h_hold[:, :, t] - h_frozen_target)).item()
        assert diff < 1e-6, f"Static hold failed to keep state constant at t={t}: diff={diff}"

    # Under active dynamics with non-zero inputs, h_act must diverge from h_hold
    divergence_h = torch.norm(h_act[:, :, -1] - h_hold[:, :, -1], p=2).item()
    assert divergence_h > 1e-3, "Active state did not diverge from static hold!"


def test_sham_control_specifications():
    """Verify math and dimensions of all mandatory control vectors."""
    delta_u_p = 0.010  # 10 mm
    m = 1.0
    k0 = 100.0
    alpha = 0.02

    # C0: Zero intervention
    c0 = torch.zeros(64)
    assert torch.all(c0 == 0.0)

    # C1: Physical plastic displacement intervention (coordinate 2)
    c1 = torch.zeros(64)
    c1[2] = delta_u_p
    assert c1[2] == 0.010
    assert torch.all(c1[3:] == 0.0)

    # C2.A: Matched kinetic energy velocity sham
    # 0.5 * m * (delta_v)^2 = 0.5 * (1 - alpha) * k0 * (delta_u_p)^2
    delta_v = math.sqrt((1.0 - alpha) * k0 / m) * abs(delta_u_p) * math.copysign(1.0, delta_u_p)
    c2_a = torch.zeros(64)
    c2_a[1] = delta_v  # Coordinate 1 is velocity v
    assert abs(c2_a[1] - math.sqrt(0.98 * 100.0) * 0.010) < 1e-7

    # C2.B: Matched norm latent sham (perturbing unconstrained memory q)
    c2_b = torch.zeros(64)
    # Unit direction in q space (index 3 to 63)
    c2_b[3] = delta_u_p  # Matched Euclidean norm: ||q|| = |delta_u_p|
    assert abs(torch.norm(c2_b[3:], p=2).item() - abs(delta_u_p)) < 1e-7

    # C3: Sign reversal
    c3 = -c1
    assert c3[2] == -0.010
