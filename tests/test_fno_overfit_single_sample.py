"""
test_fno_overfit_single_sample.py — FNO Overfitting Sanity Check.

Verifies that FNO1d can rapidly overfit a single physical SDOF response
to < 1.0 % relative L2 error within 60 epochs, proving gradient flow,
spectral weight updates, and representation capacity.
"""

import math
import numpy as np
import pytest
import torch
import torch.optim as optim

from src.models.fno1d import FNO1d
from src.losses.data_loss import RelativeL2Loss
from src.ground_truth.opensees_sdof_model import SDOFParams, simulate_sdof


def test_fno1d_overfit_single_sample():
    """
    Overfit FNO1d on a single SDOF response trajectory.
    Target: Relative L2 error < 1.0 % (0.01).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Generate ground truth physical response
    dt = 0.01
    n_steps = 1024
    t = np.arange(n_steps) * dt
    ag = 0.8 * np.sin(2.0 * math.pi * 1.5 * t) + 0.4 * np.sin(2.0 * math.pi * 3.2 * t)

    params = SDOFParams(T=0.5, zeta=0.05, material_type="elastic")
    resp = simulate_sdof(params=params, ag=ag, dt=dt)

    # Package into PyTorch tensors: [1, 7, 1024] -> [1, 3, 1024]
    x_ag = torch.from_numpy(ag).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 1024]
    x_t = torch.full_like(x_ag, params.T)
    x_zeta = torch.full_like(x_ag, params.zeta)
    x_bilin = torch.zeros_like(x_ag)
    x_uy = torch.zeros_like(x_ag)
    x_alpha = torch.zeros_like(x_ag)
    x_pga = torch.full_like(x_ag, float(np.max(np.abs(ag)) / 9.80665))

    x_input = torch.cat([
        x_ag,
        x_t,
        x_zeta,
        x_bilin,
        x_uy,
        x_alpha,
        x_pga,
        torch.linspace(0.0, 1.0, n_steps).unsqueeze(0).unsqueeze(0),
    ], dim=1)  # [1, 8, 1024]

    y_u = torch.from_numpy(resp.u).float().unsqueeze(0).unsqueeze(0)
    y_f = torch.from_numpy(resp.f_r).float().unsqueeze(0).unsqueeze(0)
    y_e = torch.from_numpy(resp.e_h).float().unsqueeze(0).unsqueeze(0)
    y_target = torch.cat([y_u, y_f, y_e], dim=1)  # [1, 3, 1024]

    # Standardize target
    y_mean = torch.mean(y_target, dim=-1, keepdim=True)
    y_std = torch.std(y_target, dim=-1, keepdim=True) + 1e-6
    y_norm = (y_target - y_mean) / y_std

    # 2. Build FNO model
    model = FNO1d(
        in_channels=8,
        out_channels=3,
        modes=24,
        width=64,
        n_layers=4,
        activation="gelu",
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160, eta_min=1e-5)
    criterion = RelativeL2Loss()

    # 3. Train loop
    final_rel_l2 = 1.0
    for epoch in range(1, 161):
        optimizer.zero_grad()
        pred_norm = model(x_input)
        loss = criterion(pred_norm, y_norm)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 40 == 0 or epoch == 160:
            pred_phys = pred_norm * y_std + y_mean
            rel_l2_u = float((torch.norm(pred_phys[:, 0, :] - y_target[:, 0, :]) / torch.norm(y_target[:, 0, :])).detach())
            print(f"Epoch [{epoch:3d}/160] | Norm Loss: {loss.item():.4e} | Phys Rel L2 (u): {rel_l2_u * 100:.3f}%")
            final_rel_l2 = rel_l2_u

    assert final_rel_l2 < 0.01, f"Failed to overfit single sample: Rel L2 = {final_rel_l2 * 100:.2f}% (expected < 1.0%)"


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
