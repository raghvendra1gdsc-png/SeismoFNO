"""
build_exp3r_cache.py — Authoritative Cache Builder for EXP 3-R.

Builds train and validation caches for the pure recurrent state-space model:
  Inputs:  x in R^{N x 5 x 2048} [a_g, T, zeta, u_y, alpha]
  Targets: y in R^{N x 3 x 2048} [u, F_R, E_diss]
  States:  s in R^{N x 3 x 2048} [u, v, u_p]

Normalizers:
  Fitted STRICTLY on train partition only.
  Test partition is NEVER loaded or accessed.
"""

from pathlib import Path
import torch
import numpy as np

from src.data_pipeline.dataset_builder import UnitGaussianNormalizer


def build_exp3r_cache():
    in_dir = Path("data/processed/exp3_cache")
    out_dir = Path("data/processed/exp3r_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== BUILDING EXP 3-R DATASET CACHE ===")

    # 1. Process Train Cache
    print("Loading raw train data from exp3 cache...")
    tc = torch.load(in_dir / "train_cache.pt", map_location="cpu")
    x_train_raw = tc["x"]  # [5740, 10, 2048]
    y_train_raw = tc["y"]  # [5740, 3, 2048]

    # Select 5 channels: [a_g, T, zeta, u_y, alpha] -> indices [0, 1, 4, 6, 7]
    x_train = x_train_raw[:, [0, 1, 4, 6, 7], :].clone()

    u_tr = y_train_raw[:, 0, :]
    f_r_tr = y_train_raw[:, 1, :]
    T_tr = x_train_raw[:, 1, 0:1]
    zeta_tr = x_train_raw[:, 4, 0:1]
    u_y_tr = x_train_raw[:, 6, 0:1]
    alpha_tr = x_train_raw[:, 7, 0:1]
    k0_tr = (2.0 * np.pi / T_tr) ** 2
    fy_tr = k0_tr * u_y_tr

    # Exact physical state
    u_p_tr = (u_tr - f_r_tr / k0_tr) / (1.0 - alpha_tr)
    # Zero out numerical noise below 1e-6 for elastic cases
    u_p_tr = torch.where(torch.abs(u_p_tr) < 1e-7, torch.zeros_like(u_p_tr), u_p_tr)

    # Irreversible hysteretic dissipation: E_diss = (1 - alpha) * fy * integral |du_p|
    du_p_tr = torch.diff(u_p_tr, dim=-1)
    e_diss_tr = torch.zeros_like(u_tr)
    e_diss_tr[:, 1:] = torch.cumsum((1.0 - alpha_tr) * fy_tr * torch.abs(du_p_tr), dim=-1)

    # Velocity v: numerical gradient with dt = 0.01
    v_tr = torch.gradient(u_tr, spacing=0.01, dim=-1)[0]

    y_train = torch.stack([u_tr, f_r_tr, e_diss_tr], dim=1)
    s_train = torch.stack([u_tr, v_tr, u_p_tr], dim=1)

    print(f"Train processed: x {x_train.shape}, y {y_train.shape}, s {s_train.shape}")

    # 2. Fit Normalizers on Train Data ONLY
    print("Fitting UnitGaussianNormalizers strictly on Train partition...")
    x_norm = UnitGaussianNormalizer().fit(x_train)
    y_norm = UnitGaussianNormalizer().fit(y_train)
    s_norm = UnitGaussianNormalizer().fit(s_train)

    # 3. Process Validation Cache
    print("Loading raw validation data from exp3 cache...")
    vc = torch.load(in_dir / "val_cache.pt", map_location="cpu")
    x_val_raw = vc["x"]  # [1120, 10, 2048]
    y_val_raw = vc["y"]  # [1120, 3, 2048]

    x_val = x_val_raw[:, [0, 1, 4, 6, 7], :].clone()

    u_val = y_val_raw[:, 0, :]
    f_r_val = y_val_raw[:, 1, :]
    T_val = x_val_raw[:, 1, 0:1]
    zeta_val = x_val_raw[:, 4, 0:1]
    u_y_val = x_val_raw[:, 6, 0:1]
    alpha_val = x_val_raw[:, 7, 0:1]
    k0_val = (2.0 * np.pi / T_val) ** 2
    fy_val = k0_val * u_y_val

    u_p_val = (u_val - f_r_val / k0_val) / (1.0 - alpha_val)
    u_p_val = torch.where(torch.abs(u_p_val) < 1e-7, torch.zeros_like(u_p_val), u_p_val)

    du_p_val = torch.diff(u_p_val, dim=-1)
    e_diss_val = torch.zeros_like(u_val)
    e_diss_val[:, 1:] = torch.cumsum((1.0 - alpha_val) * fy_val * torch.abs(du_p_val), dim=-1)

    v_val = torch.gradient(u_val, spacing=0.01, dim=-1)[0]

    y_val = torch.stack([u_val, f_r_val, e_diss_val], dim=1)
    s_val = torch.stack([u_val, v_val, u_p_val], dim=1)

    print(f"Val processed:   x {x_val.shape}, y {y_val.shape}, s {s_val.shape}")

    # 4. Save Caches
    print("Saving processed exp3r caches...")
    torch.save({"x": x_train, "y": y_train, "s": s_train}, out_dir / "train_cache.pt")
    torch.save({"x": x_val, "y": y_val, "s": s_val}, out_dir / "val_cache.pt")
    torch.save({"x_norm": x_norm, "y_norm": y_norm, "s_norm": s_norm}, out_dir / "normalizers.pt")

    print(f"Saved: {out_dir / 'train_cache.pt'}")
    print(f"Saved: {out_dir / 'val_cache.pt'}")
    print(f"Saved: {out_dir / 'normalizers.pt'}")
    print(">>> EXP 3-R DATASET CACHE BUILD COMPLETE (100% TRAIN-ONLY NORMALIZATION) <<<")


if __name__ == "__main__":
    build_exp3r_cache()
