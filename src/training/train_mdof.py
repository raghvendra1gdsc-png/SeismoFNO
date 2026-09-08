"""
train_mdof.py — Spatiotemporal MDOF Surrogate Training Engine.

Trains and evaluates:
  1. FNO2d (2D Fourier Neural Operator over story-time domain)
  2. MDOFLSTMBaseline (Recurrent sequence model)
  3. MDOFMLPBaseline (Pointwise residual MLP)
using leakage-free data partitions (held-out-earthquake, held-out-structure)
and composite physics-informed losses.
"""

from typing import Dict, Any, Optional, Tuple
import os
import time
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_pipeline.splits import load_split
from src.data_pipeline.mdof_dataset import SeismicMDOFDataset, UnitGaussianNormalizer2D
from src.models.fno2d import FNO2d
from src.models.mdof_baselines import MDOFLSTMBaseline, MDOFMLPBaseline
from src.losses.mdof_losses import MDOFCompositeLoss, MDOFDataLoss


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_mdof_model(cfg: Dict[str, Any]) -> nn.Module:
    m_cfg = cfg["model"]
    m_type = m_cfg.get("model_type", "fno2d").lower()

    if m_type == "fno2d":
        return FNO2d(
            in_channels=m_cfg.get("in_channels", 10),
            out_channels=m_cfg.get("out_channels", 3),
            modes1=m_cfg.get("modes1", 4),
            modes2=m_cfg.get("modes2", 64),
            width=m_cfg.get("width", 48),
            n_layers=m_cfg.get("n_layers", 4),
            activation=m_cfg.get("activation", "gelu"),
        )
    elif m_type == "lstm":
        return MDOFLSTMBaseline(
            in_channels=m_cfg.get("in_channels", 10),
            out_channels=m_cfg.get("out_channels", 3),
            max_stories=m_cfg.get("max_stories", 5),
            hidden_dim=m_cfg.get("hidden_dim", 128),
            num_layers=m_cfg.get("num_layers", 3),
            dropout=m_cfg.get("dropout", 0.1),
        )
    elif m_type == "mlp":
        return MDOFMLPBaseline(
            in_channels=m_cfg.get("in_channels", 10),
            out_channels=m_cfg.get("out_channels", 3),
            hidden_dim=m_cfg.get("hidden_dim", 128),
            num_layers=m_cfg.get("num_layers", 4),
            dropout=m_cfg.get("dropout", 0.05),
        )
    else:
        raise ValueError(f"Unknown MDOF model_type: {m_type}")


def train_mdof(config: Dict[str, Any]) -> Dict[str, Any]:
    exp_name = config.get("experiment_name", "mdof_experiment")
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[{exp_name}] Using device: {device}")

    # Output directories
    out_cfg = config.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", f"experiments/{exp_name}/checkpoints"))
    res_dir = Path(out_cfg.get("results_dir", f"experiments/{exp_name}/results"))
    log_dir = Path(out_cfg.get("log_dir", f"experiments/{exp_name}/logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    d_cfg = config["data"]
    df = pd.read_csv(d_cfg["index_csv"])

    # Filter material type if requested
    if "filter_material" in d_cfg and d_cfg["filter_material"]:
        mat_filter = d_cfg["filter_material"]
        df = df[df["material_type"] == mat_filter].reset_index(drop=True)
        print(f"Filtered MDOF dataset for material_type='{mat_filter}': {len(df)} samples remaining.")

    split_file = d_cfg["split_file"]
    split = load_split(split_file)

    train_df = df[df["sim_id"].isin(set(split.train_ids))].reset_index(drop=True)
    val_df = df[df["sim_id"].isin(set(split.val_ids))].reset_index(drop=True)
    test_df = df[df["sim_id"].isin(set(split.test_ids))].reset_index(drop=True)

    print(f"MDOF Data partition: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    target_time_steps = d_cfg.get("target_time_steps", 2048)
    batch_size = d_cfg.get("batch_size", 32)
    normalize = d_cfg.get("normalize", True)

    # Compute normalizers on training set
    train_ds_raw = SeismicMDOFDataset(train_df, target_time_steps=target_time_steps)
    x_norm, y_norm = None, None
    if normalize and len(train_ds_raw) > 0:
        sample_n = min(len(train_ds_raw), 256)
        x_s, y_s = [], []
        for i in range(sample_n):
            xs, ys = train_ds_raw[i]
            x_s.append(xs)
            y_s.append(ys)
        x_norm = UnitGaussianNormalizer2D().fit(torch.stack(x_s, dim=0))
        y_norm = UnitGaussianNormalizer2D().fit(torch.stack(y_s, dim=0))

    train_ds = SeismicMDOFDataset(train_df, target_time_steps=target_time_steps, x_normalizer=x_norm, y_normalizer=y_norm)
    val_ds = SeismicMDOFDataset(val_df, target_time_steps=target_time_steps, x_normalizer=x_norm, y_normalizer=y_norm)
    test_ds = SeismicMDOFDataset(test_df, target_time_steps=target_time_steps, x_normalizer=x_norm, y_normalizer=y_norm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Build model
    model = build_mdof_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model constructed with {n_params:,} trainable parameters.")

    # Setup loss
    l_cfg = config.get("loss", {})
    loss_fn = MDOFCompositeLoss(
        data_weights=l_cfg.get("data_weights", [1.0, 1.0, 1.0]),
        lambda_energy=l_cfg.get("lambda_energy", 0.1),
        lambda_boundary=l_cfg.get("lambda_boundary", 0.05),
    )
    val_loss_fn = MDOFDataLoss(data_weights=[1.0, 1.0, 1.0])

    # Optimizer & Scheduler
    t_cfg = config.get("training", {})
    epochs = t_cfg.get("epochs", 40)
    lr = float(t_cfg.get("learning_rate", 0.005))
    weight_decay = float(t_cfg.get("weight_decay", 1e-5))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(t_cfg.get("min_lr", 1e-5))
    )

    best_val_loss = float("inf")
    start_time = time.time()

    print(f"Starting MDOF training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_accum = 0.0
        n_train_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss, _ = loss_fn(pred, y)
            loss.backward()
            if t_cfg.get("clip_grad_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["clip_grad_norm"])
            optimizer.step()
            train_loss_accum += loss.item()
            n_train_batches += 1

        scheduler.step()
        train_loss_avg = train_loss_accum / max(1, n_train_batches)

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_rel_u_accum = 0.0
        val_rel_e_accum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)

                if y_norm is not None:
                    pred_phys = y_norm.decode(pred)
                    y_phys = y_norm.decode(y)
                else:
                    pred_phys, y_phys = pred, y

                _, metrics = val_loss_fn(pred_phys, y_phys)
                val_loss_accum += metrics["data_loss"]
                val_rel_u_accum += metrics["loss_u"]
                val_rel_e_accum += metrics["loss_e"]
                n_val_batches += 1

        val_loss_avg = val_loss_accum / max(1, n_val_batches)
        val_rel_u = val_rel_u_accum / max(1, n_val_batches)
        val_rel_e = val_rel_e_accum / max(1, n_val_batches)
        epoch_time = time.time() - t0

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Train Loss: {train_loss_avg:.4e} | "
                f"Val Loss: {val_loss_avg:.4e} | "
                f"Rel L2 u: {val_rel_u * 100:.2f}% | "
                f"Rel L2 E_h: {val_rel_e * 100:.2f}% | "
                f"Time: {epoch_time:.2f}s",
                flush=True,
            )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} s. Evaluating best checkpoint on test set...")

    # Load best checkpoint for final test evaluation
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()

    # Disaggregated evaluation: overall, elastic, post-yield
    test_metrics = evaluate_mdof_test_set(
        model=model,
        test_df=test_df,
        target_time_steps=target_time_steps,
        x_norm=x_norm,
        y_norm=y_norm,
        device=device,
    )

    results_payload = {
        "experiment_name": exp_name,
        "config": config,
        "trainable_parameters": n_params,
        "total_train_time_s": total_time,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
    }

    with open(res_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"MDOF Experiment '{exp_name}' Results")
    print("=" * 70)
    print(f"  [Overall Test Set: {len(test_df)} records]")
    print(f"    - Relative L2 Displacement u(s, t) : {test_metrics['overall_rel_l2_u']:.4e} ({test_metrics['overall_rel_l2_u']*100:.2f} %)")
    print(f"    - Relative L2 Restoring Force F_R  : {test_metrics['overall_rel_l2_f_r']:.4e} ({test_metrics['overall_rel_l2_f_r']*100:.2f} %)")
    print(f"    - Relative L2 Hysteretic E_h       : {test_metrics['overall_rel_l2_e_h']:.4e} ({test_metrics['overall_rel_l2_e_h']*100:.2f} %)")
    if "elastic_rel_l2_u" in test_metrics:
        print(f"  [Elastic Regime: {test_metrics['n_elastic']} records]")
        print(f"    - Relative L2 Displacement u(s, t) : {test_metrics['elastic_rel_l2_u']*100:.2f} %")
    if "post_yield_rel_l2_u" in test_metrics:
        print(f"  [Post-Yield Regime: {test_metrics['n_post_yield']} records]")
        print(f"    - Relative L2 Displacement u(s, t) : {test_metrics['post_yield_rel_l2_u']*100:.2f} %")
        print(f"    - Relative L2 Hysteretic E_h       : {test_metrics['post_yield_rel_l2_e_h']*100:.2f} %")
    print(f"  Total Training Time : {total_time:.2f} s")
    print("=" * 70)

    return results_payload


def evaluate_mdof_test_set(
    model: nn.Module,
    test_df: pd.DataFrame,
    target_time_steps: int,
    x_norm: Optional[UnitGaussianNormalizer2D],
    y_norm: Optional[UnitGaussianNormalizer2D],
    device: torch.device,
) -> Dict[str, Any]:
    """Compute overall and regime-disaggregated test metrics."""
    test_ds = SeismicMDOFDataset(test_df, target_time_steps=target_time_steps, x_normalizer=x_norm, y_normalizer=y_norm)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_u_err, all_f_err, all_e_err = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            if y_norm is not None:
                pred_phys = y_norm.decode(pred)
                y_phys = y_norm.decode(y)
            else:
                pred_phys, y_phys = pred, y

            b, _, max_s, t = pred_phys.shape
            for i in range(b):
                # Extract actual story count S from input tensor channel 5
                s_actual = int(x[i, 5, 0, 0].item()) if x[i, 5, 0, 0].item() > 0 else max_s
                u_p = pred_phys[i, 0, :s_actual, :]
                u_t = y_phys[i, 0, :s_actual, :]
                f_p = pred_phys[i, 1, :s_actual, :]
                f_t = y_phys[i, 1, :s_actual, :]
                e_p = pred_phys[i, 2, :s_actual, :]
                e_t = y_phys[i, 2, :s_actual, :]

                err_u = (torch.norm(u_p - u_t) / (torch.norm(u_t) + 1e-6)).item()
                err_f = (torch.norm(f_p - f_t) / (torch.norm(f_t) + 1e-6)).item()
                err_e = (torch.norm(e_p - e_t) / (torch.norm(e_t) + 1e-6)).item()

                all_u_err.append(err_u)
                all_f_err.append(err_f)
                all_e_err.append(err_e)

    metrics = {
        "overall_rel_l2_u": float(np.mean(all_u_err)),
        "overall_rel_l2_f_r": float(np.mean(all_f_err)),
        "overall_rel_l2_e_h": float(np.mean(all_e_err)),
    }

    # Disaggregate by elastic vs post-yield if material info present
    elastic_mask = (test_df["material_type"] == "elastic").values
    post_yield_mask = (test_df["material_type"] == "bilinear").values

    if np.any(elastic_mask):
        metrics["n_elastic"] = int(np.sum(elastic_mask))
        metrics["elastic_rel_l2_u"] = float(np.mean([all_u_err[i] for i in range(len(all_u_err)) if elastic_mask[i]]))
        metrics["elastic_rel_l2_e_h"] = float(np.mean([all_e_err[i] for i in range(len(all_e_err)) if elastic_mask[i]]))

    if np.any(post_yield_mask):
        metrics["n_post_yield"] = int(np.sum(post_yield_mask))
        metrics["post_yield_rel_l2_u"] = float(np.mean([all_u_err[i] for i in range(len(all_u_err)) if post_yield_mask[i]]))
        metrics["post_yield_rel_l2_e_h"] = float(np.mean([all_e_err[i] for i in range(len(all_e_err)) if post_yield_mask[i]]))

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_mdof(cfg)
