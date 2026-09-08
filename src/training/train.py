"""
train.py — Training & Evaluation Script for SeismoFNO Models.

Supports:
  - Phase 4: FNO MVP on linear-elastic SDOF dataset
  - Phase 5: Bilinear-hysteretic SDOF core model with physics losses (energy consistency, boundary)
             and history-augmented conditioning across held-out earthquake splits
  - Phase 8: MDOF multi-story extensions

Logs every run with exact configs, hyperparameters, and measured metrics
disaggregated by elastic vs post-yield regime per AGENTS.md Hard Rules 2, 3, & 4.
"""

from typing import Dict, Any, Optional
import argparse
from pathlib import Path
import json
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.fno1d import FNO1d
from src.models.lstm_baseline import LSTMBaseline
from src.models.mlp_baseline import MLPBaseline
from src.losses.data_loss import (
    RelativeL2Loss,
    MultiChannelRelativeL2Loss,
    CompositePhysicsLoss,
)
from src.data_pipeline.dataset_builder import SeismicSDOFDataset, UnitGaussianNormalizer
from src.data_pipeline.splits import (
    split_random,
    split_held_out_earthquake,
    split_held_out_structure,
    load_split,
)
from src.training.trainer import FNOTrainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML experiment configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_fno(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete FNO model training workflow."""
    exp_name = config.get("experiment_name", "fno_experiment")
    seed = config.get("data", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[{exp_name}] Using device: {device}")

    # 1. Load data index
    data_cfg = config["data"]
    index_csv = data_cfg["index_csv"]
    df = pd.read_csv(index_csv)

    # Filter material type if requested
    filter_mat = data_cfg.get("filter_material", None)
    if filter_mat:
        df = df[df["material_type"] == filter_mat].reset_index(drop=True)
        print(f"Filtered dataset for material_type='{filter_mat}': {len(df)} samples remaining.")

    # 2. Partition dataset
    split_file = data_cfg.get("split_file", None)
    split_strategy = data_cfg.get("split_strategy", "held_out_earthquake")

    if split_file and Path(split_file).exists():
        split = load_split(split_file)
        train_ids = set(split.train_ids)
        val_ids = set(split.val_ids)
        test_ids = set(split.test_ids)

        train_df = df[df["sim_id"].isin(train_ids)].reset_index(drop=True)
        val_df = df[df["sim_id"].isin(val_ids)].reset_index(drop=True)
        test_df = df[df["sim_id"].isin(test_ids)].reset_index(drop=True)
    elif split_strategy == "held_out_earthquake":
        split = split_held_out_earthquake(df, seed=seed)
        train_df = df.iloc[split.train_indices].reset_index(drop=True)
        val_df = df.iloc[split.val_indices].reset_index(drop=True)
        test_df = df.iloc[split.test_indices].reset_index(drop=True)
    elif split_strategy == "held_out_structure":
        split = split_held_out_structure(df, seed=seed)
        train_df = df.iloc[split.train_indices].reset_index(drop=True)
        val_df = df.iloc[split.val_indices].reset_index(drop=True)
        test_df = df.iloc[split.test_indices].reset_index(drop=True)
    else:
        train_ratio = data_cfg.get("train_ratio", 0.70)
        val_ratio = data_cfg.get("val_ratio", 0.15)
        test_ratio = data_cfg.get("test_ratio", 0.15)
        split = split_random(df, train_frac=train_ratio, val_frac=val_ratio, test_frac=test_ratio, seed=seed)
        train_df = df.iloc[split.train_indices].reset_index(drop=True)
        val_df = df.iloc[split.val_indices].reset_index(drop=True)
        test_df = df.iloc[split.test_indices].reset_index(drop=True)

    print(f"Data partition ({split_strategy}): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 3. Build Datasets and compute normalization statistics
    target_time_steps = data_cfg.get("target_time_steps", 2048)
    target_channels = data_cfg.get("target_channels", ["u", "f_r", "e_h"])
    use_history_channel = data_cfg.get("use_history_channel", False)
    normalize = data_cfg.get("normalize", True)

    train_ds_raw = SeismicSDOFDataset(
        train_df,
        target_time_steps=target_time_steps,
        target_channels=target_channels,
        use_history_channel=use_history_channel,
    )

    x_norm = None
    y_norm = None
    if normalize and len(train_ds_raw) > 0:
        sample_size = min(len(train_ds_raw), 512)
        x_samples = []
        y_samples = []
        for i in range(sample_size):
            xs, ys = train_ds_raw[i]
            x_samples.append(xs)
            y_samples.append(ys)
        x_norm = UnitGaussianNormalizer().fit(torch.stack(x_samples, dim=0))
        y_norm = UnitGaussianNormalizer().fit(torch.stack(y_samples, dim=0))

    train_ds = SeismicSDOFDataset(
        train_df,
        target_time_steps=target_time_steps,
        target_channels=target_channels,
        use_history_channel=use_history_channel,
        x_normalizer=x_norm,
        y_normalizer=y_norm,
    )
    val_ds = SeismicSDOFDataset(
        val_df,
        target_time_steps=target_time_steps,
        target_channels=target_channels,
        use_history_channel=use_history_channel,
        x_normalizer=x_norm,
        y_normalizer=y_norm,
    )
    # Test dataset configured to return metadata for regime disaggregation
    test_ds = SeismicSDOFDataset(
        test_df,
        target_time_steps=target_time_steps,
        target_channels=target_channels,
        use_history_channel=use_history_channel,
        return_meta=True,
        x_normalizer=x_norm,
        y_normalizer=y_norm,
    )

    batch_size = data_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 4. Build Model
    model_cfg = config["model"]
    model_type = model_cfg.get("model_type", "fno").lower()
    in_channels = 12 if use_history_channel else model_cfg.get("in_channels", 10)
    out_channels = model_cfg.get("out_channels", 3)

    if model_type == "lstm":
        hidden_dim = model_cfg.get("hidden_dim", 128)
        num_layers = model_cfg.get("num_layers", 3)
        dropout = model_cfg.get("dropout", 0.1)
        bidirectional = model_cfg.get("bidirectional", False)
        model = LSTMBaseline(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        print(f"LSTM Baseline constructed with {model.get_num_parameters():,} trainable parameters (hidden_dim={hidden_dim}, layers={num_layers}).")
    elif model_type == "mlp":
        hidden_dim = model_cfg.get("hidden_dim", 256)
        num_layers = model_cfg.get("num_layers", 4)
        dropout = model_cfg.get("dropout", 0.0)
        model = MLPBaseline(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        print(f"MLP Baseline constructed with {model.get_num_parameters():,} trainable parameters (hidden_dim={hidden_dim}, layers={num_layers}).")
    else:
        modes = model_cfg.get("modes", 128)
        width = model_cfg.get("width", 48)
        n_layers = model_cfg.get("n_layers", 4)
        activation = model_cfg.get("activation", "gelu")
        use_norm = model_cfg.get("use_norm", False)
        model = FNO1d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            width=width,
            n_layers=n_layers,
            activation=activation,
            use_norm=use_norm,
        )
        print(f"FNO Model constructed with {model.get_num_parameters():,} trainable parameters (in_channels={in_channels}, modes={modes}, width={width}).")

    # 5. Optimizer, Scheduler, and Loss
    train_cfg = config["training"]
    epochs = train_cfg.get("epochs", 40)
    lr = float(train_cfg.get("learning_rate", 0.005))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(train_cfg.get("min_lr", 1e-5))
    )

    loss_cfg = config.get("loss", {})
    loss_weights = loss_cfg.get("data_weights", [1.0, 1.0, 1.0])
    lambda_energy = float(loss_cfg.get("lambda_energy", 0.0))
    lambda_boundary = float(loss_cfg.get("lambda_boundary", 0.0))

    criterion = CompositePhysicsLoss(
        data_weights=loss_weights,
        lambda_energy=lambda_energy,
        lambda_boundary=lambda_boundary,
    )
    print(f"Loss Configuration: DataWeights={loss_weights}, Lambda_Energy={lambda_energy}, Lambda_Boundary={lambda_boundary}")

    # 6. Output Directories
    out_cfg = config.get("output", {})
    ckpt_dir = Path(out_cfg.get("checkpoint_dir", f"experiments/{exp_name}/checkpoints"))
    res_dir = Path(out_cfg.get("results_dir", f"experiments/{exp_name}/results"))
    log_dir = Path(out_cfg.get("log_dir", f"experiments/{exp_name}/logs"))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 7. Trainer
    trainer = FNOTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        clip_grad_norm=float(train_cfg.get("clip_grad_norm", 1.0)),
        y_normalizer=y_norm,
    )

    print(f"Starting training for {epochs} epochs...")
    t_start = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=ckpt_dir,
        save_best=True,
        print_interval=5,
    )
    total_train_time = time.time() - t_start

    # 8. Disaggregated Test Set Evaluation
    print("\nEvaluating best checkpoint on unseen test set (disaggregated by regime)...")
    test_metrics = trainer.evaluate_disaggregated(test_loader, unnormalize=True)

    # Disaggregated metrics
    u_all = test_metrics["overall_rel_l2_u"]
    e_all = test_metrics["overall_rel_l2_e_h"]
    u_el = test_metrics["elastic_rel_l2_u"]
    e_el = test_metrics["elastic_rel_l2_e_h"]
    u_post = test_metrics["post_yield_rel_l2_u"]
    e_post = test_metrics["post_yield_rel_l2_e_h"]

    print("=" * 70)
    print(f"SeismoFNO Experiment '{exp_name}' Results (Held-Out-Earthquake Split)")
    print("=" * 70)
    print(f"  [Overall Test Set: {test_metrics['n_total']} records]")
    print(f"    - Relative L2 Displacement u(t) : {u_all:.4e} ({u_all * 100:.2f} %)")
    print(f"    - Relative L2 Hysteretic E_h(t) : {e_all:.4e} ({e_all * 100:.2f} %)")
    print(f"  [Elastic Regime (mu <= 1.0): {test_metrics['n_elastic']} records]")
    print(f"    - Relative L2 Displacement u(t) : {u_el:.4e} ({u_el * 100:.2f} %)")
    print(f"    - Relative L2 Hysteretic E_h(t) : {e_el:.4e} ({e_el * 100:.2f} %)")
    print(f"  [Post-Yield Regime (mu > 1.0): {test_metrics['n_post_yield']} records]")
    print(f"    - Relative L2 Displacement u(t) : {u_post:.4e} ({u_post * 100:.2f} %)")
    print(f"    - Relative L2 Hysteretic E_h(t) : {e_post:.4e} ({e_post * 100:.2f} %)")
    print(f"  Total Training Time : {total_train_time:.2f} s")
    print("=" * 70)

    # 9. Save trace metrics and artifacts
    run_summary = {
        "experiment_name": exp_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "test_metrics": test_metrics,
        "total_train_time_s": total_train_time,
        "n_parameters": model.get_num_parameters(),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    }

    metrics_file = res_dir / "test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    history_file = log_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return run_summary


def main():
    parser = argparse.ArgumentParser(description="Train SeismoFNO model.")
    parser.add_argument("--config", type=str, default="configs/phase5_a_data_only.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_fno(cfg)


if __name__ == "__main__":
    main()
