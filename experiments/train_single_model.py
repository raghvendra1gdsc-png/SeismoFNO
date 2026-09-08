"""
train_single_model.py — Fast, focused MPS GPU trainer for individual EXP6 models.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.graph_dataset import GraphNormalizer
from src.data_pipeline.modal_dataset import (
    SeismicMDOFModalGraphDataset,
    ModalNormalizer,
    collate_modal_structural_graphs,
)
from src.models.conditioned_gno import ConditionedSpatiotemporalGNO


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff_norm = torch.norm(pred - target, p=2, dim=-1)
    target_norm = torch.norm(target, p=2, dim=-1) + 1e-6
    return torch.mean(diff_norm / target_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, choices=["t1_gno", "multimodal_gno", "shuffled_modal_gno"])
    parser.add_argument("--cond-mode", type=str, required=True, choices=["t1", "multimodal"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-cond", action="store_true", default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== TRAINING {args.model_name.upper()} on {device} (batch_size={args.batch_size}, max_epochs={args.max_epochs}) ===", flush=True)

    exp6_dir = PROJECT_ROOT / "results" / "experiments" / "exp6"
    train_dir = exp6_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Load split manifest
    split_df = pd.read_csv(exp6_dir / "split_manifest.csv")
    train_df = split_df[split_df["partition"] == "train"].reset_index(drop=True)
    val_df = split_df[split_df["partition"] == "val"].reset_index(drop=True)

    # Load scalers
    scalers = torch.load(exp6_dir / "scalers.pt", map_location="cpu")
    x_norm = GraphNormalizer(scalers["x_mean"], scalers["x_std"])
    y_norm = GraphNormalizer(scalers["y_mean"], scalers["y_std"])

    if args.cond_mode == "t1":
        modal_norm = ModalNormalizer(scalers["modal_t1_mean"], scalers["modal_t1_std"])
        cond_dim = 1
    else:
        modal_norm = ModalNormalizer(scalers["modal_mm_mean"], scalers["modal_mm_std"])
        cond_dim = 6

    train_ds = SeismicMDOFModalGraphDataset(
        train_df, x_normalizer=x_norm, y_normalizer=y_norm, modal_normalizer=modal_norm,
        cond_mode=args.cond_mode, shuffle_cond=args.shuffle_cond,
    )
    val_ds = SeismicMDOFModalGraphDataset(
        val_df, x_normalizer=x_norm, y_normalizer=y_norm, modal_normalizer=modal_norm,
        cond_mode=args.cond_mode, shuffle_cond=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_modal_structural_graphs, pin_memory=False, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_modal_structural_graphs, pin_memory=False, num_workers=0,
    )

    model = ConditionedSpatiotemporalGNO(
        in_channels=10,
        out_channels=3,
        width=48,
        modes=64,
        n_layers=4,
        cond_dim=cond_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-5)

    history = {
        "epoch": [], "lr": [], "train_loss": [], "val_loss": [], "val_rel_l2_u": [],
        "epoch_time_s": [], "cumulative_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    patience_cnt = 0
    t_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_acc = 0.0
        n_train_batches = 0

        for b in train_loader:
            b = b.to(device)
            optimizer.zero_grad()
            pred = model(b.x, b.edge_index, b.edge_attr, cond=b.modal_cond, batch_idx=b.batch_idx)
            loss = relative_l2_loss(pred, b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_acc += loss.item()
            n_train_batches += 1

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]
        avg_train_loss = train_loss_acc / n_train_batches

        # Validation loop
        model.eval()
        val_loss_acc = 0.0
        val_u_acc = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for b in val_loader:
                b = b.to(device)
                pred = model(b.x, b.edge_index, b.edge_attr, cond=b.modal_cond, batch_idx=b.batch_idx)
                val_loss_acc += relative_l2_loss(pred, b.y).item()

                pred_phys = y_norm.decode(pred)
                y_phys = y_norm.decode(b.y)
                u_p = pred_phys[:, 0, :]
                u_t = y_phys[:, 0, :]
                rel_l2_u = float(torch.norm(u_p - u_t, p=2) / (torch.norm(u_t, p=2) + 1e-8))
                val_u_acc += rel_l2_u
                n_val_batches += 1

        avg_val_loss = val_loss_acc / n_val_batches
        avg_val_u = val_u_acc / n_val_batches
        epoch_time = time.time() - t0
        cum_time = time.time() - t_start

        history["epoch"].append(epoch)
        history["lr"].append(curr_lr)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_rel_l2_u"].append(avg_val_u * 100.0)
        history["epoch_time_s"].append(epoch_time)
        history["cumulative_time_s"].append(cum_time)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_cnt = 0
            ckpt_path = train_dir / f"best_{args.model_name}.pt"
            torch.save({
                "epoch": epoch,
                "model_name": args.model_name,
                "cond_mode": args.cond_mode,
                "cond_dim": cond_dim,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_rel_l2_u": avg_val_u,
                "n_params": n_params,
                "seed": args.seed,
            }, ckpt_path)
        else:
            patience_cnt += 1

        if device.type == "mps":
            torch.mps.empty_cache()

        print(
            f"Epoch [{epoch:2d}/{args.max_epochs:2d}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Rel L2 u: {avg_val_u * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
            f"{' -> BEST' if is_best else ''}",
            flush=True,
        )

        if patience_cnt >= args.patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(train_dir / f"history_{args.model_name}.csv", index=False)

    summary = {
        "model_name": args.model_name,
        "cond_mode": args.cond_mode,
        "cond_dim": cond_dim,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history["epoch"]),
        "total_training_time_s": cum_time,
        "n_params": n_params,
    }
    with open(train_dir / f"summary_{args.model_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[COMPLETED] {args.model_name}: Best Epoch={best_epoch}, Best Val Loss={best_val_loss:.4f} in {cum_time:.1f}s.")


if __name__ == "__main__":
    main()
