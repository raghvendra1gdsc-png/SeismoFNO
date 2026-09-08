"""
trainer.py — Modular Trainer Engine for Fourier Neural Operators with Physics-Informed Losses.

Handles:
  - Model forward/backward passes with CompositePhysicsLoss (data, energy consistency, boundary)
  - Relative L2 loss tracking across training and validation
  - Channel-specific evaluation (displacement u, force F_R, energy E_h)
  - Disaggregated evaluation: elastic regime (mu <= 1.0) vs. post-yield regime (mu > 1.0)
  - Learning rate scheduling and checkpoint management
  - Traceable experiment logging per AGENTS.md Hard Rule 3
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import time
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.losses.data_loss import RelativeL2Loss, MultiChannelRelativeL2Loss, CompositePhysicsLoss
from src.data_pipeline.dataset_builder import UnitGaussianNormalizer


class FNOTrainer:
    """Trainer class for FNO dynamic models with physics loss support and regime disaggregation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        clip_grad_norm: float = 1.0,
        y_normalizer: Optional[UnitGaussianNormalizer] = None,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else RelativeL2Loss()
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.y_normalizer = y_normalizer
        self.eval_criterion = RelativeL2Loss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_energy_loss = 0.0
        total_boundary_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)

            # Unnormalize prediction if needed for physical loss terms
            pred_phys = self.y_normalizer.decode(pred) if self.y_normalizer is not None else pred

            if isinstance(self.criterion, CompositePhysicsLoss):
                loss_dict = self.criterion(pred, y, pred_physical=pred_phys)
                loss = loss_dict["loss"]
                total_data_loss += float(loss_dict["data_loss"].item())
                total_energy_loss += float(loss_dict["energy_loss"].item())
                total_boundary_loss += float(loss_dict["boundary_loss"].item())
            else:
                loss = self.criterion(pred, y)

            loss.backward()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "train_loss": total_loss / max(1, n_batches),
            "train_data_loss": total_data_loss / max(1, n_batches),
            "train_energy_loss": total_energy_loss / max(1, n_batches),
            "train_boundary_loss": total_boundary_loss / max(1, n_batches),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, unnormalize: bool = True) -> Dict[str, float]:
        """
        Standard evaluation returning overall metrics on validation set.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_targets = []

        for batch in dataloader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            pred_phys = self.y_normalizer.decode(pred) if (unnormalize and self.y_normalizer is not None) else pred
            y_phys = self.y_normalizer.decode(y) if (unnormalize and self.y_normalizer is not None) else y

            if isinstance(self.criterion, CompositePhysicsLoss):
                loss_dict = self.criterion(pred, y, pred_physical=pred_phys)
                loss = loss_dict["loss"]
            else:
                loss = self.criterion(pred, y)

            total_loss += float(loss.item())
            all_preds.append(pred_phys.cpu())
            all_targets.append(y_phys.cpu())
            n_batches += 1

        preds_cat = torch.cat(all_preds, dim=0)    # [N, C, T]
        targets_cat = torch.cat(all_targets, dim=0) # [N, C, T]

        metrics = {"val_loss": total_loss / max(1, n_batches)}

        channel_names = ["u", "f_r", "e_h", "v", "total_accel"]
        n_channels = preds_cat.size(1)

        for c in range(n_channels):
            c_name = channel_names[c] if c < len(channel_names) else f"ch_{c}"
            p_c = preds_cat[:, c, :]
            t_c = targets_cat[:, c, :]

            diff_l2 = torch.norm(p_c - t_c, p=2, dim=1)
            targ_l2 = torch.norm(t_c, p=2, dim=1) + 1e-7
            rel_l2_mean = float(torch.mean(diff_l2 / targ_l2).item())

            metrics[f"rel_l2_{c_name}"] = rel_l2_mean

        diff_all = torch.norm(preds_cat.view(preds_cat.size(0), -1) - targets_cat.view(targets_cat.size(0), -1), p=2, dim=1)
        targ_all = torch.norm(targets_cat.view(targets_cat.size(0), -1), p=2, dim=1) + 1e-7
        metrics["rel_l2_global"] = float(torch.mean(diff_all / targ_all).item())

        return metrics

    @torch.no_grad()
    def evaluate_disaggregated(
        self,
        dataloader: DataLoader,
        unnormalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Disaggregated evaluation on test set:
          1. Overall test set
          2. Elastic regime subset (ductility mu <= 1.0)
          3. Post-yield regime subset (ductility mu > 1.0)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_targets = []
        all_ductility = []
        all_is_post_yield = []

        for batch in dataloader:
            if len(batch) == 3:
                x, y, meta = batch
                duct = meta["ductility"].cpu().numpy()
                post_y = meta["is_post_yield"].cpu().numpy()
            else:
                x, y = batch
                duct = np.zeros(x.size(0))
                post_y = np.zeros(x.size(0))

            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            pred_phys = self.y_normalizer.decode(pred) if (unnormalize and self.y_normalizer is not None) else pred
            y_phys = self.y_normalizer.decode(y) if (unnormalize and self.y_normalizer is not None) else y

            if isinstance(self.criterion, CompositePhysicsLoss):
                loss_dict = self.criterion(pred, y, pred_physical=pred_phys)
                loss = loss_dict["loss"]
            else:
                loss = self.criterion(pred, y)

            total_loss += float(loss.item())
            all_preds.append(pred_phys.cpu())
            all_targets.append(y_phys.cpu())
            all_ductility.extend(duct.tolist())
            all_is_post_yield.extend(post_y.tolist())
            n_batches += 1

        preds_cat = torch.cat(all_preds, dim=0)     # [N, C, T]
        targets_cat = torch.cat(all_targets, dim=0) # [N, C, T]
        ductility_arr = np.array(all_ductility)
        is_post_yield_arr = np.array(all_is_post_yield)

        def calc_rel_l2_subset(p: torch.Tensor, t: torch.Tensor, mask: np.ndarray) -> Dict[str, float]:
            if np.sum(mask) == 0:
                return {"rel_l2_u": float("nan"), "rel_l2_f_r": float("nan"), "rel_l2_e_h": float("nan"), "rel_l2_global": float("nan")}
            p_sub = p[mask]
            t_sub = t[mask]

            res = {}
            channel_names = ["u", "f_r", "e_h", "v", "total_accel"]
            for c in range(min(3, p_sub.size(1))):
                c_name = channel_names[c]
                diff = torch.norm(p_sub[:, c, :] - t_sub[:, c, :], p=2, dim=1)
                denom = torch.norm(t_sub[:, c, :], p=2, dim=1) + 1e-7
                res[f"rel_l2_{c_name}"] = float(torch.mean(diff / denom).item())

            diff_all = torch.norm(p_sub.view(p_sub.size(0), -1) - t_sub.view(t_sub.size(0), -1), p=2, dim=1)
            targ_all = torch.norm(t_sub.view(t_sub.size(0), -1), p=2, dim=1) + 1e-7
            res["rel_l2_global"] = float(torch.mean(diff_all / targ_all).item())
            return res

        # 1. Overall
        mask_all = np.ones(len(ductility_arr), dtype=bool)
        metrics_overall = calc_rel_l2_subset(preds_cat, targets_cat, mask_all)

        # 2. Elastic regime (mu <= 1.0)
        mask_elastic = ductility_arr <= 1.0
        metrics_elastic = calc_rel_l2_subset(preds_cat, targets_cat, mask_elastic)

        # 3. Post-yield regime (mu > 1.0)
        mask_post_yield = ductility_arr > 1.0
        metrics_post_yield = calc_rel_l2_subset(preds_cat, targets_cat, mask_post_yield)

        return {
            "val_loss": total_loss / max(1, n_batches),
            "n_total": int(len(ductility_arr)),
            "n_elastic": int(np.sum(mask_elastic)),
            "n_post_yield": int(np.sum(mask_post_yield)),
            # Overall
            "overall_rel_l2_u": metrics_overall.get("rel_l2_u", np.nan),
            "overall_rel_l2_f_r": metrics_overall.get("rel_l2_f_r", np.nan),
            "overall_rel_l2_e_h": metrics_overall.get("rel_l2_e_h", np.nan),
            "overall_rel_l2_global": metrics_overall.get("rel_l2_global", np.nan),
            # Elastic regime (mu <= 1.0)
            "elastic_rel_l2_u": metrics_elastic.get("rel_l2_u", np.nan),
            "elastic_rel_l2_f_r": metrics_elastic.get("rel_l2_f_r", np.nan),
            "elastic_rel_l2_e_h": metrics_elastic.get("rel_l2_e_h", np.nan),
            "elastic_rel_l2_global": metrics_elastic.get("rel_l2_global", np.nan),
            # Post-yield regime (mu > 1.0)
            "post_yield_rel_l2_u": metrics_post_yield.get("rel_l2_u", np.nan),
            "post_yield_rel_l2_f_r": metrics_post_yield.get("rel_l2_f_r", np.nan),
            "post_yield_rel_l2_e_h": metrics_post_yield.get("rel_l2_e_h", np.nan),
            "post_yield_rel_l2_global": metrics_post_yield.get("rel_l2_global", np.nan),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        save_best: bool = True,
        print_interval: int = 10,
    ) -> Dict[str, Any]:
        """Train model for multiple epochs and log metrics."""
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "rel_l2_u": [],
            "rel_l2_f_r": [],
            "rel_l2_e_h": [],
            "rel_l2_global": [],
            "time_per_epoch_s": [],
        }

        best_val_loss = float("inf")
        best_state = None

        if checkpoint_dir is not None:
            ckpt_path = Path(checkpoint_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            t_epoch = time.time() - t0

            history["epoch"].append(epoch)
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["rel_l2_u"].append(val_metrics.get("rel_l2_u", np.nan))
            history["rel_l2_f_r"].append(val_metrics.get("rel_l2_f_r", np.nan))
            history["rel_l2_e_h"].append(val_metrics.get("rel_l2_e_h", np.nan))
            history["rel_l2_global"].append(val_metrics.get("rel_l2_global", np.nan))
            history["time_per_epoch_s"].append(t_epoch)

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                if checkpoint_dir is not None and save_best:
                    torch.save(best_state, Path(checkpoint_dir) / "best_model.pt")

            if epoch == 1 or epoch % print_interval == 0 or epoch == epochs:
                u_err = val_metrics.get("rel_l2_u", 0.0)
                e_err = val_metrics.get("rel_l2_e_h", 0.0)
                print(
                    f"Epoch [{epoch:3d}/{epochs:3d}] | "
                    f"Train Loss: {train_metrics['train_loss']:.4e} | "
                    f"Val Loss: {val_metrics['val_loss']:.4e} | "
                    f"Rel L2 u: {u_err * 100:.2f}% | "
                    f"Rel L2 E_h: {e_err * 100:.2f}% | "
                    f"Time: {t_epoch:.2f}s"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "best_val_loss": best_val_loss,
            "final_val_metrics": val_metrics,
            "history": history,
        }
