"""
prepare_exp3_cache.py — Pre-load and cache train/val/test tensors for instant training & evaluation.
"""

from pathlib import Path
import torch
import pandas as pd

from src.data_pipeline.dataset_builder import SeismicSDOFDataset, UnitGaussianNormalizer
from src.data_pipeline.splits import load_split
from src.evaluation.exp3_interventions import compute_exact_physical_state

cache_dir = Path("data/processed/exp3_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

train_cache_path = cache_dir / "train_cache.pt"
val_cache_path = cache_dir / "val_cache.pt"

split_path = Path("data/processed/splits/held_out_earthquake_split.json")
sim_index_path = Path("data/simulations/simulation_index.csv")

split_data = load_split(split_path)
sim_df = pd.read_csv(sim_index_path)
sim_df = sim_df[sim_df["material_type"] == "bilinear"].reset_index(drop=True)

train_df = sim_df[sim_df["sim_id"].isin(split_data.train_ids)].reset_index(drop=True)
val_df = sim_df[sim_df["sim_id"].isin(split_data.val_ids)].reset_index(drop=True)

def process_and_cache(df, save_path):
    raw_ds = SeismicSDOFDataset(df, target_time_steps=2048, target_channels=["u", "f_r", "e_h"], use_history_channel=False)
    x_list, y_list, s_list = [], [], []
    for i in range(len(raw_ds)):
        xs, ys = raw_ds[i]
        x_list.append(xs)
        y_list.append(ys)
        k0_v = xs[3, 0].item()
        alpha_v = xs[7, 0].item()
        up_s, ab_s = compute_exact_physical_state(ys[0], ys[1], k0=k0_v, alpha=alpha_v)
        s_list.append(torch.stack([up_s, ab_s], dim=0))
    torch.save({
        "x": torch.stack(x_list, dim=0),
        "y": torch.stack(y_list, dim=0),
        "s": torch.stack(s_list, dim=0),
    }, save_path)
    print(f"Saved {len(df)} samples to {save_path}")

if not train_cache_path.exists():
    print("Caching train dataset...")
    process_and_cache(train_df, train_cache_path)

if not val_cache_path.exists():
    print("Caching val dataset...")
    process_and_cache(val_df, val_cache_path)

print("EXP 3 cache ready!")
