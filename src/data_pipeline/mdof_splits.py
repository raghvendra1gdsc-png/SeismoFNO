"""
mdof_splits.py — Leakage-Free Data Split Generators for MDOF Surrogate Training.

Implements three split strategies for MDOF simulation datasets:
  1. Random split (baseline benchmark only)
  2. Held-Out-Earthquake split (strict zero earthquake leakage)
  3. Held-Out-Structure split (strict zero structural archetype leakage)
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.data_pipeline.splits import SplitResult, save_split, load_split


def create_mdof_splits(
    index_csv: str = "data/simulations/mdof/simulation_index.csv",
    output_dir: str = "data/processed/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, SplitResult]:
    """
    Generate all three leakage-free split strategies for MDOF dataset.

    Args:
        index_csv: Path to MDOF simulation manifest
        output_dir: Output directory for JSON split files
        train_ratio, val_ratio, test_ratio: Target partition ratios
        seed: Random seed for reproducibility

    Returns:
        Dictionary of {split_name: SplitResult}
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_csv)
    np.random.seed(seed)
    n_total = len(df)

    splits = {}

    # --------------------------------------------------------------------------
    # 1. Random Split (Baseline only)
    # --------------------------------------------------------------------------
    shuffled_indices = np.random.permutation(n_total).tolist()
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx_rand = shuffled_indices[:n_train]
    val_idx_rand = shuffled_indices[n_train : n_train + n_val]
    test_idx_rand = shuffled_indices[n_train + n_val :]

    train_ids_rand = df.iloc[train_idx_rand]["sim_id"].tolist()
    val_ids_rand = df.iloc[val_idx_rand]["sim_id"].tolist()
    test_ids_rand = df.iloc[test_idx_rand]["sim_id"].tolist()

    random_split_res = SplitResult(
        split_strategy="random",
        train_indices=train_idx_rand,
        val_indices=val_idx_rand,
        test_indices=test_idx_rand,
        train_ids=train_ids_rand,
        val_ids=val_ids_rand,
        test_ids=test_ids_rand,
        train_earthquakes=sorted(df.iloc[train_idx_rand]["earthquake_id"].unique().tolist()),
        val_earthquakes=sorted(df.iloc[val_idx_rand]["earthquake_id"].unique().tolist()),
        test_earthquakes=sorted(df.iloc[test_idx_rand]["earthquake_id"].unique().tolist()),
        train_structures=sorted(df.iloc[train_idx_rand]["struct_id"].unique().tolist()),
        val_structures=sorted(df.iloc[val_idx_rand]["struct_id"].unique().tolist()),
        test_structures=sorted(df.iloc[test_idx_rand]["struct_id"].unique().tolist()),
        n_train=len(train_ids_rand),
        n_val=len(val_ids_rand),
        n_test=len(test_ids_rand),
        total=n_total,
    )
    save_split(random_split_res, str(out_path / "mdof_random_split.json"))
    splits["random"] = random_split_res

    # --------------------------------------------------------------------------
    # 2. Held-Out-Earthquake Split (Zero earthquake event leakage)
    # --------------------------------------------------------------------------
    unique_eqs = sorted(df["earthquake_id"].unique().tolist())
    np.random.shuffle(unique_eqs)
    n_eq = len(unique_eqs)
    n_eq_train = int(n_eq * train_ratio)
    n_eq_val = int(n_eq * val_ratio)

    train_eqs = set(unique_eqs[:n_eq_train])
    val_eqs = set(unique_eqs[n_eq_train : n_eq_train + n_eq_val])
    test_eqs = set(unique_eqs[n_eq_train + n_eq_val :])

    train_mask_eq = df["earthquake_id"].isin(train_eqs)
    val_mask_eq = df["earthquake_id"].isin(val_eqs)
    test_mask_eq = df["earthquake_id"].isin(test_eqs)

    train_idx_eq = df[train_mask_eq].index.tolist()
    val_idx_eq = df[val_mask_eq].index.tolist()
    test_idx_eq = df[test_mask_eq].index.tolist()

    eq_split_res = SplitResult(
        split_strategy="held_out_earthquake",
        train_indices=train_idx_eq,
        val_indices=val_idx_eq,
        test_indices=test_idx_eq,
        train_ids=df.iloc[train_idx_eq]["sim_id"].tolist(),
        val_ids=df.iloc[val_idx_eq]["sim_id"].tolist(),
        test_ids=df.iloc[test_idx_eq]["sim_id"].tolist(),
        train_earthquakes=sorted(list(train_eqs)),
        val_earthquakes=sorted(list(val_eqs)),
        test_earthquakes=sorted(list(test_eqs)),
        train_structures=sorted(df.iloc[train_idx_eq]["struct_id"].unique().tolist()),
        val_structures=sorted(df.iloc[val_idx_eq]["struct_id"].unique().tolist()),
        test_structures=sorted(df.iloc[test_idx_eq]["struct_id"].unique().tolist()),
        n_train=len(train_idx_eq),
        n_val=len(val_idx_eq),
        n_test=len(test_idx_eq),
        total=n_total,
    )
    save_split(eq_split_res, str(out_path / "mdof_held_out_earthquake_split.json"))
    splits["held_out_earthquake"] = eq_split_res

    # --------------------------------------------------------------------------
    # 3. Held-Out-Structure Split (Zero structural archetype leakage)
    # --------------------------------------------------------------------------
    unique_structs = sorted(df["struct_id"].unique().tolist())
    test_structs = {"3S_T090", "5S_T120"}
    train_val_structs = [s for s in unique_structs if s not in test_structs]
    np.random.shuffle(train_val_structs)

    val_structs = {train_val_structs[0]}
    train_structs = set(train_val_structs[1:])

    train_mask_st = df["struct_id"].isin(train_structs)
    val_mask_st = df["struct_id"].isin(val_structs)
    test_mask_st = df["struct_id"].isin(test_structs)

    train_idx_st = df[train_mask_st].index.tolist()
    val_idx_st = df[val_mask_st].index.tolist()
    test_idx_st = df[test_mask_st].index.tolist()

    struct_split_res = SplitResult(
        split_strategy="held_out_structure",
        train_indices=train_idx_st,
        val_indices=val_idx_st,
        test_indices=test_idx_st,
        train_ids=df.iloc[train_idx_st]["sim_id"].tolist(),
        val_ids=df.iloc[val_idx_st]["sim_id"].tolist(),
        test_ids=df.iloc[test_idx_st]["sim_id"].tolist(),
        train_earthquakes=sorted(df.iloc[train_idx_st]["earthquake_id"].unique().tolist()),
        val_earthquakes=sorted(df.iloc[val_idx_st]["earthquake_id"].unique().tolist()),
        test_earthquakes=sorted(df.iloc[test_idx_st]["earthquake_id"].unique().tolist()),
        train_structures=sorted(list(train_structs)),
        val_structures=sorted(list(val_structs)),
        test_structures=sorted(list(test_structs)),
        n_train=len(train_idx_st),
        n_val=len(val_idx_st),
        n_test=len(test_idx_st),
        total=n_total,
    )
    save_split(struct_split_res, str(out_path / "mdof_held_out_structure_split.json"))
    splits["held_out_structure"] = struct_split_res

    print("MDOF splits generated:")
    for name, sp in splits.items():
        print(f"  {name:25s}: Train={sp.n_train}, Val={sp.n_val}, Test={sp.n_test}")

    return splits


if __name__ == "__main__":
    create_mdof_splits()
