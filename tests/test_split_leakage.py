"""
test_split_leakage.py — Zero-Leakage Split Verification Suite.

Per AGENTS.md Hard Rules 1, 2, and Phase 3 specifications:
  1. Held-out-earthquake split MUST have strictly 0 % overlap in earthquake events
     between train and test, and between train and validation.
  2. Held-out-structure split MUST have strictly 0 % overlap in structure configurations
     between train and test, and between train and validation.
  3. Every split must have pairwise disjoint index sets.
  4. Random split is verified as a baseline but flagged so it cannot be used for generalization claims.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.splits import (
    SplitResult,
    split_random,
    split_held_out_earthquake,
    split_held_out_structure,
    verify_no_leakage,
    save_split,
    load_split,
)


@pytest.fixture
def sample_simulation_df():
    """Build a realistic mock/cached simulation index dataframe."""
    manifest_path = Path("data/simulations/dataset_manifest.csv")
    if manifest_path.exists():
        df_manifest = pd.read_csv(manifest_path)
        # Create a combined dataframe with simulated structures
        struct_ids = ["SDOF_T0p20s_ELAS", "SDOF_T0p50s_ELAS", "SDOF_T1p00s_ELAS",
                      "SDOF_T0p50s_BILIN_uy0p01", "SDOF_T1p00s_BILIN_uy0p02", "SDOF_T1p50s_BILIN_uy0p03"]
        rows = []
        for _, rec in df_manifest.head(100).iterrows():
            for sid in struct_ids:
                rows.append({
                    "sim_id": f"{rec['record_id']}__{sid}",
                    "record_id": rec["record_id"],
                    "base_record_id": rec["base_record_id"],
                    "earthquake_name": rec["earthquake_name"],
                    "magnitude": rec["magnitude"],
                    "r_rup_km": rec["r_rup_km"],
                    "vs30_ms": rec["vs30_ms"],
                    "target_pga_g": rec["target_pga_g"],
                    "resulting_pga_g": rec["resulting_pga_g"],
                    "struct_id": sid,
                })
        return pd.DataFrame(rows)
    else:
        # Synthetic fallback for pure unit testing
        eq_names = ["Imperial Valley", "Loma Prieta", "Northridge", "Kobe", "Chi-Chi", "Landers", "Kocaeli", "San Fernando"]
        struct_ids = [f"SDOF_T0p{i}0s" for i in range(1, 7)]
        rows = []
        for eq in eq_names:
            for r in range(10):
                for s in struct_ids:
                    rows.append({
                        "sim_id": f"{eq}_R{r}_{s}",
                        "earthquake_name": eq,
                        "struct_id": s,
                        "target_pga_g": 0.4,
                    })
        return pd.DataFrame(rows)


def test_held_out_earthquake_split_zero_leakage(sample_simulation_df):
    """
    CRITICAL ASSERTION:
    Assert zero overlap of earthquake IDs between train, val, and test splits.
    """
    df = sample_simulation_df
    split = split_held_out_earthquake(df, test_eq_frac=0.25, val_eq_frac=0.15, seed=42)

    train_eqs = set(split.train_earthquakes)
    val_eqs = set(split.val_earthquakes)
    test_eqs = set(split.test_earthquakes)

    print(f"\n[Held-Out Earthquake Split]")
    print(f"  Train Earthquakes ({len(train_eqs)}) : {sorted(train_eqs)}")
    print(f"  Val Earthquakes   ({len(val_eqs)})   : {sorted(val_eqs)}")
    print(f"  Test Earthquakes  ({len(test_eqs)})  : {sorted(test_eqs)}")

    # 1. Assert non-empty splits
    assert len(train_eqs) > 0, "Train earthquakes set is empty"
    assert len(test_eqs) > 0, "Test earthquakes set is empty"
    assert len(val_eqs) > 0, "Val earthquakes set is empty"

    # 2. Strict zero earthquake overlap assertions
    overlap_train_test = train_eqs.intersection(test_eqs)
    overlap_train_val = train_eqs.intersection(val_eqs)
    overlap_val_test = val_eqs.intersection(test_eqs)

    assert len(overlap_train_test) == 0, f"Earthquake leakage between train and test: {overlap_train_test}"
    assert len(overlap_train_val) == 0, f"Earthquake leakage between train and val: {overlap_train_val}"
    assert len(overlap_val_test) == 0, f"Earthquake leakage between val and test: {overlap_val_test}"

    # 3. Assert index disjointness
    train_idx = set(split.train_indices)
    val_idx = set(split.val_indices)
    test_idx = set(split.test_indices)

    assert train_idx.isdisjoint(test_idx), "Index leakage between train and test"
    assert train_idx.isdisjoint(val_idx), "Index leakage between train and val"
    assert val_idx.isdisjoint(test_idx), "Index leakage between val and test"

    # 4. Assert all samples accounted for
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    # 5. Formal verification helper
    ok, diag = verify_no_leakage(split, df)
    assert ok, f"verify_no_leakage failed: {diag}"


def test_held_out_structure_split_zero_leakage(sample_simulation_df):
    """
    CRITICAL ASSERTION:
    Assert zero overlap of structure IDs between train, val, and test splits.
    """
    df = sample_simulation_df
    split = split_held_out_structure(df, test_struct_frac=0.25, val_struct_frac=0.15, seed=42)

    train_structs = set(split.train_structures)
    val_structs = set(split.val_structures)
    test_structs = set(split.test_structures)

    print(f"\n[Held-Out Structure Split]")
    print(f"  Train Structures ({len(train_structs)}) : {sorted(train_structs)}")
    print(f"  Val Structures   ({len(val_structs)})   : {sorted(val_structs)}")
    print(f"  Test Structures  ({len(test_structs)})  : {sorted(test_structs)}")

    # 1. Assert non-empty splits
    assert len(train_structs) > 0, "Train structures set is empty"
    assert len(test_structs) > 0, "Test structures set is empty"
    assert len(val_structs) > 0, "Val structures set is empty"

    # 2. Strict zero structure overlap assertions
    overlap_train_test = train_structs.intersection(test_structs)
    overlap_train_val = train_structs.intersection(val_structs)
    overlap_val_test = val_structs.intersection(test_structs)

    assert len(overlap_train_test) == 0, f"Structure leakage between train and test: {overlap_train_test}"
    assert len(overlap_train_val) == 0, f"Structure leakage between train and val: {overlap_train_val}"
    assert len(overlap_val_test) == 0, f"Structure leakage between val and test: {overlap_val_test}"

    # 3. Assert index disjointness
    train_idx = set(split.train_indices)
    val_idx = set(split.val_indices)
    test_idx = set(split.test_indices)

    assert train_idx.isdisjoint(test_idx), "Index leakage between train and test"
    assert train_idx.isdisjoint(val_idx), "Index leakage between train and val"
    assert val_idx.isdisjoint(test_idx), "Index leakage between val and test"

    # 4. Assert all samples accounted for
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    # 5. Formal verification helper
    ok, diag = verify_no_leakage(split, df)
    assert ok, f"verify_no_leakage failed: {diag}"


def test_random_split_properties(sample_simulation_df):
    """Verify basic integrity of the baseline random split."""
    df = sample_simulation_df
    split = split_random(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42)

    assert split.split_strategy == "random"
    assert set(split.train_indices).isdisjoint(set(split.test_indices))
    assert set(split.train_indices).isdisjoint(set(split.val_indices))
    assert set(split.val_indices).isdisjoint(set(split.test_indices))
    assert split.n_train + split.n_val + split.n_test == len(df)


def test_split_serialization_roundtrip(sample_simulation_df, tmp_path: Path):
    """Verify saving and loading splits to JSON preserves exact index sets."""
    df = sample_simulation_df
    orig_split = split_held_out_earthquake(df, seed=42)
    save_path = tmp_path / "test_split.json"

    save_split(orig_split, save_path)
    loaded_split = load_split(save_path)

    assert orig_split.split_strategy == loaded_split.split_strategy
    assert orig_split.train_indices == loaded_split.train_indices
    assert orig_split.val_indices == loaded_split.val_indices
    assert orig_split.test_indices == loaded_split.test_indices
    assert orig_split.train_earthquakes == loaded_split.train_earthquakes
    assert orig_split.test_earthquakes == loaded_split.test_earthquakes


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
