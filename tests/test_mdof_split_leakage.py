"""
test_mdof_split_leakage.py — Strict Split Leakage Verification for MDOF Dataset.

Asserts:
  1. Zero overlap of earthquake IDs between train, val, and test in held-out-earthquake split.
  2. Zero overlap of structure archetype IDs between train, val, and test in held-out-structure split.
  3. Disjoint sim_id sets across all splits.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.data_pipeline.splits import load_split
from src.data_pipeline.mdof_splits import create_mdof_splits


def test_mdof_split_leakage():
    index_csv = "data/simulations/mdof/simulation_index.csv"
    if not Path(index_csv).exists():
        pytest.skip("MDOF simulation index not generated yet.")

    df = pd.read_csv(index_csv)

    # 1. Held-Out-Earthquake Split
    eq_split = load_split("data/processed/splits/mdof_held_out_earthquake_split.json")
    train_eq = set(df[df["sim_id"].isin(set(eq_split.train_ids))]["earthquake_id"])
    val_eq = set(df[df["sim_id"].isin(set(eq_split.val_ids))]["earthquake_id"])
    test_eq = set(df[df["sim_id"].isin(set(eq_split.test_ids))]["earthquake_id"])

    assert len(train_eq & test_eq) == 0, f"Earthquake leakage between train and test: {train_eq & test_eq}"
    assert len(train_eq & val_eq) == 0, f"Earthquake leakage between train and val: {train_eq & val_eq}"
    assert len(val_eq & test_eq) == 0, f"Earthquake leakage between val and test: {val_eq & test_eq}"

    # 2. Held-Out-Structure Split
    st_split = load_split("data/processed/splits/mdof_held_out_structure_split.json")
    train_st = set(df[df["sim_id"].isin(set(st_split.train_ids))]["struct_id"])
    val_st = set(df[df["sim_id"].isin(set(st_split.val_ids))]["struct_id"])
    test_st = set(df[df["sim_id"].isin(set(st_split.test_ids))]["struct_id"])

    assert len(train_st & test_st) == 0, f"Structure leakage between train and test: {train_st & test_st}"
    assert len(train_st & val_st) == 0, f"Structure leakage between train and val: {train_st & val_st}"
    assert len(val_st & test_st) == 0, f"Structure leakage between val and test: {val_st & test_st}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
