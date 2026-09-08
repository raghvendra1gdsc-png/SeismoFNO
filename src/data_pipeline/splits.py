"""
splits.py — Dataset Splitting Strategies for SeismoFNO.

Implements three rigorous partitioning strategies per AGENTS.md:
  1. random:
     Baseline random partition.
     WARNING: Per AGENTS.md Hard Rule 2, NEVER evaluate or claim "zero-shot generalization"
     using this split.

  2. held_out_earthquake:
     Partitions by distinct earthquake events (e.g. Kobe, Chi-Chi, Northridge).
     All ground motions and scaled variants from test/validation earthquakes NEVER appear
     in the training set.

  3. held_out_structure:
     Partitions by structural configurations (periods T, yield displacements u_y, alpha).
     All responses for held-out structures NEVER appear in the training set.

Guarantees zero leakage with formal verification assertions.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from pathlib import Path
import json
import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    """Dataclass holding indices, identifiers, and metadata for a dataset split."""
    split_strategy: str                       # 'random', 'held_out_earthquake', 'held_out_structure'
    train_indices: List[int]                  # Row indices in dataframe
    val_indices: List[int]
    test_indices: List[int]
    train_ids: List[str]                      # sim_id or record_id
    val_ids: List[str]
    test_ids: List[str]
    train_earthquakes: List[str]
    val_earthquakes: List[str]
    test_earthquakes: List[str]
    train_structures: List[str]
    val_structures: List[str]
    test_structures: List[str]
    n_train: int
    n_val: int
    n_test: int
    total: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize split result to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplitResult":
        """Reconstruct split result from dictionary."""
        return cls(**data)


def split_random(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    id_col: str = "sim_id",
) -> SplitResult:
    """
    Standard randomized train/val/test split.

    WARNING: For baseline comparison only. Must NEVER be used for generalization claims.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}")

    n = len(df)
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(n)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_idx = shuffled_idx[:n_train].tolist()
    val_idx = shuffled_idx[n_train : n_train + n_val].tolist()
    test_idx = shuffled_idx[n_train + n_val :].tolist()

    id_key = id_col if id_col in df.columns else df.columns[0]
    eq_col = "earthquake_name" if "earthquake_name" in df.columns else None
    struct_col = "struct_id" if "struct_id" in df.columns else None

    return SplitResult(
        split_strategy="random",
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        train_ids=df.iloc[train_idx][id_key].tolist(),
        val_ids=df.iloc[val_idx][id_key].tolist(),
        test_ids=df.iloc[test_idx][id_key].tolist(),
        train_earthquakes=sorted(df.iloc[train_idx][eq_col].unique().tolist()) if eq_col else [],
        val_earthquakes=sorted(df.iloc[val_idx][eq_col].unique().tolist()) if eq_col else [],
        test_earthquakes=sorted(df.iloc[test_idx][eq_col].unique().tolist()) if eq_col else [],
        train_structures=sorted(df.iloc[train_idx][struct_col].unique().tolist()) if struct_col else [],
        val_structures=sorted(df.iloc[val_idx][struct_col].unique().tolist()) if struct_col else [],
        test_structures=sorted(df.iloc[test_idx][struct_col].unique().tolist()) if struct_col else [],
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        total=n,
    )


def split_held_out_earthquake(
    df: pd.DataFrame,
    held_out_test_eqs: Optional[List[str]] = None,
    held_out_val_eqs: Optional[List[str]] = None,
    test_eq_frac: float = 0.20,
    val_eq_frac: float = 0.10,
    seed: int = 42,
    eq_col: str = "earthquake_name",
    id_col: str = "sim_id",
) -> SplitResult:
    """
    Partition dataset by whole earthquake events.

    Guarantees that test and validation earthquake events are completely unseen
    during training (zero earthquake overlap).
    """
    if eq_col not in df.columns:
        raise ValueError(f"Column '{eq_col}' not found in dataframe.")

    all_eqs = sorted(df[eq_col].unique().tolist())
    n_eqs = len(all_eqs)

    if held_out_test_eqs is not None:
        test_eqs = set(held_out_test_eqs)
        for eq in test_eqs:
            if eq not in all_eqs:
                raise ValueError(f"Specified test earthquake '{eq}' not found in data.")
    else:
        rng = np.random.default_rng(seed)
        n_test_eqs = max(1, int(round(test_eq_frac * n_eqs)))
        test_eqs = set(rng.choice(all_eqs, size=n_test_eqs, replace=False).tolist())

    remaining_eqs = [eq for eq in all_eqs if eq not in test_eqs]

    if held_out_val_eqs is not None:
        val_eqs = set(held_out_val_eqs)
        for eq in val_eqs:
            if eq not in remaining_eqs:
                raise ValueError(f"Validation earthquake '{eq}' overlaps with test or not in data.")
    else:
        rng = np.random.default_rng(seed + 1)
        n_val_eqs = max(1, int(round(val_eq_frac * n_eqs)))
        val_eqs = set(rng.choice(remaining_eqs, size=n_val_eqs, replace=False).tolist())

    train_eqs = [eq for eq in remaining_eqs if eq not in val_eqs]

    # Verify set disjointness
    assert set(train_eqs).isdisjoint(test_eqs), "Leakage: train and test earthquakes overlap!"
    assert set(train_eqs).isdisjoint(val_eqs), "Leakage: train and val earthquakes overlap!"
    assert val_eqs.isdisjoint(test_eqs), "Leakage: val and test earthquakes overlap!"

    # Map to dataframe indices
    train_mask = df[eq_col].isin(train_eqs)
    val_mask = df[eq_col].isin(val_eqs)
    test_mask = df[eq_col].isin(test_eqs)

    train_idx = df[train_mask].index.tolist()
    val_idx = df[val_mask].index.tolist()
    test_idx = df[test_mask].index.tolist()

    id_key = id_col if id_col in df.columns else df.columns[0]
    struct_col = "struct_id" if "struct_id" in df.columns else None

    return SplitResult(
        split_strategy="held_out_earthquake",
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        train_ids=df.iloc[train_idx][id_key].tolist(),
        val_ids=df.iloc[val_idx][id_key].tolist(),
        test_ids=df.iloc[test_idx][id_key].tolist(),
        train_earthquakes=sorted(train_eqs),
        val_earthquakes=sorted(list(val_eqs)),
        test_earthquakes=sorted(list(test_eqs)),
        train_structures=sorted(df.iloc[train_idx][struct_col].unique().tolist()) if struct_col else [],
        val_structures=sorted(df.iloc[val_idx][struct_col].unique().tolist()) if struct_col else [],
        test_structures=sorted(df.iloc[test_idx][struct_col].unique().tolist()) if struct_col else [],
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        total=len(df),
    )


def split_held_out_structure(
    df: pd.DataFrame,
    held_out_test_structs: Optional[List[str]] = None,
    held_out_val_structs: Optional[List[str]] = None,
    test_struct_frac: float = 0.20,
    val_struct_frac: float = 0.10,
    seed: int = 42,
    struct_col: str = "struct_id",
    id_col: str = "sim_id",
) -> SplitResult:
    """
    Partition dataset by structural system configurations.

    Guarantees that test and validation structures (periods, yield displacement, ductility)
    are completely unseen during training (zero structure overlap).
    """
    if struct_col not in df.columns:
        raise ValueError(f"Column '{struct_col}' not found in dataframe.")

    all_structs = sorted(df[struct_col].unique().tolist())
    n_structs = len(all_structs)

    if held_out_test_structs is not None:
        test_structs = set(held_out_test_structs)
        for s in test_structs:
            if s not in all_structs:
                raise ValueError(f"Specified test structure '{s}' not found in data.")
    else:
        rng = np.random.default_rng(seed)
        n_test_structs = max(1, int(round(test_struct_frac * n_structs)))
        test_structs = set(rng.choice(all_structs, size=n_test_structs, replace=False).tolist())

    remaining_structs = [s for s in all_structs if s not in test_structs]

    if held_out_val_structs is not None:
        val_structs = set(held_out_val_structs)
        for s in val_structs:
            if s not in remaining_structs:
                raise ValueError(f"Validation structure '{s}' overlaps with test or not in data.")
    else:
        rng = np.random.default_rng(seed + 1)
        n_val_structs = max(1, int(round(val_struct_frac * n_structs)))
        val_structs = set(rng.choice(remaining_structs, size=n_val_structs, replace=False).tolist())

    train_structs = [s for s in remaining_structs if s not in val_structs]

    # Verify set disjointness
    assert set(train_structs).isdisjoint(test_structs), "Leakage: train and test structures overlap!"
    assert set(train_structs).isdisjoint(val_structs), "Leakage: train and val structures overlap!"
    assert val_structs.isdisjoint(test_structs), "Leakage: val and test structures overlap!"

    # Map to dataframe indices
    train_mask = df[struct_col].isin(train_structs)
    val_mask = df[struct_col].isin(val_structs)
    test_mask = df[struct_col].isin(test_structs)

    train_idx = df[train_mask].index.tolist()
    val_idx = df[val_mask].index.tolist()
    test_idx = df[test_mask].index.tolist()

    id_key = id_col if id_col in df.columns else df.columns[0]
    eq_col = "earthquake_name" if "earthquake_name" in df.columns else None

    return SplitResult(
        split_strategy="held_out_structure",
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        train_ids=df.iloc[train_idx][id_key].tolist(),
        val_ids=df.iloc[val_idx][id_key].tolist(),
        test_ids=df.iloc[test_idx][id_key].tolist(),
        train_earthquakes=sorted(df.iloc[train_idx][eq_col].unique().tolist()) if eq_col else [],
        val_earthquakes=sorted(df.iloc[val_idx][eq_col].unique().tolist()) if eq_col else [],
        test_earthquakes=sorted(df.iloc[test_idx][eq_col].unique().tolist()) if eq_col else [],
        train_structures=sorted(train_structs),
        val_structures=sorted(list(val_structs)),
        test_structures=sorted(list(test_structs)),
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        total=len(df),
    )


def verify_no_leakage(split: SplitResult, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    r"""
    Formally assert zero leakage between train, validation, and test partitions.

    Checks:
      1. Train/Val/Test index disjointness: Train \cap Test = \emptyset, Train \cap Val = \emptyset.
      2. For held_out_earthquake: Train Earthquakes \cap (Val U Test Earthquakes) = \emptyset.
      3. For held_out_structure: Train Structures \cap (Val U Test Structures) = \emptyset.
    """
    train_idx = set(split.train_indices)
    val_idx = set(split.val_indices)
    test_idx = set(split.test_indices)

    # 1. Index checks
    idx_train_test_overlap = train_idx.intersection(test_idx)
    idx_train_val_overlap = train_idx.intersection(val_idx)
    idx_val_test_overlap = val_idx.intersection(test_idx)

    passed = True
    diagnostics: Dict[str, Any] = {
        "index_overlap_train_test": len(idx_train_test_overlap),
        "index_overlap_train_val": len(idx_train_val_overlap),
        "index_overlap_val_test": len(idx_val_test_overlap),
    }

    if idx_train_test_overlap or idx_train_val_overlap or idx_val_test_overlap:
        passed = False

    # 2. Earthquake leakage checks
    if split.split_strategy == "held_out_earthquake":
        train_eqs = set(split.train_earthquakes)
        val_eqs = set(split.val_earthquakes)
        test_eqs = set(split.test_earthquakes)
        
        eq_leak_test = train_eqs.intersection(test_eqs)
        eq_leak_val = train_eqs.intersection(val_eqs)
        diagnostics["earthquake_leakage_train_test"] = list(eq_leak_test)
        diagnostics["earthquake_leakage_train_val"] = list(eq_leak_val)
        
        if eq_leak_test or eq_leak_val:
            passed = False

    # 3. Structure leakage checks
    if split.split_strategy == "held_out_structure":
        train_structs = set(split.train_structures)
        val_structs = set(split.val_structures)
        test_structs = set(split.test_structures)
        
        struct_leak_test = train_structs.intersection(test_structs)
        struct_leak_val = train_structs.intersection(val_structs)
        diagnostics["structure_leakage_train_test"] = list(struct_leak_test)
        diagnostics["structure_leakage_train_val"] = list(struct_leak_val)
        
        if struct_leak_test or struct_leak_val:
            passed = False

    diagnostics["passed"] = passed
    return passed, diagnostics


def save_split(split: SplitResult, output_path: Union[str, Path]) -> None:
    """Save split result to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2)


def load_split(input_path: Union[str, Path]) -> SplitResult:
    """Load split result from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SplitResult.from_dict(data)


def generate_and_save_all_splits(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "data/processed/splits",
) -> Dict[str, SplitResult]:
    """Generate all three canonical splits and persist to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "random": split_random(df, seed=42),
        "held_out_earthquake": split_held_out_earthquake(df, seed=42),
        "held_out_structure": split_held_out_structure(df, seed=42),
    }

    for name, s in splits.items():
        ok, diag = verify_no_leakage(s, df)
        assert ok, f"Leakage verification failed for split '{name}': {diag}"
        save_split(s, out_path / f"{name}_split.json")
        print(f"Saved {name} split -> {out_path / f'{name}_split.json'} (Train={s.n_train}, Val={s.n_val}, Test={s.n_test})")

    return splits
