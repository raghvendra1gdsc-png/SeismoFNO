"""
test_exp2_preflight.py — Pre-Flight Verification Tests for EXP 2 Readiness.
"""

import json
import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline.splits import load_split
from src.data_pipeline.dataset_builder import SeismicSDOFDataset
from src.evaluation.metrics import compute_clustered_bootstrap_ci
from src.evaluation.linear_probe import LatentPlasticStateProbe
from src.models.fno1d import FNO1d
from src.models.causal_tcn import CausalTCN
from src.models.state_augmented_tcn import StateAugmentedCausalTCN
from src.models.s4_operator import S4Operator


def test_gate0_cardinalities():
    """Verify exact Gate 0 bilinear dataset cardinalities (5,740 / 1,120 / 1,540)."""
    df = pd.read_csv("data/simulations/simulation_index.csv")
    df_bilinear = df[df["material_type"] == "bilinear"].reset_index(drop=True)
    split = load_split("data/processed/splits/held_out_earthquake_split.json")

    train_df = df_bilinear[df_bilinear["sim_id"].isin(set(split.train_ids))]
    val_df = df_bilinear[df_bilinear["sim_id"].isin(set(split.val_ids))]
    test_df = df_bilinear[df_bilinear["sim_id"].isin(set(split.test_ids))]

    assert len(train_df) == 5740, f"Expected 5,740 train samples, got {len(train_df)}"
    assert len(val_df) == 1120, f"Expected 1,120 val samples, got {len(val_df)}"
    assert len(test_df) == 1540, f"Expected 1,540 test samples, got {len(test_df)}"

    assert train_df["earthquake_name"].nunique() == 11, f"Expected 11 train earthquakes, got {train_df['earthquake_name'].nunique()}"
    assert val_df["earthquake_name"].nunique() == 2, f"Expected 2 val earthquakes, got {val_df['earthquake_name'].nunique()}"
    assert test_df["earthquake_name"].nunique() == 3, f"Expected 3 test earthquakes, got {test_df['earthquake_name'].nunique()}"

    # Disjointness check
    train_eqs = set(train_df["earthquake_name"].unique())
    val_eqs = set(val_df["earthquake_name"].unique())
    test_eqs = set(test_df["earthquake_name"].unique())

    assert train_eqs.isdisjoint(val_eqs), "Train and Val earthquakes overlap!"
    assert train_eqs.isdisjoint(test_eqs), "Train and Test earthquakes overlap!"
    assert val_eqs.isdisjoint(test_eqs), "Val and Test earthquakes overlap!"


def test_clustered_bootstrap_execution():
    """Verify clustered bootstrap computation across clusters."""
    dummy_records = [
        {"err": 10.0, "earthquake": "EQ1"},
        {"err": 12.0, "earthquake": "EQ1"},
        {"err": 20.0, "earthquake": "EQ2"},
        {"err": 22.0, "earthquake": "EQ2"},
        {"err": 30.0, "earthquake": "EQ3"},
    ]
    mean_val, low, high = compute_clustered_bootstrap_ci(
        dummy_records, metric_key="err", cluster_key="earthquake", n_boot=200, seed=42
    )

    assert low <= mean_val <= high, f"Invalid CI: {low} <= {mean_val} <= {high}"


def test_latent_probe_zero_leakage():
    """Verify linear probe fit on train and evaluate on test."""
    n_train, n_test, d_latent, l_seq = 20, 10, 128, 64
    rng = np.random.default_rng(42)

    h_train = rng.standard_normal((n_train, d_latent, l_seq)).astype(np.float32)
    # Synthetic target with linear relation to feature channel 0
    up_train = h_train[:, 0, :] * 2.5 + rng.standard_normal((n_train, l_seq)) * 0.1

    h_test = rng.standard_normal((n_test, d_latent, l_seq)).astype(np.float32)
    up_test = h_test[:, 0, :] * 2.5 + rng.standard_normal((n_test, l_seq)) * 0.1

    probe = LatentPlasticStateProbe(alpha=1.0)
    probe.fit(h_train, up_train)
    results = probe.evaluate(h_test, up_test)

    assert "r2_score" in results
    assert "rmse_mm" in results
    assert results["r2_score"] > 0.90, f"Expected high R2 on synthetic linear data, got {results['r2_score']}"


def test_five_model_parameter_matching():
    """Verify all 5 models in the EXP 2 matrix are within +-1.5% of target budget (~1.192M)."""
    target = 1192448
    models = {
        "Standard FNO": FNO1d(in_channels=10, out_channels=3, modes=128, width=48, n_layers=4),
        "Causal TCN": CausalTCN(in_channels=10, out_channels=3, num_channels=[136]*11, kernel_size=3),
        "State-Augmented TCN": StateAugmentedCausalTCN(in_channels=10, out_channels=3, num_channels=[136]*11, state_dim=4),
        "Continuous S4 SSM": S4Operator(in_channels=10, out_channels=3, d_model=128, d_state=64, n_blocks=6, d_ff=578),
        "Memory-Truncated S4": S4Operator(in_channels=10, out_channels=3, d_model=128, d_state=64, n_blocks=6, d_ff=578, memory_truncated=True),
    }

    for name, m in models.items():
        p = m.get_num_parameters() if hasattr(m, "get_num_parameters") else sum(param.numel() for param in m.parameters() if param.requires_grad)
        delta_pct = abs(p - target) / target * 100.0
        assert delta_pct < 1.5, f"{name} param count {p} deviates by {delta_pct:.2f}% from target {target} (limit 1.5%)"
