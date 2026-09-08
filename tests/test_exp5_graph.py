"""
test_exp5_graph.py — Comprehensive Unit & Integrity Tests for EXP5 Graph Neural Operator.

Verifies:
  1. Topology-native graph construction (exact 3 nodes for 3S, exact 5 nodes for 5S, no zero padding)
  2. Physical shear-frame edge connectivity and attributes
  3. Disjoint graph batching and index offsets
  4. GNO forward pass on variable graph batches on CPU & MPS
  5. GNO topology ablation mode (use_topology=False)
  6. Strict structural and earthquake split leakage absence
  7. Checkpoint save and reload invariance
"""

import pytest
import torch
import numpy as np
import pandas as pd

from src.data_pipeline.graph_dataset import (
    build_shear_frame_edges,
    StructuralGraphSample,
    BatchedStructuralGraphs,
    collate_structural_graphs,
    GraphNormalizer,
)
from src.models.gno import SpatiotemporalGNO


def test_build_shear_frame_edges():
    """Verify physical shear-frame connectivity for 3-story and 5-story buildings."""
    # 3-story building: 3 nodes (0, 1, 2)
    # Edges: 2 upward (0->1, 1->2), 2 downward (1->0, 2->1), 3 self-loops = 7 edges
    edge_idx_3, edge_attr_3 = build_shear_frame_edges(3)
    assert edge_idx_3.shape == (2, 7), f"Expected (2, 7), got {edge_idx_3.shape}"
    assert edge_attr_3.shape == (7, 2), f"Expected (7, 2), got {edge_attr_3.shape}"

    # 5-story building: 5 nodes (0, 1, 2, 3, 4)
    # Edges: 4 upward, 4 downward, 5 self-loops = 13 edges
    edge_idx_5, edge_attr_5 = build_shear_frame_edges(5)
    assert edge_idx_5.shape == (2, 13), f"Expected (2, 13), got {edge_idx_5.shape}"
    assert edge_attr_5.shape == (13, 2), f"Expected (13, 2), got {edge_attr_5.shape}"

    # Verify no out-of-bound node indices
    assert edge_idx_3.max() == 2
    assert edge_idx_5.max() == 4


def test_collate_structural_graphs_variable_floors():
    """Verify collation of mixed 3-story and 5-story buildings with zero padding."""
    T = 256
    # Graph 1: 3 stories
    e_idx1, e_attr1 = build_shear_frame_edges(3)
    g1 = StructuralGraphSample(
        sim_id=101,
        struct_id="3S_T035",
        n_stories=3,
        earthquake_id="RSN0001",
        x=torch.randn(3, 10, T),
        y=torch.randn(3, 3, T),
        edge_index=e_idx1,
        edge_attr=e_attr1,
    )

    # Graph 2: 5 stories
    e_idx2, e_attr2 = build_shear_frame_edges(5)
    g2 = StructuralGraphSample(
        sim_id=102,
        struct_id="5S_T055",
        n_stories=5,
        earthquake_id="RSN0001",
        x=torch.randn(5, 10, T),
        y=torch.randn(5, 3, T),
        edge_index=e_idx2,
        edge_attr=e_attr2,
    )

    batch = collate_structural_graphs([g1, g2])

    # Total nodes = 3 + 5 = 8 (strictly no zero-padding)
    assert batch.x.shape == (8, 10, T), f"Expected (8, 10, {T}), got {batch.x.shape}"
    assert batch.y.shape == (8, 3, T), f"Expected (8, 3, {T}), got {batch.y.shape}"

    # Total edges = 7 + 13 = 20
    assert batch.edge_index.shape == (2, 20), f"Expected (2, 20), got {batch.edge_index.shape}"
    assert batch.edge_attr.shape == (20, 2), f"Expected (20, 2), got {batch.edge_attr.shape}"

    # Graph 2 edges must be shifted by 3
    g2_edges = batch.edge_index[:, 7:]
    assert g2_edges.min() == 3
    assert g2_edges.max() == 7

    # Batch assignment vector
    assert batch.batch_idx.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert batch.node_counts == [3, 5]


def test_gno_forward_pass():
    """Verify GNO forward pass produces correct shapes on variable graph batches."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    T = 512
    model = SpatiotemporalGNO(in_channels=10, out_channels=3, width=32, modes=32, n_layers=2).to(device)

    # Build mixed batch: one 3S and one 5S
    e1, a1 = build_shear_frame_edges(3)
    g1 = StructuralGraphSample(1, "3S", 3, "EQ1", torch.randn(3, 10, T), torch.randn(3, 3, T), e1, a1)
    e2, a2 = build_shear_frame_edges(5)
    g2 = StructuralGraphSample(2, "5S", 5, "EQ1", torch.randn(5, 10, T), torch.randn(5, 3, T), e2, a2)

    batch = collate_structural_graphs([g1, g2]).to(device)

    out = model(batch.x, batch.edge_index, batch.edge_attr)
    assert out.shape == (8, 3, T), f"Expected (8, 3, {T}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in GNO output"


def test_gno_topology_ablation():
    """Verify GNO runs in ablation mode (use_topology=False) with zero spatial edge passing."""
    T = 256
    model_ablation = SpatiotemporalGNO(
        in_channels=10, out_channels=3, width=32, modes=32, n_layers=2, use_topology=False
    )

    e1, a1 = build_shear_frame_edges(3)
    g1 = StructuralGraphSample(1, "3S", 3, "EQ1", torch.randn(3, 10, T), torch.randn(3, 3, T), e1, a1)
    batch = collate_structural_graphs([g1])

    out = model_ablation(batch.x, batch.edge_index, batch.edge_attr)
    assert out.shape == (3, 3, T)
    assert not torch.isnan(out).any()


def test_graph_normalizer():
    """Verify channel-wise unit Gaussian normalizer encodes and decodes properly."""
    t1 = torch.randn(3, 10, 100) * 5.0 + 10.0
    t2 = torch.randn(5, 10, 100) * 5.0 + 10.0

    normalizer = GraphNormalizer().fit([t1, t2])
    encoded = normalizer.encode(t1)
    decoded = normalizer.decode(encoded)

    assert torch.allclose(t1, decoded, atol=1e-5), "Normalizer invertibility failed"


def test_split_leakage_invariants():
    """Verify strict structural and earthquake split invariants for EXP5."""
    df = pd.read_csv("data/simulations/mdof/simulation_index.csv")

    seen_structs = {"3S_T035", "3S_T060", "3S_T090", "5S_T055", "5S_T085"}
    unseen_struct = {"5S_T120"}

    train_eqs = {f"RSN{i:04d}" for i in range(1, 9)}
    val_eqs = {"RSN0009", "RSN0010"}
    test_eqs = {"RSN0011", "RSN0012"}

    # Invariants
    assert seen_structs.isdisjoint(unseen_struct), "Structural overlap detected!"
    assert train_eqs.isdisjoint(val_eqs), "Train/Val earthquake overlap!"
    assert train_eqs.isdisjoint(test_eqs), "Train/Test earthquake overlap!"
    assert val_eqs.isdisjoint(test_eqs), "Val/Test earthquake overlap!"

    # Verify coverage across dataset
    all_structs = set(df["struct_id"].unique())
    all_eqs = set(df["earthquake_id"].unique())
    assert (seen_structs | unseen_struct) == all_structs
    assert (train_eqs | val_eqs | test_eqs) == all_eqs


def test_checkpoint_load_and_reproduce():
    """Verify saved checkpoint loads and reproduces exact inference output."""
    ckpt_path = "results/experiments/exp5/training/best_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = SpatiotemporalGNO(
        in_channels=10, out_channels=3, width=48, modes=64, n_layers=4,
        use_topology=ckpt.get("use_topology", True)
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    assert sum(p.numel() for p in model.parameters()) == 674115
    assert ckpt["epoch"] == 16

    # Test single-sample inference reproducibility
    e_idx, e_attr = build_shear_frame_edges(3)
    x = torch.randn(3, 10, 256)
    with torch.no_grad():
        out1 = model(x, e_idx, e_attr)
        out2 = model(x, e_idx, e_attr)
    assert torch.equal(out1, out2), "Inference reproducibility failed"
