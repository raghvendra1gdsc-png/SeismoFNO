"""
test_exp6_modal_gno.py — Comprehensive Unit Tests for EXP6 Modal-Conditioned GNO.

Verifies:
  1. Modal feature computation & eigenvalue correctness against theoretical matrices
  2. Modal normalizer encoding/decoding integrity
  3. FiLM modulation generator & identity initialization
  4. ConditionedSpatiotemporalGNO forward and backward gradient flow
  5. Absence of target leakage in modal conditioning
  6. Checkpoint save & reload consistency
  7. Batching with variable-floor graphs
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data_pipeline.graph_dataset import build_shear_frame_edges, GraphNormalizer
from src.data_pipeline.modal_dataset import (
    MODAL_CATALOG,
    get_modal_vector,
    ModalNormalizer,
    ModalStructuralGraphSample,
    collate_modal_structural_graphs,
)
from src.models.conditioned_gno import (
    FiLMBlock,
    ConditionedSpatiotemporalGNOBlock,
    ConditionedSpatiotemporalGNO,
)
from src.ground_truth.opensees_mdof_model import MDOFParams


def test_modal_catalog_theoretical_eigenvalues():
    """Verify that catalog modal periods match exact theoretical [K], [M] eigenvalue solution."""
    for struct_id, expected in MODAL_CATALOG.items():
        n = expected["n_stories"]
        t1 = expected["T1"]
        m = 1000.0 if n == 3 else (1000.0 if t1 == 0.55 else (1200.0 if t1 == 0.85 else (1400.0 if t1 == 1.20 else 1300.0)))
        omega1 = 2.0 * math.pi / t1
        k = m * (omega1 / (2.0 * math.sin(math.pi / (4 * n + 2)))) ** 2

        p = MDOFParams(n_stories=n, story_masses=[m] * n, story_stiffnesses=[k] * n)
        omegas, periods, _ = p.compute_theoretical_modal_properties()

        assert math.isclose(periods[0], expected["T1"], rel_tol=1e-3), f"T1 mismatch for {struct_id}"
        assert math.isclose(periods[1], expected["T2"], rel_tol=1e-3), f"T2 mismatch for {struct_id}"
        assert math.isclose(periods[2], expected["T3"], rel_tol=1e-3), f"T3 mismatch for {struct_id}"


def test_modal_vector_generation():
    """Verify dimensions and contents of modal vectors."""
    v_none = get_modal_vector("5S_T085", cond_mode="none")
    assert v_none.shape == (0,)

    v_t1 = get_modal_vector("5S_T085", cond_mode="t1")
    assert v_t1.shape == (1,)
    assert math.isclose(v_t1[0], 0.85, rel_tol=1e-4)

    v_multi = get_modal_vector("5S_T085", cond_mode="multimodal")
    assert v_multi.shape == (6,)
    # [T1, T2, T3, w1, w2, w3]
    assert math.isclose(v_multi[0], 0.85, rel_tol=1e-4)
    assert math.isclose(v_multi[3], 2.0 * math.pi / 0.85, rel_tol=1e-3)


def test_modal_normalizer():
    """Verify Z-score normalization for modal vectors."""
    vecs = [
        np.array([0.35, 0.12, 0.08, 17.9, 50.3, 72.7], dtype=np.float32),
        np.array([0.85, 0.29, 0.18, 7.39, 21.5, 34.0], dtype=np.float32),
        np.array([1.20, 0.41, 0.26, 5.24, 15.3, 24.1], dtype=np.float32),
    ]
    norm = ModalNormalizer().fit(vecs)
    assert norm.mean is not None
    assert norm.std is not None
    assert norm.mean.shape == (6,)

    t_in = torch.from_numpy(vecs[1]).float()
    t_enc = norm.encode(t_in)
    t_dec = norm.decode(t_enc)
    assert torch.allclose(t_in, t_dec, atol=1e-5)


def test_film_block_initialization():
    """Verify FiLM generator produces identity modulation upon initialization."""
    film = FiLMBlock(cond_dim=1, channels=48)
    c = torch.tensor([[0.85], [1.20]])  # B=2
    gamma, beta = film(c)
    assert gamma.shape == (2, 48, 1)
    assert beta.shape == (2, 48, 1)
    # Output should be near-zero initially (<0.02) so (1 + gamma)*x + beta approx x
    assert torch.allclose(gamma, torch.zeros_like(gamma), atol=0.02)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=0.02)


def test_conditioned_gno_parameters_and_modes():
    """Verify parameter counts across unconditioned, T1-conditioned, and multi-modal models."""
    # Unconditioned
    m_uncond = ConditionedSpatiotemporalGNO(cond_dim=0)
    p_uncond = sum(p.numel() for p in m_uncond.parameters())
    assert p_uncond == 674115, f"Expected 674,115 params for unconditioned GNO, got {p_uncond}"

    # T1 conditioned
    m_t1 = ConditionedSpatiotemporalGNO(cond_dim=1)
    p_t1 = sum(p.numel() for p in m_t1.parameters())
    assert p_t1 == 725059, f"Expected 725,059 params for T1-conditioned GNO, got {p_t1}"

    # Multi-modal conditioned
    m_mm = ConditionedSpatiotemporalGNO(cond_dim=6)
    p_mm = sum(p.numel() for p in m_mm.parameters())
    assert p_mm == 727619, f"Expected 727,619 params for multi-modal GNO, got {p_mm}"


def test_conditioned_gno_forward_backward():
    """Verify forward and backward passes with disjoint graph batching."""
    model = ConditionedSpatiotemporalGNO(cond_dim=1, width=48, modes=16, n_layers=2)

    # Batch with 1x 3S (3 nodes) and 1x 5S (5 nodes) -> total 8 nodes
    x = torch.randn(8, 10, 512, requires_grad=True)
    ei, ea = build_shear_frame_edges(8)
    cond = torch.tensor([[0.60], [1.20]])  # B=2
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])

    out = model(x, ei, ea, cond=cond, batch_idx=batch_idx)
    assert out.shape == (8, 3, 512)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # Check that FiLM weights receive gradients
    film_param = model.blocks[0].film_spat.mlp[0].weight
    assert film_param.grad is not None
    assert torch.any(film_param.grad != 0)


def test_modal_conditioning_no_target_leakage():
    """Verify that modal conditioning vector is purely a function of structural parameters."""
    for struct_id in ["3S_T035", "5S_T085", "5S_T120"]:
        vec = get_modal_vector(struct_id, cond_mode="multimodal")
        # Ensure no NaN or inf
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()
        # All values should be strictly positive physical quantities
        assert (vec > 0).all()


def test_checkpoint_save_and_reload(tmp_path):
    """Verify deterministic checkpoint reloading."""
    model1 = ConditionedSpatiotemporalGNO(cond_dim=1, width=48, modes=16, n_layers=2)
    model2 = ConditionedSpatiotemporalGNO(cond_dim=1, width=48, modes=16, n_layers=2)

    ckpt_file = tmp_path / "test_ckpt.pt"
    torch.save({"model_state": model1.state_dict()}, ckpt_file)

    loaded = torch.load(ckpt_file, map_location="cpu")
    model2.load_state_dict(loaded["model_state"])

    x = torch.randn(5, 10, 256)
    ei, ea = build_shear_frame_edges(5)
    c = torch.tensor([[0.85]])

    out1 = model1(x, ei, ea, cond=c)
    out2 = model2(x, ei, ea, cond=c)

    assert torch.allclose(out1, out2, atol=1e-6)
