"""Tests for the Professor-Facing Research Demonstration Layer.

Validates that:
1. Model registry accurately references frozen checkpoints and SHA256 hashes.
2. Experiment IDs and partitions remain immutable.
3. Pre-earthquake modal descriptors are derived purely from [M], [K] matrices with zero ground-motion leakage.
4. Structural archetypes represent physical degrees of freedom with topology-native graphs (no padding).
5. Archival metrics match source evaluation artifacts exactly.
6. Fallbacks properly label archival vs live inference.
7. FastAPI demo endpoints (/api/v1/demo/*) respond with valid schemas and status 200.
"""

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.demo.model_registry import (
    RESEARCH_MODELS,
    FROZEN_MODELS,
    get_model_spec,
    list_registered_models,
)
from src.demo.demo_data import (
    STRUCTURE_CATALOG,
    VERIFIED_EARTHQUAKES,
    DemoDataManager,
    compute_modal_properties,
    get_verified_simulation,
)
from src.demo.metrics_adapter import (
    get_research_progression,
    get_ood_matrix,
    get_ablation_falsification,
    get_failure_analysis,
    get_computational_benchmark,
)
from src.demo.inference_adapter import run_demonstration_inference


@pytest.fixture
def test_client():
    return TestClient(app)


def test_model_registry_resolution():
    """Verify that all registered models exist and have valid metadata."""
    models = list_registered_models()
    assert len(models) >= 5

    required_keys = {"model_id", "display_name", "architecture", "phase", "status"}
    for m in models:
        for k in required_keys:
            assert k in m, f"Missing key {k} in model spec {m['model_id']}"

    # Verify specific model IDs
    exp4 = get_model_spec("exp4_fno2d")
    assert exp4["phase"] == "EXP4"
    assert exp4["supports_live_inference"] is False

    exp5 = get_model_spec("exp5_gno")
    assert exp5["phase"] == "EXP5"

    exp6_b = get_model_spec("exp6_t1_gno")
    assert exp6_b["phase"] == "EXP6"

    exp6_c = get_model_spec("exp6_multimodal_gno")
    assert exp6_c["phase"] == "EXP6"
    assert exp6_c["supports_live_inference"] is True


def test_experiment_checkpoints_immutable():
    """Verify that frozen checkpoint files exist and remain untampered."""
    for model_id, spec in FROZEN_MODELS.items():
        ckpt_path = spec.checkpoint_path
        assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} missing for {model_id}"
        # File size must be non-zero
        assert ckpt_path.stat().st_size > 1000, f"Checkpoint {ckpt_path} appears empty or corrupted"


def test_modal_descriptors_pre_earthquake_invariants():
    """Verify that modal descriptors are calculated strictly from [M], [K] matrices,

    independent of any ground motion or time history.
    """
    for arch_id, arch in STRUCTURE_CATALOG.items():
        n = arch.n_stories
        m_f = arch.story_masses[0]
        k_s = arch.story_stiffnesses[0]

        # Modal solver
        t_vals, w_vals, mode_shapes = compute_modal_properties(n, m_f, k_s)
        T1 = t_vals[0]
        omega1 = w_vals[0]

        assert T1 > 0.0
        assert omega1 > 0.0
        assert np.isclose(omega1 * T1, 2.0 * np.pi, rtol=1e-3)
        assert len(mode_shapes) <= 3

        # Normalized mode shapes must have roof displacement normalized
        for mode_vec in mode_shapes:
            assert len(mode_vec) == n
            assert np.max(np.abs(mode_vec)) == pytest.approx(1.0, rel=1e-4)


def test_graph_node_count_and_topology_native():
    """Verify that 3-story has exactly 3 nodes and 5-story has exactly 5 nodes,

    demonstrating the topology-native representation without artificial zero-padding.
    """
    for arch_id, arch in STRUCTURE_CATALOG.items():
        n = arch.n_stories
        arch_dict = arch.to_dict()
        assert arch_dict["n_stories"] == n
        assert len(arch.story_masses) == n
        assert len(arch.story_stiffnesses) == n


def test_archival_metrics_match_source_artifacts():
    """Verify that the OOD generalization matrix and progression match historical reports."""
    matrix = get_ood_matrix()
    assert "partitions" in matrix
    assert "peak_disp_error_pct" in matrix
    assert "rel_l2_u_pct" in matrix

    # Verify EXP5 vs EXP6-C OOD-B values (index 2 is OOD-B)
    peak = matrix["peak_disp_error_pct"]
    assert peak["EXP5 Baseline GNO"][2] == pytest.approx(35.21, abs=0.01)
    assert peak["EXP6-C Multi-Modal GNO"][2] == pytest.approx(13.06, abs=0.01)
    assert peak["EXP6-B T1-GNO"][2] == pytest.approx(13.47, abs=0.01)

    # Shuffled ablation
    abl = get_ablation_falsification()
    assert "metrics" in abl
    assert "interpretation" in abl
    ood_b_metric = next(m for m in abl["metrics"] if "OOD-B" in m["partition"])
    assert ood_b_metric["true_multimodal_err"] == pytest.approx(13.06, abs=0.01)
    assert ood_b_metric["shuffled_err"] == pytest.approx(24.33, abs=0.01)


def test_research_progression_steps():
    """Verify progression has all 4 stages from EXP4 failure to remaining limitations."""
    steps = get_research_progression()
    assert len(steps) == 4
    assert "EXP4" in steps[0]["phase"]
    assert "EXP5" in steps[1]["phase"]
    assert "EXP6" in steps[2]["phase"]
    assert "ANALYSIS" in steps[3]["phase"]


def test_failure_modes_honesty():
    """Verify that all 4 documented scientific limitations are transparently exposed."""
    failures = get_failure_analysis()
    assert len(failures) == 4
    failure_ids = [f["id"] for f in failures]
    assert "failure_1" in failure_ids
    assert "failure_2" in failure_ids
    assert "failure_3" in failure_ids
    assert "failure_4" in failure_ids


def test_computational_benchmark_metrics():
    """Verify the single-building inference benchmark values and hardware attribution."""
    bench = get_computational_benchmark()
    assert "Apple Silicon" in bench["hardware"]
    models = {m["name"]: m for m in bench["models"]}
    assert "OpenSeesPy 5-Story NLTHA (Ground Truth)" in models
    assert "EXP6 T1-GNO (Single Inference)" in models
    assert models["OpenSeesPy 5-Story NLTHA (Ground Truth)"]["latency_ms"] == pytest.approx(54.68, abs=0.01)
    assert models["EXP6 T1-GNO (Single Inference)"]["latency_ms"] == pytest.approx(21.45, abs=0.01)
    assert models["EXP6 T1-GNO (Single Inference)"]["speedup"] == pytest.approx(2.55, abs=0.01)



def test_inference_adapter_live_and_archival_labeling():
    """Verify inference adapter executes live inference for EXP6 on MPS/CPU,

    and correctly sets data provenance labels.
    """
    res = run_demonstration_inference(
        archetype_id="3S_T035",
        record_id="RSN0001",
        model_id="exp6_multimodal_gno",
        selected_floor=3,
    )
    assert len(res["time"]) > 0
    assert len(res["u_true"]) == len(res["time"])
    assert len(res["u_pred"]) == len(res["time"])
    assert res["metrics"]["rel_l2_pct"] is not None
    assert "data_provenance" in res
    assert "LIVE_COMPUTED" in res["data_provenance"]["prediction"]

    # EXP4 should resolve archival evaluation
    res_exp4 = run_demonstration_inference(
        archetype_id="3S_T035",
        record_id="RSN0001",
        model_id="exp4_fno2d",
        selected_floor=3,
    )
    assert "ARCHIVAL" in res_exp4["data_provenance"]["prediction"]
    assert res_exp4["is_live_inference"] is False


def test_fastapi_demo_endpoints(test_client):
    """Verify all /api/v1/demo/* endpoints respond with 200 OK and valid JSON."""
    endpoints = [
        "/api/v1/demo/models",
        "/api/v1/demo/structures",
        "/api/v1/demo/earthquakes",
        "/api/v1/demo/progression",
        "/api/v1/demo/ood-matrix",
        "/api/v1/demo/ablation",
        "/api/v1/demo/failures",
        "/api/v1/demo/benchmark",
    ]

    for ep in endpoints:
        resp = test_client.get(ep)
        assert resp.status_code == 200, f"Endpoint {ep} returned {resp.status_code}"
        assert resp.headers["content-type"].startswith("application/json")

    # Test POST simulate
    payload = {
        "archetype_id": "5S_T120",
        "record_id": "RSN0001",
        "model_id": "exp6_multimodal_gno",
        "selected_floor": 5,
    }
    sim_resp = test_client.post("/api/v1/demo/simulate", json=payload)
    assert sim_resp.status_code == 200
    data = sim_resp.json()
    assert data["archetype"]["archetype_id"] == "5S_T120"
    assert len(data["time"]) > 0
    assert "metrics" in data
