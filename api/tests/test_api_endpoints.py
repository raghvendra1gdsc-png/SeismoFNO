"""
api/tests/test_api_endpoints.py — End-to-End Integration Tests for FastAPI Endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_endpoint_health():
    """Verify health endpoint returns status 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "device" in data


def test_endpoint_system_info():
    """Verify system info endpoint introspects model parameters and provenance."""
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_parameters" in data
    assert "checkpoint_path" in data
    assert "scientific_integrity" in data


def test_endpoint_earthquakes():
    """Verify earthquake catalog listing returns both Indian catalog and PEER records."""
    response = client.get("/api/v1/earthquakes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_indian"] > 0
    assert data["total_peer"] > 0
    assert len(data["indian_catalog"]) == data["total_indian"]


def test_endpoint_earthquake_details():
    """Verify fetching specific ground motion details."""
    response = client.get("/api/v1/earthquakes/IND-2001-BHUJ")
    assert response.status_code == 200
    data = response.json()
    assert "pga_g" in data
    assert "preview" in data


def test_endpoint_buildings():
    """Verify structural archetype listing."""
    response = client.get("/api/v1/buildings")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 4
    assert any(b["id"] == "BLD-RC-03" for b in data["buildings"])


def test_endpoint_scenario_validate():
    """Verify scenario validation endpoint."""
    payload = {
        "earthquake_id": "RSN0001_Imperial_Valley-06.AT2",
        "pga_g": 0.40,
        "T0": 0.50,
        "damping_ratio": 0.05,
        "yield_displacement_m": 0.010,
        "post_yield_ratio": 0.05,
        "material_type": "bilinear",
        "mass_kg": 1.0,
    }
    response = client.post("/api/v1/scenario/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["domain_status"] == "IN_DOMAIN"


def test_endpoint_scenario_predict_surrogate_only():
    """Verify prediction endpoint runs fast surrogate inference."""
    payload = {
        "earthquake_id": "RSN0001_Imperial_Valley-06.AT2",
        "pga_g": 0.35,
        "T0": 0.60,
        "damping_ratio": 0.05,
        "yield_displacement_m": 0.015,
        "post_yield_ratio": 0.05,
        "material_type": "bilinear",
        "include_ground_truth": False,
        "stride": 2,
    }
    response = client.post("/api/v1/scenario/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "trajectories" in data
    assert "metrics" in data
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] > 0
    assert len(data["trajectories"]["u"]) > 100
    assert data["validation"] is None  # Ground truth not requested


def test_endpoint_scenario_predict_with_ground_truth():
    """Verify prediction with concurrent OpenSeesPy ground truth comparison."""
    payload = {
        "earthquake_id": "RSN0001_Imperial_Valley-06.AT2",
        "pga_g": 0.30,
        "T0": 0.50,
        "damping_ratio": 0.05,
        "yield_displacement_m": 0.012,
        "post_yield_ratio": 0.05,
        "material_type": "bilinear",
        "include_ground_truth": True,
        "stride": 2,
    }
    response = client.post("/api/v1/scenario/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["validation"] is not None
    val = data["validation"]
    assert "relative_l2_u_percent" in val
    assert "speedup_factor" in val
    assert "opensees_latency_ms" in val
    assert data["trajectories"]["u_gt"] is not None
