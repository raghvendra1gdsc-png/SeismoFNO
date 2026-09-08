"""
test_dashboard.py — Unit Tests for SeismoFNO Interactive Dashboard Application.
"""

import json
import math
import threading
import time
import urllib.request
from http.server import HTTPServer

import numpy as np
import pytest
import torch

from dashboard.app import DashboardEngine, SeismoFNODashboardHandler


@pytest.fixture(scope="module")
def dashboard_engine():
    """Create a DashboardEngine instance on CPU for fast testing."""
    return DashboardEngine(device="cpu")


@pytest.fixture(scope="module")
def local_server(dashboard_engine):
    """Start local test HTTP server on an ephemeral port."""
    SeismoFNODashboardHandler.engine = dashboard_engine
    server = HTTPServer(("127.0.0.1", 0), SeismoFNODashboardHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def test_dashboard_engine_initialization(dashboard_engine):
    """Verify DashboardEngine initializes with model and presets."""
    assert dashboard_engine.model is not None
    assert len(dashboard_engine.available_presets) > 0
    assert dashboard_engine.x_norm is not None
    assert dashboard_engine.y_norm is not None


def test_load_preset_and_synthetic(dashboard_engine):
    """Verify loading PEER and synthetic acceleration records."""
    # Synthetic
    ag_synth, dt_synth, name_synth = dashboard_engine.load_preset_acceleration("synthetic_ricker", target_pga_g=0.3)
    assert len(ag_synth) == 2048
    assert dt_synth == 0.01
    assert "Synthetic" in name_synth
    assert np.isclose(np.max(np.abs(ag_synth)) / 9.80665, 0.3, atol=0.01)

    # PEER preset
    if dashboard_engine.available_presets:
        preset_id = dashboard_engine.available_presets[0]["id"]
        ag_peer, dt_peer, name_peer = dashboard_engine.load_preset_acceleration(preset_id, target_pga_g=0.5)
        assert len(ag_peer) == 2048
        assert dt_peer == 0.01


def test_parse_uploaded_file(dashboard_engine):
    """Verify parsing user-uploaded ground motion files."""
    # Synthetic text content
    data_vals = [f"{0.1 * math.sin(i * 0.1):.6f}" for i in range(500)]
    content = "\n".join(data_vals)
    ag, dt, name = dashboard_engine.parse_uploaded_file(content, "custom_record.txt")
    assert len(ag) == 2048
    assert dt == 0.01
    assert name == "custom_record.txt"


def test_run_simulation_bilinear(dashboard_engine):
    """Verify end-to-end simulation execution for bilinear SDOF."""
    ag_synth, dt, _ = dashboard_engine.load_preset_acceleration("synthetic_ricker", target_pga_g=0.4)
    res = dashboard_engine.run_simulation(
        ag=ag_synth,
        dt=dt,
        T=0.5,
        zeta=0.05,
        material_type="bilinear",
        u_y=0.01,
        alpha=0.05,
    )
    assert "time" in res
    assert "u_gt" in res
    assert "u_pred" in res
    assert "fr_gt" in res
    assert "fr_pred" in res
    assert "eh_gt" in res
    assert "eh_pred" in res
    assert "metrics" in res

    m = res["metrics"]
    assert "err_u_rel_l2" in m
    assert "err_fr_rel_l2" in m
    assert "err_eh_rel_l2" in m
    assert "speedup" in m
    assert m["ductility_mu"] > 0


def test_run_simulation_elastic(dashboard_engine):
    """Verify end-to-end simulation execution for linear-elastic SDOF."""
    ag_synth, dt, _ = dashboard_engine.load_preset_acceleration("synthetic_sine", target_pga_g=0.2)
    res = dashboard_engine.run_simulation(
        ag=ag_synth,
        dt=dt,
        T=0.4,
        zeta=0.05,
        material_type="elastic",
        u_y=0.02,
        alpha=0.0,
    )
    assert len(res["u_pred"]) > 0
    assert res["metrics"]["ductility_mu"] == 1.0


def test_http_api_endpoints(local_server):
    """Test HTTP REST endpoints: GET /, GET /api/presets, GET /api/health, POST /api/simulate."""
    # 1. GET /
    with urllib.request.urlopen(f"{local_server}/") as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "SEISMOFNO" in html.upper() or "SEISMOAGENT" in html.upper()
        assert "STRUCTURAL DYNAMICS" in html.upper() or "STRUCTURAL ENGINEERING" in html.upper()

    # 2. GET /api/presets
    with urllib.request.urlopen(f"{local_server}/api/presets") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "presets" in data
        assert len(data["presets"]) > 0

    # 3. GET /api/health
    with urllib.request.urlopen(f"{local_server}/api/health") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "online"
        assert data["num_params"] > 0

    # 4. POST /api/simulate
    payload = {
        "preset_id": "synthetic_ricker",
        "pga_g": 0.35,
        "T": 0.6,
        "zeta": 0.05,
        "material_type": "bilinear",
        "u_y": 0.01,
        "alpha": 0.05,
    }
    req = urllib.request.Request(
        f"{local_server}/api/simulate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "u_pred" in data
        assert "u_gt" in data
        assert "metrics" in data
        assert data["metrics"]["err_u_rel_l2"] >= 0.0
