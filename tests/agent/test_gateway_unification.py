"""
tests/agent/test_gateway_unification.py — Verification of unified API Gateway.

Validates that api.main:app seamlessly routes both:
  1. Original Digital-Twin routes (/api/v1/...)
  2. SeismoAgent Autonomous Engineering & NVIDIA Nemotron routes (/api/agent/..., /api/simulate, /api/presets)
"""

import pytest
import json
from fastapi.testclient import TestClient

from api.main import app
from seismo_agent.server.dependencies import get_orchestrator
from seismo_agent.core.orchestrator import SeismoOrchestrator
from tests.agent.test_orchestrator import MockLLMClient


@pytest.fixture
def mock_orchestrator():
    client = MockLLMClient([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_unified_01",
                "type": "function",
                "function": {
                    "name": "analyze_earthquake_record",
                    "arguments": json.dumps({"record_id": "synthetic_ricker", "target_pga_g": 0.35}),
                }
            }]
        },
        {
            "role": "assistant",
            "content": "Unified gateway dispatch concluded successfully.",
        }
    ])
    return SeismoOrchestrator(llm_client=client)


def test_unified_gateway_digital_twin_endpoints():
    """Verify digital-twin routes on api.main:app."""
    with TestClient(app) as client:
        # Health
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

        # System info
        res_info = client.get("/api/v1/system/info")
        assert res_info.status_code == 200
        assert res_info.json()["status"] == "operational"


def test_unified_gateway_agent_endpoints(mock_orchestrator):
    """Verify SeismoAgent & NVIDIA Nemotron endpoints on api.main:app."""
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    try:
        with TestClient(app) as client:
            # 1. Agent Health
            res_h = client.get("/api/agent/health")
            assert res_h.status_code == 200
            data_h = res_h.json()
            assert data_h["service"] == "seismoagent"
            assert data_h["tools_count"] == 8

            # 2. Tool Registry
            res_t = client.get("/api/agent/tools")
            assert res_t.status_code == 200
            data_t = res_t.json()
            assert data_t["count"] == 8
            tool_names = [t["name"] for t in data_t["tools"]]
            assert "run_seismo_inference" in tool_names
            assert "run_physics_simulation" in tool_names

            # 3. Presets
            res_p = client.get("/api/presets")
            assert res_p.status_code == 200
            data_p = res_p.json()
            assert len(data_p["presets"]) >= 2

            # 4. Agent Run
            res_r = client.post("/api/agent/run", json={"message": "Analyze synthetic ricker at 0.35g"})
            assert res_r.status_code == 200
            data_r = res_r.json()
            assert data_r["status"] == "completed"
            assert "concluded successfully" in data_r["final_response"]

            # 5. Direct Simulate
            sim_payload = {
                "preset_id": "synthetic_ricker",
                "pga_g": 0.35,
                "T": 0.5,
                "zeta": 0.05,
                "material_type": "bilinear",
                "u_y": 0.015,
                "alpha": 0.05,
                "mass": 1000.0,
                "solver": "opensees",
            }
            res_s = client.post("/api/simulate", json=sim_payload)
            assert res_s.status_code == 200
            data_s = res_s.json()
            assert data_s["status"] == "success"
            assert "metrics" in data_s
            assert "u_pred" in data_s
            assert "u_gt" in data_s
    finally:
        app.dependency_overrides.clear()
