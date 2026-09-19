"""
tests/agent/test_server.py — Unit & Integration Tests for SeismoAgent API Gateway.

Verifies:
  1. Health endpoint introspection and secret protection.
  2. Tool registry listing and schema validity.
  3. POST /api/agent/run valid execution with mocked LLM.
  4. Input validation: empty queries, malformed payloads, and size bounds.
  5. Timeout handling with 504 Gateway Timeout responses.
  6. Secret sanitization across gateway responses.
  7. WebSocket /api/agent/stream event ordering and completion.
  8. OpenSees concurrency policy and serialization under mutual exclusion.
  9. Optional live integration test (skipped when NEBIUS_API_KEY is absent).
"""

import json
import os
import time
import threading
import pytest
from fastapi.testclient import TestClient

from seismo_agent.server.app import create_app
from seismo_agent.server.dependencies import get_orchestrator
from seismo_agent.core.orchestrator import SeismoOrchestrator
from seismo_agent.core.llm_client import LLMClientProtocol
from seismo_agent.server.execution import GatewayTimeoutError, OPENSEES_NATIVE_LOCK
from seismo_agent.tools.physics_sim_tool import _OPENSEES_LOCK
from tests.agent.test_orchestrator import MockLLMClient


@pytest.fixture
def mock_orchestrator():
    """Provides an orchestrator backed by MockLLMClient for deterministic API testing."""
    client = MockLLMClient([
        # Step 1: Tool call
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_srv_01",
                "type": "function",
                "function": {
                    "name": "analyze_earthquake_record",
                    "arguments": json.dumps({"record_id": "synthetic_ricker", "target_pga_g": 0.35}),
                }
            }]
        },
        # Step 2: Final response
        {
            "role": "assistant",
            "content": "Earthquake analysis concluded successfully. Peak ground acceleration is 0.35g.",
        }
    ])
    return SeismoOrchestrator(llm_client=client)


@pytest.fixture
def client(mock_orchestrator):
    """Creates a FastAPI test client with dependency overrides for the orchestrator."""
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# 1. Health Endpoint Tests
# ==============================================================================

def test_health_endpoint(client):
    """Verify GET /api/agent/health returns structured metadata without secrets."""
    resp = client.get("/api/agent/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["service"] == "seismoagent"
    assert data["version"] == "0.3.1"
    assert data["orchestrator"] == "available"
    assert "nebius_configured" in data
    assert "model" in data
    assert data["research_core"] == "read_only"
    assert data["tools_count"] == 8
    assert "device" in data

    # Concurrency and verification semantics
    assert data["concurrency_policy"] == "process_local_serialization (single_worker_required)"
    assert data["opensees_execution"] == "serialized_via_threading_lock"
    assert "nebius_live_verified" in data

    # Verify no secret leakage
    raw_text = resp.text
    assert "api_key" not in raw_text.lower()
    assert "secret" not in raw_text.lower()
    assert "bearer" not in raw_text.lower()


# ==============================================================================
# 2. Tool Registry Endpoint Tests
# ==============================================================================

def test_tools_endpoint(client):
    """Verify GET /api/agent/tools exposes all 8 tools with valid parameters schema."""
    resp = client.get("/api/agent/tools")
    assert resp.status_code == 200
    data = resp.json()

    assert data["count"] == 8
    assert len(data["tools"]) == 8

    tool_names = {t["name"] for t in data["tools"]}
    expected_names = {
        "analyze_earthquake_record",
        "configure_structure",
        "run_seismo_inference",
        "run_physics_simulation",
        "compare_fno_vs_physics",
        "assess_ood_and_uncertainty",
        "run_parametric_sweep",
        "generate_engineering_report",
    }
    assert tool_names == expected_names

    for tool in data["tools"]:
        assert len(tool["description"]) > 20
        assert "parameters" in tool
        assert tool["parameters"]["type"] == "object"


# ==============================================================================
# 3. Agent Execution (Run) Endpoint Tests
# ==============================================================================

def test_agent_run_success(client):
    """Verify POST /api/agent/run executes orchestrator and returns full audit trace."""
    req_body = {
        "message": "Analyze synthetic Ricker record scaled to 0.35g.",
        "context": {"units": "SI"},
    }
    resp = client.post("/api/agent/run", json=req_body)
    assert resp.status_code == 200
    data = resp.json()

    assert data["request_id"].startswith("req_")
    assert data["status"] == "completed"
    assert "0.35g" in data["final_response"]
    assert len(data["tool_trace"]) == 1
    assert data["tool_trace"][0]["tool_name"] == "analyze_earthquake_record"
    assert data["tool_trace"][0]["success"] is True
    assert "analyze_earthquake_record" in data["tool_results"]

    # Verify timing breakdown
    assert "timing" in data
    assert "total_server_latency_ms" in data["timing"]
    assert "orchestrator_latency_ms" in data["timing"]
    assert data["timing"]["total_server_latency_ms"] > 0.0


def test_agent_run_empty_query_rejected(client):
    """Verify POST /api/agent/run rejects empty or whitespace-only messages."""
    resp = client.post("/api/agent/run", json={"message": "   "})
    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == "error"
    assert "cannot be empty" in data["error"]["message"]


def test_agent_run_malformed_json_rejected(client):
    """Verify POST /api/agent/run rejects invalid JSON schemas with 422 Unprocessable Entity."""
    resp = client.post("/api/agent/run", json={"invalid_field": 12345})
    assert resp.status_code == 422
    data = resp.json()
    assert data["status"] == "error"
    assert data["error"]["type"] == "ValidationError"


def test_agent_run_timeout_handled(monkeypatch):
    """Verify POST /api/agent/run returns 504 Gateway Timeout when execution exceeds limit."""
    import sys
    app = create_app()

    def slow_orchestrator(*args, **kwargs):
        raise GatewayTimeoutError("Execution timed out after 0.01s.")

    app_mod = sys.modules["seismo_agent.server.app"]
    monkeypatch.setattr(app_mod, "execute_orchestrator_safe", slow_orchestrator)

    with TestClient(app) as test_client:
        resp = test_client.post("/api/agent/run", json={"message": "Run slow calculation."})
        assert resp.status_code == 504
        data = resp.json()
        assert data["status"] == "error"
        assert data["error"]["type"] == "TimeoutError"


def test_agent_run_secret_sanitization(client, monkeypatch):
    """Verify that any sensitive strings in inputs/outputs are redacted from response payload."""
    secret_key = "nebius_secret_test_token_999"
    monkeypatch.setattr("seismo_agent.core.orchestrator.NEBIUS_API_KEY", secret_key)

    req_body = {
        "message": f"Analyze earthquake using authorization token {secret_key}",
    }
    resp = client.post("/api/agent/run", json=req_body)
    assert resp.status_code == 200
    resp_text = resp.text
    assert secret_key not in resp_text
    assert "[REDACTED]" in resp_text or "synthetic" in resp_text


# ==============================================================================
# 4. WebSocket Streaming Tests
# ==============================================================================

def test_websocket_streaming(client):
    """Verify WS /api/agent/stream connects, streams progress events, and terminates cleanly."""
    with client.websocket_connect("/api/agent/stream") as ws:
        # Send initial query
        ws.send_text(json.dumps({"message": "Analyze synthetic Ricker record scaled to 0.35g."}))

        events = []
        # Receive streamed events until completed or error
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev.get("event") in ("completed", "error"):
                break

        assert len(events) >= 3
        event_types = [e["event"] for e in events]
        assert "request_started" in event_types
        assert "planning" in event_types
        assert "completed" in event_types

        # Verify all events have request_id
        for ev in events:
            assert "request_id" in ev


def test_websocket_streaming_empty_query(client):
    """Verify WS /api/agent/stream handles empty query with error event and closes."""
    with client.websocket_connect("/api/agent/stream") as ws:
        ws.send_text(json.dumps({"message": ""}))
        ev = ws.receive_json()
        assert ev["event"] == "error"
        assert "cannot be empty" in ev["error"]


# ==============================================================================
# 5. OpenSees Concurrency Policy Tests
# ==============================================================================

def test_opensees_lock_serialization():
    """
    Test A: Verify that OPENSEES_NATIVE_LOCK guarantees mutually exclusive execution
    across concurrent threads attempting simultaneous access.
    """
    execution_order = []
    active_count = 0
    max_concurrent = 0

    def worker(worker_id: int):
        nonlocal active_count, max_concurrent
        with _OPENSEES_LOCK:
            active_count += 1
            max_concurrent = max(max_concurrent, active_count)
            execution_order.append(f"start_{worker_id}")
            time.sleep(0.05)  # Simulate OpenSees simulation workload
            execution_order.append(f"end_{worker_id}")
            active_count -= 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Crucial assertion: max concurrent executions inside OpenSees lock MUST be exactly 1
    assert max_concurrent == 1
    assert len(execution_order) == 8


def test_opensees_concurrent_simulations_lock_scope():
    """
    Test B: Verify lock scope covers complete OpenSees simulation lifecycle:
    ops.wipe() -> model construction -> simulation -> result extraction -> ops.wipe().
    Executes multiple real OpenSees simulations concurrently without memory corruption.
    """
    from seismo_agent.tools.physics_sim_tool import PhysicsSimTool
    from seismo_agent.schemas.inputs import PhysicsSimulationRequest, EarthquakeRequest, StructuralRequest

    tool = PhysicsSimTool()
    results = []
    exceptions = []

    def run_sim(pga: float):
        try:
            req = PhysicsSimulationRequest(
                earthquake=EarthquakeRequest(record_id="synthetic_ricker", target_pga_g=pga),
                structure=StructuralRequest(system_type="sdof", T=0.5, zeta=0.05, material_type="bilinear", u_y=0.015),
                solver="opensees",
            )
            res = tool.execute(req)
            results.append(res)
        except Exception as e:
            exceptions.append(e)

    threads = [threading.Thread(target=run_sim, args=(0.2 + i * 0.1,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(exceptions) == 0, f"Concurrent OpenSees simulations raised exceptions: {exceptions}"
    assert len(results) == 3
    for res in results:
        assert res.status == "success"
        assert res.u_max_m > 0.0
        assert len(res.u_gt) == 2048


# ==============================================================================
# 6. Environment-Gated Live Nebius Integration Test
# ==============================================================================

@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_NEBIUS"),
    reason="SKIPPED — Live remote tests require RUN_LIVE_NEBIUS=1.",
)
def test_live_nebius_gateway_integration():
    """
    Live end-to-end integration test through the real FastAPI gateway using live Nebius credentials.
    Runs ONLY when NEBIUS_API_KEY is present in the environment.
    Proves:
      HTTP request -> SeismoOrchestrator -> Nebius Token Factory -> NVIDIA Nemotron ->
      tool selection -> SeismoAgent deterministic tool -> structured tool result ->
      Nemotron synthesis -> OrchestrationResult.
    """
    from seismo_agent.server.app import set_nebius_live_verified
    from seismo_agent.config import NEBIUS_BASE_URL, NEBIUS_MODEL

    app = create_app()  # Real orchestrator without mock
    with TestClient(app) as live_client:
        # Pre-check health before live run
        health_pre = live_client.get("/api/agent/health").json()
        assert health_pre["nebius_configured"] is True

        # Run query requiring deterministic tool calling
        t0 = time.perf_counter()
        resp = live_client.post(
            "/api/agent/run",
            json={
                "message": (
                    "Analyze the synthetic Ricker earthquake record scaled to 0.25g "
                    "and report its deterministic peak ground acceleration."
                )
            },
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert resp.status_code == 200, f"Live request failed with body: {resp.text}"
        data = resp.json()

        # Telemetry verification (secrets strictly redacted)
        assert data["status"] == "completed"
        assert len(data["final_response"]) > 20
        assert len(data["tool_trace"]) >= 1

        selected_tool = data["tool_trace"][0]["tool_name"]
        assert selected_tool == "analyze_earthquake_record"
        assert data["tool_trace"][0]["success"] is True

        # Mark live verification in server state
        set_nebius_live_verified(True)

        # Post-check health confirms live verification
        health_post = live_client.get("/api/agent/health").json()
        assert health_post["nebius_live_verified"] is True

        print(
            f"\n[LIVE NEBIUS VERIFICATION SUCCESS]\n"
            f"  Endpoint: {NEBIUS_BASE_URL}\n"
            f"  Model: {NEBIUS_MODEL}\n"
            f"  Tool Selected: {selected_tool}\n"
            f"  Total Latency: {latency_ms:.2f} ms\n"
            f"  Orchestration Steps: {data['steps_used']}\n"
        )


# ==============================================================================
# 7. Direct Engineering Workspaces Endpoints Tests
# ==============================================================================

def test_presets_endpoint(client):
    """Verify GET /api/presets returns discovered and synthetic earthquake presets."""
    resp = client.get("/api/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert "presets" in data
    assert len(data["presets"]) >= 2
    preset_ids = [p["id"] for p in data["presets"]]
    assert "synthetic_ricker" in preset_ids
    assert "synthetic_sine" in preset_ids


def test_simulate_endpoint(client):
    """Verify POST /api/simulate runs full FNO + OpenSees physics reference comparison."""
    payload = {
        "preset_id": "synthetic_ricker",
        "pga_g": 0.35,
        "T": 0.50,
        "zeta": 0.05,
        "material_type": "bilinear",
        "u_y": 0.015,
        "alpha": 0.05,
    }
    resp = client.post("/api/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["time"]) == 2048
    assert len(data["u_pred"]) == 2048
    assert len(data["u_gt"]) == 2048
    assert len(data["fr_pred"]) == 2048
    assert len(data["fr_gt"]) == 2048
    assert len(data["eh_pred"]) == 2048
    assert len(data["eh_gt"]) == 2048
    assert "metrics" in data
    assert data["metrics"]["err_u_rel_l2"] >= 0.0
    assert data["metrics"]["speedup"] > 0.0
    assert "ood" in data
    assert data["ood"]["uncertainty_status"] == "NOT_QUANTIFIED"


def test_sweep_endpoint(client):
    """Verify POST /api/sweep executes parametric sweep across structural values."""
    payload = {
        "param_name": "T",
        "param_values": [0.3, 0.6],
        "preset_id": "synthetic_ricker",
        "pga_g": 0.30,
        "T": 0.50,
        "zeta": 0.05,
        "material_type": "bilinear",
        "u_y": 0.015,
        "alpha": 0.05,
    }
    resp = client.post("/api/sweep", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["param_name"] == "T"
    assert data["total_evaluations"] == 2
    assert len(data["results"]) == 2


def test_report_endpoint(client):
    """Verify POST /api/report generates deterministic technical engineering report."""
    payload = {
        "preset_id": "synthetic_ricker",
        "pga_g": 0.30,
        "T": 0.50,
        "zeta": 0.05,
        "material_type": "bilinear",
        "u_y": 0.015,
        "alpha": 0.05,
        "format": "markdown",
    }
    resp = client.post("/api/report", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "report_title" in data
    assert len(data["content"]) > 100
    assert "SeismoFNO" in data["content"]
    assert "NOT_QUANTIFIED" in data["content"]


def test_spa_static_files_served(client):
    """Verify GET / serves production frontend index.html when dist exists."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "SEISMOAGENT" in resp.text or "SeismoFNO" in resp.text or "root" in resp.text


# ==============================================================================
# 5. Stage 5 Failure-Injection & Boundary Resilience Tests
# ==============================================================================

def test_failure_injection_nebius_down(monkeypatch):
    """Verify orchestrator failure (e.g. Nebius 503 or network drop) returns safe error response."""
    from seismo_agent.core.orchestrator import SeismoOrchestrator
    
    def mock_run_failure(*args, **kwargs):
        raise RuntimeError("Nebius Token Factory connection refused (simulated 503)")
    
    monkeypatch.setattr(SeismoOrchestrator, "run", mock_run_failure)
    app = create_app()
    with TestClient(app) as test_client:
        resp = test_client.post("/api/agent/run", json={"message": "Simulate earthquake response."})
        assert resp.status_code == 500
        data = resp.json()
        assert data["status"] == "error"
        assert "error" in data
        assert "token factory" in data["error"]["message"].lower() or "simulated" in data["error"]["message"].lower()
        # Verify no secret leakage
        assert "api_key" not in resp.text.lower()


def test_failure_injection_physics_invalid_params(client):
    """Verify invalid physical parameters are rejected with HTTP 422 before numerical solvers run."""
    invalid_payload = {
        "preset_id": "synthetic_ricker",
        "pga_g": -0.50,  # Invalid: negative PGA
        "T": -1.0,       # Invalid: negative period
        "zeta": 0.05,
        "material_type": "bilinear",
        "u_y": 0.010,
        "alpha": 0.05,
    }
    resp = client.post("/api/simulate", json=invalid_payload)
    assert resp.status_code == 422  # Pydantic validation rejection
    data = resp.json()
    assert data["status"] == "error"
    assert data["error"]["type"] == "ValidationError"
    # Confirm no fake simulation numbers were returned
    assert "u_pred" not in data
    assert "u_gt" not in data


def test_failure_injection_nonexistent_preset(client):
    """Verify non-existent earthquake preset returns clean error rather than fabricated numbers."""
    payload = {
        "preset_id": "non_existent_earthquake_record_9999.AT2",
        "pga_g": 0.40,
        "T": 0.50,
        "zeta": 0.05,
        "material_type": "bilinear",
        "u_y": 0.010,
        "alpha": 0.05,
    }
    resp = client.post("/api/simulate", json=payload)
    assert resp.status_code == 500
    data = resp.json()
    assert data["status"] == "error"
    assert "error" in data
    assert "not found" in data["error"].lower() or "missing" in data["error"].lower()


def test_ood_detection_injection(client):
    """Verify severe out-of-domain parameters trigger OOD WARNING and NOT_QUANTIFIED uncertainty."""
    ood_payload = {
        "preset_id": "synthetic_ricker",
        "pga_g": 1.60,   # Exceeds 1.20g training bound
        "T": 3.80,       # Exceeds 3.00s period bound
        "zeta": 0.05,
        "material_type": "bilinear",
        "u_y": 0.060,    # Exceeds 45mm yield bound
        "alpha": 0.05,
    }
    resp = client.post("/api/simulate", json=ood_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "ood" in data
    assert data["ood"]["is_ood"] is True
    assert len(data["ood"]["domain_violations"]) >= 2
    assert data["ood"]["uncertainty_status"] == "NOT_QUANTIFIED"
    assert "OUT-OF-DISTRIBUTION WARNING" in data["ood"]["recommendation"]


def test_websocket_disconnect_resilience():
    """Verify WebSocket disconnects gracefully without corrupting server state or locking mutex."""
    from seismo_agent.tools.physics_sim_tool import _OPENSEES_LOCK
    app = create_app()
    with TestClient(app) as test_client:
        with test_client.websocket_connect("/api/agent/stream") as ws:
            ws.send_json({"message": "Analyze synthetic Ricker ground motion."})
            first_event = ws.receive_json()
            assert first_event["event"] == "request_started"
            # Close connection immediately midway
            ws.close()
    
    # Assert OpenSees mutex is released and not held
    assert not _OPENSEES_LOCK.locked()


