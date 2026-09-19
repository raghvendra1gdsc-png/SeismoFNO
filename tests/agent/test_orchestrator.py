"""
tests/agent/test_orchestrator.py — Unit Tests for SeismoOrchestrator & Nemotron Integration.

Verifies:
  1. Tool registry exposure and spec conversion.
  2. Single tool dispatch and execution.
  3. Multi-step engineering workflow planning and sequential execution.
  4. Whitelist enforcement and rejection of unauthorized/unknown tools.
  5. Pydantic validation failure handling for malformed tool arguments.
  6. Controlled recovery from deterministic tool failures without fabrication.
  7. Bounded execution when MAX_TOOL_CALLS is exceeded.
  8. Scientific honesty enforcement (no fabricated confidence percentages).
  9. Out-of-distribution (OOD) explicit warning propagation.
  10. Secrets masking across execution traces and telemetry.
  11. Optional live Nebius smoke test (runs only when NEBIUS_API_KEY is present).
"""

import json
import os
import pytest
from typing import List, Dict, Any, Optional

from seismo_agent.core.orchestrator import SeismoOrchestrator, OrchestrationResult, ToolCallTrace
from seismo_agent.core.llm_client import LLMClientProtocol, NemotronClient, AuthenticationError
from seismo_agent.tools import get_all_tools


class MockLLMClient:
    """
    Deterministic mock for NemotronClient that returns pre-queued responses.
    Simulates function/tool calls and conversational text generation.
    """

    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None):
        self.responses: List[Dict[str, Any]] = responses or []
        self.call_history: List[Dict[str, Any]] = []
        self._call_count = 0

    def queue_response(self, response: Dict[str, Any]):
        self.responses.append(response)

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        self.call_history.append({
            "messages": copy_messages(messages),
            "tools": tools,
        })
        self._call_count += 1
        if self.responses:
            return self.responses.pop(0)
        # Default fallback: plain text response
        return {
            "role": "assistant",
            "content": "Analysis completed using verified deterministic tools.",
        }


def copy_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper to deep-copy message list."""
    return json.loads(json.dumps(messages))


# ==============================================================================
# 1. Tool Registry Exposure
# ==============================================================================

def test_tool_registry_exposure():
    """Test 1: Verify all 8 Stage 1 tools are exposed with valid specs."""
    orchestrator = SeismoOrchestrator(llm_client=MockLLMClient())
    assert len(orchestrator.tools) == 8
    assert len(orchestrator.tools_spec) == 8

    expected_tools = {
        "analyze_earthquake_record",
        "configure_structure",
        "run_seismo_inference",
        "run_physics_simulation",
        "compare_fno_vs_physics",
        "assess_ood_and_uncertainty",
        "run_parametric_sweep",
        "generate_engineering_report",
    }
    registered_names = set(orchestrator.tools_by_name.keys())
    assert expected_tools == registered_names

    for spec in orchestrator.tools_spec:
        assert spec["type"] == "function"
        fn = spec["function"]
        assert fn["name"] in expected_tools
        assert len(fn["description"]) > 20
        assert "parameters" in fn


# ==============================================================================
# 2. Single Tool Call Dispatch
# ==============================================================================

def test_single_tool_dispatch():
    """Test 2: Mock Nemotron requesting earthquake analysis."""
    mock_client = MockLLMClient([
        # Turn 1: Nemotron requests analyze_earthquake_record
        {
            "role": "assistant",
            "content": "I will analyze the specified earthquake ground motion.",
            "tool_calls": [
                {
                    "id": "call_eq_01",
                    "type": "function",
                    "function": {
                        "name": "analyze_earthquake_record",
                        "arguments": json.dumps({"record_id": "synthetic_ricker", "target_pga_g": 0.35}),
                    }
                }
            ]
        },
        # Turn 2: Nemotron receives tool result and provides final explanation
        {
            "role": "assistant",
            "content": "Earthquake analysis complete. Peak ground acceleration is measured at 0.35g.",
        }
    ])

    orchestrator = SeismoOrchestrator(llm_client=mock_client)
    res = orchestrator.run("Analyze the synthetic Ricker earthquake record scaled to 0.35g.")

    assert isinstance(res, OrchestrationResult)
    assert res.status == "completed"
    assert len(res.tool_trace) == 1
    trace = res.tool_trace[0]
    assert trace.tool_name == "analyze_earthquake_record"
    assert trace.success is True
    assert "pga_g" in trace.output_summary
    assert pytest.approx(trace.output_summary["pga_g"], rel=1e-2) == 0.35
    assert "0.35g" in res.final_response


# ==============================================================================
# 3. Multi-Step Engineering Workflow
# ==============================================================================

def test_multi_step_workflow():
    """Test 3: Mock sequential execution of the complete 6-stage engineering audit."""
    mock_client = MockLLMClient([
        # Step 1: Ingest ground motion
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "analyze_earthquake_record",
                    "arguments": json.dumps({"record_id": "synthetic_ricker", "target_pga_g": 0.3}),
                }
            }]
        },
        # Step 2: Configure structure
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "configure_structure",
                    "arguments": json.dumps({"system_type": "sdof", "T": 0.5, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.015}),
                }
            }]
        },
        # Step 3: Run SeismoFNO surrogate inference
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_3",
                "type": "function",
                "function": {
                    "name": "run_seismo_inference",
                    "arguments": json.dumps({
                        "earthquake": {"record_id": "synthetic_ricker", "target_pga_g": 0.3},
                        "structure": {"system_type": "sdof", "T": 0.5, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.015},
                    }),
                }
            }]
        },
        # Step 4: Run OpenSeesPy ground truth
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_4",
                "type": "function",
                "function": {
                    "name": "run_physics_simulation",
                    "arguments": json.dumps({
                        "earthquake": {"record_id": "synthetic_ricker", "target_pga_g": 0.3},
                        "structure": {"system_type": "sdof", "T": 0.5, "zeta": 0.05, "material_type": "bilinear", "u_y": 0.015},
                    }),
                }
            }]
        },
        # Step 5: Final engineering explanation
        {
            "role": "assistant",
            "content": "Comprehensive analysis complete. Both SeismoFNO and OpenSeesPy executed deterministically.",
        }
    ])

    orchestrator = SeismoOrchestrator(llm_client=mock_client)
    res = orchestrator.run("Run complete surrogate and physics comparison on a 0.5s bilinear oscillator.")

    assert res.status == "completed"
    assert len(res.tool_trace) == 4
    executed_names = [t.tool_name for t in res.tool_trace]
    assert executed_names == [
        "analyze_earthquake_record",
        "configure_structure",
        "run_seismo_inference",
        "run_physics_simulation",
    ]
    for trace in res.tool_trace:
        assert trace.success is True
    assert "run_seismo_inference" in res.tool_results
    assert "run_physics_simulation" in res.tool_results


# ==============================================================================
# 4. Whitelist Enforcement & Unknown Tool Rejection
# ==============================================================================

def test_unauthorized_tool_rejected():
    """Test 4: Verify unauthorized tools or shell commands are rejected with structured error."""
    mock_client = MockLLMClient([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_bad",
                "type": "function",
                "function": {
                    "name": "execute_shell_command",
                    "arguments": json.dumps({"command": "ls -la"}),
                }
            }]
        },
        {
            "role": "assistant",
            "content": "I apologize, execute_shell_command is not an authorized tool.",
        }
    ])

    orchestrator = SeismoOrchestrator(llm_client=mock_client)
    res = orchestrator.run("Run bash command.")

    assert len(res.tool_trace) == 1
    assert res.tool_trace[0].tool_name == "execute_shell_command"
    assert res.tool_trace[0].success is False
    assert "not a registered SeismoAgent engineering tool" in res.tool_trace[0].output_summary["error"]


# ==============================================================================
# 5. Pydantic Argument Validation
# ==============================================================================

def test_malformed_arguments_caught():
    """Test 5: Verify Pydantic schema catches invalid arguments (e.g. negative period)."""
    mock_client = MockLLMClient([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_bad_args",
                "type": "function",
                "function": {
                    "name": "configure_structure",
                    "arguments": json.dumps({"system_type": "sdof", "T": -0.5}),
                }
            }]
        },
        {
            "role": "assistant",
            "content": "The structural period must be strictly positive.",
        }
    ])

    orchestrator = SeismoOrchestrator(llm_client=mock_client)
    res = orchestrator.run("Configure structure with T=-0.5s.")

    assert len(res.tool_trace) == 1
    assert res.tool_trace[0].success is False
    assert len(res.errors) >= 1


# ==============================================================================
# 6. Tool Failure Recovery
# ==============================================================================

def test_deterministic_tool_failure_handled():
    """Test 6: Simulated tool failure is passed cleanly back to Nemotron without fabrication."""
    mock_client = MockLLMClient([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_fail",
                "type": "function",
                "function": {
                    "name": "analyze_earthquake_record",
                    "arguments": json.dumps({"record_id": "nonexistent_earthquake_123.AT2"}),
                }
            }]
        },
        {
            "role": "assistant",
            "content": "The requested preset record was not found in data/raw/ground_motions.",
        }
    ])

    orchestrator = SeismoOrchestrator(llm_client=mock_client)
    res = orchestrator.run("Analyze nonexistent record.")

    assert len(res.tool_trace) == 1
    assert res.tool_trace[0].success is False
    assert "not found" in res.tool_trace[0].output_summary["error"]
    assert "not found" in res.final_response


# ==============================================================================
# 7. Maximum Tool Calls Limit (Infinite Loop Protection)
# ==============================================================================

def test_max_tool_calls_bounded():
    """Test 7: Force infinite tool loop and verify MAX_TOOL_CALLS stops safely."""
    # Endless tool call generator
    infinite_responses = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": f"call_loop_{i}",
                "type": "function",
                "function": {
                    "name": "configure_structure",
                    "arguments": json.dumps({"system_type": "sdof", "T": 0.5}),
                }
            }]
        }
        for i in range(15)
    ]
    # Final response after limit reached
    infinite_responses.append({
        "role": "assistant",
        "content": "Execution terminated because tool call limit was reached.",
    })

    mock_client = MockLLMClient(infinite_responses)
    orchestrator = SeismoOrchestrator(llm_client=mock_client, max_tool_calls=3)
    res = orchestrator.run("Loop test.")

    assert res.status == "limit_reached"
    assert len(res.tool_trace) <= 4


# ==============================================================================
# 8. Scientific Honesty: Uncertainty Handled Without Fabrication
# ==============================================================================

def test_scientific_honesty_unquantified_uncertainty():
    """Test 8: Verify 'NOT_QUANTIFIED' uncertainty status is preserved without fabricated confidence."""
    mock_client = MockLLMClient([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_ood",
                "type": "function",
                "function": {
                    "name": "assess_ood_and_uncertainty",
                    "arguments": json.dumps({
                        "structure": {"system_type": "sdof", "T": 0.5, "zeta": 0.05},
                        "pga_g": 0.40,
                    }),
                }
            }]
        },
        {
            "role": "assistant",
            "content": "The system is in-distribution. Epistemic uncertainty status is NOT_QUANTIFIED.",
        }
    ])

    orchestrator = SeismoOrchestrator(llm_client=mock_client)
    res = orchestrator.run("Check uncertainty for T=0.5s, PGA=0.4g.")

    assert "assess_ood_and_uncertainty" in res.tool_results
    ood_data = res.tool_results["assess_ood_and_uncertainty"]
    assert ood_data["uncertainty_status"] == "NOT_QUANTIFIED"
    assert "NOT_QUANTIFIED" in res.final_response
    # Assert no fabricated percentages in response
    assert "95% confident" not in res.final_response
    assert "90% certain" not in res.final_response


# ==============================================================================
# 9. Out-of-Distribution Warning Propagation
# ==============================================================================

def test_ood_warning_propagation():
    """Test 9: Verify out-of-distribution input triggers an explicit warning."""
    mock_client = MockLLMClient([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_ood_extreme",
                "type": "function",
                "function": {
                    "name": "assess_ood_and_uncertainty",
                    "arguments": json.dumps({
                        "structure": {"system_type": "sdof", "T": 10.0, "zeta": 0.05},
                        "pga_g": 5.0,
                    }),
                }
            }]
        },
        {
            "role": "assistant",
            "content": "WARNING: The parameters T=10.0s and PGA=5.0g are OUT-OF-DISTRIBUTION. SeismoFNO surrogate inference cannot be verified.",
        }
    ])

    orchestrator = SeismoOrchestrator(llm_client=mock_client)
    res = orchestrator.run("Analyze skyscraper T=10s under 5.0g shaking.")

    assert "assess_ood_and_uncertainty" in res.tool_results
    assert res.tool_results["assess_ood_and_uncertainty"]["is_ood"] is True
    assert "OUT-OF-DISTRIBUTION" in res.final_response


# ==============================================================================
# 10. Secrets Masking
# ==============================================================================

def test_secrets_redacted_from_telemetry():
    """Test 10: Verify API keys or sensitive arguments are scrubbed from execution trace."""
    mock_client = MockLLMClient([
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_sec",
                "type": "function",
                "function": {
                    "name": "configure_structure",
                    "arguments": json.dumps({
                        "system_type": "sdof",
                        "T": 0.5,
                        "api_key": "secret_key_12345",
                    }),
                }
            }]
        },
        {
            "role": "assistant",
            "content": "Configured structure.",
        }
    ])

    orchestrator = SeismoOrchestrator(llm_client=mock_client)
    res = orchestrator.run("Test secret masking.")

    trace_json = json.dumps(res.model_dump())
    assert "secret_key_12345" not in trace_json


# ==============================================================================
# 11. Optional Live Nebius API Smoke Test
# ==============================================================================

@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_NEBIUS"),
    reason="SKIPPED — Live remote tests require RUN_LIVE_NEBIUS=1.",
)
def test_live_nebius_smoke():
    """
    Live smoke test that executes against Nebius Token Factory ONLY if NEBIUS_API_KEY is present.
    Verifies authentication, connectivity, and basic Nemotron response.
    """
    client = NemotronClient()
    res = client.test_connection()
    assert res["status"] == "success"
    assert len(res["response"]) > 0
