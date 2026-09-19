"""
seismo_agent/core/orchestrator.py — Autonomous Engineering Reasoning & Tool Orchestrator.

Orchestrates NVIDIA Nemotron through the Nebius Token Factory API to dynamically plan,
dispatch deterministic engineering tools, inspect physics results, and narrate structural audits.
Enforces hard safety boundaries, Pydantic input validation, and zero numerical hallucination.
"""

import json
import time
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field

from seismo_agent.config import (
    MAX_TOOL_CALLS,
    MAX_ORCHESTRATION_STEPS,
    NEBIUS_API_KEY,
)
from seismo_agent.core.llm_client import LLMClientProtocol, NemotronClient, NebiusAPIError
from seismo_agent.core.prompts import SEISMO_AGENT_SYSTEM_PROMPT
from seismo_agent.tools.base import BaseTool, ToolExecutionError
from seismo_agent.tools import get_all_tools


class ToolCallTrace(BaseModel):
    """Execution telemetry and audit trace for a single tool call."""
    step: int = Field(description="Step index within the orchestration loop.")
    tool_name: str = Field(description="Name of the deterministic tool executed.")
    tool_input: Dict[str, Any] = Field(description="Sanitized input payload passed to the tool.")
    success: bool = Field(description="Whether tool execution completed without error.")
    runtime_ms: float = Field(description="Measured tool wall-clock execution time in milliseconds.")
    output_summary: Dict[str, Any] = Field(default_factory=dict, description="Condensed output payload or error summary.")


class OrchestrationResult(BaseModel):
    """Complete structured result returned by the autonomous orchestrator."""
    user_query: str = Field(description="Original user engineering prompt.")
    final_response: str = Field(description="Nemotron's synthesized engineering explanation.")
    tool_trace: List[ToolCallTrace] = Field(default_factory=list, description="Ordered audit trace of all tool calls.")
    tool_results: Dict[str, Any] = Field(default_factory=dict, description="Complete raw structured outputs keyed by tool name.")
    status: Literal["completed", "limit_reached", "error"] = Field(description="Workflow termination status.")
    steps_used: int = Field(description="Number of orchestration steps executed.")
    total_llm_latency_ms: float = Field(description="Cumulative time spent in Nemotron LLM inference.")
    total_tool_latency_ms: float = Field(description="Cumulative time spent in deterministic tool execution.")
    total_latency_ms: float = Field(description="End-to-end wall-clock workflow runtime in milliseconds.")
    errors: List[str] = Field(default_factory=list, description="List of caught non-fatal or fatal error messages.")


class SeismoOrchestrator:
    """
    Autonomous ReAct reasoning orchestrator connecting NVIDIA Nemotron with SeismoAgent tools.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClientProtocol] = None,
        tools: Optional[List[BaseTool]] = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_steps: int = MAX_ORCHESTRATION_STEPS,
        system_prompt: str = SEISMO_AGENT_SYSTEM_PROMPT,
    ):
        self.llm_client = llm_client or NemotronClient()
        self.tools = tools if tools is not None else get_all_tools()
        self.tools_by_name: Dict[str, BaseTool] = {t.name: t for t in self.tools}
        self.tools_spec = [t.to_tool_spec() for t in self.tools]
        self.max_tool_calls = max_tool_calls
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def _sanitize_secrets(self, data: Any) -> Any:
        """Deep-scrub sensitive values (API keys, bearer tokens) from telemetry traces."""
        if isinstance(data, dict):
            scrubbed = {}
            for k, v in data.items():
                if any(sec in k.lower() for sec in ["api_key", "token", "secret", "password", "authorization"]):
                    scrubbed[k] = "[REDACTED]"
                else:
                    scrubbed[k] = self._sanitize_secrets(v)
            return scrubbed
        elif isinstance(data, list):
            return [self._sanitize_secrets(item) for item in data]
        elif isinstance(data, str):
            if NEBIUS_API_KEY and len(NEBIUS_API_KEY) > 6 and NEBIUS_API_KEY in data:
                return data.replace(NEBIUS_API_KEY, "[REDACTED]")
            return data
        return data

    def run(
        self,
        user_query: str,
        event_callback: Optional[Any] = None,
    ) -> OrchestrationResult:
        """
        Execute autonomous reasoning and tool orchestration for a user engineering query.
        Optionally emits progress events to `event_callback` for live streaming.
        """
        def emit(event_dict: Dict[str, Any]):
            if event_callback and callable(event_callback):
                try:
                    event_callback(self._sanitize_secrets(event_dict))
                except Exception:
                    pass

        t_start = time.perf_counter()
        tool_trace: List[ToolCallTrace] = []
        tool_results: Dict[str, Any] = {}
        errors: List[str] = []

        total_llm_ms = 0.0
        total_tool_ms = 0.0
        tool_calls_count = 0
        step = 0
        status: Literal["completed", "limit_reached", "error"] = "completed"

        emit({"event": "request_started", "step": 0, "query": user_query})

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]

        final_response = ""

        while step < self.max_steps:
            step += 1
            emit({"event": "planning", "step": step})

            # 1. Invoke Nemotron Reasoning with Tools Spec
            t_llm0 = time.perf_counter()
            try:
                msg = self.llm_client.chat_completion(
                    messages=messages,
                    tools=self.tools_spec,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except Exception as e:
                err_msg = f"LLM client communication error at step {step}: {e}"
                errors.append(err_msg)
                status = "error"
                final_response = (
                    f"Execution halted due to LLM client failure: {e}. "
                    f"Deterministic tool results acquired so far are preserved."
                )
                break
            llm_step_ms = (time.perf_counter() - t_llm0) * 1000.0
            total_llm_ms += llm_step_ms

            tool_calls = msg.get("tool_calls", [])

            # 2. Check if Nemotron produced a final text response without tool calls
            if not tool_calls:
                final_response = msg.get("content", "").strip()
                status = "completed"
                break

            # Append assistant message with its tool calls
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)

            # 3. Process each tool call sequentially
            for call in tool_calls:
                tool_calls_count += 1
                call_id = call.get("id", f"call_{step}_{tool_calls_count}")
                fn = call.get("function", {})
                tool_name = fn.get("name", "")
                raw_args_str = fn.get("arguments", "{}")

                # Safety Limit Check
                if tool_calls_count > self.max_tool_calls:
                    status = "limit_reached"
                    warn_msg = f"Maximum tool call limit reached ({self.max_tool_calls}). Terminating workflow safely."
                    errors.append(warn_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps({"status": "limit_reached", "error": warn_msg}),
                    })
                    break

                # Parse arguments JSON
                try:
                    args_dict = json.loads(raw_args_str) if isinstance(raw_args_str, str) else raw_args_str
                    # Auto-deserialize stringified nested objects for Pydantic models
                    if isinstance(args_dict, dict):
                        for k, v in list(args_dict.items()):
                            if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
                                try:
                                    args_dict[k] = json.loads(v)
                                except Exception:
                                    pass
                except Exception as e:
                    err_msg = f"Invalid JSON arguments supplied for tool '{tool_name}': {e}"
                    errors.append(err_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps({"status": "error", "error": err_msg}),
                    })
                    tool_trace.append(ToolCallTrace(
                        step=step,
                        tool_name=tool_name,
                        tool_input={"raw_arguments": raw_args_str},
                        success=False,
                        runtime_ms=0.0,
                        output_summary={"error": err_msg},
                    ))
                    continue

                # Tool Whitelist Verification
                if tool_name not in self.tools_by_name:
                    err_msg = f"Tool '{tool_name}' is not a registered SeismoAgent engineering tool."
                    errors.append(err_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps({"status": "error", "error": err_msg}),
                    })
                    tool_trace.append(ToolCallTrace(
                        step=step,
                        tool_name=tool_name,
                        tool_input=self._sanitize_secrets(args_dict),
                        success=False,
                        runtime_ms=0.0,
                        output_summary={"error": err_msg},
                    ))
                    continue

                tool = self.tools_by_name[tool_name]

                # 4. Execute the Deterministic Engineering Tool
                emit({"event": "tool_started", "step": step, "tool": tool_name, "input": self._sanitize_secrets(args_dict)})
                t_tool0 = time.perf_counter()
                try:
                    result_obj = tool.execute(args_dict)
                    tool_runtime_ms = (time.perf_counter() - t_tool0) * 1000.0
                    total_tool_ms += tool_runtime_ms

                    result_dict = result_obj.model_dump()
                    tool_results[tool_name] = result_dict

                    # Create condensed summary for the trace
                    summary = {
                        k: v for k, v in result_dict.items()
                        if not isinstance(v, list) or len(v) <= 5
                    }

                    tool_trace.append(ToolCallTrace(
                        step=step,
                        tool_name=tool_name,
                        tool_input=self._sanitize_secrets(args_dict),
                        success=True,
                        runtime_ms=round(tool_runtime_ms, 2),
                        output_summary=self._sanitize_secrets(summary),
                    ))

                    emit({
                        "event": "tool_completed",
                        "step": step,
                        "tool": tool_name,
                        "status": "success",
                        "elapsed_ms": round(tool_runtime_ms, 2),
                        "summary": self._sanitize_secrets(summary),
                    })

                    # Feed result back into conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps(result_dict),
                    })

                except Exception as e:
                    tool_runtime_ms = (time.perf_counter() - t_tool0) * 1000.0
                    total_tool_ms += tool_runtime_ms
                    err_msg = f"Tool '{tool_name}' execution failed: {e}"
                    errors.append(err_msg)

                    tool_trace.append(ToolCallTrace(
                        step=step,
                        tool_name=tool_name,
                        tool_input=self._sanitize_secrets(args_dict),
                        success=False,
                        runtime_ms=round(tool_runtime_ms, 2),
                        output_summary={"error": str(e)},
                    ))

                    emit({
                        "event": "tool_failed",
                        "step": step,
                        "tool": tool_name,
                        "status": "error",
                        "elapsed_ms": round(tool_runtime_ms, 2),
                        "error": str(e),
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps({"status": "error", "error": str(e)}),
                    })

            if status == "limit_reached":
                # Give Nemotron one final turn to explain what was accomplished without tool calls
                emit({"event": "synthesis_started", "step": step, "reason": "limit_reached"})
                try:
                    final_msg = self.llm_client.chat_completion(
                        messages=messages,
                        tools=None,
                        temperature=0.2,
                    )
                    final_response = final_msg.get("content", "").strip()
                except Exception:
                    final_response = "Workflow reached the maximum tool call limit. Partial results are preserved."
                break

        if step >= self.max_steps and not final_response:
            status = "limit_reached"
            final_response = f"Workflow stopped because the maximum step limit ({self.max_steps}) was reached."

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        emit({
            "event": "completed",
            "status": status,
            "steps_used": step,
            "final_response": final_response,
            "total_latency_ms": round(total_latency_ms, 2),
        })

        return OrchestrationResult(
            user_query=user_query,
            final_response=final_response,
            tool_trace=tool_trace,
            tool_results=tool_results,
            status=status,
            steps_used=step,
            total_llm_latency_ms=round(total_llm_ms, 2),
            total_tool_latency_ms=round(total_tool_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            errors=errors,
        )
