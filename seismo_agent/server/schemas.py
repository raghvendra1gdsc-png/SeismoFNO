"""
seismo_agent/server/schemas.py — Request & Response Schemas for SeismoAgent API Gateway.

Strongly typed Pydantic v2 schemas defining the REST and WebSocket contracts.
Enforces request bounding, rejects untrusted input, and guarantees JSON serializability.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Payload for GET /api/agent/health."""
    status: str = Field(default="ok", description="Overall gateway health status.")
    service: str = Field(default="seismoagent", description="Service identifier.")
    version: str = Field(default="0.3.1", description="Gateway release version.")
    orchestrator: str = Field(default="available", description="State of the reasoning orchestrator.")
    nebius_configured: bool = Field(description="True if NEBIUS_API_KEY environment variable is present.")
    nebius_live_verified: bool = Field(default=False, description="True only if a live request to Nebius has succeeded during runtime.")
    model: str = Field(description="NVIDIA Nemotron model identifier currently configured.")
    research_core: str = Field(default="read_only", description="Status of the protected scientific core.")
    tools_count: int = Field(description="Number of registered deterministic engineering tools.")
    device: str = Field(description="Target compute device (mps / cuda / cpu).")
    concurrency_policy: str = Field(
        default="process_local_serialization (single_worker_required)",
        description="Concurrency execution model for OpenSeesPy. Requires single-process deployment.",
    )
    opensees_execution: str = Field(
        default="serialized_via_threading_lock",
        description="Process-local mutex serialization of native OpenSees operations.",
    )


class ToolInfo(BaseModel):
    """Metadata schema for a registered deterministic engineering tool."""
    name: str = Field(description="Unique tool identifier.")
    description: str = Field(description="Detailed operational description for LLM reasoning.")
    parameters: Dict[str, Any] = Field(description="JSON schema defining valid parameters.")


class ToolsResponse(BaseModel):
    """Payload for GET /api/agent/tools."""
    count: int = Field(description="Total number of available tools.")
    tools: List[ToolInfo] = Field(description="List of registered tool specifications.")


class AgentRunRequest(BaseModel):
    """Payload for POST /api/agent/run."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural-language structural engineering query or analysis command.",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional execution context (e.g. preferred checkpoint, units).",
    )


class AgentRunResponse(BaseModel):
    """Payload returned by POST /api/agent/run."""
    request_id: str = Field(description="Unique UUID tracking this execution.")
    status: Literal["completed", "limit_reached", "error"] = Field(description="Final orchestration state.")
    final_response: str = Field(description="Nemotron's synthesized engineering explanation.")
    tool_trace: List[Dict[str, Any]] = Field(description="Chronological audit trace of tool executions.")
    tool_results: Dict[str, Any] = Field(description="Complete raw JSON outputs from all executed tools.")
    steps_used: int = Field(description="Number of reasoning/tool-execution cycles.")
    errors: List[str] = Field(default_factory=list, description="Non-fatal or fatal error notices.")
    timing: Dict[str, float] = Field(description="Latency breakdown in milliseconds.")


class ErrorDetail(BaseModel):
    type: str = Field(description="Machine-readable error classification.")
    message: str = Field(description="Safe error message (secrets redacted).")


class ErrorResponse(BaseModel):
    status: str = "error"
    request_id: str
    error: ErrorDetail


class StreamEvent(BaseModel):
    """Typed event emitted over WebSocket WS /api/agent/stream."""
    event: Literal[
        "request_started",
        "planning",
        "tool_started",
        "tool_completed",
        "tool_failed",
        "synthesis_started",
        "completed",
        "error",
    ]
    request_id: str
    step: Optional[int] = None
    tool: Optional[str] = None
    status: Optional[str] = None
    elapsed_ms: Optional[float] = None
    summary: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# ==============================================================================
# Simulation & Workspace Schemas for Frontend Copilot Workstation
# ==============================================================================

class PresetItem(BaseModel):
    """Ground motion preset metadata for earthquake selection."""
    id: str
    name: str
    path: str
    pga_g: Optional[float] = None
    duration_s: Optional[float] = None
    dt_s: Optional[float] = None


class PresetsResponse(BaseModel):
    """Payload for GET /api/presets."""
    presets: List[PresetItem]


class SimulateRequest(BaseModel):
    """Payload for POST /api/simulate."""
    preset_id: Optional[str] = Field(default="RSN0001_Imperial_Valley-06.AT2", description="Preset earthquake identifier.")
    pga_g: float = Field(default=0.40, gt=0.0, description="Target Peak Ground Acceleration in g.")
    T: float = Field(default=0.50, gt=0.0, description="Fundamental period in seconds.")
    zeta: float = Field(default=0.05, ge=0.0, description="Viscous damping ratio.")
    material_type: Literal["bilinear", "elastic"] = Field(default="bilinear", description="Constitutive material type.")
    u_y: float = Field(default=0.010, gt=0.0, description="Yield displacement in meters.")
    alpha: float = Field(default=0.05, ge=0.0, le=1.0, description="Post-yield stiffness ratio.")
    mass: float = Field(default=1.0, gt=0.0, description="Lumped oscillator mass in kg.")
    custom_content: Optional[str] = Field(default=None, description="Uploaded file text content.")
    custom_filename: Optional[str] = Field(default=None, description="Uploaded filename.")
    solver: Literal["opensees", "newmark"] = Field(default="opensees", description="Physics verification solver.")


class SimulationMetrics(BaseModel):
    """Dense technical engineering metrics comparing FNO surrogate vs physics reference."""
    err_u_rel_l2: float
    err_fr_rel_l2: float
    err_eh_rel_l2: float
    err_umax_rel: float
    u_max_gt_m: float
    u_max_pred_m: float
    ductility_mu: float
    pga_g: float
    t_fno_ms: float
    t_gt_ms: float
    speedup: float
    device: str
    phase_error_rad: Optional[float] = None
    yield_time_error_ms: Optional[float] = None
    residual_drift_error_m: Optional[float] = None


class OODSummary(BaseModel):
    """Trust and domain boundary evaluation."""
    is_ood: bool
    domain_violations: List[str]
    uncertainty_status: str = "NOT_QUANTIFIED"
    recommendation: str


class SimulateResponse(BaseModel):
    """Payload for POST /api/simulate."""
    time: List[float]
    ag: List[float]
    u_gt: List[float]
    u_pred: List[float]
    fr_gt: List[float]
    fr_pred: List[float]
    eh_gt: List[float]
    eh_pred: List[float]
    metrics: SimulationMetrics
    ood: OODSummary
    record_name: str
    status: str = "success"
    error: Optional[str] = None


class SweepApiRequest(BaseModel):
    """Payload for POST /api/sweep."""
    param_name: Literal["pga_g", "T", "zeta", "u_y", "alpha", "mass"]
    param_values: List[float]
    preset_id: str = "RSN0001_Imperial_Valley-06.AT2"
    pga_g: float = 0.40
    T: float = 0.50
    zeta: float = 0.05
    material_type: Literal["bilinear", "elastic"] = "bilinear"
    u_y: float = 0.015
    alpha: float = 0.05


class SweepApiResponse(BaseModel):
    """Payload for POST /api/sweep."""
    param_name: str
    results: List[Dict[str, Any]]
    total_evaluations: int


class ReportApiRequest(BaseModel):
    """Payload for POST /api/report."""
    preset_id: str = "RSN0001_Imperial_Valley-06.AT2"
    pga_g: float = 0.40
    T: float = 0.50
    zeta: float = 0.05
    material_type: Literal["bilinear", "elastic"] = "bilinear"
    u_y: float = 0.015
    alpha: float = 0.05
    format: Literal["markdown", "json"] = "markdown"


class ReportApiResponse(BaseModel):
    """Payload for POST /api/report."""
    report_title: str
    content: str
    format: str
    metrics: Dict[str, Any]

