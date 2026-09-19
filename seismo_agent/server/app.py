"""
seismo_agent/server/app.py — FastAPI Production Application Gateway for SeismoAgent.

Exposes REST and WebSocket endpoints for autonomous structural engineering analysis:
  - GET  /api/agent/health : Health check, device status, and Nebius configuration state.
  - GET  /api/agent/tools  : Introspection of the 8 registered deterministic tools.
  - POST /api/agent/run    : Synchronous/bounded agent execution with full audit trace.
  - WS   /api/agent/stream : Real-time execution progress and telemetry streaming.
  - GET  /api/presets      : Ground motion presets discovery.
  - POST /api/simulate     : Direct synchronized FNO + OpenSeesPy physics reference.
  - POST /api/sweep        : Parametric structural/earthquake sweep.
  - POST /api/report       : Automated civil engineering compliance report.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, Request, Depends, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles

from seismo_agent.config import (
    REPO_ROOT,
    NEBIUS_API_KEY,
    NEBIUS_MODEL,
    get_default_device,
)
from seismo_agent.server.schemas import (
    HealthResponse,
    ToolsResponse,
    ToolInfo,
    AgentRunRequest,
    AgentRunResponse,
    ErrorResponse,
    ErrorDetail,
    PresetItem,
    PresetsResponse,
    SimulateRequest,
    SimulateResponse,
    SimulationMetrics,
    OODSummary,
    SweepApiRequest,
    SweepApiResponse,
    ReportApiRequest,
    ReportApiResponse,
)
from seismo_agent.schemas.inputs import (
    EarthquakeRequest,
    StructuralRequest,
    InferenceRequest,
    PhysicsSimulationRequest,
    ComparisonRequest,
    OODRequest,
    SweepRequest as ToolSweepRequest,
    ReportRequest as ToolReportRequest,
)
from seismo_agent.tools.earthquake_tool import EarthquakeTool
from seismo_agent.tools.structural_tool import StructuralTool
from seismo_agent.tools.inference_tool import InferenceTool
from seismo_agent.tools.physics_sim_tool import PhysicsSimTool
from seismo_agent.tools.comparison_tool import ComparisonTool
from seismo_agent.tools.ood_detector_tool import OODDetectorTool
from seismo_agent.tools.sweep_tool import SweepTool
from seismo_agent.tools.report_tool import ReportTool
from seismo_agent.server.dependencies import (
    generate_request_id,
    get_cached_tools,
    get_orchestrator,
    get_llm_client,
)
from seismo_agent.server.execution import execute_orchestrator_safe, GatewayTimeoutError
from seismo_agent.server.streaming import handle_agent_stream_websocket
from seismo_agent.core.orchestrator import SeismoOrchestrator


# Setup structured logger (never logs secrets)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("seismo_agent.server")


# Global runtime tracking for live Nebius verification
_NEBIUS_LIVE_VERIFIED: bool = False


def set_nebius_live_verified(status: bool = True):
    """Mark that a live Nebius call has completed successfully during runtime."""
    global _NEBIUS_LIVE_VERIFIED
    _NEBIUS_LIVE_VERIFIED = status


# -----------------------------------------------------------------------------
# Agent & Simulation Endpoints Router
# -----------------------------------------------------------------------------
agent_router = APIRouter()


@agent_router.get(
    "/api/agent/health",
    response_model=HealthResponse,
    summary="Service Health and Configuration Status",
    tags=["System"],
)
def health_check() -> HealthResponse:
    tools = get_cached_tools()
    llm = get_llm_client()
    device = str(get_default_device())

    return HealthResponse(
        status="ok",
        service="seismoagent",
        version="0.3.1",
        orchestrator="available",
        nebius_configured=llm.is_configured(),
        nebius_live_verified=_NEBIUS_LIVE_VERIFIED,
        model=NEBIUS_MODEL,
        research_core="read_only",
        tools_count=len(tools),
        device=device,
        concurrency_policy="process_local_serialization (single_worker_required)",
        opensees_execution="serialized_via_threading_lock",
    )


@agent_router.get(
    "/api/agent/tools",
    response_model=ToolsResponse,
    summary="Inspect Available Deterministic Engineering Tools",
    tags=["Tools"],
)
def list_tools() -> ToolsResponse:
    tools = get_cached_tools()
    specs: List[ToolInfo] = []
    for t in tools:
        spec = t.to_tool_spec()["function"]
        specs.append(
            ToolInfo(
                name=spec["name"],
                description=spec["description"],
                parameters=spec["parameters"],
            )
        )
    return ToolsResponse(count=len(specs), tools=specs)


@agent_router.post(
    "/api/agent/run",
    response_model=AgentRunResponse,
    responses={
        422: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Execute Autonomous Engineering Analysis Query",
    tags=["Agent"],
)
def run_agent(
    request_body: AgentRunRequest,
    orchestrator: SeismoOrchestrator = Depends(get_orchestrator),
) -> AgentRunResponse:
    request_id = generate_request_id()
    t0 = time.perf_counter()
    logger.info(f"Received agent run request: {request_id}")

    # Input bounding check
    if len(request_body.message.strip()) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(type="BadRequest", message="Query message cannot be empty."),
            ).model_dump(),
        )

    try:
        # Execute through worker pool with explicit timeout
        timeout_sec = float(os.environ.get("SEISMO_REQUEST_TIMEOUT", "120.0"))
        orch_res = execute_orchestrator_safe(
            orchestrator=orchestrator,
            user_query=request_body.message,
            timeout_seconds=timeout_sec,
        )
    except GatewayTimeoutError as e:
        logger.error(f"Request timeout ({request_id}): {e}")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(type="TimeoutError", message=str(e)),
            ).model_dump(),
        )
    except Exception as e:
        logger.error(f"Internal execution failure ({request_id}): {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(type="ExecutionError", message=f"Internal agent execution error: {e}"),
            ).model_dump(),
        )

    total_server_latency_ms = (time.perf_counter() - t0) * 1000.0

    if orch_res.status == "completed" and orch_res.total_llm_latency_ms > 0:
        set_nebius_live_verified(True)

    timing_data = {
        "llm_latency_ms": orch_res.total_llm_latency_ms,
        "tool_latency_ms": orch_res.total_tool_latency_ms,
        "orchestrator_latency_ms": orch_res.total_latency_ms,
        "total_server_latency_ms": round(total_server_latency_ms, 2),
    }

    # Convert tool traces to plain dictionaries
    traces = [t.model_dump() for t in orch_res.tool_trace]

    return AgentRunResponse(
        request_id=request_id,
        status=orch_res.status,
        final_response=orch_res.final_response,
        tool_trace=traces,
        tool_results=orch_res.tool_results,
        steps_used=orch_res.steps_used,
        errors=orch_res.errors,
        timing=timing_data,
    )


@agent_router.websocket("/api/agent/stream")
async def agent_stream_endpoint(
    websocket: WebSocket,
    orchestrator: SeismoOrchestrator = Depends(get_orchestrator),
):
    await handle_agent_stream_websocket(
        websocket=websocket,
        orchestrator=orchestrator,
    )


@agent_router.get("/api/health", response_model=HealthResponse, tags=["System"])
def standard_health():
    return health_check()


@agent_router.get("/api/presets", response_model=PresetsResponse, tags=["Engineering"])
def get_presets():
    presets_dir = Path(REPO_ROOT) / "data" / "raw" / "ground_motions"
    preset_items: List[PresetItem] = []
    if presets_dir.exists():
        for p in sorted(presets_dir.glob("*.AT2")):
            name = p.stem.replace("_", " ")
            preset_items.append(
                PresetItem(
                    id=p.name,
                    name=name,
                    path=str(p),
                    pga_g=0.35,
                    duration_s=20.48,
                    dt_s=0.01,
                )
            )
    # Always provide synthetic records for deterministic verification
    preset_items.insert(
        0,
        PresetItem(
            id="synthetic_ricker",
            name="Synthetic Ricker Wavelet (2 Hz)",
            path="",
            pga_g=0.35,
            duration_s=20.48,
            dt_s=0.01,
        ),
    )
    preset_items.insert(
        1,
        PresetItem(
            id="synthetic_sine",
            name="Harmonic Damped Sine (1 Hz)",
            path="",
            pga_g=0.35,
            duration_s=20.48,
            dt_s=0.01,
        ),
    )
    return PresetsResponse(presets=preset_items)


@agent_router.post("/api/simulate", response_model=SimulateResponse, tags=["Engineering"])
def simulate_structure(payload: SimulateRequest):
    t0 = time.perf_counter()
    try:
        eq_tool = EarthquakeTool()
        struct_tool = StructuralTool()
        ood_tool = OODDetectorTool()
        inf_tool = InferenceTool()
        phys_tool = PhysicsSimTool(eq_tool=eq_tool)
        comp_tool = ComparisonTool()

        # 1. Process Earthquake
        eq_req = EarthquakeRequest(
            record_id=payload.preset_id if not payload.custom_content else None,
            target_pga_g=payload.pga_g,
            custom_content=payload.custom_content,
            custom_filename=payload.custom_filename,
        )
        eq_res = eq_tool.execute(eq_req)

        # 2. Configure Structure
        st_req = StructuralRequest(
            system_type="sdof",
            T=payload.T,
            zeta=payload.zeta,
            material_type=payload.material_type,
            u_y=payload.u_y,
            alpha=payload.alpha,
            mass=payload.mass,
        )
        st_res = struct_tool.execute(st_req)

        # 3. Assess OOD / Trust Boundary
        ood_req = OODRequest(
            structure=st_req,
            pga_g=eq_res.pga_g,
            dt=eq_res.dt,
            duration_s=eq_res.duration_s,
        )
        ood_res = ood_tool.execute(ood_req)
        ood_summary = OODSummary(
            is_ood=ood_res.is_ood,
            domain_violations=ood_res.domain_violations,
            uncertainty_status="NOT_QUANTIFIED",
            recommendation=ood_res.recommendation,
        )

        # 4. Neural Operator Inference
        inf_req = InferenceRequest(earthquake=eq_req, structure=st_req)
        inf_res = inf_tool.execute(inf_req)

        # 5. Physics Simulation Reference
        phys_req = PhysicsSimulationRequest(
            earthquake=eq_req,
            structure=st_req,
            solver=payload.solver,
        )
        phys_res = phys_tool.execute(phys_req)

        # 6. Discrepancy & Agreement Comparison
        comp_req = ComparisonRequest(
            u_pred=inf_res.u_pred,
            u_ref=phys_res.u_gt,
            fr_pred=inf_res.fr_pred,
            fr_ref=phys_res.fr_gt,
            eh_pred=inf_res.eh_pred,
            eh_ref=phys_res.eh_gt,
            dt=inf_res.dt,
            u_y=payload.u_y,
            fno_runtime_ms=inf_res.runtime_ms,
            physics_runtime_ms=phys_res.runtime_ms,
        )
        comp_res = comp_tool.execute(comp_req)

        metrics = SimulationMetrics(
            err_u_rel_l2=comp_res.err_u_rel_l2_pct,
            err_fr_rel_l2=comp_res.err_fr_rel_l2_pct or 0.0,
            err_eh_rel_l2=comp_res.err_eh_rel_l2_pct or 0.0,
            err_umax_rel=comp_res.err_umax_rel_pct,
            u_max_gt_m=phys_res.u_max_m,
            u_max_pred_m=inf_res.u_max_m,
            ductility_mu=phys_res.ductility_mu,
            pga_g=eq_res.pga_g,
            t_fno_ms=inf_res.runtime_ms,
            t_gt_ms=phys_res.runtime_ms,
            speedup=comp_res.speedup_factor or 1.0,
            device=inf_res.device_used,
            phase_error_rad=comp_res.phase_error_rad,
            yield_time_error_ms=comp_res.yield_time_error_ms,
            residual_drift_error_m=comp_res.residual_drift_error_m,
        )

        # Accelerations in g for plotting
        ag_ms2, _, _ = eq_tool.get_processed_acceleration_array(eq_req)
        ag_g = [float(a / 9.80665) for a in ag_ms2]

        return SimulateResponse(
            time=phys_res.time,
            ag=ag_g,
            u_gt=phys_res.u_gt,
            u_pred=inf_res.u_pred,
            fr_gt=phys_res.fr_gt,
            fr_pred=inf_res.fr_pred,
            eh_gt=phys_res.eh_gt,
            eh_pred=inf_res.eh_pred,
            metrics=metrics,
            ood=ood_summary,
            record_name=eq_res.record_name,
            status="success",
        )
    except Exception as e:
        logger.error(f"Simulation endpoint failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)},
        )


@agent_router.post("/api/sweep", response_model=SweepApiResponse, tags=["Engineering"])
def run_sweep_api(payload: SweepApiRequest):
    try:
        sweep_tool = SweepTool()
        eq_req = EarthquakeRequest(record_id=payload.preset_id, target_pga_g=payload.pga_g)
        st_req = StructuralRequest(
            system_type="sdof",
            T=payload.T,
            zeta=payload.zeta,
            material_type=payload.material_type,
            u_y=payload.u_y,
            alpha=payload.alpha,
        )
        sweep_req = ToolSweepRequest(
            earthquake=eq_req,
            base_structure=st_req,
            parameter_name=payload.param_name,
            parameter_values=payload.param_values,
            eval_fno=True,
            eval_physics=True,
        )
        res = sweep_tool.execute(sweep_req)
        return SweepApiResponse(
            param_name=payload.param_name,
            results=[p.model_dump() for p in res.points],
            total_evaluations=len(res.points),
        )
    except Exception as e:
        logger.error(f"Sweep endpoint failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@agent_router.post("/api/report", response_model=ReportApiResponse, tags=["Engineering"])
def generate_report_api(payload: ReportApiRequest):
    try:
        report_tool = ReportTool()
        eq_req = EarthquakeRequest(record_id=payload.preset_id, target_pga_g=payload.pga_g)
        st_req = StructuralRequest(
            system_type="sdof",
            T=payload.T,
            zeta=payload.zeta,
            material_type=payload.material_type,
            u_y=payload.u_y,
            alpha=payload.alpha,
        )
        report_req = ToolReportRequest(
            earthquake=eq_req,
            structure=st_req,
            eval_physics=True,
            format=payload.format,
        )
        res = report_tool.execute(report_req)
        return ReportApiResponse(
            report_title=res.title,
            content=res.content,
            format=res.format,
            metrics=res.executive_summary,
        )
    except Exception as e:
        logger.error(f"Report endpoint failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


def create_app() -> FastAPI:
    """Application factory for SeismoAgent Gateway."""
    app = FastAPI(
        title="SeismoAgent API Gateway",
        version="0.3.1",
        description=(
            "Autonomous AI Structural Engineering Analysis System connecting NVIDIA Nemotron "
            "with the SeismoFNO neural operator research core and OpenSeesPy physics simulation."
        ),
    )

    # 1. CORS Configuration
    raw_origins = os.environ.get(
        "SEISMO_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173,http://localhost:8000",
    )
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 2. Global Validation Exception Handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = generate_request_id()
        logger.warning(f"Request validation error ({req_id}): {exc}")
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                status="error",
                request_id=req_id,
                error=ErrorDetail(
                    type="ValidationError",
                    message=f"Request payload validation failed: {exc.errors()}",
                ),
            ).model_dump(),
        )

    # 3. Include Agent Router
    app.include_router(agent_router)

    # 4. Mount Production Static Frontend Bundle (SPA)
    dist_dir = REPO_ROOT / "frontend" / "dist"
    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

    return app


# Default ASGI application instance
app = create_app()
