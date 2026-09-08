"""
api/main.py — FastAPI Application Gateway for SeismoFNO Digital-Twin Demonstrator.

Exposes REST endpoints:
  - GET  /health
  - GET  /api/v1/system/info
  - POST /api/v1/scenario/validate
  - POST /api/v1/scenario/predict
  - GET  /api/v1/earthquakes
  - GET  /api/v1/earthquakes/{record_id}
  - GET  /api/v1/buildings
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.adapter import SeismoFNOInferenceAdapter, InferenceOutput
from api.services.engineering_metrics import compute_engineering_metrics
from api.services.earthquake_service import EarthquakeService
from api.services.building_service import BuildingService
from api.schemas import (
    SystemInfoResponse,
    ScenarioInput,
    ScenarioValidationResult,
    ScenarioPredictionResponse,
    TrajectoryData,
    ValidationComparison,
)

# OpenSeesPy ground truth import (isolated execution)
from src.ground_truth.opensees_sdof_model import simulate_sdof, SDOFParams, SDOFResponse

# Global Thread Lock for OpenSeesPy C-runtime safety
OPENSEES_LOCK = threading.Lock()

# Initialize FastAPI Application
app = FastAPI(
    title="SeismoFNO Digital-Twin Application API",
    description="Production-grade engineering decision-support layer around the SeismoFNO surrogate model.",
    version="1.0.0",
)

# Enable CORS for local Vite development and Docker containers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
inference_adapter = SeismoFNOInferenceAdapter()
earthquake_service = EarthquakeService()
building_service = BuildingService()


# -----------------------------------------------------------------------------
# 1. Health & System Information Endpoints
# -----------------------------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Liveness and readiness probe."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "model_loaded": inference_adapter.is_loaded,
        "device": str(inference_adapter.device),
    }


@app.get("/api/v1/system/info", response_model=SystemInfoResponse)
def get_system_info() -> SystemInfoResponse:
    """Introspect active runtime, model checkpoint configuration, and compute device."""
    meta = inference_adapter.metadata()
    model_info = meta.get("model_info", {})
    return SystemInfoResponse(
        status="operational",
        version="1.0.0",
        model_loaded=inference_adapter.is_loaded,
        device=str(inference_adapter.device),
        model_parameters=model_info.get("parameters", 1196931 if inference_adapter.is_loaded else 0),
        checkpoint_path=str(inference_adapter.checkpoint_path),
        config_path=str(inference_adapter.config_path),
        modes=model_info.get("modes", 128),
        width=model_info.get("width", 48),
        is_mock=not inference_adapter.is_loaded,
        scientific_integrity="Strict — Frozen Research Core",
    )


# -----------------------------------------------------------------------------
# 2. Seismic & Structural Catalogs
# -----------------------------------------------------------------------------

@app.get("/api/v1/earthquakes")
def list_earthquakes(
    include_indian: bool = Query(True, description="Include historical Indian seismic catalog"),
    limit_peer: int = Query(50, description="Max PEER records to return"),
) -> Dict[str, Any]:
    """Retrieve verified ground motion records from PEER and Indian seismic catalogs."""
    peer_records = earthquake_service.get_peer_records(limit=limit_peer)
    indian_catalog = earthquake_service.get_indian_catalog() if include_indian else []
    return {
        "indian_catalog": indian_catalog,
        "peer_records": peer_records,
        "total_indian": len(indian_catalog),
        "total_peer": len(peer_records),
    }


@app.get("/api/v1/earthquakes/{record_id}")
def get_earthquake_details(record_id: str) -> Dict[str, Any]:
    """Get detailed ground motion information and initial acceleration preview."""
    try:
        ag, dt, name = earthquake_service.load_ground_motion(record_id, n_steps=1024)
        pga_g = float(np.max(np.abs(ag)) / 9.80665)
        time_arr = np.linspace(0, (len(ag) - 1) * dt, len(ag))
        return {
            "record_id": record_id,
            "earthquake_name": name,
            "dt": dt,
            "pga_g": round(pga_g, 4),
            "n_points": len(ag),
            "preview": {
                "time": time_arr[::4].tolist(),
                "ag": ag[::4].tolist(),
            }
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Ground motion record '{record_id}' not found: {str(exc)}")


@app.get("/api/v1/buildings")
def list_buildings() -> Dict[str, Any]:
    """Retrieve predefined civil engineering building archetypes."""
    archetypes = building_service.get_building_archetypes()
    return {
        "buildings": archetypes,
        "total": len(archetypes),
    }


# -----------------------------------------------------------------------------
# 3. Scenario Validation & Prediction
# -----------------------------------------------------------------------------

@app.post("/api/v1/scenario/validate", response_model=ScenarioValidationResult)
def validate_scenario(scenario: ScenarioInput) -> ScenarioValidationResult:
    """
    Validate scenario physical consistency and evaluate domain bounds.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Structural period sanity
    if scenario.T0 < 0.05 or scenario.T0 > 4.0:
        errors.append(f"Fundamental period T0={scenario.T0}s is outside valid physical range [0.05s, 4.0s].")

    # 2. Damping sanity
    if scenario.damping_ratio < 0.0 or scenario.damping_ratio > 0.30:
        errors.append(f"Damping ratio {scenario.damping_ratio} is outside [0.0, 0.30].")

    # 3. Yield displacement sanity
    if scenario.yield_displacement_m <= 0.0:
        errors.append("Yield displacement must be strictly positive.")

    # 4. Computed properties
    omega_n = 2.0 * np.pi / max(1e-4, scenario.T0)
    k0 = scenario.mass_kg * (omega_n ** 2)
    fy = k0 * scenario.yield_displacement_m if scenario.material_type == "bilinear" else float("inf")

    # 5. Training domain bounds checking (Phase 5 empirical training distribution)
    # Training bounds: T ∈ [0.1, 2.0]s, PGA ∈ [0.05, 1.2]g, zeta = 0.05
    domain_status = "IN_DOMAIN"
    if scenario.T0 < 0.10 or scenario.T0 > 2.0:
        warnings.append(f"Period T0={scenario.T0}s is outside the primary training regime [0.10s, 2.0s]. Operator will extrapolate.")
        domain_status = "EXTRAPOLATION_WARNING"
    if scenario.pga_g > 1.20:
        warnings.append(f"PGA={scenario.pga_g}g exceeds training bounds (max 1.2g). Severe plastic deformation expected.")
        domain_status = "EXTRAPOLATION_WARNING"
    if abs(scenario.damping_ratio - 0.05) > 0.05:
        warnings.append(f"Damping ratio {scenario.damping_ratio} differs substantially from standard 5% damping.")

    is_valid = len(errors) == 0
    return ScenarioValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        domain_status=domain_status,
        computed_properties={
            "omega_n_rad_s": round(omega_n, 4),
            "stiffness_k0_N_m": round(k0, 2),
            "yield_force_N": round(fy, 2) if fy != float("inf") else "elastic",
            "frequency_hz": round(omega_n / (2.0 * np.pi), 3),
        }
    )


@app.post("/api/v1/scenario/predict", response_model=ScenarioPredictionResponse)
def predict_scenario(scenario: ScenarioInput) -> ScenarioPredictionResponse:
    """
    Execute instant SeismoFNO digital-twin response prediction with optional OpenSeesPy ground truth.
    """
    # 1. Validate input scenario
    val_res = validate_scenario(scenario)
    if not val_res.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": val_res.errors, "warnings": val_res.warnings},
        )

    # 2. Retrieve / Preprocess Ground Motion
    n_steps = 2048
    try:
        if scenario.custom_record_content:
            # Parse custom file upload
            from dashboard.app import DashboardEngine
            engine_helper = DashboardEngine()
            ag, dt, rec_name = engine_helper.parse_uploaded_file(
                scenario.custom_record_content,
                filename="custom_record.AT2",
                target_dt=scenario.dt,
            )
            # Scale to requested PGA
            curr_pga = float(np.max(np.abs(ag)) / 9.80665)
            if scenario.pga_g > 0 and curr_pga > 1e-5:
                ag = ag * (scenario.pga_g / curr_pga)
        else:
            ag, dt, rec_name = earthquake_service.load_ground_motion(
                record_identifier=scenario.earthquake_id,
                target_pga_g=scenario.pga_g,
                target_dt=scenario.dt,
                n_steps=n_steps,
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ground motion loading failed: {str(exc)}")

    # 3. Execute SeismoFNO Surrogate Inference
    scenario_dict = {
        "T0": scenario.T0,
        "damping": scenario.damping_ratio,
        "yield_displacement": scenario.yield_displacement_m,
        "alpha": scenario.post_yield_ratio,
        "material_type": scenario.material_type,
        "mass": scenario.mass_kg,
        "pga_g": scenario.pga_g,
    }

    try:
        fno_out: InferenceOutput = inference_adapter.predict(
            scenario=scenario_dict,
            ag=ag,
            dt=dt,
            n_steps=n_steps,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Surrogate inference failed: {str(exc)}")

    # 4. Compute Engineering Metrics
    metrics_summary = compute_engineering_metrics(
        time=fno_out.time,
        ag=fno_out.ag,
        u=fno_out.u,
        fr=fno_out.fr,
        eh=fno_out.eh,
        v=fno_out.v,
        dt=dt,
        uy=scenario.yield_displacement_m,
        mass=scenario.mass_kg,
        height=scenario.building_height_m,
        T0=scenario.T0,
    )

    # 5. Optional Ground Truth Execution via OpenSeesPy
    validation_comp: Optional[ValidationComparison] = None
    u_gt_list: Optional[List[float]] = None
    fr_gt_list: Optional[List[float]] = None
    eh_gt_list: Optional[List[float]] = None

    if scenario.include_ground_truth:
        with OPENSEES_LOCK:
            t0_gt = time.perf_counter()
            gt_params = SDOFParams(
                T=scenario.T0,
                zeta=scenario.damping_ratio,
                material_type="bilinear" if scenario.material_type == "bilinear" else "elastic",
                u_y=scenario.yield_displacement_m if scenario.material_type == "bilinear" else None,
                alpha=scenario.post_yield_ratio,
                mass=scenario.mass_kg,
            )
            try:
                gt_resp: SDOFResponse = simulate_sdof(gt_params, ag=ag, dt=dt)
                opensees_ms = (time.perf_counter() - t0_gt) * 1000.0

                u_gt = gt_resp.u
                fr_gt = gt_resp.f_r
                eh_gt = gt_resp.e_h

                # Compute quantitative validation errors
                norm_u_gt = float(np.linalg.norm(u_gt) + 1e-7)
                norm_fr_gt = float(np.linalg.norm(fr_gt) + 1e-7)
                norm_eh_gt = float(np.linalg.norm(eh_gt) + 1e-7)

                rel_l2_u = float(np.linalg.norm(u_gt - fno_out.u) / norm_u_gt * 100.0)
                rel_l2_fr = float(np.linalg.norm(fr_gt - fno_out.fr) / norm_fr_gt * 100.0)
                rel_l2_eh = float(np.linalg.norm(eh_gt - fno_out.eh) / norm_eh_gt * 100.0)

                u_max_gt = float(np.max(np.abs(u_gt)))
                u_max_fno = float(np.max(np.abs(fno_out.u)))
                peak_err = float(abs(u_max_gt - u_max_fno) / max(1e-6, u_max_gt) * 100.0)
                rmse_mm = float(np.sqrt(np.mean((u_gt - fno_out.u) ** 2)) * 1000.0)
                speedup = float(opensees_ms / max(0.01, fno_out.latency_ms))

                validation_comp = ValidationComparison(
                    relative_l2_u_percent=round(rel_l2_u, 2),
                    relative_l2_fr_percent=round(rel_l2_fr, 2),
                    relative_l2_eh_percent=round(rel_l2_eh, 2),
                    peak_u_error_percent=round(peak_err, 2),
                    rmse_u_mm=round(rmse_mm, 3),
                    speedup_factor=round(speedup, 1),
                    fno_latency_ms=round(fno_out.latency_ms, 3),
                    opensees_latency_ms=round(opensees_ms, 3),
                    benchmark_status="Empirically Measured Ground Truth",
                )

                stride = scenario.stride
                u_gt_list = u_gt[::stride].tolist()
                fr_gt_list = fr_gt[::stride].tolist()
                eh_gt_list = eh_gt[::stride].tolist()
            except Exception as exc:
                print(f"[Warning] OpenSeesPy execution failed: {exc}")

    stride = scenario.stride
    trajectories = TrajectoryData(
        time=fno_out.time[::stride].tolist(),
        ag=fno_out.ag[::stride].tolist(),
        u=fno_out.u[::stride].tolist(),
        v=fno_out.v[::stride].tolist(),
        fr=fno_out.fr[::stride].tolist(),
        up=fno_out.up[::stride].tolist(),
        eh=fno_out.eh[::stride].tolist(),
        u_gt=u_gt_list,
        fr_gt=fr_gt_list,
        eh_gt=eh_gt_list,
    )

    return ScenarioPredictionResponse(
        scenario=scenario.model_dump(),
        trajectories=trajectories,
        metrics=metrics_summary.to_dict(),
        validation=validation_comp,
        inference_time_ms=round(fno_out.latency_ms, 3),
        is_mock=fno_out.is_mock,
        uncertainty_note="Uncertainty estimation: research module pending",
        model_provenance=fno_out.metadata,
    )


# -----------------------------------------------------------------------------
# 5. Professor-Facing Research Demonstration Endpoints (/api/v1/demo/*)
# -----------------------------------------------------------------------------

from pydantic import BaseModel, Field
from src.demo.model_registry import ModelRegistry
from src.demo.demo_data import DemoDataManager
from src.demo.metrics_adapter import MetricsAdapter
from src.demo.inference_adapter import simulate_and_compare


class DemoSimulationRequest(BaseModel):
    archetype_id: str = Field(default="3S_T035", description="Structural archetype identifier")
    record_id: str = Field(default="RSN0001", description="Earthquake record identifier")
    model_id: str = Field(default="exp6_t1_gno", description="Model architecture identifier")
    selected_floor: Optional[int] = Field(default=None, description="Specific floor index (1 to N, default Roof)")


@app.get("/api/v1/demo/models")
def get_demo_models() -> List[Dict[str, Any]]:
    """Return catalog of immutable research models from EXP4, EXP5, and EXP6."""
    return ModelRegistry.list_models()


@app.get("/api/v1/demo/structures")
def get_demo_structures() -> List[Dict[str, Any]]:
    """Return structural archetypes with precomputed theoretical modal properties."""
    return DemoDataManager.list_structures()


@app.get("/api/v1/demo/earthquakes")
def get_demo_earthquakes() -> List[Dict[str, Any]]:
    """Return verified PEER ground motion records."""
    return DemoDataManager.list_earthquakes()


@app.get("/api/v1/demo/earthquakes/live")
def get_live_earthquakes(
    minmagnitude: Optional[float] = Query(None, description="Minimum magnitude threshold (e.g. 2.5, 4.5)"),
    limit: int = Query(50, ge=1, le=200, description="Max earthquake events to return"),
    hours: Optional[float] = Query(24.0, description="Time window in hours"),
    latitude: Optional[float] = Query(None, description="Scenario latitude for distance calculation"),
    longitude: Optional[float] = Query(None, description="Scenario longitude for distance calculation"),
    radius_km: Optional[float] = Query(None, description="Radius in km from scenario coordinates"),
) -> Dict[str, Any]:
    """Retrieve normalized real-world earthquake events from USGS live feed with cached fallback."""
    from src.demo.usgs_client import usgs_client
    return usgs_client.fetch_live_feed(
        min_magnitude=minmagnitude,
        limit=limit,
        hours=hours,
        scenario_lat=latitude,
        scenario_lon=longitude,
        radius_km=radius_km,
    )


@app.get("/api/v1/demo/earthquakes/{event_id}")
def get_live_earthquake_detail(event_id: str) -> Dict[str, Any]:
    """Retrieve detailed metadata for a specific observed USGS earthquake event."""
    from src.demo.usgs_client import usgs_client
    detail = usgs_client.get_event_detail(event_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Earthquake event '{event_id}' not found in live feed or cache")
    return detail


@app.get("/api/v1/demo/progression")
def get_demo_progression() -> List[Dict[str, Any]]:
    """Return verified scientific research progression (EXP4 -> EXP5 -> EXP6 -> Boundary)."""
    return MetricsAdapter.get_research_progression()


@app.get("/api/v1/demo/ood-matrix")
def get_demo_ood_matrix() -> Dict[str, Any]:
    """Return out-of-distribution evaluation matrix across partitions."""
    return MetricsAdapter.get_ood_matrix()


@app.get("/api/v1/demo/ablation")
def get_demo_ablation() -> Dict[str, Any]:
    """Return shuffled-conditioning falsification ablation metrics."""
    return MetricsAdapter.get_ablation_falsification()


@app.get("/api/v1/demo/failures")
def get_demo_failures() -> List[Dict[str, Any]]:
    """Return documented first-class scientific failure modes and limitations."""
    return MetricsAdapter.get_failure_analysis()


@app.get("/api/v1/demo/benchmark")
def get_demo_benchmark() -> Dict[str, Any]:
    """Return measured inference speedup benchmarks on Apple Silicon GPU."""
    return MetricsAdapter.get_computational_benchmark()


@app.post("/api/v1/demo/simulate")
def run_demo_simulation(req: DemoSimulationRequest) -> Dict[str, Any]:
    """
    Execute simulation or retrieve verified response trajectory with live vs archival distinction.
    """
    try:
        return simulate_and_compare(
            archetype_id=req.archetype_id,
            record_id=req.record_id,
            model_id=req.model_id,
            selected_floor=req.selected_floor,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

