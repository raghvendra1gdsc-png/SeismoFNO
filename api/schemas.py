"""
api/schemas.py — Pydantic Schemas for SeismoFNO REST API.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class SystemInfoResponse(BaseModel):
    """System health, device runtime, and model status."""
    status: str
    version: str
    model_loaded: bool
    device: str
    model_parameters: int
    checkpoint_path: str
    config_path: str
    modes: int
    width: int
    is_mock: bool
    scientific_integrity: str = "Strict — Frozen Research Core"


class ScenarioInput(BaseModel):
    """Structural oscillator scenario definition with physical parameter bounds."""
    earthquake_id: str = Field(
        default="RSN0001_Imperial_Valley-06.AT2",
        description="PEER ground motion record filename or Indian seismic catalog ID.",
    )
    pga_g: float = Field(
        default=0.40,
        ge=0.01,
        le=5.0,
        description="Peak Ground Acceleration target in units of gravity (g).",
    )
    dt: float = Field(
        default=0.01,
        ge=0.001,
        le=0.05,
        description="Simulation time step in seconds.",
    )
    duration_sec: float = Field(
        default=20.48,
        ge=1.0,
        le=60.0,
        description="Total duration of simulation in seconds.",
    )
    T0: float = Field(
        default=0.50,
        ge=0.05,
        le=5.0,
        description="Fundamental elastic structural period T_n in seconds.",
    )
    damping_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=0.30,
        description="Viscous damping ratio zeta (dimensionless, e.g. 0.05 for 5%).",
    )
    stiffness: Optional[float] = Field(
        default=None,
        description="Initial elastic lateral stiffness k_0 in N/m (computed from T0 and mass if None).",
    )
    yield_displacement_m: float = Field(
        default=0.010,
        gt=0.0,
        le=2.0,
        description="Structural yield displacement u_y in meters.",
    )
    post_yield_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Post-yield stiffness ratio alpha (dimensionless, slope ratio k_p / k_0).",
    )
    material_type: Literal["bilinear", "elastic"] = Field(
        default="bilinear",
        description="Constitutive material law: 'bilinear' (elastoplastic) or 'elastic'.",
    )
    mass_kg: float = Field(
        default=1.0,
        gt=0.0,
        description="Tributary structural mass in kilograms.",
    )
    building_height_m: Optional[float] = Field(
        default=3.5,
        gt=0.5,
        le=200.0,
        description="Building tributary height for inter-story drift calculation in meters.",
    )
    custom_record_content: Optional[str] = Field(
        default=None,
        description="Raw string content of uploaded ground motion (.AT2, .csv, or whitespace numbers).",
    )
    include_ground_truth: bool = Field(
        default=False,
        description="If True, execute concurrent OpenSeesPy NLTHA ground truth for exact validation comparison.",
    )
    stride: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Downsampling stride factor for returned time history trajectories.",
    )


class ScenarioValidationResult(BaseModel):
    """Validation response indicating scenario physical consistency."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    domain_status: str = Field(
        default="IN_DOMAIN",
        description="'IN_DOMAIN' or 'EXTRAPOLATION_WARNING'",
    )
    computed_properties: Dict[str, Any] = Field(default_factory=dict)


class TrajectoryData(BaseModel):
    """Time history response trajectories."""
    time: List[float]
    ag: List[float]
    u: List[float]
    v: List[float]
    fr: List[float]
    up: List[float]
    eh: List[float]

    # Optional Ground Truth trajectories (when include_ground_truth=True)
    u_gt: Optional[List[float]] = None
    fr_gt: Optional[List[float]] = None
    eh_gt: Optional[List[float]] = None


class ValidationComparison(BaseModel):
    """Comparison metrics between SeismoFNO prediction and OpenSeesPy ground truth."""
    relative_l2_u_percent: float
    relative_l2_fr_percent: float
    relative_l2_eh_percent: float
    peak_u_error_percent: float
    rmse_u_mm: float
    speedup_factor: float
    fno_latency_ms: float
    opensees_latency_ms: float
    benchmark_status: str = "Empirically Measured Ground Truth"


class ScenarioPredictionResponse(BaseModel):
    """Comprehensive digital-twin prediction response."""
    scenario: Dict[str, Any]
    trajectories: TrajectoryData
    metrics: Dict[str, Any]
    validation: Optional[ValidationComparison] = None
    inference_time_ms: float
    is_mock: bool
    uncertainty_note: str = "Uncertainty estimation: research module pending"
    model_provenance: Dict[str, Any]
