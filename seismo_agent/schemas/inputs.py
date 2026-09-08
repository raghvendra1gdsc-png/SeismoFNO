"""
seismo_agent/schemas/inputs.py — Strongly typed Pydantic input schemas.

All requests are strictly validated before passing to deterministic tools.
"""

from typing import List, Optional, Literal, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class EarthquakeRequest(BaseModel):
    """Input payload for earthquake record ingestion, scaling, and characterization."""
    record_id: Optional[str] = Field(
        default=None,
        description="Preset record identifier from data/raw/ground_motions (e.g. 'RSN0001_Imperial_Valley-06.AT2') or 'synthetic_ricker'."
    )
    custom_content: Optional[str] = Field(
        default=None,
        description="Raw text content of an uploaded .AT2, .csv, or whitespace-separated table."
    )
    custom_filename: Optional[str] = Field(
        default=None,
        description="Filename for the uploaded custom record (e.g. 'el_centro.AT2')."
    )
    target_pga_g: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Target Peak Ground Acceleration in units of g for amplitude scaling (e.g. 0.40g)."
    )
    apply_baseline_correction: bool = Field(
        default=True,
        description="Whether to apply seismological baseline drift correction."
    )
    baseline_method: Literal["polynomial", "mean", "highpass"] = Field(
        default="polynomial",
        description="Baseline correction algorithm: 'polynomial', 'mean', or 'highpass'."
    )
    target_dt: float = Field(
        default=0.01,
        gt=0.0,
        description="Target uniform sampling time step in seconds (default: 0.01s = 100 Hz)."
    )
    n_steps: int = Field(
        default=2048,
        gt=0,
        description="Required number of discrete time points (default: 2048)."
    )
    raw_accel_series: Optional[List[float]] = Field(
        default=None,
        description="Direct array of acceleration values (in m/s^2 or g) if already parsed."
    )

    @model_validator(mode="after")
    def validate_source_provided(self) -> "EarthquakeRequest":
        has_preset = bool(self.record_id)
        has_custom = bool(self.custom_content and self.custom_filename)
        has_direct = bool(self.raw_accel_series and len(self.raw_accel_series) > 0)
        if not (has_preset or has_custom or has_direct):
            raise ValueError("EarthquakeRequest must specify either 'record_id', ('custom_content' + 'custom_filename'), or 'raw_accel_series'.")
        return self


class StructuralRequest(BaseModel):
    """Input payload for SDOF or MDOF structural dynamic configuration."""
    system_type: Literal["sdof", "mdof"] = Field(
        default="sdof",
        description="Type of structural oscillator/system: 'sdof' (single-degree) or 'mdof' (multi-story shear building)."
    )
    # SDOF parameters
    T: float = Field(
        default=0.5,
        gt=0.0,
        description="Natural elastic period in seconds (T > 0)."
    )
    zeta: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Viscous damping ratio (dimensionless, 0 <= zeta <= 1.0, typically 0.02 to 0.05)."
    )
    material_type: Literal["elastic", "bilinear"] = Field(
        default="bilinear",
        description="Constitutive material law: 'elastic' (linear) or 'bilinear' (kinematic hardening elastoplastic)."
    )
    u_y: Optional[float] = Field(
        default=0.015,
        description="Yield displacement in meters (required and > 0 for bilinear material)."
    )
    alpha: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Post-yield stiffness ratio k_post / k_0 (dimensionless, typically 0.02 to 0.10)."
    )
    mass: float = Field(
        default=1.0,
        gt=0.0,
        description="Structural lumped mass in kg (default 1.0 kg)."
    )
    damping_type: Literal["mass", "initial_stiffness"] = Field(
        default="mass",
        description="Damping matrix construction: 'mass' or 'initial_stiffness'."
    )
    # MDOF parameters
    n_stories: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of stories for MDOF system (e.g. 3 or 5)."
    )
    story_masses: Optional[List[float]] = Field(
        default=None,
        description="Lumped floor masses [m_1, ..., m_N] in kg."
    )
    story_heights: Optional[List[float]] = Field(
        default=None,
        description="Story heights [h_1, ..., h_N] in meters."
    )
    story_stiffnesses: Optional[List[float]] = Field(
        default=None,
        description="Inter-story lateral stiffnesses [k_1, ..., k_N] in N/m."
    )
    yield_displacements: Optional[List[float]] = Field(
        default=None,
        description="Inter-story yield drift thresholds [u_y,1, ..., u_y,N] in meters."
    )
    zeta_1: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Target Rayleigh damping ratio for Mode 1."
    )
    zeta_2: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Target Rayleigh damping ratio for Mode 2."
    )

    @model_validator(mode="after")
    def validate_yield_displacement(self) -> "StructuralRequest":
        if self.material_type == "bilinear":
            if self.system_type == "sdof":
                if self.u_y is None or self.u_y <= 0.0:
                    raise ValueError(f"Valid positive yield displacement 'u_y' is required for bilinear SDOF (got {self.u_y}).")
        return self


class InferenceRequest(BaseModel):
    """Input payload for SeismoFNO neural operator surrogate inference."""
    earthquake: EarthquakeRequest = Field(
        description="Earthquake acceleration record definition."
    )
    structure: StructuralRequest = Field(
        description="Structural oscillator configuration."
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to model checkpoint (.pt). If None, uses default verified checkpoint."
    )
    config_path: Optional[str] = Field(
        default=None,
        description="Path to experiment config YAML. If None, uses default research config."
    )
    normalizer_path: Optional[str] = Field(
        default=None,
        description="Path to normalizers (.pt). If None, resolves from checkpoint directory."
    )
    device: Optional[str] = Field(
        default=None,
        description="Target compute device: 'mps', 'cuda', or 'cpu'."
    )


class PhysicsSimulationRequest(BaseModel):
    """Input payload for deterministic OpenSeesPy or independent numerical simulation."""
    earthquake: EarthquakeRequest = Field(
        description="Earthquake acceleration record definition."
    )
    structure: StructuralRequest = Field(
        description="Structural oscillator configuration."
    )
    solver: Literal["opensees", "independent_newmark", "analytical"] = Field(
        default="opensees",
        description="Numerical solver engine: 'opensees' (OpenSeesPy NLTHA), 'independent_newmark' (pure NumPy Newmark-beta), or 'analytical' (closed form)."
    )
    u0: float = Field(
        default=0.0,
        description="Initial displacement at t=0 in meters."
    )
    v0: float = Field(
        default=0.0,
        description="Initial velocity at t=0 in m/s."
    )


class ComparisonRequest(BaseModel):
    """Input payload for evaluating discrepancy between surrogate predictions and reference physics."""
    u_pred: List[float] = Field(description="Surrogate predicted relative displacement trajectory.")
    u_ref: List[float] = Field(description="Ground truth reference relative displacement trajectory.")
    fr_pred: Optional[List[float]] = Field(default=None, description="Surrogate predicted restoring force trajectory.")
    fr_ref: Optional[List[float]] = Field(default=None, description="Ground truth reference restoring force trajectory.")
    eh_pred: Optional[List[float]] = Field(default=None, description="Surrogate predicted hysteretic energy trajectory.")
    eh_ref: Optional[List[float]] = Field(default=None, description="Ground truth reference hysteretic energy trajectory.")
    dt: float = Field(default=0.01, gt=0.0, description="Time step in seconds.")
    u_y: Optional[float] = Field(default=None, description="Yield displacement threshold for yield onset error.")
    fno_runtime_ms: Optional[float] = Field(default=None, description="Measured FNO inference latency in ms.")
    physics_runtime_ms: Optional[float] = Field(default=None, description="Measured physics solver latency in ms.")


class OODRequest(BaseModel):
    """Input payload for out-of-distribution (OOD) and domain boundary assessment."""
    structure: StructuralRequest = Field(
        description="Structural parameters to verify against the calibrated training envelope."
    )
    pga_g: float = Field(
        gt=0.0,
        description="Measured ground motion Peak Ground Acceleration in g."
    )
    dt: float = Field(
        default=0.01,
        gt=0.0,
        description="Ground motion sampling time step in seconds."
    )
    duration_s: Optional[float] = Field(
        default=20.48,
        description="Ground motion record duration in seconds."
    )
    config_path: Optional[str] = Field(
        default=None,
        description="Optional path to training domain YAML config (e.g. configs/data_generation.yaml)."
    )


class SweepRequest(BaseModel):
    """Input payload for parametric sensitivity sweeps / Incremental Dynamic Analysis."""
    parameter_name: Literal["pga_g", "T", "zeta", "u_y", "alpha", "mass"] = Field(
        description="Structural or ground motion parameter to sweep across."
    )
    parameter_values: List[float] = Field(
        min_length=2,
        description="List of parameter values to evaluate (e.g. PGAs [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])."
    )
    base_structure: StructuralRequest = Field(
        description="Baseline structural configuration held constant except for the swept parameter."
    )
    earthquake: EarthquakeRequest = Field(
        description="Base ground motion record applied across all sweep points."
    )
    eval_fno: bool = Field(
        default=True,
        description="Whether to execute SeismoFNO inference for each sweep point."
    )
    eval_physics: bool = Field(
        default=True,
        description="Whether to execute OpenSeesPy ground truth for each sweep point."
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional custom model checkpoint path."
    )


class ReportRequest(BaseModel):
    """Input payload for compiling a deterministic engineering audit report."""
    experiment_id: str = Field(
        default="seismo_analysis_001",
        description="Identifier for the analysis run or report."
    )
    title: Optional[str] = Field(
        default="Autonomous Seismic Structural Response Audit",
        description="Title for the engineering report."
    )
    earthquake_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output payload from EarthquakeTool."
    )
    structural_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output payload from StructuralTool."
    )
    inference_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output payload from InferenceTool."
    )
    physics_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output payload from PhysicsSimTool."
    )
    comparison_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output payload from ComparisonTool."
    )
    ood_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output payload from OODDetectorTool."
    )
    sweep_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output payload from SweepTool if a sweep was executed."
    )
    report_format: Literal["markdown", "json"] = Field(
        default="markdown",
        description="Desired output format: 'markdown' or 'json'."
    )
