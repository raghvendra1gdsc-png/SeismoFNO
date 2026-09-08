"""
seismo_agent/schemas/outputs.py — Strongly typed Pydantic output schemas.

Every deterministic tool produces a validated, JSON-serializable result payload.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class EarthquakeResult(BaseModel):
    """Structured output from earthquake record ingestion and intensity measure analysis."""
    record_id: str = Field(description="Record identifier string.")
    record_name: str = Field(description="Human-readable event or station name.")
    n_points: int = Field(description="Total number of discrete time points.")
    dt: float = Field(description="Sampling time step in seconds.")
    duration_s: float = Field(description="Total record duration in seconds.")
    pga_g: float = Field(description="Peak Ground Acceleration in units of g.")
    pga_ms2: float = Field(description="Peak Ground Acceleration in m/s^2.")
    pgv_ms: float = Field(description="Peak Ground Velocity in m/s.")
    pgd_m: float = Field(description="Peak Ground Displacement in meters.")
    arias_intensity_ms: float = Field(description="Arias Intensity in m/s.")
    significant_duration_d5_95_s: float = Field(description="5-95% significant duration in seconds.")
    scale_factor_applied: float = Field(default=1.0, description="Amplitude scaling factor applied.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Header metadata extracted from file.")
    accel_preview: List[float] = Field(default_factory=list, description="Downsampled preview of acceleration series (max 100 points).")


class StructuralResult(BaseModel):
    """Structured output from structural dynamic parameter setup and modal analysis."""
    system_type: str = Field(description="'sdof' or 'mdof'.")
    T_n: float = Field(description="Fundamental elastic period in seconds.")
    omega_n: float = Field(description="Fundamental circular frequency in rad/s.")
    k0: float = Field(description="Initial lateral stiffness in N/m.")
    zeta: float = Field(description="Damping ratio.")
    material_type: str = Field(description="'elastic' or 'bilinear'.")
    u_y: Optional[float] = Field(default=None, description="Yield displacement in meters.")
    Fy: Optional[float] = Field(default=None, description="Yield lateral force capacity in Newtons.")
    alpha: float = Field(description="Post-yield stiffness ratio.")
    mass: float = Field(description="Total lumped mass in kg.")
    damping_type: str = Field(description="Damping formulation used.")
    # MDOF modal properties (if applicable)
    n_stories: Optional[int] = Field(default=None, description="Number of stories for MDOF.")
    modal_periods: Optional[List[float]] = Field(default=None, description="Modal periods [T_1, ..., T_N] in seconds.")
    modal_frequencies: Optional[List[float]] = Field(default=None, description="Modal frequencies [omega_1, ..., omega_N] in rad/s.")
    mode_shapes: Optional[List[List[float]]] = Field(default=None, description="Mass-normalized mode shape matrix.")
    rayleigh_alpha: Optional[float] = Field(default=None, description="Rayleigh mass damping coefficient alpha_M.")
    rayleigh_beta: Optional[float] = Field(default=None, description="Rayleigh stiffness damping coefficient beta_K.")


class InferenceResult(BaseModel):
    """Structured output from SeismoFNO surrogate model inference."""
    status: str = Field(description="'success' or 'error'.")
    model_architecture: str = Field(description="Model family, e.g. 'FNO1d' or 'FNO2d'.")
    checkpoint_used: str = Field(description="Path to model weights checkpoint.")
    device_used: str = Field(description="Compute device: 'mps', 'cuda', or 'cpu'.")
    runtime_ms: float = Field(description="Measured forward-pass wall-clock time in milliseconds.")
    n_points: int = Field(description="Number of time history points.")
    dt: float = Field(description="Time step in seconds.")
    u_max_m: float = Field(description="Peak predicted relative displacement in meters.")
    fr_max_n: float = Field(description="Peak predicted restoring force in Newtons.")
    eh_total_j: float = Field(description="Total cumulative dissipated hysteretic energy in Joules.")
    ductility_mu: float = Field(description="Predicted ductility demand ratio u_max / u_y.")
    time: List[float] = Field(description="Time coordinate array.")
    u_pred: List[float] = Field(description="Predicted relative displacement time history.")
    fr_pred: List[float] = Field(description="Predicted restoring force time history.")
    eh_pred: List[float] = Field(description="Predicted cumulative hysteretic energy time history.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic model details.")


class PhysicsResult(BaseModel):
    """Structured output from OpenSeesPy or independent numerical simulation."""
    status: str = Field(description="'success' or 'error'.")
    solver_used: str = Field(description="Solver engine identifier (e.g. 'OpenSeesPy NLTHA').")
    runtime_ms: float = Field(description="Measured simulation wall-clock time in milliseconds.")
    n_points: int = Field(description="Number of time history points.")
    dt: float = Field(description="Time step in seconds.")
    u_max_m: float = Field(description="Peak ground truth relative displacement in meters.")
    fr_max_n: float = Field(description="Peak ground truth restoring force in Newtons.")
    eh_total_j: float = Field(description="Total cumulative dissipated hysteretic energy in Joules.")
    ductility_mu: float = Field(description="Ground truth ductility demand ratio u_max / u_y.")
    time: List[float] = Field(description="Time coordinate array.")
    u_gt: List[float] = Field(description="Ground truth relative displacement time history.")
    v_gt: List[float] = Field(default_factory=list, description="Ground truth relative velocity time history.")
    a_gt: List[float] = Field(default_factory=list, description="Ground truth relative acceleration time history.")
    fr_gt: List[float] = Field(description="Ground truth restoring force time history.")
    eh_gt: List[float] = Field(description="Ground truth cumulative hysteretic energy time history.")
    energy_balance_residual: Optional[float] = Field(default=None, description="Energy conservation residual.")


class ComparisonResult(BaseModel):
    """Structured output from cross-validation comparison and discrepancy analysis."""
    status: str = Field(description="'success' or 'error'.")
    err_u_rel_l2_pct: float = Field(description="Relative L2 displacement trajectory error in percent.")
    err_fr_rel_l2_pct: Optional[float] = Field(default=None, description="Relative L2 restoring force trajectory error in percent.")
    err_eh_rel_l2_pct: Optional[float] = Field(default=None, description="Relative L2 hysteretic energy trajectory error in percent.")
    err_umax_rel_pct: float = Field(description="Relative error in peak absolute displacement |u_max - u_hat_max| / u_max * 100%.")
    err_frmax_rel_pct: Optional[float] = Field(default=None, description="Relative error in peak restoring force in percent.")
    phase_error_rad: Optional[float] = Field(default=None, description="Instantaneous Hilbert transform phase coherence discrepancy in radians.")
    yield_time_error_ms: Optional[float] = Field(default=None, description="Timing discrepancy in initial plastic yield onset in ms.")
    residual_drift_error_m: Optional[float] = Field(default=None, description="Permanent residual plastic drift discrepancy |u(T_end) - u_hat(T_end)| in meters.")
    fno_runtime_ms: Optional[float] = Field(default=None, description="Inference runtime in ms.")
    physics_runtime_ms: Optional[float] = Field(default=None, description="Physics solver runtime in ms.")
    speedup_factor: Optional[float] = Field(default=None, description="Measured speedup factor: physics_runtime / fno_runtime.")
    agreement_summary: str = Field(description="Factual technical summary of surrogate fidelity.")


class OODResult(BaseModel):
    """Structured output from domain boundary and uncertainty evaluation."""
    status: str = Field(description="'success' or 'error'.")
    is_ood: bool = Field(description="True if any input feature violates the calibrated training distribution boundaries.")
    within_structural_domain: bool = Field(description="True if period, damping, yield point, and post-yield stiffness are within bounds.")
    within_ground_motion_domain: bool = Field(description="True if PGA, sampling rate, and duration are within bounds.")
    resolution_supported: bool = Field(description="True if time step dt is supported by zero-shot resolution invariance.")
    domain_violations: List[str] = Field(default_factory=list, description="Explicit list of parameters falling outside the training envelope.")
    domain_warnings: List[str] = Field(default_factory=list, description="Advisory notices on near-boundary or extreme ductility conditions.")
    uncertainty_status: str = Field(
        default="NOT_QUANTIFIED",
        description="Factual uncertainty status. Never manufactured: returns 'NOT_QUANTIFIED' or 'CALIBRATED_TRAINING_DISTRIBUTION'."
    )
    recommendation: str = Field(description="Concrete operational recommendation (e.g. valid for FNO surrogate vs. requires OpenSeesPy verification).")


class SweepPointResult(BaseModel):
    """Single data point in a parametric sweep."""
    parameter_value: float = Field(description="Value of the swept parameter.")
    u_max_fno: Optional[float] = Field(default=None, description="Peak displacement from FNO in meters.")
    u_max_physics: Optional[float] = Field(default=None, description="Peak displacement from physics in meters.")
    ductility_fno: Optional[float] = Field(default=None, description="Ductility demand from FNO.")
    ductility_physics: Optional[float] = Field(default=None, description="Ductility demand from physics.")
    rel_l2_u_pct: Optional[float] = Field(default=None, description="Relative L2 displacement discrepancy in percent.")
    fno_runtime_ms: Optional[float] = Field(default=None, description="FNO inference runtime in ms.")
    physics_runtime_ms: Optional[float] = Field(default=None, description="Physics solver runtime in ms.")


class SweepResult(BaseModel):
    """Structured output from parametric sensitivity sweeps / Incremental Dynamic Analysis."""
    experiment_id: str = Field(description="Experiment identifier.")
    parameter_name: str = Field(description="Name of the swept parameter.")
    values_tested: List[float] = Field(description="List of tested values.")
    num_cases: int = Field(description="Number of completed sweep points.")
    points: List[SweepPointResult] = Field(description="Detailed results per sweep point.")
    avg_speedup_factor: Optional[float] = Field(default=None, description="Average measured speedup across completed sweep points.")
    total_runtime_s: float = Field(description="Total sweep wall-clock runtime in seconds.")


class ReportResult(BaseModel):
    """Structured output from deterministic engineering audit report generation."""
    experiment_id: str = Field(description="Experiment identifier.")
    title: str = Field(description="Report title.")
    generated_at: str = Field(description="ISO timestamp of report generation.")
    content: str = Field(description="Full formatted report text (Markdown or JSON string).")
    format: str = Field(description="'markdown' or 'json'.")
    executive_summary: Dict[str, Any] = Field(description="Key numerical figures from the analysis.")
    limitations: List[str] = Field(description="Scientific limitations and epistemic boundaries.")
