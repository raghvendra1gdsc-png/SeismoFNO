"""
seismo_agent/schemas — Strongly typed Pydantic models for tool inputs and outputs.
"""

from seismo_agent.schemas.inputs import (
    EarthquakeRequest,
    StructuralRequest,
    InferenceRequest,
    PhysicsSimulationRequest,
    ComparisonRequest,
    OODRequest,
    SweepRequest,
    ReportRequest,
)

from seismo_agent.schemas.outputs import (
    EarthquakeResult,
    StructuralResult,
    InferenceResult,
    PhysicsResult,
    ComparisonResult,
    OODResult,
    SweepResult,
    SweepPointResult,
    ReportResult,
)

__all__ = [
    "EarthquakeRequest",
    "StructuralRequest",
    "InferenceRequest",
    "PhysicsSimulationRequest",
    "ComparisonRequest",
    "OODRequest",
    "SweepRequest",
    "ReportRequest",
    "EarthquakeResult",
    "StructuralResult",
    "InferenceResult",
    "PhysicsResult",
    "ComparisonResult",
    "OODResult",
    "SweepResult",
    "SweepPointResult",
    "ReportResult",
]
