"""
src/losses package initialization.
"""

from src.losses.data_loss import (
    RelativeL2Loss,
    MultiChannelRelativeL2Loss,
    CompositePhysicsLoss,
)
from src.losses.energy_consistency_loss import (
    EnergyConsistencyLoss,
    compute_cumulative_trapezoidal_work,
)
from src.losses.boundary_loss import BoundaryLoss

__all__ = [
    "RelativeL2Loss",
    "MultiChannelRelativeL2Loss",
    "CompositePhysicsLoss",
    "EnergyConsistencyLoss",
    "compute_cumulative_trapezoidal_work",
    "BoundaryLoss",
]
