"""
api/services/building_service.py — Structural Building Archetype Service.

Provides realistic building classifications, structural dynamics properties,
and ductility parameters derived from standard civil engineering codes (ASCE 7-22, IS 1893:2016).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass
class BuildingArchetype:
    """Predefined building structural archetype."""
    id: str
    name: str
    structural_system: str
    stories: int
    total_height_m: float
    story_height_m: float
    fundamental_period_s: float
    damping_ratio: float
    yield_displacement_m: float
    post_yield_ratio: float
    material_type: str
    seismic_design_category: str
    description: str


BUILDING_ARCHETYPES: List[BuildingArchetype] = [
    BuildingArchetype(
        id="BLD-RC-03",
        name="3-Story Reinforced Concrete Moment Frame",
        structural_system="Special RC Moment-Resisting Frame (SMRF)",
        stories=3,
        total_height_m=9.6,
        story_height_m=3.2,
        fundamental_period_s=0.45,
        damping_ratio=0.05,
        yield_displacement_m=0.012,
        post_yield_ratio=0.05,
        material_type="bilinear",
        seismic_design_category="D",
        description="Typical modern commercial low-rise structure engineered with ductile detailing per ACI 318 / IS 13920.",
    ),
    BuildingArchetype(
        id="BLD-STEEL-06",
        name="6-Story Steel Moment Frame",
        structural_system="Steel Special Moment Frame (SMF)",
        stories=6,
        total_height_m=22.5,
        story_height_m=3.75,
        fundamental_period_s=0.85,
        damping_ratio=0.03,
        yield_displacement_m=0.024,
        post_yield_ratio=0.03,
        material_type="bilinear",
        seismic_design_category="E",
        description="Mid-rise flexible steel frame with wide-flange columns and reduced beam sections (RBS) for plastic hinge formation.",
    ),
    BuildingArchetype(
        id="BLD-WALL-08",
        name="8-Story RC Shear Wall Building",
        structural_system="Special Reinforced Concrete Shear Wall",
        stories=8,
        total_height_m=25.6,
        story_height_m=3.2,
        fundamental_period_s=0.55,
        damping_ratio=0.05,
        yield_displacement_m=0.010,
        post_yield_ratio=0.08,
        material_type="bilinear",
        seismic_design_category="D",
        description="Stiff residential shear wall core building with high lateral stiffness and moderate ductility capacity.",
    ),
    BuildingArchetype(
        id="BLD-MASONRY-02",
        name="2-Story Unreinforced Masonry (URM)",
        structural_system="Unreinforced Brick Masonry bearing walls",
        stories=2,
        total_height_m=6.0,
        story_height_m=3.0,
        fundamental_period_s=0.20,
        damping_ratio=0.07,
        yield_displacement_m=0.003,
        post_yield_ratio=0.01,
        material_type="bilinear",
        seismic_design_category="C",
        description="High-vulnerability traditional masonry structure with brittle in-plane shear cracking and low displacement capacity.",
    ),
    BuildingArchetype(
        id="BLD-ISOLATED-05",
        name="5-Story Base-Isolated Hospital",
        structural_system="Lead-Rubber Bearing (LRB) Isolated Frame",
        stories=5,
        total_height_m=17.5,
        story_height_m=3.5,
        fundamental_period_s=2.20,
        damping_ratio=0.15,
        yield_displacement_m=0.065,
        post_yield_ratio=0.10,
        material_type="bilinear",
        seismic_design_category="F",
        description="Critical care healthcare facility protected by seismic elastomeric isolators shifting the fundamental period beyond dominant earthquake frequencies.",
    ),
]


class BuildingService:
    """Manages building structural archetypes."""

    def get_building_archetypes(self) -> List[Dict[str, Any]]:
        return [asdict(b) for b in BUILDING_ARCHETYPES]

    def get_archetype_by_id(self, archetype_id: str) -> Optional[BuildingArchetype]:
        for b in BUILDING_ARCHETYPES:
            if b.id == archetype_id:
                return b
        return None
