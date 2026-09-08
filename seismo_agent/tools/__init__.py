"""
seismo_agent/tools — Unified deterministic engineering tool harness for SeismoAgent.
"""

from typing import List
from seismo_agent.tools.base import BaseTool, ToolExecutionError
from seismo_agent.tools.earthquake_tool import EarthquakeTool
from seismo_agent.tools.structural_tool import StructuralTool
from seismo_agent.tools.inference_tool import InferenceTool
from seismo_agent.tools.physics_sim_tool import PhysicsSimTool
from seismo_agent.tools.comparison_tool import ComparisonTool
from seismo_agent.tools.ood_detector_tool import OODDetectorTool
from seismo_agent.tools.sweep_tool import SweepTool
from seismo_agent.tools.report_tool import ReportTool

__all__ = [
    "BaseTool",
    "ToolExecutionError",
    "EarthquakeTool",
    "StructuralTool",
    "InferenceTool",
    "PhysicsSimTool",
    "ComparisonTool",
    "OODDetectorTool",
    "SweepTool",
    "ReportTool",
    "get_all_tools",
]


def get_all_tools() -> List[BaseTool]:
    """Instantiate and return the suite of all 8 deterministic engineering tools."""
    eq_tool = EarthquakeTool()
    struct_tool = StructuralTool()
    inf_tool = InferenceTool(eq_tool=eq_tool)
    phys_tool = PhysicsSimTool(eq_tool=eq_tool)
    cmp_tool = ComparisonTool()
    ood_tool = OODDetectorTool()
    swp_tool = SweepTool(
        inference_tool=inf_tool,
        physics_tool=phys_tool,
        comparison_tool=cmp_tool,
    )
    rep_tool = ReportTool()

    return [
        eq_tool,
        struct_tool,
        inf_tool,
        phys_tool,
        cmp_tool,
        ood_tool,
        swp_tool,
        rep_tool,
    ]
