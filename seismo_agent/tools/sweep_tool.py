"""
seismo_agent/tools/sweep_tool.py — Deterministic Parametric Sensitivity & IDA Sweep Tool.

Conducts controlled multi-point sensitivity sweeps and Incremental Dynamic Analysis (IDA).
Evaluates response trajectories across parameter grids (e.g. varying PGA, T, u_y, zeta),
enforcing strict OpenSeesPy state wipes between runs and base configuration immutability.
"""

import copy
import time
import uuid
from typing import Optional, List
import numpy as np

from seismo_agent.schemas.inputs import SweepRequest, InferenceRequest, PhysicsSimulationRequest, ComparisonRequest
from seismo_agent.schemas.outputs import SweepResult, SweepPointResult
from seismo_agent.tools.base import BaseTool, ToolExecutionError
from seismo_agent.tools.inference_tool import InferenceTool
from seismo_agent.tools.physics_sim_tool import PhysicsSimTool
from seismo_agent.tools.comparison_tool import ComparisonTool


class SweepTool(BaseTool):
    """
    Conducts systematic parametric sweeps and Incremental Dynamic Analysis (IDA).
    """
    name = "run_parametric_sweep"
    description = (
        "Executes deterministic multi-point parametric sweeps across structural or seismic variables "
        "(such as scaling PGA for Incremental Dynamic Analysis, sweeping natural period T, yield displacement u_y, "
        "or damping ratio zeta). Preserves base configuration immutability and enforces clean OpenSeesPy C-state "
        "wiping between consecutive simulation points. Measures scaling behavior, ductility growth, and speedup."
    )
    input_schema = SweepRequest
    output_schema = SweepResult

    def __init__(
        self,
        inference_tool: Optional[InferenceTool] = None,
        physics_tool: Optional[PhysicsSimTool] = None,
        comparison_tool: Optional[ComparisonTool] = None,
    ):
        self.inference_tool = inference_tool or InferenceTool()
        self.physics_tool = physics_tool or PhysicsSimTool()
        self.comparison_tool = comparison_tool or ComparisonTool()

    def _run(self, request: SweepRequest) -> SweepResult:
        exp_id = f"sweep_{request.parameter_name}_{uuid.uuid4().hex[:8]}"
        param_name = request.parameter_name
        values = request.parameter_values

        if not values:
            raise ToolExecutionError(self.name, "Parameter values list cannot be empty.")

        points: List[SweepPointResult] = []
        speedups: List[float] = []

        t_start = time.perf_counter()

        for val in values:
            # Immutability: deep-copy structural request and earthquake request for each point
            struct_case = copy.deepcopy(request.base_structure)
            eq_case = copy.deepcopy(request.earthquake)

            # Apply swept parameter
            if param_name == "pga_g":
                eq_case.target_pga_g = float(val)
            elif param_name == "T":
                struct_case.T = float(val)
            elif param_name == "zeta":
                struct_case.zeta = float(val)
            elif param_name == "u_y":
                struct_case.u_y = float(val)
            elif param_name == "alpha":
                struct_case.alpha = float(val)
            elif param_name == "mass":
                struct_case.mass = float(val)
            else:
                raise ToolExecutionError(self.name, f"Unsupported sweep parameter: '{param_name}'.")

            inf_res = None
            phys_res = None
            rel_l2 = None

            # 1. Execute FNO inference if enabled
            if request.eval_fno:
                inf_req = InferenceRequest(
                    earthquake=eq_case,
                    structure=struct_case,
                    checkpoint_path=request.checkpoint_path,
                )
                inf_res = self.inference_tool.execute(inf_req)

            # 2. Execute Physics simulation if enabled
            if request.eval_physics:
                phys_req = PhysicsSimulationRequest(
                    earthquake=eq_case,
                    structure=struct_case,
                    solver="opensees",
                )
                phys_res = self.physics_tool.execute(phys_req)

            # 3. Cross-compare if both evaluated
            if inf_res and phys_res:
                cmp_req = ComparisonRequest(
                    u_pred=inf_res.u_pred,
                    u_ref=phys_res.u_gt,
                    fr_pred=inf_res.fr_pred,
                    fr_ref=phys_res.fr_gt,
                    eh_pred=inf_res.eh_pred,
                    eh_ref=phys_res.eh_gt,
                    dt=inf_res.dt,
                    u_y=struct_case.u_y,
                    fno_runtime_ms=inf_res.runtime_ms,
                    physics_runtime_ms=phys_res.runtime_ms,
                )
                cmp_res = self.comparison_tool.execute(cmp_req)
                rel_l2 = cmp_res.err_u_rel_l2_pct
                if cmp_res.speedup_factor:
                    speedups.append(cmp_res.speedup_factor)

            point = SweepPointResult(
                parameter_value=float(val),
                u_max_fno=inf_res.u_max_m if inf_res else None,
                u_max_physics=phys_res.u_max_m if phys_res else None,
                ductility_fno=inf_res.ductility_mu if inf_res else None,
                ductility_physics=phys_res.ductility_mu if phys_res else None,
                rel_l2_u_pct=rel_l2,
                fno_runtime_ms=inf_res.runtime_ms if inf_res else None,
                physics_runtime_ms=phys_res.runtime_ms if phys_res else None,
            )
            points.append(point)

        total_runtime = time.perf_counter() - t_start
        avg_speedup = round(float(np.mean(speedups)), 1) if speedups else None

        return SweepResult(
            experiment_id=exp_id,
            parameter_name=param_name,
            values_tested=[float(v) for v in values],
            num_cases=len(points),
            points=points,
            avg_speedup_factor=avg_speedup,
            total_runtime_s=round(total_runtime, 3),
        )
