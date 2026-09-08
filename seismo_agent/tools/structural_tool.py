"""
seismo_agent/tools/structural_tool.py — Deterministic Structural Configuration Tool.

Wraps existing research classes `SDOFParams` and `MDOFParams` to instantiate,
validate, and compute mass, stiffness, natural period, circular frequency,
yield capacity, and modal eigenvalue properties for structural oscillators and frames.
"""

import math
from typing import Optional, List
import numpy as np

from seismo_agent.schemas.inputs import StructuralRequest
from seismo_agent.schemas.outputs import StructuralResult
from seismo_agent.tools.base import BaseTool, ToolExecutionError
from src.ground_truth.opensees_sdof_model import SDOFParams
from src.ground_truth.opensees_mdof_model import MDOFParams


class StructuralTool(BaseTool):
    """
    Configures structural properties and solves undamped modal eigenvalue dynamics.
    """
    name = "configure_structure"
    description = (
        "Configures single-degree-of-freedom (SDOF) or multi-degree-of-freedom (MDOF) "
        "structural dynamic systems. Computes elastic natural frequencies, periods, "
        "lateral stiffnesses, yield capacities, mode shapes, and Rayleigh damping constants. "
        "Wraps verified structural parameter dataclasses without mutating solver state."
    )
    input_schema = StructuralRequest
    output_schema = StructuralResult

    def _run(self, request: StructuralRequest) -> StructuralResult:
        if request.system_type == "sdof":
            # Instantiate SDOFParams
            try:
                sdof = SDOFParams(
                    T=request.T,
                    zeta=request.zeta,
                    material_type="bilinear" if request.material_type == "bilinear" else "elastic",
                    u_y=request.u_y if request.material_type == "bilinear" else None,
                    alpha=request.alpha,
                    mass=request.mass,
                    damping_type=request.damping_type,
                )
            except Exception as e:
                raise ToolExecutionError(self.name, f"Failed to instantiate SDOF parameters: {e}") from e

            return StructuralResult(
                system_type="sdof",
                T_n=round(float(sdof.T), 4),
                omega_n=round(float(sdof.omega_n), 4),
                k0=round(float(sdof.k0), 4),
                zeta=round(float(sdof.zeta), 4),
                material_type=sdof.material_type,
                u_y=round(float(sdof.u_y), 6) if sdof.u_y is not None else None,
                Fy=round(float(sdof.Fy), 4) if sdof.Fy is not None else None,
                alpha=round(float(sdof.alpha), 4),
                mass=round(float(sdof.mass), 4),
                damping_type=sdof.damping_type,
            )

        elif request.system_type == "mdof":
            n_stories = request.n_stories
            masses = request.story_masses or [request.mass * 1000.0] * n_stories
            heights = request.story_heights or [3.0] * n_stories
            stiffnesses = request.story_stiffnesses or [1.0e6] * n_stories
            yield_disps = request.yield_displacements or [0.005 * h for h in heights]

            try:
                mdof = MDOFParams(
                    n_stories=n_stories,
                    story_masses=masses,
                    story_heights=heights,
                    story_stiffnesses=stiffnesses,
                    material_type="bilinear" if request.material_type == "bilinear" else "elastic",
                    yield_displacements=yield_disps,
                    alpha=request.alpha,
                    zeta_1=request.zeta_1,
                    zeta_2=request.zeta_2,
                )
                omegas, periods, phi = mdof.compute_theoretical_modal_properties()
                alpha_m, beta_k = mdof.compute_rayleigh_constants(omegas[0], omegas[1] if len(omegas) > 1 else omegas[0])
            except Exception as e:
                raise ToolExecutionError(self.name, f"Failed to instantiate MDOF parameters or solve modal dynamics: {e}") from e

            return StructuralResult(
                system_type="mdof",
                T_n=round(float(periods[0]), 4),
                omega_n=round(float(omegas[0]), 4),
                k0=round(float(stiffnesses[0]), 4),
                zeta=round(float(request.zeta_1), 4),
                material_type=mdof.material_type,
                u_y=round(float(yield_disps[0]), 6),
                Fy=round(float(stiffnesses[0] * yield_disps[0]), 4),
                alpha=round(float(mdof.alpha), 4),
                mass=round(float(sum(masses)), 4),
                damping_type="rayleigh_modal",
                n_stories=n_stories,
                modal_periods=[round(float(p), 4) for p in periods],
                modal_frequencies=[round(float(w), 4) for w in omegas],
                mode_shapes=[[round(float(v), 5) for v in col] for col in phi.T],
                rayleigh_alpha=round(float(alpha_m), 6),
                rayleigh_beta=round(float(beta_k), 6),
            )
        else:
            raise ToolExecutionError(self.name, f"Unsupported system_type: '{request.system_type}'. Must be 'sdof' or 'mdof'.")
