"""
seismo_agent/tools/physics_sim_tool.py — Deterministic Physics Simulation Tool.

Wraps the verified high-fidelity OpenSeesPy numerical solver (SDOF / MDOF NLTHA)
and independent zero-dependency Newmark-beta / analytical reference solvers.
Ensures process-local serialization of OpenSeesPy calls and C-state cleaning via `ops.wipe()`.
"""

import time
import threading
from typing import Optional, Dict, Any
import numpy as np

# Process-local lock ensuring sequential serialization of OpenSeesPy C-state operations within the current process
_OPENSEES_LOCK = threading.Lock()

from seismo_agent.schemas.inputs import PhysicsSimulationRequest
from seismo_agent.schemas.outputs import PhysicsResult
from seismo_agent.tools.base import BaseTool, ToolExecutionError
from seismo_agent.tools.earthquake_tool import EarthquakeTool
from src.ground_truth.opensees_sdof_model import simulate_sdof, SDOFParams, SDOFResponse
from src.ground_truth.opensees_mdof_model import simulate_mdof, MDOFParams, MDOFResponse
from src.ground_truth.independent_solvers import newmark_nonlinear_sdof
import openseespy.opensees as ops


class PhysicsSimTool(BaseTool):
    """
    Executes rigorous numerical time-history analysis (NLTHA) using OpenSeesPy or independent solvers.
    """
    name = "run_physics_simulation"
    description = (
        "Executes high-fidelity, deterministic physics-based nonlinear dynamic time-history analysis (NLTHA). "
        "Solves the equations of motion for SDOF or MDOF structural systems subjected to ground accelerations. "
        "Supports OpenSeesPy (Newmark-beta average acceleration method with Newton-Raphson iterations) "
        "and independent pure-Python Newmark solvers. Enforces process-local serialization of OpenSeesPy calls."
    )
    input_schema = PhysicsSimulationRequest
    output_schema = PhysicsResult

    def __init__(self, eq_tool: Optional[EarthquakeTool] = None):
        self.eq_tool = eq_tool or EarthquakeTool()

    def _run(self, request: PhysicsSimulationRequest) -> PhysicsResult:
        # 1. Obtain processed ground acceleration series
        ag, dt, rec_name = self.eq_tool.get_processed_acceleration_array(request.earthquake)
        n_steps = len(ag)
        st = request.structure

        t0 = time.perf_counter()

        if request.solver == "opensees":
            if st.system_type == "sdof":
                sdof_params = SDOFParams(
                    T=st.T,
                    zeta=st.zeta,
                    material_type="bilinear" if st.material_type == "bilinear" else "elastic",
                    u_y=st.u_y if st.material_type == "bilinear" else None,
                    alpha=st.alpha,
                    mass=st.mass,
                    damping_type=st.damping_type,
                )

                with _OPENSEES_LOCK:
                    try:
                        # Explicit OpenSees C-state wipe before and after to avoid memory contamination
                        ops.wipe()
                        res: SDOFResponse = simulate_sdof(
                            params=sdof_params,
                            ag=ag,
                            dt=dt,
                            u0=request.u0,
                            v0=request.v0,
                        )
                    except Exception as e:
                        ops.wipe()
                        raise ToolExecutionError(
                            self.name,
                            f"OpenSeesPy SDOF simulation failed: {e}"
                        ) from e
                    finally:
                        ops.wipe()

                runtime_ms = (time.perf_counter() - t0) * 1000.0

                u_max = float(np.max(np.abs(res.u)))
                fr_max = float(np.max(np.abs(res.f_r)))
                eh_total = float(res.e_h[-1]) if len(res.e_h) > 0 else 0.0
                ductility = float(u_max / max(1e-6, st.u_y)) if (st.material_type == "bilinear" and st.u_y) else 1.0

                return PhysicsResult(
                    status="success",
                    solver_used="OpenSeesPy SDOF NLTHA (Newmark-beta average acceleration)",
                    runtime_ms=round(runtime_ms, 3),
                    n_points=n_steps,
                    dt=round(dt, 5),
                    u_max_m=round(u_max, 6),
                    fr_max_n=round(fr_max, 4),
                    eh_total_j=round(eh_total, 4),
                    ductility_mu=round(ductility, 3),
                    time=res.time.tolist(),
                    u_gt=res.u.tolist(),
                    v_gt=res.v.tolist(),
                    a_gt=res.a.tolist(),
                    fr_gt=res.f_r.tolist(),
                    eh_gt=res.e_h.tolist(),
                    energy_balance_residual=None,
                )

            elif st.system_type == "mdof":
                n_stories = st.n_stories
                masses = st.story_masses or [st.mass * 1000.0] * n_stories
                heights = st.story_heights or [3.0] * n_stories
                stiffnesses = st.story_stiffnesses or [1.0e6] * n_stories
                yield_disps = st.yield_displacements or [0.005 * h for h in heights]

                mdof_params = MDOFParams(
                    n_stories=n_stories,
                    story_masses=masses,
                    story_heights=heights,
                    story_stiffnesses=stiffnesses,
                    material_type="bilinear" if st.material_type == "bilinear" else "elastic",
                    yield_displacements=yield_disps,
                    alpha=st.alpha,
                    zeta_1=st.zeta_1,
                    zeta_2=st.zeta_2,
                )

                with _OPENSEES_LOCK:
                    try:
                        ops.wipe()
                        res_mdof: MDOFResponse = simulate_mdof(
                            params=mdof_params,
                            ag=ag,
                            dt=dt,
                        )
                    except Exception as e:
                        ops.wipe()
                        raise ToolExecutionError(
                            self.name,
                            f"OpenSeesPy MDOF simulation failed: {e}"
                        ) from e
                    finally:
                        ops.wipe()

                runtime_ms = (time.perf_counter() - t0) * 1000.0

                # Top-story displacement metrics
                top_u = res_mdof.u[-1]
                top_fr = res_mdof.f_r[-1]
                total_eh = float(np.sum(res_mdof.e_h[:, -1]))
                u_max = float(np.max(np.abs(top_u)))
                fr_max = float(np.max(np.abs(top_fr)))
                ductility = float(u_max / max(1e-6, yield_disps[-1]))

                return PhysicsResult(
                    status="success",
                    solver_used=f"OpenSeesPy MDOF NLTHA ({n_stories}-story shear frame)",
                    runtime_ms=round(runtime_ms, 3),
                    n_points=n_steps,
                    dt=round(dt, 5),
                    u_max_m=round(u_max, 6),
                    fr_max_n=round(fr_max, 4),
                    eh_total_j=round(total_eh, 4),
                    ductility_mu=round(ductility, 3),
                    time=res_mdof.time.tolist(),
                    u_gt=top_u.tolist(),
                    v_gt=res_mdof.v[-1].tolist(),
                    a_gt=res_mdof.a[-1].tolist(),
                    fr_gt=top_fr.tolist(),
                    eh_gt=res_mdof.e_h[-1].tolist(),
                    energy_balance_residual=None,
                )
            else:
                raise ToolExecutionError(self.name, f"Unknown system type: {st.system_type}")

        elif request.solver == "independent_newmark":
            # Pure-Python Newmark-beta solver (Zero OpenSeesPy dependency)
            k0 = st.mass * ((2.0 * np.pi / max(1e-4, st.T)) ** 2)
            u_y_val = st.u_y if st.material_type == "bilinear" and st.u_y else 0.015

            try:
                res_dict = newmark_nonlinear_sdof(
                    mass=st.mass,
                    k0=k0,
                    zeta=st.zeta,
                    ag=ag,
                    dt=dt,
                    material_type="bilinear" if st.material_type == "bilinear" else "linear",
                    u_y=u_y_val,
                    alpha=st.alpha,
                )
            except Exception as e:
                raise ToolExecutionError(self.name, f"Independent Newmark solver failed: {e}") from e

            runtime_ms = (time.perf_counter() - t0) * 1000.0

            u_arr = res_dict["u"]
            v_arr = res_dict["v"]
            a_arr = res_dict["a"]
            fr_arr = res_dict["f_r"]
            eh_arr = res_dict["e_h"]
            time_arr = res_dict["time"]

            u_max = float(np.max(np.abs(u_arr)))
            fr_max = float(np.max(np.abs(fr_arr)))
            eh_total = float(eh_arr[-1]) if len(eh_arr) > 0 else 0.0
            ductility = float(u_max / max(1e-6, u_y_val)) if st.material_type == "bilinear" else 1.0

            return PhysicsResult(
                status="success",
                solver_used="Independent Pure-Python Newmark-Beta Integrator",
                runtime_ms=round(runtime_ms, 3),
                n_points=n_steps,
                dt=round(dt, 5),
                u_max_m=round(u_max, 6),
                fr_max_n=round(fr_max, 4),
                eh_total_j=round(eh_total, 4),
                ductility_mu=round(ductility, 3),
                time=time_arr.tolist(),
                u_gt=u_arr.tolist(),
                v_gt=v_arr.tolist(),
                a_gt=a_arr.tolist(),
                fr_gt=fr_arr.tolist(),
                eh_gt=eh_arr.tolist(),
                energy_balance_residual=None,
            )
        else:
            raise ToolExecutionError(self.name, f"Unsupported solver: '{request.solver}'. Choose 'opensees' or 'independent_newmark'.")
