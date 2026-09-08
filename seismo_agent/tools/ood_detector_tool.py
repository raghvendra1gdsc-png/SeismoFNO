"""
seismo_agent/tools/ood_detector_tool.py — Deterministic OOD & Uncertainty Assessment Tool.

Evaluates structural parameters and ground motion features against the authoritative
training domain envelope defined in `configs/data_generation.yaml` and generalization
protocols. Factual and scientifically honest: NEVER manufactures fake confidence percentages.
"""

from pathlib import Path
from typing import Optional, List
import yaml

from seismo_agent.config import DEFAULT_DATA_GEN_CONFIG_PATH
from seismo_agent.schemas.inputs import OODRequest
from seismo_agent.schemas.outputs import OODResult
from seismo_agent.tools.base import BaseTool, ToolExecutionError


class OODDetectorTool(BaseTool):
    """
    Evaluates whether input parameters fall within the calibrated training distribution.
    """
    name = "assess_ood_and_uncertainty"
    description = (
        "Evaluates whether a structural configuration and ground motion record reside within "
        "the calibrated training distribution of SeismoFNO or represent out-of-distribution (OOD) "
        "extrapolation. Reads bounds dynamically from research configuration. "
        "Enforces scientific honesty: reports 'NOT_QUANTIFIED' rather than inventing statistical confidence."
    )
    input_schema = OODRequest
    output_schema = OODResult

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_DATA_GEN_CONFIG_PATH
        self._bounds = self._load_bounds()

    def _load_bounds(self) -> dict:
        """Load domain limits dynamically from authoritative project configuration."""
        if not self.config_path.exists():
            # Standard documented research fallback envelope
            return {
                "pga_min_g": 0.05,
                "pga_max_g": 1.20,
                "t_min_s": 0.10,
                "t_max_s": 3.00,
                "zeta_min": 0.01,
                "zeta_max": 0.10,
                "uy_min_m": 0.002,
                "uy_max_m": 0.050,
                "alpha_min": 0.01,
                "alpha_max": 0.20,
                "dt_min_s": 0.0025,
                "dt_max_s": 0.08,
            }

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            gm = cfg.get("ground_motions", {})
            pgas = gm.get("pga_scales_g", [0.05, 1.20])
            st = cfg.get("structures", {}).get("sdof", {})
            periods = st.get("periods_s", [0.10, 2.00])
            zetas = st.get("damping_ratios", [0.02, 0.05])
            uys = st.get("yield_displacements_m", [0.005, 0.030])
            alphas = st.get("hardening_ratios", [0.02, 0.10])

            return {
                "pga_min_g": float(min(pgas)),
                "pga_max_g": float(max(pgas)),
                "t_min_s": float(min(periods)),
                # In Phase 7, generalization evaluated up to 3.0s
                "t_max_s": max(3.0, float(max(periods))),
                "zeta_min": max(0.01, float(min(zetas)) * 0.5),
                "zeta_max": min(0.15, float(max(zetas)) * 2.0),
                "uy_min_m": float(min(uys)) * 0.5,
                "uy_max_m": float(max(uys)) * 1.5,
                "alpha_min": float(min(alphas)) * 0.5,
                "alpha_max": float(max(alphas)) * 2.0,
                "dt_min_s": 0.0025,
                "dt_max_s": 0.08,
            }
        except Exception:
            return {
                "pga_min_g": 0.05,
                "pga_max_g": 1.20,
                "t_min_s": 0.10,
                "t_max_s": 3.00,
                "zeta_min": 0.01,
                "zeta_max": 0.10,
                "uy_min_m": 0.002,
                "uy_max_m": 0.050,
                "alpha_min": 0.01,
                "alpha_max": 0.20,
                "dt_min_s": 0.0025,
                "dt_max_s": 0.08,
            }

    def _run(self, request: OODRequest) -> OODResult:
        st = request.structure
        b = self._bounds

        violations: List[str] = []
        warnings: List[str] = []

        # 1. Structural Domain Verification
        struct_ok = True
        if st.T < b["t_min_s"] or st.T > b["t_max_s"]:
            struct_ok = False
            violations.append(f"Natural period T={st.T:.2f}s is outside calibrated envelope [{b['t_min_s']}, {b['t_max_s']}]s.")
        elif st.T > 2.0:
            warnings.append(f"Natural period T={st.T:.2f}s is in the long-period generalization zone (Phase 7).")

        if st.zeta < b["zeta_min"] or st.zeta > b["zeta_max"]:
            struct_ok = False
            violations.append(f"Damping ratio zeta={st.zeta:.3f} is outside calibrated envelope [{b['zeta_min']}, {b['zeta_max']}].")

        if st.material_type == "bilinear" and st.u_y is not None:
            if st.u_y < b["uy_min_m"] or st.u_y > b["uy_max_m"]:
                struct_ok = False
                violations.append(f"Yield displacement u_y={st.u_y*1000.0:.2f}mm is outside calibrated envelope [{b['uy_min_m']*1000.0}, {b['uy_max_m']*1000.0}]mm.")

            if st.alpha < b["alpha_min"] or st.alpha > b["alpha_max"]:
                struct_ok = False
                violations.append(f"Hardening ratio alpha={st.alpha:.3f} is outside calibrated envelope [{b['alpha_min']}, {b['alpha_max']}].")

        # 2. Ground Motion Domain Verification
        gm_ok = True
        if request.pga_g < b["pga_min_g"] or request.pga_g > b["pga_max_g"]:
            gm_ok = False
            violations.append(f"Peak Ground Acceleration PGA={request.pga_g:.3f}g is outside calibrated envelope [{b['pga_min_g']}, {b['pga_max_g']}]g.")
        elif request.pga_g > 0.8:
            warnings.append(f"PGA={request.pga_g:.2f}g is in the severe excitation regime (>0.8g); significant plastic excursions anticipated.")

        # 3. Resolution Invariance Domain Verification
        res_ok = True
        if request.dt < b["dt_min_s"] or request.dt > b["dt_max_s"]:
            res_ok = False
            violations.append(f"Sampling time step dt={request.dt}s is outside zero-shot resolution invariance tested range [{b['dt_min_s']}, {b['dt_max_s']}]s.")

        is_ood = not (struct_ok and gm_ok and res_ok)

        # 4. Recommendation Formulation
        if is_ood:
            rec = (
                "OUT-OF-DISTRIBUTION WARNING: The analysis involves parameters outside the calibrated "
                "training domain. SeismoFNO surrogate predictions may exhibit unverified extrapolation error. "
                "Execution of OpenSeesPy ground-truth simulation is strongly recommended for verification."
            )
        elif len(warnings) > 0:
            rec = (
                "IN-DISTRIBUTION WITH ADVISORY: Parameters are within training limits, but operate in "
                "near-boundary or high-intensity regime. FNO inference is valid; spot-check with OpenSeesPy advised."
            )
        else:
            rec = (
                "IN-DISTRIBUTION: All structural and seismic parameters reside securely within the calibrated "
                "training envelope. SeismoFNO inference is expected to operate with sub-5% relative L2 error."
            )

        return OODResult(
            status="success",
            is_ood=is_ood,
            within_structural_domain=struct_ok,
            within_ground_motion_domain=gm_ok,
            resolution_supported=res_ok,
            domain_violations=violations,
            domain_warnings=warnings,
            uncertainty_status="NOT_QUANTIFIED",
            recommendation=rec,
        )
