"""
tests/agent/test_agent_tools.py — Comprehensive Unit Tests for SeismoAgent Deterministic Tools.

Verifies:
  1. BaseTool interface and Nemotron/OpenAI tool specification generation.
  2. EarthquakeTool parsing, baseline correction, resampling, scaling, and input validation.
  3. StructuralTool SDOF and MDOF modal dynamics.
  4. InferenceTool surrogate forward pass, device handling, and checkpoint loading.
  5. PhysicsSimTool OpenSeesPy SDOF/MDOF and independent Newmark solver.
  6. ComparisonTool metric calculation, length mismatch handling, and speedup tracking.
  7. OODDetectorTool training domain boundary verification and uncertainty reporting.
  8. SweepTool parametric sweep execution, base config immutability, and state isolation.
  9. ReportTool deterministic markdown audit compilation.
"""

import copy
import math
import pytest
import numpy as np
from pydantic import ValidationError

from seismo_agent.config import REPO_ROOT, DEFAULT_SDOF_CHECKPOINT
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
    ReportResult,
)
from seismo_agent.tools.base import BaseTool, ToolExecutionError
from seismo_agent.tools.earthquake_tool import EarthquakeTool
from seismo_agent.tools.structural_tool import StructuralTool
from seismo_agent.tools.inference_tool import InferenceTool
from seismo_agent.tools.physics_sim_tool import PhysicsSimTool
from seismo_agent.tools.comparison_tool import ComparisonTool
from seismo_agent.tools.ood_detector_tool import OODDetectorTool
from seismo_agent.tools.sweep_tool import SweepTool
from seismo_agent.tools.report_tool import ReportTool
from seismo_agent.tools import get_all_tools


# ==============================================================================
# 1. BaseTool & Interface Verification
# ==============================================================================

class TestToolInterfaces:
    """Verify that all 8 tools satisfy the BaseTool protocol and export valid specs."""

    def test_all_tools_instantiate(self):
        tools = get_all_tools()
        assert len(tools) == 8
        for t in tools:
            assert isinstance(t, BaseTool)
            assert isinstance(t.name, str) and len(t.name) > 0
            assert isinstance(t.description, str) and len(t.description) > 20
            assert t.input_schema is not None
            assert t.output_schema is not None

    def test_tool_spec_generation(self):
        tools = get_all_tools()
        for t in tools:
            spec = t.to_tool_spec()
            assert spec["type"] == "function"
            fn = spec["function"]
            assert fn["name"] == t.name
            assert fn["description"] == t.description.strip()
            assert "parameters" in fn
            params = fn["parameters"]
            assert params["type"] == "object"
            assert "properties" in params


# ==============================================================================
# 2. EarthquakeTool Tests
# ==============================================================================

class TestEarthquakeTool:
    """Test parsing, baseline correction, resampling, and error handling."""

    @pytest.fixture
    def eq_tool(self):
        return EarthquakeTool()

    def test_synthetic_ricker_record(self, eq_tool):
        req = EarthquakeRequest(
            record_id="synthetic_ricker",
            target_pga_g=0.5,
            target_dt=0.01,
            n_steps=2048,
        )
        res = eq_tool.execute(req)
        assert isinstance(res, EarthquakeResult)
        assert res.n_points == 2048
        assert res.dt == 0.01
        assert pytest.approx(res.pga_g, rel=1e-2) == 0.50
        assert res.arias_intensity_ms > 0.0
        assert res.significant_duration_d5_95_s > 0.0
        assert len(res.accel_preview) > 0

    def test_real_peer_preset_record(self, eq_tool):
        preset_file = "RSN0001_Imperial_Valley-06.AT2"
        req = EarthquakeRequest(
            record_id=preset_file,
            target_pga_g=0.40,
            target_dt=0.01,
            n_steps=2048,
        )
        res = eq_tool.execute(req)
        assert isinstance(res, EarthquakeResult)
        assert res.n_points == 2048
        assert pytest.approx(res.pga_g, rel=1e-2) == 0.40

    def test_custom_csv_parsing(self, eq_tool):
        # Create a CSV string with >10 points
        rows = ["time,accel"]
        for i in range(25):
            val = 0.15 * math.sin(i * 0.4)
            rows.append(f"{i*0.01:.2f},{val:.4f}")
        csv_text = "\n".join(rows) + "\n"
        req = EarthquakeRequest(
            custom_content=csv_text,
            custom_filename="test_ground_motion.csv",
            target_dt=0.01,
            n_steps=2048,
        )
        res = eq_tool.execute(req)
        assert isinstance(res, EarthquakeResult)
        assert res.n_points == 2048
        assert res.pga_g > 0.0

    def test_empty_content_fails(self, eq_tool):
        # Whitespace-only content should fail execution parsing
        req = EarthquakeRequest(
            custom_content="   \n   \n",
            custom_filename="empty.csv",
        )
        with pytest.raises(ToolExecutionError, match="empty"):
            eq_tool.execute(req)

    def test_nan_values_fail(self, eq_tool):
        nan_series = [0.0, 0.1, float("nan"), 0.2] + [0.0] * 20
        req = EarthquakeRequest(
            raw_accel_series=nan_series,
            target_dt=0.01,
        )
        with pytest.raises(ToolExecutionError, match="NaN"):
            eq_tool.execute(req)

    def test_missing_source_fails_validation(self):
        with pytest.raises(ValidationError):
            EarthquakeRequest()


# ==============================================================================
# 3. StructuralTool Tests
# ==============================================================================

class TestStructuralTool:
    """Test SDOF and MDOF physical parameter validation and modal dynamics."""

    @pytest.fixture
    def st_tool(self):
        return StructuralTool()

    def test_valid_sdof_bilinear(self, st_tool):
        req = StructuralRequest(
            system_type="sdof",
            T=0.5,
            zeta=0.05,
            material_type="bilinear",
            u_y=0.015,
            alpha=0.05,
            mass=1.0,
        )
        res = st_tool.execute(req)
        assert isinstance(res, StructuralResult)
        assert res.system_type == "sdof"
        assert res.T_n == 0.5
        assert pytest.approx(res.omega_n, rel=1e-3) == 12.566
        assert pytest.approx(res.k0, rel=1e-3) == 157.91
        assert res.Fy is not None
        assert pytest.approx(res.Fy, rel=1e-3) == 157.91 * 0.015

    def test_valid_mdof_eigenvalues(self, st_tool):
        req = StructuralRequest(
            system_type="mdof",
            n_stories=3,
            story_masses=[1000.0, 1000.0, 1000.0],
            story_stiffnesses=[1.0e6, 1.0e6, 1.0e6],
            zeta_1=0.05,
            zeta_2=0.05,
        )
        res = st_tool.execute(req)
        assert isinstance(res, StructuralResult)
        assert res.system_type == "mdof"
        assert res.n_stories == 3
        assert len(res.modal_periods) == 3
        assert res.modal_periods[0] > res.modal_periods[1] > res.modal_periods[2]
        assert res.rayleigh_alpha is not None
        assert res.rayleigh_beta is not None

    def test_invalid_negative_period(self):
        with pytest.raises(ValidationError):
            StructuralRequest(T=-0.5)

    def test_bilinear_without_yield_displacement(self):
        with pytest.raises(ValidationError):
            StructuralRequest(material_type="bilinear", u_y=None)


# ==============================================================================
# 4. InferenceTool Tests
# ==============================================================================

class TestInferenceTool:
    """Test SeismoFNO surrogate inference execution, shapes, and error handling."""

    @pytest.fixture
    def inf_tool(self):
        return InferenceTool()

    def test_valid_inference_execution(self, inf_tool):
        if not DEFAULT_SDOF_CHECKPOINT.exists():
            pytest.skip(f"Default checkpoint not found at {DEFAULT_SDOF_CHECKPOINT}")

        eq_req = EarthquakeRequest(record_id="synthetic_ricker", target_pga_g=0.3)
        st_req = StructuralRequest(system_type="sdof", T=0.5, zeta=0.05, material_type="bilinear", u_y=0.015)
        inf_req = InferenceRequest(earthquake=eq_req, structure=st_req)

        res = inf_tool.execute(inf_req)
        assert isinstance(res, InferenceResult)
        assert res.status == "success"
        assert res.n_points == 2048
        assert res.runtime_ms > 0.0
        assert len(res.u_pred) == 2048
        assert len(res.fr_pred) == 2048
        assert len(res.eh_pred) == 2048
        assert res.u_max_m > 0.0
        assert res.ductility_mu > 0.0

    def test_missing_checkpoint_raises_error(self, inf_tool):
        eq_req = EarthquakeRequest(record_id="synthetic_ricker")
        st_req = StructuralRequest(system_type="sdof", T=0.5)
        inf_req = InferenceRequest(
            earthquake=eq_req,
            structure=st_req,
            checkpoint_path="nonexistent_checkpoint_weights.pt",
        )
        with pytest.raises(ToolExecutionError, match="checkpoint not found"):
            inf_tool.execute(inf_req)


# ==============================================================================
# 5. PhysicsSimTool Tests
# ==============================================================================

class TestPhysicsSimTool:
    """Test OpenSeesPy and independent Newmark nonlinear simulations."""

    @pytest.fixture
    def phys_tool(self):
        return PhysicsSimTool()

    def test_opensees_sdof_simulation(self, phys_tool):
        eq_req = EarthquakeRequest(record_id="synthetic_ricker", target_pga_g=0.3)
        st_req = StructuralRequest(system_type="sdof", T=0.5, zeta=0.05, material_type="bilinear", u_y=0.015)
        phys_req = PhysicsSimulationRequest(earthquake=eq_req, structure=st_req, solver="opensees")

        res = phys_tool.execute(phys_req)
        assert isinstance(res, PhysicsResult)
        assert res.status == "success"
        assert res.n_points == 2048
        assert len(res.u_gt) == 2048
        assert len(res.fr_gt) == 2048
        assert len(res.eh_gt) == 2048
        assert res.u_max_m > 0.0
        assert res.runtime_ms > 0.0

    def test_opensees_mdof_simulation(self, phys_tool):
        eq_req = EarthquakeRequest(record_id="synthetic_ricker", target_pga_g=0.2)
        st_req = StructuralRequest(system_type="mdof", n_stories=3)
        phys_req = PhysicsSimulationRequest(earthquake=eq_req, structure=st_req, solver="opensees")

        res = phys_tool.execute(phys_req)
        assert isinstance(res, PhysicsResult)
        assert res.status == "success"
        assert res.n_points == 2048
        assert res.u_max_m > 0.0

    def test_independent_newmark_solver(self, phys_tool):
        eq_req = EarthquakeRequest(record_id="synthetic_ricker", target_pga_g=0.25)
        st_req = StructuralRequest(system_type="sdof", T=0.4, zeta=0.05, material_type="bilinear", u_y=0.01)
        phys_req = PhysicsSimulationRequest(earthquake=eq_req, structure=st_req, solver="independent_newmark")

        res = phys_tool.execute(phys_req)
        assert isinstance(res, PhysicsResult)
        assert "Independent" in res.solver_used
        assert res.n_points == 2048
        assert res.u_max_m > 0.0


# ==============================================================================
# 6. ComparisonTool Tests
# ==============================================================================

class TestComparisonTool:
    """Test metric computation, speedup evaluation, and edge cases."""

    @pytest.fixture
    def cmp_tool(self):
        return ComparisonTool()

    def test_perfect_agreement(self, cmp_tool):
        signal = [0.0, 0.01, 0.05, 0.02, -0.01, 0.0]
        req = ComparisonRequest(
            u_pred=signal,
            u_ref=signal,
            fr_pred=signal,
            fr_ref=signal,
            eh_pred=signal,
            eh_ref=signal,
            dt=0.01,
            fno_runtime_ms=1.0,
            physics_runtime_ms=100.0,
        )
        res = cmp_tool.execute(req)
        assert isinstance(res, ComparisonResult)
        assert res.status == "success"
        assert res.err_u_rel_l2_pct == 0.0
        assert res.err_umax_rel_pct == 0.0
        assert res.residual_drift_error_m == 0.0
        assert res.speedup_factor == 100.0

    def test_mismatched_lengths_handled_cleanly(self, cmp_tool):
        pred = [0.0, 0.01, 0.02, 0.03, 0.04]
        ref = [0.0, 0.01, 0.02]  # shorter
        req = ComparisonRequest(u_pred=pred, u_ref=ref, dt=0.01)
        res = cmp_tool.execute(req)
        assert isinstance(res, ComparisonResult)
        assert res.err_u_rel_l2_pct >= 0.0

    def test_empty_signal_raises_error(self, cmp_tool):
        req = ComparisonRequest(u_pred=[], u_ref=[0.01])
        with pytest.raises(ToolExecutionError, match="cannot be empty"):
            cmp_tool.execute(req)


# ==============================================================================
# 7. OODDetectorTool Tests
# ==============================================================================

class TestOODDetectorTool:
    """Test domain boundary checks and uncertainty status reporting."""

    @pytest.fixture
    def ood_tool(self):
        return OODDetectorTool()

    def test_in_domain_request(self, ood_tool):
        st = StructuralRequest(system_type="sdof", T=0.5, zeta=0.05, u_y=0.015, alpha=0.05)
        req = OODRequest(structure=st, pga_g=0.4, dt=0.01)
        res = ood_tool.execute(req)
        assert isinstance(res, OODResult)
        assert not res.is_ood
        assert res.within_structural_domain
        assert res.within_ground_motion_domain
        assert res.uncertainty_status == "NOT_QUANTIFIED"
        assert len(res.domain_violations) == 0

    def test_out_of_domain_request(self, ood_tool):
        # T = 10.0s (far outside 0.1-3.0s), PGA = 4.0g (far outside 0.05-1.2g)
        st = StructuralRequest(system_type="sdof", T=10.0, zeta=0.05, u_y=0.015, alpha=0.05)
        req = OODRequest(structure=st, pga_g=4.0, dt=0.01)
        res = ood_tool.execute(req)
        assert isinstance(res, OODResult)
        assert res.is_ood
        assert not res.within_structural_domain
        assert not res.within_ground_motion_domain
        assert len(res.domain_violations) >= 2
        assert "OUT-OF-DISTRIBUTION" in res.recommendation


# ==============================================================================
# 8. SweepTool Tests
# ==============================================================================

class TestSweepTool:
    """Test parametric sweeps, IDA execution, and base config immutability."""

    @pytest.fixture
    def swp_tool(self):
        return SweepTool()

    def test_pga_sweep_immutability(self, swp_tool):
        if not DEFAULT_SDOF_CHECKPOINT.exists():
            pytest.skip("Default checkpoint required for sweep")

        base_st = StructuralRequest(system_type="sdof", T=0.5, zeta=0.05, material_type="bilinear", u_y=0.015)
        eq = EarthquakeRequest(record_id="synthetic_ricker")
        original_T = base_st.T

        req = SweepRequest(
            parameter_name="pga_g",
            parameter_values=[0.2, 0.4],
            base_structure=base_st,
            earthquake=eq,
            eval_fno=True,
            eval_physics=True,
        )
        res = swp_tool.execute(req)
        assert isinstance(res, SweepResult)
        assert res.num_cases == 2
        assert len(res.points) == 2
        assert res.points[0].u_max_fno is not None
        assert res.points[1].u_max_fno is not None
        # Peak displacement should grow with PGA
        assert res.points[1].u_max_fno > res.points[0].u_max_fno

        # Verify base_structure was NOT mutated
        assert base_st.T == original_T


# ==============================================================================
# 9. ReportTool Tests
# ==============================================================================

class TestReportTool:
    """Test deterministic report compilation and limitations disclosure."""

    @pytest.fixture
    def rep_tool(self):
        return ReportTool()

    def test_report_compilation(self, rep_tool):
        req = ReportRequest(
            experiment_id="test_exp_001",
            title="Seismic Verification Report",
            earthquake_result={"record_id": "RSN0001", "pga_g": 0.40, "n_points": 2048, "dt": 0.01},
            structural_result={"system_type": "sdof", "T_n": 0.5, "zeta": 0.05, "k0": 157.91},
            inference_result={"u_max_m": 0.035, "runtime_ms": 0.75, "ductility_mu": 2.33},
            physics_result={"u_max_m": 0.034, "runtime_ms": 110.0, "ductility_mu": 2.27},
            comparison_result={"err_u_rel_l2_pct": 3.5, "speedup_factor": 146.7},
            ood_result={"is_ood": False, "uncertainty_status": "NOT_QUANTIFIED"},
            report_format="markdown",
        )
        res = rep_tool.execute(req)
        assert isinstance(res, ReportResult)
        assert res.experiment_id == "test_exp_001"
        assert "# Seismic Verification Report" in res.content
        assert "146.7x Speedup" in res.content
        assert "3.5%" in res.content
        assert len(res.limitations) >= 3
