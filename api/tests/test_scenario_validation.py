"""
api/tests/test_scenario_validation.py — Unit Tests for Scenario Validation.
"""

import pytest
from api.schemas import ScenarioInput, ScenarioValidationResult
from api.main import validate_scenario


def test_scenario_validation_valid():
    """Verify standard in-domain scenario validates successfully."""
    scenario = ScenarioInput(
        earthquake_id="RSN0001_Imperial_Valley-06.AT2",
        pga_g=0.40,
        T0=0.50,
        damping_ratio=0.05,
        yield_displacement_m=0.010,
        post_yield_ratio=0.05,
        material_type="bilinear",
        mass_kg=1.0,
    )
    result = validate_scenario(scenario)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.domain_status == "IN_DOMAIN"
    assert "omega_n_rad_s" in result.computed_properties
    assert "stiffness_k0_N_m" in result.computed_properties


def test_scenario_validation_extrapolation_warnings():
    """Verify warning generated when scenario parameters are outside primary training regime."""
    scenario = ScenarioInput(
        earthquake_id="RSN0001_Imperial_Valley-06.AT2",
        pga_g=1.50,  # > 1.2g
        T0=3.0,     # > 2.0s
        damping_ratio=0.15,
        yield_displacement_m=0.02,
    )
    result = validate_scenario(scenario)
    assert result.is_valid is True
    assert result.domain_status == "EXTRAPOLATION_WARNING"
    assert len(result.warnings) >= 2


def test_scenario_validation_pydantic_bounds():
    """Verify Pydantic validation rejects physically impossible inputs."""
    with pytest.raises(Exception):
        ScenarioInput(pga_g=-0.5)  # PGA must be positive

    with pytest.raises(Exception):
        ScenarioInput(T0=0.001)  # T0 must be >= 0.05s

    with pytest.raises(Exception):
        ScenarioInput(damping_ratio=0.80)  # Damping must be <= 0.30
