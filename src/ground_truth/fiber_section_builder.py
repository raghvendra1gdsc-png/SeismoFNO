"""
fiber_section_builder.py — OpenSeesPy Fiber Section Discretization Engine for MDOF Systems.

Constructs nonlinear fiber sections for reinforced concrete (RC) and structural steel members:
  - Uniaxial constitutive materials (Concrete01/02, Steel01/02, Elastic)
  - Rectangular RC sections with confined core, unconfined cover, and reinforcing steel layers
  - Wide-flange (W-shape) steel fiber sections discretized into web and flange fiber patches
  - Elastic section definitions for closed-form modal and dynamic benchmarks
"""

from typing import Dict, Any, Optional, Tuple
import math
import openseespy.opensees as ops


def create_elastic_material(mat_tag: int, E: float) -> int:
    """Create uniaxial Elastic material in OpenSees."""
    ops.uniaxialMaterial("Elastic", mat_tag, float(E))
    return mat_tag


def create_steel01_material(
    mat_tag: int,
    Fy: float,
    E0: float,
    b: float = 0.02,
    a1: float = 0.0,
    a2: float = 1.0,
    a3: float = 0.0,
    a4: float = 1.0,
) -> int:
    """
    Create uniaxial Steel01 (bilinear kinematic hardening) material.

    Args:
        mat_tag: Unique integer tag
        Fy: Yield strength
        E0: Initial elastic modulus
        b: Strain hardening ratio (E_post / E0)
    """
    ops.uniaxialMaterial("Steel01", mat_tag, float(Fy), float(E0), float(b), float(a1), float(a2), float(a3), float(a4))
    return mat_tag


def create_steel02_material(
    mat_tag: int,
    Fy: float,
    E0: float,
    b: float = 0.02,
    R0: float = 18.5,
    cR1: float = 0.925,
    cR2: float = 0.15,
) -> int:
    """Create uniaxial Steel02 (Menegotto-Pinto) material with smooth hysteretic transitions."""
    ops.uniaxialMaterial("Steel02", mat_tag, float(Fy), float(E0), float(b), float(R0), float(cR1), float(cR2))
    return mat_tag


def create_concrete01_material(
    mat_tag: int,
    fpc: float,
    epsc0: float = -0.002,
    fpcu: float = 0.0,
    epsu: float = -0.006,
) -> int:
    """
    Create uniaxial Concrete01 (zero tensile strength, Kent-Scott-Park backbone).

    Args:
        mat_tag: Unique tag
        fpc: Peak compressive strength (negative value in OpenSees convention)
        epsc0: Strain at peak compressive strength
        fpcu: Crushing strength
        epsu: Strain at crushing strength
    """
    fpc_val = -abs(float(fpc))
    fpcu_val = -abs(float(fpcu))
    epsc0_val = -abs(float(epsc0))
    epsu_val = -abs(float(epsu))
    ops.uniaxialMaterial("Concrete01", mat_tag, fpc_val, epsc0_val, fpcu_val, epsu_val)
    return mat_tag


def create_elastic_section(
    sec_tag: int,
    E: float,
    A: float,
    Iz: float,
) -> int:
    """Create an Elastic Section for 2D frame elements."""
    ops.section("Elastic", sec_tag, float(E), float(A), float(Iz))
    return sec_tag


def create_fiber_section_rc_rect(
    sec_tag: int,
    core_mat_tag: int,
    cover_mat_tag: int,
    steel_mat_tag: int,
    d: float,
    b: float,
    cover: float,
    num_bars_top: int,
    num_bars_bot: int,
    bar_area: float,
    n_fibers_d: int = 16,
    n_fibers_b: int = 8,
) -> int:
    """
    Construct a rectangular reinforced concrete (RC) fiber section in 2D (bending about z).

    Args:
        sec_tag: Unique section tag
        core_mat_tag: Confined concrete material tag
        cover_mat_tag: Unconfined concrete material tag
        steel_mat_tag: Reinforcing steel material tag
        d: Total section depth (y direction)
        b: Section width (z direction)
        cover: Clear cover to rebar centroid
        num_bars_top: Number of top rebar bars
        num_bars_bot: Number of bottom rebar bars
        bar_area: Cross-sectional area of individual rebar
        n_fibers_d: Discretization fibers along depth
        n_fibers_b: Discretization fibers along width
    """
    y1 = -d / 2.0
    y2 = d / 2.0
    z1 = -b / 2.0
    z2 = b / 2.0

    core_y1 = y1 + cover
    core_y2 = y2 - cover
    core_z1 = z1 + cover
    core_z2 = z2 - cover

    ops.section("Fiber", sec_tag)

    # Core confined patch
    ops.patch("quad", core_mat_tag, n_fibers_b, n_fibers_d, core_y1, core_z1, core_y1, core_z2, core_y2, core_z2, core_y2, core_z1)

    # Cover unconfined patches: bottom, top, left, right
    ops.patch("quad", cover_mat_tag, n_fibers_b, 2, y1, z1, y1, z2, core_y1, z2, core_y1, z1)
    ops.patch("quad", cover_mat_tag, n_fibers_b, 2, core_y2, z1, core_y2, z2, y2, z2, y2, z1)
    ops.patch("quad", cover_mat_tag, 2, n_fibers_d, core_y1, z1, core_y1, core_z1, core_y2, core_z1, core_y2, z1)
    ops.patch("quad", cover_mat_tag, 2, n_fibers_d, core_y1, core_z2, core_y1, z2, core_y2, z2, core_y2, core_z2)

    # Reinforcing steel straight layers
    if num_bars_top > 0:
        ops.layer("straight", steel_mat_tag, num_bars_top, bar_area, core_y2, core_z1, core_y2, core_z2)
    if num_bars_bot > 0:
        ops.layer("straight", steel_mat_tag, num_bars_bot, bar_area, core_y1, core_z1, core_y1, core_z2)

    return sec_tag


def create_fiber_section_w_shape(
    sec_tag: int,
    steel_mat_tag: int,
    d: float,
    bf: float,
    tf: float,
    tw: float,
    n_fibers_flange: int = 4,
    n_fibers_web: int = 16,
) -> int:
    """
    Construct a structural steel wide-flange (W-shape) fiber section in 2D.

    Args:
        sec_tag: Section tag
        steel_mat_tag: Steel material tag
        d: Total depth
        bf: Flange width
        tf: Flange thickness
        tw: Web thickness
    """
    y1 = -d / 2.0
    y2 = d / 2.0
    z1 = -bf / 2.0
    z2 = bf / 2.0

    ops.section("Fiber", sec_tag)

    # Bottom flange
    ops.patch("quad", steel_mat_tag, 8, n_fibers_flange, y1, z1, y1, z2, y1 + tf, z2, y1 + tf, z1)

    # Top flange
    ops.patch("quad", steel_mat_tag, 8, n_fibers_flange, y2 - tf, z1, y2 - tf, z2, y2, z2, y2, z1)

    # Web
    ops.patch("quad", steel_mat_tag, 4, n_fibers_web, y1 + tf, -tw / 2.0, y1 + tf, tw / 2.0, y2 - tf, tw / 2.0, y2 - tf, -tw / 2.0)

    return sec_tag
