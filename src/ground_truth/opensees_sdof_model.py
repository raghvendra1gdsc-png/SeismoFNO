"""
opensees_sdof_model.py — OpenSeesPy SDOF Oscillator Simulator.

Supports:
  (a) Linear-elastic restoring force
  (b) Bilinear-hysteretic (kinematic hardening) restoring force

Parameterized by:
  - Natural period T [s] (or angular frequency omega = 2*pi / T)
  - Damping ratio zeta (dimensionless, e.g., 0.02 or 0.05)
  - Yield displacement u_y [m] (for bilinear-hysteretic)
  - Post-yield stiffness ratio alpha = k_post / k_elastic (dimensionless)
  - Mass m [kg] (default = 1.0)

Returns time-history response:
  - u(t): Relative displacement [m]
  - v(t): Relative velocity [m/s]
  - a(t): Relative acceleration [m/s^2]
  - total_accel(t): Absolute acceleration [m/s^2] = a(t) + a_g(t)
  - F_R(t): Restoring force [N]
  - E_h(t): Cumulative hysteretic energy / restoring force work [J] = integral(F_R du)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
import math
import numpy as np
import openseespy.opensees as ops


@dataclass
class SDOFParams:
    """Parameters defining the single-degree-of-freedom oscillator."""
    T: float                                            # Natural period [s]
    zeta: float = 0.05                                 # Viscous damping ratio
    material_type: Literal["elastic", "bilinear"] = "elastic"
    u_y: Optional[float] = None                        # Yield displacement [m] (required for bilinear)
    alpha: float = 0.02                                # Post-yield stiffness ratio k_post / k_0
    mass: float = 1.0                                  # Mass [kg]
    damping_type: Literal["mass", "initial_stiffness"] = "mass"

    @property
    def omega_n(self) -> float:
        """Natural circular frequency [rad/s]."""
        return 2.0 * math.pi / self.T

    @property
    def k0(self) -> float:
        """Initial elastic stiffness [N/m]."""
        return self.mass * (self.omega_n ** 2)

    @property
    def Fy(self) -> Optional[float]:
        """Yield strength [N]."""
        if self.u_y is not None:
            return self.k0 * self.u_y
        return None

    @property
    def c(self) -> float:
        """Viscous damping coefficient [N*s/m]."""
        return 2.0 * self.zeta * self.mass * self.omega_n


@dataclass
class SDOFResponse:
    """Time-series response output from SDOF dynamic simulation."""
    time: np.ndarray          # Time vector [s] (N,)
    u: np.ndarray             # Relative displacement [m] (N,)
    v: np.ndarray             # Relative velocity [m/s] (N,)
    a: np.ndarray             # Relative acceleration [m/s^2] (N,)
    total_accel: np.ndarray   # Total acceleration [m/s^2] (N,)
    f_r: np.ndarray           # Restoring force [N] (N,)
    e_h: np.ndarray           # Cumulative hysteretic energy integral(F_R du) [J] (N,)
    e_d: np.ndarray           # Cumulative viscous damping energy integral(F_D du) [J] (N,)
    e_k: np.ndarray           # Kinetic energy 0.5 * m * v^2 [J] (N,)
    e_s: np.ndarray           # Elastic strain energy [J] (N,)
    ag: np.ndarray            # Ground acceleration input [m/s^2] (N,)
    dt: float                 # Time step [s]

    def to_dict(self) -> Dict[str, Any]:
        """Convert response dataclass to dictionary."""
        return {
            "time": self.time,
            "u": self.u,
            "v": self.v,
            "a": self.a,
            "total_accel": self.total_accel,
            "f_r": self.f_r,
            "e_h": self.e_h,
            "e_d": self.e_d,
            "e_k": self.e_k,
            "e_s": self.e_s,
            "ag": self.ag,
            "dt": self.dt,
        }


def compute_cumulative_energy(f_r: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Compute cumulative hysteretic energy / restoring force work:
        E_h(t) = integral_0^t F_R(tau) du(tau)
    using the trapezoidal rule over discrete time steps.
    """
    n = len(u)
    e_h = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return e_h
    du = np.diff(u)
    f_avg = 0.5 * (f_r[:-1] + f_r[1:])
    de_h = f_avg * du
    e_h[1:] = np.cumsum(de_h)
    return e_h


def simulate_sdof(
    params: Union[SDOFParams, Dict[str, Any]],
    ag: Optional[np.ndarray] = None,
    dt: float = 0.01,
    n_steps: Optional[int] = None,
    u0: float = 0.0,
    v0: float = 0.0,
    gamma: float = 0.5,
    beta: float = 0.25,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> SDOFResponse:
    """
    Run an OpenSeesPy dynamic analysis of an SDOF oscillator.

    Parameters
    ----------
    params : SDOFParams or dict
        Oscillator parameters (T, zeta, material_type, u_y, alpha, mass, damping_type).
    ag : np.ndarray or None
        Ground acceleration time series a_g(t) [m/s^2]. If None, a zero-acceleration
        array of length `n_steps` is used (useful for free vibration).
    dt : float
        Time step [s].
    n_steps : int, optional
        Number of time steps. If ag is provided, defaults to len(ag).
    u0 : float
        Initial displacement at t=0 [m].
    v0 : float
        Initial velocity at t=0 [m/s].
    gamma : float
        Newmark integrator parameter gamma (default 0.5 for average acceleration).
    beta : float
        Newmark integrator parameter beta (default 0.25 for average acceleration).
    tol : float
        Convergence tolerance for unbalance norm.
    max_iter : int
        Maximum iterations per step.

    Returns
    -------
    SDOFResponse dataclass containing u, v, a, total_accel, f_r, e_h, etc.
    """
    if isinstance(params, dict):
        params = SDOFParams(**params)

    if ag is not None:
        ag = np.asarray(ag, dtype=np.float64)
        n_steps = len(ag)
    else:
        if n_steps is None:
            raise ValueError("Either `ag` or `n_steps` must be provided.")
        ag = np.zeros(n_steps, dtype=np.float64)

    time_vec = np.arange(n_steps, dtype=np.float64) * dt

    # Initialise OpenSees model
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Nodes: 1 (base, fixed), 2 (mass, free)
    ops.node(1, 0.0)
    ops.node(2, 0.0)
    ops.fix(1, 1)
    ops.mass(2, params.mass)

    # Material definition
    mat_tag = 1
    elem_tag = 1
    k0 = params.k0

    if params.material_type in ("elastic", "linear"):
        ops.uniaxialMaterial("Elastic", mat_tag, k0)
    elif params.material_type in ("bilinear", "hysteretic", "steel01"):
        if params.u_y is None or params.u_y <= 0.0:
            raise ValueError(f"Valid positive yield displacement u_y is required for bilinear material, got {params.u_y}")
        fy = params.Fy
        alpha = params.alpha
        # Steel01: matTag, Fy, E0, b (isotropic hardening parameters default to 0)
        ops.uniaxialMaterial("Steel01", mat_tag, fy, k0, alpha)
    else:
        raise ValueError(f"Unsupported material_type: {params.material_type}. Choose 'elastic' or 'bilinear'.")

    ops.element("zeroLength", elem_tag, 1, 2, "-mat", mat_tag, "-dir", 1)

    # Damping setup
    if params.damping_type == "mass":
        alpha_m = 2.0 * params.zeta * params.omega_n
        beta_k = 0.0
    elif params.damping_type == "initial_stiffness":
        alpha_m = 0.0
        beta_k = 2.0 * params.zeta / params.omega_n
    else:
        raise ValueError(f"Unsupported damping_type: {params.damping_type}")

    ops.rayleigh(alpha_m, 0.0, beta_k, 0.0)

    # Set initial state if nonzero
    if abs(u0) > 1e-15:
        ops.setNodeDisp(2, 1, u0, "-commit")
    if abs(v0) > 1e-15:
        ops.setNodeVel(2, 1, v0, "-commit")

    # Set up ground motion if non-zero
    has_motion = np.any(np.abs(ag) > 1e-15)
    if has_motion:
        ts_tag = 1
        pattern_tag = 1
        ops.timeSeries("Path", ts_tag, "-dt", dt, "-values", *ag.tolist())
        ops.pattern("UniformExcitation", pattern_tag, 1, "-accel", ts_tag)

    # Analysis setup
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGen")
    ops.test("NormUnbalance", tol, max_iter)
    
    if params.material_type in ("elastic", "linear") and not has_motion:
        ops.algorithm("Linear")
    else:
        ops.algorithm("Newton")

    ops.integrator("Newmark", gamma, beta)
    ops.analysis("Transient")

    # Storage arrays
    u_hist = np.zeros(n_steps, dtype=np.float64)
    v_hist = np.zeros(n_steps, dtype=np.float64)
    a_hist = np.zeros(n_steps, dtype=np.float64)
    f_hist = np.zeros(n_steps, dtype=np.float64)

    # Initial state at t = 0
    u_hist[0] = u0
    v_hist[0] = v0
    # Initial restoring force
    if params.material_type in ("elastic", "linear"):
        f_hist[0] = k0 * u0
    else:
        # For bilinear, sign of u0 determines force
        if abs(u0) <= (params.u_y or 1e-12):
            f_hist[0] = k0 * u0
        else:
            sign_u = 1.0 if u0 > 0 else -1.0
            f_hist[0] = sign_u * (params.Fy + params.alpha * k0 * (abs(u0) - params.u_y))

    # Initial acceleration: m * a0 + c * v0 + f0 = -m * ag[0]
    c_val = params.c
    a_hist[0] = (-ag[0] * params.mass - c_val * v0 - f_hist[0]) / params.mass

    # Transient time stepping
    algorithms_fallback = ["Newton", "ModifiedNewton", "KrylovNewton", "NewtonLineSearch", "BFGS"]

    for i in range(1, n_steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            # Attempt fallback algorithms
            converged = False
            for alg in algorithms_fallback:
                ops.algorithm(alg)
                test_ok = ops.analyze(1, dt)
                if test_ok == 0:
                    converged = True
                    break
            if not converged:
                ops.wipe()
                raise RuntimeError(
                    f"OpenSees transient analysis failed to converge at step {i} "
                    f"(t = {i * dt:.4f} s) for T={params.T}, material={params.material_type}"
                )

        u_hist[i] = ops.nodeDisp(2, 1)
        v_hist[i] = ops.nodeVel(2, 1)
        a_hist[i] = ops.nodeAccel(2, 1)
        f_hist[i] = ops.basicForce(elem_tag)[0]

    ops.wipe()

    # Absolute acceleration
    total_accel = a_hist + ag

    # Energy calculations
    e_h = compute_cumulative_energy(f_hist, u_hist)
    
    # Viscous damping energy integral(c * v du) = integral(c * v^2 dt)
    # Using trapezoidal rule on c * v^2 * dt
    c_force = c_val * v_hist
    e_d = compute_cumulative_energy(c_force, u_hist)

    # Kinetic energy 0.5 * m * v^2
    e_k = 0.5 * params.mass * (v_hist ** 2)

    # Elastic strain energy
    if params.material_type in ("elastic", "linear"):
        e_s = 0.5 * k0 * (u_hist ** 2)
    else:
        # Recoverable elastic strain energy = 0.5 * F_R^2 / k0
        e_s = 0.5 * (f_hist ** 2) / k0

    return SDOFResponse(
        time=time_vec,
        u=u_hist,
        v=v_hist,
        a=a_hist,
        total_accel=total_accel,
        f_r=f_hist,
        e_h=e_h,
        e_d=e_d,
        e_k=e_k,
        e_s=e_s,
        ag=ag,
        dt=dt,
    )


class OpenSeesSDOF:
    """Object-oriented wrapper around the OpenSees SDOF simulator."""

    def __init__(
        self,
        T: float,
        zeta: float = 0.05,
        material_type: Literal["elastic", "bilinear"] = "elastic",
        u_y: Optional[float] = None,
        alpha: float = 0.02,
        mass: float = 1.0,
        damping_type: Literal["mass", "initial_stiffness"] = "mass",
    ):
        self.params = SDOFParams(
            T=T,
            zeta=zeta,
            material_type=material_type,
            u_y=u_y,
            alpha=alpha,
            mass=mass,
            damping_type=damping_type,
        )

    def simulate(
        self,
        ag: Optional[np.ndarray] = None,
        dt: float = 0.01,
        n_steps: Optional[int] = None,
        u0: float = 0.0,
        v0: float = 0.0,
        gamma: float = 0.5,
        beta: float = 0.25,
    ) -> SDOFResponse:
        """Run dynamic simulation for ground motion `ag` or free vibration."""
        return simulate_sdof(
            params=self.params,
            ag=ag,
            dt=dt,
            n_steps=n_steps,
            u0=u0,
            v0=v0,
            gamma=gamma,
            beta=beta,
        )
