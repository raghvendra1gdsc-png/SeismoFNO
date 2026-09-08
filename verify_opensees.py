"""
verify_opensees.py — Stage 0 end-to-end check.

Runs a trivial undamped SDOF (mass-spring) free-vibration analysis in
OpenSeesPy and checks that the numerical solution matches the analytical
solution to within 1 %.

Analytical solution for undamped free vibration with x(0)=0, v(0)=1 m/s:
    x(t) = v0 / omega * sin(omega * t)
where omega = sqrt(k/m).

Parameters (deliberately simple round numbers):
    m = 1.0 kg,  k = 4*pi^2 N/m  =>  T = 1.0 s,  omega = 2*pi rad/s
    v0 = 1.0 m/s
"""

import math
import sys

import numpy as np
import openseespy.opensees as ops


def run_sdof() -> dict:
    """Build and run a 1-DOF free-vibration model; return displacement history."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    # Nodes
    ops.node(1, 0.0)   # fixed base
    ops.node(2, 0.0)   # mass DOF

    ops.fix(1, 1)

    # Mass
    m = 1.0
    ops.mass(2, m)

    # Spring  (elastic truss element)
    T = 1.0                          # target period [s]
    omega = 2.0 * math.pi / T        # rad/s
    k = m * omega ** 2               # stiffness [N/m]

    ops.uniaxialMaterial("Elastic", 1, k)
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)

    # Initial velocity (v0 = 1 m/s at node 2)
    v0 = 1.0
    ops.setNodeVel(2, 1, v0, "-commit")

    # Newmark transient analysis (gamma=0.5, beta=0.25 => average acceleration)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGen")
    ops.test("NormUnbalance", 1e-12, 10)
    ops.algorithm("Linear")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    dt = 0.01   # 100 Hz sampling
    n_steps = 200  # 2 seconds of simulation
    times = []
    disps = []

    for i in range(n_steps):
        ok = ops.analyze(1, dt)
        if ok != 0:
            raise RuntimeError(f"Analysis failed at step {i}")
        times.append((i + 1) * dt)
        disps.append(ops.nodeDisp(2, 1))

    ops.wipe()
    return {
        "times": np.array(times),
        "disps": np.array(disps),
        "omega": omega,
        "v0": v0,
    }


def analytical(t, omega, v0):
    return (v0 / omega) * np.sin(omega * t)


def main():
    print("=" * 60)
    print("SeismoFNO — Stage 0 OpenSeesPy verification")
    print("=" * 60)

    import importlib.metadata as _meta
    try:
        _ops_ver = _meta.version("openseespy")
    except _meta.PackageNotFoundError:
        _ops_ver = "unknown"
    print(f"openseespy version : {_ops_ver}")
    import torch
    print(f"torch version      : {torch.__version__}")
    import neuralop as _no
    _no_ver = getattr(_no, "__version__", None) or _meta.version("neuraloperator")
    print(f"neuraloperator ver : {_no_ver}")
    print()

    result = run_sdof()
    t = result["times"]
    u_num = result["disps"]
    u_ana = analytical(t, result["omega"], result["v0"])

    rel_l2 = np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana)
    max_abs = np.max(np.abs(u_num - u_ana))
    u_peak_ana = np.max(np.abs(u_ana))

    print(f"SDOF free-vibration check (T=1 s, v0=1 m/s, dt=0.01 s, 200 steps)")
    print(f"  Analytical peak amplitude : {u_peak_ana:.6f} m")
    print(f"  Numerical  peak amplitude : {np.max(np.abs(u_num)):.6f} m")
    print(f"  Relative L2 error         : {rel_l2:.2e}")
    print(f"  Max absolute error        : {max_abs:.2e} m")

    if rel_l2 < 0.01:
        print("\n[PASS] OpenSeesPy SDOF analysis verified. Relative L2 < 1 %.")
        return 0
    else:
        print("\n[FAIL] Relative L2 error exceeds 1 %. Check model setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
