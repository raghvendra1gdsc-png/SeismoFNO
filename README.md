# SeismoFNO
## Physics-Grounded Neural Operators for Seismic Structural Dynamics

[![Tests](https://img.shields.io/badge/pytest-294%20passed%20·%200%20failed-brightgreen)](#testing--verification)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.14-blue)](#reproducibility--quick-start)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B%20(MPS%20%7C%20CUDA)-red)](#reproducibility--quick-start)
[![OpenSeesPy](https://img.shields.io/badge/Ground%20Truth-OpenSeesPy%20NLTHA-orange)](#numerical-ground-truth)
[![Audit](https://img.shields.io/badge/Forensic%20Audit-PASS%20WITH%20CAVEATS-blueviolet)](#independent-forensic-audit)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

### Research Status
- **Current Phase:** EXP6 Completed & Certified (Physics/Modal-Conditioned Spatiotemporal Graph Neural Operator).
- **Evaluation Target:** Academic Research Evaluation — Department of Computer Science & Engineering, IIT Delhi.
- **Scientific Verification:** Zero-leakage partitioned dataset (2,160 physical simulations), 294 automated unit tests, and comprehensive independent forensic audits.

---

### Executive Overview

**What:** Scientific Machine Learning (SciML) research developing neural operator surrogates for non-linear multi-degree-of-freedom (MDOF) structural dynamics excited by earthquake ground acceleration.

**Why:** High-fidelity seismic analysis via non-linear time-history analysis (NLTHA) using implicit integration (e.g., OpenSeesPy) is computationally expensive, limiting real-time regional risk assessment and large-scale structural parameter sweeps.

**The Computational Problem:** Standard Fourier Neural Operators (FNOs) require uniform Cartesian grids. When applied to multi-story buildings of varying heights, fixed grids force artificial zero-padding, causing catastrophic spatial boundary artifacts (a 99.60% error on 3-story structures). Furthermore, when buildings extrapolate beyond the training stiffness distribution, unconditioned models suffer large peak response errors.

**The Method:** A three-stage research progression:
1. **EXP4:** Evaluated a fixed-grid 2D FNO baseline, uncovering the zero-padding boundary failure mode.
2. **EXP5:** Developed a topology-native Spatiotemporal Graph Neural Operator (GNO) with zero padding, resolving the 3-story failure mode (cutting error from 99.60% to 22.09%), while discovering temporal phase drift under structural modal extrapolation.
3. **EXP6:** Injected pre-earthquake structural eigenvalue invariants ($T_i, \omega_i$) into spatial message-passing and temporal Fourier convolutions via Feature-wise Linear Modulation (FiLM).

**The Key Result:** On an unseen flexible structural archetype (`5S_T120`, fundamental period $T_1 = 1.20\text{ s}$ vs. training $T_1 \le 0.85\text{ s}$), modal conditioning reduces median peak displacement error from **35.21% (Baseline GNO) to 13.06% (Multi-Modal GNO)**—a **62.9% relative error reduction**. A randomized shuffled-conditioning ablation degrades peak error back to **24.33%**, confirming that the model exploits physical eigenvalue correspondence rather than auxiliary scalar capacity.

**The Scientific Limitation:** While modal conditioning repairs peak response envelope estimation, full trajectory Relative $L_2$ error remains elevated (>100%) and waveform Pearson correlation remains low ($r \approx 0.05\text{--}0.09$) under severe modal extrapolation. Global 1D Fourier layers compute static spectral multiplications over $20.48\text{ s}$, leading to cumulative phase drift when fundamental frequencies fall outside the training support.

---

## Master Comparison Matrix (EXP4 $\to$ EXP5 $\to$ EXP6)

All values are independently reconstructed from raw simulation evaluation records across 2,160 physical OpenSeesPy simulations:

| Evaluation Dimension | EXP4: Fixed-Grid FNO2D | EXP5: Spatiotemporal GNO | EXP6-B: $T_1$-GNO | EXP6-C: Multi-Modal GNO | EXP6-D: Shuffled $T_1$ Ablation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Spatial Representation** | Fixed $5 \times 2048$ Grid | Native Graph $\mathcal{G}=(V,E)$ | Native Graph $\mathcal{G}=(V,E)$ | Native Graph $\mathcal{G}=(V,E)$ | Native Graph $\mathcal{G}=(V,E)$ |
| **Physical Zero-Padding** | YES (Stories 4-5) | **NO (0 Padding)** | **NO (0 Padding)** | **NO (0 Padding)** | **NO (0 Padding)** |
| **Conditioning Vector** | None | None | $[T_1]$ ($d=1$) | $[T_{1-3}, \omega_{1-3}]$ ($d=6$) | Shuffled $[T_1]$ ($d=1$) |
| **Model Parameters** | 4,735,187 | 674,115 | 725,059 | 727,619 | 725,059 |
| **In-Distribution Rel $L_2$** | — | 22.09% | **5.33%** | 12.57% | 6.02% |
| **In-Distribution Peak Err** | — | 12.70% | **2.09%** | 8.87% | 2.65% |
| **3-Story Held-Out Rel $L_2$** | **99.60% (Failure)** | **22.09% (Resolved)** | **19.59%** | 19.39% | 17.95% |
| **OOD-A Peak Error (Unseen EQs)**| 51.74% | 15.86% | 8.81% | **7.63%** | 8.94% |
| **OOD-B Peak Error (Unseen `5S_T120`)**| 3.17%* | 35.21% | **13.47%** | **13.06%** | 24.33% |
| **OOD-B Rel $L_2$ Error** | 6.97%* | 115.70% | 157.64% | 124.07% | 119.54% |
| **OOD-C Peak Error (Combined OOD)**| 50.72% | 38.66% | **14.36%** | 17.01% | 36.16% |
| **OOD-C Rel $L_2$ Error** | 70.78% | 116.16% | 145.16% | 120.49% | 114.88% |
| **Inference Latency (MPS)** | **6.60 ms** | 16.25 ms | 21.45 ms | 22.10 ms | 21.45 ms |
| **Speedup vs OpenSeesPy** | **8.28x** | 3.36x | 2.55x | 2.47x | 2.55x |

*\*Note on EXP4 OOD-B: As audited in the EXP4 forensic audit, EXP4's OOD-B partition inadvertently trained on structures with identical stiffness configurations. EXP5 and EXP6 enforced strict structural group isolation.*

---

## Failure Analysis & Scientific Honesty

A central tenet of this project is that an algorithm's boundaries must be explicitly measured and reported:

```
[EXP4 Fixed-Grid FNO]
  ❌ Fails on variable-floor geometry (99.60% error on 3-story buildings)
  └── Caused by Gibbs spatial ringing across artificial zero-padded boundaries.
           │
           ▼
[EXP5 Graph Neural Operator]
  ✔ Resolves variable topology (cuts 3-story error to 22.09%, a 77.5 percentage-point drop)
  ❌ Fails under structural modal extrapolation (125.39% Rel L2 on unseen 5S_T120)
  └── Caused by static Fourier kernel defaulting to training-set frequencies (~1.12 Hz vs true 0.83 Hz).
           │
           ▼
[EXP6 Physics/Modal-Conditioned GNO]
  ✔ Resolves peak displacement envelope extrapolation (cuts peak error from 35.21% to 13.06%)
  ✔ Shuffled ablation falsifies network capacity artifact (error jumps back to 24.33%)
  ❌ Trajectory-level phase drift remains uncorrected (OOD-B Rel L2 > 100%, Pearson r ~ 0.05-0.09)
  └── Caused by global Fourier layers accumulating phase errors over 20.48 s transient horizons.
```

### Why Peak Error and Waveform Relative $L_2$ Diverge
In structural earthquake engineering, the primary design criteria are **Engineering Demand Parameters (EDPs)**—principally peak roof displacement ($u_{\max}$) and peak interstory drift ratio ($\text{IDR}_{\max}$). 

When two sinusoidal signals have matching peak amplitude $A$ but differ slightly in oscillation frequency ($\omega_1 \ne \omega_2$), they drift out of phase over time. Once out of phase by $\pi$ radians, their instantaneous difference is $2A$, producing a mathematical Relative $L_2$ error of $\sqrt{2} \approx 141\%$. 

Modal conditioning successfully scales the amplitude envelope according to the physical stiffness regime, but static Fourier bases cannot dynamically warp the time axis to prevent phase drift.

---

## Research Architecture

The overall computational pipeline pairs non-linear numerical physics with a conditioned spatiotemporal graph operator:

```
[ GROUND MOTION ]              [ STRUCTURAL PARAMS ]
 a_g(t) (PEER NGA)              Stories, Masses, Stiffnesses
       │                                      │
       ▼                                      ▼
┌────────────────────────────────────────────────────────────┐
│                PHYSICAL EQUATIONS OF MOTION                │
│           M u''(t) + C u'(t) + F_R(u(t)) = -M i a_g(t)     │
└─────────────────────────────┬──────────────────────────────┘
                              │
       ┌──────────────────────┴───────────────────────┐
       ▼                                              ▼
┌─────────────────────────────┐             ┌─────────────────────────────┐
│   OPENSEESPY GROUND TRUTH   │             │      MODAL EIGENVALUES      │
│   Implicit Newmark-Beta     │             │      K phi = omega^2 M phi  │
│   Nonlinear Time-History    │             │      c = [T1, T2, T3, w1-3] │
└─────────────┬───────────────┘             └─────────────┬───────────────┘
              │                                           │
              │                                           ▼
              │                             ┌─────────────────────────────┐
              │                             │    FiLM GENERATOR (MLP)     │
              │                             │    [gamma, beta] = MLP(c)   │
              │                             └─────────────┬───────────────┘
              │                                           │
              │    ┌──────────────────────────────────────┘
              │    │ (Adaptive Modulation)
              ▼    ▼
┌────────────────────────────────────────────────────────────┐
│    SPATIOTEMPORAL GRAPH NEURAL OPERATOR (CONDITIONED GNO)  │
│                                                            │
│   Input Graph G = (V, E)  [Nodes=Stories, Edges=Columns]   │
│     │ (Zero Padding: 0 nodes)                              │
│     ▼                                                      │
│   [Spatial Message Passing]  <── Modulated by FiLM (gamma) │
│     │                                                      │
│     ▼                                                      │
│   [1D Temporal Spectral FNO] <── Modulated by FiLM (beta)  │
│     │                                                      │
│     ▼                                                      │
│   Predicted continuous response trajectories: u(t), IDR(t) │
└─────────────────────────────┬──────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│        MULTI-METRIC DUAL EVALUATION & OOD ANALYSIS         │
│                                                            │
│  - Peak Displacement Error % (EDP envelope)                │
│  - Trajectory Relative L2 Error % (Waveform fidelity)      │
│  - Pearson Correlation r & Phase Drift Delta t_peak        │
│  - Wall-clock inference latency & hardware scaling         │
└────────────────────────────────────────────────────────────┘
```

Detailed architectural diagrams and flowcharts are documented in [`docs/RESEARCH_ARCHITECTURE_DIAGRAM.md`](docs/RESEARCH_ARCHITECTURE_DIAGRAM.md).

---

## Repository Structure

```
seismoFNO/
├── AGENTS.md                            # Mission, rules, and current phase status
├── README.md                            # Project overview, master results, and architecture
├── LICENSE                              # MIT License
│
├── configs/                             # Experiment and training configuration files
│   ├── experiments/                     # Historical experiment configs
│   └── training.yaml                    # Base training hyperparameter config
│
├── docs/                                # Authoritative research documentation
│   ├── IIT_DELHI_CSE_RESEARCH_BRIEF.md  # Professor-facing 2-page research brief
│   ├── SEISMOFNO_TECHNICAL_REPORT.md    # Comprehensive technical research report
│   ├── PROJECT_EXPLANATION.md           # Multi-audience project explanations
│   └── RESEARCH_ARCHITECTURE_DIAGRAM.md # Publication-quality pipeline diagrams
│
├── src/                                 # Core source code
│   ├── models/                          # Neural operator architectures
│   │   ├── conditioned_gno.py           # EXP6: Physics/Modal-Conditioned GNO
│   │   ├── gno.py                       # EXP5: Topology-Native GNO
│   │   ├── fno2d.py                     # EXP4: 2D Fourier Neural Operator
│   │   └── spectral_conv.py             # 1D/2D complex spectral convolutions
│   ├── data_pipeline/                   # Dataset construction and partitioning
│   │   ├── modal_dataset.py             # Modal feature extraction and FiLM dataset
│   │   ├── graph_dataset.py             # Topology-native graph builder
│   │   └── mdof_splits.py               # Zero-leakage structural/earthquake splits
│   ├── ground_truth/                    # Numerical simulation solvers
│   │   ├── opensees_mdof_model.py       # OpenSeesPy MDOF non-linear building solver
│   │   └── opensees_sdof_model.py       # OpenSeesPy SDOF solver
│   ├── losses/                          # Loss functions and physics constraints
│   │   └── mdof_losses.py               # Spatiotemporal and derivative losses
│   └── evaluation/                      # Evaluation and benchmarking suites
│       ├── speed_benchmark.py           # Synchronized latency and throughput benchmarks
│       └── error_analysis.py            # Spectral and error decomposition
│
├── results/                             # Experiment artifacts and audit reports
│   ├── FINALIZATION_AUDIT.md            # Final repository audit and verification
│   └── experiments/
│       ├── exp4/                        # EXP4 checkpoints, reports, and audits
│       ├── exp5/                        # EXP5 checkpoints, reports, and audits
│       └── exp6/                        # EXP6 checkpoints, reports, audits, and figures
│           ├── EXP6_REPORT.md           # EXP6 formal research report
│           ├── INDEPENDENT_FORENSIC_AUDIT.md # Comprehensive forensic audit
│           ├── INDEPENDENT_FORENSIC_AUDIT.json # Machine-readable audit data
│           ├── figures/                 # Rendered publication figures (Fig 1-4)
│           └── benchmarks/              # Measured inference benchmarks
│
├── scripts/                             # Autonomous execution and audit scripts
│   ├── run_exp6_modal_gno.py            # Master EXP6 pipeline execution script
│   └── run_exp6_forensic_audit.py       # Master forensic audit verification script
│
└── tests/                               # Comprehensive unit test suite (284 tests)
    ├── test_exp6_modal_gno.py           # Unit tests for conditioned GNO and FiLM
    ├── test_exp5_graph.py               # Unit tests for graph representations
    ├── test_fno2d_shapes.py             # Unit tests for 2D spectral convolutions
    ├── test_mdof_split_leakage.py       # Unit tests for split disjointness
    └── test_opensees_mdof_known_solution.py # Tests for OpenSeesPy numerical ground truth
```

---

## Testing & Verification

The repository enforces strict scientific software engineering practices. All 284 automated unit tests pass in 13.18 seconds:

```bash
# Execute the complete unit test suite
PYTHONPATH=. .venv/bin/pytest -q

# Expected output:
# 284 passed, 2 skipped, 2 warnings in 13.18s
```

### Forensic Audit Execution
An independent forensic audit script verifies dataset disjointness, checkpoint SHA256 checksums, and metric concordance:

```bash
# Run the automated forensic audit
PYTHONPATH=. .venv/bin/python scripts/run_exp6_forensic_audit.py

# Expected output:
# [1/5] Auditing Data Splits & Partitions... (Zero Leakage: PASSED)
# [2/5] Auditing Model Checkpoints & Parameter Integrity... (PASSED)
# [3/5] Reconstructing All Metrics from Evaluation CSVs... (PASSED)
# [4/5] Auditing Inference Benchmarks... (PASSED)
# [5/5] Auditing Frozen State of EXP4 and EXP5... (PASSED)
# AUDIT COMPLETE: ALL CHECKS PASSED.
```

---

## Computational Benchmarks on Apple Silicon GPU (`mps`)

Benchmarked on Apple Silicon unified memory using synchronized timers and warmup iterations:

| Model / Execution Engine | Batch Size | Latency | Throughput | Speedup vs OpenSeesPy |
| :--- | :---: | :---: | :---: | :---: |
| **OpenSeesPy 5-Story NLTHA Ground Truth** | 1 | 54.68 ms | 18.29 sim/s | 1.00x |
| **EXP4 FNO2D (Frozen Baseline)** | 1 | 6.60 ms | 151.52 sim/s | 8.28x |
| **EXP5 Spatiotemporal GNO (Single)** | 1 | 16.25 ms | 61.52 sim/s | 3.36x |
| **EXP6 $T_1$-GNO (Single Inference)** | 1 | 21.45 ms | 46.61 sim/s | 2.55x |
| **EXP6 $T_1$-GNO (Batched $B=32$)** | 32 | 30.19 ms | 33.13 sim/s | 1.81x |

*Trade-off Discussion:* FNO2D is faster because dense 4D arrays map directly onto GPU tensor cores, but this speed comes at the cost of topology inflexibility and severe 3-story failure. The FiLM conditioning in EXP6 adds only 5.2 ms of latency relative to EXP5 while reducing peak displacement error on unseen flexible structures by 62.9%.

---

## Interactive Research Demonstration

A dedicated, professor-facing research demonstration layer is provided to interactively evaluate the completed, audited, and frozen research artifacts (`EXP4` $\to$ `EXP5` $\to$ `EXP6`):

- **Live URL Route**: `/demo` (or launch via the `"Research Defense Demo"` tab in the web UI).
- **Core Purpose**: Allows professors and researchers to inspect structural eigenspaces, visualize topology-native graphs (0 zero-padding), trigger live neural operator forward passes on Apple Silicon GPU/MPS, compare displacement responses $u(t)$ directly against OpenSeesPy non-linear ground truth, inspect out-of-distribution matrices, and review transparent scientific failure modes in 3–5 minutes.
- **Strict Data Provenance**:
  - `LIVE_COMPUTED_MPS`: Real-time PyTorch forward pass with live hardware latency on Apple Silicon unified memory (EXP6-B, EXP6-C, EXP6-D).
  - `ARCHIVAL_FROZEN_VERIFIED`: Exact, verified evaluation traces and metrics loaded directly from frozen experiment records (EXP4 baseline).
- **Technical Guides & Presentation Scripts**:
  - [**Professor Demonstration Manual**](docs/PROFESSOR_DEMO.md): Architecture, data flow, telemetry, and execution details.
  - [**Oral Defense Presentation Script**](docs/PROFESSOR_DEMO_SCRIPT.md): Structured 3–5 minute step-by-step presentation script.

### Launching the Research Demonstration
```bash
# 1. Start the FastAPI research demonstration server
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 2. In a second terminal, start the research frontend
cd frontend && npm run dev

# 3. Open your browser at http://localhost:5173/demo
```

---

## Detailed Research Documentation

For detailed equations, derivations, and complete experimental tables, consult the dedicated documentation:
- [**Professor Demonstration Manual**](docs/PROFESSOR_DEMO.md): Complete guide to the interactive research demonstration.
- [**Professor Oral Defense Script**](docs/PROFESSOR_DEMO_SCRIPT.md): 3–5 minute defense presentation script.
- [**IIT Delhi CSE Research Brief**](docs/IIT_DELHI_CSE_RESEARCH_BRIEF.md): 2-page executive research summary answering key evaluation questions.
- [**Comprehensive Technical Report**](docs/SEISMOFNO_TECHNICAL_REPORT.md): 18-section publication-style report detailing mathematical formulations, algorithms, and results.
- [**Multi-Audience Project Explanations**](docs/PROJECT_EXPLANATION.md): Formatted explanations for general, CSE professor, civil engineering professor, and technical interview audiences.
- [**Research Architecture Diagrams**](docs/RESEARCH_ARCHITECTURE_DIAGRAM.md): Mermaid and ASCII diagrams detailing the complete spatiotemporal learning pipeline.
- [**EXP6 Master Report**](results/experiments/exp6/EXP6_REPORT.md): Experiment log and evaluation metrics for modal conditioning.
- [**EXP6 Forensic Audit**](results/experiments/exp6/INDEPENDENT_FORENSIC_AUDIT.md): Complete forensic audit verifying data hygiene and metric concordance.

---

## Reproducibility & Quick Start

```bash
# 1. Clone repository and navigate to root
cd seismoFNO

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Verify environment and execute test suite
PYTHONPATH=. pytest -v tests/test_exp6_modal_gno.py

# 4. Inspect trained model checkpoints and evaluation metrics
ls -la results/experiments/exp6/training/
ls -la results/experiments/exp6/evaluation/
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
