# SEISMOFNO: PROFESSOR-FACING INTERACTIVE RESEARCH DEMONSTRATION
## Technical Reference Manual & Demonstration Architecture

---

## 1. PURPOSE & RESEARCH TARGET

The **Professor-Facing Interactive Research Demonstration Layer** provides an academically rigorous, live computational interface designed specifically for:
- IIT Delhi Computer Science & Engineering / AI faculty
- Scientific Machine Learning and Neural Operator researchers
- Computational mechanics and non-linear structural dynamics professors
- Advanced technical research presentations and internship defense

### Core Research Thesis
Can continuous neural operators generalize non-linear dynamic structural responses across both **discrete topological variations** (variable floor counts) and **distribution shifts in structural modal properties** (fundamental natural period $T_1$ and circular frequency $\omega_1$)?

The interface demonstrates the systematic resolution of representation failures from fixed-grid FNOs (EXP4) to topology-native Graph Neural Operators (EXP5) and physics/modal-conditioned Graph Neural Operators (EXP6), backed by 2,160 OpenSeesPy non-linear time-history ground-truth simulations.

---

## 2. SYSTEM ARCHITECTURE

```
                                  [Structure Selector]
                         (3-Story, 5-Story, 5S_T120 Target OOD)
                                           │
                                           ▼
                            [Pre-Earthquake Modal Engine]
                            Theoretical [M], [K] Matrices
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                Undamped Eigenvalues                 Vertical Mode Shapes
              T1-3, omega1-3 Invariants               phi_1, phi_2, phi_3
                        │                                     │
                        │    [Ground Motion Excitation]       │
                        │        PEER NGA West-2 a_g(t)       │
                        │                  │                  │
                        └──────────┐       │                  │
                                   ▼       ▼                  ▼
                            [Topology-Native Graph]    [Modal Deformation UI]
                              V: physical stories         Building wireframe
                              E: physical columns         deflection shape
                              No zero-padding
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │        SURROGATE MODEL LAYER         │
                ├──────────────────────────────────────┤
                │ EXP4: Fixed-Grid FNO2D (Archival)    │ ── Gibbs Ringing Failure (99.60%)
                │ EXP5: Unconditioned GNO (Live/Arch)  │ ── Variable Topology Success (22.09%)
                │ EXP6-B: T1-Conditioned GNO (Live)    │ ── FiLM on T1 Scalar
                │ EXP6-C: Multi-Modal GNO (Live)       │ ── FiLM on [T1-3, w1-3] (13.06% Peak)
                │ EXP6-D: Shuffled Modal (Live/Arch)   │ ── Falsification Control (24.33% Peak)
                └──────────────────────────────────────┘
                                   │
                                   ▼
                      [Response & Error Evaluation]
                 OpenSeesPy Non-Linear Ground Truth vs
                 Surrogate Predicted Displacements u(t)
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
       [Live vs Archival]                    [Scientific Honesty]
     Exact Provenance Labeling              Waveform Phase Drift
     LIVE_COMPUTED_MPS vs ARCHIVAL          Pearson r ~ 0.05 - 0.09
```

### Module Organization
The demonstration layer wraps the frozen research core using zero-intrusion adapters under `src/demo/`:
- `src/demo/model_registry.py`: Central, immutable metadata registry describing parameter counts, checkpoint locations, and SHA256 integrity hashes for EXP4, EXP5, EXP6-B, EXP6-C, and EXP6-D.
- `src/demo/demo_data.py`: Pre-earthquake modal eigenvalue solver, structural archetype catalog (8 verified configurations), and verified PEER accelerogram provider.
- `src/demo/metrics_adapter.py`: Authoritative provider of experimental progression, verified OOD generalization matrix, shuffled ablation data, failure analysis, and Apple Silicon MPS computational benchmarks.
- `src/demo/inference_adapter.py`: Live inference execution pipeline running `ConditionedSpatiotemporalGNO` forward passes on Apple Silicon MPS/CPU with fallback to archival verified records.
- `api/main.py`: REST API endpoints mounted at `/api/v1/demo/*`.
- `frontend/src/components/ResearchDemo/`: Comprehensive technical UI implementing Sections A through P of the master directive.

---

## 3. HOW THE DEMO MAPS TO COMPLETED RESEARCH

| Demonstration Section | Scientific Phase | Repository Source Artifact | Verified Key Finding |
|---|---|---|---|
| **Section B & C: Structural & Modal Engine** | EXP6 Pre-processing | `src/demo/demo_data.py` | Exact eigenvalues $\omega_i^2 \mathbf{M}\phi_i = \mathbf{K}\phi_i$ computed prior to earthquake response. |
| **Section E & F: Graph Representation** | EXP5 Architecture | `src/models/graph_neural_operator.py` | 3 nodes for 3-story, 5 nodes for 5-story; complete elimination of Cartesian zero-padding. |
| **Section G & H: Response Prediction** | EXP4/5/6 Evaluation | `results/experiments/exp6/` | Real-time trajectory comparison against OpenSeesPy ground truth. |
| **Section I: Research Progression** | EXP4 $\to$ EXP5 $\to$ EXP6 | `docs/SEISMOFNO_TECHNICAL_REPORT.md` | 4-stage empirical progression from Gibbs failure to modal envelope scaling. |
| **Section J: OOD Matrix** | EXP6 Generalization | `results/experiments/exp6/eval_summary.json` | 62.9% relative peak error reduction on unseen structure 5S_T120 (35.21% $\to$ 13.06%). |
| **Section K: Failure Analysis** | Scientific Limitations | `results/FINALIZATION_AUDIT.md` | 4 documented failure modes including high-horizon waveform phase drift. |
| **Section L: Speedup Benchmark** | Computational Efficiency | `results/speed_benchmark/` | Single-building inference: 21.45 ms vs 54.68 ms OpenSeesPy ($2.55\times$ wall-clock on MPS). |
| **Section M: Shuffled Ablation** | Scientific Rigor | `results/experiments/exp6/` | Shuffling $T_1$ degrades OOD-B error from 13.06% to 24.33%, falsifying parameter capacity hypothesis. |

---

## 4. STRICT DISTINCTION: LIVE COMPUTED VS. ARCHIVAL VERIFIED

A primary scientific integrity mandate of SeismoFNO is that **no archival metric may be presented as live, and no live inference may fabricate missing numbers**:

1. **`LIVE_COMPUTED_MPS` (Green Badge)**:
   - When the user selects EXP6-B, EXP6-C, or EXP6-D with a supported structure and earthquake, the backend loads the PyTorch checkpoint onto Apple Silicon MPS (or CPU), scales inputs via `scalers.pt`, executes `model(x, edge_index, edge_attr, cond, batch_idx)`, and synchronizes hardware timers.
   - Latency, peak displacement, relative $L_2$ error, and Pearson $r$ are computed in real time.
2. **`ARCHIVAL_FROZEN_VERIFIED` (Slate Badge)**:
   - EXP4 (Fixed-grid FNO2D) requires 2D grid reshaping and zero-padding which was historically evaluated under dedicated experiment scripts.
   - For EXP4, the system loads verified historical evaluation curves and metrics directly from `results/experiments/exp4/` and displays the `ARCHIVAL_FROZEN_VERIFIED` badge.

---

## 5. MODEL REGISTRY & CHECKPOINT INTEGRITY

All models are registered in `src/demo/model_registry.py` with immutable SHA256 checksums:

| Model ID | Phase | Architecture | Parameters | Conditioning | Checkpoint Path |
|---|---|---|---|---|---|
| `exp4_fno2d` | EXP4 | Fixed-Grid FNO2D | 4,735,187 | None (0D) | `results/experiments/exp4/training/best_checkpoint.pt` |
| `exp5_gno` | EXP5 | Spatiotemporal GNO | 674,115 | None (0D) | `results/experiments/exp5/training/best_checkpoint.pt` |
| `exp6_t1_gno` | EXP6-B | Conditioned GNO | 725,059 | $T_1$ Scalar (1D FiLM) | `results/experiments/exp6/training/best_t1_gno.pt` |
| `exp6_multimodal_gno` | EXP6-C | Multi-Modal GNO | 727,619 | $[T_{1-3}, \omega_{1-3}]$ (6D FiLM) | `results/experiments/exp6/training/best_multimodal_gno.pt` |
| `exp6_shuffled_gno` | EXP6-D | Shuffled GNO (Ablation) | 725,059 | Permuted $T_1$ (1D FiLM) | `results/experiments/exp6/training/best_shuffled_modal_gno.pt` |

---

## 6. REPRODUCIBILITY TELEMETRY

The demonstration UI continuously displays active runtime telemetry:
- **Python Runtime**: Python 3.14.5
- **Deep Learning Framework**: PyTorch 2.13.0
- **Compute Acceleration Backend**: Apple Silicon MPS (`torch.backends.mps.is_available() == True`)
- **Random Seed**: 42 (Frozen across all splits and initializations)
- **Ground-Truth Engine**: OpenSeesPy 3.5.1.3 (Non-Linear Time-History Analysis with Newmark-$\beta$ integration)
- **Validation Audit**: 294 unit tests passing with zero regressions (`results/FINALIZATION_AUDIT.md`)

---

## 7. DOCUMENTED SCIENTIFIC BOUNDARIES & LIMITATIONS

The demonstration does not claim universal generalization. It explicitly showcases 4 failure boundaries:
1. **Fixed-Grid Topology Failure (EXP4)**: Zero-padding 3-story buildings onto a 5-story grid causes spatial boundary discontinuities and Gibbs ringing ($99.60\%$ relative $L_2$ error).
2. **Modal Extrapolation Waveform Drift (EXP5/EXP6)**: Static 1D Fourier layers compute global frequency convolutions over the entire duration. Under extreme modal shift ($T_1 = 1.20$s vs training $T_1 \le 0.85$s), minor period discrepancies integrate over 1,024 timesteps, causing waveform phase drift (Relative $L_2 > 100\%$, Pearson $r \approx 0.05\text{--}0.09$). Peak displacement envelopes remain accurate ($13.06\%$ error).
3. **Dynamic Period Elongation Under Severe Yielding**: Static elastic invariants $[T_1, \omega_1]$ derived from undamped $[K],[M]$ cannot represent instantaneous period elongation during severe plastic hinging ($\mu > 4.0$, PGA $\ge 0.8$g).
4. **Planar Shear Building Idealization**: Validation is currently restricted to planar lumped-mass shear-building systems. 3D torsional coupling and diaphragm flexibility represent future research directions.

---

## 8. HOW TO RUN THE DEMONSTRATION

### Option 1: Full-Stack Integrated Launch (Recommended)
1. **Start FastAPI Backend**:
   ```bash
   cd /Users/rahul/seismoFNO
   .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. **Start Frontend Development Server**:
   ```bash
   cd /Users/rahul/seismoFNO/frontend
   npm run dev
   ```
3. **Open Browser**:
   Navigate to `http://localhost:5173/demo` (or click the "Research Defense Demo" tab).

### Option 2: Automated Verification Run
Execute the complete test suite verifying both the frozen research core and the demonstration layer:
```bash
PYTHONPATH=. .venv/bin/pytest -v tests/test_demo_layer.py
PYTHONPATH=. .venv/bin/pytest -v tests/test_usgs_integration.py
PYTHONPATH=. .venv/bin/pytest -q
```
Expected result: **304 passed, 0 failed**.

---

## 9. FROM RESEARCH PROTOTYPE TO REAL-WORLD ENGINEERING WORKFLOW

To bridge the gap between academic research and engineering practice while maintaining absolute scientific honesty, SeismoFNO integrates an observed real-world earthquake layer (USGS) with the following provenance workflow:

```
                  Observed Earthquake (USGS Feed)
                                ↓
                          Event Metadata
             (Magnitude Mw, Hypocenter, Origin Time)
                                ↓
               Ground-Motion Availability Check
           ("Compatible waveform unavailable for direct
              SeismoFNO inference from catalog alone")
                                ↓
                   Compatible Ground Motion
            (Verified Historical Record / Sensor a_g(t))
                                ↓
                        Structural Model
               (5-Story Archetype [M], [K], T1-3)
                                ↓
                            SeismoFNO
                  (Frozen EXP6 Neural Operator)
                                ↓
                     Rapid Response Estimate
                      u(t), Peak Disp, Rel L2
                                ↓
                     Engineering Screening
        (Portfolio Prioritization, Targeted NLTHA Trigger)
```

### Scientific Honesty Note
USGS event metadata is used to connect the research prototype to observed earthquake events. SeismoFNO inference requires an appropriate ground-motion input and does not infer a structural response from earthquake magnitude/location alone.

> **Future Work Requirement:**
> "Future work would require validated waveform acquisition, site-specific ground-motion characterization, structural sensing, uncertainty quantification, and extensive external validation before operational engineering deployment."

