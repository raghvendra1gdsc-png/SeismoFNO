# SEISMOFNO EXP5 — INDEPENDENT FORENSIC AUDIT

**Audit Date:** September 7, 2026  
**Auditor:** Independent Scientific Audit Agent  
**Audited Target:** EXP 5 — Spatiotemporal Graph Neural Operator (GNO) for Multi-Story Structural Dynamics  
**Repository:** `SeismoFNO` (Strict Project Boundary Enforced)  
**Overall Verdict:** **PASS WITH CAVEATS**

---

## A. Executive Verdict

EXP5 is **scientifically sound, methodologically rigorous, and reproducible**. The central hypothesis **H1 (Topology Invariance)** is confirmed by repository evidence: eliminating fixed-grid zero-padding through native structural graphs directly resolves the 3-story failure mode from EXP4, dropping 3-story relative $L_2$ displacement error from **99.60% (EXP4 FNO2D)** to **22.09% (EXP5 GNO on unseen earthquakes)** and **11.28% (in-distribution)**. Furthermore, strict structural-group and earthquake-group partitioning eliminated the structural contamination observed in EXP4, with zero sample overlap across train, validation, and OOD partitions.

The qualification **PASS WITH CAVEATS** reflects two minor, non-fatal scientific findings that must be accurately contextualized for civil/structural engineering experts:
1. **Modal Extrapolation Phase Drift in OOD-B:** The high relative $L_2$ error on unseen structure `5S_T120` (125.39%) was proven by FFT forensic analysis to stem from fundamental modal period extrapolation ($T_1 = 1.20\text{ s}$ vs. training $T_1 \le 0.85\text{ s}$), where the model accurately predicts displacement envelope amplitude (**15.11% median peak error**) but defaults its oscillation frequency to $\sim 1.12\text{ Hz}$ instead of the true $0.83\text{ Hz}$.
2. **Language Precision in Reporting:** The report refers to the 3-story improvement as a "77.5% absolute error reduction," which conflates absolute percentage-point reduction ($77.51$ percentage points) with relative error reduction ($77.82\%$).

---

## B. Claim Verification Table

| Scientific Claim | Status | Forensic Evidence | Severity / Note |
| :--- | :---: | :--- | :---: |
| **Native graph representation** | 🟢 **VERIFIED** | `src/data_pipeline/graph_dataset.py` constructs exact 3-node graphs for 3-story and 5-node graphs for 5-story buildings with physical column connectivity. | None |
| **No physical zero-padding** | 🟢 **VERIFIED** | Batch collation in `collate_structural_graphs` uses disjoint block-diagonal union ($\sum N_b$ nodes). Zero padding nodes are completely absent. | None |
| **OOD-A earthquake isolation** | 🟢 **VERIFIED** | Events RSN0011 and RSN0012 have exactly 0 simulations in train, val, and ID test partitions. | None |
| **OOD-B structural isolation** | 🟢 **VERIFIED** | Archetype `5S_T120` has exactly 0 simulations in train, val, and ID test partitions. Evaluated on 240 held-out simulations. | None |
| **OOD-C combined isolation** | 🟢 **VERIFIED** | All 60 simulations in OOD-C are simultaneously `5S_T120` and RSN0011/RSN0012, completely excluded from training/validation. | None |
| **Training-only scaler** | 🟢 **VERIFIED** | `GraphNormalizer` fitted strictly on `train_df` (1,080 samples). Scaler tensors in `scalers.pt` verified uninfluenced by val or test. | None |
| **Validation-only checkpoint** | 🟢 **VERIFIED** | Checkpoint saved strictly at validation loss minimum (Epoch 16, Val Loss = 0.2485). Test and OOD data never accessed during training. | None |
| **11.28% 3-Story ID Error** | 🟢 **VERIFIED** | Recomputed independently from `id_results.csv`: **11.2751%** median relative $L_2$ error. | None |
| **22.09% 3-Story OOD-A Error** | 🟢 **VERIFIED** | Recomputed independently from `ood_a_results.csv`: **22.0862%** median relative $L_2$ error. | None |
| **15.11% OOD-B Peak Error** | 🟢 **VERIFIED** | Recomputed independently from `ood_b_results.csv`: **15.1101%** median peak displacement error. | None |
| **125.39% OOD-B Rel $L_2$** | 🟢 **VERIFIED** | Recomputed independently from `ood_b_results.csv`: **125.3922%**. Forensic FFT confirmed modal frequency phase drift ($f_{\text{true}}=0.83$ Hz vs $f_{\text{pred}}=1.12$ Hz). | Physics Extrapolation |
| **24.23% OOD-C Peak Error** | 🟢 **VERIFIED** | Recomputed independently from `ood_c_results.csv`: **24.2277%** median peak displacement error. | None |
| **Topology Ablation Effect** | 🟢 **VERIFIED** | Verified difference between `best_checkpoint.pt` (0.2485 val loss) and `best_ablation_no_topology.pt` (0.2503 val loss, $+1.28$ percentage points Rel $L_2$). | Caveat: Single Seed |
| **2.0x Inference Speedup** | 🟢 **VERIFIED** | OpenSeesPy (31.89 ms) vs GNO Single ($B=1$, 15.82 ms) yields $31.885 / 15.817 = \mathbf{2.02\times}$. | Single-sample optimal |
| **674,115 Parameters** | 🟢 **VERIFIED** | Exact parameter count of `SpatiotemporalGNO` reconstructed from checkpoint is **674,115** ($7.0\times$ smaller than EXP4 FNO2D). | None |
| **EXP4 Comparison** | 🟡 **PARTIALLY VERIFIED** | The 180 3-story test simulations are 100% identical between EXP4 and EXP5. However, phrasing as a "77.5% absolute error reduction" is imprecise terminology. | Metrology Phrasing |

---

## C. Leakage Audit & Partition Matrix

The complete 2,160-simulation dataset was cross-tabulated against the generated `split_manifest.csv`:

### 1. Structural Archetype vs. Partition Matrix
| Structural Archetype | Stories | $T_1$ (s) | Train | Val | ID Test | OOD-A | OOD-B | OOD-C | Excluded Val | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `3S_T035` | 3 | 0.35 | 216 | 60 | 24 | 60 | 0 | 0 | 0 | **360** |
| `3S_T060` | 3 | 0.60 | 216 | 60 | 24 | 60 | 0 | 0 | 0 | **360** |
| `3S_T090` | 3 | 0.90 | 216 | 60 | 24 | 60 | 0 | 0 | 0 | **360** |
| `5S_T055` | 5 | 0.55 | 216 | 60 | 24 | 60 | 0 | 0 | 0 | **360** |
| `5S_T085` | 5 | 0.85 | 216 | 60 | 24 | 60 | 0 | 0 | 0 | **360** |
| `5S_T120` | 5 | 1.20 | **0** | **0** | **0** | 0 | 240 | 60 | 60 | **360** |
| **Total** | — | — | **1080** | **300** | **120** | **300** | **240** | **60** | **60** | **2160** |

### 2. Earthquake Event vs. Partition Matrix
| Earthquake Events | Role | Train | Val | ID Test | OOD-A | OOD-B | OOD-C | Excluded Val | Total |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RSN0001–RSN0008** (8 events) | Training / Known EQs | 1080 | 0 | 120 | 0 | 240 | 0 | 0 | **1440** |
| **RSN0009–RSN0010** (2 events) | Validation EQs | 0 | 300 | 0 | 0 | 0 | 0 | 60 | **360** |
| **RSN0011–RSN0012** (2 events) | Held-Out Test EQs | **0** | **0** | **0** | 300 | 0 | 60 | 0 | **360** |
| **Total** | — | **1080** | **300** | **120** | **300** | **240** | **60** | **60** | **2160** |

**Forensic Conclusion:** Zero structural leakage and zero earthquake leakage. The orthogonality between train/val and OOD sets is complete.

---

## D. Metric Reconstruction

All metrics were recomputed independently directly from the raw evaluation CSV files:

```python
rel_l2 = norm(u_pred - u_true) / (norm(u_true) + 1e-8) * 100.0
peak_err = abs(max(|u_pred|) - max(|u_true|)) / (max(|u_true|) + 1e-8) * 100.0
```

| Evaluation Partition | Sample Count | Claimed Rel $L_2$ (%) | Recomputed Rel $L_2$ (%) | Claimed Peak Err (%) | Recomputed Peak Err (%) | Concordance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ID (Overall)** | 120 | 11.44% | **11.4448%** | 8.20% | **8.2004%** | Exact Match |
| — 3-Story Subgroup | 72 | 11.28% | **11.2751%** | 8.20% | **8.2004%** | Exact Match |
| — 5-Story Subgroup | 48 | 11.86% | **11.8567%** | 8.01% | **8.0074%** | Exact Match |
| **OOD-A (Overall)** | 300 | 17.34% | **17.3355%** | 6.11% | **6.1109%** | Exact Match |
| — 3-Story Subgroup | 180 | 22.09% | **22.0862%** | 5.47% | **5.4727%** | Exact Match |
| — 5-Story Subgroup | 120 | 15.17% | **15.1742%** | 7.07% | **7.0700%** | Exact Match |
| **OOD-B (`5S_T120`)** | 240 | 125.39% | **125.3922%** | 15.11% | **15.1101%** | Exact Match |
| **OOD-C (Combined OOD)** | 60 | 119.68% | **119.6780%** | 24.23% | **24.2277%** | Exact Match |

---

## E. EXP4 vs. EXP5 Comparability Analysis

### 1. Sample Concordance
Forensic verification of simulation identifiers confirmed that the **180 3-story simulations evaluated in EXP4 held-out test set are 100% identical** (same simulation IDs, same earthquake accelerograms RSN0011-12, same structural parameters) to the 180 3-story simulations evaluated in EXP5 OOD-A.

### 2. Classification: **DIRECTLY COMPARABLE**
Because the underlying physical systems, input ground motions, and metric definitions are identical, the comparison directly isolates the effect of the spatial neural operator representation:
- **EXP4 FNO2D (Fixed $5 \times 2048$ Grid with Zero-Padding):** **99.60%** median Rel $L_2$ error.
- **EXP5 GNO (Topology-Native Discrete Graph, 0 Padding):** **22.09%** median Rel $L_2$ error.

### 3. Metrological Correction
- **Absolute percentage-point reduction:** $99.60\% - 22.09\% = \mathbf{77.51\text{ percentage points}}$.
- **Relative error reduction:** $\frac{99.60 - 22.09}{99.60} \times 100\% = \mathbf{77.82\%}$.
*(The original report stated "77.5% absolute error reduction", which is an imprecise colloquialism for a 77.51 percentage-point drop).*

---

## F. Forensic Investigation: The 125.39% OOD-B Result

A forensic investigation was conducted on individual response trajectories in OOD-B to determine whether the $125.39\%$ relative $L_2$ error was an implementation artifact or physical behavior:

1. **Bug Audits:**
   - Sign convention mismatch: **NONE**.
   - Time-grid indexing shift: **NONE**.
   - Scaler denormalization bug: **NONE**.
   - Amplitude scale mismatch: **NONE** (Predicted peak roof displacement matches ground truth within **15.11%**).
2. **Spectral Frequency Analysis:**
   - Ground truth structure `5S_T120` has fundamental frequency $f_1 = \frac{1}{1.20\text{ s}} = \mathbf{0.833\text{ Hz}}$.
   - FFT spectral analysis of predicted trajectories reveals a dominant oscillation frequency of $\mathbf{1.12\text{ Hz}}$ ($T_1 \approx 0.89\text{ s}$).
   - **Root Cause:** In the training partition, the longest period 5-story frame was `5S_T085` ($T_1 = 0.85\text{ s}$, $f_1 = 1.18\text{ Hz}$). The model defaulted its temporal Fourier convolution modes towards the upper limit of the training frequency distribution ($\sim 1.12\text{ Hz}$), failing to extrapolate down to $0.833\text{ Hz}$.
3. **Mathematical Plausibility:**
   Two sinusoidal signals with identical amplitude $A$ oscillating at uncorrelated frequencies ($f_1 = 0.83\text{ Hz}$ vs. $f_2 = 1.12\text{ Hz}$, Pearson $r = 0.078$) drift out of phase over $20.48$ seconds. The mathematical expectation of their relative $L_2$ difference is:
   $$\frac{\|A \sin(\omega_1 t) - A \sin(\omega_2 t)\|_2}{\|A \sin(\omega_1 t)\|_2} \approx \sqrt{2} \approx 141.4\%$$
   The observed median relative $L_2$ error of **125.39%** is consistent with two oscillations having accurate amplitudes but drifting out of phase.

---

## G. Topology Ablation Audit

Ablation B evaluated whether physical column/floor message passing improves structural prediction by training `SpatiotemporalGNO(use_topology=False)` on the identical training set:

- **Full GNO (with physical column edges):** Validation Loss = **0.2485** | Val Rel $L_2 = \mathbf{18.20\%}$
- **Ablation B (isolated nodes, no column edges):** Validation Loss = **0.2503** | Val Rel $L_2 = \mathbf{19.48\%}$
- **Auditor Assessment:** The ablation demonstrates a modest positive effect size ($\Delta \text{Rel } L_2 = 1.28$ percentage points) in favor of explicit interstory message passing. However, because this was conducted on a single authoritative seed (seed 42), the scientific conclusion is conservatively stated as:
  > *"The ablation provides modest empirical evidence consistent with a beneficial contribution from explicit structural topology."*

---

## H. Inference Benchmark Audit

Inference timings were re-evaluated on Apple Silicon M1 (MPS backend) with pre-warmup iterations and `torch.mps.synchronize()`:

1. **Single-Sample Inference ($B=1$):**
   - OpenSeesPy 5-story NLTHA: **31.89 ms / simulation** (31.4 sim/s).
   - EXP5 GNO ($B=1$): **15.82 ms / simulation** (63.2 sim/s).
   - **Measured Speedup:** $\mathbf{2.02\times}$.
2. **Batched Inference ($B=32$):**
   - EXP5 GNO ($B=32$): **20.25 ms / simulation** (49.4 sim/s).
   - **Finding on Batched Scaling:** In GNO, a batch of 32 buildings forms a disjoint graph of 128 nodes. On Apple Silicon MPS, computing 4 layers of 1D complex spectral convolutions over 128 nodes increases total floating-point workload by $25.6\times$, which scales execution time from 15.8 ms up to ~650 ms per batch ($20.25$ ms per simulation).
   - **Conclusion:** Unlike fixed-grid FNO2D (which benefits from dense 4D AMX matrix tiling), **single-sample inference is the optimal operational configuration for GNO on Apple Silicon unified memory**, providing a direct **2.0x wall-clock speedup** over OpenSeesPy.

---

## I. Critical Issues & Required Corrections

### Ranked Severity:
- **CRITICAL:** None.
- **HIGH:** None.
- **MEDIUM:** None.
- **LOW:**
  1. *Metrology Phrasing:* Update any documentation stating "77.5% absolute error reduction" to "77.51 percentage-point reduction (77.82% relative error reduction)."
  2. *Serving Recommendation:* Explicitly document that single-sample inference ($B=1$, 15.8 ms) is the recommended deployment mode on Apple Silicon MPS.

---

## J. Final Scientific Conclusion

1. **Is EXP5 scientifically defensible as a research experiment?**  
   **YES.** The experimental design, code, data splitting, training convergence, and metric calculations are verified. The core hypothesis that native graph representations eliminate fixed-grid Fourier zero-padding failure is demonstrated.
2. **Is EXP5 ready to be presented to an expert civil/structural engineering researcher?**  
   **YES, WITH DOCUMENTED FINDINGS.** The report honestly presents both the strengths (77.5 percentage-point error reduction on 3-story frames, 6.1% peak error on unseen earthquakes, 2.0x speedup) and the physical limits of neural operators (phase drift during modal period extrapolation beyond the training envelope).
