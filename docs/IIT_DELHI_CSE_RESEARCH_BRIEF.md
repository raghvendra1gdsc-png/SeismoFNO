# SeismoFNO — Research Brief
## Physics-Grounded Neural Operators for Seismic Structural Dynamics

**Author:** Raghvendra Singh Gahlot (B.Tech, Civil Engineering)  
**Focus Area:** Scientific Machine Learning (SciML), Neural Operators, Graph Representation Learning, Out-of-Distribution (OOD) Generalization  
**Repository:** `SeismoFNO`  
**Version:** September 2026 | **Status:** Research Core FROZEN & AUDITED | **305 Passed, 2 Skipped, 0 Failed**  

---

### 1. What is the Computational Problem?
Seismic structural analysis requires predicting multi-story building responses—continuous floor displacements $u_i(t)$, interstory drift ratios $\text{IDR}_i(t)$, and restoring shears $F_{R,i}(t)$—excited by transient non-stationary ground acceleration $a_g(t)$. The underlying physics is governed by a coupled second-order matrix differential equation:
$$M \ddot{u}(t) + C \dot{u}(t) + f_{\text{int}}(u(t), \dot{u}(t)) = -M r a_g(t)$$
Solving this system numerically via non-linear time-history analysis (NLTHA) using implicit integration (e.g., Newton-Raphson in OpenSeesPy) is computationally prohibitive for real-time hazard screening, large-scale parametric sensitivity analysis, and regional risk assessment.

From a Computer Science perspective, this is an **operator learning problem**: approximating a continuous, non-linear mapping $\mathcal{G}: \mathcal{A} \times \mathcal{S} \to \mathcal{U}$ between infinite-dimensional function spaces (ground acceleration input to multi-story continuous trajectory outputs) across varying graph topologies and physical stiffness parameters.

---

### 2. Why is it Computationally Interesting for CSE / AI?
1. **Irregular Non-Grid Topologies:** Civil structures vary in story count, floor masses, and stiffness distributions. Standard grid-based operators (such as standard Fourier Neural Operators, FNO) require uniform discretization, failing when structural topology changes.
2. **Inductive Biases & Continuous Operator Approximation:** Seismic waves propagate continuously in time. Discrete auto-regressive networks (e.g., LSTMs) accumulate severe temporal drift and lack mesh-invariance. Neural operators provide resolution-invariant continuous operator approximations.
3. **Severe Distribution Shift (OOD):** Structures encounter seismic ground motions with unforeseen spectral content (Earthquake OOD), and models must generalize to buildings with stiffness and fundamental vibration periods outside the training envelope (Structural OOD).
4. **Disentangling Envelope Accuracy from Waveform Phase Tracking:** Unlike standard regression benchmarks, oscillatory dynamical systems exhibit orthogonal error modes: amplitude/envelope scaling versus cumulative temporal phase drift.

---

### 3. What were the Research Hypotheses?
- **Hypothesis 1 (Topology Invariance via Graphs, EXP5):** Replacing regular-grid spatial discretization with a topology-native Graph Neural Operator (GNO)—where nodes represent physical stories and edges represent interstory columns—eliminates boundary discontinuities caused by zero-padding variable-floor buildings, resolving 3-story response degradation.
- **Hypothesis 2 (Physics-Informed Modal Conditioning, EXP6):** Conditioning spatiotemporal operator blocks on structural eigenvalue invariants ($T_1, \omega_1$) via Feature-wise Linear Modulation (FiLM) enables the operator to dynamically rescale spectral convolutions, significantly improving peak displacement estimation under structural modal-period extrapolation ($T_1 > T_{1,\text{train}}$).

---

### 4. What were the Baselines & What Failed?

#### Baseline 1: EXP4 — 2D Fourier Neural Operator (Fixed Grid)
- **Architecture:** Fixed $5 \times 2048$ tensor (5 spatial stories $\times$ 2,048 time steps). 3-story buildings were zero-padded at stories 4 and 5.
- **The Failure:** While 5-story structures achieved a median Relative $L_2$ error of **19.29%**, 3-story structures suffered catastrophic failure: **99.60% median Relative $L_2$ error**.
- **Root Cause:** The global spatial Fourier kernel attempted to enforce spatial periodicity across the artificial zero-boundary, causing high-frequency Gibbs-like ringing and severe non-physical boundary artifacts.

#### Baseline 2: EXP5 — Spatiotemporal Graph Neural Operator (Unconditioned)
- **Architecture:** Topology-native graph $\mathcal{G}=(V, E)$ with $|V| = N_{\text{stories}}$ (0 zero-padding), combined with 1D temporal Fourier spectral convolutions.
- **Result:** Successfully eliminated the 3-story boundary failure, dropping 3-story error from **99.60% to 22.09%** (a 77.51 percentage-point reduction).
- **The Failure:** Under structural distribution shift (unseen flexible archetype `5S_T120`, fundamental period $T_1 = 1.20\text{ s}$ vs. training $T_1 \le 0.85\text{ s}$), EXP5 incurred a **125.39% Relative $L_2$ error**.
- **Root Cause:** FFT forensic analysis confirmed the model accurately predicted peak displacement amplitude (**15.11% peak error**), but defaulted its oscillation frequency to the training distribution limit ($\sim 1.12\text{ Hz}$ vs. true $0.83\text{ Hz}$), causing destructive phase cancellation over $20.48\text{ s}$.

---

### 5. What Algorithm was Introduced in EXP6?
We designed the **Physics/Modal-Conditioned Spatiotemporal Graph Neural Operator** (`ConditionedSpatiotemporalGNO`):
1. **Pre-Earthquake Modal Invariants:** From theoretical undamped stiffness and mass matrices ($K \phi_i = \omega_i^2 M \phi_i$), we extract natural periods $T_i = 2\pi / \omega_i$ and circular frequencies $\omega_i$. Zero ground-motion or response trajectory features are leaked.
2. **Dual-Branch FiLM Modulation:** Structural invariants $c \in \mathbb{R}^{d_{\text{cond}}}$ are mapped via an MLP generator to scale ($\gamma$) and shift ($\beta$) vectors:
   $$[\gamma_l, \beta_l] = \text{MLP}_l(c), \quad h_{\text{mod}} = (1 + \gamma_l) \odot h + \beta_l$$
   Modulation is applied to both the spatial graph message-passing representations and the temporal 1D Fourier spectral channels across 4 spatiotemporal blocks.

---

### 6. What Experiments were Performed?
Evaluated across **2,160 physical OpenSeesPy simulations** strictly partitioned by structural group and earthquake event:
- **In-Distribution (ID, 120 sims):** Known structures ($T_1 \in [0.35, 0.85]\text{ s}$), training earthquakes (RSN0001–08).
- **OOD-A (300 sims):** Known structures, held-out earthquakes (RSN0011–12).
- **OOD-B (240 sims):** Unseen flexible structural archetype `5S_T120` ($T_1 = 1.20\text{ s}$), training earthquakes.
- **OOD-C (60 sims):** Unseen structure `5S_T120` combined with held-out earthquakes RSN0011–12.
- **Progressive OOD (120 sims):** Progressive stiffness degradation benchmarks (`5S_T105`, $T_1=1.05\text{ s}$; `5S_T140`, $T_1=1.40\text{ s}$).
- **Ablation D (Shuffled Conditioning):** Randomized permutation of conditioning vectors across batch instances to falsify whether performance gains arise from physical correspondence or auxiliary parameter capacity.

---

### 7. What Did the Results Show?

#### Master Empirical Findings:
1. **Peak Displacement Envelope Generalization (OOD-B, $T_1 = 1.20\text{ s}$):**
   - **Baseline GNO:** Median Peak Error = **35.21%**
   - **$T_1$-Conditioned GNO:** Median Peak Error = **13.47%**
   - **Multi-Modal GNO:** Median Peak Error = **13.06%**
   - **Empirical Gain:** **62.9% relative reduction in peak displacement error** on unseen flexible structures.
2. **Falsification of Capacity Artifacts (Ablation D):**
   When conditioning vectors were shuffled randomly across the batch, OOD-B peak error regressed to **24.33%** and OOD-C error regressed to **36.16%**. This provides evidence that performance gains depend on physically meaningful modal correspondence rather than merely additional conditioning capacity.
3. **Inference Latency on Apple Silicon GPU (`mps`):**
   - OpenSeesPy 5-story NLTHA: **54.68 ms** per simulation.
   - EXP6 $T_1$-GNO ($B=1$): **21.45 ms** per simulation (**2.55x speedup**).
   - EXP6 $T_1$-GNO ($B=32$): **30.19 ms** batch latency (**1.81x speedup**).

---

### 8. What Remains Unsolved? (Scientific Honesty)
**Cumulative Waveform Phase Extrapolation Remains Unresolved:**
While modal conditioning successfully informs the model of the structural stiffness scale—yielding accurate peak displacement envelopes (13.06% error)—the trajectory-level Relative $L_2$ error on OOD-B remains elevated (**124.07%** for Multi-Modal GNO, **157.64%** for $T_1$-GNO), and waveform Pearson correlation remains low ($r \approx 0.05\text{--}0.09$).

**Mechanistic Cause:** Global 1D Fourier layers compute static frequency convolutions over the entire time horizon ($T = 20.48\text{ s}$, 1,024 steps). When the fundamental natural period extrapolates substantially outside the training support ($T_1 > 0.85\text{ s}$), slight frequency discrepancies accumulate linearly over time, resulting in phase opposition where $\|u_{\text{pred}} - u_{\text{true}}\|_2 / \|u_{\text{true}}\|_2 \approx \sqrt{2} \approx 141\%$. Modal conditioning does not dynamically warp the temporal Fourier basis functions.

---

### 9. Relevance to CSE / AI Research
- **Operator Learning on Dynamic Graphs:** Demonstrates how to couple graph message-passing with spectral convolutions for physical systems that exhibit dual spatial-discrete and temporal-continuous characteristics.
- **Out-of-Distribution Physics Generalization:** Provides an empirical case study showing that physics conditioning can repair amplitude/envelope generalization while leaving global spectral phase drift uncorrected.
- **Scientific Software Engineering & Forensic Auditing:** Full repository integrity enforced through 305 passing unit tests, zero-leakage hash verification, frozen baseline artifacts, and automated forensic auditing.

---

### 10. Logical Next Research Direction
To eliminate the long-horizon phase drift inherent to global Fourier spectral layers, future work should investigate **Physics-Guided Structured State-Space Models (SSMs / S4 / Mamba)** or **Continuous-Time Neural ODEs** coupled with graph representations. A state-space formulation processes time causally and recursively, allowing instantaneous period changes and hysteretic degradation to be integrated step-by-step rather than relying on a static global Fourier basis.
