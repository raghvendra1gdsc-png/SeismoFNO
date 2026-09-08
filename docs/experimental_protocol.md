# Master Experimental Protocol & Benchmarking Guidelines

This document establishes the scientific execution protocol for all experiments in the SeismoFNO research program. To ensure reproducible, peer-reviewed evaluation, all models, baselines, and numerical solvers must adhere to these exact constraints.

---

## 1. Ground Truth & Data Partitioning Protocol

### 1.1 Earthquake Ground Motion Database
- **Source:** Curated subset of the **PEER NGA-West2 Database** (140 unique physical earthquake events, including Imperial Valley, Northridge, Kobe, Chi-Chi, Loma Prieta, Landers).
- **Sampling & Duration:** $100\text{ Hz}$ sampling rate ($\Delta t = 0.01\text{ s}$), fixed duration of $N = 2,048$ time steps ($20.48\text{ s}$).
- **Preprocessing:** Linear baseline detrending followed by a 4th-order causal Butterworth bandpass filter ($0.1\text{ Hz} \le f \le 25.0\text{ Hz}$).
- **Intensity Scaling:** 10 Peak Ground Acceleration (PGA) scale factors per earthquake:
  $$\text{PGA} \in \{0.05g, 0.10g, 0.20g, 0.30g, 0.40g, 0.50g, 0.60g, 0.80g, 1.00g, 1.20g\}$$
  yielding $1,400$ distinct ground motion time histories.

---

### 1.2 Structural Parameter Space
- **SDOF Bilinear Systems:**
  - Natural Periods $T_n \in [0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]\text{ s}$
  - Viscous Damping Ratio $\zeta = 0.05$ ($5\%$ critical damping)
  - Yield Displacements $u_y \in [0.005, 0.010, 0.020, 0.030]\text{ m}$ ($5\text{ mm} \dots 30\text{ mm}$)
  - Post-Yield Hardening Ratio $\alpha = 0.05$
  - Mass $m = 1.0\text{ kg}$ (nominal normalized mass, $k_0 = m(2\pi/T_n)^2$)
- **Simulation Total:** $8,400$ high-fidelity Nonlinear Time-History Analyses (NLTHA) generated via OpenSeesPy C++ and cross-verified against the independent NumPy Newmark-$\beta$ solver.

---

### 1.3 Strict Zero-Leakage Split Strategies
1. **Held-Out Earthquake Split (`held_out_earthquake_split.json`):**
   - **Protocol:** Random partition by root PEER Record Sequence Number (`base_record_id`).
   - **Training Set (70%):** 98 earthquake events (5,740 simulations).
   - **Validation Set (15%):** 21 earthquake events (1,120 simulations).
   - **Test Set (15%):** 21 earthquake events (1,540 simulations).
   - **Constraint:** $\text{Set}(\text{train\_rsn}) \cap \text{Set}(\text{test\_rsn}) = \emptyset$. All PGA scalings of a test earthquake are strictly quarantined in the test set.
2. **Held-Out Structure Split (`held_out_structure_split.json`):**
   - **Protocol:** Partition by natural period $T_n$. Training on $T_n \in \{0.2, 0.5, 1.0, 1.5, 2.0\}\text{ s}$; testing zero-shot on $T_n \in \{0.3, 0.75, 1.25, 1.75\}\text{ s}$.

---

## 2. Model Input & Target Normalization Protocol

### 2.1 Input Tensor Layout
All surrogate models receive an identical 10-channel input tensor $\mathbf{X} \in \mathbb{R}^{B \times 10 \times 2048}$:
- Channel 0: Ground acceleration $a_g(t)$ $[m/s^2]$
- Channel 1: Natural period $T_n$ $[s]$ (constant along time)
- Channel 2: Damping ratio $\zeta$ (constant along time)
- Channel 3: Yield displacement $u_y$ $[m]$ (constant along time)
- Channel 4: Hardening ratio $\alpha$ (constant along time)
- Channel 5: Mass $m$ $[kg]$ (constant along time)
- Channel 6: Initial stiffness $k_0$ $[N/m]$ (constant along time)
- Channel 7: Yield strength $F_y = k_0 u_y$ $[N]$ (constant along time)
- Channel 8: Target PGA $[g]$ (constant along time)
- Channel 9: Normalized time coordinate $\tau = t / T_{\max} \in [0, 1]$

### 2.2 Normalization Protocol
- **Unit Gaussian Normalizer:** $\tilde{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$
- **Strict Rule:** Mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ are fitted **strictly on the 5,740 training samples**. The validation and test sets are transformed using frozen training statistics (zero test-leakage).

---

## 3. Training & Optimization Protocol

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Weight decay $= 1.0 \times 10^{-5}$ |
| **Initial Learning Rate** | $0.005$ | Base learning rate for batch size 32 |
| **LR Scheduler** | CosineAnnealingLR | $T_{\max} = 50$ epochs, $\eta_{\min} = 1.0 \times 10^{-5}$ |
| **Batch Size** | 32 | Mini-batch size across all deep models |
| **Max Epochs** | 50 | With early stopping patience = 12 epochs on validation loss |
| **Gradient Clipping** | $1.0$ | Max gradient $L_2$ norm to prevent exploding gradients |
| **Random Seed** | 42 | Deterministic seed across PyTorch, NumPy, Python random |
| **Hardware** | Fixed Hardware | Single workstation CPU/GPU to ensure apples-to-apples timing |

---

## 4. Disaggregation Requirements

Aggregate metrics (overall Mean Relative $L_2$) conceal critical localized failures. Every experiment must report results disaggregated across:

1. **Ductility Bins ($\mu = u_{\max} / u_y$):**
   - **Elastic Regime:** $\mu \le 1.0$
   - **Low Inelasticity:** $1.0 < \mu \le 2.0$
   - **Moderate Inelasticity:** $2.0 < \mu \le 4.0$
   - **Severe Inelasticity:** $\mu > 4.0$
2. **Earthquake Event Breakdown:**
   - Per-earthquake distributions (median, 16th, and 84th percentiles) across the 21 held-out test events to measure record-to-record dispersion ($\sigma_{\ln \text{error}}$).
3. **Temporal Progression:**
   - Pre-yield ($t < t_{\text{yield}}$) vs. Post-yield ($t \ge t_{\text{yield}}$) segment errors.
