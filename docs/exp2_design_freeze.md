# EXP 2 DESIGN FREEZE SPECIFICATION (Gate 1 Certification)

**Document Identifier:** DOCS-EXP2-FREEZE-V1  
**Authority:** Master Scientific Design Freeze (Reconciled from GStack Gate 0 Audit, Hostile Review, Plan-Eng-Review, and Plan-Design-Review)  
**Governing Standard:** Top-Tier Computational Mechanics (*CMAME*) & Structural Dynamics (*EESD*) Publication Criteria  
**Date of Freeze:** 2026-08-30  
**Implementation Status:** FROZEN — Ready for Source Implementation  

---

## 1. Primary Scientific Question & Falsifiable Hypotheses

### 1.1 The Central Research Question
> **"Why do causal feedforward neural operators suffer severe trajectory drift under non-stationary elastoplastic yielding, and is explicit recursive internal state representation both necessary and sufficient to restore hysteretic accuracy while preserving continuous temporal resolution invariance?"**

### 1.2 Formal Popperian Hypotheses
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPOTHESIS 2 (Mechanism B — The Internal State Memory Hypothesis):          │
│ In non-stationary elastoplastic dynamics, yielding produces an irreversible │
│ shifting of the elastic equilibrium origin:                                 │
│                   u_p(t) = \int_0^t \dot{u}_p(\tau) \, d\tau                │
│ Finite Impulse Response (FIR) feedforward operators (such as Causal TCN)    │
│ cannot track this unbounded memory integral from input history alone.       │
│ Augmenting the operator with an explicit recursive internal state h(t)      │
│ (either via continuous state-space S4 or a recurrent state channel) is      │
│ necessary to prevent cumulative plastic drift and reduce severe post-yield  │
│ displacement error (\mu > 4) by at least 40% absolute.                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ COUNTER-HYPOTHESIS 2_NULL (The Architectural Superiority Counter-Claim):    │
│ Any performance improvement observed in S4 is not caused by internal state  │
│ memory (Mechanism B), but rather by S4's specific architectural inductive   │
│ biases (orthogonal HiPPO polynomial basis, Cauchy kernel parameterization,  │
│ LayerNorm dynamics, or continuous spectral filtering).                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Experimental Factors & 2x2 Factorial Decomposition

To definitively decouple **Causality**, **State Memory**, and **Representation Domain**, EXP 2 evaluates a controlled matrix across four distinct experimental factors:

```
                          2x2 FACTORIAL DECOMPOSITION
┌──────────────────────────────────────┬──────────────────────────────────────┐
│                                      │ RECEPTIVE FIELD MEMORY MECHANISM     │
│ TEMPORAL CAUSALITY                   ├──────────────────┬───────────────────┤
│                                      │ State-Free (FIR) │ State-Bearing(IIR)│
├──────────────────────────────────────┼──────────────────┼───────────────────┤
│ Acausal (Bidirectional FFT / Conv)   │ Standard FNO-1D  │ [Acausal SSM]     │
├──────────────────────────────────────┼──────────────────┼───────────────────┤
│ Strictly Causal (Left-Padded / Rec)  │ Causal TCN       │ Continuous S4 SSM │
│                                      │                  │ State-Aug TCN     │
└──────────────────────────────────────┴──────────────────┴───────────────────┘
```

---

## 3. The 5-Model Controlled Matrix (Parameter-Matched ~1.19M)

To eliminate architectural confounding, EXP 2 tests **five strictly controlled models** containing within-family controls:

```
                            THE 5-MODEL BENCHMARK MATRIX
┌───┬─────────────────────────────┬─────────────┬───────────┬──────────────┬─────────────────────────┐
│ # │ Model Name                  │ Parameters  │ Causality │ State Memory │ Scientific Function     │
├───┼─────────────────────────────┼─────────────┼───────────┼──────────────┼─────────────────────────┤
│ 1 │ Standard FNO-1D             │ 1,196,931   │ ❌ Acausal│ ❌ None (FIR)│ Acausal Spectral Baseline│
│ 2 │ Causal TCN (State-Free)     │ 1,188,763   │ Strict    │ ❌ None (FIR)│ Causal Feedforward Base │
│ 3 │ State-Augmented Causal TCN  │ 1,191,203   │ Strict    │ ✅ Recurrent │ Within-TCN State Control│
│ 4 │ Continuous S4 SSM (Full)    │ 1,192,448   │ Strict    │ ✅ Continuous│ Continuous State Model  │
│ 5 │ Memory-Truncated S4 SSM     │ 1,192,448   │ Strict    │ ❌ Truncated │ Within-S4 Memory Control│
└───┴─────────────────────────────┴─────────────┴───────────┴──────────────┴─────────────────────────┘
```

### Detailed Role of Within-Family Controls:
1. **Model 3 (State-Augmented Causal TCN):** Uses the exact 11-layer dilated TCN backbone from Model 2, but prepends a 1D recurrent state cell $s(t) = \tanh(\mathbf{W}_s s(t-1) + \mathbf{W}_x x(t))$. This tests Mechanism B **strictly within the TCN architecture**.
2. **Model 5 (Memory-Truncated S4 SSM):** Uses the exact S4 backbone from Model 4, but sets the state transition matrix decay rate $\lambda \to \infty$, artificially constraining S4's effective memory horizon to $< 10$ time steps ($R \to 0$). This tests Mechanism B **strictly within the S4 architecture**.

---

## 4. Comprehensive Confound Audit & Safeguards

| Potential Confounding Variable | How It Distorts Experiments | Strict Experimental Control / Safeguard |
| :--- | :--- | :--- |
| **Parameter Count Mismatch** | Capacity differences mask inductive bias. | All 5 models are tightly parameter-matched to **$1.19\text{M} \pm 1.5\%$** ($1,188,763 - 1,196,931$ parameters). |
| **Normalization Discrepancy** | Different feature scales alter loss landscapes. | A single `UnitGaussianNormalizer` instance is fitted strictly over the **entire 5,740 training records** and shared across all models. |
| **Receptive Field Span** | Model cannot see past seismic peaks. | Receptive field of Causal TCN is $4,095$ steps ($40.95\text{ s}$), exceeding the $2,048$-step ($20.48\text{ s}$) record length. S4 has infinite impulse response ($R=\infty$). |
| **Activation Function Bias** | GELU vs. ReLU differences. | FNO uses GELU; Causal TCN uses ReLU; S4 uses GELU. Within-family controls hold activations $100\%$ identical. |
| **Optimization Budget Bias** | One model trained longer or with higher LR. | Identical AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay $10^{-4}$), Cosine Annealing scheduler ($\eta_{\min} = 10^{-6}$), 50 epochs, batch size 32, fixed seed 42. |
| **Inter-Record State Bleeding** | State persists across batch samples. | All recurrent and state-space models explicitly initialize $h(0) = 0$ for every sample in every batch. |

---

## 5. Dataset Freeze (Gate 0 Reconciled)

The experiment is strictly frozen to the verified Gate 0 bilinear dataset partition:

```
                      AUTHORITATIVE DATASET PARTITION
┌─────────────────────────────────────────────────────────────────────────────┐
│ SPLIT FILE       : data/processed/splits/held_out_earthquake_split.json     │
│ INDEX FILE       : data/simulations/simulation_index.csv                    │
│ FILTER RULE      : material_type == "bilinear"                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. TRAINING SET  : 11 Parent Earthquakes | 82 RSNs | 5,740 Simulations      │
│    Events: Chi-Chi (Taiwan-01), Darfield (New Zealand), Duzce (Turkey),     │
│            Hector Mine, Imperial Valley-06, Kobe (Hyogo-ken Nanbu),         │
│            Landers, Parkfield-02, San Fernando, Tottori (Japan),            │
│            Whittier Narrows-01.                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. VALIDATION SET: 2 Parent Earthquakes | 16 RSNs | 1,120 Simulations       │
│    Events: Kocaeli (Turkey), Loma Prieta.                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. TEST SET      : 3 Parent Earthquakes | 22 RSNs | 1,540 Simulations       │
│    Events: Christchurch, Morgan Hill, Northridge-01.                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ TOTAL NONLINEAR BENCHMARK: 16 Parent Earthquakes | 120 RSNs | 8,400 Sims    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Leakage Quarantine

1. **Input Tensor Specification:** Model input tensor $x \in \mathbb{R}^{B \times 10 \times 2048}$ is strictly limited to 10 known channels:
   $$x(t) = \left[ a_g(t), T_0, \omega_n, k_0, \zeta, \text{is\_bilinear}, u_y, \alpha, \text{PGA}, \tau_{\text{grid}} \right]$$
2. **Quarantine of History Channels:** `use_history_channel = False` is hardcoded. Zero displacement ($u$), velocity ($\dot{u}$), restoring force ($F_R$), or energy ($E_h$) enters the input pipeline.
3. **Partition Disjointness:** Verified disjointness at parent earthquake level (`Train vs Val: True`, `Train vs Test: True`, `Val vs Test: True`).

---

## 7. Mathematical Metric Definitions

Every model is evaluated across 7 core mathematical metrics:

### 7.1 Relative Trajectory $L_2$ Error
$$\text{Rel } L_2(u) = \frac{\|u(t) - \hat{u}(t)\|_2}{\|u(t)\|_2 + \epsilon} \times 100\%$$

### 7.2 Severe-Yielding Trajectory Error
$$\text{Rel } L_2(u)_{\mu > 4} = \left. \frac{\|u(t) - \hat{u}(t)\|_2}{\|u(t)\|_2 + \epsilon} \times 100\% \right|_{\mu = \frac{u_{\max}}{u_y} > 4.0}$$

### 7.3 Residual Plastic Drift Error
$$\text{Err}(u_{\text{residual}}) = |u(T) - \hat{u}(T)| \times 1000.0 \quad (\text{mm})$$

### 7.4 Hysteresis Loop Dissipation Area Error
$$\text{Err}(A_{\text{loop}}) = \frac{\left| \oint F_R \, du - \oint \hat{F}_R \, d\hat{u} \right|}{\max\left(\left|\oint F_R \, du\right|, \epsilon\right)} \times 100\%$$

### 7.5 Scale-Regularized Absorbed Hysteretic Energy Error
$$\text{Rel } L_2(E_h) = \frac{\|E_h(t) - \hat{E}_h(t)\|_2}{\max\left(\|E_h(t)\|_2, E_i, 1.0\text{ J}\right)} \times 100\%$$

### 7.6 Past-Perturbation Causality Discrepancy Metric
For future noise perturbation $\tilde{a}_g(t) = a_g(t) + \eta(t)$ injected at $t \ge t_0$ ($t_0 = 0.5 T$):
$$\text{Discr}_{\text{past}}(t_0) = \max_{t < t_0} \left| u_{\text{clean}}(t) - u_{\text{perturbed}}(t) \right| \times 1000.0 \quad (\text{mm})$$

### 7.7 Latent-State Plastic Offset Linear Probing $R^2$
For frozen latent feature representation $h(t) \in \mathbb{R}^{128}$ and true plastic offset $u_p(t) = u(t) - F_R(t)/k_0$:
$$\hat{u}_p(t) = \mathbf{w}^T h(t) + b, \quad R^2(u_p) = 1 - \frac{\sum (u_p(t) - \hat{u}_p(t))^2}{\sum (u_p(t) - \bar{u}_p)^2}$$

---

## 8. Statistical Protocol: Earthquake-Clustered Block Bootstrap

To account for intra-earthquake covariance among recording stations:
1. **Statistical Cluster Unit:** Parent Earthquake Event ($K = 3$ test clusters: `Christchurch`, `Morgan Hill`, `Northridge-01`).
2. **Bootstrap Iterations:** $B = 2,000$ resamples with replacement.
3. **Resampling Procedure:**
   - Sample 3 earthquake clusters with replacement.
   - Pool all $N_{\text{cluster}}$ simulation records belonging to the sampled clusters.
   - Compute mean metric $\bar{\theta}_b$ for iteration $b$.
4. **$95\%$ Confidence Interval:** $\text{CI}_{95} = \left[ q_{0.025}(\bar{\theta}), \, q_{0.975}(\bar{\theta}) \right]$.

---

## 9. Formal Falsification Logic & Decision Tree

```
                       EXP 2 SCIENTIFIC DECISION TREE
┌─────────────────────────────────────────────────────────────────────────────┐
│ CASE 1: STRONG CONFIRMATION OF MECHANISM B                                  │
│ Condition:                                                                  │
│   • Continuous S4 SSM achieves severe yield error (\mu > 4) < 40.0%.        │
│   • State-Augmented TCN beats State-Free TCN by > 25.0% absolute.           │
│   • Full S4 beats Memory-Truncated S4 by > 30.0% absolute.                  │
│   • Linear Probe achieves R^2(u_p) > 0.85 on S4 and State-Aug TCN.          │
│ Scientific Conclusion: Internal state memory is mathematically necessary    │
│ and sufficient to resolve plastic drift in causal neural operators.         │
├─────────────────────────────────────────────────────────────────────────────┤
│ CASE 2: ARCHITECTURAL CONFOUNDING CONFIRMED (Mechanism B Falsified)         │
│ Condition:                                                                  │
│   • S4 SSM outperforms TCN, BUT:                                            │
│   • State-Augmented TCN fails to improve over State-Free TCN (< 5% delta).  │
│   • Memory-Truncated S4 achieves identical accuracy to Full S4.             │
│   • Linear Probe yields R^2(u_p) < 0.40 across all models.                  │
│ Scientific Conclusion: S4's performance is driven by HiPPO polynomial bases │
│ or LayerNorm dynamics, NOT by internal state tracking of plastic offset.    │
├─────────────────────────────────────────────────────────────────────────────┤
│ CASE 3: LINEAR STATE INSUFFICIENCY (Profound Negative Discovery)            │
│ Condition:                                                                  │
│   • Both S4 SSM and State-Augmented TCN fail on severe yielding (> 60% err).│
│ Scientific Conclusion: Continuous linear state-space models cannot capture  │
│ non-smooth yield surface bifurcations without explicit nonlinear yield laws.│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Latent-State Linear Probing Protocol

- **Objective:** Determine whether the latent hidden state $h(t)$ physically encodes the unobserved internal plastic offset $u_p(t) = \int \dot{u}_p d\tau$.
- **Probe Architecture:** Single-layer linear ridge regression ($\hat{u}_p(t) = \mathbf{w}^T h(t) + b$, $\alpha = 1.0$).
- **Data Partitioning for Probe (Zero Leakage):**
  - Train Probe on latent representations extracted from the **5,740 training records**.
  - Evaluate Probe on latent representations extracted from the **1,540 test records**.
  - Base neural operator weights remain strictly **frozen** during probe training.

---

## 11. Publication-Grade Figure Plan

The EXP 2 report must automatically generate the following 5 publication figures:

1. `fig1_exp2_error_vs_ductility_clustered.png`: 4-model comparison across ductility regimes with clustered bootstrap $95\%$ CI shaded bands (log-scale Y-axis).
2. `fig2_within_family_state_ablation.png`: Direct 2-panel ablation showing (Left) State-Free TCN vs. State-Augmented TCN, and (Right) Full S4 vs. Memory-Truncated S4.
3. `fig3_residual_drift_boxplots.png`: Distribution of endpoint plastic offset error $|u(T) - \hat{u}(T)|$ (mm) across models.
4. `fig4_severe_hysteresis_comparison.png`: 4-way side-by-side $F_R(t) - u(t)$ hysteresis loops against OpenSees NLTHA ground truth for severe yielding records ($\mu > 6$).
5. `fig5_latent_state_probe_decodability.png`: True plastic offset $u_p(t)$ vs. linearly decoded probe prediction $\hat{u}_p(t)$, reporting $R^2$ values.

---

## 12. Reproducibility & Environment Freeze

```
                     REPRODUCIBILITY SPECIFICATIONS
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ Random Seed                          │ 42 (PyTorch, NumPy, Python RNG)      │
│ Floating-Point Precision             │ float32 (torch.float32)              │
│ Optimizer                            │ AdamW (lr=1e-3, weight_decay=1e-4)   │
│ Learning Rate Schedule               │ CosineAnnealingLR (T_max=50, eta=1e-6)│
│ Training Epochs & Batch Size         │ 50 Epochs | Batch Size 32            │
│ Checkpoint Selection Rule            │ Minimum Validation Loss (Best Epoch) │
│ Hardware Reporting                   │ Apple Silicon Metal (MPS) / CPU      │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 13. Gate 1 Certification Checklist

```
┌───┬────────────────────────────────────────────────────────┬────────┐
│ # │ Certification Item                                     │ Status │
├───┼────────────────────────────────────────────────────────┼────────┤
│ 1 │ Falsifiable hypothesis separated from S4 architecture  │  PASS  │
│ 2 │ 5-model matrix with within-family controls frozen      │  PASS  │
│ 3 │ Confounder audit & parameter matching (+-1.5%) verified│  PASS  │
│ 4 │ Dataset partition strictly frozen to 5,740 / 1,120 / 1,540 PASS │
│ 5 │ Target & history channel leakage strictly quarantined  │  PASS  │
│ 6 │ Mathematical metric equations formalized               │  PASS  │
│ 7 │ Earthquake-clustered block bootstrap protocol defined  │  PASS  │
│ 8 │ Latent state probe protocol with zero leakage frozen   │  PASS  │
│ 9 │ Publication figure suite specified                     │  PASS  │
│ 10│ Reproducibility environment parameters frozen          │  PASS  │
└───┴────────────────────────────────────────────────────────┴────────┘
```

---

## 14. Final Gate 1 Decision Verdict

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GATE 1 DESIGN SPECIFICATION: APPROVED & FROZEN                              │
│ VERDICT: PASS — PROCEED TO SOURCE CODE IMPLEMENTATION OF EXP 2             │
└─────────────────────────────────────────────────────────────────────────────┘
```
