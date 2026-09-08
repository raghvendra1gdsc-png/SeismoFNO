# EXP 3-R AUTHORITATIVE TRAINING & EVALUATION PROTOCOL

**Document:** `docs/EXP3R_TRAINING_PROTOCOL.md`  
**Status:** FROZEN BEFORE TRAINING  
**Date:** September 2, 2026  
**Auditing Panel:** Hostile PhD-Level Research Committee  
**Operating Constraint:** Zero training executed. No access to test-set performance. Strict scientific pre-registration.

---

## 1. Absolute Scientific Immutability & Separation

1. **Prior Experiment Immutability:**
   - EXP 1 (FNO vs Baseline Operators), EXP 2 (State Memory & Continuous S4), and EXP 3 (Physics-Guided State Supervision & Forensic Audit) are **STRICTLY FROZEN**.
   - No code, datasets, splits, checkpoints, metrics, figures, or reports in `results/experiments/exp1_*`, `results/experiments/exp2_*`, or `results/experiments/exp3_*` shall be modified, overwritten, re-trained, or re-interpreted.
2. **Independent Experiment Identity:**
   - EXP 3-R is an entirely separate experiment housed strictly within `results/experiments/exp3r_ssm/`.
   - The primary objective of EXP 3-R is to test the causal state hypothesis using a **pure recurrent state-space model with direct physical-state coordinate pinning and zero decoder bypass**.

---

## 2. Frozen Dataset Partition & Zero-Leakage Guarantee

The experiment uses the authoritative zero-leakage partition from `data/processed/splits/held_out_earthquake_split.json`:

```
+---------------------------------------------------------------------------------------------------+
|                                  FROZEN DATASET PARTITION BREAKDOWN                               |
+--------------------------+---------------------+-------------------+------------------------------+
| Split Partition          | RSN Record Count    | Parent Event Count| Bilinear SDOF Simulations    |
+--------------------------+---------------------+-------------------+------------------------------+
| Train Partition          | 82 RSNs             | 11 Earthquakes    | 5,740 simulations            |
| Validation Partition     | 16 RSNs             | 2 Earthquakes     | 1,120 simulations            |
| Test Partition           | 22 RSNs             | 3 Earthquakes     | 1,540 simulations            |
+--------------------------+---------------------+-------------------+------------------------------+
| Test Earthquakes: ['Christchurch', 'Morgan Hill', 'Northridge-01']                                 |
+---------------------------------------------------------------------------------------------------+
```

### Strict Partition Invariants:
1. **Parent Earthquake Separation:** Test ground motions originate from 3 tectonic events (*Christchurch*, *Morgan Hill*, *Northridge-01*) that are strictly disjoint from the 11 training and 2 validation events.
2. **Zero Test Set Access:**
   - Normalization statistics (mean, std) are computed **strictly on the 5,740 training simulations**.
   - The validation set is used **exclusively for checkpoint selection and early stopping**.
   - The test set is evaluated **exactly once** after training and checkpoint selection are locked.
   - Zero hyperparameter tuning, loss weight adjustment, or model modifications are permitted based on test set results.

---

## 3. Frozen Model Architecture: PureRecurrentSSM

Implemented in `src/models/exp3r_ssm.py`:

```
+---------------------------------------------------------------------------------------------------+
|                                  EXP 3-R PURE RECURRENT STATE-SPACE MODEL                         |
+---------------------------------------------------------------------------------------------------+
| Input Features (5 Channels):                                                                      |
|       x_t = [ a_g(t), T, zeta, u_y, alpha ]^T in R^5                                              |
|       (STRICTLY 5 CHANNELS: NO PGA, NO GLOBAL SUMMARY, NO FUTURE LEAKAGE, NO HISTORY CHANNELS)   |
|                                                                                                   |
| Recurrent State Vector (Total Dimension = 64):                                                    |
|       h_t = [ s_t^phys , q_t ]^T in R^64                                                          |
|       where:                                                                                      |
|             s_t^phys = [ u_t, v_t, u_p,t ]^T in R^3  (Physically Supervised Mechanical State)      |
|             q_t in R^61                             (Unconstrained Complementary Latent Memory)   |
|                                                                                                   |
| Recurrent Dynamic Transition:                                                                     |
|       h_{t+1} = h_t + F_theta( h_t, x_t )                                                         |
|       where F_theta is a 4-layer ResNet MLP: Linear(69 -> 601) -> LayerNorm -> GELU ->            |
|             3x [Linear(601 -> 601) -> LayerNorm -> GELU] -> Linear(601 -> 64)                    |
|                                                                                                   |
| Output Pointwise Readout:                                                                         |
|       y_t = G_theta( h_t ) = [ u(t), F_R(t), E_diss(t) ]^T in R^3                                 |
|       where G_theta is a 2-layer MLP: Linear(64 -> 300) -> LayerNorm -> GELU -> Linear(300 -> 3) |
|                                                                                                   |
| Strict Structural Constraints:                                                                    |
|       - ZERO direct path x_t -> y_t                                                               |
|       - ZERO encoder-decoder bypass                                                               |
|       - ZERO future feature injection                                                             |
+---------------------------------------------------------------------------------------------------+
```

### Parameter Budget Lock:
- **Trainable Parameters:** **1,191,815**
- **Target Budget:** **1,192,448**
- **Budget Deviation:** **$-0.053\%$** (well within the $\pm 1.0\%$ tolerance window of $1,180,524$ to $1,204,372$).

---

## 4. Precommitted Random Seeds

Training will execute across exactly **5 precommitted random seeds**:
```json
{
  "training_seeds": [42, 123, 456, 789, 1024]
}
```
For each seed:
- PyTorch CPU/GPU seed, NumPy seed, and Python random seed are set identically.
- Model parameters are initialized from scratch using PyTorch default Kaiming initialization.
- DataLoaders are shuffled deterministically using a seeded PyTorch generator.
- Checkpoints, training logs, and evaluations are saved into seed-specific subdirectories (`seed_42/`, `seed_123/`, etc.).

---

## 5. Locked Optimizer & Hyperparameters

```json
{
  "optimizer": "AdamW",
  "learning_rate": 1e-3,
  "weight_decay": 1e-4,
  "gradient_clip_max_norm": 1.0,
  "scheduler": "CosineAnnealingLR",
  "t_max_epochs": 50,
  "eta_min": 1e-6,
  "batch_size": 32,
  "max_epochs": 50,
  "early_stopping_patience": 12,
  "num_workers": 4
}
```
**Strict Prohibition:** No post-hoc learning rate sweeps, batch size sweeps, or optimizer modifications are permitted.

---

## 6. Locked Multi-Task Loss Function & Normalization

### Normalization Pipeline (Train-Only Statistics):
All inputs $x$, outputs $y$, and supervised states $s^{\text{phys}}$ are normalized using `UnitGaussianNormalizer` fitted strictly on `train_df`:
$$\tilde{x} = \frac{x - \mu_x^{\text{train}}}{\sigma_x^{\text{train}} + \epsilon}, \quad \tilde{y} = \frac{y - \mu_y^{\text{train}}}{\sigma_y^{\text{train}} + \epsilon}, \quad \tilde{s}^{\text{phys}} = \frac{s^{\text{phys}} - \mu_s^{\text{train}}}{\sigma_s^{\text{train}} + \epsilon}$$

### Total Composite Loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{response}} + \lambda_{\text{state}} \mathcal{L}_{\text{state}}$$

#### 1. Response Loss ($\mathcal{L}_{\text{response}}$):
$$\mathcal{L}_{\text{response}} = w_u \cdot \text{MSE}(\tilde{u}, \hat{\tilde{u}}) + w_F \cdot \text{MSE}(\tilde{F}_R, \hat{\tilde{F}}_R) + w_E \cdot \text{MSE}(\tilde{E}_{\text{diss}}, \hat{\tilde{E}}_{\text{diss}})$$
- $w_u = 1.0$ (Displacement primary tracking)
- $w_F = 0.5$ (Restoring force constitutive constraint)
- $w_E = 0.5$ (Cumulative hysteretic dissipation constraint)

#### 2. Physical State Auxiliary Loss ($\mathcal{L}_{\text{state}}$):
Supervises the first 3 coordinates of $h_t$ against ground-truth $[u(t), v(t), u_p(t)]$:
$$\mathcal{L}_{\text{state}} = w_{s,u} \cdot \text{MSE}(\tilde{u}, \hat{\tilde{h}}_0) + w_{s,v} \cdot \text{MSE}(\tilde{v}, \hat{\tilde{h}}_1) + w_{s,u_p} \cdot \text{MSE}(\tilde{u}_p, \hat{\tilde{h}}_2)$$
- $w_{s,u} = 1.0$
- $w_{s,v} = 0.5$
- $w_{s,u_p} = 1.0$
- Auxiliary state loss weight: $\lambda_{\text{state}} = 0.20$

**Non-Redefinition Clause:** The physical state loss pins coordinate index 2 to $u_p(t)$. The causal intervention is strictly restricted to coordinate index 2.

---

## 7. Checkpoint Selection Protocol

- **Selection Metric:** **Validation Response Loss** ($\text{Val } \mathcal{L}_{\text{response}}$).
  $$\text{Checkpoint Metric} = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \mathcal{L}_{\text{response}}(y_i^{\text{val}}, \hat{y}_i^{\text{val}})$$
- **Condition:** Evaluated at the end of each epoch on the 1,120 validation simulations.
- **Decision Rule:** The checkpoint that minimizes $\text{Val } \mathcal{L}_{\text{response}}$ across the 50 epochs is designated `best_model.pt`.
- **Absolute Rule:** The test set is **never evaluated** during checkpoint selection.

---

## 8. Causal Intervention & Control Protocol

### Dynamic Yield Onset:
$$t_y = \min \{ t \mid |u(t)| \ge u_y \}$$

### Intervention Operator:
$$\mathbf{P}_{\text{phys}} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ \mathbf{0}_{61} \end{bmatrix} \in \mathbb{R}^{64} \implies \Delta h = \mathbf{P}_{\text{phys}} \Delta u_p = \begin{bmatrix} 0 \\ 0 \\ \Delta u_p \\ \mathbf{0}_{61} \end{bmatrix}$$
- Doses: $\Delta u_p \in \{-10.0, -5.0, -2.5, 0.0, +2.5, +5.0, +10.0\}\text{ mm}$.

### Mandatory Control Suite:
1. **C0 (Zero Intervention):** $\Delta u_p = 0.0\text{ mm}$. Must reproduce baseline trajectory to machine precision ($< 10^{-6}\text{ mm}$).
2. **C1 (Physical Plastic Intervention):** Primary causal intervention $\Delta h = [0, 0, \Delta u_p, \mathbf{0}_{61}]^T$.
3. **C2.A (Velocity Sham):** Perturbs velocity coordinate with matched kinetic energy:
   $$\Delta v = \sqrt{\frac{(1 - \alpha) k_0}{m}} |\Delta u_p| \cdot \text{sign}(\Delta u_p), \quad \Delta h_{\text{vel}} = [0, \Delta v, 0, \mathbf{0}_{61}]^T$$
4. **C2.B (Latent Memory Sham):** Perturbs complementary latent memory $q$ with matched Euclidean norm:
   $$\|\Delta \mathbf{q}\|_2 = |\Delta u_p|, \quad \Delta h_{\text{latent}} = [0, 0, 0, \Delta u_p, 0, \dots, 0]^T$$
5. **C3 (Sign Reversal):** Injects $-\Delta u_p$ to evaluate directional symmetry.
6. **C4 (Post-Shaking Intervention):** Injected at $t_{\text{post}}$ after shaking ceases ($a_g \approx 0$). Reference static equilibrium condition.
7. **C5 (Initial-Condition Diagnostic):** Injected at $t=0$ as an initial-state diagnostic.
8. **C6 (Elastic Constitutive Control):** Evaluated on linear-elastic records ($\mu \le 1.0$). Unperturbed execution must satisfy $\hat{u}_p(t) \approx 0$ and $E_{\text{diss}}(t) \equiv 0.000\text{ J}$.
9. **Static-Hold Diagnostic:** Evaluates dynamic divergence $D_h(t) = \|h_t^{\text{active}} - h_t^{\text{hold}}\|_2$ and $D_y(t) = \|y_t^{\text{active}} - y_t^{\text{hold}}\|_2$.

---

## 9. Pre-Registered Dose-Response Criteria: Regime Separation

### Regime A: Monotonic Post-Yield Decay (No Secondary Yielding)
- Condition: The structure yields at $t_y$, after which excitation stays within the post-yield elastic unloading limits ($|F_R(t) - \alpha_b(t)| < F_y$).
- **Theoretically Verified Slope:** $m_{\text{expected}} = 1 - \alpha = 1 - 0.02 = \mathbf{+0.98000}$.
- **Pre-Registered Acceptance Interval:**
  $$m_{\text{Regime A}} \in [0.900, 1.050]$$

### Regime B: Multi-Pulse Cyclic / Reverse Yielding
- Condition: Subsequent seismic pulses drive the oscillator into reverse plastic yielding ($\Delta u_{\text{reverse}} < -2 u_y$).
- **No Universal Linear Constraint:** Multi-pulse ratcheting and shakedown introduce accelerogram-dependent variance.
- **Reporting Requirement:** Report the empirical distribution of $m_i$, median, interquartile range (IQR), clustered bootstrap 95% CI, and percentage of positive slopes ($\Pr(m_i > 0)$).
- **Hypothesis:** $\mathbb{E}[m_{\text{Regime B}}] > 0$.

---

## 10. Specificity Acceptance Standard

$$\text{Specificity Ratio } \mathcal{S} = \frac{|\Delta u_{\text{res}}(\text{C2})|}{|\Delta u_{\text{res}}(\text{C1})|} \le 0.20 \quad (\ge 80\%\text{ specificity over sham controls})$$

---

## 11. Statistical Analysis & Cluster Bootstrap

- **Clustered Bootstrap:** Resampling clustered by parent earthquake ($K=3$, $B=2,000$ bootstrap replicates).
- **Primary Reporting:**
  1. Per-earthquake causal slope, specificity, and relative error reported separately for *Christchurch*, *Morgan Hill*, and *Northridge-01*.
  2. Pooled descriptive effect with 95% bootstrap confidence interval.
  3. Cochran's $Q$ and $I^2$ reported strictly as secondary descriptive heterogeneity statistics.
  4. Explicit disclaimer that $K=3$ limits population-level generalization.

---

## 12. Primary Scientific Endpoints

```
+---------------------------------------------------------------------------------------------------+
|                                  PRIMARY VS. SECONDARY ENDPOINTS                                  |
+--------------------------+------------------------------------+-----------------------------------+
| Category                 | Endpoint Name                      | Acceptance Standard               |
+--------------------------+------------------------------------+-----------------------------------+
| Primary Response         | Relative L2 Displacement Error     | Outperforms EXP 3 Supervised      |
|                          |                                    | PG-TCN (172.37%)                  |
| Primary Causal           | Regime A Asymptotic Slope          | m in [0.90, 1.05]                 |
| Primary Specificity      | C2 Latent & Velocity Sham Ratio    | S <= 0.20 (>= 80% specificity)    |
| Primary State            | u_p Tracking Error                 | R^2 >= 0.85 against OpenSees      |
| Primary Dynamic          | Dynamic Divergence D_y(t)          | Non-zero, excitation correlated   |
+--------------------------+------------------------------------+-----------------------------------+
| Secondary Metrics        | Peak u error (%), Force error (%), E_diss error (J), Residual drift  |
|                          | (mm), Regime B slope distribution, Elastic unperturbed zero-check.|
+---------------------------------------------------------------------------------------------------+
```

---

## 13. Stop Conditions

### Stop Before Test Evaluation:
1. Discovery of train/test data leakage.
2. Parameter count deviating $> \pm 1.0\%$ from $1,192,448$.
3. Autograd dependency check detecting direct $x_t \to y_t$ feedforward connection.
4. Non-zero past perturbation ($\max_{t < t_y} |\Delta y(t)| > 0.000\text{ mm}$).
5. Normalization pipeline accessing validation or test tensors.

### Stop During Training:
1. Encountering `NaN` or `Inf` in losses, activations, or gradients.
2. Gradient clipping saturation ($> 90\%$ of batches exceeding $\text{norm} = 1.0$ after epoch 5).
3. Non-finite validation loss.

**Protocol is FROZEN. No modifications permitted during execution.**
