# EXP 3 Forensic Causal Audit: Deconstruction of State Interventions & Causal Claims

**Document:** `docs/EXP3_CAUSAL_FORENSIC_AUDIT.md`  
**Author:** Research Lead (SeismoFNO Scientific Audit)  
**Date:** September 2, 2026  
**Target Artifact:** `results/experiments/exp3_physics_guided_state/EXP3_FINAL_REPORT.md`  
**Execution Environment:** Frozen PyTorch 3.14 / macOS ARM64 / 1,540 Held-Out Bilinear Test Records  
**Status:** COMPLETE (EXPLICIT CAUSAL RECOMMENDATION ISSUED)

---

## 1. Executive Summary & Explicit Recommendation

In accordance with **Hard Rule 1, 3, 4, and 6** of the SeismoFNO Charter (*scientific honesty over clean-looking results; no silent tuning; explicit logging of failures; unvarnished critique of physics claims*), this document executes a rigorous forensic audit attacking the causal interpretation of the EXP 3 counterfactual intervention experiment (Hypothesis H3B).

### Final Recommendation
$$\mathbf{H3B\ REJECTED}$$

*(The causal interpretation of the latent physical state in PG-TCN is **REJECTED**. While the architecture satisfies mathematical input-output causality and exhibits high $R^2$ linearity, this linearity is an artifact of direct-decoder additive feedthrough. The intermediate latent bottleneck does **not** function as a physical causal state in a dynamical system.)*

### Summary of Hypotheses Status Post-Audit

| Hypothesis | Investigated Mechanism | Pre-Registered Standard | Empirical Audit Finding | Final Scientific Status |
| :--- | :--- | :--- | :--- | :---: |
| **H3A** | Physics-Supervised Plastic State | $\ge 40\%$ Residual Drift Error Reduction vs EXP 2 | Residual drift error increased by **+5.3%** ($58.30\text{ mm} \to 61.40\text{ mm}$); Rel $L_2(u)$ worsened from $140.63\%$ to $172.37\%$ | **REJECTED (COMPLETED FAILURE)** |
| **H3B** | Counterfactual Causal State Intervention | Past Invariance ($< 10^{-5}$ mm), Monotonicity, Specificity, & Causal Slope | Past invariance is an **architectural tautology** of left-padding; slope is **attenuated by $150\times$ and inverted in sign** ($m = -0.006$ vs $+0.98$); sham controls match $86\%$ of effect | **REJECTED (CONFIRMATION RESCINDED)** |
| **H3C** | Minimal State Sufficiency (2D vs 64D) | 2D Model within $\le 5\%$ Error of 64D Model | 2D model error is **$+63.41\%$ worse** than 64D unconstrained model ($172.37\%$ vs $105.48\%$); 64D model has lower residual drift ($52.83\text{ mm}$) | **REJECTED (COMPLETED FAILURE)** |

---

## 2. Granular Evaluation of Causal Audit Criteria

To evaluate whether the counterfactual intervention experiment in PG-TCN demonstrates genuine physical causal agency or merely numerical artifacting, we subjected the model and telemetry to six adversarial attack vectors:

| Audit Criterion | Adversarial Audit Probe | Quantitative Evidence | Audit Verdict |
| :--- | :--- | :--- | :---: |
| **Criterion 1: Past Invariance** | Is past invariance ($0.000000\text{ mm}$) an empirical validation of physics, or a structural tautology of left-padding? | In `pg_tcn.py`, left-padding enforces $y[t] = f(x[:t])$ algebraically. Any perturbation restricted to $t \ge t_y$ has a receptive field disjoint from $t < t_y$. | **FAIL** *(Architectural Tautology)* |
| **Criterion 2: Dose-Response Slope** | Does the dose-response slope match the constitutive mechanics of bilinear kinematic hardening? | Expected: $m \approx +(1 - \alpha) = \mathbf{+0.98}$. Measured: $\text{Mean } m = \mathbf{-0.00625}$ ($95\%\text{ CI: } [-0.01150, -0.00101]$). Attenuated by $156\times$ and **inverted in sign** ($62.4\%$ negative). | **FAIL** *(Physical Incoherence)* |
| **Criterion 3: Sham & Specificity Controls** | Does an unphysical/orthogonal perturbation fail to reproduce the downstream drift shift? | Physical shift: $\mathbf{0.3871\text{ mm}}$. Sham 1 (orthogonal $+\Delta u_p, -\Delta \alpha_b$): $\mathbf{0.3332\text{ mm}}$ ($86.1\%$). Sham 2 (pure back-stress): $\mathbf{0.3224\text{ mm}}$ ($83.3\%$). | **FAIL** *(Lack of State Specificity)* |
| **Criterion 4: Constitutive Validity of $t_y$** | Does the model refuse or damp plastic interventions on linear-elastic records that never yielded? | $N=23$ elastic records ($\mu \le 1.0$) were perturbed at arbitrary step $1024$. The model shifted drift with slope $\mathbf{-0.01407}$, which is **$2.8\times$ larger** than on true yielding records ($-0.00491$). | **FAIL** *(Constitutive Blindness)* |
| **Criterion 5: Decoder Confounding** | Is $s_{\text{active}}$ an internal dynamical state or a direct additive bias into the decoder stack? | $s_{\text{active}}$ is concatenated directly with encoder output $z$ before 5 decoder dilated convolutional blocks. It acts as an additive linear feedthrough channel. | **FAIL** *(Direct Feedthrough Confounding)* |
| **Criterion 6: Dynamic Waveform Modification** | Does the intervention induce altered yielding dynamics, period elongation, and hysteresis changes? | Perturbation induces minor high-frequency ripple ($\text{std} = 0.3759\text{ mm}$) around an attenuated DC shift, but produces zero subsequent yielding changes or altered hysteretic dissipation. | **FAIL** *(Superficial Waveform Perturbation)* |

---

## 3. Deep-Dive Forensic Analyses

### 3.1 Criterion 1: Deconstructing Past Invariance as an Architectural Tautology
In the EXP 3 Final Report, the observation that:
$$\max_{t < t_y} |\hat{u}^{\mathrm{cf}}(t) - \hat{u}^{\mathrm{base}}(t)| = 0.000000\mathrm{\ mm}$$
was cited as evidence confirming Hypothesis H3B.

**The Forensic Attack:**
Reviewing `src/models/pg_tcn.py` (lines 179–185):
```python
if 0 <= t_idx < s_active.shape[-1]:
    s_active[:, 0, t_idx:] += delta_u
    s_active[:, 1, t_idx:] += delta_alpha
```
And the decoder definition (lines 116–131):
```python
dec_layers.append(
    TemporalBlock(
        in_channels=curr_in,
        out_channels=decoder_dim,
        kernel_size=kernel_size,
        dilation=dilation,
        dropout=dropout,
    )
)
```
In `TemporalBlock` (`src/models/causal_tcn.py`), every convolution is preceded by:
$$\text{Chomp1d}(\text{padding}), \quad \text{padding} = (k - 1) \cdot d$$
This trims the right-hand side of the tensor so that:
$$\text{Output}[t] = \sum_{\tau=0}^{k-1} W[\tau] \cdot \text{Input}[t - \tau \cdot d]$$
Because every layer in both the encoder, state cell, and decoder uses exclusively causal left-padding, the output at any time $t < t_y$ depends strictly and exclusively on the slice $\text{Input}[0 : t]$. 

Because $\Delta s$ was added **only** to time steps $\tau \ge t_y$, the input tensors to the decoder for all $\tau \le t < t_y$ are **bit-for-bit identical** between the baseline run and the counterfactual run.

**Forensic Verdict:**
A machine precision difference of $0.000000\text{ mm}$ is an **algebraic tautology** of causal convolutional padding. It would occur for *any* arbitrary tensor added at $t \ge t_y$, whether physically meaningful, pure Gaussian noise, or random garbage. Treating this as empirical evidence that the neural operator understands physical causality is a Category Error.

---

### 3.2 Criterion 2: Dose-Response Linearity vs. Constitutive Incoherence
The EXP 3 Final Report highlighted that the causal dose-response relationship achieved a stellar mean $R^2 = 0.9983$ across doses $\Delta u_p \in \{-10, -5, 0, +5, +10\}\text{ mm}$.

**The Forensic Attack:**
1. **The Magnitude Discrepancy ($156\times$ Attenuation):**
   In a bilinear kinematic hardening SDOF oscillator (Steel01), the governing constitutive relationship is:
   $$u(t) = u_p(t) + \frac{F_R(t) - \alpha_b(t)}{k_0 (1 - \alpha)}$$
   At the end of an earthquake excitation ($t = t_{\text{end}}$), the system enters linear elastic free vibration about its displaced plastic equilibrium position. In the absence of external ground acceleration and with viscous damping decaying velocities to zero, the restoring force $F_R(t_{\text{end}}) \approx \alpha_b(t_{\text{end}}) = \alpha k_0 u_p(t_{\text{end}})$.
   Therefore, the residual displacement is:
   $$u_{\text{res}} = (1 - \alpha) u_p(t_{\text{end}}) + \frac{\alpha k_0 u_p(t_{\text{end}})}{k_0} = u_p(t_{\text{end}})$$
   If an external intervention injects an additional plastic displacement $\Delta u_p = +10.0\text{ mm}$ into the physical state at yield onset, the theoretical downstream residual displacement must shift by:
   $$\Delta u_{\text{res}}^{\text{theory}} \approx (1 - \alpha) \Delta u_p \approx (1 - 0.02) \times 10.0\text{ mm} = \mathbf{+9.80\text{ mm}}$$
   The theoretical slope is:
   $$m_{\text{theoretical}} = \frac{\Delta u_{\text{res}}}{\Delta u_p} \approx \mathbf{+0.98}$$

2. **The Measured Empirical Slope:**
   Running our forensic probe across all 157 evaluated held-out test simulations reveals:
   - **Mean Measured Slope:** $m = \mathbf{-0.00625}$
   - **95% Confidence Interval:** $[-0.01150, -0.00101]$
   - **Median Measured Slope:** $m = \mathbf{-0.00434}$
   - **Slope Sign Distribution:** **$62.4\%$ Negative**, $37.6\%$ Positive

3. **Why $R^2$ Was Misleading:**
   A linear regression of 5 collinear points $y = m x + b$ yields $R^2 \approx 1.0$ whenever the points fall on a straight line, *regardless of how flat the line is or what sign it has*.
   Because the decoder's convolutional layers apply linear operations (convolutions and residual adds) to $s_{\text{active}}$, feeding a constant step function $\Delta u_p \cdot \mathbf{1}_{t \ge t_y}$ into the decoder naturally passes through the linear convolutional filters, producing a scaled constant offset at the output. 
   
   The fact that the slope is **$-0.006$** proves two devastating facts:
   1. The model attenuates the injected physical state by **$99.4\%$** ($0.006$ vs $0.98$).
   2. The injected state pushes the residual displacement in the **opposite physical direction** ($m < 0$) in the majority of cases!

---

### 3.3 Criterion 3: Sham Controls and Lack of Physical State Specificity
If the PG-TCN architecture were genuinely utilizing $[u_p, \alpha_b]$ as coupled thermodynamic/constitutive internal state variables, the model's response should be highly sensitive to whether the perturbation respects the constitutive manifold:
$$\Delta \alpha_b = \alpha k_0 \Delta u_p$$

**The Forensic Attack:**
We evaluated four intervention conditions on the identical set of yielding earthquake simulations ($N=20$, $\mu > 4.0$, dose $\Delta u_p = 10.0\text{ mm}$):
1. **Physical Intervention:** $(\Delta u_p, +\Delta \alpha_b)$  
   $\to \text{Mean Drift Shift} = \mathbf{0.3871\text{ mm}}$
2. **Sham 1 (Orthogonal Perturbation):** $(\Delta u_p, -\Delta \alpha_b)$  
   $\to \text{Mean Drift Shift} = \mathbf{0.3332\text{ mm}}$ ($86.1\%$ of physical effect)
3. **Sham 2 (Pure Back-Stress Perturbation):** $(0, +\Delta \alpha_b)$  
   $\to \text{Mean Drift Shift} = \mathbf{0.3224\text{ mm}}$ ($83.3\%$ of physical effect)
4. **Sham 3 (Pure Plastic Displacement):** $(\Delta u_p, 0)$  
   $\to \text{Mean Drift Shift} = \mathbf{0.1104\text{ mm}}$

**Forensic Verdict:**
An unphysical perturbation that explicitly violates the constitutive kinematic hardening relationship by inverting the back-stress sign ($\text{Sham } 1$) produces **$86.1\%$** of the exact same downstream drift as the "physical" perturbation. Perturbing back-stress alone with zero plastic displacement ($\text{Sham } 2$) produces **$83.3\%$** of the drift.

The decoder has not learned a physically constrained state manifold. It treats channel 0 and channel 1 of $s_{\text{active}}$ as generic, uncoupled latent channels. Any non-zero signal injected into these channels simply trickles through the decoder weights to shift the output displacement.

---

### 3.4 Criterion 4: Constitutive Validity of Yield Onset Detection
In `src/evaluation/exp3_interventions.py`, yield onset was defined as:
$$t_y = \min \{ t \mid |u(t)| \ge u_y \}, \quad \text{fallback: } t_y = L // 2 = 1024$$

**The Forensic Attack:**
In the held-out test set, **23 of the evaluated records were strictly linear-elastic** ($\mu \le 1.0$, peak displacement $|u(t)| < u_y$). In these records, the ground-truth plastic displacement is identically zero for all time:
$$u_p(t) \equiv 0, \quad \alpha_b(t) \equiv 0, \quad \forall t \in [0, T]$$
Because $|u(t)| < u_y$, the yield onset detector fell back to $t_y = 1024$.

When a "plastic state intervention" $\Delta u_p$ was applied at $t = 1024$ to these purely elastic records:
- **Elastic Records Mean Slope:** $m = \mathbf{-0.01407}$
- **Inelastic Records Mean Slope:** $m = \mathbf{-0.00491}$

**Forensic Verdict:**
The model produces a **$2.8\times$ larger response** to a "plastic intervention" when applied to an elastic structure than when applied to a yielding structure!
In real physics, an elastic structure cannot sustain plastic deformation; injecting plastic displacement into an elastic equation of motion without yielding is constitutively impossible. The fact that the model responds even more aggressively on elastic records proves that the decoder is completely unaware of the structural yielding state, treating $s_{\text{active}}$ merely as an external input channel.

---

### 3.5 Criterion 5: Architectural / Direct-Decoder Confounding
In PG-TCN, the latent state $s_{\text{active}}$ is fed into the network via concatenation with the encoder feature map:
$$h_{\text{combined}}(t) = [z(t) \,\|\, s_{\text{active}}(t)] \in \mathbb{R}^{137}$$
It is then passed into 5 dilated causal residual blocks ($d=64, 128, 256, 512, 1024$) followed by a pointwise projection head.

**The Forensic Attack:**
Because the decoder contains residual skip connections:
$$\text{Block}(x) = x + \text{Conv}(\text{GELU}(\text{Conv}(x)))$$
and because the projection head is a linear $1 \times 1$ convolution:
$$y(t) = W_{\text{head}} h_{\text{dec}}(t) + b_{\text{head}}$$
there exists an uninhibited, direct feedforward pathway from the input channels of the decoder to the output $y(t)$.

When $s_{\text{active}}$ is subjected to a constant step perturbation $\Delta s \cdot \mathbf{1}_{t \ge t_y}$, the linear component of the decoder convolutions acts as a low-pass filter on that step function. The resulting output displacement $\hat{u}(t)$ naturally exhibits a step-like shift $\Delta \hat{u}$ for $t \ge t_y$.

This mechanism is **not** dynamical causal intervention. It is identical to adding a step voltage into an audio amplifier: the output shifts because of circuit feedthrough, not because the amplifier possesses an internal cognitive or dynamical state.

---

## 4. Audit of H3A and H3C as Completed Scientific Failures

In compliance with the directive (*"Audit H3A and H3C as completed failures and do not retune them to recover the hypotheses"*), the empirical performance of H3A and H3C is audited below:

### 4.1 Audit of H3A: Physics-Supervised Plastic State Learning
- **Pre-registered Claim:** Supervising intermediate latent representations with exact physical plastic states $[u_p(t), \alpha_b(t)]$ will reduce residual drift error by $\ge 40\%$ compared to the EXP 2 state-augmented baseline.
- **Empirical Measurement:**
  - EXP 2 Baseline Residual Drift: $\mathbf{58.30\text{ mm}}$
  - EXP 3 Supervised PG-TCN Residual Drift: $\mathbf{61.40\text{ mm}}$
  - Relative Change: **$+5.3\%$ error increase** (FAILED $\ge 40\%$ reduction).
  - Relative Displacement Error: EXP 2 was $140.63\%$, while Supervised PG-TCN was **$172.37\%$** ($+31.74\%$ absolute degradation).
- **Physical Root Cause:**
  1. **Objective Competition:** The composite loss $\mathcal{L} = \mathcal{L}_{\text{resp}} + \lambda_{\text{state}} \mathcal{L}_{\text{state}} + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}}$ forces the network to simultaneously solve an end-to-end multi-step prediction task and an intermediate 2D state regression task.
  2. **Gradient Penalty:** In `run_exp3.py`, the unsupervised ablation model ($\lambda_{\text{state}} = 0.0$) achieved a relative displacement error of **$132.90\%$**, vastly outperforming the supervised model ($172.37\%$). Forcing the intermediate features to adhere to the classical 2D constitutive equations severely crippled the representation learning of the causal convolutional backbone.
- **Verdict:** **COMPLETED SCIENTIFIC FAILURE**. H3A is definitively rejected.

### 4.2 Audit of H3C: Minimal State Sufficiency (2D Physics vs 64D Latent Space)
- **Pre-registered Claim:** A 2D physics-constrained state space is sufficient for nonlinear seismic response prediction and will perform within $5\%$ error of an unconstrained 64D state model.
- **Empirical Measurement:**
  - 64D Unconstrained Model Error: $\mathbf{105.48\%}$ Rel $L_2(u)$, $\mathbf{52.83\text{ mm}}$ Residual Drift.
  - 2D Physics Supervised Model Error: $\mathbf{172.37\%}$ Rel $L_2(u)$, $\mathbf{61.40\text{ mm}}$ Residual Drift.
  - Discrepancy: The 2D model is **$+63.41\%$ worse** than the 64D model (FAILED $\le 5\%$ tolerance).
- **Physical Root Cause:**
  Classical rate-independent plasticity theory assumes that the current state of a 1D kinematic hardening oscillator is fully described by displacement $u$, plastic displacement $u_p$, and back-stress $\alpha_b$. However, a neural operator mapping continuous ground motion $a_g(t)$ to $u(t)$ must internally solve the full second-order differential equation, track instantaneous velocity $\dot{u}(t)$, integrate hysteretic energy dissipation, and retain multi-scale memory across varying ground motion frequencies. A 64-dimensional unconstrained distributed representation provides the network with the latent capacity necessary to track these multi-scale dynamical features. Constraining the latent space to 2 dimensions creates an informational bottleneck that degrades generalization.
- **Verdict:** **COMPLETED SCIENTIFIC FAILURE**. H3C is definitively rejected.

---

## 5. Summary of Forensic Pass/Fail Scorecard

| Audit Item | Pre-Registered Expectation | Forensic Metric / Evidence | Scorecard Verdict |
| :--- | :--- | :--- | :---: |
| **Criterion 1** | Past Invariance $< 10^{-5}$ mm | Receptive field separation via left-padding makes past discrepancy mathematically $0.000000\text{ mm}$ for *any* input. | <span style="color:red; font-weight:bold;">FAIL (TAUTOLOGY)</span> |
| **Criterion 2** | Theoretical Slope $m \approx +0.98$ | Measured mean slope is $m = -0.00625$ ($156\times$ attenuation, $62.4\%$ inverted negative sign). | <span style="color:red; font-weight:bold;">FAIL</span> |
| **Criterion 3** | Sham Control Specificity | Inverting back-stress sign ($\text{Sham } 1$) produces $86.1\%$ of physical effect; pure back-stress produces $83.3\%$. | <span style="color:red; font-weight:bold;">FAIL</span> |
| **Criterion 4** | Yield Onset Validity | Intervening on linear-elastic records ($\mu \le 1.0$) yields a $2.8\times$ *larger* drift shift than on yielding records. | <span style="color:red; font-weight:bold;">FAIL</span> |
| **Criterion 5** | Architectural Independence | $s_{\text{active}}$ is concatenated directly into the decoder, establishing a trivial additive bypass. | <span style="color:red; font-weight:bold;">FAIL</span> |
| **Criterion 6** | Dynamic Modification | Trajectory changes are high-frequency ripples around an attenuated DC offset, with zero yield threshold shifts. | <span style="color:red; font-weight:bold;">FAIL</span> |
| **Hypothesis H3A** | Residual Drift Reduction $\ge 40\%$ | Error increased by $+5.3\%$ ($58.30\text{ mm} \to 61.40\text{ mm}$); Rel $L_2$ degraded to $172.37\%$. | <span style="color:red; font-weight:bold;">FAIL (REJECTED)</span> |
| **Hypothesis H3C** | 2D within $5\%$ of 64D Model | 2D model is $+63.41\%$ worse ($172.37\%$ vs $105.48\%$). | <span style="color:red; font-weight:bold;">FAIL (REJECTED)</span> |

---

## 6. Authoritative Scientific Conclusion & Guidance for Next Phases

1. **Retraction of H3B Confirmation:**
   The claim in `EXP3_FINAL_REPORT.md` that Hypothesis H3B was "CONFIRMED" is scientifically untenable. The high $R^2$ was an artifact of linear decoder feedthrough, and the past invariance was an artifact of causal convolution padding. The physical mechanism of causal plastic state intervention was **NOT** confirmed.
2. **Definitive Status of EXP 3:**
   - **H3A:** REJECTED (COMPLETED FAILURE)
   - **H3B:** REJECTED (CONFIRMATION RESCINDED UPON FORENSIC AUDIT)
   - **H3C:** REJECTED (COMPLETED FAILURE)
3. **Core Research Lesson:**
   Intermediate supervision of physical states using classical 1D constitutive formulas hurts end-to-end neural operator performance, while higher-dimensional unconstrained latent representations ($d=64$) provide superior predictive accuracy. Neural operators do not naturally adopt classical constitutive equations simply because a loss function penalizes state error; instead, they find degenerate shortcuts through their decoder architectures.
4. **Pre-requisite for EXP 4:**
   EXP 4 must not assume that 2D physical state conditioning is effective. Any future causal operator work must address the direct-decoder confounding identified here, or pivot toward recurrent state-space formulations (such as Continuous S4 or state-space neural ODEs) where state transitions are dynamically coupled to the underlying time-evolution operator.
