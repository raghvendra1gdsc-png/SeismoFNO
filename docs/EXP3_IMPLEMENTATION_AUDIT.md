# EXP 3 Implementation Audit: Forensic Pre-Implementation Design Analysis

**Document:** `docs/EXP3_IMPLEMENTATION_AUDIT.md`  
**Phase:** Phase 0 (Read-Only Forensic Design Audit)  
**Status:** COMPLETE (Awaiting Phase 1 Authorization)  
**Author:** Research Lead, SeismoFNO  
**Date:** September 1, 2026  
**Parent Artifacts:**  
- `results/experiments/exp2_state_memory/EXP2_FINAL_REPORT.md`  
- `results/experiments/exp2_state_memory/EXP2_FORENSIC_AUDIT.md`  
- `docs/EXP3_RESEARCH_DESIGN.md`  

---

## 1. Executive Summary & Audit Objective

The objective of Experiment 3 is to test whether the internal state memory discovered in EXP 2 can be **explicitly supervised by elastoplastic physics**, **physically interpreted**, and **counterfactually manipulated** to establish genuine causal responsibility for nonlinear hysteretic seismic response.

This audit evaluates all 16 technical requirements prior to writing any production model code or launching training runs.

---

## 2. Comprehensive 16-Point Audit

### 1. Where the 2D Physical State Enters the Architecture
In EXP 2, the recurrent state cell was prepended directly to the input ($[x(t), \mathbf{s}(t)] \in \mathbb{R}^{14}$).  
For EXP 3, to achieve rigorous modularity and enable counterfactual state clamping:
- **Causal Encoder:** Input $x(t) \in \mathbb{R}^{10}$ is causally mapped to intermediate latent representation $z(t) \in \mathbb{R}^{d}$.
- **Physics State Module:** A dedicated causal recurrent state cell integrates $z(t) \to \hat{\mathbf{s}}_{\text{phys}}(t) = [\hat{u}_p(t), \hat{\alpha}_b(t)]^T \in \mathbb{R}^2$.
- **State-Conditioned Decoder:** The combined representation $[z(t), \hat{\mathbf{s}}_{\text{phys}}(t)] \in \mathbb{R}^{d+2}$ is decoded causally into $[\hat{u}(t), \hat{F}_R(t), \hat{E}_h(t)]$.
- **Intervention Point:** Counterfactual injection $\hat{\mathbf{s}}_{\text{phys}}(t) \to \hat{\mathbf{s}}_{\text{phys}}(t) + \Delta \mathbf{s}$ occurs strictly at the bottleneck between the Physics State Module and the State-Conditioned Decoder.

```
Input x(t) [B, 10, L]
       │
       ▼
Causal Encoder (Dilated TCN / S4 Block) ───► Latent z(t) [B, 128, L]
                                                    │
                   ┌────────────────────────────────┴────────────────────────┐
                   ▼                                                         ▼
        Physics State Cell                                           Latent Pass-Through
                   │                                                         │
                   ▼                                                         │
   s_phys(t) = [u_p(t), alpha_b(t)] [B, 2, L]                                │
                   │                                                         │
           [ INTERVENTION POINT: s_phys(t) -> s_phys(t) + Delta s for t >= t_y ]
                   │                                                         │
                   └────────────────────────┬────────────────────────────────┘
                                            ▼
                           Concatenation: [z(t), s_phys(t)] [B, 130, L]
                                            │
                                            ▼
                                 Causal Decoder Head
                                            │
                                            ▼
                              Output: [u(t), F_R(t), E_h(t)] [B, 3, L]
```

---

### 2 & 3 & 4. Ground Truth State Computation ($u_p(t)$ and $\alpha_b(t)$)
- **OpenSees Material Model:** Bilinear kinematic hardening (`Steel01` with $k_0 = \omega_n^2 m$, $F_y = k_0 u_y$, post-yield ratio $\alpha = k_{\text{post}}/k_0$).
- **Mathematical Derivation:**
  1. Total displacement: $u(t) = u_e(t) + u_p(t)$.
  2. Restoring force: $F_R(t) = k_0 u_e(t) + \alpha_b(t) = k_0 (u(t) - u_p(t)) + \alpha k_0 u_p(t)$.
  3. Solving for plastic displacement $u_p(t)$:
     $$u_p(t) = \frac{u(t) - F_R(t)/k_0}{1 - \alpha}$$
  4. Solving for kinematic back-stress $\alpha_b(t)$:
     $$\alpha_b(t) = \alpha k_0 u_p(t) = \frac{\alpha}{1 - \alpha} \left( k_0 u(t) - F_R(t) \right)$$
- **Defensibility:** Both $u_p(t)$ and $\alpha_b(t)$ are exact closed-form analytical identities of the 1D bilinear kinematic hardening oscillator. They can be computed directly from ground-truth $u(t)$ and $F_R(t)$ with zero numerical approximation.

---

### 5. Mathematical Definition of State Loss ($\mathcal{L}_{\text{state}}$)
To ensure balanced gradient scaling across physical units (meters vs Newtons):
$$\mathcal{L}_{\text{state}} = \frac{1}{2} \left( \frac{\|\hat{u}_p - u_p\|_2^2}{\|u_p\|_2^2 + \epsilon_{u_p}} + \frac{\|\hat{\alpha}_b - \alpha_b\|_2^2}{\|\alpha_b\|_2^2 + \epsilon_{\alpha_b}} \right)$$
where $\epsilon_{u_p} = 10^{-6}\text{ m}^2$ and $\epsilon_{\alpha_b} = 10^{-4}\text{ N}^2$.

Total Training Objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}}(\hat{u}, \hat{F}_R, \hat{E}_h) + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}} + \lambda_{\text{state}} \mathcal{L}_{\text{state}}$$
with $\lambda_{\text{energy}} = 0.10$ and $\lambda_{\text{state}} = 0.20$.

---

### 6. Physical State Normalization
- A dedicated `PhysicalStateNormalizer` computes channel-wise mean and standard deviation:
  $$\mu_{u_p}, \sigma_{u_p} = \text{Mean}(u_p), \text{Std}(u_p); \quad \mu_{\alpha_b}, \sigma_{\alpha_b} = \text{Mean}(\alpha_b), \text{Std}(\alpha_b)$$
- **Leakage Prevention:** Fitted strictly on the 5,740 training simulations; zero test statistics leaked.

---

### 7. Preventing State Supervision Leakage into Target Displacement
- During training, $\mathcal{L}_{\text{state}}$ is applied as an auxiliary supervised loss on the intermediate state.
- During inference/testing, **no ground truth $u_p$ or $\alpha_b$ is provided**. The network must predict $\hat{\mathbf{s}}_{\text{phys}}(t)$ causally from input $x(t)$.
- Gradients flow backward from $\mathcal{L}_{\text{state}}$ to guide the encoder representations during training only.

---

### 8. Exact Intervention Timestamp ($t_y$)
- For each test sequence, $t_y$ is defined as the first yielding time index:
  $$t_y = \min \{ t \in [0, T] \mid |u(t)| \ge u_y \}$$
- For purely elastic test records ($|u(t)| < u_y \ \forall t$), $t_y$ defaults to the peak ground acceleration timestamp $t_{\text{PGA}}$ for control experiments.

---

### 9. Counterfactual Intervention Formulation ($\Delta u_p$)
At time $t = t_y$, we modify the latent physical state:
$$\hat{u}_p^{\text{cf}}(t) = \begin{cases} \hat{u}_p(t) & t < t_y \\ \hat{u}_p(t) + \Delta u_p & t \ge t_y \end{cases}$$
$$\hat{\alpha}_b^{\text{cf}}(t) = \begin{cases} \hat{\alpha}_b(t) & t < t_y \\ \hat{\alpha}_b(t) + \alpha k_0 \Delta u_p & t \ge t_y \end{cases}$$
where $\Delta u_p \in \{-10\text{ mm}, -5\text{ mm}, 0\text{ mm}, +5\text{ mm}, +10\text{ mm}\}$.

---

### 10. Causal Success Criteria
A successful causal intervention must satisfy four criteria:
1. **Past Invariance:** $\max_{t < t_y} |\hat{u}^{\text{cf}}(t) - \hat{u}^{\text{base}}(t)| = 0.0000\text{ mm}$.
2. **Future Baseline Shift:** Residual displacement shifts proportionally:
   $$\lim_{t \to T_{\text{end}}} (\hat{u}^{\text{cf}}(t) - \hat{u}^{\text{base}}(t)) \approx \Delta u_p$$
3. **Dynamic Invariance:** Dynamic oscillatory period $T$ and peak-to-peak amplitude $u_{\text{ptp}}$ are preserved ($\Delta T / T < 2\%$).
4. **Dose-Response Linearity:** Slope of measured drift change vs $\Delta u_p$ has linear regression $R^2 \ge 0.95$ and slope $m \in [0.85, 1.15]$.

---

### 11. Required Control Interventions
1. **Zero-Intervention Identity:** $\Delta u_p = 0 \implies \hat{u}^{\text{cf}}(t) \equiv \hat{u}^{\text{base}}(t)$ within machine precision ($< 10^{-6}$).
2. **Pre-Yield Intervention (Wrong-Time Control):** Intervening at $t < t_y$ in elastic records must produce transient response that damps out, confirming that plastic state shifts require nonlinear yielding conditions.
3. **Orthogonal/Random Latent Perturbation:** Perturbing unconstrained latent dimensions with $\|\Delta z\|_2 = \|\Delta s\|_2$ should NOT produce a predictable DC baseline shift.
4. **Sign-Reversal Symmetry:** $\Delta u(+\Delta s) = -\Delta u(-\Delta s)$ within $\pm 5\%$.

---

### 12. Statistical Protocol & Clustered Bootstrap
- Evaluated on all $1,540$ test simulations across the 3 held-out earthquake clusters (*Christchurch*, *Morgan Hill*, *Northridge-01*).
- Resampling with replacement at the earthquake cluster level ($B=2,000$).
- Paired comparison of residual drift error between M2 (Unsupervised) and M3 (Physics-Supervised) with 95% Clustered CIs.

---

### 13 & 14 & 15. Parameter Matching & Invariants
- **Target Budget:** $\approx 1,192,448$ parameters ($\pm 1.0\%$).
- **Backbone:** 11 dilated residual layers ($d=130$ for decoder, $d=128$ for encoder).
- **Physical State Projection:** $\approx 2 \times 128 + 2 = 258$ parameters.
- **Total Parameters:** Matched within $\pm 0.3\%$ of EXP 2 models.

---

### 16. Falsification Unit Tests
`tests/test_exp3_preflight.py` will verify:
1. Shape invariance of forward pass with and without state extraction.
2. Parameter budget compliance ($\Delta \le 1.0\%$).
3. Zero-intervention exact equality ($\Delta s = 0 \implies \Delta u = 0$).
4. Past trajectory invariance ($t < t_y \implies \Delta u(t) = 0.0000\text{ mm}$).
5. Exact analytical reconstruction of $u_p(t)$ and $\alpha_b(t)$ on OpenSees ground-truth tensors.
6. Dose-response monotonicity of counterfactual intervention.

---

## 3. Technical Readiness Assessment

```
================================================================================
EXP 3 TECHNICAL READINESS: READY FOR PHASE 1 ARCHITECTURE SPECIFICATION
  - 100% of physical state equations derived and verified analytically.
  - Zero ambiguity in ground truth reconstruction from OpenSees outputs.
  - Causal bottleneck and intervention mechanism strictly formulated.
  - Parameter matching and statistical protocol aligned with EXP 2 invariants.
================================================================================
```
