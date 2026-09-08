# EXP 3 Research Design: Physics-Guided Latent State Supervision & Counterfactual Intervention

**Document Version:** 1.0 (Pre-Registration Draft)  
**Author:** Research Lead, SeismoFNO  
**Date:** September 1, 2026  
**Parent Study:** SeismoFNO Nonlinear Dynamics Benchmark  
**Prerequisites:** EXP 1 (Causality Benchmark — Complete) | EXP 2 (State Memory & Latent Probing — Validated with Caveats)  

---

## 1. Scientific Motivation & Core Research Question

### Background & Forensic Findings from EXP 2:
Experiment 2 established that:
1. Augmenting causal neural operators with internal recurrent state memory reduces nonlinear trajectory error by **55.7%** (State-Augmented TCN vs. Causal TCN).
2. Unsupervised latent representations spontaneously encode the unobserved plastic drift offset ($u_p(t) = u(t) - F_R(t)/k_0$) with linear probe decodability $R^2 = 0.897$.
3. However, EXP 2 demonstrated only an **observational correlation** between state capacity and performance. It did not establish whether the learned latent state is **causally responsible** for the physical path-dependent mechanics of yielding.

### Central Research Question for EXP 3:
> **"Can the learned latent state be explicitly constrained by elastoplastic physics and counterfactually manipulated to prove causal responsibility for path-dependent seismic response?"**

We aim to bridge the gap between empirical sequence modeling and computational plasticity by transitioning from **unsupervised memory** to **physically supervised internal state variables**.

---

## 2. Formal Hypotheses & Falsification Criteria

### Hypothesis 3A: Explicit Plastic-State Supervision
*Supervising a dedicated 2-dimensional sub-vector of the latent state $\mathbf{s}_t = [u_p(t), \alpha_b(t)]^T$ (where $u_p$ is plastic excursion and $\alpha_b$ is kinematic back-stress) using an auxiliary physics loss $\mathcal{L}_{\text{state}}$ will reduce residual drift error $|u(T_{\text{end}}) - \hat{u}(T_{\text{end}})|$ by $\ge 40\%$ compared to unsupervised state memory, under identical parameter budgets.*

- **Falsification Criterion 3A:** If residual drift error fails to decrease by at least $25\%$ with non-overlapping 95% Clustered CIs, Hypothesis 3A is falsified.

### Hypothesis 3B: Counterfactual Latent State Intervention
*Intervening directly on the latent plastic state at yield onset ($t = t_{\text{yield}}$) by clamping or shifting $\mathbf{s}_{t_{\text{yield}}} \to \mathbf{s}_{t_{\text{yield}}} + \Delta u_p$ will causally shift subsequent baseline displacement by exactly $\Delta u_p$ ($R^2_{\text{shift}} \ge 0.95$) without distorting the underlying elastic oscillatory dynamics.*

- **Falsification Criterion 3B:** If intervening on the latent state fails to produce a proportional future trajectory offset ($R^2_{\text{shift}} < 0.80$), or causes high-frequency instability, the hypothesis of causal latent state representation is falsified.

### Hypothesis 3C: Minimal State Dimension Sufficiency
*A structured 2D physical state ($d_{\text{state}} = 2$) with thermodynamic dissipation constraints achieves parity ($\le 5\%$ error difference) with high-dimensional unconstrained state spaces ($d_{\text{state}} = 64$).*

- **Falsification Criterion 3C:** If $d_{\text{state}} = 2$ incurs $> 20\%$ higher trajectory error than $d_{\text{state}} = 64$, minimal thermodynamic state sufficiency is falsified.

---

## 3. Mathematical Formulation & Architecture

### 3.1 Physics-Guided Neural State-Space Architecture (PG-SSM)
The discrete-time state update is governed by:
$$\begin{aligned}
\mathbf{z}_t &= \mathbf{W}_x x_t + \mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{b} \\
\mathbf{s}_t &= \begin{bmatrix} s_t^{\text{plastic}} \\ s_t^{\text{backstress}} \\ \mathbf{s}_t^{\text{latent}} \end{bmatrix} = \begin{bmatrix} \text{cumsum}(\text{ReLU}(|\dot{u}_t| - \dot{u}_y) \cdot \text{sgn}(F_{R,t})) \\ \text{clip}(F_{R,t} - k_0 (u_t - u_{p,t}), -F_y, F_y) \\ \tanh(\mathbf{W}_z \mathbf{z}_t) \end{bmatrix}
\end{aligned}$$

### 3.2 Composite Loss Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}}(\hat{u}, \hat{F}_R, \hat{E}_h) + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}} + \lambda_{\text{state}} \mathcal{L}_{\text{state}}$$
where:
- $\mathcal{L}_{\text{data}}$: Relative $L_2$ trajectory loss over $[u, F_R, E_h]$.
- $\mathcal{L}_{\text{energy}}$: Dynamic energy balance residual penalty ($\lambda_{\text{energy}} = 0.10$).
- $\mathcal{L}_{\text{state}}$: Explicit plastic state supervision loss ($\lambda_{\text{state}} = 0.20$):
  $$\mathcal{L}_{\text{state}} = \frac{1}{T} \sum_{t=1}^T \left( \frac{|\hat{s}_t^{\text{plastic}} - u_p(t)|^2}{\sigma_{u_p}^2 + \epsilon} \right)$$

---

## 4. Controlled Model Matrix & Parameter Matching

All models will be strictly matched to the baseline parameter budget of **~1.192M parameters ($\pm 1.0\%$)**:

| Model Index | Model Identifier | Architecture | State Mechanism | State Supervision | Target Parameters |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **M1** | `Causal TCN (State-Free)` | Dilated 1D Conv | None (FIR) | None | $1,192,448$ |
| **M2** | `State-Augmented TCN (Unsupervised)` | Dilated 1D Conv + RNN | Unconstrained $\mathbb{R}^4$ | None ($\lambda_{\text{state}} = 0$) | $1,192,448$ |
| **M3** | `Physics-Supervised TCN (PG-TCN)` | Dilated 1D Conv + RNN | Explicit $[u_p, \alpha_b, \mathbf{s}_{\text{lat}}]$ | Supervised ($\lambda_{\text{state}} = 0.20$) | $1,192,448$ |
| **M4** | `Continuous S4 SSM (Unsupervised)` | S4D Layer Stack | Unconstrained $\mathbb{C}^{64}$ | None ($\lambda_{\text{state}} = 0$) | $1,192,448$ |
| **M5** | `Physics-Supervised S4 (PG-S4)` | S4D Layer + State Head | Explicit $[u_p, \alpha_b]$ Head | Supervised ($\lambda_{\text{state}} = 0.20$) | $1,192,448$ |

---

## 5. Experimental Protocol & Invariants

### 5.1 Dataset & Splits (Frozen Invariant)
- Master simulation index: `data/simulations/simulation_index.csv` (8,400 bilinear simulations).
- Split file: `data/processed/splits/held_out_earthquake_split.json` (SHA-256: `d79f22f7...`).
  - **Training:** 11 earthquakes (5,740 simulations).
  - **Validation:** 2 earthquakes (1,120 simulations).
  - **Test (Zero-Shot):** 3 unseen earthquakes (1,540 simulations: *Christchurch*, *Morgan Hill*, *Northridge-01*).

### 5.2 Counterfactual Intervention Protocol
For 100 randomly sampled test records in the yielding regime ($\mu > 2$):
1. Run standard forward pass $\to \hat{u}_{\text{orig}}(t)$.
2. Identify first yield timestamp $t_y = \min \{t \mid |u(t)| \ge u_y\}$.
3. At $t = t_y$, inject an instantaneous state perturbation $\Delta s \in \{+5\text{ mm}, -5\text{ mm}, +10\text{ mm}, -10\text{ mm}\}$.
4. Continue causal forward recurrence from $t_y$ to $T_{\text{end}} \to \hat{u}_{\text{pert}}(t)$.
5. Measure trajectory shift $\delta u(t) = \hat{u}_{\text{pert}}(t) - \hat{u}_{\text{orig}}(t)$ and verify if $\lim_{t \to T_{\text{end}}} \delta u(t) = \Delta s$.

---

## 6. Required Deliverables & Figures

### Planned Publication Figures:
1. **Figure 1: State Supervision Ablation vs. Ductility:** Comparative Relative $L_2$ across ductility regimes with 95% Clustered CIs for M1–M5.
2. **Figure 2: Residual Drift Error Reduction:** Boxplots and CDFs of residual plastic offset error $|u(T_{\text{end}}) - \hat{u}(T_{\text{end}})|$.
3. **Figure 3: Counterfactual State Intervention Trajectories:** Time-series demonstration of causal baseline shifting upon $\Delta s$ injection.
4. **Figure 4: Latent State Phase-Space Trajectories:** Predicted $(\hat{u}(t), \hat{s}_t)$ vs. analytical $(u(t), u_p(t))$ phase portraits.
5. **Figure 5: State Dimension Ablation Sweep:** Performance vs. $d_{\text{state}} \in \{1, 2, 4, 8, 16, 32, 64\}$.

---

## 7. Expected Risks & Failure Modes

1. **Optimization Trade-Off:** $\mathcal{L}_{\text{state}}$ might compete with data loss $\mathcal{L}_{\text{data}}$, slightly increasing high-frequency elastic error while decreasing drift. *Mitigation: Curriculum weighting on $\lambda_{\text{state}}$ ($0 \to 0.20$).*
2. **Counterfactual Decay:** In continuous S4, state perturbations might decay if $\text{Re}(\lambda) < 0$. *Mitigation: Zero-eigenvalue integrator mode ($\lambda_0 = 0$) for plastic accumulation.*
3. **Cluster Covariance:** Statistical testing across 3 test earthquake clusters. *Mitigation: Explicit reporting of cluster bootstrap intervals.*
