# Master SeismoFNO Research Protocol (v2.0)

**Authority Level:** Authoritative Master Research Protocol (Reconciled from GStack Hostile, Engineering, PI, and Design Audits)  
**Governing Standard:** Top-Tier Computational Mechanics (*CMAME*) & Structural Dynamics (*EESD*) Journal Publication Criteria  
**Revision Date:** 2026-08-30  

---

## 1. Executive Mission & Epistemic Hierarchy

The primary mission of **SeismoFNO** is to discover, mathematically formalize, and empirically evaluate the exact boundary where continuous neural operators break down under non-smooth, path-dependent material bifurcations (elastoplasticity), and to establish the necessary and sufficient architectural inductive biases (causality, continuous state-space memory, thermodynamic conservation) required for surrogate modeling of nonlinear structural dynamics.

### Epistemic Hierarchy of Decisions
When experimental, engineering, or visual proposals conflict, decisions are strictly resolved by the following hierarchy of authority:
1. **Tier 1 — Scientific Validity & Hostile Falsification:** Physical conservation laws, strict causality, zero data leakage, and unconfounded hypotheses override all other concerns.
2. **Tier 2 — Engineering Isolation:** Controlled parameter matching ($\pm 1.5\%$), standardized training budgets, and identical numerical discretizations override model-specific tuning.
3. **Tier 3 — Research Strategy:** Focus on high-information-gain mechanistic discoveries over broad deep-learning architectural bake-offs.
4. **Tier 4 — Visual & Practical Usability:** Clear, honest, and uncompressed publication figures that faithfully reflect uncertainty and limitations.

---

## 2. Explicit Reconciliation of Literature & Previous Review Conflicts

```
                          CONFLICT RESOLUTION MATRIX
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ Prior Conflicted Claim               │ Reconciled Authoritative Rule        │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ "EXP 1 proved Mechanism B."          │ ❌ FALSE. EXP 1 falsified H1 (that   │
│                                      │ causality alone fixes FNO) and gave  │
│                                      │ diagnostic motivation for Mech B.    │
│                                      │ Mech B is tested in EXP 2.           │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ "Use Hilbert-transform phase error   │ ❌ INVALID for plastic drift.        │
│ on total displacement u(t)."         │ Hilbert transform of non-zero-mean   │
│                                      │ drifted signals violates Bedrosian   │
│                                      │ theorem. Use Zero-Crossing Lag and   │
│                                      │ DTW, or Hilbert on elastic comp only.│
├──────────────────────────────────────┼──────────────────────────────────────┤
│ "S4 is automatically resolution      │ ❌ UNVERIFIED A PRIORI. Discreti-    │
│ invariant."                          │ zation (Bilinear/ZOH) can warp poles.│
│                                      │ Resolution invariance must be proven │
│                                      │ empirically across 25–200 Hz in EXP6.│
├──────────────────────────────────────┼──────────────────────────────────────┤
│ "Use unclustered bootstrap CIs."     │ ❌ INVALID. Ignores intra-earthquake │
│                                      │ correlation. Use Clustered Block     │
│                                      │ Bootstrap over earthquake clusters.  │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ "use_history_channel provides state."│ 🔴 FATAL DATA LEAKAGE. Legacy Phase  │
│                                      │ 5D input of u(t) path is excised.    │
│                                      │ Inputs are strictly foundation ag(t) │
│                                      │ and known structural properties.     │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 3. The 5 Formal Failure Hypotheses (Popperian Formulation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Hypothesis 1 (Temporal Acausality — Mechanism A):                         │
│    Global Fourier operators leak future excitation into pre-yield states.   │
│    Status: CONFIRMED in EXP 1 (48.90% future-to-past leakage measured).     │
│    However, causality alone does NOT restore nonlinear plastic accuracy.    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Hypothesis 2 (Internal State Memory — Mechanism B):                      │
│    Feedforward FIR models fail in plasticity because they lack continuous   │
│    recursive state integration to track shifted plastic origin u_p(t).      │
│    Test: EXP 2 (Continuous State-Space Operator vs State-Free Causal TCN).  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Hypothesis 3 (Spectral Mode Truncation — Mechanism C):                   │
│    FNO failure is an artifact of cutting Fourier modes at K=128, which      │
│    filters high-frequency harmonics of sharp yielding slope discontinuities.│
│    Test: EXP 3 (Mode Cutoff Sweep K in [16, 32, 64, 128, 256, 512]).        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Hypothesis 4 (Structural Parameter Modulation — Mechanism D):            │
│    Concatenated static channels (Tn, u_y) fail to modulate spectral weights.│
│    Test: EXP 4 (Channel Concatenation vs. FiLM Hypernetwork on held-out Tn).│
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. Hypothesis 5 (Thermodynamic Consistency — Mechanism E):                  │
│    Scale-regularized hysteretic energy loss enforces the 2nd Law of Thermo- │
│    dynamics, preventing non-physical stiffness generation during unloading. │
│    Test: EXP 5 (Loss penalty ablation lambda_energy in [0.0, 0.01, 0.1, 0.5]).│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Ground Motion Provenance & Partitioning Rules

1. **Ground Motion Database:**
   - **Primary Real Set:** 120 PEER NGA-West2 recorded ground motions from shallow crustal earthquakes ($M_w 5.5 - 7.9$, $R_{\text{rup}} 5 - 70\text{ km}$, $V_{s30} 180 - 800\text{ m/s}$).
   - **Provenance Clarity:** Never describe synthetically scaled waveforms as separate natural earthquakes. The primary earthquake entity is the PEER Record Sequence Number (RSN).
2. **Zero-Leakage Partitioning:**
   - **Partitioning Unit:** Unique PEER RSN.
   - **Split Allocation:** 82 RSNs Train ($68.3\%$, 5,740 simulations) / 16 RSNs Validation ($13.3\%$, 1,120 simulations) / 22 RSNs Test ($18.3\%$, 1,540 simulations).
   - **Zero-Overlap Guarantee:** No earthquake event or scaled waveform appearing in testing is ever seen during training or validation.

---

## 5. Statistical Methodology & Metric Definitions

### 5.1 Clustered Block Bootstrap for Uncertainty
Because simulations originating from the same earthquake event share seismic frequency content (intra-event covariance), classical i.i.d. bootstrap underestimates error variance.  
All reported $95\%$ confidence intervals must be computed via **Cluster-Level Block Bootstrap**:
1. Resample $N_{\text{clusters}} = 22$ earthquake events with replacement ($B = 2,000$ bootstrap iterations).
2. Extract all simulations belonging to the resampled clusters.
3. Compute metric distribution across bootstrap iterations: $\text{CI}_{95} = [q_{0.025}, q_{0.975}]$.

### 5.2 Mandatory Engineering Metrics
Every model evaluation must report the following 6 core metrics disaggregated by ductility demand ($\mu \le 1$, $1 < \mu \le 2$, $2 < \mu \le 4$, $\mu > 4$):
1. **Trajectory Relative $L_2$ Error:** $\text{Rel } L_2(u) = \|u - \hat{u}\|_2 / (\|u\|_2 + \epsilon)$ (%).
2. **Peak Displacement Error:** $\text{Err}(u_{\max}) = |u_{\max} - \hat{u}_{\max}| / u_y$ (dimensionless ductility units).
3. **Residual Plastic Drift Error:** $\text{Err}(u_{\text{residual}}) = |u(T) - \hat{u}(T)|$ (mm).
4. **Hysteresis Loop Enclosed Area Error:** $\text{Err}(A_{\text{loop}}) = |A_{\text{loop}} - \hat{A}_{\text{loop}}| / A_{\text{loop}}$ (%).
5. **Hysteretic Energy Dissipation Error:** $\text{Rel } L_2(E_h) = \|E_h - \hat{E}_h\|_2 / (\|E_h\|_2 + \epsilon)$ (%).
6. **Future-Input Causality Discrepancy:** $\text{Discr}_{\text{past}}(t_0) = \max_{t < t_0} |u_{\text{clean}}(t) - u_{\text{perturbed}}(t)|$ (mm).

---

## 6. Execution Gates for Future Phases

| Phase / Experiment | Objective | Entry Gate Requirement |
| :--- | :--- | :--- |
| **EXP 1 (Completed)** | Diagnostic Causality Proof | Passed (48.9% FNO leakage, TCN FIR drift failure). |
| **EXP 2 (Next)** | State-Space Operator Evaluation | Architecture parameter-matched to $1.19\text{M}$ ($\pm 1.5\%$), zero leakage. |
| **EXP 3 (Control)** | Mode Cutoff Sweep ($K=16\dots 512$) | Standard FNO baseline trained across $K \in \{16, 32, 64, 128, 256, 512\}$. |
| **EXP 4 (Surrogacy)**| Period Generalization (FiLM) | Held-out structural period split ($T_{\text{test}} \notin T_{\text{train}}$). |
| **EXP 6 (Operator)** | Discretization Invariance | Zero-shot evaluation across $\Delta t \in \{0.04\text{ s}, 0.02\text{ s}, 0.01\text{ s}, 0.005\text{ s}\}$. |
