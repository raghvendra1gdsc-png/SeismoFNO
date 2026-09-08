# EXP 3 Final Scientific Report: Physics-Guided State Supervision & Counterfactual Causal Interventions

**Status:** COMPLETE (GATE 4 FINAL SCIENTIFIC REPORT)  
**Execution Date:** September 2, 2026  
**Protocol:** Frozen Gate 0–4 Scientific Protocol  
**Test Partition:** 1,540 Held-Out Bilinear Simulations (3 Unseen Parent Earthquakes: *Christchurch*, *Morgan Hill*, *Northridge-01*)  

---

## 1. Executive Summary & Hypothesis Verdicts

| Hypothesis | Tested Mechanism | Success Threshold | Measured Empirical Result | Scientific Verdict |
| :--- | :--- | :--- | :---: | :---: |
| **H3A** | Physics-Supervised Plastic State | $\ge 40\%$ Residual Drift Reduction vs EXP 2 | **-5.3%** (58.30 mm $\to$ 61.40 mm) | **NOT CONFIRMED** |
| **H3B** | Counterfactual Causal State Intervention | Past error $< 10^{-5}$ mm & Linear $R^2 \ge 0.90$ | Max Past: **0.000000 mm**, Mean $R^2 = \mathbf{0.9983}$, Slope $m = \mathbf{-0.01}$ | **CONFIRMED** |
| **H3C** | Minimal State Sufficiency (2D vs 64D) | 2D Error $\le 5\%$ from 64D Unconstrained | Delta: **+63.41%** (2D: 172.37% vs 64D: 105.48%) | **NOT CONFIRMED** |

---

## 2. Complete EXP 3 Benchmark Table (1,540 Held-Out Test Records)

| Model Architecture | Parameters | Rel $L_2(u)$ Mean (%) | 95% Clustered CI [%] | Peak $u$ Err (%) | Force Rel $L_2$ (%) | Energy Rel $L_2$ (%) | Residual Drift (mm) | Rel $L_2(u_p)$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP 2 Baseline (State-TCN)** | 1,182,451 | 140.63 | [134.16, 144.69] | 493.90 | 33.08 | 303.56 | 58.30 | 84.94 |
| **EXP 3 PG-TCN (Physics-Supervised)** | 1,192,849 | 172.37 | [166.44, 178.43] | 82.23 | 112.15 | 2715.48 | 61.40 | 8344364.00 |
| **EXP 3 PG-TCN (Unsupervised Ablation)**| 1,192,849 | 132.90 | [130.32, 135.48] | 72.62 | 97.94 | 1813.72 | 78.12 | 362589152.00 |
| **EXP 3 64D Unconstrained State** | 1,192,255 | 105.48 | [104.11, 106.46] | 76.41 | 99.19 | 3640.14 | 52.83 | 487407392.00 |

---

## 3. Counterfactual Causal Intervention Results (H3B)

Across all yielding held-out test simulations, clamping the latent physical state at yield onset ($t = t_y$) via:
$$s_{\mathrm{cf}}(t) = \hat{s}_{\mathrm{phys}}(t) + \Delta s, \quad \forall t \ge t_y$$
across doses $\Delta u_p \in \{-10, -5, 0, +5, +10\}\mathrm{\ mm}$ demonstrated:
1. **Strict Past Invariance:** $\max_{t < t_y} |\hat{u}^{\mathrm{cf}}(t) - \hat{u}^{\mathrm{base}}(t)| = \mathbf{0.000000\mathrm{\ mm}}$ (zero past modification).
2. **Dose-Response Linearity:** Mean $R^2 = \mathbf{0.9983}$ with response slope $m = \mathbf{-0.01}$.
3. **Negative Controls:** Zero intervention ($\Delta = 0$) produced exact baseline reproduction ($< 10^{-6}$ mm), and pre-yield interventions decayed cleanly.

---

## 4. Generated Artifacts
1. `figures/fig1_convergence.png` — Loss convergence history.
2. `figures/fig2_predicted_vs_true_state.png` — True vs. predicted physical state trajectories.
3. `figures/fig3_residual_drift_comparison.png` — Residual drift error bar charts.
4. `figures/fig4_error_vs_ductility.png` — Relative error vs ductility regimes.
5. `figures/fig5_counterfactual_dose_response.png` — Causal dose-response curves.
6. `figures/fig6_past_invariance.png` — Past trajectory invariance verification.
7. `figures/fig7_sign_symmetry.png` — Counterfactual sign symmetry.
8. `figures/fig8_negative_controls.png` — Negative control evaluations.
9. `figures/fig9_2d_vs_64d_sufficiency.png` — Minimal state sufficiency (2D vs 64D).
10. `metrics/summary_metrics.csv` — Earthquake-clustered bootstrap summary metrics.
11. `raw/test_records_detailed_metrics.csv` — Full 1,540 test simulation metrics.
12. `interventions/counterfactual_intervention_results.json` — Granular intervention telemetry.