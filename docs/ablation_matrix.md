# Systematic Ablation Matrix & Controlled Experiment Protocol

This document specifies the 9 controlled ablation experiments designed to isolate the failure mechanisms of neural operators in nonlinear structural dynamics, audit inherited numerical claims, and establish the ranked implementation sequence based on scientific information gain.

---

## 1. Audit of Numerical Claims & Evidence Classification

To prevent ungrounded claims, every metric in SeismoFNO is strictly classified under one of four evidence tiers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EVIDENCE CLASSIFICATION TIERS                         │
├──────────────────────────┬──────────────────────────────────────────────────┤
│ 1. [OUR MEASUREMENT]     │ Measured directly from scripts on this codebase. │
│ 2. [REPRODUCED RESULT]   │ Re-implemented and verified against literature.  │
│ 3. [LITERATURE-REPORTED] │ Published number cited from external paper.      │
│ 4. [HYPOTHESIS]          │ Scientific prediction to be tested/falsified.    │
└──────────────────────────┴──────────────────────────────────────────────────┘
```

### Audited Benchmark Summary
- **Linear SDOF FNO Error:** $[2.68\%]$ — *[OUR MEASUREMENT, Phase 4]*
- **Standard FNO Post-Yield Error ($\mu > 1$):** $[57.48\%]$ — *[OUR MEASUREMENT, Table 1, Held-Out Earthquakes]*
- **Published FNO Nonlinear Benchmark:** $[35\% - 40\%]$ — *[LITERATURE-REPORTED: Moya et al. 2023, Rahman et al. 2024]*
- **Standard FNO Peak $u_{\max}$ Error:** $[15.18\%]$ — *[OUR MEASUREMENT, Table 1]*
- **LSTM Baseline Post-Yield Error:** $[74.44\%]$ — *[OUR MEASUREMENT, Table 1]*
- **Chopra Capacity Spectrum Peak Error:** $[89.86\%]$ — *[OUR MEASUREMENT, Table 1]*
- **Target Causal State Post-Yield Error:** $[< 35.0\%]$ — *[HYPOTHESIS]*

---

## 2. The 9 Controlled Ablation Experiments

---

### EXP 1: Global Spectral vs. Strictly Causal Temporal Processing
- **Goal:** Directly test **Hypothesis 1 (Spectral Acausality)** by comparing global Fourier spectral convolutions against strictly left-padded causal convolutions.
- **Independent Variable:** Temporal receptive field causality (Acausal Global FFT vs. Strictly Causal Dilated Conv).
- **Dependent Variables:** Rel $L_2$ $u(t)$, Phase error $\Delta \Phi$, Post-yield error ($\mu > 4$).
- **Controlled Variables:** Parameters ($\sim 1.1\text{M}$), depth (11 layers / 4 blocks), dataset, AdamW optimizer, cosine schedule.
- **Training Split:** `held_out_earthquake_split.json` (Train: 5,740 records).
- **Test Split:** `held_out_earthquake_split.json` (Test: 1,540 records).
- **Expected Outcome:** Causal model eliminates precursor oscillations and achieves $< 38\%$ post-yield error.
- **Falsification Result:** If Causal TCN achieves post-yield error $\ge 55\%$ (matching FNO), acausality is **FALSIFIED** as the cause of failure.

---

### EXP 2: Latent Internal State vs. State-Free Representation
- **Goal:** Directly test **Hypothesis 2 (Internal State Tracking)** for plastic offset memory.
- **Independent Variable:** State representation (State-free Causal TCN vs. Continuous State-Space SSM / Explicit $z(t)$ tracking).
- **Dependent Variables:** Residual plastic drift error $|\hat{u}_{\text{end}} - u_{\text{end}}|$, Hysteresis loop area error $\text{Err}_{A_{\text{loop}}}$.
- **Controlled Variables:** Causal receptive field length ($> 4,000$ steps), parameter count.
- **Training & Test Split:** `held_out_earthquake_split.json`.
- **Expected Outcome:** State-aware SSM tracks shifted elastic origin, reducing residual drift error by $> 30\%$.
- **Falsification Result:** If residual plastic drift error improves by $< 10\%$, explicit state representation is **FALSIFIED** as necessary for SDOF dynamics.

---

### EXP 3: Fourier Mode Cutoff Sweep ($K \in [16, 32, 64, 128, 256, 512]$)
- **Goal:** Directly test **Hypothesis 3 (Spectral Truncation)**.
- **Independent Variable:** Truncated Fourier modes $K \in \{16, 32, 64, 128, 256, 512\}$.
- **Dependent Variables:** Acceleration error $\ddot{u}(t)$, Yield instant error $\Delta t_{\text{yield}}$, High-frequency power spectral density (PSD).
- **Controlled Variables:** FNO width (48), depth (4), training split.
- **Training & Test Split:** `held_out_earthquake_split.json`.
- **Expected Outcome:** Higher $K$ improves high-frequency acceleration but saturates without improving post-yield phase lag.
- **Falsification Result:** If increasing $K$ to 512 drives post-yield displacement error below $25\%$, Hypothesis 1 is **FALSIFIED** (proving spectral bandwidth was the true bottleneck).

---

### EXP 4: Structural Parameter Conditioning Mechanisms
- **Goal:** Test **Hypothesis 4 (Parameter Modulation)**.
- **Independent Variable:** Conditioning method (Constant channel concatenation vs. FiLM scale-shift vs. Dynamic hypernetwork).
- **Dependent Variables:** Interpolation error on held-out periods $T_n$, Ductility error.
- **Training Split:** `held_out_structure_split.json` ($T_n \in \{0.2, 0.5, 1.0, 1.5, 2.0\}\text{ s}$).
- **Test Split:** `held_out_structure_split.json` ($T_n \in \{0.3, 0.75, 1.25, 1.75\}\text{ s}$).
- **Expected Outcome:** FiLM modulation reduces parameter interpolation error from $663\% \to < 25\%$.
- **Falsification Result:** If FiLM achieves $> 100\%$ error on held-out $T_n$, conditioning alone cannot resolve parameter extrapolation.

---

### EXP 5: Thermodynamic Loss Weighting ($\lambda_{\text{energy}} \in \{0.0, 0.01, 0.10, 0.50\}$)
- **Goal:** Test **Hypothesis 5 (Thermodynamic Loss Regularization)**.
- **Independent Variable:** Energy loss penalty coefficient $\lambda_{\text{energy}}$.
- **Dependent Variables:** Restoring force error $F_R(t)$, Energy balance residual $\mathcal{R}_{\text{energy}}(t)$.
- **Controlled Variables:** Standard FNO architecture, training seed 42.
- **Expected Outcome:** $\lambda_{\text{energy}} = 0.10$ produces optimal balance between force accuracy and work consistency.
- **Falsification Result:** If $\lambda_{\text{energy}} = 0.0$ and $\lambda_{\text{energy}} = 0.10$ yield identical restoring force metrics ($\le 1.5\%$ difference), physics regularization is **FALSIFIED** as an effective inductive bias.

---

### EXP 6: Temporal Super-Resolution & Discretization Invariance
- **Goal:** Evaluate continuous resolution invariance under variable sampling rates.
- **Independent Variable:** Evaluation sampling frequency $f_s \in \{25\text{ Hz}, 50\text{ Hz}, 100\text{ Hz}, 200\text{ Hz}\}$ ($\Delta t \in \{0.04s, 0.02s, 0.01s, 0.005s\}$).
- **Dependent Variables:** Relative $L_2$ degradation $\Delta \text{Err}(f_s) = \text{Err}(f_s) - \text{Err}(100\text{ Hz})$.
- **Controlled Variables:** Models trained strictly at $100\text{ Hz}$ with zero fine-tuning.
- **Comparison:** FNO vs. Causal TCN vs. S4 SSM vs. LSTM.
- **Expected Outcome:** FNO and S4 SSM preserve zero-shot invariance ($< 5\%$ error rise); discrete TCN and LSTM degrade significantly.

---

### EXP 7: Ductility Level Disaggregation ($\mu \le 1, 1 < \mu \le 2, 2 < \mu \le 4, \mu > 4$)
- **Goal:** Measure error scaling as structural nonlinearity and plastic dissipation intensify.
- **Independent Variable:** Peak ductility demand $\mu = u_{\max} / u_y$.
- **Dependent Variables:** Trajectory Rel $L_2$, Peak $u_{\max}$ error, Hysteresis area discrepancy.
- **Expected Outcome:** FNO error scales monotonically with ductility ($\approx 10\% \to 70\%$); causal state models maintain bounded error ($< 35\%$).

---

### EXP 8: Earthquake-Event Generalization & Aleatory Dispersion
- **Goal:** Quantify record-to-record variability across diverse tectonic mechanisms (strike-slip, reverse, near-fault pulse).
- **Independent Variable:** 21 individual held-out test earthquakes.
- **Dependent Variables:** Per-event median error, logarithmic standard deviation $\sigma_{\ln \text{error}}$.
- **Expected Outcome:** Near-fault velocity pulse records exhibit highest error across all architectures.

---

### EXP 9: Computational Efficiency & Throughput Scaling
- **Goal:** Benchmark execution wall-clock time, GPU memory footprint, and batch throughput.
- **Independent Variable:** Batch size $B \in \{1, 8, 32, 64, 128, 512\}$.
- **Dependent Variables:** Inference latency ($ms$), Throughput ($rec/s$), Peak VRAM ($MB$).
- **Comparison:** OpenSeesPy (Single-thread & 8-core multi-process) vs. NumPy Newmark vs. FNO vs. Causal TCN vs. S4 SSM.

---

## 3. Ranked Implementation Order (By Scientific Information Gain)

To discover whether our central hypothesis is true with minimal computational waste, we execute experiments in strict order of scientific priority:

```
                   RANKED SCIENTIFIC EXECUTION SEQUENCE
┌─────────────────────────────────────────────────────────────────────────────┐
│ RANK 1: EXP 1 — The Causal vs. Global Spectral Showdown                     │
│   • Runs FIRST. Trains Causal TCN on held-out earthquake split.             │
│   • Scientific Information Gain: MAXIMUM. Immediately determines if         │
│     acausality is the root cause of the 57% post-yield breakdown.           │
├─────────────────────────────────────────────────────────────────────────────┤
│ RANK 2: EXP 3 — Fourier Mode Cutoff Sweep (K = 16 to 512)                   │
│   • Tests whether spectral truncation is the confounding factor in FNO.     │
├─────────────────────────────────────────────────────────────────────────────┤
│ RANK 3: EXP 2 — Continuous State-Space (S4/SSM) Evaluation                  │
│   • Evaluates whether continuous state representation beats state-free TCN  │
│     while preserving resolution invariance.                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ RANK 4: EXP 5 — Thermodynamic Loss Regularization Ablation                  │
│   • Quantifies genuine physical benefit of the normalized energy loss.      │
├─────────────────────────────────────────────────────────────────────────────┤
│ RANK 5: EXP 6 — Zero-Shot Temporal Resolution Stress Test                   │
│   • Evaluates sampling rate robustness (25 Hz to 200 Hz).                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ RANK 6: EXP 4 — Structural Parameter Conditioning (FiLM)                    │
│   • Tests out-of-distribution period generalization.                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ RANK 7: EXP 7, 8, 9 — Multi-Dimensional Disaggregation & Speed Scaling      │
│   • Compiles the master multi-metric publication tables.                    │
└─────────────────────────────────────────────────────────────────────────────┘
```
