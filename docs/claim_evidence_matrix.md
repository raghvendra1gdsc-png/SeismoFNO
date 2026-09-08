# SeismoFNO Master Claim-to-Evidence Matrix

**Purpose:** Comprehensive audit tracing every scientific, architectural, and numerical claim in the SeismoFNO project to verifiable code, scripts, literature citations, or formal falsifiable hypotheses.  
**Strict Honesty Rule:** No number may be presented in a report, paper, or README without an explicit Evidence Classification Tier.

---

## 1. Evidence Classification Tiers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: [OUR MEASUREMENT]     │ Directly measured from code execution in   │
│                               │ this repository with logged seeds & scripts.│
├───────────────────────────────┼─────────────────────────────────────────────┤
│ TIER 2: [REPRODUCED RESULT]   │ Literature method re-implemented and        │
│                               │ verified locally on our exact dataset split.│
├───────────────────────────────┼─────────────────────────────────────────────┤
│ TIER 3: [LITERATURE-REPORTED] │ Published number directly cited from an     │
│                               │ external peer-reviewed academic paper.      │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ TIER 4: [HYPOTHESIS]          │ Theoretical prediction awaiting formal      │
│                               │ empirical falsification / confirmation.     │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

---

## 2. Complete Traceability Matrix of Numerical & Scientific Claims

| # | Project Claim | Evidence Tier | Numerical Value / Finding | Primary Source / Artifact Path | Status / Notes |
| :- | :--- | :---: | :---: | :--- | :--- |
| **1** | Linear SDOF FNO accuracy | **[OUR MEASUREMENT]** | **$2.68\%$** Rel $L_2$ error | `results/phase4_mvp/linear_eval.json` | Verified in Phase 4. |
| **2** | Standard FNO elastic error ($\mu \le 1$) | **[OUR MEASUREMENT]** | **$10.77\%$** Rel $L_2$ error | `results/experiments/exp1_causality/summary_metrics.csv` | Measured in EXP 1 ($N=242$). |
| **3** | Standard FNO severe yield error ($\mu > 4$) | **[OUR MEASUREMENT]** | **$57.68\%$** Rel $L_2$ error | `results/experiments/exp1_causality/summary_metrics.csv` | Measured in EXP 1 ($N=830$). |
| **4** | Standard FNO overall test error | **[OUR MEASUREMENT]** | **$49.11\%$** Rel $L_2$ error | `results/experiments/exp1_causality/summary_metrics.csv` | Measured in EXP 1 ($N=1,540$). |
| **5** | Standard FNO future-input leakage ($t < t_0$) | **[OUR MEASUREMENT]** | **$48.90\%$** mean past alteration ($353.1\text{ mm}$ max) | `results/experiments/exp1_causality/causality_intervention.json` | Proves Mechanism A (Acausality). |
| **6** | Causal TCN future-input invariance | **[OUR MEASUREMENT]** | **$0.000000\%$** past alteration ($0.00\text{ mm}$) | `results/experiments/exp1_causality/causality_intervention.json` | Proves exact causal invariance. |
| **7** | Causal TCN severe yield error ($\mu > 4$) | **[OUR MEASUREMENT]** | **$92.43\%$** Rel $L_2$ error | `results/experiments/exp1_causality/summary_metrics.csv` | Falsifies H1 as sole bottleneck. |
| **8** | Causal TCN elastic error ($\mu \le 1$) | **[OUR MEASUREMENT]** | **$384.43\%$** Rel $L_2$ error | `results/experiments/exp1_causality/summary_metrics.csv` | Identified as FIR phase cancellation. |
| **9** | LSTM Baseline overall test error | **[REPRODUCED RESULT]**| **$67.74\%$** Rel $L_2$ error | `results/experiments/exp1_causality/summary_metrics.csv` | Evaluated on 1,540 held-out records. |
| **10**| MLP Baseline overall test error | **[REPRODUCED RESULT]**| **$106.01\%$** Rel $L_2$ error | `results/experiments/exp1_causality/summary_metrics.csv` | Evaluated on 1,540 held-out records. |
| **11**| Chopra Capacity Spectrum peak error | **[REPRODUCED RESULT]**| **$89.86\%$** peak error | `results/tables/unified_baseline_benchmark.csv` | Classical engineering benchmark. |
| **12**| Published nonlinear FNO error in literature | **[LITERATURE-REPORTED]**| **$\sim 35\% - 40\%$** Rel $L_2$ | Moya et al. (2023), Rahman et al. (2024) | Benchmark cited in `docs/baselines.md`. |
| **13**| S4 SSM severe yield accuracy ($\mu > 4$) | **[HYPOTHESIS]** | **$< 40.0\%$** Rel $L_2$ error | Target of `docs/exp2_state_memory_protocol.md` | To be tested in EXP 2. |
| **14**| S4 SSM residual plastic drift error | **[HYPOTHESIS]** | **$> 40\%$ reduction** vs TCN | Target of `docs/exp2_state_memory_protocol.md` | To be tested in EXP 2. |
| **15**| Zero-shot temporal resolution invariance | **[HYPOTHESIS]** | **$< 5\%$ degradation** from 25 to 200 Hz | Target of `docs/ablation_matrix.md` (EXP 6) | To be tested in EXP 6. |

---

## 3. Retracted & Prohibited Claims

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ❌ RETRACTED CLAIM 1: "Phase 5D proved state augmentation works."            │
│    Reason: Phase 5D used use_history_channel=True, which leaked the target  │
│    displacement trajectory u(t) into the input channels. Result is invalid.│
├─────────────────────────────────────────────────────────────────────────────┤
│ ❌ RETRACTED CLAIM 2: "FNO failure is caused purely by acausality."          │
│    Reason: Falsified by EXP 1 (Causal TCN eliminated acausality but had     │
│    92.43% severe yield error due to lack of recursive state integration).   │
├─────────────────────────────────────────────────────────────────────────────┤
│ ❌ PROHIBITED CLAIM 3: "S4 is inherently resolution-invariant in practice." │
│    Reason: Discretization algorithms (Bilinear/ZOH) and time-step scaling   │
│    can introduce pole distortion. Must be empirically verified in EXP 6.   │
└─────────────────────────────────────────────────────────────────────────────┘
```
