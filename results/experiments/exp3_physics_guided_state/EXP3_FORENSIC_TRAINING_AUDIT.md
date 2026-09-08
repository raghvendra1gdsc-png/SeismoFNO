# EXP 3 Forensic Training Audit: Scientific Invariants & Invariant Verification

**Audit Date:** September 2, 2026  
**Parent Artifact:** `docs/EXP3_ARCHITECTURE.md`  

---

## 1. Protocol & Safety Invariant Checklist

| Audit Item | Invariant Rule | Verification Method | Status |
| :--- | :--- | :--- | :---: |
| **1. Zero Data Leakage** | Normalizers fit on train split only (5,740 sims) | Normalizer state audit | **PASSED** |
| **2. Checkpoint Selection** | Selected strictly on validation loss (1,120 sims) | Training loop callback | **PASSED** |
| **3. Test Partition** | Exactly 1,540 held-out simulations across 3 unseen earthquakes | Row count & cluster check | **PASSED** |
| **4. Parameter Matching** | All 4 models within $\pm 1.0\%$ of $1,192,448$ params | Explicit model parameter sum | **PASSED** |
| **5. Strict Causality** | Zero future noise past contamination ($0.0000$ mm) | Future noise intervention | **PASSED** |
| **6. Zero State Transfer** | Initial state $h_0 = \mathbf{0}$ reset per batch | Sequence permutation test | **PASSED** |
| **7. Inference Purity** | No ground truth $u_p$ or $\alpha_b$ provided during test inference | Model signature inspection | **PASSED** |

---

## 2. Parameter Budget Audit Table

| Model Identifier | Trainable Parameters | Target Budget | Delta (%) | Tolerance Status |
| :--- | :---: | :---: | :---: | :---: |
| **EXP 2 Baseline (State-TCN)** | 1,182,451 | 1,192,448 | -0.84% | **PASSED** |
| **EXP 3 PG-TCN (Physics-Supervised)** | 1,192,849 | 1,192,448 | +0.03% | **PASSED** |
| **EXP 3 PG-TCN (Unsupervised Ablation)** | 1,192,849 | 1,192,448 | +0.03% | **PASSED** |
| **EXP 3 64D Unconstrained State** | 1,192,255 | 1,192,448 | -0.02% | **PASSED** |
