# EXP 3-R TRAINING RECOVERY AUDIT

**Document:** `results/experiments/exp3r_ssm/TRAINING_RECOVERY_AUDIT.md`  
**Date:** September 2, 2026, 14:40 UTC  
**Auditor:** Hostile Scientific Research Custodian  

---

## 1. Frozen Training Protocol Verification (Phase 1)

All 14 frozen protocol requirements were verified against `config_locked.json`, `seed_manifest.json`, dataset caches, and model definitions:

```
+---------------------------------------------------------------------------------------------------------+
| PROTOCOL COMPLIANCE AUDIT TABLE                                                                         |
+------------------------------+---------------------------+------------------------+---------------------+
| Dimension                    | Frozen Requirement        | Active State           | Verification Status |
+------------------------------+---------------------------+------------------------+---------------------+
| Train Partition Cardinality  | 5,740 bilinear sims       | 5,740 bilinear sims    | 100% MATCH (PASS)   |
| Validation Partition         | 1,120 bilinear sims       | 1,120 bilinear sims    | 100% MATCH (PASS)   |
| Held-Out Test Partition      | 1,540 bilinear sims       | 1,540 bilinear sims    | 100% MATCH (PASS)   |
| Held-Out Parent Earthquakes  | Christchurch, Morgan Hill,| Christchurch, Morgan H,| 100% MATCH (PASS)   |
|                              | Northridge-01             | Northridge-01          |                     |
| Input Channels x_t           | 5 [ag, T, zeta, u_y, alpha| 5 [ag, T, zeta, u_y, a]| 100% MATCH (PASS)   |
| Prohibited Channel Audit     | Zero PGA, tau, future     | Zero prohibited inputs | 100% MATCH (PASS)   |
| State Dimension              | 64 [u, v, u_p, q_1..q_61] | 64 [u, v, u_p, q_1..q] | 100% MATCH (PASS)   |
| Transition Dynamics          | Pure Recurrent ResNet MLP | Pure Recurrent ResNet  | 100% MATCH (PASS)   |
| Readout Map                  | Pointwise G_theta(h_t)    | Pointwise G_theta(h_t) | 100% MATCH (PASS)   |
| Direct Feedthrough Edge      | Zero x_t -> y_t bypass    | Zero bypass (verified) | 100% MATCH (PASS)   |
| Parameter Count              | 1,192,448 (+/- 1.5%)      | 1,191,815 (-0.053%)    | 100% MATCH (PASS)   |
| Optimizer & Hyperparameters  | AdamW, lr=1e-3, wd=1e-4,  | AdamW, lr=1e-3, wd=1e-4| 100% MATCH (PASS)   |
|                              | clip=1.0, Cosine Annealing| clip=1.0, Cosine Sched |                     |
| Training Epochs Target       | 50 epochs per seed        | 50 epochs per seed     | 100% MATCH (PASS)   |
| Normalization Pipeline       | Train-only UnitGaussian   | Train-only UnitGaussian| 100% MATCH (PASS)   |
| Precommitted Seed List       | [42, 123, 456, 789, 1024] | [42, 123, 456, 789, 102| 100% MATCH (PASS)   |
+------------------------------+---------------------------+------------------------+---------------------+
```

---

## 2. Existing Seed Training State Inspection (Phase 2)

```
+---------------------------------------------------------------------------------------------------------+
| SEED TRAINING STATUS MATRIX                                                                             |
+------+-----------+--------------------+---------------------+------------------+------------------------+
| Seed | Max Epoch | Checkpoint Path    | Checkpoint SHA-256  | Best Val Resp L  | Status                 |
+------+-----------+--------------------+---------------------+------------------+------------------------+
| 42   | 3 / 50    | best_checkpoint.pt | e7f25b8c9d049661... | 1.20374          | Halted prematurely     |
| 123  | 0 / 50    | None               | N/A                 | N/A              | Pending execution      |
| 456  | 0 / 50    | None               | N/A                 | N/A              | Pending execution      |
| 789  | 0 / 50    | None               | N/A                 | N/A              | Pending execution      |
| 1024 | 0 / 50    | None               | N/A                 | N/A              | Pending execution      |
+------+-----------+--------------------+---------------------+------------------+------------------------+
```

### Checkpoint Integrity & Selection Verification:
- **Selection Criterion:** Strictly `val_response_loss < best_val_resp_loss`.
- **Test Set Access:** ZERO. The training loop never imported, loaded, or evaluated test samples.
- **Random Seed Determinism:** Fully preserved via PyTorch and NumPy seed locking.

---

## 3. Stop Condition B: Contamination Assessment

We explicitly evaluated whether test data influenced:
1. Checkpoint selection: **NO** (Validation response loss only).
2. Early stopping: **NO** (No early stopping triggered; process was halted by user foreground request).
3. Hyperparameter choice: **NO** (Locked in `config_locked.json`).
4. Architecture choice: **NO** (Graph audited and frozen).
5. Loss weighting: **NO** ($\lambda_{\text{state}} = 0.20, \lambda_{\text{energy}} = 0.10$ locked).
6. Training duration: **NO** (50 epochs preregistered).

### Authoritative Contamination Declaration:
# **"NO TRAINING-PROTOCOL CONTAMINATION DETECTED."**
