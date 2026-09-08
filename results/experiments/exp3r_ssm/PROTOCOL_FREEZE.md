# EXP 3-R PROTOCOL FREEZE DECLARATION

**Experiment:** EXP 3-R — Pure Recurrent State-Space Neural Operator for Path-Dependent Hysteresis  
**Location:** `results/experiments/exp3r_ssm/`  
**Status:** FROZEN PRIOR TO TRAINING  
**Timestamp:** September 2, 2026  
**Auditing Panel:** Hostile Research Committee  

---

## 1. Immutability Affirmation
- EXP 1, EXP 2, and EXP 3 artifacts, splits, checkpoints, and reports are **100% frozen and untouched**.
- Zero training runs for EXP 3-R have been executed.
- The model architecture, loss function, optimizer, hyperparameters, seeds, intervention definitions, control batteries, and evaluation metrics are **locked**.

---

## 2. Model & Parameter Budget Verification
- **Architecture:** `PureRecurrentSSM` (`src/models/exp3r_ssm.py`)
- **Input Channels:** $5$ ($a_g(t), T, \zeta, u_y, \alpha$) — Zero PGA, zero future features.
- **State Dimension:** $64$ ($[u, v, u_p] \in \mathbb{R}^3$, $q \in \mathbb{R}^{61}$)
- **Output Channels:** $3$ ($u(t), F_R(t), E_{\text{diss}}(t)$)
- **Trainable Parameters:** **1,191,815** (Target: 1,192,448, Deviation: $-0.053\%$).
- **Computational Graph:** Verified via PyTorch autograd — zero direct $x_t \to y_t$ feedforward bypass.

---

## 3. Precommitted Seed Manifest
- **Seeds:** `[42, 123, 456, 789, 1024]`
- Full initialization from scratch for each seed.

---

## 4. Causal Protocol Summary
- **Intervention:** $\Delta h = [0, 0, \Delta u_p, \mathbf{0}_{61}]^T$ at $t_y = \min \{ t \mid |u(t)| \ge u_y \}$.
- **Doses:** $\{-10.0, -5.0, -2.5, 0.0, +2.5, +5.0, +10.0\}\text{ mm}$.
- **Regime A Slope Interval:** $[0.900, 1.050]$ (Theoretically expected: $+0.980$).
- **Controls:** C0 (Zero), C1 (Physical), C2.A (Velocity Sham), C2.B (Latent Sham), C3 (Sign Reversal), C4 (Post-Shaking), C5 (Initial Condition), C6 (Elastic).
- **Specificity Threshold:** $\mathcal{S} \le 0.20$.
- **Static-Hold:** Treated strictly as a diagnostic, not proof of causality.

**PROTOCOL LOCKED. ZERO MODIFICATIONS PERMITTED.**
