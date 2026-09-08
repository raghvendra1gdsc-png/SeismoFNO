# EXP 3-R EVALUATION PROTOCOL

**Document:** `results/experiments/exp3r_ssm/evaluation_protocol.md`  
**Status:** FROZEN  
**Target:** 1,540 Held-Out Bilinear Test Simulations Across 3 Unseen Earthquakes  

---

## 1. Single-Pass Test Evaluation Rule
- The test set is evaluated **exactly once** per seed after model training and validation-based checkpoint selection are completed.
- Zero tuning or threshold adjustment is permitted after viewing test set results.

---

## 2. Statistical Reporting Hierarchy (K = 3 Clusters)

```
+---------------------------------------------------------------------------------------------------+
|                                 TEST CLUSTER REPORTING SPECIFICATION                              |
+--------------------------+---------------------+-------------------+------------------------------+
| Earthquake Cluster       | Station RSN Count   | Simulation Count  | Tectonic Regime              |
+--------------------------+---------------------+-------------------+------------------------------+
| Christchurch             | 7 RSNs              | 520 simulations   | Strike-slip / Complex fault  |
| Morgan Hill              | 7 RSNs              | 500 simulations   | Strike-slip (Calaveras)      |
| Northridge-01            | 8 RSNs              | 520 simulations   | Blind thrust (San Fernando)  |
+--------------------------+---------------------+-------------------+------------------------------+
| Total Test Partition     | 22 RSNs             | 1,540 simulations | Pooled Clustered Evaluation  |
+--------------------------+---------------------+-------------------+------------------------------+
```

### Primary Reporting Obligations:
1. **Disaggregated Cluster Reporting:** Every primary metric must be reported individually for *Christchurch*, *Morgan Hill*, and *Northridge-01*.
2. **Pooled Clustered Bootstrap:** $B = 2,000$ bootstrap iterations clustered by parent earthquake.
3. **Heterogeneity Metrics:** Cochran's $Q$ and Higgins' $I^2$ reported strictly as secondary descriptive indicators.
4. **Generalization Scope:** Population-level generalization claims are explicitly forbidden.

---

## 3. Pre-Registered Causal Hypothesis Criteria

### Criterion 1: Regime A Slope Concordance
- Evaluated on test records with zero secondary reverse yielding.
- Expected slope: $m_{\text{Regime A}} = 1 - \alpha = +0.980$.
- Pre-registered acceptance:
  $$m_{\text{Regime A}} \in [0.900, 1.050]$$

### Criterion 2: Physical Specificity over Sham Controls
- Evaluated against matched-energy velocity sham (C2.A) and matched-norm latent memory sham (C2.B):
  $$\mathcal{S} = \frac{|\Delta u_{\text{res}}(\text{sham})|}{|\Delta u_{\text{res}}(\text{phys})|} \le 0.20 \quad (\ge 80\%\text{ specificity})$$

### Criterion 3: Regime B Qualitative Ratcheting
- For cyclic multi-yield records, report empirical slope distribution ($m_i$), median, and percentage of positive slopes:
  $$\text{Hypothesis: } \Pr(m_{\text{Regime B}} > 0) \ge 0.75$$

### Criterion 4: Static-Hold Diagnostic Dynamic Divergence
- Dynamic divergence between active recurrence and static state hold:
  $$D_y(t) = \|y_t^{\text{active}} - y_t^{\text{hold}}\|_2$$
  Must correlate with cumulative post-yield seismic input energy: $\text{Corr}(D_y(t), \int_{t_y}^t a_g^2 d\tau) \ge 0.60$.

### Criterion 5: Elastic Purity (C6)
- On unperturbed elastic records ($\mu \le 1.0$):
  $$\max_t |\hat{u}_p(t)| < 10^{-4}\text{ mm}, \quad \hat{E}_{\text{diss}}(T_{\text{end}}) \equiv 0.000\text{ J}$$

---

## 4. Primary vs. Secondary Endpoints

- **Primary Response:** Rel $L_2(u)$ (%) across all 1,540 test records.
- **Primary Causal:** Regime A slope $m$, sham specificity ratio $\mathcal{S}$, and sign consistency.
- **Primary State:** $u_p$ tracking error ($R^2 \ge 0.85$).
- **Secondary:** Peak $u$ error (%), $F_R$ error (%), $E_{\text{diss}}$ error (J), residual drift error (mm), Regime B slope distribution.

**NO ENDPOINT MAY BE PROMOTED OR REORDERED AFTER OBSERVING TEST RESULTS.**
