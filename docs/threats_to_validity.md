# Threats to Scientific & Engineering Validity in SeismoFNO

**Purpose:** Comprehensive risk assessment and methodological limitations document conforming to ACM / IEEE / ASCE peer-review standards.  
**Governing Rule:** A paper that candidly states its scientific threats and experimental bounds is vastly more credible than one presenting ungrounded universal claims.

---

## 1. Construct Validity (Are We Measuring the True Physical Phenomena?)

### 1.1 The Elastic Denominator Artifact in Relative $L_2$ Loss
- **Threat:** In the linear elastic regime ($\mu \le 1$), the ground-truth displacement $u(t)$ has a small amplitude ($1 - 4\text{ mm}$), making the relative $L_2$ denominator $\|u_{\text{gt}}\|_2$ small. A minor high-frequency phase drift produces a misleadingly massive percentage error ($384.4\%$).
- **Mitigation:** In all reports and papers, relative $L_2$ percentages must be accompanied by **Absolute Peak Error ($|u_{\max} - \hat{u}_{\max}|$ in mm)** and **Normalized Displacement Error ($\Delta u / u_y$)**.

### 1.2 Breakdown of the Hilbert Transform Under Non-Zero-Mean Plastic Drift
- **Threat:** In linear vibration, the Hilbert transform $\tilde{u}(t) = \mathcal{H}\{u\}(t)$ yields an instantaneous phase $\theta(t) = \arctan(\tilde{u}/u)$. However, plastic yielding produces a permanent monotonic drift $u_p(t) \ne 0$. The Bedrosian theorem breaks down for non-zero-mean signals, creating artificial phase singularities.
- **Mitigation:** Hilbert phase error is restricted strictly to zero-mean elastic signals. For elastoplastic signals, phase alignment is measured via **Zero-Crossing Timing Lag ($\Delta t_{\text{zero}}$)** and **Dynamic Time Warping (DTW)**.

### 1.3 Ground-Truth Simulator Numerical Accuracy
- **Threat:** If OpenSeesPy NLTHA numerical integration diverges or accumulates step-truncation errors, the neural surrogate learns solver artifacts rather than continuum physics.
- **Mitigation:** All OpenSeesPy simulations are cross-validated against an independent pure NumPy Newmark-$\beta$ solver with Newton-Raphson equilibrium iterations ($100\%$ unit test pass rate in `tests/test_energy_conservation.py`).

---

## 2. Internal Validity (Are the Experimental Comparisons Unconfounded?)

### 2.1 Target Leakage in History Augmentation
- **Threat:** Incorporating ground-truth displacement history $u(t)$ into input channels (as done in legacy Phase 5D) provides direct target leakage.
- **Mitigation:** The flag `use_history_channel = False` is hardcoded across all active configs. Model inputs are strictly restricted to foundation ground acceleration $a_g(t)$ and known structural properties ($T_0, \zeta, u_y, \alpha$).

### 2.2 Architectural Confounding (Causality vs. Representation Basis)
- **Threat:** Comparing Standard FNO (Global Fourier) with Causal TCN (Dilated 1D Conv) changes both causality AND the basis function (continuous global harmonics vs discrete local FIR taps).
- **Mitigation:** In EXP 2, the Continuous State-Space Model (S4) isolates state memory by matching continuous operator representation and strict left-padded causality while varying only the recursive state formulation ($h_t$).

### 2.3 Parameter Count Mismatches
- **Threat:** A model with $3\times$ more parameters could outperform baselines purely through brute-force memorization capacity.
- **Mitigation:** Strict parameter matching to within **$\pm 1.5\%$** ($1.19\text{M} \pm 18\text{k}$ parameters) across FNO, Causal TCN, and S4 SSM.

---

## 3. Statistical Conclusion Validity (Are the Inferences Statistically Sound?)

### 3.1 Intra-Earthquake Correlation (Pseudo-Replication)
- **Threat:** Standard bootstrap randomly resamples individual simulation records independently. Because multiple simulations share the same earthquake ground motion $a_g(t)$ (varying only $T_n$ and $u_y$), standard bootstrap underestimates confidence interval widths due to unmodeled intra-event covariance.
- **Mitigation:** All statistical tests use **Earthquake-Clustered Block Bootstrap** (resampling entire earthquake RSN events with replacement across 2,000 iterations).

### 3.2 Sample Size & Power
- **Status:** Evaluation is conducted across 1,540 test records generated from 22 distinct held-out earthquakes, providing high statistical power ($> 99\%$ power to detect a $5\%$ effect size at $\alpha = 0.01$).

---

## 4. External Validity (What Are the Real-World Engineering Limits?)

### 4.1 Ground Motion Domain Bounds
- **Threat:** All training records are drawn from shallow crustal earthquakes in the PEER NGA-West2 database ($M_w 5.5 - 7.9$, $R_{\text{rup}} 5 - 70\text{ km}$).
- **Limitation:** The surrogate is not validated on subduction zone mega-thrust earthquakes (e.g. 2011 Tohoku, durations $> 120\text{ s}$) or induced seismicity with high-frequency content ($> 25\text{ Hz}$).

### 4.2 Structural System Complexity
- **Threat:** Bilinear / elastoplastic SDOF systems do not exhibit strength deterioration, pinching hysteresis, or multi-mode frame interactions.
- **Limitation:** SDOF is an analytical laboratory for operator learning. Application to multi-story buildings requires the MDOF extension outlined in Phase 8.

### 4.3 Resolution Invariance Claims
- **Threat:** Standard FNO and S4 are claimed to be continuous operators, but evaluations are conducted at $\Delta t = 0.01\text{ s}$.
- **Limitation:** Continuous operator property remains a theoretical hypothesis until verified empirically in EXP 6 across sampling rates $\Delta t \in \{0.04\text{ s}, 0.02\text{ s}, 0.01\text{ s}, 0.005\text{ s}\}$.
