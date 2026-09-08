# Comprehensive Evaluation Metrics Specification

This document provides the mathematical definitions, units, and engineering interpretations for all 13 evaluation metrics used to assess structural response surrogates in SeismoFNO.

---

## 1. Trajectory-Level Relative Errors

### 1.1 Relative $L_2$ Bochner Norm Error
Measures global waveform fidelity over the full sequence $N = 2,048$:
$$\text{Rel } L_2(y, \hat{y}) = \frac{\|y - \hat{y}\|_2}{\|y\|_2 + \epsilon} \times 100\% = \frac{\sqrt{\sum_{n=1}^N (y_n - \hat{y}_n)^2}}{\sqrt{\sum_{n=1}^N y_n^2} + \epsilon} \times 100\%$$
- **Evaluated on:**
  - Displacement $u(t)$ $[\%]$
  - Velocity $v(t) = \dot{u}(t)$ $[\%]$
  - Acceleration $a(t) = \ddot{u}(t)$ $[\%]$
  - Nonlinear Restoring Force $F_R(t)$ $[\%]$

---

## 2. Engineering Peak Demand Metrics

### 2.1 Peak Relative Displacement Error
Measures accuracy on maximum drift demand $u_{\max} = \max_t |u(t)|$, the critical parameter governing building damage states:
$$\text{Err}_{u_{\max}} = \frac{|\max_t |\hat{u}(t)| - \max_t |u(t)||}{\max_t |u(t)| + \epsilon} \times 100\%$$

### 2.2 Peak Restoring Force Error
Measures maximum base shear demand $F_{R,\max} = \max_t |F_R(t)|$:
$$\text{Err}_{F_{R,\max}} = \frac{|\max_t |\hat{F}_R(t)| - \max_t |F_R(t)||}{\max_t |F_R(t)| + \epsilon} \times 100\%$$

### 2.3 Residual Plastic Drift Error
Measures permanent plastic offset at the end of ground motion excitation ($t = T_{\text{end}}$):
$$\text{Err}_{u_{\text{residual}}} = |\hat{u}(T_{\text{end}}) - u(T_{\text{end}})| \quad [\text{mm}]$$

---

## 3. Dynamic Phase & Bifurcation Metrics

### 3.1 Yield Onset Timing Error
Measures the temporal discrepancy in predicting the exact instant of initial plastic yielding $t_y = \inf \{t : |u(t)| \ge u_y\}$:
$$\Delta t_{\text{yield}} = |\hat{t}_{\text{yield}} - t_{\text{yield}}| \quad [\text{ms}]$$

### 3.2 Instantaneous Phase Coherence Error
Measures phase lag/lead across time via the Hilbert transform $\mathcal{H}[\cdot]$:
$$\phi(t) = \arctan\left(\frac{\mathcal{H}[u(t)]}{u(t)}\right), \quad \Delta \Phi = \frac{1}{T} \int_0^T |\hat{\phi}(t) - \phi(t)| dt \quad [\text{rad}]$$

---

## 4. Hysteretic & Thermodynamic Metrics

### 4.1 Hysteresis Loop Dissipation Area Discrepancy
Measures error in total energy enclosed by cyclic loops in the $F_R - u$ plane:
$$A_{\text{loop}} = \oint F_R du, \quad \text{Err}_{A_{\text{loop}}} = \frac{|\hat{A}_{\text{loop}} - A_{\text{loop}}|}{A_{\text{loop}} + \epsilon} \times 100\%$$

### 4.2 Scale-Regularized Cumulative Absorbed Energy Error
Measures error in the cumulative work curve $E_h(t) = \int_0^t F_R du$, normalized against total seismic input energy $E_i(t)$:
$$\text{Err}_{E_h} = \frac{\|\hat{E}_h - E_h\|_2}{\max(\|E_h\|_2, \|E_i\|_2, 1.0\text{ J})} \times 100\%$$

### 4.3 Total Dynamic Energy Balance Residual
Measures violation of the conservation law $E_k(t) + E_d(t) + E_s(t) + E_h(t) = E_i(t)$:
$$\mathcal{R}_{\text{energy}}(t) = |E_k(t) + E_d(t) + E_s(t) + E_h(t) - E_i(t)| \quad [\text{J}]$$

---

## 5. Computational & Profiling Metrics

| Metric | Definition & Protocol | Target Unit |
| :--- | :--- | :--- |
| **Inference Latency (Batch 1)** | Wall-clock time per single record forward pass on standard CPU | Milliseconds ($ms$) |
| **Batch Throughput** | Evaluations per second with batch size $B=64$ on GPU | Records / sec ($rec/s$) |
| **Peak VRAM / Memory** | Max memory allocated during forward inference | Megabytes ($MB$) |
| **Trainable Parameters** | Total non-frozen weights in architecture | Count ($M$) |
