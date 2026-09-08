# Formal Research Question & Scientific Problem Formulation

---

## 1. Primary Research Question

> **"Why do global Fourier Neural Operators suffer systematic accuracy breakdown under non-stationary elastoplastic yielding, and can strict temporal causality and explicit state representations improve hysteretic response prediction while preserving computational efficiency and temporal-resolution robustness?"**

---

## 2. Theoretical Motivation & Problem Statement

### 2.1 The Linear-Elastic Success vs. Inelastic Breakdown
In linear elastodynamics, the governing equation of motion for a Single-Degree-of-Freedom (SDOF) or Multi-Degree-of-Freedom (MDOF) system subjected to earthquake ground acceleration $a_g(t)$ is a **Linear Time-Invariant (LTI)** differential operator:
$$\mathcal{L} u(t) = m \ddot{u}(t) + c \dot{u}(t) + k_0 u(t) = -m a_g(t)$$

Because complex harmonic sinusoids $e^{i \omega t}$ are **exact eigenfunctions** of any LTI differential operator, the Fourier transform diagonalizes $\mathcal{L}$:
$$\mathcal{F}\{\mathcal{L} u\}(\omega) = \left( -m \omega^2 + i c \omega + k_0 \right) \hat{u}(\omega) = -m \hat{a}_g(\omega)$$
$$\hat{u}(\omega) = H(\omega) \hat{a}_g(\omega), \quad \text{where } H(\omega) = \frac{-m}{-m\omega^2 + ic\omega + k_0}$$

Standard Fourier Neural Operators (FNOs) learn this continuous frequency-domain transfer multiplier $H(\omega)$ with extreme accuracy, achieving **$< 2.7\%$ relative $L_2$ error** on linear-elastic records.

---

### 2.2 The Inelastic Bifurcation & Breakdown of Time-Translation Invariance
When a structural element yields ($|u(t)| > u_y$), the restoring force $F_R(u(t), z(t))$ undergoes an **irreversible, non-smooth bifurcation**:
$$F_R(t) = \alpha k_0 u(t) + (1 - \alpha) k_0 z(t)$$
$$\dot{z}(t) = \dot{u}(t) \left[ 1 - \text{rect}\left(\frac{z(t)}{u_y}\right) \text{sign}(z(t) \dot{u}(t)) \right]$$

This transition produces three fundamental challenges:
1. **Broken Time-Translation Invariance:** The tangent stiffness $k_t(t)$ drops instantaneously from $k_0 \to \alpha k_0$. The natural period elongates by a factor of $\sqrt{1/\alpha}$ (e.g., $4.47\times$ for $\alpha = 0.05$).
2. **Non-Markovian Memory & Path-Dependence:** Unloading from a plastic excursion occurs from a permanently shifted plastic displacement $u_p(t) = \int_0^t \dot{u}_p(\tau) d\tau$. The restoring force cannot be determined from instantaneous kinematic variables $(u(t), \dot{u}(t))$ alone.
3. **Global Fourier Contamination:** The discrete Fourier transform $\hat{u}(k) = \sum_{n=0}^{N-1} u(t_n) e^{-i 2\pi k n / N}$ integrates over the entire duration $[0, T]$ simultaneously. Post-yield low-frequency content leaks into pre-yield time steps, creating non-causal precursor oscillations and accumulating phase lag across hysteretic cycles.

---

## 3. Disentangling Competing Failure Hypotheses

An engineering reviewer will rightly demand proof that FNO's $\sim 57\%$ post-yield error is specifically caused by acausality, rather than competing confounding mechanisms. We explicitly formulate and isolate **five distinct candidate hypotheses**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             CANDIDATE FAILURE MECHANISMS IN SPECTRAL OPERATORS              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Mechanism A: Fourier / Acausal Temporal Mixing                              │
│   Global FFT convolves future post-yield states into past pre-yield states. │
├─────────────────────────────────────────────────────────────────────────────┤
│ Mechanism B: Insufficient Representation of Internal State                  │
│   Model lacks an explicit internal state variable z(t) to track plastic     │
│   offset u_p, forcing the network to infer memory from input history.       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Mechanism C: Spectral Truncation / High-Frequency Loss                      │
│   Mode truncation at cutoff K removes high-frequency harmonics generated    │
│   by sharp non-smooth yielding slope discontinuities.                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Mechanism D: Inadequate Structural Parameter Conditioning                   │
│   Constant scalar channels (T_n, u_y, α) provide insufficient modulation   │
│   of spectral weights across diverse structural frequencies.                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Mechanism E: Optimization & Loss Landscape Bottlenecks                      │
│   L2 loss gradients are dominated by high-energy elastic coda, causing the  │
│   optimizer to under-weight the sharp, localized yield transition.          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Expected Impact of Resolving this Question

1. **For Scientific Machine Learning (SciML):** Establishes clear mathematical boundaries on where global spectral operators are theoretically valid vs. where causal state-space representations are mathematically required in non-smooth, path-dependent continuum mechanics.
2. **For Structural Earthquake Engineering:** Provides a validated, physics-consistent, sub-millisecond surrogate capable of evaluating nonlinear seismic response across regional building portfolios without sacrificing hysteretic energy dissipation or peak displacement accuracy.
