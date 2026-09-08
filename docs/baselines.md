# Baseline Models & Reference Solvers Specification

This document details the exact mathematical formulations, network architectures, parameter budgets, and causality protocols for all 8 reference solvers and baseline models in the SeismoFNO benchmarking battery.

---

## 1. Summary of Model Portfolio

| Model / Solver | Model Type | Causality Protocol | Parameter Budget | Primary Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **1. Analytical Reference** | Closed-Form Exact | Causal ($t \ge 0$) | 0 (Exact Math) | Duhamel Integral / Damped Sinusoid |
| **2. Independent Newmark** | Vectorized Integrator | Causal Step-by-Step | 0 (Numerical) | Average Accel ($\beta=0.25$) + Newton-Raphson |
| **3. OpenSeesPy Reference** | C++ FE Engine | Causal Step-by-Step | 0 (Numerical) | Fiber Beam-Column / Uniaxial Steel01 |
| **4. Standard FNO 1D** | Global Spectral NO | Acausal (Global FFT) | $1,196,931$ | 4 Spectral Blocks, 128 Modes, Width 48 |
| **5. Causal Dilated TCN** | Dilated 1D Conv | Strictly Causal ($t \le \tau$) | $1,024,515$ | 11 Residual Blocks ($d = 2^0 \dots 2^{10}$), Left Pad |
| **6. Multi-Layer LSTM** | Recurrent Sequence | Causal (Hidden State) | $414,851$ | 3 LSTM Layers, Hidden Dim 128 |
| **7. Deep Residual MLP** | Pointwise Feedforward | Acausal (Pointwise) | $564,995$ | 4 Linear Layers, Width 256, Skip Connections |
| **8. Structured State-Space (S4)** | Continuous SSM | Strictly Causal | $1,085,200$ | HiPPO State Matrices, Bilinear Discretization |
| **9. Chopra CSM** | Equivalent Linear | Semi-empirical Spectral | 0 (Analytical) | Effective Period $T_{\text{eff}}$ & Damping $\zeta_{\text{eff}}$ |

---

## 2. Detailed Mathematical Formulations

### 2.1 Independent Nonlinear Newmark-$\beta$ Solver
- **Module:** [`src/ground_truth/independent_solvers.py`](file:///Users/rahul/seismoFNO/src/ground_truth/independent_solvers.py)
- **Method:** Constant average acceleration method ($\gamma = 0.5, \beta = 0.25$).
- **Equilibrium Iteration:** At each time step $i$, trial displacement $u^{(k+1)} = u^{(k)} + \Delta u$ is iterated via Newton-Raphson:
  $$r^{(k)} = \hat{p}_i - F_R\left(u^{(k)}, z^{(k)}\right) - a_1 u^{(k)}$$
  $$\Delta u = \frac{r^{(k)}}{k_t^{(k)} + a_1}$$
  where $a_1 = \frac{m}{\beta \Delta t^2} + \frac{\gamma c}{\beta \Delta t}$, until residual $\|r\| < 10^{-10}\text{ N}$.
- **Zero-Dependency Guarantee:** Completely implemented in pure NumPy with zero OpenSees dependencies, providing a verified independent cross-check.

---

### 2.2 Standard 1D Fourier Neural Operator (FNO-1D)
- **Module:** [`src/models/fno1d.py`](file:///Users/rahul/seismoFNO/src/models/fno1d.py)
- **Architecture:**
  - Lifting Layer: $P: \mathbb{R}^{10} \to \mathbb{R}^{48}$
  - 4 Spectral Convolution Blocks with 128 modes:
    $$h_{l+1}(t) = \text{GELU}\left( W_l h_l(t) + \mathcal{F}^{-1} \left[ R_l(k) \cdot \mathcal{F}[h_l](k) \right] \right)$$
  - Projection Layer: $Q: \mathbb{R}^{48} \to \mathbb{R}^{3}$ predicting $(u(t), F_R(t), E_h(t))$.
- **Causality Classification:** **Acausal**. The discrete real FFT computes global integrals spanning $[0, T]$ simultaneously.

---

### 2.3 Causal Dilated Temporal Convolutional Network (Causal TCN)
- **Module:** [`src/models/causal_tcn.py`](file:///Users/rahul/seismoFNO/src/models/causal_tcn.py)
- **Architecture:**
  - 11 residual dilated blocks with dilation factors $d \in \{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024\}$.
  - Strictly left-padded: `pad_left = (kernel_size - 1) * dilation`, `pad_right = 0`.
  - Receptive field:
    $$\text{RF} = 1 + \sum_{l=0}^{10} 2 \cdot (3 - 1) \cdot 2^l = 1 + 4 \cdot (2^{11} - 1) = 8,189\text{ time steps} > 2,048$$
- **Causality Classification:** **Strictly Causal**. Output at time $t$ depends exclusively on inputs at $\tau \le t$.

---

### 2.4 Structured State-Space Model (S4 / S4D Baseline)
- **Mathematical Principle:** Parameterizes the continuous-time linear differential system:
  $$\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t), \quad \mathbf{y}(t) = \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t)$$
  using Diagonal State Spaces (S4D) with HiPPO initialization to preserve long-range memory.
- **Discretization:** Bilinear (Tustin) transformation evaluated in parallel during training via FFT convolution, and step-by-step during inference.
- **Causality Classification:** **Strictly Causal**.

---

### 2.5 Multi-Layer Recurrent LSTM Baseline
- **Module:** [`src/models/lstm_baseline.py`](file:///Users/rahul/seismoFNO/src/models/lstm_baseline.py)
- **Architecture:** 3 LSTM layers with hidden dimension $d = 128$, input feature lifting, and linear projection head.
- **Causality Classification:** **Strictly Causal**, but computationally sequential ($O(T)$ steps).

---

### 2.6 Chopra Capacity Spectrum Method (Equivalent Linearization)
- **Module:** [`src/ground_truth/independent_solvers.py`](file:///Users/rahul/seismoFNO/src/ground_truth/independent_solvers.py)
- **Method:** Estimates peak nonlinear displacement demand $u_{\max}$ using effective elongated period $T_{\text{eff}}$ and hysteretic damping $\zeta_{\text{eff}}$:
  $$T_{\text{eff}} = T_0 \sqrt{\frac{\mu}{1 + \alpha(\mu - 1)}}, \quad \zeta_{\text{eff}} = \zeta_0 + \frac{2}{\pi} \frac{(\mu - 1)(1 - \alpha)}{\mu (1 + \alpha\mu - \alpha)}$$
- **Causality Classification:** Semi-empirical spectral method (envelope estimate only; no time history).
