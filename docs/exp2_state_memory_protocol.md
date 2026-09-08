# EXP 2 Experimental Protocol: Continuous State-Space Operator vs. State-Free Feedforward Baseline

**Experiment Identifier:** EXP-02 (State-Memory Investigation)  
**Primary Research Hypothesis:** **Hypothesis 2 (Mechanism B)** — *Lack of explicit continuous internal state representation causes trajectory drift and phase breakdown in non-smooth elastoplastic dynamics.*  
**Status:** Protocol Specification Complete — Ready for Execution Gate.  

---

## 1. Scientific Motivation & Problem Formulation

### 1.1 The Theoretical Gap Exposed by EXP 1
In EXP 1, standard FNO-1D exhibited **$48.90\%$ future-to-past acausal leakage**, proving that global spectral basis functions violate temporal causality in physical dynamics.  
However, replacing spectral convolutions with a strictly causal finite-impulse-response (FIR) feedforward architecture (Causal TCN) eliminated acausality ($0.000000\%$) but suffered **$92.43\%$ severe yielding error** ($\mu > 4$) and $149.68\%$ overall error.

### 1.2 The Mathematical Cause of Feedforward Failure
In linear elastodynamics, the response is a linear time-invariant convolution with fixed equilibrium origin. In elastoplastic dynamics, yielding produces an **irreversible, time-accumulating plastic offset**:
$$u_p(t) = \int_0^t \dot{u}_p(\tau) \, d\tau, \quad u(t) = u_e(t) + u_p(t)$$
An FIR convolutional kernel approximates response over a finite window of past inputs:
$$u(t) = \sum_{\tau=0}^{R} W(\tau) a_g(t - \tau)$$
Because plastic offset $u_p(t)$ is an indefinite integral that persists for all $t > t_{\text{yield}}$, an FIR model must continuously approximate this offset from input history alone, causing cumulative drift errors during multiple cyclic reversals.

### 1.3 The Continuous State-Space Hypothesis
By representing dynamics via a continuous-time state-space equation:
$$\dot{h}(t) = \mathbf{A} h(t) + \mathbf{B} x(t), \quad y(t) = \mathbf{C} h(t) + \mathbf{D} x(t)$$
where $\mathbf{A} \in \mathbb{R}^{H \times H}$ is parameterized via the **HiPPO (High-Order Polynomial Projection Operators)** matrix, the latent state $h(t)$ acts as an **infinite impulse response (IIR) memory store** capable of continuously tracking the shifting plastic origin $u_p(t)$.

---

## 2. Models Under Comparison (Controlled & Parameter-Matched)

To ensure pure scientific isolation, all 4 models operate on identical inputs, splits, and loss functions:

```
                            PARAMETER MATCHING AUDIT
┌──────────────────────────────────────┬─────────────┬───────────┬──────────────────────────┐
│ Model                                │ Target Params│ Causality │ State Memory Store       │
├──────────────────────────────────────┼─────────────┼───────────┼──────────────────────────┤
│ 1. Standard FNO-1D (Control 1)       │ 1,196,931   │ ❌ Acausal│ ❌ None (Spectral global)│
│ 2. Causal TCN (Control 2)            │ 1,188,763   │ Strict    │ ❌ None (FIR Receptive)  │
│ 3. Continuous S4 SSM (Treatment)     │ 1,192,448   │ Strict    │ ✅ Continuous Latent h(t)│
│ 4. LSTM Baseline (Discrete RNN)      │ 414,851     │ Strict    │ ⚠️ Discrete Hidden (h, c)│
└──────────────────────────────────────┴─────────────┴───────────┴──────────────────────────┘
```

---

## 3. Detailed Architecture Specification for Continuous S4 Operator

```
                    CONTINUOUS S4 STATE-SPACE OPERATOR (1.192M Params)
Input: x [Batch, 10, 2048]
  │
  ├── 1. Linear Input Projection:
  │      • in_features = 10 -> d_model = 128 (with bias)
  │
  ├── 2. Stack of 6 S4 Residual Blocks:
  │      For block l in 1 ... 6:
  │      ┌────────────────────────────────────────────────────────┐
  │      │ LayerNorm(d_model = 128)                               │
  │      │ Continuous S4 Layer (State Dimension N = 64):          │
  │      │   • HiPPO-LegS matrix initialization for A in C^{N}    │
  │      │   • Low-rank NPLR parameterization (A - P Q^*)         │
  │      │   • Discretization: Bilinear (Tustin) with Delta t=0.01│
  │      │   • Convolutional kernel generation:                   │
  │      │       K_bar = (C_bar B_bar, C_bar A_bar B_bar, ...)    │
  │      │   • Causal FFT-based 1D convolution with strict        │
  │      │     left-padding (length 2048)                         │
  │      │ Activation: GELU(x)                                    │
  │      │ Dropout(p = 0.05)                                      │
  │      │ Positionwise Feedforward: Linear(128 -> 256 -> 128)    │
  │      │ Residual Skip Connection: x = x + Block(x)             │
  │      └────────────────────────────────────────────────────────┘
  │
  └── 3. Linear Decoder Head:
         • LayerNorm(128) -> Linear(128 -> 3)
         • Output: [u(t), F_R(t), E_h(t)]
```

---

## 4. Experimental Controls & Anti-Confounding Protocol

| Control Dimension | Implementation Rule |
| :--- | :--- |
| **Data Split** | `held_out_earthquake_split.json` (5,740 Train / 1,120 Val / 1,540 Test across 22 held-out RSN earthquakes). |
| **Input Representation** | Exactly 10 channels: $[a_g(t), T_0, \omega_n, k_0, \zeta, \text{is\_bilinear}, u_y, \alpha, \text{PGA}, \tau_{\text{grid}}]$. |
| **Strict Data Quarantine** | `use_history_channel = False` strictly enforced (zero target displacement feedback). |
| **Normalization** | Channel-wise zero-mean unit-variance fitted strictly on the 5,740 training records. |
| **Optimization Budget** | AdamW optimizer ($\text{lr} = 10^{-3}$, weight decay $10^{-4}$), Cosine Annealing scheduler ($\eta_{\min} = 10^{-6}$), 50 epochs, batch size 32, seed 42. |
| **Loss Function** | Normalized multi-target Relative $L_2$ Loss on $(u, F_R, E_h)$ + regularized energy loss ($\lambda_{\text{energy}} = 0.10$). |

---

## 5. Formal Popperian Falsification Criteria for Mechanism B

```
                      FALSIFICATION & CONFIRMATION CRITERIA
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. CONFIRMATION THRESHOLD (Mechanism B is Supported):                       │
│    • S4 SSM reduces severe post-yield error (\mu > 4) from 92.4% (TCN) to   │
│      < 40.0% Relative L2.                                                   │
│    • S4 SSM reduces residual plastic drift error |u_end - \hat{u}_end| by   │
│      > 40% relative to Causal TCN and > 25% relative to Standard FNO.       │
│    • S4 SSM maintains exact future-perturbation invariance (0.000000% past  │
│      discrepancy at t < t0).                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. FALSIFICATION THRESHOLD (Mechanism B is Refuted):                        │
│    • If S4 SSM achieves severe post-yield error >= 55.0% (matching standard │
│      FNO's 57.7% and Causal TCN's 92.4%), then Mechanism B is FALSIFIED,   │
│      proving that continuous latent state memory alone is insufficient to   │
│      model non-smooth plastic transitions.                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Output Artifacts & Deliverables

1. **Model Checkpoint:** `experiments/exp2_state_memory/checkpoints/best_model.pt`.
2. **Evaluation Metrics Table:** `results/experiments/exp2_state_memory/summary_metrics.csv` (with Earthquake-Clustered Block Bootstrap $95\%$ CIs).
3. **Per-Record Diagnostics:** `results/experiments/exp2_state_memory/test_records_detailed_metrics.csv`.
4. **Causality & Prefix Verification:** `causality_intervention.json` and `prefix_online_causality.json`.
5. **Publication Visualizations (300 DPI PNG + Vector PDF):**
   - `fig1_error_vs_ductility_clustered_ci.png` (log-scale with clustered bootstrap bands)
   - `fig2_residual_drift_distribution.png` (plastic offset error boxplots)
   - `fig3_severe_hysteresis_comparison.png` (OpenSees vs FNO vs TCN vs S4)
   - `fig4_zero_crossing_timing_error.png` (phase alignment diagnostics)
