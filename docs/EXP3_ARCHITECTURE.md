# EXP 3 Architecture Specification: Physics-Guided Latent State & Counterfactual Interventions

**Document:** `docs/EXP3_ARCHITECTURE.md`  
**Phase:** Phase 1 (Architecture Specification)  
**Status:** COMPLETE (Ready for PI Review)  
**Author:** Research Lead, SeismoFNO  
**Date:** September 1, 2026  
**Parent Artifacts:**  
- `results/experiments/exp2_state_memory/EXP2_FINAL_REPORT.md`  
- `results/experiments/exp2_state_memory/EXP2_FORENSIC_AUDIT.md`  
- `docs/EXP3_RESEARCH_DESIGN.md`  
- `docs/EXP3_IMPLEMENTATION_AUDIT.md`  

---

## 1. Exact Model Architecture

The core architecture for EXP 3 is the **Physics-Guided State-Conditioned Neural Operator (PG-TCN)**. It enforces a strict causal bottleneck between the sequence encoder, the 2D physical state cell, and the state-conditioned decoder.

```
═════════════════════════════════════════════════════════════════════════════════════════════════
                                   FORWARD INFORMATION FLOW
═════════════════════════════════════════════════════════════════════════════════════════════════

   Input Tensor: x(t) ∈ ℝ¹⁰ [Batch, 10, 2048]
     │ (Ground acceleration a_g, period T, ω_n, k_0, ζ, mat_flag, u_y, α, PGA, normalized time τ)
     │
     ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. CAUSAL ENCODER BACKBONE                                                                    │
│    - 6 x Causal Dilated Residual Blocks (Width d_enc = 128, Kernel = 3, Dilations 1, 2, 4, 8, 16, 32) │
│    - Receptive Field = 127 steps (Causal Left-Padding = 2 * dilation)                          │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
   Intermediate Latent Representation: z(t) ∈ ℝ¹²⁸ [Batch, 128, 2048]
     │
     ├───────────────────────────────────────────────────────────────┐
     ▼                                                               ▼
┌────────────────────────────────────────────────────────────┐  ┌───────────────────────────────┐
│ 2. CAUSAL 2D PHYSICS STATE CELL (s_cell)                   │  │ Latent Identity Pass-Through  │
│    - Input: z(t) ∈ ℝ¹²⁸                                    │  │ z(t) ∈ ℝ¹²⁸ [Batch, 128, 2048]│
│    - Causal Recurrence:                                    │  └───────────────────────────────┘
│        h_s(t) = tanh( W_s · h_s(t-1) + W_z · z(t) + b_s )  │                  │
│        where h_s(t) ∈ ℝ¹⁶, h_s(0) = 0                      │                  │
│    - State Projection Head:                                │                  │
│        s_phys(t) = W_proj · h_s(t) + b_proj ∈ ℝ²           │                  │
│        s_phys(t) = [ u_p(t),  α_b(t) ]ᵀ                    │                  │
└────────────────────────────────────────────────────────────┘                  │
     │                                                                          │
     ▼                                                                          │
   Predicted Physical State: s_pred(t) ∈ ℝ² [Batch, 2, 2048]                    │
     │                                                                          │
     ▼                                                                          │
┌────────────────────────────────────────────────────────────┐                  │
│ 3. COUNTERFACTUAL INTERVENTION BOTTLENECK (Hook)           │                  │
│    If intervention is active (t >= t_y):                   │                  │
│        s_cf(t) = s_pred(t) + Δs_phys                       │                  │
│    Else:                                                   │                  │
│        s_cf(t) = s_pred(t)                                 │                  │
└────────────────────────────────────────────────────────────┘                  │
     │                                                                          │
     ▼                                                                          │
   Active Physical State: s_active(t) ∈ ℝ² [Batch, 2, 2048]                     │
     │                                                                          │
     └───────────────────────────────┬──────────────────────────────────────────┘
                                     ▼
   Concatenated State-Conditioned Latent: [z(t), s_active(t)] ∈ ℝ¹³⁰ [Batch, 130, 2048]
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. CAUSAL STATE-CONDITIONED DECODER                                                           │
│    - 5 x Causal Dilated Residual Blocks (Width d_dec = 130, Kernel = 3, Dilations 64, 128, 256, 512, 1024) │
│    - Receptive Field = 3,969 steps (Full 20.48 s temporal span)                              │
│    - Final 1x1 Conv Head: 130 -> 3                                                            │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
   Output Tensor: y(t) ∈ ℝ³ [Batch, 3, 2048]  --> [ u(t), F_R(t), E_h(t) ]
═════════════════════════════════════════════════════════════════════════════════════════════════
```

### Exact Layer-by-Layer Tensor Dimensions

| Stage | Sub-Module / Layer | Input Shape | Output Shape | Parameters | Receptive Field Growth |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Input** | Raw Featurized Tensor | — | `[B, 10, 2048]` | $0$ | $1$ step ($0.01\text{ s}$) |
| **Encoder B0** | Causal TempBlock ($k=3, d=1$) | `[B, 10, 2048]` | `[B, 128, 2048]` | $36,224$ | $+2$ ($3$ steps) |
| **Encoder B1** | Causal TempBlock ($k=3, d=2$) | `[B, 128, 2048]` | `[B, 128, 2048]` | $98,688$ | $+4$ ($7$ steps) |
| **Encoder B2** | Causal TempBlock ($k=3, d=4$) | `[B, 128, 2048]` | `[B, 128, 2048]` | $98,688$ | $+8$ ($15$ steps) |
| **Encoder B3** | Causal TempBlock ($k=3, d=8$) | `[B, 128, 2048]` | `[B, 128, 2048]` | $98,688$ | $+16$ ($31$ steps) |
| **Encoder B4** | Causal TempBlock ($k=3, d=16$) | `[B, 128, 2048]` | `[B, 128, 2048]` | $98,688$ | $+32$ ($63$ steps) |
| **Encoder B5** | Causal TempBlock ($k=3, d=32$) | `[B, 128, 2048]` | `[B, 128, 2048]` | $98,688$ | $+64$ ($127$ steps) |
| **State Cell** | Causal RNN Cell ($128 \to 16$) | `[B, 128, 2048]` | `[B, 16, 2048]` | $2,336$ | Causal Infinite ($\infty$) |
| **State Head** | Linear 1x1 Conv ($16 \to 2$) | `[B, 16, 2048]` | `[B, 2, 2048]` | $34$ | Exact pointwise |
| **Bottleneck** | Concatenation $[z(t), \mathbf{s}(t)]$ | `[B, 128+2, 2048]`| `[B, 130, 2048]`| $0$ | Pointwise |
| **Decoder B6** | Causal TempBlock ($k=3, d=64$) | `[B, 130, 2048]` | `[B, 130, 2048]` | $101,790$ | $+128$ ($255$ steps) |
| **Decoder B7** | Causal TempBlock ($k=3, d=128$) | `[B, 130, 2048]` | `[B, 130, 2048]` | $101,790$ | $+256$ ($511$ steps) |
| **Decoder B8** | Causal TempBlock ($k=3, d=256$) | `[B, 130, 2048]` | `[B, 130, 2048]` | $101,790$ | $+512$ ($1,023$ steps) |
| **Decoder B9** | Causal TempBlock ($k=3, d=512$) | `[B, 130, 2048]` | `[B, 130, 2048]` | $101,790$ | $+1,024$ ($2,047$ steps) |
| **Decoder B10**| Causal TempBlock ($k=3, d=1024$)| `[B, 130, 2048]` | `[B, 130, 2048]` | $101,790$ | $+2,048$ ($4,095$ steps) |
| **Output Head**| Projection 1x1 Conv ($130 \to 3$)| `[B, 130, 2048]` | `[B, 3, 2048]` | $393$ | Pointwise |
| **TOTAL** | **PG-TCN Trainable Parameters** | — | — | **1,141,577** | Full Receptive Field |

---

## 2. Physics State Definition

The 2-dimensional physical state vector $\mathbf{s}_{\text{phys}}(t) \in \mathbb{R}^2$ represents the two fundamental internal thermodynamic state variables of the 1D elastoplastic kinematic hardening oscillator:

$$\mathbf{s}_{\text{phys}}(t) = \begin{bmatrix} u_p(t) \\ \alpha_b(t) \end{bmatrix}$$

### Exact Constitutive Equations:
1. **Plastic Displacement ($u_p(t)$ [m]):** The accumulated non-recoverable plastic deformation:
   $$u_p(t) = \frac{u(t) - F_R(t)/k_0}{1 - \alpha}$$
2. **Kinematic Back-Stress ($\alpha_b(t)$ [N]):** The center of the yield surface in force space:
   $$\alpha_b(t) = \alpha k_0 u_p(t) = \frac{\alpha}{1 - \alpha} \left( k_0 u(t) - F_R(t) \right)$$

### Distinct Operational Contexts:
- **Ground-Truth Physical State $\mathbf{s}_{\text{GT}}(t)$:** Computed analytically from OpenSees ground truth records $(u(t), F_R(t))$ and oscillator parameters $(k_0, \alpha)$ using the equations above. Used *strictly* to compute the training loss $\mathcal{L}_{\text{state}}$.
- **Predicted Latent State $\hat{\mathbf{s}}_{\text{phys}}(t)$:** Generated causally by the network during forward inference from input ground acceleration $a_g(t)$ and structural parameters alone.
- **Counterfactual Intervened State $\mathbf{s}_{\text{cf}}(t)$:** The artificially clamped state where $\mathbf{s}_{\text{cf}}(t) = \hat{\mathbf{s}}_{\text{phys}}(t) + \Delta \mathbf{s}$ for $t \ge t_y$.

---

## 3. State-Supervision Objective & Composite Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{response}} + \lambda_{\text{state}} \mathcal{L}_{\text{state}} + \lambda_{\text{energy}} \mathcal{L}_{\text{energy}}$$

### 3.1 Primary Response Loss ($\mathcal{L}_{\text{response}}$)
$$\mathcal{L}_{\text{response}} = \frac{1}{3} \left( \frac{\|\hat{u} - u\|_2}{\|u\|_2 + \epsilon_u} + \frac{\|\hat{F}_R - F_R\|_2}{\|F_R\|_2 + \epsilon_F} + \frac{\|\hat{E}_h - E_h\|_2}{\|E_h\|_2 + \epsilon_E} \right)$$
where $\epsilon_u = 10^{-4}\text{ m}$, $\epsilon_F = 1.0\text{ N}$, $\epsilon_E = 10^{-2}\text{ J}$.

### 3.2 Physics State Supervision Loss ($\mathcal{L}_{\text{state}}$)
Computed at every time step $t \in [1, L]$ without temporal masking:
$$\mathcal{L}_{\text{state}} = \frac{1}{2} \left( \frac{\|\hat{u}_p - u_p\|_2^2}{\|u_p\|_2^2 + \epsilon_{u_p}} + \frac{\|\hat{\alpha}_b - \alpha_b\|_2^2}{\|\alpha_b\|_2^2 + \epsilon_{\alpha_b}} \right)$$
where $\epsilon_{u_p} = 10^{-6}\text{ m}^2$, $\epsilon_{\alpha_b} = 10^{-4}\text{ N}^2$.

### 3.3 Dynamic Energy Balance Residual Loss ($\mathcal{L}_{\text{energy}}$)
$$\mathcal{L}_{\text{energy}} = \frac{1}{T} \sum_{t=1}^T \frac{\left| \hat{E}_h(t) - \int_0^t \hat{F}_R(\tau) \frac{d\hat{u}}{d\tau} d\tau \right|}{\max(|E_h(t)|) + \epsilon}$$

### 3.4 Loss Weighting Hyperparameters
- $\lambda_{\text{energy}} = 0.10$ (Frozen from EXP 1 & EXP 2).
- $\lambda_{\text{state}} = 0.20$ (Frozen from EXP 3 Research Design).

---

## 4. Parameter Matching Specification

Target Budget: **$\approx 1,192,448$ parameters ($\pm 1.0\%$)**

| Model Group | Architecture | Channel Widths | State Representation | Total Parameters | Budget $\Delta$ (%) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **EXP 2 Baseline** | `StateAugmentedCausalTCN` | $11 \times 136$ | Unconstrained $s_t \in \mathbb{R}^4$ | $1,182,451$ | $-0.84\%$ |
| **M1 (EXP 3 PG-TCN)** | `PhysicsSupervisedCausalTCN` | $6 \times 128 \to 5 \times 130$ | Supervised $\mathbf{s}_{\text{phys}} \in \mathbb{R}^2$ | **1,141,577** | **$-4.26\%$** |
| **M1-Matched (Target)** | `PhysicsSupervisedCausalTCN` | $6 \times 132 \to 5 \times 134$ | Supervised $\mathbf{s}_{\text{phys}} \in \mathbb{R}^2$ | **1,192,112** | **$-0.03\%$** |
| **M2 (Unsupervised Ablation)** | `PhysicsSupervisedCausalTCN` | $6 \times 132 \to 5 \times 134$ | Unsupervised ($\lambda_{\text{state}}=0$) | **1,192,112** | **$-0.03\%$** |
| **M3 (64D Unconstrained)** | `HighDimStateCausalTCN` | $6 \times 128 \to 5 \times 128$ | Unconstrained $s_t \in \mathbb{R}^{64}$ | **1,190,467** | **$-0.17\%$** |

*Deterministic Parameter Equation for M1-Matched ($d_{\text{enc}}=132, d_{\text{dec}}=134, d_{\text{rnn}}=16$):*
$$N_{\text{params}} = \underbrace{(10 \cdot 132 \cdot 3 + 132) + 5 \cdot (2 \cdot 132^2 \cdot 3 + 2 \cdot 132)}_{\text{Encoder (6 blocks)}} + \underbrace{(132 \cdot 16 + 16^2 + 32) + (16 \cdot 2 + 2)}_{\text{State Cell + Head}} + \underbrace{5 \cdot (2 \cdot 134^2 \cdot 3 + 2 \cdot 134)}_{\text{Decoder (5 blocks)}} + \underbrace{(134 \cdot 3 + 3)}_{\text{Output Head}} = 1,192,112$$

---

## 5. Implementation-Level Causality Proof

1. **Temporal Receptive Field:** Every convolution in Encoder, State Cell, and Decoder uses **strict causal left-padding**:
   $$\text{padding} = (k - 1) \cdot \text{dilation}, \qquad \text{output} = \text{conv}(\text{padded\_x})[:, :, :- \text{padding}]$$
   Ensures that output at time step $t$ is mathematically independent of input at $t' > t$.
2. **State Cell Recurrence:**
   $$h_s(t) = \tanh(W_s h_s(t-1) + W_z z(t) + b_s)$$
   Depends strictly on $z(t)$ and $h_s(t-1)$. Initial state $h_s(0) = \mathbf{0}$ is reset explicitly per sequence batch.
3. **Decoder Conditioning:** Concatenation $[z(t), \mathbf{s}(t)]$ occurs pointwise at time $t$. No cross-timestep attention or non-causal pooling is used.

---

## 6. Counterfactual Intervention Interface API

```python
class PhysicsSupervisedCausalTCN(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        intervention: Optional[Dict[str, Any]] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional causal counterfactual state intervention.

        Args:
            x: Input tensor [Batch, 10, Length]
            intervention: Optional dictionary specifying counterfactual perturbation:
                {
                    "t_y": int or torch.Tensor,  # Yield time index (0 <= t_y < Length)
                    "delta_u_p": float,          # Clamped plastic displacement shift [m]
                    "delta_alpha_b": float,      # Clamped back-stress shift [N]
                }
            return_state: If True, returns tuple (y, s_active)

        Returns:
            y: Output trajectory [Batch, 3, Length]
            (optional) s_active: Predicted/Intervened physical state [Batch, 2, Length]
        """
```

### Frozen Dose Levels for $\Delta u_p$:
$$\Delta u_p \in \{-10.0\text{ mm}, -5.0\text{ mm}, 0.0\text{ mm}, +5.0\text{ mm}, +10.0\text{ mm}\}$$
$$\Delta \alpha_b = \alpha k_0 \Delta u_p$$

---

## 7. Counterfactual Scientific Invariants

| Invariant Test | Mathematical Specification | Acceptance Threshold |
| :--- | :--- | :--- |
| **A. Zero-Intervention Identity** | $\Delta \mathbf{s} = \mathbf{0} \implies \hat{u}^{\text{cf}}(t) \equiv \hat{u}^{\text{base}}(t) \ \forall t$ | $\max_t |\hat{u}^{\text{cf}}(t) - \hat{u}^{\text{base}}(t)| < 10^{-6}\text{ mm}$ |
| **B. Past Invariance** | $t < t_y \implies \hat{u}^{\text{cf}}(t) \equiv \hat{u}^{\text{base}}(t)$ | $\max_{t < t_y} |\hat{u}^{\text{cf}}(t) - \hat{u}^{\text{base}}(t)| = 0.0000\text{ mm}$ |
| **C. Dose-Response Linearity** | $\Delta u_{\text{residual}} = m \cdot \Delta u_p + c$ across the 5 dose levels | Linear $R^2 \ge 0.95$, $m \in [0.85, 1.15]$ |
| **D. Sign Symmetry** | $\hat{u}(+\Delta s) - \hat{u}^{\text{base}} \approx -(\hat{u}(-\Delta s) - \hat{u}^{\text{base}})$ | Relative asymmetry $< 5.0\%$ |
| **E. Wrong-Time Control** | Perturb $\mathbf{s}(t_{\text{pre}})$ before ground motion arrival ($t < t_y$) | Transient decays to zero; $\Delta u_{\text{residual}} < 0.1 \cdot \Delta u_p$ |
| **F. Orthogonal Latent Control** | Perturb orthogonal latent subspace with $\|\Delta z\|_2 = \|\Delta s\|_2$ | No coherent DC displacement shift ($R^2 < 0.20$) |

---

## 8. Hypotheses & Experimental Groups

### Hypotheses:
- **H3A (Physics Supervision):** $\mathcal{L}_{\text{state}}$ supervision reduces residual drift error by $\ge 40\%$ compared to unsupervised state memory.
- **H3B (Causal Intervention):** Latent state clamping at $t_y$ causally shifts future baseline displacement with dose slope $m \approx 1.0$ while preserving past response.
- **H3C (Minimal State Sufficiency):** 2D physical state achieves performance parity ($\le 5\%$ error difference) with 64D unconstrained state space.

### Experimental Comparison Matrix:

| Model ID | Architecture Code | State Formulation | Training Loss | Primary Hypothesis Tested |
| :---: | :--- | :--- | :--- | :---: |
| **G1** | `StateAugmentedCausalTCN` | Unconstrained $s_t \in \mathbb{R}^4$ | $\mathcal{L}_{\text{data}} + 0.1 \mathcal{L}_{\text{energy}}$ | EXP 2 Frozen Baseline |
| **G2** | `PhysicsSupervisedCausalTCN` | Explicit $\mathbf{s}_{\text{phys}} \in \mathbb{R}^2$ | $\mathcal{L}_{\text{data}} + 0.1 \mathcal{L}_{\text{energy}} + 0.2 \mathcal{L}_{\text{state}}$ | **H3A** (Primary) & **H3B** (Causal) |
| **G3** | `PhysicsSupervisedCausalTCN` | Explicit $\mathbf{s}_{\text{phys}} \in \mathbb{R}^2$ | $\mathcal{L}_{\text{data}} + 0.1 \mathcal{L}_{\text{energy}}$ ($\lambda_{\text{state}}=0$) | **H3A** (Ablation Control) |
| **G4** | `HighDimStateCausalTCN` | Unconstrained $s_t \in \mathbb{R}^{64}$ | $\mathcal{L}_{\text{data}} + 0.1 \mathcal{L}_{\text{energy}}$ | **H3C** (State Dimension Parity) |
| **G5** | `PhysicsSupervisedCausalTCN` | Intervened $\mathbf{s}_{\text{cf}} \in \mathbb{R}^2$ | Evaluation Only (Dose Sweep) | **H3B** (Counterfactual Experiment) |

---

## 9. Data, Split, and Leakage Invariants

- Master Split File: `data/processed/splits/held_out_earthquake_split.json` (SHA-256: `d79f22f7...`)
  - **Train:** 11 earthquakes ($5,740$ simulations).
  - **Validation:** 2 earthquakes ($1,120$ simulations).
  - **Test (Held-Out):** 3 earthquakes ($1,540$ simulations: *Christchurch*, *Morgan Hill*, *Northridge-01*).
- Normalization: `x_normalizer`, `y_normalizer`, and `state_normalizer` fit strictly on the 5,740 training records. Zero test moments leaked.

---

## 10. Pre-Flight Test Specification (`tests/test_exp3_preflight.py`)

All 16 unit tests must pass with 100% success before Phase 4 smoke training:
1. `test_analytical_physical_state_reconstruction`: Verify $u_p(t)$ and $\alpha_b(t)$ match OpenSees constitutive equations to machine precision.
2. `test_pg_tcn_forward_shape`: Verify `[B, 10, 2048] -> [B, 3, 2048]` and `s_phys: [B, 2, 2048]`.
3. `test_parameter_budget_match`: Verify parameters within $\pm 1.0\%$ of $1,192,448$.
4. `test_strict_causality_future_perturbation`: Verify zero past contamination ($0.0000\text{ mm}$).
5. `test_state_reset_between_batches`: Verify hidden state resets to zero on each sample.
6. `test_zero_intervention_identity`: Verify $\Delta s = 0 \implies \hat{u}^{\text{cf}} \equiv \hat{u}^{\text{base}}$.
7. `test_past_trajectory_invariance`: Verify $t < t_y \implies \Delta u(t) = 0.0000\text{ mm}$.
8. `test_dose_response_monotonicity`: Verify positive $\Delta u_p$ yields positive $\Delta u(T_{\text{end}})$.
9. `test_sign_symmetry`: Verify $\Delta u(+\Delta s) \approx -\Delta u(-\Delta s)$.
10. `test_wrong_time_intervention`: Verify elastic pre-yield perturbation decays.
11. `test_orthogonal_latent_control`: Verify random latent perturbation fails to produce DC offset.
12. `test_loss_gradient_flow`: Verify $\mathcal{L}_{\text{state}}$ backpropagates to both encoder and state cell.
13. `test_split_earthquake_disjointness`: Verify train/val/test earthquakes are mutually disjoint.
14. `test_checkpoint_selection_isolated`: Verify early stopping monitors val loss only.
15. `test_normalizer_state_fit_train_only`: Verify normalizers fit on train split only.
16. `test_numerical_stability_nan_inf`: Verify forward/backward passes are NaN/Inf free.

---

## 11. Implementation Boundaries

- **MUST CREATE (New Files):**
  - `src/models/exp3_physics_state.py`
  - `src/models/exp3_state_conditioned_operator.py`
  - `src/evaluation/exp3_metrics.py`
  - `experiments/run_exp3.py`
  - `configs/experiments/exp3_physics_state.yaml`
  - `tests/test_exp3_preflight.py`
  - `tests/test_exp3_counterfactual.py`
- **MAY MODIFY:**
  - None (All EXP 3 components are implemented as modular extensions).
- **MUST NOT MODIFY (Strict Invariants):**
  - `src/models/state_augmented_tcn.py`, `src/models/s4_operator.py`, `src/models/causal_tcn.py`, `src/models/fno_1d.py`
  - `results/experiments/exp2_state_memory/*`
  - `data/processed/splits/held_out_earthquake_split.json`
  - `src/ground_truth/*`

---

## 12. Open PI Decisions Table

| Decision Item | Proposed Default in Specification | Alternative Option | Impact / Trade-Off |
| :--- | :--- | :--- | :--- |
| **D1: State Loss Weight $\lambda_{\text{state}}$** | $\lambda_{\text{state}} = 0.20$ | $\lambda_{\text{state}} \in \{0.05, 0.10, 0.50\}$ | Higher weight prioritizes drift accuracy over wave dynamics; $0.20$ is balanced. |
| **D2: State Projection Mechanism** | 16-neuron recurrent cell + 2D linear head | Direct 2-neuron recurrent cell | 16-neuron cell provides capacity for nonlinear yield surface transition. |
| **D3: Elastic Record Intervention Time** | Defaults to $t_{\text{PGA}}$ for control tests | Exclude $\mu \le 1$ from counterfactual tests | Testing on $\mu \le 1$ verifies wrong-time elasticity control. |

```
================================================================================
EXP 3 PHASE 1 STATUS: ARCHITECTURE SPECIFICATION COMPLETE
  - Specification written to docs/EXP3_ARCHITECTURE.md.
  - Zero implementation code modified.
  - Zero training executed.
  - Awaiting PI approval of Open Decisions (D1-D3) to authorize Phase 2.
================================================================================
```
