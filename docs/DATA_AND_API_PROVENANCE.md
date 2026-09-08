# Data & API Provenance Documentation — SeismoFNO

This document provides a comprehensive accounting of data provenance, seismic record sources, structural archetype definitions, numerical ground truth methodology, and the operational boundaries between live computation and archival research data.

---

## 1. System Scope & Live-Data Clarification

> [!IMPORTANT]
> **Real-World Earthquake Event Integration Layer (USGS):**
> SeismoFNO incorporates an isolated **Real-World Earthquake Event Integration Layer** powered by public USGS GeoJSON services.
> 
> **Scientific Boundaries & Operational Scope:**
> - **USGS Data Purpose:** Real-world earthquake event metadata (origin time, magnitude, hypocenter depth, geographic coordinates).
> - **Role:** Event discovery and context. **NOT** used as a substitute for the research acceleration waveform dataset.
> - **Waveform Boundary:** "USGS event metadata is used to connect the research prototype to observed earthquake events. SeismoFNO inference requires an appropriate ground-motion input and does not infer a structural response from earthquake magnitude/location alone."
> - **Disclaimers:** The system does **NOT** perform earthquake prediction, earthquake forecasting, live accelerometer streaming, live building sensor ingestion, real-time structural health monitoring, earthquake early warning, or automatic structural safety certifications.
> 
> All dynamic simulations are conditioned on verified historical ground motions ($a_g(t)$) from peer-reviewed databases (PEER NGA-West2 / Indian catalog) paired with validated finite element structural archetypes.

### Provenance Labels Explained
When interacting with the system (e.g., via the research demo at `/demo`, `/live`, or the API at `/api/v1/demo/simulate`), every output is labeled with explicit provenance metadata:

1. **`LIVE EVENT FEED`:**
   - Real-time or recent observed earthquake event metadata ingested from the public USGS GeoJSON catalog. Requires no API key.
   - If network connectivity is interrupted, the system automatically transitions to `OFFLINE CACHED EVENT DATA`.

2. **`LIVE_COMPUTED_MPS` (or `LIVE_COMPUTED`):**
   - The response trajectory $u(t)$ was computed in real time by executing a PyTorch neural operator forward pass on the local hardware (Apple Silicon MPS or CPU).
   - Latency figures reflect true, live wall-clock execution time.
   - Used for EXP6-B ($T_1$-GNO), EXP6-C (Multi-Modal GNO), and EXP6-D (Shuffled Ablation).

3. **`ARCHIVAL_FROZEN_VERIFIED` (or `ARCHIVAL GROUND MOTION`):**
   - The ground motion waveform or response trajectory is loaded directly from frozen, audited evaluation datasets and simulation records.
   - Used for the EXP4 2D-FNO baseline and verified historical earthquake records (PEER NGA-West2).

4. **`OPENSEESPY_NLTHA_GROUND_TRUTH` (or `FROZEN RESEARCH RESULT`):**
   - Ground truth response obtained via non-linear time-history analysis (NLTHA) using OpenSeesPy.

---

## 2. Ground Motion Records Provenance

The seismic catalog utilized in SeismoFNO originates from two primary sources:

### A. PEER NGA-West2 Ground Motion Database
Records sourced from the Pacific Earthquake Engineering Research (PEER) Center NGA-West2 database:

| Record ID | Earthquake Event | Year | Station | Magnitude ($M_w$) | Peak Ground Accel (PGA) |
| :--- | :--- | :---: | :--- | :---: | :---: |
| `RSN0001` | Imperial Valley | 1940 | El Centro Array #9 | 6.95 | 0.35 g |
| `RSN0006` | Imperial Valley | 1979 | Bonds Corner | 6.53 | 0.78 g |
| `RSN0180` | Imperial Valley | 1979 | El Centro Array #4 | 6.53 | 0.49 g |
| `RSN0802` | Loma Prieta | 1989 | Saratoga - Aloha Ave | 6.93 | 0.51 g |
| `RSN1011` | Northridge | 1994 | Sylmar - Converter Station | 6.69 | 0.84 g |

*Pre-processing standards:* All records are baseline-corrected, filtered, and resampled to uniform time steps of $\Delta t = 0.01\text{ s}$ over $N = 2048$ time points ($20.48\text{ s}$ total transient window).

### B. Indian Seismic Catalog
Historic earthquake records from the Indian subcontinent, digitized and processed in accordance with IS 1893 (Part 1): 2016:
- **Bhuj Earthquake (2001):** $M_w = 7.7$, PGA $\approx 0.38\text{ g}$.
- **Chamoli Earthquake (1999):** $M_w = 6.8$, PGA $\approx 0.36\text{ g}$.
- **Uttarkashi Earthquake (1991):** $M_w = 6.8$, PGA $\approx 0.31\text{ g}$.

---

## 3. Structural Archetypes Provenance

Structural models are formulated as multi-degree-of-freedom (MDOF) shear-building frames.

### Archetype Definitions
| Archetype ID | Stories ($N$) | Fundamental Period ($T_1$) | Story Stiffness ($k_i$) | Story Mass ($m_i$) | Role in Research |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `3S_T035` | 3 | 0.35 s | $1.8 \times 10^7\text{ N/m}$ | $1.5 \times 10^4\text{ kg}$ | Held-out 3-story geometry (exposed EXP4 99.60% failure) |
| `5S_T055` | 5 | 0.55 s | $2.5 \times 10^7\text{ N/m}$ | $2.0 \times 10^4\text{ kg}$ | In-distribution regular frame |
| `5S_T120` | 5 | 1.20 s | $0.5 \times 10^7\text{ N/m}$ | $2.0 \times 10^4\text{ kg}$ | Out-of-distribution OOD-B flexible frame ($T_1 > 0.85\text{ s}$) |
| `8S_T085` | 8 | 0.85 s | $3.2 \times 10^7\text{ N/m}$ | $2.5 \times 10^4\text{ kg}$ | Taller structural boundary validation |

### Modal Invariant Calculation
Theoretical structural periods $T_i$ and circular natural frequencies $\omega_i$ are derived from the mass matrix $\mathbf{M}$ and stiffness matrix $\mathbf{K}$ via the generalized linear eigenvalue problem:
$$(\mathbf{K} - \omega_i^2 \mathbf{M})\boldsymbol{\phi}_i = \mathbf{0}$$
$$T_i = \frac{2\pi}{\omega_i}$$

These eigenvalue invariants ($T_i, \omega_i$) serve as the pre-earthquake physical conditioning vector for EXP6 via Feature-wise Linear Modulation (FiLM).

---

## 4. Ground Truth Numerical Simulation

Ground truth non-linear dynamic responses are generated using **OpenSeesPy** (Python binding for the Open System for Earthquake Engineering Simulation):

- **Equation of Motion:**
  $$\mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{C} \dot{\mathbf{u}}(t) + \mathbf{f}_r(\mathbf{u}, \dot{\mathbf{u}}) = -\mathbf{M} \boldsymbol{\iota} \ddot{u}_g(t)$$
- **Numerical Integration:** Newmark-$\beta$ implicit step-by-step integration ($\gamma = 0.5$, $\beta = 0.25$, unconditional numerical stability for linear systems).
- **Non-Linear Solver:** Newton-Raphson iteration with Krylov-Newton and Modified Newton fallbacks.
- **Constitutive Model:** Bilinear hysteretic material law (Steel01) with kinematic hardening ratio $\alpha = 0.05$.
- **Damping:** Rayleigh damping configured for $5\%$ critical damping ($\zeta = 0.05$) pinned at the first two natural modes.

---

## 5. Quantitative Verification Metrics

All validation comparisons compute standard engineering and mathematical metrics:

1. **Relative $L_2$ Trajectory Error:**
   $$\text{Rel } L_2 = \frac{\|\mathbf{u}_{\text{pred}} - \mathbf{u}_{\text{true}}\|_2}{\|\mathbf{u}_{\text{true}}\|_2 + \epsilon} \times 100\%$$

2. **Peak Displacement Error:**
   $$\text{Peak Err} = \frac{|\max_t |u_{\text{pred}}(t)| - \max_t |u_{\text{true}}(t)||}{\max_t |u_{\text{true}}(t)| + \epsilon} \times 100\%$$

3. **Pearson Waveform Correlation ($r$):**
   $$r = \frac{\sum_{t} (u_{\text{pred}} - \bar{u}_{\text{pred}})(u_{\text{true}} - \bar{u}_{\text{true}})}{\sqrt{\sum_t (u_{\text{pred}} - \bar{u}_{\text{pred}})^2 \sum_t (u_{\text{true}} - \bar{u}_{\text{true}})^2}}$$

4. **Peak Timing Phase Lag ($\Delta t_{\text{peak}}$):**
   $$\Delta t_{\text{peak}} = |t_{\text{peak, pred}} - t_{\text{peak, true}}| \quad (\text{seconds})$$

---

## 6. External API Dependencies & Keys

- **USGS Public Earthquake Services (`src/demo/usgs_client.py`):**
  - **Purpose:** Real-world earthquake event metadata discovery and engineering context.
  - **API Key:** **NONE REQUIRED** (Open public GeoJSON feeds: `https://earthquake.usgs.gov/earthquakes/feed/v1.0/`).
  - **Offline Fallback:** Automatically falls back to verified cached event records (`data/usgs_cache.json`) if network is unavailable or times out.
  - **Waveform Boundary:** "USGS event metadata is used to connect the research prototype to observed earthquake events. SeismoFNO inference requires an appropriate ground-motion input and does not infer a structural response from earthquake magnitude/location alone."
- **Core Surrogate & Demo:** Requires **ZERO** external APIs, cloud subscriptions, or remote services. All PyTorch inference and OpenSeesPy simulations execute locally.
- **AI Copilot (`seismo_agent`):** Utilizes an optional `NEBIUS_API_KEY` for natural language structural consultations. If the key is absent or invalid, the system automatically falls back to a deterministic, offline rule-based engineering advisory engine. No secrets are ever exposed to the client browser.

