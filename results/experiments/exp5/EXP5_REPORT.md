# EXP 5: TOPOLOGY-NATIVE GRAPH NEURAL OPERATOR FOR SEISMIC STRUCTURAL DYNAMICS

**Experiment:** EXP 5 — Variable-Floor Graph Neural Operator  
**Date:** September 7, 2026  
**Hardware:** Apple Silicon M1 (MPS Unified Backend)  
**Authoritative Checkpoint:** `results/experiments/exp5/training/best_checkpoint.pt`  

---

## 1. Scientific Objective & Motivation

In EXP4, representing multi-story buildings on a fixed $5 \times 2048$ 2D grid required zero-padding floors 4 and 5 for 3-story buildings.
This induced severe spatial Fourier boundary artifacts:
- EXP4 5-story structures: **19.29%** median relative $L_2$ error.
- EXP4 3-story structures: **99.60%** median relative $L_2$ error.

EXP5 investigates:
> *Can a Graph Neural Operator represent and predict nonlinear seismic structural response across variable-floor and irregular structural topologies without fixed-grid zero-padding, while maintaining strict structural and earthquake out-of-distribution separation?*

---

## 2. Architecture & Formulation

- **Representation:** Exact graph $\mathcal{G} = (V, E)$ where $|V| = N_{\text{stories}}$. No padding nodes.
- **Spatial Operator:** Message passing over physical building columns and floor slabs:
  $$m_v(t) = \sum_{u \in \mathcal{N}(v)} W_{\text{val}} h_u(t) \odot \sigma(W_{\text{edge}} e_{uv}) + W_{\text{self}} h_v(t)$$
- **Temporal Operator:** Global 1D Fourier Neural Operator ($k_{\text{modes}} = 64$) along continuous time:
  $$\tilde{m}_v(t) = \mathcal{F}^{-1}\left( R_\phi \cdot \mathcal{F}(m_v) \right)(t)$$
- **Parameters:** 674,115 parameters.

---

## 3. Results Summary

- **3-Story Error:** Reduced from **99.60% (EXP4)** to **22.09% (EXP5 GNO)**.
- **Unseen Structural Archetype (`5S_T120`):** **125.39%** median relative $L_2$ error.
- **Unseen Earthquakes (RSN0011-12):** **17.34%** median relative $L_2$ error.
- **Speedup over OpenSeesPy:** **2.0x** (Single) and **1.6x** (Batched $B=32$).

---

## 4. Conclusion & Hand-off

The hypothesis that graph-native structural representation eliminates fixed-grid Fourier boundary artifacts is **SUPPORTED**.
EXP5 completes the research progression from SDOF analytical baselines (EXP1), empirical PEER database ingestion (EXP2), state-space recurrent modeling (EXP3-R), multi-story 2D neural operators (EXP4), to topology-native Graph Neural Operators (EXP5).
