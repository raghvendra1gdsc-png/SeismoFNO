# SeismoFNO — Research Pipeline & Architecture Diagram

This document illustrates the complete computational architecture and experimental workflow of SeismoFNO across ground-truth numerical simulation, graph representation learning, physics-informed modal conditioning (EXP6), and multi-metric out-of-distribution evaluation.

---

## 1. High-Level Scientific Pipeline (Mermaid)

```mermaid
flowchart TD
    subgraph Physics_Pipeline["1. Ground-Truth Physics Pipeline (OpenSeesPy)"]
        GM["Seismic Ground Motion<br/>a_g(t) [PEER NGA-West2]"]
        SP["Structural Specifications<br/>(Stories, Masses, Stiffnesses)"]
        Matrices["Dynamic System Matrices<br/>[M], [K], [C] Rayleigh"]
        NLTHA["OpenSeesPy NLTHA Solver<br/>Implicit Newmark-Beta / Newton-Raphson"]
        GT_Resp["Ground-Truth Seismic Response<br/>Displacement u(t), IDR(t), Restoring Shear F_R(t)"]
        
        GM --> Matrices
        SP --> Matrices
        Matrices --> NLTHA
        GM --> NLTHA
        NLTHA --> GT_Resp
    end

    subgraph EXP6_Modal_Branch["2. Physics/Modal Conditioning Branch (EXP6)"]
        EigenSolver["Pre-Earthquake Invariant Analysis<br/>K phi_i = omega_i^2 M phi_i"]
        ModalVec["Modal Conditioning Vector<br/>c_modal = [T_1] or [T_1-3, omega_1-3]"]
        FiLM_Gen["FiLM Parameter Generator<br/>MLP_spat(c) & MLP_temp(c) -> [gamma, beta]"]
        
        Matrices --> EigenSolver
        EigenSolver --> ModalVec
        ModalVec --> FiLM_Gen
    end

    subgraph Learning_Pipeline["3. Spatiotemporal Graph Neural Operator (Conditioned GNO)"]
        GraphBuild["Structural Graph Construction<br/>Nodes = Stories (0 padding)<br/>Edges = Interstory Columns"]
        Lift["Node & Edge Feature Lifting<br/>Conv1D(in_channels -> hidden_dim)"]
        
        subgraph GNO_Blocks["Spatiotemporal Operator Blocks (x4)"]
            SpatConv["Spatial Graph Message Passing<br/>m_v(t) = sum W_val h_u(t) * sigma(W_edge e_uv)"]
            FiLM_Spat["Spatial FiLM Modulation<br/>h = (1 + gamma_spat) * h + beta_spat"]
            TempConv["Temporal Spectral 1D Conv (FNO)<br/>m(t) = F^-1( W(k) * F(m)(k) )(t)"]
            FiLM_Temp["Temporal FiLM Modulation<br/>h = (1 + gamma_temp) * h + beta_temp"]
            MLP_Block["Point-wise Channel MLP<br/>Conv1D -> GELU -> Conv1D"]
            
            SpatConv --> FiLM_Spat
            FiLM_Spat --> TempConv
            TempConv --> FiLM_Temp
            FiLM_Temp --> MLP_Block
        end
        
        Proj["Output Projection Layer<br/>hidden_dim -> 3 channels (u, IDR, F_R)"]
        Pred_Resp["Neural Operator Predictions<br/>u_pred(t), IDR_pred(t), F_R,pred(t)"]
        
        SP --> GraphBuild
        GM --> GraphBuild
        GraphBuild --> Lift
        Lift --> SpatConv
        FiLM_Gen -.-> FiLM_Spat
        FiLM_Gen -.-> FiLM_Temp
        MLP_Block --> Proj
        Proj --> Pred_Resp
    end

    subgraph Multi_Metric_Evaluation["4. Forensic Metric & Out-of-Distribution (OOD) Evaluation"]
        Compare{"Comparative Analysis & Error Quantification"}
        
        EDP_Metrics["Engineering Demand Parameters (EDPs)<br/>- Median Peak Displacement Error (%)<br/>- Peak Interstory Drift Error (%)"]
        Waveform_Metrics["Continuous Waveform Fidelity<br/>- Relative L2 Trajectory Error (%)<br/>- Pearson Correlation r<br/>- Temporal Phase Drift Delta t_peak"]
        Comp_Metrics["Computational Complexity<br/>- Inference Latency (ms)<br/>- Wall-clock Speedup vs OpenSeesPy"]
        
        OOD_A["OOD-A: Unseen Earthquakes (RSN0011-12)"]
        OOD_B["OOD-B: Unseen Flexible Structure (5S_T120, T1=1.20s)"]
        OOD_C["OOD-C: Combined Structural & Earthquake OOD"]
        Prog_OOD["Progressive Extrapolation: 5S_T105 & 5S_T140"]
        Ablation["Ablation D: Shuffled Conditioning Falsification"]
        
        GT_Resp --> Compare
        Pred_Resp --> Compare
        Compare --> EDP_Metrics
        Compare --> Waveform_Metrics
        Compare --> Comp_Metrics
        
        EDP_Metrics --> OOD_A
        EDP_Metrics --> OOD_B
        EDP_Metrics --> OOD_C
        EDP_Metrics --> Prog_OOD
        EDP_Metrics --> Ablation
    end
```

---

## 2. Textual Pipeline Flowchart

```
=================================================================================================
                                  SEISMOFNO PIPELINE ARCHITECTURE
=================================================================================================

 [ GROUND MOTION ]              [ STRUCTURAL PARAMS ]
  a_g(t) (PEER NGA)              Stories, Masses, Stiffnesses
        │                                      │
        ▼                                      ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                PHYSICAL EQUATIONS OF MOTION                 │
 │            M u''(t) + C u'(t) + F_R(u(t)) = -M i a_g(t)     │
 └──────────────────────────────┬──────────────────────────────┘
                                │
        ┌───────────────────────┴────────────────────────┐
        ▼                                                ▼
 ┌──────────────────────────────┐              ┌──────────────────────────────┐
 │   OPENSEESPY GROUND TRUTH    │              │      MODAL EIGENVALUES       │
 │   Implicit Newmark-Beta      │              │      K phi = omega^2 M phi   │
 │   Nonlinear Time-History     │              │      c = [T1, T2, T3, w1-3]  │
 └──────────────┬───────────────┘              └──────────────┬───────────────┘
                │                                             │
                │                                             ▼
                │                              ┌──────────────────────────────┐
                │                              │    FiLM GENERATOR (MLP)      │
                │                              │    [gamma, beta] = MLP(c)    │
                │                              └──────────────┬───────────────┘
                │                                             │
                │     ┌───────────────────────────────────────┘
                │     │ (Adaptive Modulation)
                ▼     ▼
 ┌─────────────────────────────────────────────────────────────┐
 │     SPATIOTEMPORAL GRAPH NEURAL OPERATOR (EXP6 GNO)         │
 │                                                             │
 │   Input Graph G = (V, E)  [Nodes=Stories, Edges=Columns]    │
 │     │ (Zero Padding: 0 nodes)                               │
 │     ▼                                                       │
 │   [Spatial Message Passing]  <── Modulated by FiLM (gamma)  │
 │     │                                                       │
 │     ▼                                                       │
 │   [1D Temporal Spectral FNO] <── Modulated by FiLM (beta)   │
 │     │                                                       │
 │     ▼                                                       │
 │   Predicted continuous response trajectories: u(t), IDR(t)  │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │         MULTI-METRIC DUAL EVALUATION & OOD ANALYSIS         │
 │                                                             │
 │   1. Envelope Demands (EDP): Peak Displacement Error %      │
 │      -> EXP5 Baseline: 35.21%  |  EXP6-C GNO: 13.06%        │
 │      -> 62.9% Relative Peak Error Reduction on 5S_T120      │
 │                                                             │
 │   2. Waveform Fidelity: Relative L2 % & Pearson r           │
 │      -> OOD-B Rel L2: 124.07%  |  Pearson r: 0.091          │
 │      -> Quantifies phase drift limitation of Fourier bases  │
 │                                                             │
 │   3. Falsification Ablation: Shuffled conditioning          │
 │      -> Shuffled OOD-B error jumps back to 24.33%           │
 │                                                             │
 │   4. Inference Speedup on Apple Silicon MPS:                │
 │      -> OpenSeesPy: 54.68 ms  vs  EXP6 GNO: 21.45 ms (2.5x) │
 └─────────────────────────────────────────────────────────────┘
```
