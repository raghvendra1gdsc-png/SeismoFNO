# Unified Baseline Benchmark Table (Held-Out-Earthquake Split)

| Model / Solver | Causality | Overall Rel L2 u(t) (%) | Peak u_max Error (%) | Elastic u(t) (%) | Post-Yield u(t) (%) | Force L2 (%) | Latency (ms) | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NumPy Newmark | Causal (Time-stepping) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 6.51 | Exact Newmark |
| Standard FNO | Acausal (Global Spectral) | 50.01% | 15.18% | 10.77% | 57.48% | 25.01% | 6.55 | 1,196,931 |
| LSTM Baseline | Causal (Recurrent Hidden State) | 71.62% | 20.32% | 56.84% | 74.44% | 32.83% | 92.43 | 414,851 |
| MLP Baseline | Acausal (Pointwise) | 106.76% | 94.80% | 139.76% | 100.47% | 100.00% | 7.59 | 564,995 |
| Chopra CSM | Spectral (Equivalent Linear) | 89.86% | 89.86% | 89.47% | 89.94% | 0.00% | 0.00 | Analytical |
