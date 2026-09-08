# Phase 7.1: Zero-Shot Generalization on Held-Out Earthquakes

Evaluated on completely unseen ground motion time series:

| Model | Overall Rel L2 u(t) | Overall Rel L2 E_h(t) | Elastic (mu <= 1.0) u(t) | Elastic (mu <= 1.0) E_h(t) | Post-Yield (mu > 1.0) u(t) | Post-Yield (mu > 1.0) E_h(t) |
| --- | --- | --- | --- | --- | --- | --- |
| FNO (+ Energy + Boundary) | 49.11% | 81.07% | 10.81% | 413.61% | 55.90% | 22.09% |
| FNO (+ History Channel) | 44.23% | 90.85% | 10.83% | 492.30% | 50.15% | 19.64% |
| Baseline: LSTM | 67.74% | 638.74% | 59.20% | 3695.23% | 69.26% | 96.61% |
| Baseline: MLP | 106.01% | 299.49% | 139.01% | 1354.76% | 100.15% | 112.31% |
