# Phase 7.3: Zero-Shot Resolution Invariance of SeismoFNO

Model trained strictly at base resolution $N = 2048$ ($\Delta t = 0.01$ s) and evaluated across sampling frequencies from 12.5 Hz to 400 Hz **without retraining**:

| Resolution (Steps) | dt (s) | Sampling Freq (Hz) | Displacement Rel L2 Error (%) | Force Rel L2 Error (%) | Hysteretic Energy Rel L2 Error (%) |
| --- | --- | --- | --- | --- | --- |
| 256 | 0.08000 | 12.5 | 61.02% | 43.85% | 66.90% |
| 512 | 0.04000 | 25.0 | 51.06% | 26.40% | 64.31% |
| 1024 | 0.02000 | 50.0 | 50.52% | 25.48% | 64.21% |
| 2048 | 0.01000 | 100.0 | 50.55% | 25.52% | 64.19% |
| 4096 | 0.00500 | 200.0 | 50.52% | 25.46% | 64.19% |
| 8192 | 0.00250 | 400.0 | 50.52% | 25.46% | 64.18% |
