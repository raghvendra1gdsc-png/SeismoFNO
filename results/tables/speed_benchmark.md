# Phase 9: Speed & Computational Throughput Benchmark

Measured on identical hardware (mps) excluding disk I/O and data loading overhead.
OpenSeesPy SDOF NLTHA baseline: **4.21 ms** per simulation (237.3 rec/s).
OpenSeesPy MDOF 3-story NLTHA baseline: **16.44 ms** per simulation (60.8 rec/s).

## Inference Speedup Across Batch Sizes

| Model / Method | Batch Size | Per-Record Latency | Throughput (rec/s) | Speedup Distribution (Mean +/- Std) | Speedup 5th-95th %ile |
| --- | --- | --- | --- | --- | --- |
| OpenSeesPy SDOF NLTHA (Ground Truth) | 1 (Sequential) | 4.21 +/- 0.67 ms | 237.3 | 1.0x (Baseline) | [1.0x, 1.0x] |
| OpenSeesPy MDOF 3-Story NLTHA | 1 (Sequential) | 16.44 +/- 0.92 ms | 60.8 | 1.0x (Baseline) | [1.0x, 1.0x] |
| SeismoFNO 1D (SDOF) | 1 | 2.2473 +/- 0.2380 ms | 445.0 | 1.9x +/- 0.2x | [1.5x, 2.1x] |
| SeismoFNO 1D (SDOF) | 8 | 0.9226 +/- 0.0092 ms | 1083.9 | 4.6x +/- 0.0x | [4.5x, 4.6x] |
| SeismoFNO 1D (SDOF) | 16 | 0.8234 +/- 0.0424 ms | 1214.5 | 5.1x +/- 0.3x | [4.8x, 5.5x] |
| SeismoFNO 1D (SDOF) | 32 | 0.8146 +/- 0.0374 ms | 1227.6 | 5.2x +/- 0.2x | [4.8x, 5.5x] |
| SeismoFNO 1D (SDOF) | 64 | 0.7215 +/- 0.0295 ms | 1385.9 | 5.8x +/- 0.2x | [5.5x, 6.1x] |
| SeismoFNO 1D (SDOF) | 128 | 0.7496 +/- 0.0706 ms | 1334.1 | 5.7x +/- 0.5x | [4.9x, 6.2x] |
| SeismoFNO 1D (SDOF) | 256 | 0.7536 +/- 0.0459 ms | 1326.9 | 5.6x +/- 0.3x | [5.2x, 6.2x] |
| LSTM Baseline (SDOF) | 1 | 66.4882 +/- 2.3698 ms | 15.0 | 0.1x +/- 0.0x | [0.1x, 0.1x] |
| LSTM Baseline (SDOF) | 8 | 16.6478 +/- 5.8999 ms | 60.1 | 0.3x +/- 0.1x | [0.1x, 0.3x] |
| LSTM Baseline (SDOF) | 16 | 7.1861 +/- 1.0268 ms | 139.2 | 0.6x +/- 0.1x | [0.4x, 0.6x] |
| LSTM Baseline (SDOF) | 32 | 4.2818 +/- 0.3072 ms | 233.5 | 1.0x +/- 0.1x | [0.9x, 1.0x] |
| LSTM Baseline (SDOF) | 64 | 4.8863 +/- 1.4387 ms | 204.7 | 0.9x +/- 0.2x | [0.6x, 1.2x] |
| MLP Baseline (SDOF) | 1 | 3.7405 +/- 0.8857 ms | 267.3 | 1.2x +/- 0.2x | [0.8x, 1.4x] |
| MLP Baseline (SDOF) | 8 | 1.5270 +/- 0.1646 ms | 654.9 | 2.8x +/- 0.3x | [2.3x, 3.1x] |
| MLP Baseline (SDOF) | 16 | 1.2722 +/- 0.0882 ms | 786.0 | 3.3x +/- 0.2x | [3.0x, 3.6x] |
| MLP Baseline (SDOF) | 32 | 1.1881 +/- 0.0294 ms | 841.7 | 3.5x +/- 0.1x | [3.4x, 3.7x] |
| MLP Baseline (SDOF) | 64 | 1.1443 +/- 0.0211 ms | 873.9 | 3.7x +/- 0.1x | [3.6x, 3.8x] |
| MLP Baseline (SDOF) | 128 | 1.0849 +/- 0.0776 ms | 921.7 | 3.9x +/- 0.2x | [3.3x, 4.1x] |
| MLP Baseline (SDOF) | 256 | 1.0603 +/- 0.0333 ms | 943.1 | 4.0x +/- 0.1x | [3.8x, 4.1x] |
| SeismoFNO 2D (MDOF 5-Story) | 1 | 6.5045 +/- 0.7143 ms | 153.7 | 2.6x +/- 0.3x | [2.1x, 2.9x] |
| SeismoFNO 2D (MDOF 5-Story) | 8 | 4.3593 +/- 0.0727 ms | 229.4 | 3.8x +/- 0.1x | [3.7x, 3.9x] |
| SeismoFNO 2D (MDOF 5-Story) | 16 | 4.2217 +/- 0.5484 ms | 236.9 | 3.9x +/- 0.4x | [3.1x, 4.2x] |
| SeismoFNO 2D (MDOF 5-Story) | 32 | 3.8310 +/- 0.2298 ms | 261.0 | 4.3x +/- 0.2x | [3.8x, 4.5x] |
| SeismoFNO 2D (MDOF 5-Story) | 64 | 4.0058 +/- 0.3670 ms | 249.6 | 4.1x +/- 0.4x | [3.5x, 4.5x] |

## 10,000-Record Regional Seismic Risk Assessment Case Study

- **OpenSeesPy Serial NLTHA**: 0.7 minutes (0.01 hours)
- **SeismoFNO (Single-record interactive)**: 22.47 seconds (1.9x speedup)
- **SeismoFNO (Batched regional evaluation)**: **7.54 seconds** (**5.6x speedup**)
