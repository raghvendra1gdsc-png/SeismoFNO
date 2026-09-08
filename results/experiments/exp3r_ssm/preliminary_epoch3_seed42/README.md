# EXP 3-R PRELIMINARY INTERMEDIATE DIAGNOSTIC (EPOCH 3, SEED 42)

**Preservation Date:** September 2, 2026, 14:39 UTC  
**Checkpoint Path:** `results/experiments/exp3r_ssm/preliminary_epoch3_seed42/epoch3_checkpoint.pt`  
**Checkpoint SHA-256:**  
`e7f25b8c9d0496617841ac6cf28a43fd15248eccd3048f6fb07d500cf43cebc4`


## Mandatory Forensic Disclaimer
> **Epoch-3 seed-42 evaluation is an intermediate diagnostic only.**  
> **It does not replace the frozen 5-seed × 50-epoch protocol.**  
> No test-set tuning may be performed using this diagnostic.  
> This artifact is strictly preserved for forensic audit and diagnostic comparison.

## Diagnostic Summary (Seed 42, Epoch 3)
- **Epochs Completed:** 3 of 50
- **Validation Response Loss:** 1.20374
- **Validation Total Loss:** 164.20
- **Validation State Loss:** 814.98 (unconverged)
- **Held-Out Test Relative L2(u):** 100.93%
- **Regime A Median Slope m:** -0.0000 (Expected: 0.9800)
- **Sham Specificity S_v:** 6.61 (Expected <= 0.20)
- **Audited Cause:** Incomplete training (Epoch 3 of 50); readout head G_theta suppressed unconverged coordinate 2 (u_p) by 3.3x relative to unconstrained latent memory q.
- **Reference Audit:** [`EXP3R_FORENSIC_AUDIT.md`](file:///Users/rahul/seismoFNO/results/experiments/exp3r_ssm/preliminary_epoch3_seed42/EXP3R_FORENSIC_AUDIT.md)
