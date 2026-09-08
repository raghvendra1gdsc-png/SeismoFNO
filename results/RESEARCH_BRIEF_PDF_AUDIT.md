# SeismoFNO — Research Brief PDF Verification & Integrity Audit

**Audit Date:** September 8, 2026  
**Auditor:** Senior Scientific ML Researcher & Packaging Engineer  
**Target Evaluation:** Research Brief Packaging — Department of Computer Science & Engineering, IIT Delhi Application  
**Output Document:** `docs/IIT_DELHI_CSE_RESEARCH_BRIEF.pdf`  
**Overall Status:** **VERIFIED — ALL SPECIFICATIONS SATISFIED (FROZEN RESEARCH CORE)**  

---

## 1. Document Inventory & Metrics Summary

| Property | Value | Requirement / Constraint | Verification Status |
|---|---|---|---|
| **Primary PDF Path** | `docs/IIT_DELHI_CSE_RESEARCH_BRIEF.pdf` | `docs/IIT_DELHI_CSE_RESEARCH_BRIEF.pdf` | **MATCH** |
| **Markdown Source** | `docs/IIT_DELHI_CSE_RESEARCH_BRIEF.md` | Preserved and synchronized | **MATCH** |
| **Build Script** | `scripts/generate_research_brief_pdf.py` | Reproducible Python generator | **MATCH** |
| **Page Count** | **2 pages** | Target: 2 pages, Max: 3 pages | **EXACT TARGET** |
| **File Size** | **969,327 bytes (947 KB)** | Vector-rendered standalone document | **PASS** |
| **Generation Engine** | Google Chrome Headless (`--print-to-pdf`) | Vector typography + Post-metadata injection | **PASS** |
| **Clickable Hyperlinks** | **5 links** (PDF `/Subtype /Link`) | Valid relative internal repository links | **VERIFIED** |
| **QR Codes Generated** | **0** | No public URL exists; fake codes strictly prohibited | **COMPLIANT** |
| **Test Suite Status** | **294 passed, 2 skipped, 0 failed** | Zero regressions against test baseline | **PASS** |
| **Frontend Build** | **Built in 239ms (0 errors, 0 warnings)** | Clean TypeScript / Vite bundle | **PASS** |

---

## 2. Empirical Value Reconciliation & Metric Traceability

Every quantitative metric reported in `docs/IIT_DELHI_CSE_RESEARCH_BRIEF.pdf` matches frozen historical evaluation records:

| Required Metric | Location in PDF | Value | Primary Source | Reconciliation |
|---|---|---|---|---|
| **99.60%** | Section 1, 3, 4 | 3-story median Rel. L₂ error (EXP4) | `results/experiments/exp4/` | **EXACT MATCH** |
| **22.09%** | Section 1, 3, 4 | 3-story median Rel. L₂ error (EXP5) | `results/experiments/exp5/` | **EXACT MATCH** |
| **35.21%** | Section 1, 5 | OOD-B median peak displacement error (EXP5) | `results/experiments/exp5/` | **EXACT MATCH** |
| **13.47%** | Section 4, 5 | OOD-B median peak displacement error (EXP6-B) | `results/experiments/exp6/` | **EXACT MATCH** |
| **13.06%** | Section 1, 3, 4, 5 | OOD-B median peak displacement error (EXP6-C) | `results/experiments/exp6/` | **EXACT MATCH** |
| **24.33%** | Section 4 | OOD-B median peak displacement error (EXP6-D) | `results/experiments/exp6/` | **EXACT MATCH** |
| **54.68 ms** | Section 6 | OpenSeesPy 5-story NLTHA execution time | `inference_benchmark.csv` | **EXACT MATCH** |
| **21.45 ms** | Section 6 | EXP6 T₁-GNO single forward pass latency | `inference_benchmark.csv` | **EXACT MATCH** |
| **2.55×** | Section 6 | Measured wall-clock speedup ratio on Apple Silicon MPS | `inference_benchmark.csv` | **EXACT MATCH** |
| **294 passed** | Header, Section 10 | Automated regression unit tests | `tests/` via pytest | **EXACT MATCH** |

---

## 3. Scientific Language & Claims Compliance Audit

A systematic lexical audit was performed across the PDF text and Markdown companion to ensure strict scientific honesty:

| Checked Term | Status in PDF | Remediation / Audited Phrasing |
|---|---|---|
| `"proves"` | **0 occurrences** | Replaced with *"provides evidence that"* |
| `"confirms"` | **0 occurrences** | Replaced with *"provides evidence that"* or *"corroborates"* |
| `"solves"` | **0 occurrences** | Replaced with *"resolves the boundary failure observed in EXP4"* |
| `"guarantees"` | **0 occurrences** | Strictly avoided |
| `"universally generalizes"` | **0 occurrences** | Bound to *"under the evaluated structural OOD condition"* |
| `"production-ready"` | **0 occurrences** | Strictly avoided |
| `"state-of-the-art"` | **0 occurrences** | Strictly avoided |
| **EXP6-D Falsification** | **Compliant** | *"Provides evidence that performance gains depend on physically meaningful modal correspondence rather than merely additional conditioning capacity."* |
| **Active Limitations** | **Prominently Featured** | Section 7 explicitly details cumulative phase divergence ($r \approx 0.05\text{--}0.09$, Rel. $L_2 > 100\%$), pre-earthquake static modal limitations during yielding, and planar shear-frame idealization. |

---

## 4. Hyperlink Verification

The PDF contains 5 clickable `/Subtype /Link` annotations pointing to local repository resources:
1. `docs/SEISMOFNO_TECHNICAL_REPORT.md`: Comprehensive 18-section technical report.
2. `docs/PROFESSOR_DEMO.md`: Technical reference manual and demonstration architecture.
3. `results/experiments/exp6/INDEPENDENT_FORENSIC_AUDIT.md`: Cryptographic audit of EXP6.
4. `docs/RESEARCH_ARCHITECTURE_DIAGRAM.md`: Architecture specification.
5. `README.md`: Master project repository documentation.

*(The interactive web demo is explicitly labeled as `local / repository deployment`, with zero fake URLs or localhost QR codes).*

---

## 5. Freezing Confirmation & Boundary Enforcement

1. **Checkpoints & Weights:**
   - `results/experiments/exp4/training/best_checkpoint.pt` unmodified.
   - `results/experiments/exp5/training/best_checkpoint.pt` unmodified.
   - `results/experiments/exp6/training/best_t1_gno.pt` unmodified.
   - `results/experiments/exp6/training/best_multimodal_gno.pt` unmodified.
   - `results/experiments/exp6/training/best_shuffled_modal_gno.pt` unmodified.
2. **Datasets & Splits:**
   - All 2,160 physical simulation records and split manifest CSVs remain completely untouched.
3. **EXP7:**
   - Strictly not started.
4. **Conclusion:**
   - Packaging task complete. The SeismoFNO research brief is frozen, audited, and ready for academic review.
