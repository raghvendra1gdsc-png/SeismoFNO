# PDF Build Documentation: SeismoFNO Research Brief

**Target Output:** `docs/RESEARCH_BRIEF.pdf`  
**Markdown Source:** `docs/RESEARCH_BRIEF.md`  
**Build Script:** `scripts/generate_research_brief_pdf.py`  
**Version:** September 2026 | **Scientific Core:** FROZEN / AUDITED  

---

## 1. Overview

The SeismoFNO Research Brief is a 2-page, publication-grade academic research summary tailored for professor-level evaluation (specifically for research research review at the Department of Computer Science & Engineering, Academic).

The PDF is generated using a reproducible, vector-based build pipeline that ensures:
1. **Pixel-perfect layout** fitting precisely on 2 pages (A4 format).
2. **Selectable text** with native system fonts.
3. **Typeset mathematical notation** for the governing matrix equation of motion.
4. **Embedded vector diagrams** for the 4-stage research progression and computational pipeline.
5. **Clickable relative hyperlinks** to full repository reports, forensic audits, and specifications.
6. **Embedded PDF metadata** (Title, Author, Subject, Keywords).

---

## 2. Prerequisites

- **Python 3.10+** (Python 3.10, 3.11, or 3.14)
- **Google Chrome** (standard installation at `/Applications/Google Chrome.app` on macOS, or `google-chrome` on Linux)
- No specialized LaTeX or heavy desktop publishing distribution is required.

---

## 3. How to Build the PDF

Run the automated generation script from the project root:

```bash
python3 scripts/generate_research_brief_pdf.py
```

### Execution Output:
```
==> Generating SeismoFNO Research Brief HTML...
    Wrote temporary HTML to docs/_temp_research_brief.html
==> Compiling PDF via headless Google Chrome...
    Compiled PDF: docs/RESEARCH_BRIEF.pdf
==> Patching PDF Metadata (Title, Author, Subject, Keywords)...
==> Auditing PDF structure & metrics...
    Page count: 2 (Target: 2, Max: 3)
    Clickable links detected: 5
    Verified metrics in source:
      - 99.60%      : ✓ FOUND
      - 22.09%      : ✓ FOUND
      - 35.21%      : ✓ FOUND
      - 13.47%      : ✓ FOUND
      - 13.06%      : ✓ FOUND
      - 24.33%      : ✓ FOUND
      - 54.68 ms    : ✓ FOUND
      - 21.45 ms    : ✓ FOUND
      - 2.55×       : ✓ FOUND
      - 294 passed  : ✓ FOUND
==> Research Brief PDF successfully compiled and verified!
```

---

## 4. Verification & Quality Checks

To verify that the generated PDF conforms to all publication and academic standards:

1. **Verify Page Count:**
   ```bash
   python3 -c "
   import re
   with open('docs/RESEARCH_BRIEF.pdf', 'rb') as f:
       data = f.read()
   pages = len(re.findall(rb'/Type\s*/Page\b', data))
   print(f'Page count: {pages} (Must be 2 or 3)')
   assert 2 <= pages <= 3
   "
   ```

2. **Verify Regression Test Suite:**
   ```bash
   ./.venv/bin/pytest -q
   ```
   *Expected:* `294 passed, 2 skipped, 0 failed`

3. **Verify Frontend Build:**
   ```bash
   cd frontend && npm run build
   ```
   *Expected:* `0 errors, 0 warnings`
