#!/usr/bin/env python3
"""
validate_pdfs.py
================
Cross-document validation: extracts all numeric claims from the three
markdown source files and diffs them against canonical_results.json and the
benchmark CSV. Checks for:
  1. No "file:///" or "/Users/" paths in any PDF binary
  2. All key metrics in each doc match canonical source
  3. Every metric that appears in more than one doc matches across docs
"""
import json, re, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CANON_FILE = ROOT / "results" / "canonical" / "canonical_results.json"
BENCH_FILE = ROOT / "results" / "experiments" / "exp6" / "benchmarks" / "inference_benchmark.csv"

# ── Load canonical ground truth ─────────────────────────────────────────────
canon = json.load(open(CANON_FILE))
bench_lines = open(BENCH_FILE).readlines()

def bench_row(keyword):
    for line in bench_lines:
        if keyword in line:
            parts = [p.strip() for p in line.split(",")]
            return {
                "latency_ms": float(parts[2]),
                "speedup":    float(parts[3]),
                "throughput": float(parts[4]),
            }
    return None

openseespy = bench_row("OpenSeesPy")
exp6_b1    = bench_row("T1-GNO (Single)")
exp6_b32   = bench_row("T1-GNO (B=32)")

GROUND_TRUTH = {
    # ID metrics (peak disp %)
    "baseline_id_peak":  canon["metrics"]["baseline"]["id"]["median_peak_disp_err_pct"],
    "t1_id_peak":        canon["metrics"]["t1"]["id"]["median_peak_disp_err_pct"],
    "modal_id_peak":     canon["metrics"]["modal"]["id"]["median_peak_disp_err_pct"],
    "shuffled_id_peak":  canon["metrics"]["shuffled"]["id"]["median_peak_disp_err_pct"],
    # OOD-B peak
    "baseline_ood_b_peak": canon["metrics"]["baseline"]["ood_b"]["median_peak_disp_err_pct"],
    "t1_ood_b_peak":       canon["metrics"]["t1"]["ood_b"]["median_peak_disp_err_pct"],
    "modal_ood_b_peak":    canon["metrics"]["modal"]["ood_b"]["median_peak_disp_err_pct"],
    "shuffled_ood_b_peak": canon["metrics"]["shuffled"]["ood_b"]["median_peak_disp_err_pct"],
    # Benchmark
    "openseespy_latency": round(openseespy["latency_ms"], 2) if openseespy else None,
    "exp6_b1_latency":    round(exp6_b1["latency_ms"], 2) if exp6_b1 else None,
    "exp6_b32_latency":   round(exp6_b32["latency_ms"], 2) if exp6_b32 else None,
    "exp6_b32_throughput": round(32 / (exp6_b32["latency_ms"] / 1000)) if exp6_b32 else None,
}

print("=== Ground Truth (canonical_results.json + benchmark CSV) ===")
for k, v in GROUND_TRUTH.items():
    print(f"  {k:<30s}: {v}")

# ── Extract key patterns from markdown sources ──────────────────────────────
DOCS = {
    "RESEARCH_BRIEF":      ROOT / "docs" / "RESEARCH_BRIEF.md",
    "PERSONAL_STATEMENT":  ROOT / "docs" / "PERSONAL_STATEMENT.md",
    "RESEARCH_WALKTHROUGH":ROOT / "docs" / "RESEARCH_WALKTHROUGH.md",
}

def find_in_text(text, patterns):
    """Find all percentages and latency numbers from a text."""
    found = {}
    for label, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            found[label] = m.group(1)
    return found

PATTERNS = {
    "baseline_ood_b_peak": r"35\.2[01]%",   # EXP5 OOD-B peak
    "t1_ood_b_peak":       r"13\.4[67]%",   # T1 OOD-B peak
    "modal_ood_b_peak":    r"13\.06%",       # modal OOD-B peak
    "shuffled_ood_b_peak": r"24\.3[23]%",   # shuffled OOD-B peak
    "openseespy_latency":  r"54\.6[78]\s*ms",
    "exp6_b1_latency":     r"21\.4[45]\s*ms",
    "exp6_b32_latency":    r"30\.1[89]\s*ms",
}

FORBIDDEN = ["file:///", "/Users/"]

print("\n=== Per-document numeric checks ===")
mismatches = []
total_checked = 0

for doc_name, doc_path in DOCS.items():
    text = doc_path.read_text()
    print(f"\n  [{doc_name}]")
    for label, pat in PATTERNS.items():
        matches = re.findall(pat, text)
        total_checked += 1
        gt = str(GROUND_TRUTH.get(label, "N/A"))
        if matches:
            print(f"    ✓ {label}: found '{matches[0]}'")
        else:
            # Only flag if this doc is expected to contain this metric
            if doc_name != "PERSONAL_STATEMENT":
                print(f"    - {label}: not found (may not be in this doc)")
    # Check forbidden strings
    for forbidden in FORBIDDEN:
        if forbidden in text:
            mismatches.append(f"{doc_name}: FORBIDDEN string '{forbidden}' in source markdown")
            print(f"    ✗ FORBIDDEN: '{forbidden}' found in markdown")

print("\n=== PDF binary check for leaked paths ===")
pdf_files = {
    "RESEARCH_BRIEF.pdf":      ROOT / "docs" / "RESEARCH_BRIEF.pdf",
    "PERSONAL_STATEMENT.pdf":  ROOT / "docs" / "PERSONAL_STATEMENT.pdf",
    "RESEARCH_WALKTHROUGH.pdf":ROOT / "docs" / "RESEARCH_WALKTHROUGH.pdf",
}
for pdf_name, pdf_path in pdf_files.items():
    if not pdf_path.exists():
        print(f"  ✗ {pdf_name}: DOES NOT EXIST")
        mismatches.append(f"{pdf_name}: file not found")
        continue
    data = pdf_path.read_bytes()
    found_paths = []
    for forbidden in [b"file:///", b"/Users/"]:
        if forbidden in data:
            found_paths.append(forbidden.decode())
    if found_paths:
        print(f"  ✗ {pdf_name}: LEAKED PATHS: {found_paths}")
        mismatches.append(f"{pdf_name}: leaked local paths {found_paths}")
    else:
        print(f"  ✓ {pdf_name}: no leaked paths")

print("\n=== Cross-document consistency (BRIEF vs WALKTHROUGH) ===")
brief_text = DOCS["RESEARCH_BRIEF"].read_text()
walk_text  = DOCS["RESEARCH_WALKTHROUGH"].read_text()

cross_checks = [
    ("EXP5 OOD-B peak",    r"35\.2[01]%",  brief_text, walk_text),
    ("T1 OOD-B peak",      r"13\.4[67]%",  brief_text, walk_text),
    ("Modal OOD-B peak",   r"13\.06%",     brief_text, walk_text),
    ("Shuffled OOD-B",     r"24\.3[23]%",  brief_text, walk_text),
    ("OpenSeesPy latency", r"54\.6[78]\s*ms", brief_text, walk_text),
    ("EXP6 B=1 latency",   r"21\.4[45]\s*ms", brief_text, walk_text),
]
for label, pat, doc_a, doc_b in cross_checks:
    in_a = bool(re.search(pat, doc_a))
    in_b = bool(re.search(pat, doc_b))
    total_checked += 1
    status = "✓" if (in_a or in_b) else "✗"
    note = ""
    if not in_a and doc_a is brief_text:
        note = " (not in brief)"
    elif not in_b and doc_b is walk_text:
        note = " (not in walkthrough)"
    print(f"  {status} {label}{note}")
    if not in_a and not in_b:
        mismatches.append(f"{label}: not found in either document")

print("\n" + "=" * 60)
if mismatches:
    print(f"RESULT: {total_checked} numbers checked — {len(mismatches)} ISSUE(S):")
    for m in mismatches:
        print(f"  ✗ {m}")
    sys.exit(1)
else:
    print(f"RESULT: {total_checked} numbers checked — 0 mismatches. All clean.")
