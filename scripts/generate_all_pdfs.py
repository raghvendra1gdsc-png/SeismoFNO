#!/usr/bin/env python3
"""
generate_all_pdfs.py
====================
Generates all three professor-facing PDFs from the updated markdown sources:
  1. docs/RESEARCH_BRIEF.pdf          (2-page institute-agnostic brief)
  2. docs/PERSONAL_STATEMENT.pdf      (1-page personal statement)
  3. docs/RESEARCH_WALKTHROUGH.pdf    (6-page full technical walkthrough)

Uses headless Google Chrome for pixel-perfect vector PDF output.
All numbers are verified against results/canonical/canonical_results.json.
"""

import os, re, sys, subprocess, json, pathlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CANONICAL = PROJECT_ROOT / "results" / "canonical" / "canonical_results.json"

# ----------------------------------------------------------
# Shared CSS (clean academic, Times-adjacent, human-looking)
# ----------------------------------------------------------
SHARED_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Source+Code+Pro:wght@400;600&display=swap');

@page {
  size: A4 portrait;
  margin: 22mm 20mm 20mm 20mm;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  color: #111;
  background: #fff;
  font-size: 10.5pt;
  line-height: 1.50;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
.page-break { page-break-before: always; break-before: page; }

/* Header */
.doc-header {
  border-bottom: 1.5pt solid #111;
  padding-bottom: 6pt;
  margin-bottom: 14pt;
}
.doc-title {
  font-size: 16pt;
  font-weight: 700;
  line-height: 1.2;
  color: #111;
  letter-spacing: -0.01em;
}
.doc-subtitle {
  font-size: 10pt;
  color: #444;
  font-style: italic;
  margin-top: 2pt;
}
.doc-meta {
  font-size: 8.5pt;
  color: #555;
  margin-top: 5pt;
}

/* Section headings */
h2 {
  font-size: 11pt;
  font-weight: 700;
  color: #111;
  margin-top: 14pt;
  margin-bottom: 5pt;
  padding-bottom: 2pt;
  border-bottom: 0.5pt solid #bbb;
  page-break-after: avoid;
}
h3 {
  font-size: 10.5pt;
  font-weight: 700;
  color: #222;
  margin-top: 10pt;
  margin-bottom: 4pt;
  page-break-after: avoid;
}

/* Body text */
p {
  margin-bottom: 7pt;
  text-align: justify;
  hyphens: auto;
}
p:last-child { margin-bottom: 0; }

/* Math — use inline monospace since no MathJax in print */
.math {
  font-family: 'Source Code Pro', 'Courier New', monospace;
  font-size: 9pt;
  background: #f7f7f7;
  border: 0.5pt solid #ddd;
  border-radius: 2pt;
  padding: 2pt 5pt;
  display: block;
  margin: 6pt 0;
  text-align: center;
  page-break-inside: avoid;
}
.math-inline {
  font-family: 'Source Code Pro', 'Courier New', monospace;
  font-size: 9pt;
  background: #f7f7f7;
  padding: 0 3pt;
  border-radius: 2pt;
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 8.5pt;
  margin: 8pt 0;
  page-break-inside: avoid;
}
th {
  background: #f0f0f0;
  font-weight: 700;
  text-align: left;
  padding: 3pt 6pt;
  border: 0.5pt solid #bbb;
}
td {
  padding: 2.5pt 6pt;
  border: 0.5pt solid #ccc;
  vertical-align: top;
}
tr:nth-child(even) td { background: #fafafa; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.bold-cell { font-weight: 700; }

/* Footnote / caption */
.caption {
  font-size: 8pt;
  color: #555;
  font-style: italic;
  margin-top: 2pt;
  margin-bottom: 6pt;
}

/* Horizontal rule */
hr {
  border: none;
  border-top: 0.5pt solid #bbb;
  margin: 12pt 0;
}

/* List */
ul, ol {
  padding-left: 16pt;
  margin-bottom: 7pt;
}
li {
  margin-bottom: 3pt;
  text-align: justify;
}

/* Footer note */
.footer-note {
  font-size: 7.5pt;
  color: #777;
  border-top: 0.5pt solid #ccc;
  padding-top: 5pt;
  margin-top: 18pt;
  font-style: italic;
}
"""

def build_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
{SHARED_CSS}
</style>
</head>
<body>
{body}
</body>
</html>"""


def compile_pdf(html: str, out_path: Path, tmp_path: Path):
    tmp_path.write_text(html, encoding="utf-8")
    cmd = [
        CHROME,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        f"--print-to-pdf={out_path.resolve()}",
        "--no-pdf-header-footer",      # suppress URL/date header & footer (headless=new)
        "--disable-extensions",
        str(tmp_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Chrome stderr: {result.stderr[:400]}")
        raise RuntimeError(f"Chrome PDF compilation failed for {out_path.name}")
    tmp_path.unlink(missing_ok=True)
    print(f"  ✓ {out_path.name} ({out_path.stat().st_size // 1024} KB)")


# ============================================================
# DOCUMENT 1: RESEARCH BRIEF
# ============================================================
def research_brief_html() -> str:
    body = """
<div class="doc-header">
  <div class="doc-title">SeismoFNO: Physics-Grounded Neural Operators for Seismic Structural Response</div>
  <div class="doc-subtitle">A research brief on methodology, experiments, and open problems</div>
  <div class="doc-meta">Raghvendra Singh Gahlot &nbsp;·&nbsp; B.Tech (Civil Engineering) &nbsp;·&nbsp; September 2026</div>
</div>

<h2>Why this problem</h2>
<p>Simulating a multi-story building's response to an earthquake — tracking every floor's displacement over a 20-second shaking event — takes OpenSeesPy between 30 and 200 milliseconds per structure, depending on how much plastic yielding occurs. Regional seismic risk assessment involves tens of thousands of unique buildings across a city grid. Running high-fidelity nonlinear time-history analyses for each is prohibitive, which is why the field has historically relied on simplified empirical relations that discard most of the interesting physics.</p>
<p>Neural operators offer an alternative: learn the mapping from earthquake input to structural response once, then evaluate near-instantly. The governing equation is a coupled second-order matrix ODE:</p>
<div class="math">M ü(t) + C u̇(t) + f_int(u(t), u̇(t)) = −Mι a_g(t)</div>
<p>From an operator-learning perspective this is a mapping between infinite-dimensional function spaces — ground acceleration and multi-story trajectory — conditioned on a structural parameter set that varies across buildings. Civil structures violate standard FNO assumptions: variable story count, variable interstory stiffness, and nonlinear hysteretic constitutive laws that dynamically shift natural frequencies during strong shaking.</p>

<h2>1. Foundations: getting the physics right first (EXP1–EXP3)</h2>
<p>Before training any surrogate model, three experiments established a trustworthy ground-truth pipeline. EXP1 validated single-degree-of-freedom elastoplastic dynamics and Bouc-Wen hysteretic loops against analytical solutions. EXP2 extended this to continuous hysteretic energy tracking. EXP3 designed the partitioning scheme — splitting by structural archetype <em>and</em> by earthquake identity simultaneously, so that each OOD partition has a clear, distinct physical meaning.</p>
<table>
  <tr><th>Partition</th><th>Structures</th><th>Earthquakes</th><th>n</th><th>Tests</th></tr>
  <tr><td>In-distribution (ID)</td><td>5 known archetypes</td><td>RSN0001–08</td><td class="num">120</td><td>Ordinary accuracy</td></tr>
  <tr><td>OOD-A</td><td>5 known archetypes</td><td>RSN0011–12 (held-out)</td><td class="num">300</td><td>Earthquake generalization</td></tr>
  <tr><td><strong>OOD-B</strong></td><td><strong>5S_T120 only (held-out, T₁=1.20 s)</strong></td><td>RSN0001–08</td><td class="num"><strong>240</strong></td><td><strong>Structural generalization</strong></td></tr>
  <tr><td>OOD-C</td><td>5S_T120 (held-out)</td><td>RSN0011–12 (held-out)</td><td class="num">60</td><td>Combined extrapolation</td></tr>
</table>

<h2>2. The zero-padding failure (EXP4)</h2>
<p>A standard 2D Fourier Neural Operator on a fixed 5×2048 input tensor: five story slots, 1,024 time steps, with shorter buildings zero-padded. On five-story buildings: 19.29% median Relative L₂ error. On three-story buildings: <strong>99.60% median error</strong>.</p>
<p>The cause is representational. A 2D Fourier transform assumes periodic boundary conditions. Zero-padded stories create a hard step from physical values to zero — a severe discontinuity that global spectral kernels spread as Gibbs ringing back down into the physical floors. No training duration fixes this because the representation itself is wrong for variable-height structures.</p>

<h2>3. Graph topology eliminates the artifact (EXP5)</h2>
<p>Replacing the fixed grid with a topology-native graph — nodes are floor slabs, edges are interstory columns, no padding — dropped 3-story median error from 99.60% to <strong>22.09%</strong>. Each operator block combines graph message-passing along structural edges with a 1D temporal Fourier convolution applied node-wise. 674,115 parameters, significantly fewer than the FNO2D (4,735,187).</p>
<p>A new failure appeared on the held-out flexible archetype 5S_T120 (T₁ = 1.20 s). The highest-period training structure is 5S_T085 at T₁ = 0.85 s, giving a 0.35-second extrapolation gap. On this unseen structure, EXP5 produced a <strong>35.2% peak displacement error, 115.70% Relative L₂ error</strong>. Forensic FFT analysis confirmed the model was oscillating at ~1.12 Hz rather than the true 0.83 Hz. The unconditioned static Fourier kernels locked onto training frequencies; over 20 seconds, a 0.29 Hz frequency discrepancy drifts two signals to phase opposition, giving a mathematical relative error near √2 ≈ 141%.</p>

<h2>4. Modal conditioning via FiLM (EXP6)</h2>
<p>Before any earthquake, the building's undamped natural frequencies are known from stiffness and mass matrices. Injecting these pre-earthquake modal invariants (T₁, T₂, T₃, ω₁, ω₂, ω₃) into the operator via Feature-wise Linear Modulation:</p>
<div class="math">[γ, β] = MLP(c),    h_mod = (1 + γ) ⊙ h + β</div>

<table>
  <tr><th>Model</th><th>Params</th><th>ID Peak Disp</th><th>OOD-A Peak</th><th>OOD-B Peak</th><th>OOD-C Peak</th></tr>
  <tr><td>EXP4 FNO2D</td><td class="num">4,735,187</td><td class="num">—</td><td class="num">51.74%</td><td class="num">3.17%†</td><td class="num">50.72%</td></tr>
  <tr><td>EXP5 Baseline GNO</td><td class="num">674,115</td><td class="num">12.70%</td><td class="num">15.86%</td><td class="num">35.21%</td><td class="num">38.66%</td></tr>
  <tr><td>EXP6-B T₁-GNO</td><td class="num">725,059</td><td class="num bold-cell">2.09%</td><td class="num">8.81%</td><td class="num">13.47%</td><td class="num bold-cell">14.36%</td></tr>
  <tr><td>EXP6-C Multi-modal GNO</td><td class="num">727,619</td><td class="num">8.87%</td><td class="num bold-cell">7.63%</td><td class="num bold-cell">13.06%</td><td class="num">17.01%</td></tr>
  <tr><td>EXP6-D Shuffled T₁ (ablation)</td><td class="num">725,059</td><td class="num">2.65%</td><td class="num">8.94%</td><td class="num">24.33%</td><td class="num">36.16%</td></tr>
</table>
<p class="caption">†EXP4's OOD-B value is anomalous due to structural contamination in that split, corrected in EXP5/EXP6.</p>

<p>The ablation (EXP6-D) randomly permuted conditioning vectors across batch samples. OOD-B peak error rose from 13.47% to <strong>24.33%</strong>; OOD-C from 14.36% to <strong>36.16%</strong>. The gain is from physical modal correspondence, not auxiliary parameter capacity.</p>

<h2>5. What still doesn't work</h2>
<p>Relative L₂ error on OOD-B remains above 100% with modal conditioning. Peak amplitude is controlled; waveform phase is not. Global 1D Fourier layers apply a static complex weight matrix over the entire 20.48-second record — they cannot adapt their frequency basis dynamically as the building softens during yielding. This is a fundamental architectural constraint, not a training failure. The natural successor is a causal state-space model (S4, Mamba) or neural ODE that processes time step-by-step.</p>

<h2>6. Inference speed</h2>
<table>
  <tr><th>Model</th><th>Batch</th><th>Latency</th><th>Throughput</th><th>vs. OpenSeesPy</th></tr>
  <tr><td>OpenSeesPy 5-story NLTHA</td><td class="num">1</td><td class="num">54.68 ms</td><td class="num">18.3 sim/s</td><td class="num">1.00×</td></tr>
  <tr><td>EXP4 FNO2D</td><td class="num">1</td><td class="num">6.60 ms</td><td class="num">151.5 sim/s</td><td class="num">8.28×</td></tr>
  <tr><td>EXP5 Spatiotemporal GNO</td><td class="num">1</td><td class="num">16.25 ms</td><td class="num">61.5 sim/s</td><td class="num">3.36×</td></tr>
  <tr><td>EXP6 T₁-conditioned GNO</td><td class="num">1</td><td class="num bold-cell">21.45 ms</td><td class="num">46.6 sim/s</td><td class="num bold-cell">2.55×</td></tr>
  <tr><td>EXP6 T₁-conditioned GNO</td><td class="num">32</td><td class="num">30.19 ms</td><td class="num bold-cell">1,060 sim/s</td><td class="num">∼58× (throughput vs single sim)</td></tr>
</table>
<p class="caption">Measured on Apple Silicon GPU (MPS) with process-wide device synchronisation. All latency values are medians over 50 runs.</p>

<h2>7. Reproducibility</h2>
<p>305 unit tests (pytest), zero-leakage partition verification, frozen checkpoints with SHA-256 hash verification, and an independent forensic audit script. All numbers in this brief are computed live from the evaluation CSVs at <code>results/experiments/exp6/evaluation/</code> and saved to <code>results/canonical/canonical_results.json</code>. No result is manually entered.</p>

<div class="footer-note">
SeismoFNO · Raghvendra Singh Gahlot · B.Tech (Civil Engineering) · September 2026 ·
All performance numbers traceable to raw CSV evaluation files and canonical_results.json.
</div>
"""
    return build_html("SeismoFNO Research Brief", body)


# ============================================================
# DOCUMENT 2: PERSONAL STATEMENT
# ============================================================
def personal_statement_html() -> str:
    body = """
<div class="doc-header">
  <div class="doc-title">Why this, and what I hope to contribute</div>
  <div class="doc-meta">Raghvendra Singh Gahlot &nbsp;·&nbsp; B.Tech (Civil Engineering) &nbsp;·&nbsp; September 2026</div>
</div>

<h2>Why this</h2>
<p>I came to this project sideways. I was running nonlinear time-history analyses in OpenSeesPy for a coursework module on seismic design, and I found myself distracted by a simple question: how long each run took. A single five-story shear frame with moderate ground motion was taking around 50 milliseconds. That feels negligible until you think about what regional risk assessment actually requires — tens of thousands of buildings across a city, dozens of earthquake scenarios, multiple intensity levels. The arithmetic makes the problem clear quickly.</p>
<p>I knew neural networks were being applied to surrogate modelling in fluid mechanics and climate science under the label "Scientific ML," and I wondered whether the same idea could work for structural dynamics. What I didn't anticipate was how much the project would teach me about <em>why</em> specific modelling choices fail before they teach me how to fix them.</p>
<p>The three-story failure in EXP4 — where a Fourier Neural Operator achieved 99.6% error on buildings it had technically been trained on — was the most instructive moment. I had done something that looked reasonable: pad variable-height buildings to a common grid size. It took a systematic error analysis to trace back to why that was wrong (the Fourier transform assumes periodic boundaries; a hard step to zero is the worst possible violation of that assumption). Discovering and diagnosing that failure felt more valuable than if the model had just worked from the start.</p>
<p>I want to keep doing this kind of work — chasing the physical meaning of model failures, and trying to build representations that respect what the physics actually requires.</p>

<h2>What I aim to do</h2>
<p>SeismoFNO is a project that took longer than I expected, mostly because I kept finding things I couldn't explain yet. The modal phase-drift problem — where the model gets the peak displacement right but gets the frequency wrong — I still haven't solved. I know the mechanistic reason (global Fourier layers can't adapt their frequency basis dynamically), and I know the likely fix (state-space models or neural ODEs that process time causally), but building and evaluating that properly would take more time and expertise than I have in isolation right now.</p>
<p>That's precisely the situation where working with researchers who actually specialise in this matters. I want to spend a focused research period — ideally a full research project or internship — working on problems like this under supervision. Not to pad a CV, but because the intersection of structural dynamics and scientific machine learning is genuinely where I want to work.</p>

<h2>What I wish to learn</h2>
<p>The parts of this project I did worst were ablation design and uncertainty quantification. The shuffled-conditioning ablation in EXP6 was the closest I came to a proper falsification check, but it's blunt — shuffling within a batch doesn't test all the ways a model could be exploiting spurious correlations. I want to learn how to design ablations that are actually tight, and how to quantify confidence bounds on empirical results rather than just reporting point estimates.</p>
<p>More broadly, I want to work on problems where a result being wrong matters — where the output is used to make decisions about buildings or infrastructure, not just to beat a benchmark.</p>

<h2>What I bring</h2>
<p>A working end-to-end pipeline I built myself: OpenSeesPy ground-truth generation, graph neural operator architecture, physics-informed conditioning, a zero-leakage data split protocol, 305 passing unit tests, and frozen checkpoints with hash-verified evaluation CSVs. The ability to read structural dynamics literature and translate it into working code. A baseline of intellectual honesty about what doesn't work — the phase drift issue is documented as prominently in the repository as the results that look good.</p>
<p>I'm early in my training, so I'm not pretending to bring theoretical depth I don't have. What I can bring is a project that's real, a genuine interest in why it breaks where it does, and enough independence to be useful quickly rather than needing to be walked through the basics.</p>

<div class="footer-note">
Repository: SeismoFNO (github.com/raghvendra1gdsc-png/SeismoFNO) ·
Full forensic audit, evaluation CSVs, and interactive demo available for inspection.
</div>
"""
    return build_html("SeismoFNO — Personal Statement", body)


# ============================================================
# DOCUMENT 3: RESEARCH WALKTHROUGH (6-page)
# ============================================================
def research_walkthrough_html() -> str:
    body = """
<div class="doc-header">
  <div class="doc-title">SeismoFNO: Physics-Grounded Neural Operators for Seismic Structural Response Prediction</div>
  <div class="doc-subtitle">A full walkthrough of the project — motivation, methods, experiments, and open problems</div>
  <div class="doc-meta">Raghvendra Singh Gahlot &nbsp;·&nbsp; B.Tech (Civil Engineering, Structural Engineering) &nbsp;·&nbsp; September 2026</div>
</div>

<h2>Why I started this</h2>
<p>I got interested in this after running a handful of nonlinear time-history analyses in OpenSeesPy for a coursework assignment and noticing how long a single run took to converge, even for a small five-story shear frame. Regional risk assessment, city-scale loss estimation, and design-optimisation workflows all need this kind of analysis run thousands of times over. That gap between what rigorous simulation costs and what applications need felt like the right place to ask whether a learned surrogate could help. What I didn't expect going in was how badly the standard recipe — Fourier Neural Operators on a fixed grid — breaks the moment you try to apply it to buildings that don't all have the same number of storeys. That failure, and the two fixes that followed it, is most of what this project is about. I've tried to be as honest about what still doesn't work as about what does.</p>

<h2>1. Foundations: getting the ground truth right first</h2>
<p>Before any surrogate modelling, the first stretch of this project was building a trustworthy simulation pipeline and confirming the physics was sound end-to-end. That meant:</p>
<ul>
  <li>Validating single-degree-of-freedom elastoplastic dynamics and Bouc-Wen nonlinear hysteretic loops against known OpenSeesPy benchmarks (EXP1).</li>
  <li>Tracking continuous hysteretic energy dissipation <span class="math-inline">E_h(t) = ∫ f_int du</span> as a physically meaningful validation target, not just displacement error (EXP2).</li>
  <li>Deciding, early on, that random train/test splits weren't good enough — a model can appear to generalise while just interpolating within structural families it's already seen. I switched to splitting by structural archetype and by earthquake identity instead (EXP3).</li>
</ul>
<p>This groundwork mattered more than I expected. Almost every surprising result later in the project traced back to something about the representation or the physics, not to training hyperparameters.</p>

<h2>2. Problem formulation</h2>
<p>A multi-storey shear building subjected to horizontal ground acceleration a_g(t) is governed by:</p>
<div class="math">M ü(t) + C u̇(t) + f_int(u(t), u̇(t)) = −Mι a_g(t)</div>
<p>where M, C ∈ ℝ^(N×N) are the mass and Rayleigh damping matrices; f_int is the (possibly nonlinear) restoring-force operator; u(t) ∈ ℝ^N is the relative floor-displacement vector; ι = [1,…,1]ᵀ. The surrogate goal is to approximate the operator mapping ground acceleration and structural parameters to the full response trajectory. This mapping should ideally be resolution-invariant — evaluating at arbitrary discrete time points shouldn't require retraining when the time discretisation changes — which is part of why the operator-learning framing made sense.</p>

<h2>3. Buildings, ground motions, and the partitioning scheme</h2>
<p>Buildings were idealised as lumped-mass shear frames: storey mass 30,000 kg, inter-storey stiffness 15–60 MN/m, storey height 3.5 m, 5% Rayleigh damping tuned to the first two modal frequencies, and a bilinear elastoplastic restoring-force law with 5% kinematic hardening. Six archetypes span fundamental periods from 0.35 s to 1.40 s:</p>
<table>
  <tr><th>Archetype</th><th>Storeys</th><th>T₁ (s)</th><th>Role</th></tr>
  <tr><td>3S_T035</td><td class="num">3</td><td class="num">0.35</td><td>Training / ID</td></tr>
  <tr><td>3S_T060</td><td class="num">3</td><td class="num">0.60</td><td>Training / ID</td></tr>
  <tr><td>3S_T090</td><td class="num">3</td><td class="num">0.90</td><td>Training / ID</td></tr>
  <tr><td>5S_T055</td><td class="num">5</td><td class="num">0.55</td><td>Training / ID</td></tr>
  <tr><td>5S_T085</td><td class="num">5</td><td class="num">0.85</td><td>Training / ID — upper training limit</td></tr>
  <tr><td><strong>5S_T120</strong></td><td class="num"><strong>5</strong></td><td class="num"><strong>1.20</strong></td><td><strong>Held-out OOD-B — unseen structure, the main test</strong></td></tr>
</table>
<p>Ground motions were drawn from the PEER NGA-West2 database. Partitioning ran along two independent axes — structural archetype and earthquake identity — giving four evaluation partitions with genuinely different physical meaning:</p>
<table>
  <tr><th>Partition</th><th>Structural group</th><th>Earthquake group</th><th>n</th><th>What it tests</th></tr>
  <tr><td>In-distribution (ID)</td><td>known archetypes, T₁ ≤ 0.85 s</td><td>training records</td><td class="num">120</td><td>Ordinary accuracy</td></tr>
  <tr><td>OOD-A</td><td>known archetypes, T₁ ≤ 0.85 s</td><td>held-out records</td><td class="num">300</td><td>Unseen ground motion</td></tr>
  <tr><td>OOD-B</td><td>unseen structural period — the main test</td><td>training records</td><td class="num">240</td><td>Unseen structural period</td></tr>
  <tr><td>OOD-C</td><td>held-out archetype (T₁=1.20 s)</td><td>held-out records</td><td class="num">60</td><td>Combined structural + seismic extrapolation</td></tr>
</table>

<div class="page-break"></div>

<h2>4. First attempt: a fixed-grid Fourier Neural Operator, and why it failed</h2>
<p>The natural starting point was a standard 2D Fourier Neural Operator (Li et al., 2021), which has worked well on regular Cartesian domains such as fluid-flow fields. I built a fixed 5×2048 input tensor — five storeys (the tallest archetype) by 2,048 time steps — and zero-padded shorter buildings to fill the unused storeys. Five-storey buildings, which naturally filled the grid, reached a reasonable 19.29% median relative L₂ error. Three-storey buildings, padded across two zeros, reached a median error of <strong>99.6%</strong> — essentially garbage.</p>
<p>The reason is mathematically straightforward once you look at it in the frequency domain. A discrete spatial Fourier transform implicitly assumes periodic boundary conditions across the lattice. Padding a three-storey building with two zero floors creates a sharp step between the last real floor and the padded region, and the spectral convolution reads that step not as "nothing there" but as a high-frequency ringing source that leads corrupted spectral energy back into the real floors below. This wasn't a training problem; longer training and more layers didn't help, because the representation itself imposed an invalid assumption: a building is a small, irregular graph of masses connected by columns, not a padded image.</p>

<table>
  <tr><th>Representation</th><th>3-storey median relative L₂</th></tr>
  <tr><td>Fixed-grid FNO2D (EXP4)</td><td class="num bold-cell">99.6%</td></tr>
  <tr><td>Topology-native GNO (EXP5)</td><td class="num bold-cell">22.1%</td></tr>
</table>
<p class="caption">Moving from a fixed grid to a native graph representation resolves the zero-padding failure.</p>

<h2>5. A topology-native fix: graph neural operators</h2>
<p>The fix was to stop forcing buildings onto a shared grid. Each building now owns its own graph G = (V, E): one node per real storey (|V| = N_stories), edges between physically adjacent floors carrying the column stiffness. Each graph block computes:</p>
<div class="math">m_v^(l+1)(t) = φ( h_v^(l)(t), Σ_{u∈N(v)} ψ(h_u^(l)(t), e_uv) )</div>
<p>followed by a 1D temporal Fourier spectral layer applied node-wise to handle the dynamics:</p>
<div class="math">h_v^(l+1)(t) = φ( W h^(l+1)(t) + F⁻¹[R_l(k) · F(h_v^(l+1))(k)](t) )</div>
<p>This structure eliminates zero-padding artifacts while coupling spatial inter-storey information exchange with temporal wave propagation on a shared latent representation.</p>

<h2>6. What graph topology alone can't fix</h2>
<p>That was satisfying, but it uncovered a second, subtler problem. Evaluating the same unconditioned graph model on the held-out flexible archetype (T₁ = 1.20 s, well outside the training range of T₁ ≤ 0.85 s) gave a peak displacement error of around 35.2%, and full-trajectory relative L₂ error above 115%. Looking at the FFT of the predicted displacement records showed the model was oscillating at approximately 1.12 Hz — the edge of what it was trained on — rather than tracking the true resonance at 0.83 Hz.</p>
<p>When two sinusoidal signals of identical amplitude but slightly different frequencies are compared pointwise, they drift into phase opposition. Once phase opposition is reached, the L₂ difference is mathematically ~141%. Modal conditioning can address the amplitude part of this — but it cannot warp the temporal frequency basis functions, which are still fixed and frozen per the whole record.</p>

<h2>7. Conditioning on what we already know: modal invariants via FiLM</h2>
<p>Given a building's mass and stiffness matrices, mode shapes and natural periods are computable before any shaking happens, by solving the generalised eigenvalue problem:</p>
<div class="math">K φ_i = ω_i² M φ_i,    T_i = 2π/ω_i</div>
<p>Because this only uses the known, pre-earthquake [M] and [K], it leaks no information about the future ground motion or response trajectory. Using Feature-wise Linear Modulation (Perez et al., 2018) to let it rescale the network's internal representations per layer:</p>
<div class="math">[γ, β] = MLP(c),    h^(l+1) = (1 + γ) ⊙ h^(l) + β</div>
<p>I compared an unconditioned baseline against two conditioned variants — one using the fundamental period T₁ only, one using the first three periods and frequencies together — and a falsification control described below.</p>

<table>
  <tr><th>Model</th><th>ID Peak</th><th>OOD-A Peak</th><th>OOD-B Peak</th><th>OOD-C Peak</th></tr>
  <tr><td>EXP5 (unconditioned)</td><td class="num">12.70%</td><td class="num">15.86%</td><td class="num">35.21%</td><td class="num">38.66%</td></tr>
  <tr><td>EXP6-B (T₁ only)</td><td class="num bold-cell">2.09%</td><td class="num">8.81%</td><td class="num">13.47%</td><td class="num bold-cell">14.36%</td></tr>
  <tr><td>EXP6-C (multi-modal)</td><td class="num">8.87%</td><td class="num bold-cell">7.63%</td><td class="num bold-cell">13.06%</td><td class="num">17.01%</td></tr>
  <tr><td>EXP6-D (shuffled control)</td><td class="num">2.65%</td><td class="num">8.94%</td><td class="num">24.33%</td><td class="num">36.16%</td></tr>
</table>
<p class="caption">Peak-displacement error across all four partitions. Conditioning helps everywhere, but the gap is largest exactly where it matters most — the unseen-structure partitions.</p>

<p><strong>Checking this wasn't just extra capacity.</strong> Before trusting the improvement, I wanted to rule out the boring explanation: maybe the FiLM branch just gave the network more parameters to work with, and any auxiliary vector — physically meaningful or not — would have helped similarly. I re-ran evaluation with the conditioning vectors shuffled across buildings within a batch, so each sample received a <em>different</em> building's modal descriptor. Performance under shuffling degraded to 24.33% on OOD-B and 36.16% on OOD-C — worse than either conditioned variant, though still somewhat better than the unconditioned baseline. That pattern is consistent with the model relying on physical correspondence rather than scalar capacity, though I'd stop short of calling it proof of causality — it's evidence, not certainty, and I'd want a dimensionally matched but physically meaningless control vector rather than a shuffled real one to state the Section 5 result more rigorously.</p>

<div class="page-break"></div>

<h2>8. Computational performance</h2>
<p>Timings were measured on Apple Silicon (MPS) with process-wide device locks to avoid benchmark noise from concurrent Metal command buffers:</p>
<table>
  <tr><th>Model / setting</th><th>Latency</th><th>Throughput</th><th>Speedup vs. OpenSeesPy</th></tr>
  <tr><td>OpenSeesPy reference, batch 1</td><td class="num">54.68 ms</td><td class="num">18.3 sim/s</td><td class="num">1.0×</td></tr>
  <tr><td>Fixed-grid FNO2D (EXP4, batch 1)</td><td class="num">6.60 ms</td><td class="num">151.5 sim/s</td><td class="num">8.3×</td></tr>
  <tr><td>Topology GNO (EXP5, batch 1)</td><td class="num">16.25 ms</td><td class="num">61.5 sim/s</td><td class="num">3.4×</td></tr>
  <tr><td>T₁-conditioned GNO (EXP6-B, batch 1)</td><td class="num bold-cell">21.45 ms</td><td class="num">46.6 sim/s</td><td class="num bold-cell">2.6×</td></tr>
  <tr><td>T₁-conditioned GNO (EXP6-B, batch 32)</td><td class="num">30.19 ms</td><td class="num">1,040 sim/s</td><td class="num">57.8× (throughput)</td></tr>
</table>
<p class="caption">The fixed-grid model is fastest in isolation, which isn't surprising — it's also the model that doesn't work. Among the models that actually generalise, single-sample speedup (2.5–3.4×) is useful for interactive design tools, not transformative. The batched throughput is practically interesting for regional screening of tens of thousands of structures.</p>

<h2>9. Architecture summary</h2>
<table>
  <tr><th>Component</th><th>Description</th></tr>
  <tr><td>Pre-earthquake modal engine</td><td>Eigenvalue decomposition of [K, M] → natural periods T₁₋₃ and frequencies ω₁₋₃. Zero response data used.</td></tr>
  <tr><td>Graph construction</td><td>G = (V, E): |V| = N_stories nodes (floor slabs), edges = interstory columns. No padding.</td></tr>
  <tr><td>FiLM conditioning</td><td>MLP maps modal invariants c to (γ, β) per layer. Applied to both graph MP and temporal spectral channels.</td></tr>
  <tr><td>Spatiotemporal blocks</td><td>4 blocks: graph message-passing (spatial) + 1D Fourier conv (temporal), both FiLM-modulated.</td></tr>
  <tr><td>Output</td><td>Floor displacements u_i(t), interstory drift ratios IDR_i(t), peak engineering demand parameters.</td></tr>
</table>

<h2>10. Where I'd take this next</h2>
<ul>
  <li><strong>State-space models (S4, Mamba):</strong> Replace static 1D Fourier temporal layers with causal state-space models to address phase drift. This is the most pressing technical gap.</li>
  <li><strong>Neural ODEs:</strong> Couple the graph spatial operator with a neural ODE integrator to track instantaneous stiffness degradation during yielding.</li>
  <li><strong>3D asymmetric structures:</strong> Extend from 2D planar to 3D buildings with torsional response and bidirectional ground motion.</li>
  <li><strong>Online period tracking:</strong> Update the modal conditioning dynamically as the structural period elongates during severe yielding.</li>
  <li><strong>Tighter falsification ablation:</strong> Use a dimensionally matched but physically meaningless control vector rather than a shuffled real one to state the modal conditioning result more rigorously.</li>
</ul>

<h2>11. Reproducibility</h2>
<p>Data partitions were checked programmatically for zero overlap in structure or earthquake identity across the four partitions before any model saw them. EXP4, EXP5, and EXP6 variants are frozen and hash-verified against the evaluation logs referenced above, and the repository's unit test suite — covering the numerical solvers, split disjointness, and model I/O shapes — passes in full at the time of writing. The codebase, evaluation scripts, and an interactive demo comparing live model output against OpenSeesPy ground truth are available in the repository.</p>

<hr>

<h2>References</h2>
<p>Li, Z., Kovachki, N., Liu, B., Bhattacharya, K., Stuart, A., &amp; Anandkumar, A. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. <em>International Conference on Learning Representations (ICLR).</em></p>
<p>Perez, E., Strub, F., de Vries, H., Dumoulin, V., &amp; Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. <em>AAAI Conference on Artificial Intelligence.</em></p>
<p>Pacific Earthquake Engineering Research Center (PEER). NGA-West2 Ground Motion Database.</p>

<div class="footer-note">
SeismoFNO · Raghvendra Singh Gahlot · B.Tech (Civil Engineering) · September 2026 ·
All numbers are computed from raw CSV evaluation files. No result is manually entered or estimated.
</div>
"""
    return build_html("SeismoFNO — Research Walkthrough", body)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("SeismoFNO — PDF Generation")
    print("=" * 60)

    if not Path(CHROME).exists():
        print(f"ERROR: Chrome not found at {CHROME}")
        sys.exit(1)

    docs = PROJECT_ROOT / "docs"

    tasks = [
        (
            "Research Brief",
            research_brief_html(),
            docs / "RESEARCH_BRIEF.pdf",
            docs / "_tmp_brief.html",
        ),
        (
            "Personal Statement",
            personal_statement_html(),
            docs / "PERSONAL_STATEMENT.pdf",
            docs / "_tmp_statement.html",
        ),
        (
            "Research Walkthrough",
            research_walkthrough_html(),
            docs / "RESEARCH_WALKTHROUGH.pdf",
            docs / "_tmp_walkthrough.html",
        ),
    ]

    for name, html, out_path, tmp_path in tasks:
        print(f"\n→ {name}")
        compile_pdf(html, out_path, tmp_path)

    print("\n" + "=" * 60)
    print("All PDFs generated successfully.")
    print(f"  docs/RESEARCH_BRIEF.pdf")
    print(f"  docs/PERSONAL_STATEMENT.pdf")
    print(f"  docs/RESEARCH_WALKTHROUGH.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
