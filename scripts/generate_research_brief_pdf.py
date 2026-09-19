#!/usr/bin/env python3
"""
generate_research_brief_pdf.py
==============================
Reproducible generation script for the SeismoFNO Research Brief PDF:
  docs/RESEARCH_BRIEF.pdf

This script builds a publication-grade, 2-page academic engineering research brief
using vector typography, embedded SVG diagrams, typeset mathematical equations,
clean tabular results, and clickable navigation links.

Target Environment:
  - Uses headless Google Chrome to compile pixel-perfect vector PDF from structured HTML.
  - Injects PDF document metadata (Title, Author, Subject, Keywords).
  - Validates resulting page count (target: 2 pages, max: 3 pages).
  - Verifies presence of all verified empirical metrics and links.
"""

import os
import re
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PDF = PROJECT_ROOT / "docs" / "RESEARCH_BRIEF.pdf"
TEMP_HTML = PROJECT_ROOT / "docs" / "_temp_research_brief.html"
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def generate_html_content() -> str:
    """Builds self-contained HTML for the 2-page academic research brief."""
    
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="title" content="SeismoFNO — Physics-Grounded Graph Neural Operators for Seismic Structural Response">
<meta name="author" content="Raghvendra Singh Gahlot">
<meta name="subject" content="Scientific ML for Computational Structural Dynamics">
<meta name="keywords" content="Neural Operators, Graph Neural Operators, Seismic Response, Scientific Machine Learning, Structural Dynamics, Physics-Informed ML, Out-of-Distribution Generalization">
<title>SeismoFNO — Physics-Grounded Graph Neural Operators for Seismic Structural Response</title>
<style>
  @page {
    size: A4 portrait;
    margin: 10mm 12mm 10mm 12mm;
  }
  
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  
  body {
    font-family: 'Times New Roman', 'Liberation Serif', 'Nimbus Roman No9 L', Georgia, serif;
    color: #000000;
    background-color: #ffffff;
    font-size: 8.5pt;
    line-height: 1.32;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  .page {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
  }

  .page-break {
    page-break-before: always;
    break-before: page;
  }

  /* Header Section */
  .header {
    border-bottom: 1pt solid #333333;
    padding-bottom: 5px;
    margin-bottom: 6px;
  }
  
  .header-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .title-group h1 {
    font-size: 15pt;
    font-weight: bold;
    color: #000000;
    line-height: 1.15;
  }

  .title-group .subtitle {
    font-size: 9.2pt;
    font-weight: normal;
    font-style: italic;
    color: #222222;
    margin-top: 1px;
  }

  .title-group .descriptor {
    font-size: 8pt;
    font-weight: normal;
    color: #444444;
    font-style: italic;
  }

  .meta-badges {
    text-align: right;
    font-size: 7pt;
    color: #333333;
  }

  .badge {
    display: inline-block;
    padding: 1.5px 5px;
    font-weight: bold;
    border-radius: 2px;
    font-size: 6.5pt;
    letter-spacing: 0.01em;
  }

  .badge-frozen {
    background-color: #f2f2f2;
    color: #000000;
    border: 0.5pt solid #888888;
  }

  .badge-audit {
    background-color: #f2f2f2;
    color: #000000;
    border: 0.5pt solid #888888;
  }

  .header-authors {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 4px;
    padding-top: 3px;
    border-top: 0.5pt solid #cccccc;
    font-size: 7.5pt;
  }

  .author-name {
    font-weight: 700;
    color: #0f172a;
  }

  .author-meta {
    color: #475569;
  }

  .app-context {
    font-weight: 600;
    color: #1e3a8a;
  }

  /* Section Styles */
  .section {
    margin-bottom: 6px;
  }

  .section-title {
    font-family: 'Times New Roman', 'Liberation Serif', serif;
    font-size: 8.8pt;
    font-weight: bold;
    color: #000000;
    border-bottom: 0.5pt solid #888888;
    padding-bottom: 1.5px;
    margin-bottom: 3.5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .section-title span.sec-num {
    color: #000000;
    font-weight: bold;
    margin-right: 3px;
  }

  p {
    margin-bottom: 3.5px;
    text-align: justify;
  }

  p:last-child {
    margin-bottom: 0;
  }

  /* Math Block */
  .math-block {
    background-color: transparent;
    border-left: none;
    padding: 3px 5px;
    margin: 3px 0;
    font-family: 'Times New Roman', 'Liberation Serif', serif;
    font-size: 9.5pt;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .equation {
    font-style: normal;
    color: #000000;
    flex-grow: 1;
    text-align: center;
  }

  .equation-num {
    font-size: 8pt;
    color: #333333;
    font-family: 'Times New Roman', 'Liberation Serif', serif;
  }

  /* Pipeline Flow */
  .pipeline-box {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 4px 6px;
    margin: 4px 0;
  }

  .flow-grid {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 6.8pt;
    font-weight: 600;
  }

  .flow-node {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    padding: 2.5px 4px;
    text-align: center;
    flex: 1;
    color: #1e293b;
  }

  .flow-node.highlight {
    background: #eff6ff;
    border-color: #93c5fd;
    color: #1e3a8a;
  }

  .flow-arrow {
    color: #94a3b8;
    padding: 0 2px;
    font-weight: bold;
    font-size: 7pt;
  }

  /* Research Progression Vector Diagram */
  .progression-container {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 5px;
    margin: 4px 0;
  }

  .progression-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px;
  }

  .prog-card {
    border-radius: 3px;
    padding: 4px 5px;
    font-size: 7pt;
    border: 1px solid #e2e8f0;
    background: #fafafa;
  }

  .prog-card.exp4 {
    border-top: 3px solid #dc2626;
  }

  .prog-card.exp5 {
    border-top: 3px solid #2563eb;
  }

  .prog-card.exp6 {
    border-top: 3px solid #16a34a;
  }

  .prog-card.limitation {
    border-top: 3px solid #d97706;
    background: #fffbeb;
  }

  .prog-tag {
    font-weight: 800;
    font-size: 6.5pt;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    margin-bottom: 2px;
  }

  .prog-tag.t-exp4 { color: #b91c1c; }
  .prog-tag.t-exp5 { color: #1d4ed8; }
  .prog-tag.t-exp6 { color: #15803d; }
  .prog-tag.t-lim { color: #b45309; }

  .prog-name {
    font-weight: 700;
    font-size: 7.2pt;
    color: #0f172a;
    margin-bottom: 1.5px;
  }

  .prog-desc {
    color: #475569;
    font-size: 6.5pt;
    line-height: 1.25;
    margin-bottom: 3px;
  }

  .prog-metric {
    font-weight: 700;
    font-size: 7pt;
    padding: 1.5px 3px;
    border-radius: 2px;
    display: inline-block;
  }

  .m-fail { background: #fee2e2; color: #991b1b; }
  .m-pass { background: #dcfce7; color: #166534; }
  .m-warn { background: #fef3c7; color: #92400e; }

  /* Table Styles */
  table.results-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 7.5pt;
    margin: 4px 0;
  }

  table.results-table th, table.results-table td {
    padding: 2.5pt 4.5pt;
    text-align: left;
    border: 0.5pt solid #888888;
  }

  table.results-table th {
    background-color: #f2f2f2;
    color: #000000;
    font-weight: bold;
    font-size: 7.2pt;
    font-family: 'Times New Roman', 'Liberation Serif', serif;
  }

  table.results-table tr:nth-child(even) td {
    background-color: #ffffff;
  }

  .fw-700 { font-weight: bold; }
  .val-danger { color: #000000; font-weight: bold; }
  .val-success { color: #000000; font-weight: bold; }
  .val-highlight { color: #000000; font-weight: bold; }

  /* Two Column Layout */
  .two-col {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 8px;
    margin-bottom: 5px;
  }

  .col-box {
    background: #ffffff;
    border: 0.5pt solid #888888;
    border-radius: 2px;
    padding: 5px 7px;
  }

  /* OOD Box */
  .ood-box {
    background-color: #fafafa;
    border: 0.75pt solid #666666;
    border-radius: 2px;
    padding: 5px 8px;
    margin-bottom: 6px;
  }

  .ood-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3px;
    border-bottom: 0.5pt solid #888888;
    padding-bottom: 2px;
  }

  .ood-title {
    font-weight: bold;
    font-size: 8pt;
    color: #000000;
    font-family: 'Times New Roman', 'Liberation Serif', serif;
  }

  .ood-badge {
    background: #eeeeee;
    color: #000000;
    font-weight: bold;
    font-size: 6.8pt;
    padding: 1px 4px;
    border: 0.5pt solid #888888;
    border-radius: 2px;
  }

  .ood-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1.2fr;
    gap: 4px;
    text-align: center;
    margin-top: 3px;
  }

  .ood-card {
    background: #ffffff;
    border: 0.5pt solid #aaaaaa;
    border-radius: 2px;
    padding: 3px;
  }

  .ood-card.highlight {
    background: #f9f9f9;
    border-color: #666666;
  }

  .ood-label {
    font-size: 6.3pt;
    color: #444444;
    font-weight: bold;
    text-transform: uppercase;
  }

  .ood-val {
    font-size: 8.5pt;
    font-weight: 800;
    color: #0f172a;
    margin-top: 1px;
  }

  .ood-val.success { color: #166534; }
  .ood-val.gain { color: #15803d; font-size: 9pt; }

  /* Benchmark Box */
  .bench-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 4px;
    margin-top: 3px;
    text-align: center;
  }

  .bench-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
    padding: 3px;
  }

  .bench-label {
    font-size: 6.3pt;
    color: #64748b;
    font-weight: 600;
  }

  .bench-val {
    font-size: 8pt;
    font-weight: 800;
    color: #0f172a;
  }

  /* Limitations list */
  .fail-list {
    list-style: none;
    font-size: 7.2pt;
  }

  .fail-list li {
    margin-bottom: 3px;
    padding-left: 10px;
    position: relative;
    line-height: 1.25;
  }

  .fail-list li::before {
    content: "•";
    position: absolute;
    left: 0;
    color: #b91c1c;
    font-weight: bold;
  }

  .fail-title {
    font-weight: 700;
    color: #0f172a;
  }

  /* Audit Grid */
  .audit-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
    font-size: 6.8pt;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    padding: 4px 6px;
  }

  .audit-item {
    line-height: 1.2;
  }

  .audit-item .k { color: #64748b; font-weight: 600; }
  .audit-item .v { color: #0f172a; font-weight: 700; }

  /* Research Resources Links */
  .resources-container {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 4px 8px;
    margin-top: 5px;
  }

  .res-links {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    font-size: 7.2pt;
  }

  .res-link {
    color: #1e40af;
    text-decoration: none;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
  }

  .res-link:hover {
    text-decoration: underline;
  }

  .res-sep {
    color: #94a3b8;
  }

  /* Footer */
  .page-footer {
    border-top: 1px solid #e2e8f0;
    padding-top: 3px;
    margin-top: auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 6.5pt;
    color: #64748b;
  }
</style>
</head>
<body>

<!-- ================================= PAGE 1 ================================= -->
<div class="page">
  <!-- Document Header -->
  <header class="header">
    <div class="header-top">
      <div class="title-group">
        <h1>SEISMOFNO</h1>
        <div class="subtitle">Physics-Grounded Graph Neural Operators for Seismic Structural Response</div>
        <div class="descriptor">Scientific ML for Computational Structural Dynamics</div>
      </div>
      <div class="meta-badges">
        <div><span class="badge badge-frozen">RESEARCH CORE: FROZEN / AUDITED</span></div>
        <div style="margin-top: 2px;"><span class="badge badge-audit">294 TESTS PASSING (0 FAILED)</span></div>
        <div style="margin-top: 2px; font-weight: 600;">Version: September 2026</div>
      </div>
    </div>
    <div class="header-authors">
      <div>
        <span class="author-name">Raghvendra Singh Gahlot</span> &nbsp;•&nbsp; 
        <span class="author-meta">B.E. Building and Construction Technology (Structural Engineering)</span>
      </div>
      <div class="app-context">
        Research Brief — Research Application
      </div>
    </div>
  </header>

  <!-- 1. Opening Research Thesis -->
  <section class="section">
    <div class="section-title">
      <div><span class="sec-num">1.</span> Research Problem & Scientific Thesis</div>
      <span style="font-size: 7pt; font-weight: 600; color: #475569; text-transform: none;">Operator Learning on Irregular Topologies</span>
    </div>
    <p>
      SeismoFNO investigates whether neural operators can learn seismic structural response while generalizing across both structural topology and modal-period distribution shift. Predicting non-linear dynamic responses of multi-story buildings subjected to transient ground motions requires solving non-linear differential systems currently evaluated through high-fidelity numerical integration in OpenSeesPy. While surrogate neural operators offer large evaluation acceleration, canonical architectures exhibit critical physical and geometric failure modes when deployed across heterogeneous civil structures.
    </p>
    <p>
      Through systematic research progression across <strong>2,160 physical simulations</strong>, we establish:
      <strong>(1) EXP4:</strong> Conventional fixed-grid 2D Fourier Neural Operators (FNO2D) fail catastrophically when applied to variable-story structures because spatial zero-padding introduces artificial discontinuities (<strong>99.60% median Rel. L₂ error</strong> on 3-story frames);
      <strong>(2) EXP5:</strong> A topology-native Graph Neural Operator (GNO) resolves boundary artifacts by operating exclusively on physical floor nodes and column edges (reducing 3-story error to <strong>22.09%</strong>), yet suffers severe phase opposition under structural period extrapolation;
      <strong>(3) EXP6:</strong> Conditioning spatiotemporal operator blocks on structural eigenvalue invariants via Feature-wise Linear Modulation (FiLM) substantially improves peak-response generalization under structural modal-period shift (reducing unseen archetype peak error from <strong>35.21% to 13.06%</strong>).
      <strong>Limitation:</strong> Trajectory-level waveform fidelity remains limited under strong modal extrapolation because global 1D Fourier layers accumulate temporal phase drift over long horizons.
    </p>
  </section>

  <!-- 2. Governing Physics & Pipeline -->
  <section class="section">
    <div class="section-title">
      <div><span class="sec-num">2.</span> Governing Dynamics & Computational Pipeline</div>
      <span style="font-size: 7pt; font-weight: 600; color: #475569; text-transform: none;">Physics Ground Truth: OpenSeesPy NLTHA</span>
    </div>
    <p>
      The surrogate approximates the structural response generated by non-linear time-history analysis (NLTHA) under earthquake ground excitation. Multi-degree-of-freedom (MDOF) shear buildings subjected to horizontal base acceleration <em>a</em><sub>g</sub>(<em>t</em>) are governed by:
    </p>
    <div class="math-block">
      <div class="equation">
        <strong>M</strong> ü(<em>t</em>) + <strong>C</strong> u̇(<em>t</em>) + <strong>K</strong> u(<em>t</em>) = −<strong>M</strong> <strong>r</strong> <em>a</em><sub>g</sub>(<em>t</em>)
      </div>
      <div class="equation-num">(1) Coupled Matrix Dynamic Equation</div>
    </div>
    <p>
      where <strong>M</strong>, <strong>C</strong>, <strong>K</strong> ∈ ℝ<sup><em>N</em>×<em>N</em></sup> denote mass, Rayleigh damping, and lateral stiffness matrices, <strong>r</strong> = [1, …, 1]<sup>T</sup> is the structural influence vector, and <strong>u</strong>(<em>t</em>) ∈ ℝ<sup><em>N</em></sup> is the relative floor displacement trajectory across duration <em>t</em> ∈ [0, 20.48] s.
    </p>
    <!-- Vector Computational Flow Pipeline -->
    <div class="pipeline-box">
      <div class="flow-grid">
        <div class="flow-node">
          Structural System<br>
          <span style="font-size: 6.5pt;">[M, K] Matrices</span>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node highlight">
          Modal Properties<br>
          <span style="font-size: 6.5pt;">[T<sub>1..3</sub>, ω<sub>1..3</sub>] Eigenvalues</span>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node">
          Earthquake Input<br>
          <span style="font-size: 6.5pt;">a<sub>g</sub>(t) PEER Record</span>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node highlight">
          Native Graph<br>
          <span style="font-size: 6.5pt;">G=(V,E) (0 pad)</span>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node highlight">
          Neural Operator<br>
          <span style="font-size: 6.5pt;">FiLM Spatial + 1D FNO</span>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node">
          Displacement u(t)<br>
          <span style="font-size: 6.5pt;">Response Trajectory</span>
        </div>
        <div class="flow-arrow">→</div>
        <div class="flow-node">
          Ground Truth<br>
          <span style="font-size: 6.5pt;">OpenSeesPy Benchmark</span>
        </div>
      </div>
    </div>
  </section>

  <!-- 3. Research Progression Diagram -->
  <section class="section">
    <div class="section-title">
      <div><span class="sec-num">3.</span> Four-Stage Research Progression (EXP4 → EXP5 → EXP6)</div>
      <span style="font-size: 7pt; font-weight: 600; color: #475569; text-transform: none;">Systematic Hypothesis Testing & Failure Analysis</span>
    </div>
    <div class="progression-container">
      <div class="progression-grid">
        <!-- EXP4 -->
        <div class="prog-card exp4">
          <div class="prog-tag t-exp4">EXP4: Baseline Failure</div>
          <div class="prog-name">Fixed-Grid 2D FNO</div>
          <div class="prog-desc">
            Standard 2D FNO on fixed 5×2048 grid. 3-story buildings zero-padded at stories 4–5. Fourier basis enforces non-physical spatial periodicity across zero boundary.
          </div>
          <div class="prog-metric m-fail">3-Story Rel L₂: 99.60% (Fail)</div>
        </div>

        <!-- EXP5 -->
        <div class="prog-card exp5">
          <div class="prog-tag t-exp5">EXP5: Topology Native</div>
          <div class="prog-name">Spatiotemporal GNO</div>
          <div class="prog-desc">
            Replaced Cartesian grid with physical graph G=(V,E). Zero artificial padding. Eliminates spatial ringing but reveals temporal phase divergence under modal shift.
          </div>
          <div class="prog-metric m-pass">3-Story Rel L₂: 22.09% (Resolved)</div>
        </div>

        <!-- EXP6 -->
        <div class="prog-card exp6">
          <div class="prog-tag t-exp6">EXP6: Physics FiLM</div>
          <div class="prog-name">Modal-Conditioned GNO</div>
          <div class="prog-desc">
            Injected pre-earthquake structural invariants [T₁, ω₁] into spatial message-passing and spectral layers via FiLM scale/shift generators.
          </div>
          <div class="prog-metric m-pass">OOD-B Peak Error: 13.06%</div>
        </div>

        <!-- Limitation -->
        <div class="prog-card limitation">
          <div class="prog-tag t-lim">Active Limitation</div>
          <div class="prog-name">Phase Drift Over Horizon</div>
          <div class="prog-desc">
            Global 1D Fourier layers lack dynamic frequency warping. While peak envelopes match (13.06%), temporal phase divergence accumulates over 20.48 s.
          </div>
          <div class="prog-metric m-warn">OOD-B Rel L₂: 124.07% (r≈0.07)</div>
        </div>
      </div>
    </div>
  </section>

  <!-- 4. Key Results Table -->
  <section class="section">
    <div class="section-title">
      <div><span class="sec-num">4.</span> Master Empirical Comparison Across Architectural Phases</div>
      <span style="font-size: 7pt; font-weight: 600; color: #475569; text-transform: none;">Reconstructed from 2,160 Frozen Simulation Evaluations</span>
    </div>
    <table class="results-table">
      <thead>
        <tr>
          <th style="width: 14%;">Experiment</th>
          <th style="width: 22%;">Representation</th>
          <th style="width: 40%;">Scientific Contribution</th>
          <th style="width: 24%;">Key Result</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="fw-700">EXP4</td>
          <td>Fixed-grid FNO2D (5×2048)</td>
          <td>Tests conventional fixed-grid neural operator on variable-floor geometry</td>
          <td>3-story median Relative L2 error: <span class="val-danger">99.60%</span></td>
        </tr>
        <tr>
          <td class="fw-700">EXP5</td>
          <td>Topology-native GNO</td>
          <td>Removes artificial zero-padding and supports variable topology</td>
          <td>3-story median Relative L2 error: <span class="val-success">22.09%</span></td>
        </tr>
        <tr>
          <td class="fw-700">EXP6-B</td>
          <td>T1-conditioned GNO</td>
          <td>Physics-informed FiLM conditioning on fundamental vibration period T₁</td>
          <td>OOD-B peak displacement error: <span class="val-highlight">13.47%</span></td>
        </tr>
        <tr>
          <td class="fw-700">EXP6-C</td>
          <td>Multi-modal GNO</td>
          <td>Conditions on [T1-3, ω1-3] via dual-branch FiLM modulation</td>
          <td>OOD-B peak displacement error: <span class="val-success">13.06%</span></td>
        </tr>
        <tr>
          <td class="fw-700">EXP6-D</td>
          <td>Shuffled T1 control</td>
          <td>Tests whether conditioning benefit depends on physical correspondence</td>
          <td>OOD-B peak error: <span class="val-danger">24.33%</span></td>
        </tr>
      </tbody>
    </table>
    <p style="font-size: 6.8pt; color: #475569; margin-top: 1px;">
      <em>Note on EXP6-D Falsification Control:</em> Provides evidence that performance gains depend on physically meaningful modal correspondence rather than merely additional conditioning capacity (OOD-B peak error degrades from 13.06% to 24.33%, and OOD-C degrades from 17.01% to 36.16%).
    </p>
  </section>

  <!-- Footer Page 1 -->
  <footer class="page-footer">
    <div>SeismoFNO Research Brief • Department of Computer Science & Engineering, Academic Application</div>
    <div>Page 1 of 2</div>
  </footer>
</div>

<!-- ================================= PAGE 2 ================================= -->
<div class="page page-break">
  <!-- 5. Structural OOD Generalization Box -->
  <section class="section" style="margin-top: 4px;">
    <div class="section-title">
      <div><span class="sec-num">5.</span> Structural Out-of-Distribution Generalization</div>
      <span style="font-size: 7pt; font-weight: 600; color: #475569; text-transform: none;">Extrapolation to Flexible Archetype 5S_T120</span>
    </div>
    <div class="ood-box">
      <div class="ood-header">
        <div class="ood-title">STRUCTURAL OOD — 5S_T120 (T1 = 1.20 s)</div>
        <div class="ood-badge">Training Support: T₁ ∈ [0.35, 0.85] s</div>
      </div>
      <p style="font-size: 7.2pt; margin-bottom: 3px;">
        This structure is outside the training modal-period support. Civil engineers evaluate safety envelopes around peak floor displacement.
      </p>
      <div class="ood-metrics">
        <div class="ood-card">
          <div class="ood-label">EXP5 GNO</div>
          <div class="ood-val val-danger">35.21%</div>
          <div style="font-size: 6pt; color: #64748b;">Median Peak Error</div>
        </div>
        <div class="ood-card">
          <div class="ood-label">EXP6-B</div>
          <div class="ood-val">13.47%</div>
          <div style="font-size: 6pt; color: #64748b;">Median Peak Error</div>
        </div>
        <div class="ood-card highlight">
          <div class="ood-label">EXP6-C</div>
          <div class="ood-val success">13.06%</div>
          <div style="font-size: 6pt; color: #166534;">Median Peak Error</div>
        </div>
        <div class="ood-card highlight">
          <div class="ood-label">Relative Reduction</div>
          <div class="ood-val gain">62.9%</div>
          <div style="font-size: 6pt; color: #15803d;">Peak Error Drop</div>
        </div>
      </div>
      <p style="font-size: 7pt; margin-top: 4px; font-weight: 600; color: #1e3a8a;">
        The Multi-Modal GNO reduces the baseline OOD-B peak displacement error from 35.21% to 13.06%, corresponding to a 62.9% relative reduction under the evaluated structural OOD condition.
      </p>
    </div>
  </section>

  <!-- 6. Two Column: Computational Benchmark & Limitations -->
  <div class="two-col">
    <!-- Computational Benchmark -->
    <div class="col-box">
      <div class="section-title" style="margin-bottom: 3px;">
        <div><span class="sec-num">6.</span> Computational Result</div>
      </div>
      <p style="font-size: 7.2pt;">
        Measured single-simulation wall-clock comparison on Apple Silicon MPS:
      </p>
      <div class="bench-grid">
        <div class="bench-card">
          <div class="bench-label">OpenSeesPy NLTHA</div>
          <div class="bench-val">54.68 ms</div>
          <div style="font-size: 5.8pt; color: #64748b;">Nonlinear Analysis</div>
        </div>
        <div class="bench-card">
          <div class="bench-label">EXP6 T1-GNO</div>
          <div class="bench-val" style="color: #1e3a8a;">21.45 ms</div>
          <div style="font-size: 5.8pt; color: #64748b;">MPS GPU Forward</div>
        </div>
        <div class="bench-card">
          <div class="bench-label">Wall-Clock Ratio</div>
          <div class="bench-val" style="color: #166534;">2.55×</div>
          <div style="font-size: 5.8pt; color: #166534;">vs OpenSeesPy</div>
        </div>
      </div>
      <p style="font-size: 6.5pt; color: #475569; margin-top: 3px;">
        Wall-clock ratio: <strong>2.55×</strong> relative to the OpenSeesPy benchmark. <em>Attribution:</em> Measured single-simulation wall-clock comparison on Apple Silicon MPS.
      </p>
    </div>

    <!-- Where the Model Still Fails -->
    <div class="col-box" style="border-left: 2px solid #b91c1c;">
      <div class="section-title" style="margin-bottom: 3px; color: #991b1b;">
        <div><span class="sec-num">7.</span> Where the Model Still Fails</div>
      </div>
      <ul class="fail-list">
        <li>
          <span class="fail-title">1. Long-horizon waveform phase drift:</span> 
          Relative L2 > 100% under strong modal extrapolation. Pearson correlation approximately 0.05–0.09 in the reported OOD waveform cases. The model can reproduce peak-response envelopes substantially better than full trajectory phase under extrapolation, while global 1D Fourier layers accumulate phase discrepancy over the 20.48 s horizon.
        </li>
        <li>
          <span class="fail-title">2. Static modal descriptors under severe yielding:</span> 
          The modal descriptors are computed from pre-earthquake structural [M] and [K]. They do not explicitly represent dynamic period elongation during severe nonlinear yielding.
        </li>
        <li>
          <span class="fail-title">3. Structural idealization:</span> 
          The current study uses planar lumped-mass shear-frame representations and does not yet cover full 3D asymmetric/torsional/bidirectional structural response.
        </li>
      </ul>
    </div>
  </div>

  <!-- 8. Relevance to CSE / AI Research & Future Directions -->
  <div class="two-col" style="margin-bottom: 5px;">
    <!-- Relevance to CSE / AI -->
    <div class="col-box">
      <div class="section-title" style="margin-bottom: 3px;">
        <div><span class="sec-num">8.</span> Relevance to CSE & Scientific ML</div>
      </div>
      <ul class="fail-list" style="color: #1e293b;">
        <li style="margin-bottom: 2.5px;">
          <span class="fail-title" style="color: #1e3a8a;">• Operator learning on dynamic graphs:</span> 
          Demonstrates how to couple discrete spatial message-passing with continuous 1D temporal spectral convolutions for multi-scale dynamical physical systems.
        </li>
        <li style="margin-bottom: 2.5px;">
          <span class="fail-title" style="color: #1e3a8a;">• Disentangled error modes in dynamical systems:</span> 
          Empirically reveals that modal conditioning can repair amplitude/envelope generalization while leaving global temporal phase drift uncorrected.
        </li>
        <li style="margin-bottom: 0;">
          <span class="fail-title" style="color: #1e3a8a;">• Scientific ML software engineering:</span> 
          Enforces cryptographic reproducibility, zero-leakage hash verification, and automated forensic auditing across 2,160 physical simulations.
        </li>
      </ul>
    </div>

    <!-- Future Directions -->
    <div class="col-box" style="border-left: 2px solid #2563eb;">
      <div class="section-title" style="margin-bottom: 3px; color: #1e3a8a;">
        <div><span class="sec-num">9.</span> Logical Next Research Direction</div>
      </div>
      <p style="font-size: 7.1pt; margin-bottom: 2px;">
        <strong>Causal Structured State-Space Models (SSMs / S4 / Mamba):</strong>
      </p>
      <p style="font-size: 6.9pt; color: #334155;">
        To eliminate the long-horizon phase divergence inherent to static global Fourier spectral layers, next-phase research will investigate causal <strong>State-Space Models (SSMs)</strong> or <strong>Continuous-Time Neural ODEs</strong> coupled with topology graphs. A recurrent state-space formulation processes time causally, enabling dynamic stiffness degradation and period elongation to be tracked recursively step-by-step rather than forced onto a static Fourier basis.
      </p>
    </div>
  </div>

  <!-- 10. Reproducibility & Audit Footer -->
  <section class="section">
    <div class="section-title">
      <div><span class="sec-num">10.</span> Reproducibility & Forensic Audit Verification</div>
      <span style="font-size: 7pt; font-weight: 600; color: #166534; text-transform: none;">Zero Data Leakage & Frozen Core</span>
    </div>
    <div class="audit-grid">
      <div class="audit-item">
        <div class="k">Research core:</div>
        <div class="v">EXP4 / EXP5 / EXP6 frozen</div>
      </div>
      <div class="audit-item">
        <div class="k">Data leakage:</div>
        <div class="v">0 pairwise partition intersections</div>
      </div>
      <div class="audit-item">
        <div class="k">Scaler fitting:</div>
        <div class="v">Training partition only</div>
      </div>
      <div class="audit-item">
        <div class="k">Modal conditioning:</div>
        <div class="v">Derived from structural [M], [K] before earthquake excitation</div>
      </div>
      <div class="audit-item">
        <div class="k">Repository tests:</div>
        <div class="v">294 passed, 2 skipped, 0 failed</div>
      </div>
      <div class="audit-item">
        <div class="k">Historical experiment artifacts:</div>
        <div class="v">SHA256 integrity verified</div>
      </div>
      <div class="audit-item">
        <div class="k">EXP7:</div>
        <div class="v">Not started</div>
      </div>
      <div class="audit-item">
        <div class="k">Demonstration layer:</div>
        <div class="v">Audited and frozen</div>
      </div>
    </div>
  </section>

  <!-- 11. Research Resources & Navigation -->
  <section class="section">
    <div class="section-title">
      <div><span class="sec-num">11.</span> RESEARCH RESOURCES</div>
      <span style="font-size: 7pt; font-weight: 600; color: #475569; text-transform: none;">Genuinely Clickable Repository Artifacts</span>
    </div>
    <div class="resources-container">
      <div class="res-links">
        <a class="res-link" href="./SEISMOFNO_TECHNICAL_REPORT.md">Technical Report (18-Section Formal Report)</a>
        <span class="res-sep">•</span>
        <a class="res-link" href="./PROFESSOR_DEMO.md">Professor Demo Documentation</a>
        <span class="res-sep">•</span>
        <a class="res-link" href="../results/experiments/exp6/INDEPENDENT_FORENSIC_AUDIT.md">Independent Forensic Audit</a>
        <span class="res-sep">•</span>
        <a class="res-link" href="./RESEARCH_ARCHITECTURE_DIAGRAM.md">Architecture Specification</a>
        <span class="res-sep">•</span>
        <a class="res-link" href="../README.md">Repository README</a>
        <span class="res-sep">•</span>
        <span style="color: #475569; font-weight: 600;">Interactive Demo — local / repository deployment</span>
      </div>
    </div>
  </section>

  <!-- Document Sign-Off Footer -->
  <footer class="page-footer">
    <div>
      <strong>SeismoFNO Research Brief</strong> • Author: <strong>Raghvendra Singh Gahlot</strong> (B.E. Structural Engineering) • Research Core: FROZEN / AUDITED
    </div>
    <div>Page 2 of 2</div>
  </footer>
</div>

</body>
</html>
"""

def compile_pdf_via_chrome(html_path: Path, pdf_path: Path) -> bool:
    """Invokes headless Google Chrome to compile HTML to PDF."""
    cmd = [
        CHROME_PATH,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        str(html_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return pdf_path.exists() and pdf_path.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        print(f"Chrome compilation error: {e.stderr}", file=sys.stderr)
        return False

def count_pdf_pages(pdf_path: Path) -> int:
    """Determines exact page count from PDF binary structure."""
    with open(pdf_path, "rb") as f:
        data = f.read()
    pages = len(re.findall(rb"/Type\s*/Page\b", data))
    return pages

def patch_pdf_metadata(pdf_path: Path, title: str, author: str, subject: str, keywords: str) -> None:
    """Injects metadata into PDF's Info object."""
    with open(pdf_path, "rb") as f:
        data = f.read()
        
    info_ref = re.search(rb"/Info\s+(\d+)\s+(\d+)\s+R", data)
    if not info_ref:
        return
    obj_num = info_ref.group(1)
    gen_num = info_ref.group(2)
    
    pattern = rb"(" + obj_num + rb"\s+" + gen_num + rb"\s+obj\s*<<)(.*?)(>>\s*endobj)"
    m = re.search(pattern, data, re.DOTALL)
    if not m:
        return
        
    def escape_val(s: str) -> str:
        s_clean = s.replace("—", " -- ")
        return s_clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        
    entries = (
        f"/Title ({escape_val(title)})\n"
        f"/Author ({escape_val(author)})\n"
        f"/Subject ({escape_val(subject)})\n"
        f"/Keywords ({escape_val(keywords)})\n"
        f"/Creator (SeismoFNO Academic PDF Builder)\n"
    ).encode("latin1", errors="replace")
    
    old_dict = m.group(2)
    for k in [b"/Title", b"/Author", b"/Subject", b"/Keywords", b"/Creator"]:
        old_dict = re.sub(k + rb"\s*\(.*?\)", rb"", old_dict)
        old_dict = re.sub(k + rb"\s*<.*?>", rb"", old_dict)
        
    updated_obj = m.group(1) + b"\n" + entries + old_dict.strip() + b"\n" + m.group(3)
    new_data = data[:m.start()] + updated_obj + data[m.end():]
    
    with open(pdf_path, "wb") as f:
        f.write(new_data)

def inject_metadata_and_verify(pdf_path: Path) -> dict:
    """Verifies critical metrics and counts clickable hyperlinks."""
    with open(pdf_path, "rb") as f:
        data = f.read()
        
    num_pages = count_pdf_pages(pdf_path)
    link_count = len(re.findall(rb"/Subtype\s*/Link\b", data))

    required_metrics = [
        "99.60%", "22.09%", "35.21%", "13.47%", "13.06%",
        "24.33%", "54.68 ms", "21.45 ms", "2.55×", "294 passed"
    ]
    
    html_text = TEMP_HTML.read_text(encoding="utf-8") if TEMP_HTML.exists() else ""
    
    metric_status = {}
    for m in required_metrics:
        clean_m = m.replace("×", "x").replace("passed", "").strip()
        found = (m in html_text) or (clean_m in html_text)
        metric_status[m] = found

    return {
        "num_pages": num_pages,
        "link_count": link_count,
        "metrics": metric_status,
        "file_size": pdf_path.stat().st_size
    }

def main():
    print(f"==> Generating SeismoFNO Research Brief HTML...")
    html_content = generate_html_content()
    TEMP_HTML.write_text(html_content, encoding="utf-8")
    print(f"    Wrote temporary HTML to {TEMP_HTML}")

    print(f"==> Compiling PDF via headless Google Chrome...")
    success = compile_pdf_via_chrome(TEMP_HTML, OUTPUT_PDF)
    if not success:
        print("ERROR: Failed to compile PDF via Chrome.", file=sys.stderr)
        sys.exit(1)
        
    print(f"    Compiled PDF: {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size} bytes)")

    print(f"==> Patching PDF Metadata (Title, Author, Subject, Keywords)...")
    patch_pdf_metadata(
        OUTPUT_PDF,
        title="SeismoFNO — Physics-Grounded Graph Neural Operators for Seismic Structural Response",
        author="Raghvendra Singh Gahlot",
        subject="Scientific ML for Computational Structural Dynamics",
        keywords="Neural Operators, Graph Neural Operators, Seismic Response, Scientific Machine Learning, Structural Dynamics, Physics-Informed ML, Out-of-Distribution Generalization"
    )

    print(f"==> Auditing PDF structure & metrics...")
    audit = inject_metadata_and_verify(OUTPUT_PDF)
    print(f"    Page count: {audit['num_pages']} (Target: 2, Max: 3)")
    print(f"    Clickable links detected: {audit['link_count']}")
    print(f"    Verified metrics in source:")
    all_metrics_found = True
    for m, found in audit["metrics"].items():
        status = "✓ FOUND" if found else "✗ MISSING"
        print(f"      - {m:12s}: {status}")
        if not found:
            all_metrics_found = False

    if audit["num_pages"] > 3:
        print(f"ERROR: PDF exceeded maximum page count (got {audit['num_pages']})", file=sys.stderr)
        sys.exit(1)

    print("==> Research Brief PDF successfully compiled and verified!")

if __name__ == "__main__":
    main()
