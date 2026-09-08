# SeismoFNO Web Workstation — Comprehensive User Guide

Welcome to the **SeismoFNO Scientific Neural Operator Workstation**. This guide explains how to launch, navigate, and utilize each workspace within the interactive web application for research demonstration, structural dynamic analysis, and real-world earthquake event screening.

---

## 1. Quick Start: Launching the Workstation

The application consists of a high-performance **FastAPI** backend and a responsive **React + Vite** frontend.

### Step 1: Start the Backend API
In your terminal, activate the Python virtual environment and launch Uvicorn:
```bash
cd /Users/rahul/seismoFNO
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Backend Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 2: Start the Frontend Application
In a separate terminal window:
```bash
cd /Users/rahul/seismoFNO/frontend
npm run dev
```
- **Local Workstation URL:** [http://localhost:5173/](http://localhost:5173/)
- **Direct Research Defense Demo:** [http://localhost:5173/demo](http://localhost:5173/demo)
- **Direct Live Earthquake Workspace:** [http://localhost:5173/live](http://localhost:5173/live)

---

## 2. Workspaces & Navigation Overview

The left navigation sidebar provides one-click switching between specialized engineering and research workspaces:

| Workspace | Route / Tab | Primary Function |
| :--- | :--- | :--- |
| **Research Defense Demo** | `/demo` (`research_demo`) | Academic defense view (EXP4 $\to$ EXP5 $\to$ EXP6, modal shapes, OOD matrix, falsification ablations). |
| **00 Live Earthquake** | `/live` (`live_earthquake`) | Ingestion of real-time USGS GeoJSON events, scientific map, distance calculation, and screening. |
| **Command Center** | `?tab=command_center` | System telemetry, compute device status (MPS/CPU), and executive scenario run. |
| **Earthquake Intelligence** | `?tab=earthquake_intel` | Historic Indian seismic catalog (IS 1893 zones) and PEER NGA-West2 acceleration library. |
| **Structural Digital Twin** | `?tab=structural_twin` | Non-linear SDOF/MDOF parameters ($T_0, \zeta, u_y, \alpha$, bilinear hysteresis). |
| **Scenario Lab** | `?tab=scenario_lab` | Multi-scenario comparison and sensitivity stress testing. |
| **Model Validation** | `?tab=model_validation` | Quantitative verification metrics against OpenSeesPy non-linear time-history ground truth. |
| **Research / Explainability** | `?tab=explainability` | Deep-dive Fourier frequency representations, failure boundary explanations, and citations. |

---

## 3. Deep Dive: "00 Live Earthquake" Workspace

The **00 Live Earthquake** workspace connects real-world seismic observations from the United States Geological Survey (USGS) to the multi-story SeismoFNO surrogate without violating scientific boundaries.

### Workflow Step-by-Step

```
[1] Filter USGS Events Table
        ↓
[2] Inspect Restrained Scientific Map
        ↓
[3] Select Event & Review Event Context
        ↓
[4] Click [ LOAD EVENT INTO ANALYSIS ]
        ↓
[5] Review Structural Analysis Context (5-Story Frame, T1-3)
        ↓
[6] Note Ground-Motion Availability Notice (Catalog Metadata ≠ Acceleration Waveform)
        ↓
[7] Select Verified Research Ground Motion (e.g. PEER RSN0001)
        ↓
[8] Execute Live SeismoFNO Forward Pass & Inspect Response
```

### Key UI Features

1. **Interactive Event Table & Filters:**
   - **Magnitude Filter:** Filter between All ($M \ge 0$), Minor ($M \ge 2.5$), Moderate ($M \ge 4.5$), Strong ($M \ge 6.0$), or Major ($M \ge 7.0$).
   - **Time Window:** Restrict to the past 1 Hour, 6 Hours, 24 Hours, or 7 Days.
   - **Radius Filter:** Constrain events by distance from the structural scenario target ($500\text{ km}, 1000\text{ km}, 3000\text{ km}$, or Global).
   - **Search Box:** Filter in real-time by place name or USGS event code (e.g. `California`, `Japan`, `Turkey`).
   - Click any table row to immediately select that earthquake.

2. **Restrained Scientific Map:**
   - Vector SVG Plate Carrée projection with $30^\circ$ latitude/longitude technical graticule lines.
   - Circular epicenter markers scaled by magnitude ($M_w$) and color-coded by focal depth:
     - **Red:** Shallow focus ($< 30\text{ km}$) — high surface hazard.
     - **Amber:** Intermediate focus ($30\text{--}70\text{ km}$).
     - **Blue:** Deep focus ($> 70\text{ km}$).
   - Displays target structural scenario pin and an animated geodesic connection line with computed great-circle distance (Haversine formula).

3. **Event Context Card:**
   - Displays magnitude, focal depth, exact epicenter coordinates, origin time (UTC), distance to scenario, and official USGS event page link.
   - Explicit badges: `SOURCE: USGS`, `STATUS: OBSERVED EVENT`, `DATA TYPE: CATALOG / REAL-TIME FEED`.

4. **Loading Event into Analysis:**
   - Clicking **`[ LOAD EVENT INTO ANALYSIS ]`** opens the **Structural Analysis Context**.
   - **Target Structure:** Pre-selects the 5-story frame example (`5S_T055` or flexible `5S_T120`), showing theoretical undamped modal periods ($T_1, T_2, T_3$).
   - **Ground-Motion Availability Check:** The workstation transparently notes:
     > *"Event metadata loaded. Compatible acceleration waveform unavailable for direct SeismoFNO inference."*
     *(This enforces scientific honesty: earthquake catalogs provide scalar parameters, not high-frequency accelerograms $a_g(t)$).*
   - **Waveform Selection:** Choose a verified historical acceleration record (e.g., Imperial Valley RSN0001, Loma Prieta RSN0802, Northridge RSN1011).
   - **Provenence Chain & Research Mode:** View the visual flow from USGS metadata $\to$ verified waveform $\to$ structural model $\to$ SeismoFNO response.
   - **Simulation Execution:** Click **`EXECUTE SURROGATE RESPONSE SIMULATION`** to run the PyTorch forward pass on Apple Silicon MPS / CPU and view the roof displacement trajectory $u(t)$ alongside OpenSeesPy ground truth.

5. **Offline Fallback Guarantee:**
   - If your machine is disconnected or USGS public services are unreachable, the workspace automatically activates:
     `USGS LIVE FEED UNAVAILABLE — SERVING VERIFIED CACHED EVENT DATA`
   - The workstation remains 100% functional offline using bundled historical events (2023 Turkey M7.8, 1994 Northridge M6.7, 1989 Loma Prieta M6.9, 2001 Bhuj M7.7).

---

## 4. Deep Dive: "Research Defense Demo" Workspace

The **Research Defense Demo** (`/demo`) provides the core oral defense interface for faculty, advisors, and scientific machine learning reviewers.

### Key Sections to Explore

1. **Section I — Scientific Progression (The Research Story):**
   - **Stage 1 (EXP4):** Fixed-Grid FNO2D failure ($99.60\%$ relative $L_2$ error due to zero-padding Gibbs ringing).
   - **Stage 2 (EXP5):** Topology-Native Graph Neural Operator ($22.09\%$ error, zero padding, physical story-node mapping).
   - **Stage 3 (EXP6):** Physics/Modal-Conditioned GNO ($13.06\%$ peak error on held-out flexible structure `5S_T120` via $T_1, \omega_1$ FiLM conditioning).
   - **Stage 4 (Analysis):** Honest boundary analysis documenting long-horizon waveform phase drift.

2. **Section B — Structural Archetype Selector:**
   - Select between held-out 3-story frame (`3S_T035`), in-domain 5-story frame (`5S_T055`), out-of-distribution flexible 5-story frame (`5S_T120`), and 8-story boundary frame (`8S_T085`).

3. **Section C — Theoretical Mode Shape Visualizer:**
   - Interactive tabs for **Mode 1 (Fundamental)**, **Mode 2**, and **Mode 3**.
   - Deformed building canvas illustrating dynamic modal coordinates $\boldsymbol{\phi}_i$ computed purely from initial mass $[\mathbf{M}]$ and stiffness $[\mathbf{K}]$ matrices prior to ground motion excitation.

4. **Section D & E — Excitation & Model Architecture Controls:**
   - Switch ground motion input $a_g(t)$ across verified PEER records.
   - Select surrogate neural operator:
     - `EXP4: Fixed-Grid FNO2D` (Archival baseline demonstrating Gibbs ringing).
     - `EXP5: Baseline GNO` (Topology-native unconditioned operator).
     - `EXP6-B: T1-Conditioned GNO` (Scalar period conditioning).
     - `EXP6-C: Multi-Modal GNO` (Full $[T_1\text{--}3, \omega_1\text{--}3]$ conditioning).
     - `EXP6-D: Shuffled Modal Control` (Falsification control proving physical invariance).

5. **Section G & H — Displacement Trajectory & Live Metrics:**
   - Real-time plot comparing OpenSeesPy Ground Truth (dashed black line) against SeismoFNO prediction (solid blue line).
   - Quantitative error metrics: **Relative $L_2$ Error (%)**, **Peak Displacement Error (%)**, **Pearson Correlation ($r$)**, and **Wall-Clock Latency (ms)**.

6. **Out-of-Distribution Matrix & Shuffled Falsification:**
   - Bar charts comparing EXP5 vs EXP6-B vs EXP6-C vs Shuffled Control across in-domain and out-of-distribution structural splits.
   - Proves a **62.9% relative reduction** in peak displacement error under modal shift.

---

## 5. Other Workspaces Quick Reference

### Command Center (`?tab=command_center`)
- Displays real-time device telemetry (`DEVICE: MPS` on Apple Silicon or `DEVICE: CPU`).
- Shows total neural operator parameter count (~1.2M parameters), active checkpoint SHA256 integrity, and quick scenario triggers.

### Earthquake Intelligence (`?tab=earthquake_intel`)
- Features two sub-tabs:
  - **Indian Seismic Catalog:** Interactive map of India showing historic seismic events (Bhuj 2001, Chamoli 1999, Uttarkashi 1991) classified by IS 1893 seismic zones (Zone II through Zone V).
  - **PEER Ground Motion Library:** Interactive library of PEER NGA-West2 records with PGA, duration, and time step previews.

### Structural Digital Twin (`?tab=structural_twin`)
- Customize non-linear building properties in real time:
  - Natural period $T_0$ (0.1s to 2.0s)
  - Damping ratio $\zeta$ (1% to 10%)
  - Bilinear yield displacement $u_y$ (5 mm to 50 mm)
  - Post-yield stiffness ratio $\alpha$ (0% to 20%)
  - Scaled Peak Ground Acceleration (PGA) from 0.05g to 1.50g.
- Triggers non-linear time-history analysis with hysteretic force-displacement $(f_r\text{--}u)$ loops and energy dissipation curves.

---

## 6. Understanding Provenance Badges

Every metric, trajectory, and record in the workstation is marked with an explicit provenance badge:

| Provenance Label | Significance |
| :--- | :--- |
| **`LIVE EVENT FEED`** | Ingested directly from public USGS GeoJSON catalog in real time. |
| **`OFFLINE CACHED EVENT DATA`** | Served from local verified cache (`data/usgs_cache.json`) when offline. |
| **`LIVE_COMPUTED_MPS`** | Computed in real time via local PyTorch neural operator on Apple Silicon GPU. |
| **`ARCHIVAL GROUND MOTION`** | Loaded from audited, pre-processed PEER or Indian ground-motion time series. |
| **`OPENSEESPY_NLTHA_GROUND_TRUTH`** | Numerical ground truth generated via OpenSeesPy non-linear Newmark-$\beta$ integration. |
| **`FROZEN RESEARCH RESULT`** | Immutable checkpoint or evaluation metric from the audited research core. |

---

## 7. Troubleshooting & FAQ

### Q1: The USGS live feed says "UNAVAILABLE". Why?
Your machine may be disconnected from the internet, or the USGS server took longer than 4 seconds to respond. The system automatically handles this by serving verified cached records. To retry, click the **`RETRY LIVE CONNECTION`** or **`SYNC`** button.

### Q2: Why doesn't an event from USGS directly show a displacement curve?
USGS provides earthquake occurrence metadata ($M_w$, hypocenter depth, coordinates). It does not provide the high-frequency acceleration waveform $a_g(t)$ needed by neural operators. The workstation maintains scientific honesty by requiring a verified accelerogram to simulate dynamic response.

### Q3: How do I verify that tests are passing?
Run the automated test suite from the repository root:
```bash
source .venv/bin/activate
pytest tests/test_usgs_integration.py -v
pytest tests/test_demo_layer.py -v
pytest -q
```
Expected result: **304 passed, 0 failed**.

### Q4: How do I rebuild the frontend for production?
```bash
cd frontend
npm run build
```
This compiles TypeScript and outputs production assets to `frontend/dist/`.
