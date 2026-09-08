# Local Development & Verification Guide — SeismoFNO

This guide provides instructions for setting up, running, testing, and verifying the **SeismoFNO** scientific machine learning research platform and interactive research defense demonstration on a local workstation.

---

## 1. System Requirements & Prerequisites

### Hardware
- **Processor:** Apple Silicon (M1/M2/M3/M4) recommended for MPS GPU unified memory acceleration, or x86_64 / ARM64 multi-core CPU.
- **RAM:** Minimum 8 GB (16 GB recommended for running full dataset audits).
- **Disk Space:** ~3 GB free disk space (includes checkpoints and dataset artifacts tracked via Git LFS).

### Software
- **Operating System:** macOS (Sonoma / Sequoia) or Linux (Ubuntu 22.04+).
- **Python:** Version 3.10, 3.11, or 3.14 (with `pip` and `venv`).
- **Node.js:** Node.js 18.x or 20.x+ with `npm` (v9+).
- **Git & Git LFS:** `git` and `git-lfs` (`git lfs install`).

---

## 2. Environment Setup

### A. Clone and Git LFS Initialization
```bash
# Clone the repository
git clone <repo-url> seismoFNO
cd seismoFNO

# Ensure Git LFS pointers are resolved for checkpoints and datasets
git lfs install
git lfs pull
```

### B. Python Virtual Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### C. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## 3. Environment Configuration

Copy the sample environment file:
```bash
cp .env.example .env
```

### Configuration Variables (`.env`)
```bash
# API Host and Port
PORT=8000
HOST=0.0.0.0

# Compute Device: 'mps' for Apple Silicon GPU, 'cuda' for NVIDIA, 'cpu' for standard CPU
DEVICE=mps

# Ensure unbuffered output for real-time logging
PYTHONUNBUFFERED=1

# Frontend API Target (used during build/SSR if applicable)
VITE_API_BASE_URL=http://localhost:8000

# Optional Copilot LLM Key (if unset, deterministic rule-based mock engine is used automatically)
# NEBIUS_API_KEY=your_key_here
```

---

## 4. Running the Local Application

The full interactive platform consists of two services:
1. **FastAPI Application Gateway** (Backend on `http://localhost:8000`)
2. **Vite + React Research Interface** (Frontend on `http://localhost:5173`)

### Terminal 1: Launch Backend
```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Health check:* Visit `http://localhost:8000/health` (should return `{"status": "healthy", ...}`).

### Terminal 2: Launch Frontend
```bash
cd frontend
npm run dev
```
*Demo URL:* Open your browser at **`http://localhost:5173/demo`** (or access the main dashboard at `http://localhost:5173`).

---

## 5. Verifying the System

### A. Backend Unit Tests
Execute the full pytest test suite (294 automated tests covering OpenSeesPy ground truth, FNO shapes, GNO graph edge construction, modal datasets, and API endpoints):
```bash
PYTHONPATH=. .venv/bin/pytest -q
```
*Expected output:* `294 passed, 2 skipped in ~13s`.

### B. Independent Forensic Audit
Run the automated scientific data hygiene and metric reconstruction audit:
```bash
PYTHONPATH=. .venv/bin/python scripts/run_exp6_forensic_audit.py
```
*Expected output:*
- Step 1: Auditing Data Splits & Partitions (Zero Leakage: PASSED)
- Step 2: Auditing Model Checkpoints & Parameter Integrity (PASSED)
- Step 3: Reconstructing All Metrics from Evaluation CSVs (PASSED)
- Step 4: Auditing Inference Benchmarks (PASSED)
- Step 5: Auditing Frozen State of EXP4 and EXP5 (PASSED)

### C. Frontend Production Build
Verify TypeScript compilation and asset bundling:
```bash
cd frontend
npm run build
```
*Expected output:* `built in ~250ms` with 0 errors.

---

## 6. Local Endpoints Reference

| Endpoint | Method | Purpose |
| :--- | :---: | :--- |
| `/health` | GET | Liveness and readiness probe, reporting device and model load status |
| `/api/v1/system/info` | GET | Checkpoint path, parameter count, and runtime device |
| `/api/v1/demo/models` | GET | Catalog of immutable models (EXP4, EXP5, EXP6-B, EXP6-C, EXP6-D) |
| `/api/v1/demo/structures` | GET | Precomputed physical structural archetypes (3S, 5S, 8S) |
| `/api/v1/demo/earthquakes` | GET | Verified PEER ground motion records |
| `/api/v1/demo/progression` | GET | Master scientific progression matrix (EXP4 $\to$ EXP5 $\to$ EXP6) |
| `/api/v1/demo/ood-matrix` | GET | Out-of-distribution evaluation results |
| `/api/v1/demo/ablation` | GET | Shuffled-conditioning falsification evidence |
| `/api/v1/demo/failures` | GET | First-class scientific limitations and failure modes |
| `/api/v1/demo/benchmark` | GET | Wall-clock latency benchmarks measured on Apple Silicon MPS |
| `/api/v1/demo/simulate` | POST | Live forward pass (EXP6) or verified archival response (EXP4) |

---

## 7. Troubleshooting

- **Port in use (`Errno 48: Address already in use`):**
  ```bash
  lsof -i :8000 | awk 'NR>1 {print $2}' | xargs kill -9
  lsof -i :5173 | awk 'NR>1 {print $2}' | xargs kill -9
  ```
- **Apple Silicon MPS Concurrency Error (`commit an already committed command buffer`):**
  The model forward pass is protected with an internal threading lock in `src/demo/inference_adapter.py`. If modifying adapter code, ensure thread locks remain around MPS forward passes. Alternatively, set `DEVICE=cpu` in `.env`.
- **Git LFS Pointers Not Downloaded:**
  If `.pt` or `.h5` files are ~130 bytes, run `git lfs pull` to fetch binary payloads.
