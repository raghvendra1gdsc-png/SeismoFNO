#!/usr/bin/env bash
# ==============================================================================
# SeismoFNO — Unified Localhost Research Workstation Launcher
# ==============================================================================
# Launches both the FastAPI neural surrogate backend and the React/Three.js
# research frontend in a single command.
#
# Usage:
#   chmod +x run_demo.sh
#   ./run_demo.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "======================================================================"
echo " SEISMOFNO: Physics-Grounded Structural Dynamics Neural Operator"
echo " Indian Institute of Technology (IIT) Research Submission"
echo "======================================================================"
echo ""

# 1. Check Python environment
if [ -d ".venv" ]; then
    echo "[1/4] Activating Python virtual environment (.venv)..."
    source .venv/bin/activate
elif command -v python3 &>/dev/null; then
    echo "[1/4] Using system Python: $(python3 --version)"
else
    echo "ERROR: python3 not found. Please install Python 3.10+."
    exit 1
fi

# 2. Check Node & npm
if ! command -v npm &>/dev/null; then
    echo "ERROR: npm is not installed. Please install Node.js (v18+)."
    exit 1
fi
echo "[2/4] Node environment detected: $(node --version)"

# 3. Start Backend API Server
echo "[3/4] Starting FastAPI backend on http://127.0.0.1:8000..."
BACKEND_PID=""
if lsof -i :8000 &>/dev/null; then
    echo "      Port 8000 is already active. Using existing backend instance."
else
    PYTHONPATH=. python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
    BACKEND_PID=$!
    # Wait for backend to respond
    for i in {1..30}; do
        if curl -s http://127.0.0.1:8000/api/v1/system/info &>/dev/null; then
            break
        fi
        sleep 0.5
    done
    echo "      Backend online (PID: $BACKEND_PID)."
fi

# 4. Start Frontend Dev Server
echo "[4/4] Starting Vite frontend workstation on http://localhost:5173..."
FRONTEND_PID=""
if lsof -i :5173 &>/dev/null; then
    echo "      Port 5173 is already active. Using existing frontend instance."
else
    npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173 &
    FRONTEND_PID=$!
    for i in {1..30}; do
        if curl -s http://127.0.0.1:5173/ &>/dev/null; then
            break
        fi
        sleep 0.5
    done
    echo "      Frontend online (PID: $FRONTEND_PID)."
fi

cleanup() {
    echo ""
    echo "Shutting down SeismoFNO services..."
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    echo "Done."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo ""
echo "======================================================================"
echo " ✓ SeismoFNO Workstation Ready on Localhost"
echo "======================================================================"
echo "  • ★ NVIDIA Nemotron Judge Mode:    http://localhost:5173/judge"
echo "  • Structural Dynamic Simulator:   http://localhost:5173"
echo "  • Multi-Story Research (GNO):     http://localhost:5173/demo"
echo "  • Live USGS Seismic Screening:    http://localhost:5173/live"
echo "  • OpenSeesPy Physics Reference:   http://localhost:5173/?tab=model_validation"
echo "  • Scenario Lab (Comparative):     http://localhost:5173/?tab=scenario_lab"
echo "  • Backend OpenAPI Specification:  http://127.0.0.1:8000/docs"
echo "======================================================================"
echo "Press Ctrl+C to stop both servers."
echo ""

# Keep alive
wait
