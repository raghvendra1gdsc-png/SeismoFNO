# Production & Cloud Deployment Guide — SeismoFNO

This document outlines the deployment architecture, configuration requirements, environment specifications, and cloud deployment procedures for the **SeismoFNO** scientific platform and research demonstration.

---

## 1. System Architecture

SeismoFNO is architected as a decoupled, zero-state client-server system:

```
┌────────────────────────────────────────────────────────┐
│               Frontend Presentation Layer              │
│       Vite + React 18 + Tailwind CSS + Lucide Icons     │
│   (Deployable to Vercel / Netlify / Cloudflare Pages)  │
└───────────────────────────┬────────────────────────────┘
                            │ HTTPS / REST (JSON)
                            ▼
┌────────────────────────────────────────────────────────┐
│            FastAPI Application Gateway (v1.0)           │
│  - /health (Readiness / Liveness Probe)                │
│  - /api/v1/demo/* (Research Defense Demonstration)     │
│  - /api/v1/scenario/* (Interactive Digital Twin)       │
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
              ▼                           ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│   PyTorch Neural Model    │ │   OpenSeesPy C-Runtime   │
│   Inference Engine        │ │   Non-Linear Dynamic     │
│   (Conditioned GNO / FNO) │ │   Ground Truth Engine    │
│   - CPU / CUDA Execution  │ │   - Thread-Locked C-State│
│   - Thread-Safe Locks     │ │   - OpenSeesPy NLTHA     │
└───────────────────────────┘ └──────────────────────────┘
```

---

## 2. Hardware & Device Attribution Notice

> [!IMPORTANT]
> **Benchmarking Hardware Context:**
> All wall-clock latencies and speedup ratios reported in the research documentation (e.g., **2.55× speedup** for EXP6 $T_1$-GNO, **54.68 ms OpenSeesPy** vs **21.45 ms neural inference**) were measured on **Apple Silicon GPU (`mps`) unified memory**.
> 
> When deploying to standard cloud environments (e.g., Render, Railway, Fly.io, AWS EC2, GCP Cloud Run):
> - By default, container instances run on **x86_64 or ARM64 CPUs** (`DEVICE=cpu`).
> - Neural operator inference on a cloud CPU will exhibit different absolute latencies than local Apple Silicon MPS.
> - For accelerated cloud inference, configure GPU container instances with NVIDIA drivers (`DEVICE=cuda`).

---

## 3. Large Model Checkpoints & Git LFS Handling

The neural operator checkpoints (`.pt`) and preprocessed dataset arrays (`.h5`, `.npy`) are managed via **Git Large File Storage (Git LFS)** to keep the primary git tree lightweight:

```
results/experiments/exp4/checkpoints/best_model.pt   (Git LFS)
results/experiments/exp5/checkpoints/best_model.pt   (Git LFS)
results/experiments/exp6/checkpoints/best_model.pt   (Git LFS)
results/experiments/exp6/scalers.pt                  (Git LFS)
```

### Docker / CI/CD Requirement
When building container images or deploying via automated platforms, ensure Git LFS resolves the pointer files:
```dockerfile
# Ensure Git LFS is installed in build image
RUN apt-get update && apt-get install -y git-lfs && git-lfs install
# Fetch real binary artifacts
RUN git lfs pull
```

---

## 4. Environment Variables Reference

| Variable | Default (Local) | Production Recommendation | Description |
| :--- | :---: | :---: | :--- |
| `PORT` | `8000` | Assigned by host (`$PORT`) | HTTP listening port for uvicorn |
| `HOST` | `0.0.0.0` | `0.0.0.0` | Network binding interface |
| `DEVICE` | `mps` | `cpu` (or `cuda` if GPU attached) | PyTorch device allocation (`cpu`, `cuda`, `mps`) |
| `PYTHONUNBUFFERED`| `1` | `1` | Forces stdout/stderr flush for container logs |
| `VITE_API_BASE_URL`| `http://localhost:8000` | `https://api.yourdomain.com` | Backend URL for frontend build |
| `NEBIUS_API_KEY` | *(unset)* | Optional API key | Optional LLM copilot key; uses deterministic rule engine if unset |

---

## 5. Docker Deployment (Backend)

A sample production-grade `Dockerfile` for the FastAPI backend:

```dockerfile
# Multi-stage minimal Python runtime
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application and research assets
COPY . .

ENV PORT=8000
ENV HOST=0.0.0.0
ENV DEVICE=cpu
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

> [!NOTE]
> OpenSeesPy maintains global state in its underlying C-runtime. Run Uvicorn with `--workers 1` or rely on the thread-safe `OPENSEES_LOCK` built into `api/main.py`.

---

## 6. Frontend Deployment (Vercel / Netlify)

### Vercel
1. Link your GitHub repository in the Vercel dashboard.
2. Configure project settings:
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
3. Add Environment Variable:
   - `VITE_API_BASE_URL`: `https://your-backend-service.onrender.com`
4. Deploy.

### Rewrites Configuration (`vercel.json`)
To ensure client-side routing (e.g., `/demo`) functions correctly on Vercel:
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

---

## 7. Cloud Health Checks & Monitoring

Once deployed, configure the hosting platform's health check probe to target:
- **HTTP Path:** `/health`
- **Expected Status:** `200 OK`
- **Response Format:**
  ```json
  {
    "status": "healthy",
    "timestamp": 1725821500.12,
    "version": "1.0.0",
    "model_loaded": true,
    "device": "cpu"
  }
  ```
