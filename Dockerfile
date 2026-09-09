# ==============================================================================
# SeismoFNO — Production Multi-Stage Full-Stack Docker Container for Render.com
# ==============================================================================

# --- Stage 1: Build the React 18 Frontend ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Minimal Python Runtime & Dependencies ---
FROM python:3.11-slim AS runner
WORKDIR /app

# Install system dependencies for OpenSeesPy and Git LFS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy source code and research assets
COPY . .

# Copy pre-built frontend distribution from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Ensure Git LFS pointers are pulled if repo was cloned shallowly
RUN if [ -f .gitattributes ]; then git lfs install && git lfs pull || true; fi

# Production environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV HOST=0.0.0.0
ENV DEVICE=cpu
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check targeting FastAPI /health probe
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch Uvicorn bound to dynamic Render port
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
