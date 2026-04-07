# IncidentBench — Dockerfile
# ============================
# Judges run: docker build . && docker run -p 7860:7860 <image>
# Must start cleanly and respond to /reset within seconds.
#
# Port 7860 is the HuggingFace Spaces default — we never change this.
#
# NOTE: Using mirror.gcr.io (Google's public Docker Hub mirror).
# The validator's build machine cannot reach registry-1.docker.io directly.
# mirror.gcr.io serves the same python:3.12-slim image and is universally accessible.

FROM mirror.gcr.io/library/python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies first (cached layer — only rebuilds if this changes)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer
# If requirements don't change, pip install is skipped on rebuild (faster builds)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY env.py         .
COPY graders.py     .
COPY server.py      .
COPY inference.py   .
COPY openenv.yaml   .
COPY README.md      .

# Port 7860 — HuggingFace Spaces standard, do not change
EXPOSE 7860

# Health check — Docker will mark container unhealthy if /health stops responding
# Judges can see this in docker ps
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server
# - host 0.0.0.0 is required — 127.0.0.1 won't be reachable from outside container
# - port 7860 matches EXPOSE and HuggingFace Spaces routing
# - workers=1 keeps memory under 8GB limit
# - timeout-keep-alive helps with HF Spaces proxy
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]