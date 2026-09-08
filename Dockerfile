# =================================================================
# INDUSTRIAL FIRE & PERSISTENT THERMAL SOURCE DETECTION (SIH 2026)
# Multi-Service Production Dockerfile
# Part 5.5: Deployment, Containerization & Health Monitoring
# =================================================================

FROM python:3.10-slim AS runtime

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVICE_TYPE=api \
    API_PORT=8000 \
    WEB_PORT=5000 \
    PORT=5000

# Install required build tools and libraries for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsqlite3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user for container security hardening
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Pre-create runtime directories and assign ownership
RUN mkdir -p /app/data/processed /app/data/raw /app/output/reports /app/models && \
    chown -R appuser:appgroup /app

# Install Python dependencies with pip cache optimization
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code
COPY --chown=appuser:appgroup . /app

# Switch to non-root execution
USER appuser

# Expose API and Web ports
EXPOSE 8000 5000

# Built-in container healthcheck using the standalone Python probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python scripts/healthcheck.py --service ${SERVICE_TYPE:-api} || exit 1

# Launch the target service according to SERVICE_TYPE
CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"web\" ]; then exec python main.py --part web --port ${PORT:-5000}; elif [ \"$SERVICE_TYPE\" = \"worker\" ]; then exec python main.py --part 5.1 --interval-hours 6.0; else exec uvicorn src.api.app:app --host 0.0.0.0 --port ${API_PORT:-8000}; fi"]
