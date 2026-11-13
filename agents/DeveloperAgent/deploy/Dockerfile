# DeveloperAgent Dockerfile — Production deployment
# Tier 5+ Class A — Multi-stage build for minimal image size

# ─── Stage 1: Builder ───────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

# Set build arguments
ARG PYTHON_VERSION=3.11
ARG CONDROX_VERSION=2.0.0

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Stage 2: Runtime ───────────────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

# Set labels
LABEL maintainer="Tobias Østne"
LABEL version="${CONDROX_VERSION}"
LABEL description="DeveloperAgent v2.0 — Enterprise-grade AI code assistant"
LABEL tier="Class A"

# Create non-root user
RUN groupadd -r condrox && useradd -r -g condrox condrox

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY agents/DeveloperAgent /app/agents/DeveloperAgent
COPY core /app/core
COPY config /app/config

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R condrox:condrox /app

# Switch to non-root user
USER condrox

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CONDROX_ROOT=/app
ENV CONDROX_DATA=/app/data
ENV CONDROX_LOGS=/app/logs
ENV CONDROX_CACHE=/app/cache

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "agents.DeveloperAgent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
