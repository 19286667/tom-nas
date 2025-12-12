# ToM-NAS Production Dockerfile
# Multi-stage build for optimized image size and security

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-lock.txt .
RUN pip install --no-cache-dir --user -r requirements-lock.txt

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.11-slim as production

# Security: Run as non-root user
RUN useradd --create-home --shell /bin/bash tomnas
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /home/tomnas/.local
ENV PATH=/home/tomnas/.local/bin:$PATH

# Copy application code
COPY --chown=tomnas:tomnas . .

# Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints /app/results \
    && chown -R tomnas:tomnas /app

# Switch to non-root user
USER tomnas

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOM_NAS_ENV=production \
    TOM_NAS_LOG_LEVEL=INFO \
    ENABLE_CLOUD_LOGGING=true \
    ENABLE_METRICS=true \
    PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')" || exit 1

# Expose ports
EXPOSE 8080 8501

# Default command - runs the API server
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# =============================================================================
# Stage 3: Streamlit UI (alternative entrypoint)
# =============================================================================
FROM production as streamlit

CMD ["streamlit", "run", "src/visualization/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# =============================================================================
# Stage 4: Development
# =============================================================================
FROM production as development

USER root

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    ipython \
    jupyter

USER tomnas

ENV TOM_NAS_ENV=development \
    TOM_NAS_DEBUG=true

CMD ["/bin/bash"]
