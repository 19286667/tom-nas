# ToM-NAS Simulation Engine
# Optimized for low-resource NAS deployment

FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Use CPU-only requirements for NAS
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .

# Create state directory
RUN mkdir -p /app/state /app/logs

# Non-root user for security
RUN useradd -m -u 1000 tom && chown -R tom:tom /app
USER tom

# Expose internal API port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run simulation engine
CMD ["python", "-m", "src.simulation.engine"]
