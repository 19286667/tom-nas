"""
ToM-NAS FastAPI Application

Production-ready REST API for Theory of Mind Neural Architecture Search.
Designed for Google Cloud Run deployment.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import setup_logging, get_logger, get_settings
from src.api.routes import router
from src.api.models import HealthResponse, InfoResponse

# Initialize logging
settings = get_settings()
setup_logging(
    level=settings.log_level,
    enable_cloud_logging=settings.enable_cloud_logging,
    log_file=f"{settings.logs_dir}/api.log" if not settings.is_production else None,
    structured=settings.is_production,
)

logger = get_logger(__name__)


# Application state for loaded models
class AppState:
    """Global application state."""
    models: Dict[str, Any] = {}
    ontology = None
    is_ready: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("ToM-NAS API starting up...")

    try:
        # Import components
        from src.core.ontology import SoulMapOntology

        # Initialize ontology
        state.ontology = SoulMapOntology()
        logger.info(f"Ontology initialized with {state.ontology.total_dims} dimensions")

        # Set device
        device = torch.device(settings.device)
        logger.info(f"Using device: {device}")

        state.is_ready = True
        logger.info("ToM-NAS API ready to serve requests")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        state.is_ready = False

    yield

    # Shutdown
    logger.info("ToM-NAS API shutting down...")
    state.models.clear()
    state.is_ready = False


# Create FastAPI application
app = FastAPI(
    title="ToM-NAS API",
    description="""
    Theory of Mind Neural Architecture Search REST API.

    Provides endpoints for:
    - Running inference with evolved ToM agents
    - Managing experiments and training runs
    - Accessing benchmark results
    - Health monitoring

    ## Authentication

    For production deployments, use Google Cloud IAM or API keys.

    ## Rate Limits

    - Standard tier: 100 requests/minute
    - Premium tier: 1000 requests/minute
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [
        "https://*.run.app",  # Cloud Run
        "https://*.appspot.com",  # App Engine
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.debug else None,
        }
    )


# Health check endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and orchestrators.

    Returns:
        HealthResponse: Current health status
    """
    return HealthResponse(
        status="healthy" if state.is_ready else "unhealthy",
        ready=state.is_ready,
        models_loaded=len(state.models),
    )


@app.get("/", response_model=InfoResponse, tags=["Info"])
async def root():
    """
    Root endpoint with API information.

    Returns:
        InfoResponse: API metadata
    """
    return InfoResponse(
        name="ToM-NAS API",
        version="1.0.0",
        description="Theory of Mind Neural Architecture Search",
        docs_url="/docs",
        health_url="/health",
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


# Metrics endpoint (for Prometheus)
if settings.enable_metrics:
    try:
        from prometheus_client import make_asgi_app, Counter, Histogram

        # Create metrics
        REQUEST_COUNT = Counter(
            'tom_nas_requests_total',
            'Total request count',
            ['method', 'endpoint', 'status']
        )
        REQUEST_LATENCY = Histogram(
            'tom_nas_request_latency_seconds',
            'Request latency in seconds',
            ['method', 'endpoint']
        )

        # Mount metrics endpoint
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

        logger.info("Prometheus metrics enabled at /metrics")

    except ImportError:
        logger.warning("prometheus_client not installed, metrics disabled")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
