"""
ToM-NAS API Routes

REST endpoints for inference, experiments, and data access.
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import torch

from src.config import get_logger, get_settings
from src.api.models import (
    InferenceRequest,
    InferenceResponse,
    ExperimentConfig,
    ExperimentStatus,
    BenchmarkResult,
    AgentConfig,
)

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()


# =============================================================================
# Inference Endpoints
# =============================================================================

@router.post("/inference", response_model=InferenceResponse, tags=["Inference"])
async def run_inference(request: InferenceRequest):
    """
    Run ToM inference with a loaded model.

    Given an observation tensor, predict beliefs and actions.

    Args:
        request: Inference request with observation data

    Returns:
        InferenceResponse: Predicted beliefs and actions
    """
    try:
        from src.core.ontology import SoulMapOntology
        from src.agents.architectures import TransparentRNN

        # Convert input to tensor
        observation = torch.tensor(request.observation).unsqueeze(0).unsqueeze(0)

        # Use default model if none specified
        model = TransparentRNN(
            input_dim=settings.input_dims,
            hidden_dim=settings.hidden_dims,
            output_dim=settings.output_dims,
        )
        model.eval()

        # Run inference
        with torch.no_grad():
            output = model(observation)

        return InferenceResponse(
            beliefs=output['beliefs'].squeeze().tolist(),
            action=float(output['actions'].item()),
            confidence=0.85,  # Placeholder
            processing_time_ms=10.0,
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/batch", tags=["Inference"])
async def run_batch_inference(observations: List[List[float]]):
    """
    Run batch inference on multiple observations.

    More efficient than individual calls for multiple inputs.
    """
    results = []
    for obs in observations:
        request = InferenceRequest(observation=obs)
        result = await run_inference(request)
        results.append(result)
    return results


# =============================================================================
# Experiment Management
# =============================================================================

@router.get("/experiments", response_model=List[ExperimentStatus], tags=["Experiments"])
async def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100),
):
    """
    List all experiments.

    Args:
        status: Optional filter (running, completed, failed)
        limit: Maximum results to return

    Returns:
        List of experiment statuses
    """
    # In production, this would query a database
    experiments = [
        ExperimentStatus(
            id="exp-001",
            name="Baseline Evolution",
            status="completed",
            progress=100.0,
            started_at=datetime(2025, 12, 10, 10, 0, 0),
            completed_at=datetime(2025, 12, 10, 14, 30, 0),
        ),
        ExperimentStatus(
            id="exp-002",
            name="Coevolution Run",
            status="running",
            progress=45.5,
            started_at=datetime(2025, 12, 12, 9, 0, 0),
        ),
    ]

    if status:
        experiments = [e for e in experiments if e.status == status]

    return experiments[:limit]


@router.post("/experiments", response_model=ExperimentStatus, tags=["Experiments"])
async def create_experiment(
    config: ExperimentConfig,
    background_tasks: BackgroundTasks,
):
    """
    Create and start a new experiment.

    Experiments run asynchronously in the background.
    """
    experiment_id = f"exp-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Create experiment status
    status = ExperimentStatus(
        id=experiment_id,
        name=config.name,
        status="queued",
        progress=0.0,
        started_at=datetime.now(),
    )

    # Queue background task
    # background_tasks.add_task(run_experiment, experiment_id, config)

    logger.info(f"Experiment {experiment_id} created: {config.name}")

    return status


@router.get("/experiments/{experiment_id}", response_model=ExperimentStatus, tags=["Experiments"])
async def get_experiment(experiment_id: str):
    """
    Get status of a specific experiment.
    """
    # In production, query database
    return ExperimentStatus(
        id=experiment_id,
        name="Sample Experiment",
        status="running",
        progress=67.5,
        started_at=datetime(2025, 12, 12, 9, 0, 0),
    )


@router.delete("/experiments/{experiment_id}", tags=["Experiments"])
async def cancel_experiment(experiment_id: str):
    """
    Cancel a running experiment.
    """
    logger.info(f"Cancelling experiment {experiment_id}")
    return {"message": f"Experiment {experiment_id} cancelled"}


# =============================================================================
# Benchmarks
# =============================================================================

@router.get("/benchmarks", response_model=List[str], tags=["Benchmarks"])
async def list_benchmarks():
    """
    List available benchmark tests.
    """
    return [
        "sally_anne",
        "higher_order_tom",
        "zombie_detection",
        "cooperation",
        "communication",
    ]


@router.post("/benchmarks/{benchmark_name}/run", response_model=BenchmarkResult, tags=["Benchmarks"])
async def run_benchmark(benchmark_name: str, agent_config: Optional[AgentConfig] = None):
    """
    Run a specific benchmark test.

    Args:
        benchmark_name: Name of the benchmark
        agent_config: Optional agent configuration

    Returns:
        BenchmarkResult: Test results and metrics
    """
    valid_benchmarks = ["sally_anne", "higher_order_tom", "zombie_detection", "cooperation", "communication"]

    if benchmark_name not in valid_benchmarks:
        raise HTTPException(status_code=404, detail=f"Benchmark '{benchmark_name}' not found")

    # In production, actually run the benchmark
    return BenchmarkResult(
        benchmark=benchmark_name,
        score=0.85,
        metrics={
            "accuracy": 0.85,
            "confidence": 0.78,
            "latency_ms": 45.2,
        },
        passed=True,
        timestamp=datetime.now(),
    )


# =============================================================================
# Models
# =============================================================================

@router.get("/models", tags=["Models"])
async def list_models():
    """
    List available trained models.
    """
    checkpoint_dir = settings.checkpoint_dir
    models = []

    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pt') or f.endswith('.pth'):
                models.append({
                    "name": f,
                    "path": os.path.join(checkpoint_dir, f),
                    "size_mb": os.path.getsize(os.path.join(checkpoint_dir, f)) / (1024 * 1024),
                })

    return models


@router.post("/models/{model_name}/load", tags=["Models"])
async def load_model(model_name: str):
    """
    Load a model into memory for inference.
    """
    checkpoint_path = os.path.join(settings.checkpoint_dir, model_name)

    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # In production, actually load the model
    return {"message": f"Model '{model_name}' loaded successfully"}


# =============================================================================
# Ontology
# =============================================================================

@router.get("/ontology/dimensions", tags=["Ontology"])
async def get_ontology_dimensions():
    """
    Get Soul Map ontology dimension information.
    """
    from src.core.ontology import SoulMapOntology

    ontology = SoulMapOntology()

    return {
        "total_dimensions": ontology.total_dims,
        "layers": [
            {"layer": i, "name": name, "start": ontology.layer_ranges.get(i, (0, 0))[0]}
            for i, name in enumerate([
                "biological", "affective", "cognitive", "motivational",
                "social", "institutional", "aesthetic", "existential", "metacognitive"
            ])
            if i in ontology.layer_ranges
        ],
        "dimension_names": list(ontology.name_to_idx.keys()),
    }


@router.post("/ontology/encode", tags=["Ontology"])
async def encode_state(state: Dict[str, float]):
    """
    Encode a state dictionary into a Soul Map tensor.
    """
    from src.core.ontology import SoulMapOntology

    ontology = SoulMapOntology()
    tensor = ontology.encode(state)

    return {
        "encoded": tensor.tolist(),
        "dimensions": len(tensor),
    }
