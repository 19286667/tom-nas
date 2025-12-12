"""
ToM-NAS API Pydantic Models

Request and response models for the REST API.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


# =============================================================================
# Health & Info
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status (healthy/unhealthy)")
    ready: bool = Field(..., description="Whether the service is ready")
    models_loaded: int = Field(0, description="Number of models in memory")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "ready": True,
                "models_loaded": 2
            }
        }
    }


class InfoResponse(BaseModel):
    """API information response."""
    name: str
    version: str
    description: str
    docs_url: str
    health_url: str


# =============================================================================
# Inference
# =============================================================================

class InferenceRequest(BaseModel):
    """Inference request model."""
    observation: List[float] = Field(
        ...,
        description="Input observation tensor",
        min_length=1,
    )
    model_name: Optional[str] = Field(
        None,
        description="Specific model to use (default: latest)"
    )
    return_hidden_states: bool = Field(
        False,
        description="Whether to return intermediate hidden states"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "observation": [0.5] * 191,
                "model_name": "best_evolved_agent.pt",
                "return_hidden_states": False
            }
        }
    }


class InferenceResponse(BaseModel):
    """Inference response model."""
    beliefs: List[float] = Field(..., description="Predicted belief state (181 dims)")
    action: float = Field(..., description="Predicted action value")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    processing_time_ms: float = Field(..., description="Inference time in milliseconds")
    hidden_states: Optional[List[List[float]]] = Field(
        None,
        description="Intermediate hidden states (if requested)"
    )


# =============================================================================
# Experiments
# =============================================================================

class ExperimentConfig(BaseModel):
    """Configuration for a new experiment."""
    name: str = Field(..., min_length=1, max_length=100)
    experiment_type: str = Field(
        "evolution",
        description="Type: baseline, evolution, coevolution, comparison"
    )
    population_size: int = Field(20, ge=5, le=200)
    generations: int = Field(100, ge=1, le=10000)
    architecture: str = Field(
        "hybrid",
        description="Architecture: TRN, RSAN, Transformer, hybrid"
    )
    fitness_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Custom fitness component weights"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "High Population Evolution",
                "experiment_type": "evolution",
                "population_size": 50,
                "generations": 200,
                "architecture": "hybrid"
            }
        }
    }


class ExperimentStatus(BaseModel):
    """Status of an experiment."""
    id: str
    name: str
    status: str = Field(..., description="Status: queued, running, completed, failed")
    progress: float = Field(..., ge=0, le=100, description="Completion percentage")
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    best_fitness: Optional[float] = None
    current_generation: Optional[int] = None


# =============================================================================
# Benchmarks
# =============================================================================

class BenchmarkResult(BaseModel):
    """Result of a benchmark test."""
    benchmark: str
    score: float = Field(..., ge=0, le=1)
    metrics: Dict[str, float]
    passed: bool
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# Agents & Models
# =============================================================================

class AgentConfig(BaseModel):
    """Configuration for an agent."""
    architecture: str = Field("TRN", description="TRN, RSAN, Transformer, hybrid")
    hidden_dim: int = Field(128, ge=32, le=1024)
    num_layers: int = Field(2, ge=1, le=10)
    num_heads: int = Field(4, ge=1, le=16)
    checkpoint_path: Optional[str] = None


class ModelInfo(BaseModel):
    """Information about a saved model."""
    name: str
    architecture: str
    created_at: datetime
    fitness_score: Optional[float] = None
    parameters: int
    size_mb: float


# =============================================================================
# Social World
# =============================================================================

class AgentState(BaseModel):
    """State of an agent in the social world."""
    agent_id: int
    soul_map: List[float]
    resources: float
    energy: float
    reputation: Dict[int, float]


class WorldState(BaseModel):
    """State of the social world simulation."""
    step: int
    agents: List[AgentState]
    coalitions: List[List[int]]
    recent_events: List[str]


# =============================================================================
# Beliefs
# =============================================================================

class BeliefQuery(BaseModel):
    """Query for belief state."""
    agent_id: int
    order: int = Field(..., ge=0, le=5, description="ToM order (0-5)")
    target_id: Optional[int] = None


class BeliefState(BaseModel):
    """Belief state response."""
    agent_id: int
    order: int
    target_id: Optional[int]
    content: List[float]
    confidence: float
    timestamp: int
