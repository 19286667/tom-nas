"""
ToM-NAS Web Application Backend
FastAPI server providing intuitive access to Neural Architecture Search for Theory of Mind
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import uuid
from datetime import datetime
from collections import deque

# Initialize FastAPI
app = FastAPI(
    title="ToM-NAS: Theory of Mind Neural Architecture Search",
    description="Evolve AI agents that understand minds",
    version="1.0.0"
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# State Management
# ============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.experiments = {}  # experiment_id -> experiment_data
        self.current_experiment = None
        self.evolution_logs = deque(maxlen=1000)
        self.torch_available = False
        self._check_torch()

    def _check_torch(self):
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False

state = AppState()

# ============================================================================
# Data Models
# ============================================================================

class ExperimentConfig(BaseModel):
    """Configuration for a new experiment"""
    name: str = "My ToM Experiment"
    population_size: int = 10
    num_generations: int = 20
    architecture_types: List[str] = ["TRN", "RSAN", "Transformer"]
    enable_zombies: bool = True
    num_agents: int = 6
    tom_order_target: int = 3

class QuickStartConfig(BaseModel):
    """Simplified config for one-click start"""
    mode: str = "balanced"  # "quick", "balanced", "thorough"

class BenchmarkRequest(BaseModel):
    """Request to run benchmarks"""
    experiment_id: Optional[str] = None
    tests: List[str] = ["sally_anne", "higher_order", "zombie_detection", "cooperation"]

# ============================================================================
# API Routes - System Status
# ============================================================================

@app.get("/")
async def root():
    """Serve the main application page"""
    return HTMLResponse(content=open(
        os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates', 'index.html')
    ).read())

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "ready" if state.torch_available else "limited",
        "torch_available": state.torch_available,
        "active_experiments": len(state.experiments),
        "current_experiment": state.current_experiment,
        "message": "System ready for evolution!" if state.torch_available else "Running in demo mode (PyTorch not installed)"
    }

@app.get("/api/concepts")
async def get_concepts():
    """Get explanations of key concepts - making ToM intuitive"""
    return {
        "theory_of_mind": {
            "simple": "The ability to understand that others have their own thoughts, beliefs, and intentions",
            "example": "Knowing that your friend thinks the cookies are in the jar, even though you saw them get moved",
            "why_it_matters": "Essential for social intelligence, cooperation, and understanding deception"
        },
        "neural_architecture_search": {
            "simple": "Letting AI design its own brain structure through evolution",
            "example": "Like breeding dogs for specific traits, but breeding neural networks for mind-reading ability",
            "why_it_matters": "Discovers architectures humans might never think of"
        },
        "zombie_detection": {
            "simple": "Testing if an AI truly understands minds or just fakes it",
            "example": "A chatbot might seem empathetic but not actually model your beliefs",
            "why_it_matters": "Ensures genuine understanding, not clever mimicry"
        },
        "architectures": {
            "TRN": {
                "name": "Transparent Recurrent Network",
                "simple": "A network where we can see exactly how it thinks",
                "strength": "Interpretability - we can trace every decision"
            },
            "RSAN": {
                "name": "Recursive Self-Attention Network",
                "simple": "A network that thinks about thinking about thinking...",
                "strength": "Deep recursive reasoning about beliefs"
            },
            "Transformer": {
                "name": "Transformer Agent",
                "simple": "The architecture behind ChatGPT, adapted for mind-reading",
                "strength": "Powerful pattern recognition and scalability"
            },
            "Hybrid": {
                "name": "Evolved Hybrid",
                "simple": "Evolution combines the best of all architectures",
                "strength": "Discovers novel combinations"
            }
        },
        "sally_anne_test": {
            "simple": "The classic test: Sally puts a marble in a basket, leaves, Anne moves it. Where will Sally look?",
            "answer": "The basket - because Sally doesn't know it was moved!",
            "what_it_tests": "False belief understanding - knowing others can have wrong beliefs"
        }
    }

# ============================================================================
# API Routes - Experiment Management
# ============================================================================

@app.post("/api/experiments/quick-start")
async def quick_start(config: QuickStartConfig, background_tasks: BackgroundTasks):
    """One-click experiment start with sensible defaults"""
    presets = {
        "quick": {
            "name": "Quick Exploration",
            "population_size": 6,
            "num_generations": 5,
            "description": "Fast results in ~2 minutes. Good for testing."
        },
        "balanced": {
            "name": "Balanced Evolution",
            "population_size": 12,
            "num_generations": 15,
            "description": "Good balance of speed and quality. ~10 minutes."
        },
        "thorough": {
            "name": "Deep Evolution",
            "population_size": 20,
            "num_generations": 50,
            "description": "Best results, longer runtime. ~30+ minutes."
        }
    }

    preset = presets.get(config.mode, presets["balanced"])

    experiment_id = str(uuid.uuid4())[:8]
    experiment = {
        "id": experiment_id,
        "name": preset["name"],
        "config": preset,
        "status": "initializing",
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "current_generation": 0,
        "best_fitness": 0,
        "logs": [],
        "results": None,
        "visualization_data": {
            "fitness_history": [],
            "diversity_history": [],
            "architecture_distribution": {},
            "best_architecture": None
        }
    }

    state.experiments[experiment_id] = experiment
    state.current_experiment = experiment_id

    # Start evolution in background
    if state.torch_available:
        background_tasks.add_task(run_evolution, experiment_id)
    else:
        # Demo mode - simulate evolution
        background_tasks.add_task(simulate_evolution, experiment_id)

    return {
        "experiment_id": experiment_id,
        "message": f"Started '{preset['name']}' - {preset['description']}",
        "mode": config.mode
    }

@app.post("/api/experiments/create")
async def create_experiment(config: ExperimentConfig, background_tasks: BackgroundTasks):
    """Create a new experiment with custom configuration"""
    experiment_id = str(uuid.uuid4())[:8]

    experiment = {
        "id": experiment_id,
        "name": config.name,
        "config": config.dict(),
        "status": "initializing",
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "current_generation": 0,
        "best_fitness": 0,
        "logs": [],
        "results": None,
        "visualization_data": {
            "fitness_history": [],
            "diversity_history": [],
            "architecture_distribution": {},
            "best_architecture": None
        }
    }

    state.experiments[experiment_id] = experiment
    state.current_experiment = experiment_id

    if state.torch_available:
        background_tasks.add_task(run_evolution, experiment_id)
    else:
        background_tasks.add_task(simulate_evolution, experiment_id)

    return {"experiment_id": experiment_id, "status": "started"}

@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details and progress"""
    if experiment_id not in state.experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return state.experiments[experiment_id]

@app.get("/api/experiments/{experiment_id}/stream")
async def stream_experiment(experiment_id: str):
    """Get latest updates for an experiment"""
    if experiment_id not in state.experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp = state.experiments[experiment_id]
    return {
        "status": exp["status"],
        "progress": exp["progress"],
        "current_generation": exp["current_generation"],
        "best_fitness": exp["best_fitness"],
        "latest_logs": exp["logs"][-10:],
        "visualization_data": exp["visualization_data"]
    }

@app.get("/api/experiments")
async def list_experiments():
    """List all experiments"""
    return {
        "experiments": [
            {
                "id": exp_id,
                "name": exp["name"],
                "status": exp["status"],
                "progress": exp["progress"],
                "best_fitness": exp["best_fitness"],
                "created_at": exp["created_at"]
            }
            for exp_id, exp in state.experiments.items()
        ]
    }

# ============================================================================
# API Routes - Benchmarks & Analysis
# ============================================================================

@app.post("/api/benchmarks/run")
async def run_benchmarks(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Run ToM benchmark suite"""
    benchmark_id = str(uuid.uuid4())[:8]

    result = {
        "benchmark_id": benchmark_id,
        "status": "running",
        "tests": request.tests,
        "results": None
    }

    # In demo mode, return simulated results
    if not state.torch_available:
        result["status"] = "complete"
        result["results"] = generate_demo_benchmark_results(request.tests)

    return result

@app.get("/api/interpretability/{experiment_id}")
async def get_interpretability(experiment_id: str):
    """Get interpretability data for an experiment's best model"""
    if experiment_id not in state.experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp = state.experiments[experiment_id]

    # Generate interpretability insights
    return {
        "experiment_id": experiment_id,
        "architecture": exp["visualization_data"].get("best_architecture", {}),
        "insights": {
            "attention_patterns": {
                "description": "Where the model focuses when reasoning about others",
                "interpretation": "High attention on agent presence and belief markers indicates ToM reasoning"
            },
            "gate_activations": {
                "description": "How the model controls information flow",
                "interpretation": "Update gates show what new information is integrated"
            },
            "belief_confidence": {
                "description": "How certain the model is about others' mental states",
                "interpretation": "Confidence should decrease with higher-order beliefs (uncertainty stacking)"
            }
        },
        "recommendations": [
            "Models with high Sally-Anne scores show genuine false belief understanding",
            "Watch for zombie detection ability - it indicates robust ToM vs. pattern matching",
            "Higher-order ToM (order 3+) is the hallmark of sophisticated mind-reading"
        ]
    }

# ============================================================================
# Background Tasks
# ============================================================================

async def run_evolution(experiment_id: str):
    """Run actual evolution (when PyTorch is available)"""
    exp = state.experiments[experiment_id]
    exp["status"] = "running"

    try:
        import torch
        from src.core.ontology import SoulMapOntology
        from src.core.beliefs import BeliefNetwork
        from src.world.social_world import SocialWorld4
        from src.evolution.nas_engine import NASEngine, EvolutionConfig

        config = exp["config"]
        pop_size = config.get("population_size", 10)
        num_gens = config.get("num_generations", 10)

        # Initialize components
        ontology = SoulMapOntology()
        world = SocialWorld4(num_agents=6, ontology_dim=181, num_zombies=2)
        belief_network = BeliefNetwork(num_agents=6, state_dim=181)

        # Configure evolution
        evo_config = EvolutionConfig(
            population_size=pop_size,
            num_generations=num_gens,
            device='cpu'
        )

        engine = NASEngine(evo_config, world, belief_network)
        engine.initialize_population()

        # Run evolution with progress updates
        for gen in range(num_gens):
            engine.evolve_generation()

            # Update experiment state
            progress = int((gen + 1) / num_gens * 100)
            exp["progress"] = progress
            exp["current_generation"] = gen + 1
            exp["best_fitness"] = engine.best_individual.fitness if engine.best_individual else 0

            # Update visualization data
            exp["visualization_data"]["fitness_history"].append({
                "generation": gen + 1,
                "best": engine.history["best_fitness"][-1] if engine.history["best_fitness"] else 0,
                "average": engine.history["avg_fitness"][-1] if engine.history["avg_fitness"] else 0
            })

            if engine.history.get("diversity"):
                exp["visualization_data"]["diversity_history"].append({
                    "generation": gen + 1,
                    "diversity": engine.history["diversity"][-1]
                })

            # Log
            exp["logs"].append({
                "time": datetime.now().isoformat(),
                "generation": gen + 1,
                "message": f"Generation {gen + 1}: Best fitness = {exp['best_fitness']:.4f}"
            })

            await asyncio.sleep(0.1)  # Allow other tasks

        # Store final results
        if engine.best_individual:
            exp["visualization_data"]["best_architecture"] = {
                "type": engine.best_individual.gene.gene_dict.get("arch_type", "Unknown"),
                "hidden_dim": engine.best_individual.gene.gene_dict.get("hidden_dim", 128),
                "num_layers": engine.best_individual.gene.gene_dict.get("num_layers", 2),
                "fitness": engine.best_individual.fitness
            }

        exp["status"] = "complete"
        exp["results"] = engine.get_evolution_summary()

    except Exception as e:
        exp["status"] = "error"
        exp["logs"].append({
            "time": datetime.now().isoformat(),
            "message": f"Error: {str(e)}"
        })

async def simulate_evolution(experiment_id: str):
    """Simulate evolution for demo mode"""
    import random

    exp = state.experiments[experiment_id]
    exp["status"] = "running"

    config = exp["config"]
    num_gens = config.get("num_generations", 10)

    best_fitness = 0.1

    for gen in range(num_gens):
        await asyncio.sleep(0.5)  # Simulate computation time

        # Simulate improving fitness
        improvement = random.uniform(0.01, 0.05)
        best_fitness = min(1.0, best_fitness + improvement)
        avg_fitness = best_fitness * random.uniform(0.6, 0.9)
        diversity = random.uniform(0.3, 0.7)

        progress = int((gen + 1) / num_gens * 100)
        exp["progress"] = progress
        exp["current_generation"] = gen + 1
        exp["best_fitness"] = round(best_fitness, 4)

        exp["visualization_data"]["fitness_history"].append({
            "generation": gen + 1,
            "best": round(best_fitness, 4),
            "average": round(avg_fitness, 4)
        })

        exp["visualization_data"]["diversity_history"].append({
            "generation": gen + 1,
            "diversity": round(diversity, 4)
        })

        exp["logs"].append({
            "time": datetime.now().isoformat(),
            "generation": gen + 1,
            "message": f"Generation {gen + 1}: Best fitness = {best_fitness:.4f}"
        })

    # Final architecture
    arch_types = ["TRN", "RSAN", "Transformer", "Hybrid"]
    exp["visualization_data"]["best_architecture"] = {
        "type": random.choice(arch_types),
        "hidden_dim": random.choice([128, 256]),
        "num_layers": random.choice([2, 3, 4]),
        "fitness": round(best_fitness, 4)
    }

    exp["status"] = "complete"
    exp["results"] = {
        "total_generations": num_gens,
        "best_fitness": round(best_fitness, 4),
        "demo_mode": True
    }

def generate_demo_benchmark_results(tests: List[str]) -> Dict:
    """Generate demo benchmark results"""
    import random

    results = {}

    if "sally_anne" in tests:
        results["sally_anne"] = {
            "basic": {"score": random.uniform(0.7, 1.0), "passed": True},
            "second_order": {"score": random.uniform(0.5, 0.8), "passed": random.random() > 0.3}
        }

    if "higher_order" in tests:
        results["higher_order"] = {
            f"order_{i}": {"score": max(0.3, 1.0 - i * 0.15 + random.uniform(-0.1, 0.1)), "passed": i < 4}
            for i in range(1, 6)
        }

    if "zombie_detection" in tests:
        zombie_types = ["behavioral", "belief", "causal", "metacognitive", "linguistic", "emotional"]
        results["zombie_detection"] = {
            ztype: {"score": random.uniform(0.4, 0.9), "passed": random.random() > 0.4}
            for ztype in zombie_types
        }

    if "cooperation" in tests:
        results["cooperation"] = {
            "reciprocation_rate": random.uniform(0.6, 0.9),
            "cooperation_rate": random.uniform(0.5, 0.8),
            "passed": True
        }

    return results

# ============================================================================
# Static Files
# ============================================================================

# Mount static files after all routes
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static')), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
