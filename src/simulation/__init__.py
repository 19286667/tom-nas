"""
Simulation Module - Comprehensive 3D Simulation Engine

This module implements the simulation infrastructure for recursive
Theory of Mind simulations. Key components:

- SimulationNode: The fundamental container for nested simulations
- RootSimulationNode: Special root node connected to Godot
- RSCAgent: Abstract base class for Recursive Self-Compression agents
- IntegratedAgent: Full ToM agent with beliefs, actions, and memory
- SimulationWorld: 3D simulation world with partial observability
- POETEngine: Paired Open-Ended Trailblazer for co-evolution
- BenchmarkEmbedding: Natural integration of ToM benchmarks
"""

from .fractal_node import (
    SimulationStatus,
    SimulationConfig,
    WorldStateVector,
    RSCAgent,
    NeuralRSCAgent,
    RuleBasedRSCAgent,
    SimulationNode,
    RootSimulationNode,
    create_simulation,
)

# Import additional simulation components (lazy imports for optional dependencies)
try:
    from .integrated_agent import IntegratedAgent, AgentBelief, AgentMemory
except ImportError:
    IntegratedAgent = None
    AgentBelief = None
    AgentMemory = None

try:
    from .world import SimulationWorld, WorldConfig
except ImportError:
    SimulationWorld = None
    WorldConfig = None

try:
    from .poet_engine import POETEngine, EnvironmentGene
except ImportError:
    POETEngine = None
    EnvironmentGene = None

try:
    from .benchmark_embedding import BenchmarkEmbedder, EmbeddedScenario
except ImportError:
    BenchmarkEmbedder = None
    EmbeddedScenario = None

try:
    from .liminal_integration import LiminalBridge
except ImportError:
    LiminalBridge = None

__all__ = [
    # Fractal node components
    'SimulationStatus',
    'SimulationConfig',
    'WorldStateVector',
    'RSCAgent',
    'NeuralRSCAgent',
    'RuleBasedRSCAgent',
    'SimulationNode',
    'RootSimulationNode',
    'create_simulation',
    # Extended simulation components
    'IntegratedAgent',
    'AgentBelief',
    'AgentMemory',
    'SimulationWorld',
    'WorldConfig',
    'POETEngine',
    'EnvironmentGene',
    'BenchmarkEmbedder',
    'EmbeddedScenario',
    'LiminalBridge',
]
