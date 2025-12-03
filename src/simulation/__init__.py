"""
ToM-NAS Simulation Module

Core simulation components integrating:
- Integrated agents with full ToM machinery
- 3D world with partial observability
- Benchmark embedding
- POET co-evolution
"""

from .integrated_agent import (
    IntegratedAgent,
    AgentMemory,
    NestedBelief,
    Hypothesis,
    ReasoningResult,
)

from .world import (
    SimulationWorld,
    WorldState,
    Location,
    Resource,
    Observation,
    Action,
    ActionType,
    InteractionResult,
)

from .benchmark_embedding import (
    EmbeddedBenchmark,
    BenchmarkScenario,
    ToMBenchEmbed,
    SOTOPIAEmbed,
    BenchmarkResult,
)

from .poet_engine import (
    POETEngine,
    EnvironmentGene,
    AgentArchitectureGenome,
    CoevolutionPair,
)

from .visualization import (
    TerminalRenderer,
    DetailedRenderer,
    MinimalRenderer,
    RenderConfig,
    create_renderer,
)

from .shell import (
    ToMNASShell,
    SimulationConfig,
    SimulationRunner,
)

from .menu import (
    MenuSystem,
    ExperimentConfig,
)

from .liminal_integration import (
    EnvironmentBridge,
    UnifiedAgent,
    UnifiedWorld,
    IntegrationMode,
    IntegrationConfig,
    create_integrated_environment,
)

__all__ = [
    # Agent
    'IntegratedAgent',
    'AgentMemory',
    'NestedBelief',
    'Hypothesis',
    'ReasoningResult',
    # World
    'SimulationWorld',
    'WorldState',
    'Location',
    'Resource',
    'Observation',
    'Action',
    'ActionType',
    'InteractionResult',
    # Benchmarks
    'EmbeddedBenchmark',
    'BenchmarkScenario',
    'ToMBenchEmbed',
    'SOTOPIAEmbed',
    'BenchmarkResult',
    # POET
    'POETEngine',
    'EnvironmentGene',
    'AgentArchitectureGenome',
    'CoevolutionPair',
    # Visualization
    'TerminalRenderer',
    'DetailedRenderer',
    'MinimalRenderer',
    'RenderConfig',
    'create_renderer',
    # Shell & Menu
    'ToMNASShell',
    'SimulationConfig',
    'SimulationRunner',
    'MenuSystem',
    'ExperimentConfig',
    # Liminal Integration
    'EnvironmentBridge',
    'UnifiedAgent',
    'UnifiedWorld',
    'IntegrationMode',
    'IntegrationConfig',
    'create_integrated_environment',
]
