# evolution module
from .fitness import CompositeFitnessFunction, HigherOrderToMFitness, SallyAnneFitness, ToMFitnessEvaluator
from .linas import EfficientNASPipeline, FitnessPredictor, LINASSearch, PredictorTrainer
from .mutation_controller import ControllerAnalyzer, ControllerTrainer, GuidedMutator, MutationController
from .nas_engine import EvolutionConfig, Individual, NASEngine
from .operators import (
    AdaptiveMutation,
    ArchitectureCrossover,
    ArchitectureGene,
    CoevolutionOperator,
    PopulationOperators,
    SpeciesManager,
    WeightMutation,
)
from .poet_controller import (
    AgentEnvironmentPair,
    EnvironmentGenotype,
    EnvironmentType,
    POETConfig,
    POETController,
    create_preset_environment,
)

# Advanced NAS components
from .supernet import ElasticConfig, ElasticLSTMCell, ElasticTransformer, ElasticTransparentRNN
from .tom_fitness import AdversarialToMFitness, CombinedToMFitness, EarlyTerminationFitness, ToMSpecificFitness
from .zero_cost_proxies import ArchitectureFilter, ProxyScore, ProxyValidation, ZeroCostProxy

__all__ = [
    # NAS Engine
    "NASEngine",
    "EvolutionConfig",
    "Individual",
    # Fitness
    "ToMFitnessEvaluator",
    "SallyAnneFitness",
    "HigherOrderToMFitness",
    "CompositeFitnessFunction",
    # Operators
    "ArchitectureGene",
    "WeightMutation",
    "ArchitectureCrossover",
    "PopulationOperators",
    "AdaptiveMutation",
    "SpeciesManager",
    "CoevolutionOperator",
    # POET
    "POETController",
    "POETConfig",
    "EnvironmentGenotype",
    "EnvironmentType",
    "AgentEnvironmentPair",
    "create_preset_environment",
    # Advanced NAS - Supernet
    "ElasticTransformer",
    "ElasticConfig",
    "ElasticTransparentRNN",
    "ElasticLSTMCell",
    # Advanced NAS - Zero-cost proxies
    "ZeroCostProxy",
    "ArchitectureFilter",
    "ProxyScore",
    "ProxyValidation",
    # Advanced NAS - LINAS
    "LINASSearch",
    "FitnessPredictor",
    "EfficientNASPipeline",
    "PredictorTrainer",
    # Advanced NAS - Mutation Controller
    "MutationController",
    "GuidedMutator",
    "ControllerTrainer",
    "ControllerAnalyzer",
    # Advanced NAS - ToM-specific fitness
    "ToMSpecificFitness",
    "AdversarialToMFitness",
    "CombinedToMFitness",
    "EarlyTerminationFitness",
]
