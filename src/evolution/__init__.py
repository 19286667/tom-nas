# evolution module
from .nas_engine import NASEngine, EvolutionConfig, Individual
from .fitness import (
    ToMFitnessEvaluator, SallyAnneFitness,
    HigherOrderToMFitness, CompositeFitnessFunction
)
from .operators import (
    ArchitectureGene, WeightMutation, ArchitectureCrossover,
    PopulationOperators, AdaptiveMutation, SpeciesManager, CoevolutionOperator
)
from .poet_controller import (
    POETController, POETConfig, EnvironmentGenotype, EnvironmentType,
    AgentEnvironmentPair, create_preset_environment
)

# Advanced NAS components
from .supernet import ElasticTransformer, ElasticConfig, ElasticTransparentRNN, ElasticLSTMCell
from .zero_cost_proxies import ZeroCostProxy, ArchitectureFilter, ProxyScore, ProxyValidation
from .linas import LINASSearch, FitnessPredictor, EfficientNASPipeline, PredictorTrainer
from .mutation_controller import MutationController, GuidedMutator, ControllerTrainer, ControllerAnalyzer
from .tom_fitness import ToMSpecificFitness, AdversarialToMFitness, CombinedToMFitness, EarlyTerminationFitness

__all__ = [
    # NAS Engine
    'NASEngine', 'EvolutionConfig', 'Individual',
    # Fitness
    'ToMFitnessEvaluator', 'SallyAnneFitness', 'HigherOrderToMFitness',
    'CompositeFitnessFunction',
    # Operators
    'ArchitectureGene', 'WeightMutation', 'ArchitectureCrossover',
    'PopulationOperators', 'AdaptiveMutation', 'SpeciesManager', 'CoevolutionOperator',
    # POET
    'POETController', 'POETConfig', 'EnvironmentGenotype', 'EnvironmentType',
    'AgentEnvironmentPair', 'create_preset_environment',
    # Advanced NAS - Supernet
    'ElasticTransformer', 'ElasticConfig', 'ElasticTransparentRNN', 'ElasticLSTMCell',
    # Advanced NAS - Zero-cost proxies
    'ZeroCostProxy', 'ArchitectureFilter', 'ProxyScore', 'ProxyValidation',
    # Advanced NAS - LINAS
    'LINASSearch', 'FitnessPredictor', 'EfficientNASPipeline', 'PredictorTrainer',
    # Advanced NAS - Mutation Controller
    'MutationController', 'GuidedMutator', 'ControllerTrainer', 'ControllerAnalyzer',
    # Advanced NAS - ToM-specific fitness
    'ToMSpecificFitness', 'AdversarialToMFitness', 'CombinedToMFitness', 'EarlyTerminationFitness',
]
