# evolution module
from .nas_engine import NASEngine, EvolutionConfig, Individual
from .fitness import (
    ToMFitnessEvaluator, SallyAnneFitness,
    HigherOrderToMFitness, CompositeFitnessFunction,
    ToMBenchmarkFitness, create_tom_fitness_evaluator  # NEW
)
from .operators import (
    ArchitectureGene, WeightMutation, ArchitectureCrossover,
    PopulationOperators, AdaptiveMutation, SpeciesManager, CoevolutionOperator
)

__all__ = [
    'NASEngine', 'EvolutionConfig', 'Individual',
    'ToMFitnessEvaluator', 'SallyAnneFitness', 'HigherOrderToMFitness',
    'CompositeFitnessFunction',
    'ToMBenchmarkFitness', 'create_tom_fitness_evaluator',  # NEW
    'ArchitectureGene', 'WeightMutation', 'ArchitectureCrossover',
    'PopulationOperators', 'AdaptiveMutation', 'SpeciesManager', 'CoevolutionOperator'
]
