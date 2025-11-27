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

# New experimental design components
from .search_space import (
    ArchitectureGenome, ArchitectureMetrics, CellArchitecture, CellNode,
    OPERATION_SPACE, ARCHITECTURE_METRICS, SearchSpaceFactory,
    create_random_genome, mutate_genome, crossover_genomes, genome_distance
)
from .evo_nas_jax import (
    EvosaxConfig, NumpyEvolutionEngine, ContinuousGenomeEncoder,
    run_evolutionary_search, compare_strategies, create_evolution_engine
)
from .multi_objective import (
    MultiObjectiveConfig, NSGA2Engine, run_pareto_search, analyze_pareto_front
)
from .nas_bench import (
    NASBench201Surrogate, NASBench301Surrogate,
    run_benchmark_baseline_study, compare_tom_vs_baseline
)
from .ablations import (
    AblationConfig, AblationStudy, AblationResult,
    ablation_remove_skip_connections, ablation_remove_attention,
    run_quick_ablation_study
)

__all__ = [
    # Existing exports
    'NASEngine', 'EvolutionConfig', 'Individual',
    'ToMFitnessEvaluator', 'SallyAnneFitness', 'HigherOrderToMFitness',
    'CompositeFitnessFunction',
    'ArchitectureGene', 'WeightMutation', 'ArchitectureCrossover',
    'PopulationOperators', 'AdaptiveMutation', 'SpeciesManager', 'CoevolutionOperator',

    # Search space
    'ArchitectureGenome', 'ArchitectureMetrics', 'CellArchitecture', 'CellNode',
    'OPERATION_SPACE', 'ARCHITECTURE_METRICS', 'SearchSpaceFactory',
    'create_random_genome', 'mutate_genome', 'crossover_genomes', 'genome_distance',

    # Evolutionary search
    'EvosaxConfig', 'NumpyEvolutionEngine', 'ContinuousGenomeEncoder',
    'run_evolutionary_search', 'compare_strategies', 'create_evolution_engine',

    # Multi-objective
    'MultiObjectiveConfig', 'NSGA2Engine', 'run_pareto_search', 'analyze_pareto_front',

    # NAS-Bench
    'NASBench201Surrogate', 'NASBench301Surrogate',
    'run_benchmark_baseline_study', 'compare_tom_vs_baseline',

    # Ablations
    'AblationConfig', 'AblationStudy', 'AblationResult',
    'ablation_remove_skip_connections', 'ablation_remove_attention',
    'run_quick_ablation_study',
]
