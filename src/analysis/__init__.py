"""
Analysis and Visualization Tools for ToM-NAS Experiments
"""

from .visualization import (
    visualize_cell,
    visualize_architecture,
    plot_evolution_curves,
    plot_pareto_front,
    plot_ablation_heatmap,
    create_architecture_summary,
    plot_task_comparison,
)

from .architecture_analysis import (
    compute_architecture_similarity,
    architecture_to_graph,
    analyze_architecture_families,
    cluster_architectures,
    detect_patterns,
    compare_task_architectures,
)

from .statistical_analysis import (
    compare_distributions,
    run_hypothesis_tests,
    compute_effect_sizes,
    generate_statistical_report,
    TestResult,
)

from .rsc_metrics import (
    compute_rsc_score,
    RSCAnalysis,
    analyze_rsc_across_tasks,
    compare_rsc_tom_vs_control,
    get_rsc_hypothesis_support,
)

__all__ = [
    # Visualization
    'visualize_cell',
    'visualize_architecture',
    'plot_evolution_curves',
    'plot_pareto_front',
    'plot_ablation_heatmap',
    'create_architecture_summary',
    'plot_task_comparison',

    # Architecture analysis
    'compute_architecture_similarity',
    'architecture_to_graph',
    'analyze_architecture_families',
    'cluster_architectures',
    'detect_patterns',
    'compare_task_architectures',

    # Statistical analysis
    'compare_distributions',
    'run_hypothesis_tests',
    'compute_effect_sizes',
    'generate_statistical_report',
    'TestResult',

    # RSC metrics
    'compute_rsc_score',
    'RSCAnalysis',
    'analyze_rsc_across_tasks',
    'compare_rsc_tom_vs_control',
    'get_rsc_hypothesis_support',
]
