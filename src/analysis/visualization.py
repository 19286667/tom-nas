"""
Visualization Tools for ToM-NAS Architecture Analysis

Provides:
1. Cell/architecture DAG visualization
2. Evolution progress plots
3. Pareto front visualization
4. Ablation impact heatmaps
5. Architecture comparison plots
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

# Matplotlib imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# NetworkX for graph operations
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Graphviz for high-quality graphs
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False


# Operation colors for visualization
OPERATION_COLORS = {
    'none': '#FFFFFF',           # White (transparent)
    'skip_connect': '#87CEEB',   # Light blue
    'conv_1x1': '#90EE90',       # Light green
    'conv_3x3': '#32CD32',       # Lime green
    'sep_conv_3x3': '#228B22',   # Forest green
    'dil_conv_3x3': '#006400',   # Dark green
    'avg_pool_3x3': '#FFA500',   # Orange
    'max_pool_3x3': '#FF8C00',   # Dark orange
    'self_attention': '#FF6B6B', # Light coral
    'multi_head_attn_4': '#DC143C',  # Crimson
    'multi_head_attn_8': '#8B0000',  # Dark red
    'gru_cell': '#9370DB',       # Medium purple
    'lstm_cell': '#8A2BE2',      # Blue violet
    'layer_norm': '#FFD700',     # Gold
    'batch_norm': '#DAA520',     # Goldenrod
    'residual_block': '#4169E1', # Royal blue
    'mlp_block': '#1E90FF',      # Dodger blue
    'recursive_block': '#9932CC', # Dark orchid
}


def visualize_cell(
    cell: Dict[str, Any],
    title: str = "Cell Architecture",
    output_path: Optional[str] = None,
    format: str = 'png',
) -> Optional[Any]:
    """
    Visualize a cell architecture as a directed graph.

    Args:
        cell: Cell dictionary with 'nodes' list
        title: Plot title
        output_path: Path to save visualization
        format: Output format ('png', 'pdf', 'svg')

    Returns:
        Graphviz Digraph object if graphviz available, else None
    """
    if HAS_GRAPHVIZ:
        return _visualize_cell_graphviz(cell, title, output_path, format)
    elif HAS_MATPLOTLIB and HAS_NETWORKX:
        return _visualize_cell_matplotlib(cell, title, output_path)
    else:
        print("Warning: Neither graphviz nor matplotlib+networkx available")
        return None


def _visualize_cell_graphviz(
    cell: Dict[str, Any],
    title: str,
    output_path: Optional[str],
    format: str,
) -> Digraph:
    """Create graphviz visualization of cell"""
    dot = Digraph(comment=title)
    dot.attr(rankdir='TB', label=title, fontsize='14')
    dot.attr('node', shape='box', style='filled')

    # Add input nodes
    dot.node('c_k-2', 'c_{k-2}', shape='rectangle', fillcolor='#E8E8E8')
    dot.node('c_k-1', 'c_{k-1}', shape='rectangle', fillcolor='#E8E8E8')

    nodes = cell.get('nodes', [])

    # Add intermediate nodes
    for i, node in enumerate(nodes):
        node_name = f'node_{i}'

        if isinstance(node, dict):
            op_name = node.get('operation', 'unknown')
        else:
            op_name = str(node)

        color = OPERATION_COLORS.get(op_name, '#CCCCCC')

        # Create label with operation name
        label = op_name.replace('_', '\n')
        dot.node(node_name, label, fillcolor=color)

        # Add edges from inputs
        if isinstance(node, dict):
            inputs = node.get('inputs', [0, 1])
            for inp_idx in inputs:
                if inp_idx == 0:
                    source = 'c_k-2'
                elif inp_idx == 1:
                    source = 'c_k-1'
                else:
                    source = f'node_{inp_idx - 2}'
                dot.edge(source, node_name)

    # Output node
    dot.node('c_k', 'c_k', shape='rectangle', fillcolor='#E8E8E8')
    for i in range(len(nodes)):
        dot.edge(f'node_{i}', 'c_k', style='dashed')

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        dot.render(output_path.stem, directory=output_path.parent, format=format, cleanup=True)

    return dot


def _visualize_cell_matplotlib(
    cell: Dict[str, Any],
    title: str,
    output_path: Optional[str],
) -> plt.Figure:
    """Create matplotlib visualization of cell"""
    nodes = cell.get('nodes', [])

    # Build graph
    G = nx.DiGraph()

    # Add nodes
    G.add_node('input_0', pos=(0, 1), label='c_{k-2}')
    G.add_node('input_1', pos=(0, 0), label='c_{k-1}')

    for i, node in enumerate(nodes):
        x = (i + 1) * 1.5
        y = 0.5
        G.add_node(f'node_{i}', pos=(x, y))

    G.add_node('output', pos=(len(nodes) + 1) * 1.5, label='c_k')

    # Add edges
    for i, node in enumerate(nodes):
        if isinstance(node, dict):
            inputs = node.get('inputs', [0, 1])
            for inp_idx in inputs:
                if inp_idx == 0:
                    source = 'input_0'
                elif inp_idx == 1:
                    source = 'input_1'
                else:
                    source = f'node_{inp_idx - 2}'
                G.add_edge(source, f'node_{i}')

        G.add_edge(f'node_{i}', 'output')

    # Draw
    fig, ax = plt.subplots(figsize=(12, 6))
    pos = nx.get_node_attributes(G, 'pos')

    if not pos:
        pos = nx.spring_layout(G)

    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
            node_size=2000, font_size=10, arrows=True)

    ax.set_title(title)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_architecture(
    architecture: Dict[str, Any],
    title: str = "Architecture",
    output_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Visualize full architecture with normal and reduction cells.

    Args:
        architecture: Architecture dictionary with 'normal_cell' and 'reduction_cell'
        title: Plot title
        output_path: Base path for output files
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Normal cell
    normal_cell = architecture.get('normal_cell', {})
    if isinstance(normal_cell, dict) and 'nodes' in normal_cell:
        _draw_cell_in_axis(axes[0], normal_cell, "Normal Cell")
    else:
        axes[0].text(0.5, 0.5, "No normal cell data", ha='center')
        axes[0].set_title("Normal Cell")

    # Reduction cell
    reduction_cell = architecture.get('reduction_cell', {})
    if isinstance(reduction_cell, dict) and 'nodes' in reduction_cell:
        _draw_cell_in_axis(axes[1], reduction_cell, "Reduction Cell")
    else:
        axes[1].text(0.5, 0.5, "No reduction cell data", ha='center')
        axes[1].set_title("Reduction Cell")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def _draw_cell_in_axis(ax, cell: Dict, title: str):
    """Draw cell graph in matplotlib axis"""
    if not HAS_NETWORKX:
        ax.text(0.5, 0.5, "NetworkX not available", ha='center')
        ax.set_title(title)
        return

    nodes = cell.get('nodes', [])

    G = nx.DiGraph()

    # Add nodes with positions
    G.add_node('in_0')
    G.add_node('in_1')

    for i in range(len(nodes)):
        G.add_node(f'n_{i}')

    G.add_node('out')

    # Add edges
    for i, node in enumerate(nodes):
        if isinstance(node, dict):
            inputs = node.get('inputs', [0, 1])
            for inp_idx in inputs:
                if inp_idx == 0:
                    G.add_edge('in_0', f'n_{i}')
                elif inp_idx == 1:
                    G.add_edge('in_1', f'n_{i}')
                else:
                    G.add_edge(f'n_{inp_idx - 2}', f'n_{i}')

        G.add_edge(f'n_{i}', 'out')

    # Layout and draw
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
            node_size=1500, font_size=8, arrows=True)
    ax.set_title(title)


def plot_evolution_curves(
    evolution_log: Dict[str, Any],
    title: str = "Evolution Progress",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot evolution progress curves.

    Shows:
    - Best fitness over generations
    - Mean fitness with std band
    - Diversity metric
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available")
        return None

    generations = evolution_log.get('generations', [])
    best_fitness = evolution_log.get('best_fitness', [])
    mean_fitness = evolution_log.get('mean_fitness', [])
    std_fitness = evolution_log.get('std_fitness', [])

    if not generations:
        print("Warning: No generation data to plot")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Best and mean fitness
    ax1 = axes[0, 0]
    ax1.plot(generations, best_fitness, 'b-', label='Best', linewidth=2)
    ax1.plot(generations, mean_fitness, 'g-', label='Mean', linewidth=1.5)
    if std_fitness:
        mean_arr = np.array(mean_fitness)
        std_arr = np.array(std_fitness)
        ax1.fill_between(generations, mean_arr - std_arr, mean_arr + std_arr,
                        alpha=0.3, color='green')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Over Generations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fitness improvement rate
    ax2 = axes[0, 1]
    if len(best_fitness) > 1:
        improvement = np.diff(best_fitness)
        ax2.bar(generations[1:], improvement, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Change')
    ax2.set_title('Fitness Improvement per Generation')
    ax2.grid(True, alpha=0.3)

    # Architecture metrics over time
    ax3 = axes[1, 0]
    arch_metrics = evolution_log.get('architecture_metrics', [])
    if arch_metrics:
        skip_counts = [m.get('num_skip_connections', 0) for m in arch_metrics]
        attn_counts = [m.get('num_attention_ops', 0) for m in arch_metrics]

        ax3.plot(generations[:len(skip_counts)], skip_counts, 'b-', label='Skip Connections')
        ax3.plot(generations[:len(attn_counts)], attn_counts, 'r-', label='Attention Ops')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Count')
        ax3.set_title('Architecture Metrics Over Time')
        ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Effective depth
    ax4 = axes[1, 1]
    if arch_metrics:
        depths = [m.get('effective_depth', 0) for m in arch_metrics]
        ax4.plot(generations[:len(depths)], depths, 'purple', linewidth=2)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Effective Depth')
    ax4.set_title('Architecture Depth Over Time')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_pareto_front(
    pareto_results: List[Dict[str, Any]],
    objective_names: Optional[List[str]] = None,
    title: str = "Pareto Front",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot Pareto front for multi-objective optimization.

    Args:
        pareto_results: List of Pareto-optimal solutions
        objective_names: Names of objectives
        title: Plot title
        output_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available")
        return None

    if not pareto_results:
        print("Warning: No Pareto results to plot")
        return None

    # Extract objectives
    objectives_list = [p.get('objectives', []) for p in pareto_results]
    objectives = np.array(objectives_list)

    if objectives.shape[1] < 2:
        print("Warning: Need at least 2 objectives for Pareto plot")
        return None

    objective_names = objective_names or [f'Objective {i+1}' for i in range(objectives.shape[1])]

    if objectives.shape[1] == 2:
        # 2D Pareto front
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(objectives[:, 0], objectives[:, 1], c='blue', s=100, alpha=0.7)

        # Connect Pareto front
        sorted_idx = np.argsort(objectives[:, 0])
        ax.plot(objectives[sorted_idx, 0], objectives[sorted_idx, 1], 'b--', alpha=0.5)

        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    else:
        # 3D or higher - show pairwise plots
        n_obj = min(objectives.shape[1], 4)
        fig, axes = plt.subplots(n_obj - 1, n_obj - 1, figsize=(12, 12))

        for i in range(n_obj - 1):
            for j in range(i + 1, n_obj):
                ax_idx = (i, j - 1) if n_obj > 2 else None
                ax = axes[ax_idx] if ax_idx and n_obj > 2 else axes

                ax.scatter(objectives[:, i], objectives[:, j], c='blue', s=50, alpha=0.7)
                ax.set_xlabel(objective_names[i])
                ax.set_ylabel(objective_names[j])
                ax.grid(True, alpha=0.3)

        # Hide empty subplots
        if n_obj > 2:
            for i in range(n_obj - 1):
                for j in range(i):
                    axes[i, j].set_visible(False)

        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_ablation_heatmap(
    ablation_results: Dict[str, Any],
    title: str = "Ablation Study Results",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Create heatmap showing ablation study impact.

    Shows performance drop matrix: tasks x ablation types
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available")
        return None

    performance_matrix = ablation_results.get('performance_matrix', {})

    if not performance_matrix:
        print("Warning: No performance matrix data")
        return None

    # Extract data for heatmap
    tasks = list(performance_matrix.keys())
    ablation_types = list(next(iter(performance_matrix.values())).keys())

    # Create matrix of relative drops
    data = np.zeros((len(tasks), len(ablation_types)))
    for i, task in enumerate(tasks):
        for j, ablation in enumerate(ablation_types):
            drop = performance_matrix[task][ablation].get('relative_drop', 0)
            data[i, j] = drop * 100  # Convert to percentage

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    cmap = LinearSegmentedColormap.from_list('impact', ['#2ECC71', '#F1C40F', '#E74C3C'])
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=np.max(np.abs(data)))

    # Add labels
    ax.set_xticks(np.arange(len(ablation_types)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(ablation_types, rotation=45, ha='right')
    ax.set_yticklabels(tasks)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Performance Drop (%)')

    # Add value annotations
    for i in range(len(tasks)):
        for j in range(len(ablation_types)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha='center', va='center', color='black', fontsize=10)

    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def create_architecture_summary(
    architecture: Dict[str, Any],
    metrics: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Create text/markdown summary of architecture.

    Returns:
        Markdown formatted summary string
    """
    lines = []

    lines.append("# Architecture Summary")
    lines.append("")

    # Basic info
    lines.append("## Structure")
    lines.append(f"- Number of cells: {architecture.get('num_cells', 'N/A')}")
    lines.append(f"- Initial channels: {architecture.get('init_channels', 'N/A')}")
    lines.append("")

    # Metrics
    lines.append("## Metrics")
    lines.append(f"- Skip connections: {metrics.get('num_skip_connections', 0)}")
    lines.append(f"- Attention operations: {metrics.get('num_attention_ops', 0)}")
    lines.append(f"- Effective depth: {metrics.get('effective_depth', 0)}")
    lines.append(f"- Total parameters: {metrics.get('total_parameters', 0):,}")
    lines.append(f"- Recursive depth: {metrics.get('recursive_depth', 0)}")
    lines.append("")

    # Hypothesis relevance
    lines.append("## Hypothesis Relevance")

    h_metrics = metrics.get('hypothesis_metrics', {})
    if h_metrics:
        h1 = h_metrics.get('H1_skip_connections', {})
        lines.append(f"- H1 (Skip connections): ratio = {h1.get('skip_ratio', 0):.2%}")

        h2 = h_metrics.get('H2_attention', {})
        lines.append(f"- H2 (Attention): ratio = {h2.get('attention_ratio', 0):.2%}")

        h3 = h_metrics.get('H3_complexity', {})
        lines.append(f"- H3 (Complexity): depth = {h3.get('effective_depth', 0)}")

        h4 = h_metrics.get('H4_recursive', {})
        lines.append(f"- H4 (Recursive): depth = {h4.get('recursive_depth', 0)}")

    summary = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(summary)

    return summary


def plot_task_comparison(
    results_by_task: Dict[str, List[Dict]],
    metric: str = 'best_fitness',
    title: str = "Performance Comparison Across Tasks",
    output_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Compare performance across different task types.
    """
    if not HAS_MATPLOTLIB:
        return None

    tasks = list(results_by_task.keys())
    means = []
    stds = []

    for task in tasks:
        values = [r.get(metric, 0) for r in results_by_task[task]]
        means.append(np.mean(values))
        stds.append(np.std(values))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(tasks))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)

    # Color bars by task type
    tom_tasks = ['tomi', 'bigtom', 'hitom_2', 'hitom_4', 'opentom', 'socialqa']
    for i, task in enumerate(tasks):
        if any(t in task.lower() for t in tom_tasks):
            bars[i].set_color('#E74C3C')  # Red for ToM
        else:
            bars[i].set_color('#3498DB')  # Blue for control

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)

    # Add legend
    tom_patch = mpatches.Patch(color='#E74C3C', label='ToM Tasks')
    control_patch = mpatches.Patch(color='#3498DB', label='Control Tasks')
    ax.legend(handles=[tom_patch, control_patch])

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
