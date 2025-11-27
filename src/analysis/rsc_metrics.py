"""
Recursive Self-Compression (RSC) Metrics for ToM-NAS

RSC Hypothesis: Theory of Mind requires architectures that can
represent themselves representing (strange loops / self-reference).

Key indicators for RSC:
1. Feedback connections (recurrence)
2. Hierarchical bottlenecks (compression)
3. Skip connections at multiple scales (self-reference)
4. Strange loops (paths that return to earlier layers)

These metrics capture architectural properties that may support
recursive self-modeling necessary for Theory of Mind reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Optional NetworkX for graph analysis
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class RSCAnalysis:
    """Complete RSC analysis of an architecture"""
    recurrence_score: float = 0.0
    compression_score: float = 0.0
    multiscale_score: float = 0.0
    strange_loop_count: int = 0
    rsc_total: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    interpretation: str = ""

    def to_dict(self) -> Dict:
        return {
            'recurrence_score': self.recurrence_score,
            'compression_score': self.compression_score,
            'multiscale_score': self.multiscale_score,
            'strange_loop_count': self.strange_loop_count,
            'rsc_total': self.rsc_total,
            'component_scores': self.component_scores,
            'interpretation': self.interpretation,
        }


def compute_rsc_score(architecture: Dict[str, Any]) -> RSCAnalysis:
    """
    Compute Recursive Self-Compression (RSC) score.

    The RSC score indicates how well an architecture supports
    recursive self-modeling for Theory of Mind.

    Args:
        architecture: Architecture dictionary with cell definitions

    Returns:
        RSCAnalysis with component scores and total
    """
    analysis = RSCAnalysis()

    # 1. Recurrence score - presence of feedback/recurrent connections
    analysis.recurrence_score = _compute_recurrence_score(architecture)
    analysis.component_scores['recurrence'] = analysis.recurrence_score

    # 2. Compression score - hierarchical bottlenecks
    analysis.compression_score = _compute_compression_score(architecture)
    analysis.component_scores['compression'] = analysis.compression_score

    # 3. Multi-scale skip connections - self-reference at different levels
    analysis.multiscale_score = _compute_multiscale_score(architecture)
    analysis.component_scores['multiscale'] = analysis.multiscale_score

    # 4. Strange loop detection - paths returning to earlier layers
    analysis.strange_loop_count = _detect_strange_loops(architecture)
    analysis.component_scores['strange_loops'] = min(1.0, analysis.strange_loop_count / 3.0)

    # Compute weighted total RSC score
    analysis.rsc_total = (
        0.30 * analysis.recurrence_score +
        0.25 * analysis.compression_score +
        0.25 * analysis.multiscale_score +
        0.20 * analysis.component_scores['strange_loops']
    )

    # Generate interpretation
    analysis.interpretation = _interpret_rsc(analysis)

    return analysis


def _compute_recurrence_score(architecture: Dict) -> float:
    """
    Compute recurrence score based on recurrent operations.

    Higher score indicates more feedback/recurrent processing.
    """
    recurrent_ops = {'gru_cell', 'lstm_cell', 'recursive_block'}
    count = 0
    total = 0

    for cell_type in ['normal_cell', 'reduction_cell']:
        cell = architecture.get(cell_type, {})
        nodes = cell.get('nodes', [])

        for node in nodes:
            total += 1
            if isinstance(node, dict):
                op = node.get('operation', '')
            else:
                op = str(node)

            if op in recurrent_ops or 'recur' in op.lower():
                count += 1

    if total == 0:
        return 0.0

    # Also consider effective recursive depth from num_cells
    num_cells = architecture.get('num_cells', 6)
    depth_bonus = min(0.3, num_cells / 20.0)

    base_score = count / total
    return min(1.0, base_score + depth_bonus)


def _compute_compression_score(architecture: Dict) -> float:
    """
    Compute compression score based on bottleneck structures.

    Compression indicates information distillation, important for
    abstracting and representing mental states.
    """
    compression_ops = {
        'conv_1x1': 0.3,        # Dimensionality reduction
        'avg_pool_3x3': 0.4,    # Spatial compression
        'max_pool_3x3': 0.4,    # Spatial compression
    }

    expansion_ops = {
        'mlp_block': 0.3,       # Feature expansion
        'residual_block': 0.2,  # Slight expansion
    }

    compression_count = 0.0
    expansion_count = 0.0
    total = 0

    for cell_type in ['normal_cell', 'reduction_cell']:
        cell = architecture.get(cell_type, {})
        nodes = cell.get('nodes', [])

        for node in nodes:
            total += 1
            if isinstance(node, dict):
                op = node.get('operation', '')
            else:
                op = str(node)

            if op in compression_ops:
                compression_count += compression_ops[op]
            if op in expansion_ops:
                expansion_count += expansion_ops[op]

    if total == 0:
        return 0.0

    # Ideal: compression followed by expansion (bottleneck)
    # Score higher for balanced compression-expansion
    if compression_count + expansion_count == 0:
        return 0.0

    balance = min(compression_count, expansion_count) / max(compression_count, expansion_count)
    total_intensity = (compression_count + expansion_count) / total

    return min(1.0, balance * total_intensity * 2.0)


def _compute_multiscale_score(architecture: Dict) -> float:
    """
    Compute multi-scale skip connection score.

    Multi-scale skips enable self-reference at different levels of
    abstraction - important for recursive self-modeling.
    """
    skip_scales = []  # Track scales of skip connections

    for cell_type in ['normal_cell', 'reduction_cell']:
        cell = architecture.get(cell_type, {})
        nodes = cell.get('nodes', [])

        for i, node in enumerate(nodes):
            if isinstance(node, dict):
                op = node.get('operation', '')
                inputs = node.get('inputs', [])
            else:
                op = str(node)
                inputs = [0, 1]

            if op == 'skip_connect' or op == 'residual_block':
                # Calculate skip "scale" - distance between connected layers
                for inp in inputs:
                    scale = i - inp + 2  # +2 for input nodes
                    if scale > 0:
                        skip_scales.append(scale)

    if not skip_scales:
        return 0.0

    # Score based on diversity of skip scales
    unique_scales = len(set(skip_scales))
    total_skips = len(skip_scales)

    diversity_score = unique_scales / max(1, min(total_skips, 5))
    coverage_score = min(1.0, total_skips / 4)  # 4+ skips is full coverage

    return min(1.0, 0.6 * diversity_score + 0.4 * coverage_score)


def _detect_strange_loops(architecture: Dict) -> int:
    """
    Detect strange loops - paths that return to earlier processing levels.

    In a feedforward network with recurrence, strange loops occur when:
    1. Recurrent connections feed back information
    2. Skip connections create shortcuts that enable self-reference
    3. Processing paths form cycles in the computational graph

    Note: True strange loops require recurrence; in a cell-based DAG,
    we approximate by detecting patterns that would create loops if
    unrolled across time/cells.
    """
    strange_loops = 0

    for cell_type in ['normal_cell', 'reduction_cell']:
        cell = architecture.get(cell_type, {})
        nodes = cell.get('nodes', [])

        # Build adjacency information
        recurrent_positions = []
        skip_positions = []

        for i, node in enumerate(nodes):
            if isinstance(node, dict):
                op = node.get('operation', '')
                inputs = node.get('inputs', [])
            else:
                op = str(node)
                inputs = [0, 1]

            if op in {'gru_cell', 'lstm_cell', 'recursive_block'}:
                recurrent_positions.append(i)

            if op == 'skip_connect':
                skip_positions.append((i, inputs))

        # Strange loop patterns:
        # 1. Recurrent op followed by skip that reaches before it
        for rec_pos in recurrent_positions:
            for skip_pos, skip_inputs in skip_positions:
                if skip_pos > rec_pos:
                    # Skip is after recurrent
                    for inp in skip_inputs:
                        if inp < rec_pos:
                            # Skip reaches before recurrent - potential loop
                            strange_loops += 1

        # 2. Multiple skips that create "crossing" patterns
        if len(skip_positions) >= 2:
            for i, (pos1, inputs1) in enumerate(skip_positions):
                for pos2, inputs2 in skip_positions[i+1:]:
                    # Check for crossing: one skip goes "over" another
                    if pos1 < pos2:
                        for inp1 in inputs1:
                            for inp2 in inputs2:
                                if inp2 < pos1 < inp1:
                                    strange_loops += 1

    return strange_loops


def _interpret_rsc(analysis: RSCAnalysis) -> str:
    """Generate interpretation of RSC analysis"""
    score = analysis.rsc_total

    if score < 0.2:
        level = "minimal"
        description = "Architecture lacks recursive self-modeling structures. May struggle with higher-order ToM."
    elif score < 0.4:
        level = "low"
        description = "Some recursive elements present but limited self-reference capability."
    elif score < 0.6:
        level = "moderate"
        description = "Architecture shows balanced recursive and self-referential properties."
    elif score < 0.8:
        level = "high"
        description = "Strong recursive self-compression. Suitable for complex ToM reasoning."
    else:
        level = "very high"
        description = "Architecture has extensive recursive self-modeling capability."

    components = []
    if analysis.recurrence_score > 0.5:
        components.append("strong recurrence")
    if analysis.compression_score > 0.5:
        components.append("good compression")
    if analysis.multiscale_score > 0.5:
        components.append("multi-scale connections")
    if analysis.strange_loop_count > 0:
        components.append(f"{analysis.strange_loop_count} strange loops")

    component_str = ", ".join(components) if components else "no notable components"

    return f"RSC level: {level} ({score:.2f}). {description} Components: {component_str}."


def analyze_rsc_across_tasks(
    results_by_task: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """
    Analyze RSC scores across different task types.

    Tests whether higher-order ToM tasks produce architectures
    with higher RSC scores.
    """
    task_rsc = {}

    for task_name, task_results in results_by_task.items():
        rsc_scores = []

        for result in task_results:
            arch = result.get('best_genome', result.get('architecture', result))
            if arch:
                analysis = compute_rsc_score(arch)
                rsc_scores.append(analysis.rsc_total)

        if rsc_scores:
            task_rsc[task_name] = {
                'mean_rsc': float(np.mean(rsc_scores)),
                'std_rsc': float(np.std(rsc_scores)),
                'max_rsc': float(np.max(rsc_scores)),
                'n_samples': len(rsc_scores),
            }

    # Correlation with ToM order
    tom_orders = []
    rsc_values = []

    for task_name, data in task_rsc.items():
        if 'hitom_4' in task_name.lower():
            order = 4
        elif 'hitom_3' in task_name.lower():
            order = 3
        elif 'hitom_2' in task_name.lower():
            order = 2
        elif any(t in task_name.lower() for t in ['tomi', 'bigtom', 'opentom', 'socialqa']):
            order = 1
        else:
            order = 0  # Control task

        tom_orders.append(order)
        rsc_values.append(data['mean_rsc'])

    if len(tom_orders) >= 2:
        correlation = float(np.corrcoef(tom_orders, rsc_values)[0, 1])
    else:
        correlation = 0.0

    return {
        'task_rsc_scores': task_rsc,
        'tom_order_correlation': correlation,
        'interpretation': _interpret_task_rsc(task_rsc, correlation),
    }


def _interpret_task_rsc(task_rsc: Dict, correlation: float) -> str:
    """Interpret RSC analysis across tasks"""
    if correlation > 0.7:
        corr_interp = "Strong positive correlation between ToM order and RSC score."
    elif correlation > 0.4:
        corr_interp = "Moderate positive correlation between ToM order and RSC score."
    elif correlation > 0.1:
        corr_interp = "Weak positive correlation between ToM order and RSC score."
    elif correlation > -0.1:
        corr_interp = "No correlation between ToM order and RSC score."
    else:
        corr_interp = "Negative correlation - simpler tasks have higher RSC."

    # Find highest RSC task
    if task_rsc:
        highest_task = max(task_rsc.items(), key=lambda x: x[1]['mean_rsc'])
        highest_interp = f"Highest RSC: {highest_task[0]} ({highest_task[1]['mean_rsc']:.3f})"
    else:
        highest_interp = ""

    return f"{corr_interp} {highest_interp}"


def get_rsc_hypothesis_support(analysis: RSCAnalysis) -> Dict[str, Any]:
    """
    Evaluate how well the architecture supports the RSC hypothesis.

    The RSC hypothesis states that ToM requires architectures capable
    of recursive self-representation (strange loops, self-reference).
    """
    support_indicators = {
        'has_recurrence': analysis.recurrence_score > 0.3,
        'has_compression': analysis.compression_score > 0.3,
        'has_multiscale': analysis.multiscale_score > 0.3,
        'has_strange_loops': analysis.strange_loop_count > 0,
    }

    num_indicators = sum(support_indicators.values())

    if num_indicators >= 4:
        support_level = "strong"
    elif num_indicators >= 3:
        support_level = "moderate"
    elif num_indicators >= 2:
        support_level = "weak"
    else:
        support_level = "minimal"

    return {
        'indicators': support_indicators,
        'num_indicators': num_indicators,
        'support_level': support_level,
        'rsc_total': analysis.rsc_total,
    }


def compare_rsc_tom_vs_control(
    tom_results: List[Dict],
    control_results: List[Dict],
) -> Dict[str, Any]:
    """
    Compare RSC scores between ToM and control tasks.

    Tests whether ToM tasks produce architectures with higher RSC.
    """
    tom_rsc = []
    for result in tom_results:
        arch = result.get('best_genome', result.get('architecture', result))
        if arch:
            analysis = compute_rsc_score(arch)
            tom_rsc.append(analysis.rsc_total)

    control_rsc = []
    for result in control_results:
        arch = result.get('best_genome', result.get('architecture', result))
        if arch:
            analysis = compute_rsc_score(arch)
            control_rsc.append(analysis.rsc_total)

    if not tom_rsc or not control_rsc:
        return {'error': 'Insufficient data for comparison'}

    tom_mean = float(np.mean(tom_rsc))
    control_mean = float(np.mean(control_rsc))
    difference = tom_mean - control_mean

    # Effect size
    pooled_std = np.sqrt((np.var(tom_rsc) + np.var(control_rsc)) / 2)
    effect_size = difference / pooled_std if pooled_std > 0 else 0

    return {
        'tom_mean_rsc': tom_mean,
        'control_mean_rsc': control_mean,
        'difference': difference,
        'effect_size': float(effect_size),
        'tom_higher': tom_mean > control_mean,
        'interpretation': (
            f"ToM architectures have {'higher' if difference > 0 else 'lower'} RSC "
            f"(mean: {tom_mean:.3f} vs {control_mean:.3f}, d={effect_size:.2f})"
        ),
    }
