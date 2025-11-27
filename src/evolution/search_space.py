"""
Search Space Definition for ToM-NAS Experiments
Unified search space for controlled NAS comparison across task complexity

This implements the DARTS-style cell-based search space with operations
designed to test hypotheses about skip connections, attention, and
recursive structures in Theory of Mind reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import copy


class OperationType(Enum):
    """Primitive operation types for architecture search"""
    NONE = "none"
    SKIP_CONNECT = "skip_connect"
    CONV_1X1 = "conv_1x1"
    CONV_3X3 = "conv_3x3"
    SEP_CONV_3X3 = "sep_conv_3x3"
    DIL_CONV_3X3 = "dil_conv_3x3"
    AVG_POOL_3X3 = "avg_pool_3x3"
    MAX_POOL_3X3 = "max_pool_3x3"
    SELF_ATTENTION = "self_attention"
    MULTI_HEAD_ATTN_4 = "multi_head_attn_4"
    MULTI_HEAD_ATTN_8 = "multi_head_attn_8"
    GRU_CELL = "gru_cell"
    LSTM_CELL = "lstm_cell"
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"
    RESIDUAL_BLOCK = "residual_block"
    MLP_BLOCK = "mlp_block"
    RECURSIVE_BLOCK = "recursive_block"


# Full operation space for experiments
OPERATION_SPACE = {
    'primitives': [
        'none',           # No connection (allows sparsity)
        'skip_connect',   # Identity mapping - KEY for H1
        'conv_1x1',       # Point-wise convolution
        'conv_3x3',       # Standard convolution
        'sep_conv_3x3',   # Depthwise separable
        'dil_conv_3x3',   # Dilated convolution
        'avg_pool_3x3',   # Average pooling
        'max_pool_3x3',   # Max pooling
        'self_attention', # Single-head self-attention - KEY for H2
        'multi_head_attn_4',  # 4-head attention - KEY for H2
        'multi_head_attn_8',  # 8-head attention
        'gru_cell',       # Recurrent unit
        'lstm_cell',      # LSTM unit
        'layer_norm',     # Normalization
        'batch_norm',     # Batch normalization
        'residual_block', # Pre-defined residual
        'mlp_block',      # MLP with hidden expansion
        'recursive_block', # Recursive processing block - KEY for H4
    ],

    # Cell-based search (DARTS-style)
    'cell_nodes': 4,          # Intermediate nodes per cell
    'cell_concat': 4,         # Nodes to concatenate for output

    # Macro search options
    'num_cells': [4, 6, 8, 10, 12],
    'channels': [16, 32, 64, 128, 256],
    'reduction_cells': [2, 4],  # Positions of reduction cells

    # Attention-specific options
    'num_heads_options': [1, 2, 4, 8],
    'attention_dropout': [0.0, 0.1, 0.2],

    # Recurrence options
    'max_recursion_depth': [1, 2, 3, 4, 5],
}


# Metrics to track during architecture evolution
ARCHITECTURE_METRICS = [
    'num_skip_connections',    # H1: Skip connections in ToM vs control
    'num_attention_ops',       # H2: Attention mechanisms
    'effective_depth',         # H3: Architecture complexity
    'total_parameters',        # Model size
    'path_length_variance',    # Information flow patterns
    'branching_factor',        # DAG structure
    'recursive_depth',         # H4: Nested structure depth
    'num_recurrent_ops',       # Recurrent operations count
    'compression_ratio',       # Bottleneck analysis
    'attention_span',          # Average attention distance
]


@dataclass
class CellNode:
    """Represents a node in a DARTS-style cell"""
    operation: str
    inputs: List[int]  # Indices of input nodes (0, 1 = cell inputs, 2+ = intermediate)

    def to_dict(self) -> Dict:
        return {'operation': self.operation, 'inputs': self.inputs}

    @classmethod
    def from_dict(cls, d: Dict) -> 'CellNode':
        return cls(operation=d['operation'], inputs=d['inputs'])


@dataclass
class CellArchitecture:
    """Represents a complete cell architecture"""
    nodes: List[CellNode]
    cell_type: str = 'normal'  # 'normal' or 'reduction'

    def to_dict(self) -> Dict:
        return {
            'nodes': [n.to_dict() for n in self.nodes],
            'cell_type': self.cell_type
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'CellArchitecture':
        return cls(
            nodes=[CellNode.from_dict(n) for n in d['nodes']],
            cell_type=d.get('cell_type', 'normal')
        )


@dataclass
class ArchitectureGenome:
    """
    Genome encoding for neural architectures.
    Supports both continuous (for gradient-based) and discrete (for evolutionary) representations.
    """

    num_nodes: int = 4
    num_ops: int = len(OPERATION_SPACE['primitives'])
    normal_cell: Optional[CellArchitecture] = None
    reduction_cell: Optional[CellArchitecture] = None

    # Macro-level genes
    num_cells: int = 6
    init_channels: int = 32

    # Continuous representation for gradient-based NAS
    alpha_normal: Optional[np.ndarray] = None  # Architecture weights for normal cell
    alpha_reduce: Optional[np.ndarray] = None  # Architecture weights for reduction cell

    def __post_init__(self):
        if self.normal_cell is None:
            self.normal_cell = self._random_cell('normal')
        if self.reduction_cell is None:
            self.reduction_cell = self._random_cell('reduction')

    def _random_cell(self, cell_type: str) -> CellArchitecture:
        """Generate a random cell architecture"""
        nodes = []
        primitives = OPERATION_SPACE['primitives']

        for i in range(self.num_nodes):
            # Each node takes 2 inputs from previous nodes/inputs
            max_input = i + 2  # 0, 1 are cell inputs, then intermediate nodes
            input_1 = np.random.randint(0, max_input)
            input_2 = np.random.randint(0, max_input)
            op = np.random.choice(primitives)

            nodes.append(CellNode(operation=op, inputs=[input_1, input_2]))

        return CellArchitecture(nodes=nodes, cell_type=cell_type)

    @property
    def genome_size(self) -> int:
        """Size of continuous genome representation"""
        # For each node: op_choice + 2 inputs
        return self.num_nodes * 3 * 2  # *2 for normal and reduction cells

    def to_continuous(self) -> np.ndarray:
        """Convert discrete architecture to continuous genome"""
        primitives = OPERATION_SPACE['primitives']
        genome = []

        for cell in [self.normal_cell, self.reduction_cell]:
            for node in cell.nodes:
                # Encode operation as normalized index
                op_idx = primitives.index(node.operation) if node.operation in primitives else 0
                genome.append(op_idx / self.num_ops)

                # Encode inputs
                max_input = len(cell.nodes) + 2
                genome.append(node.inputs[0] / max_input)
                genome.append(node.inputs[1] / max_input)

        return np.array(genome, dtype=np.float32)

    @classmethod
    def from_continuous(cls, genome: np.ndarray, num_nodes: int = 4) -> 'ArchitectureGenome':
        """Decode continuous genome to discrete architecture"""
        primitives = OPERATION_SPACE['primitives']
        num_ops = len(primitives)

        arch = cls(num_nodes=num_nodes)

        idx = 0
        for cell_idx, cell in enumerate([arch.normal_cell, arch.reduction_cell]):
            new_nodes = []
            for i in range(num_nodes):
                # Decode operation
                op_idx = int(genome[idx] * num_ops) % num_ops
                op = primitives[op_idx]
                idx += 1

                # Decode inputs
                max_input = i + 2
                input_1 = int(genome[idx] * max_input) % max_input
                idx += 1
                input_2 = int(genome[idx] * max_input) % max_input
                idx += 1

                new_nodes.append(CellNode(operation=op, inputs=[input_1, input_2]))

            if cell_idx == 0:
                arch.normal_cell = CellArchitecture(nodes=new_nodes, cell_type='normal')
            else:
                arch.reduction_cell = CellArchitecture(nodes=new_nodes, cell_type='reduction')

        return arch

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'num_nodes': self.num_nodes,
            'num_ops': self.num_ops,
            'normal_cell': self.normal_cell.to_dict(),
            'reduction_cell': self.reduction_cell.to_dict(),
            'num_cells': self.num_cells,
            'init_channels': self.init_channels,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'ArchitectureGenome':
        """Deserialize from dictionary"""
        arch = cls(
            num_nodes=d['num_nodes'],
            num_ops=d.get('num_ops', len(OPERATION_SPACE['primitives'])),
            num_cells=d.get('num_cells', 6),
            init_channels=d.get('init_channels', 32),
        )
        arch.normal_cell = CellArchitecture.from_dict(d['normal_cell'])
        arch.reduction_cell = CellArchitecture.from_dict(d['reduction_cell'])
        return arch


class ArchitectureMetrics:
    """
    Compute and track architectural metrics for hypothesis testing.

    H1: NAS on ToM tasks will converge to architectures with significantly more skip connections
    H2: NAS on ToM tasks will discover attention-like mechanisms at higher rates
    H3: Architecture complexity (depth, branching) will correlate with ToM task complexity
    H4: Architectures discovered for ToM will exhibit recursive/hierarchical structure
    """

    def __init__(self, genome: ArchitectureGenome):
        self.genome = genome
        self._metrics: Dict[str, float] = {}
        self._computed = False

    def compute_all(self) -> Dict[str, float]:
        """Compute all architectural metrics"""
        if self._computed:
            return self._metrics

        self._metrics = {
            'num_skip_connections': self._count_skip_connections(),
            'num_attention_ops': self._count_attention_ops(),
            'effective_depth': self._compute_effective_depth(),
            'total_parameters': self._estimate_parameters(),
            'path_length_variance': self._compute_path_variance(),
            'branching_factor': self._compute_branching_factor(),
            'recursive_depth': self._compute_recursive_depth(),
            'num_recurrent_ops': self._count_recurrent_ops(),
            'compression_ratio': self._compute_compression_ratio(),
            'attention_span': self._compute_attention_span(),
        }

        self._computed = True
        return self._metrics

    def _count_skip_connections(self) -> int:
        """Count skip connections across all cells (H1 metric)"""
        count = 0
        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            for node in cell.nodes:
                if node.operation == 'skip_connect':
                    count += 1
                elif node.operation == 'residual_block':
                    count += 1  # Residual blocks contain skip connections
        return count * self.genome.num_cells

    def _count_attention_ops(self) -> int:
        """Count attention operations (H2 metric)"""
        attention_ops = {'self_attention', 'multi_head_attn_4', 'multi_head_attn_8'}
        count = 0
        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            for node in cell.nodes:
                if node.operation in attention_ops:
                    count += 1
        return count * self.genome.num_cells

    def _compute_effective_depth(self) -> float:
        """Compute effective depth via longest path (H3 metric)"""
        max_depth = 0

        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            # Build adjacency for DAG
            num_nodes = len(cell.nodes) + 2  # +2 for input nodes
            depths = [0, 0] + [0] * len(cell.nodes)  # Input nodes have depth 0

            for i, node in enumerate(cell.nodes):
                node_idx = i + 2
                if node.operation != 'none':
                    max_input_depth = max(depths[inp] for inp in node.inputs)
                    depths[node_idx] = max_input_depth + 1

            cell_depth = max(depths) if depths else 0
            max_depth = max(max_depth, cell_depth)

        return max_depth * self.genome.num_cells

    def _estimate_parameters(self) -> int:
        """Estimate total parameters"""
        # Rough estimation based on operations
        param_estimates = {
            'none': 0,
            'skip_connect': 0,
            'conv_1x1': 1024,
            'conv_3x3': 9216,
            'sep_conv_3x3': 3456,
            'dil_conv_3x3': 9216,
            'avg_pool_3x3': 0,
            'max_pool_3x3': 0,
            'self_attention': 4096,
            'multi_head_attn_4': 16384,
            'multi_head_attn_8': 32768,
            'gru_cell': 12288,
            'lstm_cell': 16384,
            'layer_norm': 256,
            'batch_norm': 256,
            'residual_block': 18432,
            'mlp_block': 8192,
            'recursive_block': 24576,
        }

        total = 0
        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            for node in cell.nodes:
                total += param_estimates.get(node.operation, 1000)

        return total * self.genome.num_cells * (self.genome.init_channels // 32)

    def _compute_path_variance(self) -> float:
        """Compute variance in path lengths through the network"""
        all_paths = []

        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            paths = self._enumerate_paths(cell)
            all_paths.extend(paths)

        if not all_paths:
            return 0.0

        return float(np.var(all_paths))

    def _enumerate_paths(self, cell: CellArchitecture) -> List[int]:
        """Enumerate all path lengths in a cell"""
        paths = []

        def dfs(node_idx: int, depth: int):
            if node_idx < 2:  # Input nodes
                paths.append(depth)
                return

            node = cell.nodes[node_idx - 2]
            if node.operation == 'none':
                return

            for inp in node.inputs:
                dfs(inp, depth + 1)

        # Start from output (all intermediate nodes concatenated)
        for i in range(len(cell.nodes)):
            dfs(i + 2, 0)

        return paths

    def _compute_branching_factor(self) -> float:
        """Compute average branching factor"""
        total_branches = 0
        total_nodes = 0

        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            for node in cell.nodes:
                if node.operation != 'none':
                    total_branches += len(set(node.inputs))  # Unique inputs
                    total_nodes += 1

        return total_branches / max(total_nodes, 1)

    def _compute_recursive_depth(self) -> int:
        """Compute recursive structure depth (H4 metric)"""
        recursive_ops = {'recursive_block', 'gru_cell', 'lstm_cell'}
        max_recursive_chain = 0

        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            current_chain = 0
            for node in cell.nodes:
                if node.operation in recursive_ops:
                    current_chain += 1
                    max_recursive_chain = max(max_recursive_chain, current_chain)
                else:
                    current_chain = 0

        return max_recursive_chain * self.genome.num_cells

    def _count_recurrent_ops(self) -> int:
        """Count recurrent operations"""
        recurrent_ops = {'gru_cell', 'lstm_cell', 'recursive_block'}
        count = 0
        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            for node in cell.nodes:
                if node.operation in recurrent_ops:
                    count += 1
        return count * self.genome.num_cells

    def _compute_compression_ratio(self) -> float:
        """Compute information compression ratio"""
        reduction_ops = {'avg_pool_3x3', 'max_pool_3x3', 'conv_1x1'}
        expansion_ops = {'mlp_block', 'residual_block'}

        reduction_count = 0
        expansion_count = 0

        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            for node in cell.nodes:
                if node.operation in reduction_ops:
                    reduction_count += 1
                elif node.operation in expansion_ops:
                    expansion_count += 1

        total = reduction_count + expansion_count
        if total == 0:
            return 1.0

        return reduction_count / total

    def _compute_attention_span(self) -> float:
        """Compute average attention span (distance in graph)"""
        attention_ops = {'self_attention', 'multi_head_attn_4', 'multi_head_attn_8'}
        attention_positions = []

        for cell in [self.genome.normal_cell, self.genome.reduction_cell]:
            for i, node in enumerate(cell.nodes):
                if node.operation in attention_ops:
                    attention_positions.append(i)

        if len(attention_positions) < 2:
            return 0.0

        # Average distance between attention operations
        distances = []
        for i in range(len(attention_positions) - 1):
            distances.append(attention_positions[i + 1] - attention_positions[i])

        return float(np.mean(distances)) if distances else 0.0

    def get_hypothesis_metrics(self) -> Dict[str, Any]:
        """Get metrics organized by hypothesis"""
        metrics = self.compute_all()

        return {
            'H1_skip_connections': {
                'num_skip_connections': metrics['num_skip_connections'],
                'skip_ratio': metrics['num_skip_connections'] / max(self.genome.num_nodes * 2, 1),
            },
            'H2_attention': {
                'num_attention_ops': metrics['num_attention_ops'],
                'attention_ratio': metrics['num_attention_ops'] / max(self.genome.num_nodes * 2, 1),
                'attention_span': metrics['attention_span'],
            },
            'H3_complexity': {
                'effective_depth': metrics['effective_depth'],
                'total_parameters': metrics['total_parameters'],
                'path_length_variance': metrics['path_length_variance'],
                'branching_factor': metrics['branching_factor'],
            },
            'H4_recursive': {
                'recursive_depth': metrics['recursive_depth'],
                'num_recurrent_ops': metrics['num_recurrent_ops'],
                'compression_ratio': metrics['compression_ratio'],
            },
        }


class SearchSpaceFactory:
    """Factory for creating different search space configurations"""

    @staticmethod
    def full_space() -> Dict:
        """Full search space with all operations"""
        return OPERATION_SPACE.copy()

    @staticmethod
    def no_skip_space() -> Dict:
        """Ablation: Search space without skip connections"""
        space = OPERATION_SPACE.copy()
        space['primitives'] = [op for op in space['primitives']
                              if 'skip' not in op and 'residual' not in op]
        return space

    @staticmethod
    def no_attention_space() -> Dict:
        """Ablation: Search space without attention mechanisms"""
        space = OPERATION_SPACE.copy()
        space['primitives'] = [op for op in space['primitives']
                              if 'attention' not in op and 'attn' not in op]
        return space

    @staticmethod
    def no_recurrence_space() -> Dict:
        """Ablation: Search space without recurrent operations"""
        space = OPERATION_SPACE.copy()
        space['primitives'] = [op for op in space['primitives']
                              if op not in {'gru_cell', 'lstm_cell', 'recursive_block'}]
        return space

    @staticmethod
    def minimal_space() -> Dict:
        """Minimal search space for fast experiments"""
        return {
            'primitives': [
                'none',
                'skip_connect',
                'conv_3x3',
                'self_attention',
                'gru_cell',
            ],
            'cell_nodes': 2,
            'cell_concat': 2,
            'num_cells': [2, 4],
            'channels': [16, 32],
            'reduction_cells': [1],
        }

    @staticmethod
    def conv_only_space() -> Dict:
        """Convolution-only space (control baseline)"""
        return {
            'primitives': [
                'none',
                'conv_1x1',
                'conv_3x3',
                'sep_conv_3x3',
                'dil_conv_3x3',
                'avg_pool_3x3',
                'max_pool_3x3',
            ],
            'cell_nodes': 4,
            'cell_concat': 4,
            'num_cells': [4, 6, 8],
            'channels': [16, 32, 64],
            'reduction_cells': [2, 4],
        }


def create_random_genome(
    num_nodes: int = 4,
    num_cells: int = 6,
    init_channels: int = 32,
    operation_space: Optional[Dict] = None,
    seed: Optional[int] = None
) -> ArchitectureGenome:
    """Create a random architecture genome"""
    if seed is not None:
        np.random.seed(seed)

    space = operation_space or OPERATION_SPACE
    primitives = space['primitives']

    genome = ArchitectureGenome(
        num_nodes=num_nodes,
        num_ops=len(primitives),
        num_cells=num_cells,
        init_channels=init_channels,
    )

    # Override with random cells using specified operation space
    for cell_attr in ['normal_cell', 'reduction_cell']:
        nodes = []
        for i in range(num_nodes):
            max_input = i + 2
            input_1 = np.random.randint(0, max_input)
            input_2 = np.random.randint(0, max_input)
            op = np.random.choice(primitives)
            nodes.append(CellNode(operation=op, inputs=[input_1, input_2]))

        cell_type = 'normal' if cell_attr == 'normal_cell' else 'reduction'
        setattr(genome, cell_attr, CellArchitecture(nodes=nodes, cell_type=cell_type))

    return genome


def mutate_genome(
    genome: ArchitectureGenome,
    mutation_rate: float = 0.1,
    operation_space: Optional[Dict] = None
) -> ArchitectureGenome:
    """Mutate an architecture genome"""
    space = operation_space or OPERATION_SPACE
    primitives = space['primitives']

    new_genome = ArchitectureGenome(
        num_nodes=genome.num_nodes,
        num_ops=len(primitives),
        num_cells=genome.num_cells,
        init_channels=genome.init_channels,
    )

    for cell_attr in ['normal_cell', 'reduction_cell']:
        old_cell = getattr(genome, cell_attr)
        new_nodes = []

        for i, node in enumerate(old_cell.nodes):
            if np.random.random() < mutation_rate:
                # Mutate operation
                new_op = np.random.choice(primitives)
            else:
                new_op = node.operation

            new_inputs = list(node.inputs)
            for j in range(len(new_inputs)):
                if np.random.random() < mutation_rate:
                    max_input = i + 2
                    new_inputs[j] = np.random.randint(0, max_input)

            new_nodes.append(CellNode(operation=new_op, inputs=new_inputs))

        cell_type = old_cell.cell_type
        setattr(new_genome, cell_attr, CellArchitecture(nodes=new_nodes, cell_type=cell_type))

    # Mutate macro parameters
    if np.random.random() < mutation_rate:
        new_genome.num_cells = np.random.choice(space.get('num_cells', [4, 6, 8]))
    else:
        new_genome.num_cells = genome.num_cells

    if np.random.random() < mutation_rate:
        new_genome.init_channels = np.random.choice(space.get('channels', [16, 32, 64]))
    else:
        new_genome.init_channels = genome.init_channels

    return new_genome


def crossover_genomes(
    parent1: ArchitectureGenome,
    parent2: ArchitectureGenome,
    operation_space: Optional[Dict] = None
) -> Tuple[ArchitectureGenome, ArchitectureGenome]:
    """Crossover two architecture genomes"""
    space = operation_space or OPERATION_SPACE
    primitives = space['primitives']

    child1 = ArchitectureGenome(
        num_nodes=parent1.num_nodes,
        num_ops=len(primitives),
        num_cells=parent1.num_cells if np.random.random() < 0.5 else parent2.num_cells,
        init_channels=parent1.init_channels if np.random.random() < 0.5 else parent2.init_channels,
    )

    child2 = ArchitectureGenome(
        num_nodes=parent1.num_nodes,
        num_ops=len(primitives),
        num_cells=parent2.num_cells if np.random.random() < 0.5 else parent1.num_cells,
        init_channels=parent2.init_channels if np.random.random() < 0.5 else parent1.init_channels,
    )

    for cell_attr in ['normal_cell', 'reduction_cell']:
        cell1 = getattr(parent1, cell_attr)
        cell2 = getattr(parent2, cell_attr)

        # Single-point crossover
        crossover_point = np.random.randint(0, parent1.num_nodes + 1)

        nodes_c1 = []
        nodes_c2 = []

        for i in range(parent1.num_nodes):
            if i < crossover_point:
                nodes_c1.append(copy.deepcopy(cell1.nodes[i]))
                nodes_c2.append(copy.deepcopy(cell2.nodes[i]))
            else:
                nodes_c1.append(copy.deepcopy(cell2.nodes[i]))
                nodes_c2.append(copy.deepcopy(cell1.nodes[i]))

        cell_type = cell1.cell_type
        setattr(child1, cell_attr, CellArchitecture(nodes=nodes_c1, cell_type=cell_type))
        setattr(child2, cell_attr, CellArchitecture(nodes=nodes_c2, cell_type=cell_type))

    return child1, child2


def genome_distance(genome1: ArchitectureGenome, genome2: ArchitectureGenome) -> float:
    """Compute distance between two genomes for speciation"""
    distance = 0.0
    total_comparisons = 0

    for cell_attr in ['normal_cell', 'reduction_cell']:
        cell1 = getattr(genome1, cell_attr)
        cell2 = getattr(genome2, cell_attr)

        for node1, node2 in zip(cell1.nodes, cell2.nodes):
            # Operation difference
            if node1.operation != node2.operation:
                distance += 1.0

            # Input difference
            for inp1, inp2 in zip(node1.inputs, node2.inputs):
                if inp1 != inp2:
                    distance += 0.5

            total_comparisons += 1 + len(node1.inputs)

    # Macro parameter differences
    if genome1.num_cells != genome2.num_cells:
        distance += 0.5
    if genome1.init_channels != genome2.init_channels:
        distance += 0.5
    total_comparisons += 2

    return distance / max(total_comparisons, 1)
