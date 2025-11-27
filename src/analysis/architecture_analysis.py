"""
Architecture Analysis Tools for ToM-NAS

Provides:
1. Architecture similarity computation
2. Graph-based architecture analysis
3. Architecture family clustering
4. Pattern detection in discovered architectures
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Optional imports
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def architecture_to_graph(architecture: Dict[str, Any]) -> Optional['nx.DiGraph']:
    """
    Convert architecture dictionary to NetworkX graph.

    Args:
        architecture: Architecture dict with 'normal_cell' and/or 'reduction_cell'

    Returns:
        NetworkX DiGraph representing the architecture
    """
    if not HAS_NETWORKX:
        print("Warning: NetworkX not available")
        return None

    G = nx.DiGraph()

    # Process each cell type
    for cell_type in ['normal_cell', 'reduction_cell']:
        cell = architecture.get(cell_type, {})
        nodes = cell.get('nodes', [])

        if not nodes:
            continue

        prefix = 'n_' if cell_type == 'normal_cell' else 'r_'

        # Add input nodes
        G.add_node(f'{prefix}input_0', cell=cell_type, type='input')
        G.add_node(f'{prefix}input_1', cell=cell_type, type='input')

        # Add intermediate nodes
        for i, node in enumerate(nodes):
            if isinstance(node, dict):
                op = node.get('operation', 'unknown')
                inputs = node.get('inputs', [0, 1])
            else:
                op = str(node)
                inputs = [0, 1]

            node_id = f'{prefix}node_{i}'
            G.add_node(node_id, cell=cell_type, type='intermediate', operation=op)

            # Add edges
            for inp_idx in inputs:
                if inp_idx == 0:
                    source = f'{prefix}input_0'
                elif inp_idx == 1:
                    source = f'{prefix}input_1'
                else:
                    source = f'{prefix}node_{inp_idx - 2}'

                if source in G.nodes:
                    G.add_edge(source, node_id)

        # Add output node
        output_id = f'{prefix}output'
        G.add_node(output_id, cell=cell_type, type='output')
        for i in range(len(nodes)):
            G.add_edge(f'{prefix}node_{i}', output_id)

    return G


def compute_architecture_similarity(
    arch1: Dict[str, Any],
    arch2: Dict[str, Any],
    method: str = 'structural',
) -> float:
    """
    Compute similarity between two architectures.

    Args:
        arch1: First architecture
        arch2: Second architecture
        method: Similarity method ('structural', 'operation', 'graph_edit')

    Returns:
        Similarity score in [0, 1]
    """
    if method == 'structural':
        return _structural_similarity(arch1, arch2)
    elif method == 'operation':
        return _operation_similarity(arch1, arch2)
    elif method == 'graph_edit':
        return _graph_edit_similarity(arch1, arch2)
    else:
        return _structural_similarity(arch1, arch2)


def _structural_similarity(arch1: Dict, arch2: Dict) -> float:
    """Compute structural similarity based on architecture features"""
    features1 = _extract_features(arch1)
    features2 = _extract_features(arch2)

    # Compute cosine similarity
    dot = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


def _operation_similarity(arch1: Dict, arch2: Dict) -> float:
    """Compute similarity based on operation overlap"""
    ops1 = _get_all_operations(arch1)
    ops2 = _get_all_operations(arch2)

    if not ops1 or not ops2:
        return 0.0

    # Jaccard similarity
    intersection = len(set(ops1) & set(ops2))
    union = len(set(ops1) | set(ops2))

    return intersection / union if union > 0 else 0.0


def _graph_edit_similarity(arch1: Dict, arch2: Dict) -> float:
    """Compute similarity using graph edit distance"""
    if not HAS_NETWORKX:
        return _structural_similarity(arch1, arch2)

    G1 = architecture_to_graph(arch1)
    G2 = architecture_to_graph(arch2)

    if G1 is None or G2 is None:
        return 0.0

    # Use approximation for graph edit distance
    try:
        # Quick approximation: compare node/edge counts
        n1, e1 = G1.number_of_nodes(), G1.number_of_edges()
        n2, e2 = G2.number_of_nodes(), G2.number_of_edges()

        size_diff = abs(n1 - n2) + abs(e1 - e2)
        max_size = max(n1 + e1, n2 + e2)

        return 1.0 - (size_diff / max_size) if max_size > 0 else 0.0

    except Exception:
        return 0.0


def _extract_features(architecture: Dict) -> np.ndarray:
    """Extract feature vector from architecture"""
    features = []

    for cell_type in ['normal_cell', 'reduction_cell']:
        cell = architecture.get(cell_type, {})
        nodes = cell.get('nodes', [])

        # Operation counts
        op_counts = {
            'skip': 0, 'conv': 0, 'pool': 0, 'attn': 0, 'recurrent': 0, 'none': 0
        }

        for node in nodes:
            if isinstance(node, dict):
                op = node.get('operation', '')
            else:
                op = str(node)

            if 'skip' in op:
                op_counts['skip'] += 1
            elif 'conv' in op:
                op_counts['conv'] += 1
            elif 'pool' in op:
                op_counts['pool'] += 1
            elif 'attn' in op or 'attention' in op:
                op_counts['attn'] += 1
            elif 'gru' in op or 'lstm' in op or 'recursive' in op:
                op_counts['recurrent'] += 1
            elif op == 'none':
                op_counts['none'] += 1

        features.extend([
            op_counts['skip'],
            op_counts['conv'],
            op_counts['pool'],
            op_counts['attn'],
            op_counts['recurrent'],
            op_counts['none'],
            len(nodes),  # Node count
        ])

    # Add macro features
    features.append(architecture.get('num_cells', 6))
    features.append(architecture.get('init_channels', 32) / 32)

    return np.array(features, dtype=np.float32)


def _get_all_operations(architecture: Dict) -> List[str]:
    """Get list of all operations in architecture"""
    operations = []

    for cell_type in ['normal_cell', 'reduction_cell']:
        cell = architecture.get(cell_type, {})
        nodes = cell.get('nodes', [])

        for node in nodes:
            if isinstance(node, dict):
                operations.append(node.get('operation', 'unknown'))
            else:
                operations.append(str(node))

    return operations


def analyze_architecture_families(
    architectures: List[Dict[str, Any]],
    n_clusters: int = 4,
) -> Dict[str, Any]:
    """
    Cluster architectures to identify common patterns/families.

    Args:
        architectures: List of architecture dictionaries
        n_clusters: Number of clusters to find

    Returns:
        Cluster analysis results
    """
    if not architectures:
        return {'error': 'No architectures to analyze'}

    # Extract features
    features = np.array([_extract_features(arch) for arch in architectures])

    # Normalize
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + 1e-8
    features_norm = (features - feature_mean) / feature_std

    if HAS_SKLEARN:
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_norm)

        # Dimensionality reduction for visualization
        if features_norm.shape[0] > 2:
            if features_norm.shape[0] >= n_clusters * 2:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features_norm.shape[0] - 1))
                features_2d = tsne.fit_transform(features_norm)
            else:
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(features_norm)
        else:
            features_2d = features_norm

        # Analyze clusters
        cluster_analysis = {}
        for c in range(n_clusters):
            mask = clusters == c
            cluster_features = features[mask]

            if len(cluster_features) > 0:
                cluster_analysis[f'cluster_{c}'] = {
                    'size': int(mask.sum()),
                    'avg_skip_connections': float(cluster_features[:, 0].mean()),
                    'avg_attention_ops': float(cluster_features[:, 3].mean()),
                    'avg_recurrent_ops': float(cluster_features[:, 4].mean()),
                    'avg_node_count': float(cluster_features[:, 6].mean()),
                }

        return {
            'n_clusters': n_clusters,
            'cluster_labels': clusters.tolist(),
            'cluster_analysis': cluster_analysis,
            'features_2d': features_2d.tolist(),
            'feature_names': [
                'normal_skip', 'normal_conv', 'normal_pool', 'normal_attn', 'normal_recurrent', 'normal_none', 'normal_nodes',
                'reduce_skip', 'reduce_conv', 'reduce_pool', 'reduce_attn', 'reduce_recurrent', 'reduce_none', 'reduce_nodes',
                'num_cells', 'init_channels'
            ],
        }

    else:
        # Simple clustering without sklearn
        return _simple_clustering(architectures, features, n_clusters)


def _simple_clustering(
    architectures: List[Dict],
    features: np.ndarray,
    n_clusters: int,
) -> Dict[str, Any]:
    """Simple k-means without sklearn"""
    n = len(features)
    if n < n_clusters:
        n_clusters = n

    # Random initialization
    np.random.seed(42)
    centroid_idx = np.random.choice(n, n_clusters, replace=False)
    centroids = features[centroid_idx].copy()

    # Iterate
    for _ in range(100):
        # Assign clusters
        distances = np.array([
            [np.linalg.norm(f - c) for c in centroids]
            for f in features
        ])
        clusters = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = []
        for c in range(n_clusters):
            mask = clusters == c
            if mask.sum() > 0:
                new_centroids.append(features[mask].mean(axis=0))
            else:
                new_centroids.append(centroids[c])
        centroids = np.array(new_centroids)

    # Analyze
    cluster_analysis = {}
    for c in range(n_clusters):
        mask = clusters == c
        if mask.sum() > 0:
            cluster_analysis[f'cluster_{c}'] = {
                'size': int(mask.sum()),
                'avg_features': features[mask].mean(axis=0).tolist(),
            }

    return {
        'n_clusters': n_clusters,
        'cluster_labels': clusters.tolist(),
        'cluster_analysis': cluster_analysis,
    }


def cluster_architectures(
    architectures: List[Dict[str, Any]],
    labels: Optional[List[Any]] = None,
    method: str = 'kmeans',
    n_clusters: int = 4,
) -> Dict[str, Any]:
    """
    Cluster architectures and analyze cluster composition.

    Args:
        architectures: List of architectures
        labels: Optional labels (e.g., task type, ToM order)
        method: Clustering method ('kmeans', 'dbscan')
        n_clusters: Number of clusters (for kmeans)

    Returns:
        Clustering results with analysis
    """
    return analyze_architecture_families(architectures, n_clusters)


def detect_patterns(architectures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect common patterns in discovered architectures.

    Returns:
        Pattern analysis with frequencies and examples
    """
    patterns = {
        'skip_heavy': [],  # Architectures with many skip connections
        'attention_heavy': [],  # Architectures with many attention ops
        'recurrent_heavy': [],  # Architectures with recurrent structures
        'sparse': [],  # Architectures with many 'none' operations
        'deep': [],  # Architectures with high effective depth
    }

    for i, arch in enumerate(architectures):
        features = _extract_features(arch)

        # Count operations (indices from feature extraction)
        normal_skip = features[0]
        normal_attn = features[3]
        normal_recurrent = features[4]
        normal_none = features[5]
        node_count = features[6] + features[13]  # normal + reduction

        if normal_skip >= 2:
            patterns['skip_heavy'].append(i)
        if normal_attn >= 2:
            patterns['attention_heavy'].append(i)
        if normal_recurrent >= 2:
            patterns['recurrent_heavy'].append(i)
        if normal_none >= node_count / 2:
            patterns['sparse'].append(i)

    # Calculate pattern frequencies
    n = len(architectures)
    frequencies = {
        pattern: len(indices) / n if n > 0 else 0
        for pattern, indices in patterns.items()
    }

    return {
        'pattern_indices': patterns,
        'frequencies': frequencies,
        'total_architectures': n,
    }


def compare_task_architectures(
    results_by_task: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Compare architectural patterns between different task types.

    Args:
        results_by_task: Dictionary mapping task names to architecture results

    Returns:
        Comparison analysis
    """
    task_features = {}

    for task_name, results in results_by_task.items():
        architectures = [r.get('best_genome', r) for r in results if r]
        if not architectures:
            continue

        features = np.array([_extract_features(arch) for arch in architectures])
        task_features[task_name] = {
            'mean_features': features.mean(axis=0).tolist(),
            'std_features': features.std(axis=0).tolist(),
            'n_architectures': len(architectures),
        }

    # Compare ToM vs Control
    tom_tasks = ['tomi', 'bigtom', 'hitom_2', 'hitom_4', 'opentom', 'socialqa']
    control_tasks = ['simple_sequence', 'babi_1', 'babi_2', 'relational']

    tom_features = []
    control_features = []

    for task_name, data in task_features.items():
        features = data['mean_features']
        if any(t in task_name.lower() for t in tom_tasks):
            tom_features.append(features)
        elif any(t in task_name.lower() for t in control_tasks):
            control_features.append(features)

    comparison = {}
    if tom_features and control_features:
        tom_mean = np.mean(tom_features, axis=0)
        control_mean = np.mean(control_features, axis=0)

        feature_names = [
            'skip_normal', 'conv_normal', 'pool_normal', 'attn_normal', 'recurrent_normal', 'none_normal', 'nodes_normal',
            'skip_reduce', 'conv_reduce', 'pool_reduce', 'attn_reduce', 'recurrent_reduce', 'none_reduce', 'nodes_reduce',
            'num_cells', 'init_channels'
        ]

        for i, name in enumerate(feature_names):
            comparison[name] = {
                'tom_mean': float(tom_mean[i]),
                'control_mean': float(control_mean[i]),
                'difference': float(tom_mean[i] - control_mean[i]),
            }

    return {
        'task_features': task_features,
        'tom_vs_control': comparison,
    }
