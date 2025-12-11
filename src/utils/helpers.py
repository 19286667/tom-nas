"""
ToM-NAS Helper Functions
Shared utility functions for observation conversion and model creation
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


def observation_to_tensor(obs: Dict, input_dim: int = 191, max_neighbors: int = 5) -> torch.Tensor:
    """
    Convert observation dict to tensor input.
    
    Args:
        obs: Observation dictionary from SocialWorld containing:
            - own_resources: Agent's current resources
            - own_energy: Agent's current energy
            - own_coalition: Agent's coalition ID or None
            - observations: List of observations of other agents
        input_dim: Target dimension for output tensor (default: 191)
        max_neighbors: Maximum number of neighbor observations to include (default: 5)
        
    Returns:
        Tensor of shape (input_dim,) with normalized features
    """
    features = [
        obs['own_resources'] / 200.0,
        obs['own_energy'] / 100.0,
        float(obs['own_coalition'] is not None)
    ]

    # Add features from other agents
    for other_obs in obs['observations'][:max_neighbors]:
        features.extend([
            other_obs['estimated_resources'] / 200.0,
            other_obs['estimated_energy'] / 100.0,
            other_obs['reputation'],
            float(other_obs['in_same_coalition'])
        ])

    # Pad to fixed size
    while len(features) < input_dim:
        features.append(0.0)
    features = features[:input_dim]

    return torch.tensor(features, dtype=torch.float32)


def create_model(
    arch_type: str,
    input_dim: int = 191,
    hidden_dim: int = 128,
    output_dim: int = 181,
    num_layers: int = 2,
    num_heads: int = 4,
    max_recursion: int = 5,
    device: Optional[str] = None
) -> nn.Module:
    """
    Create a ToM agent model based on architecture type.
    
    Args:
        arch_type: Architecture type ('TRN', 'RSAN', 'Transformer', or 'Hybrid')
        input_dim: Input dimension (default: 191)
        hidden_dim: Hidden layer dimension (default: 128)
        output_dim: Output dimension (default: 181)
        num_layers: Number of layers for TRN/Transformer (default: 2)
        num_heads: Number of attention heads for RSAN/Transformer (default: 4)
        max_recursion: Maximum recursion depth for RSAN (default: 5)
        device: Device to place model on (default: None, stays on CPU)
        
    Returns:
        Initialized nn.Module of the specified architecture
        
    Raises:
        ValueError: If arch_type is not recognized
        
    Note:
        For 'Hybrid' architecture, this creates a RecursiveSelfAttention as the base.
        The NAS engine has special handling for full HybridArchitecture with gene dicts.
    """
    # Import here to avoid circular imports
    from ..agents.architectures import (
        TransparentRNN, RecursiveSelfAttention, TransformerToMAgent
    )
    
    if arch_type == 'TRN':
        model = TransparentRNN(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    elif arch_type == 'RSAN':
        model = RecursiveSelfAttention(
            input_dim, hidden_dim, output_dim,
            num_heads=num_heads, max_recursion=max_recursion
        )
    elif arch_type == 'Transformer':
        model = TransformerToMAgent(
            input_dim, hidden_dim, output_dim,
            num_layers=num_layers, num_heads=num_heads
        )
    elif arch_type == 'Hybrid':
        # Hybrid uses RSAN as base (best for recursive beliefs)
        # For full HybridArchitecture with gene dicts, use NAS engine's _gene_to_model
        model = RecursiveSelfAttention(
            input_dim, hidden_dim, output_dim,
            num_heads=num_heads, max_recursion=max_recursion
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")
    
    if device is not None:
        model = model.to(device)
    
    return model
