"""
Cognitive Core - Mentalese and Recursive Self-Compression

This module implements the computational heart of the Fractal Semiotic Engine:
1. Mentalese: TypeScript-style typed cognitive blocks as fundamental atomic units
2. Recursive Self-Compression (RSC): Agents running nested simulations of peers
3. Tiny Recursive Model (TRM): Neural transition function for cognitive blocks

Theoretical Foundation:
- Language of Thought (Fodor): Mental representations have compositional structure
- Simulation Theory of Mind: Understanding others by simulating their cognition
- Predictive Processing: The brain as a prediction machine

Key Insight:
Agents do not just have "state" - they possess a Virtual Machine capable of
spinning up sandboxed sub-simulations. A CognitiveBlock begins as a Percept,
shifts to Hypothesis, to Belief, to Memory archetype. This type-shifting IS
the reasoning process, traceable and transparent.

Author: ToM-NAS Project
"""

from .mentalese import (
    # Core cognitive block types
    CognitiveBlock,
    PerceptBlock,
    HypothesisBlock,
    BeliefBlock,
    IntentBlock,
    MemoryBlock,
    # Composite structures
    RecursiveBelief,
    SimulationState,
    # Block operations
    BlockTransition,
    compress_to_memory,
    expand_from_memory,
)
from .recursive_simulation import (
    RecursiveSimulationNode,
    SimulationConfig,
    SimulationResult,
    AgentModel,
    WorldModel,
)
from .trm import (
    TinyRecursiveModel,
    TRMConfig,
    CognitiveTransition,
)

__all__ = [
    # Mentalese types
    "CognitiveBlock",
    "PerceptBlock",
    "HypothesisBlock",
    "BeliefBlock",
    "IntentBlock",
    "MemoryBlock",
    "RecursiveBelief",
    "SimulationState",
    "BlockTransition",
    "compress_to_memory",
    "expand_from_memory",
    # Simulation
    "RecursiveSimulationNode",
    "SimulationConfig",
    "SimulationResult",
    "AgentModel",
    "WorldModel",
    # TRM
    "TinyRecursiveModel",
    "TRMConfig",
    "CognitiveTransition",
]
