"""
Simulation Module - Fractal Simulation Engine

This module implements the fractal container structure for recursive
Theory of Mind simulations. Key components:

- SimulationNode: The fundamental container for nested simulations
- RootSimulationNode: Special root node connected to Godot
- RSCAgent: Abstract base class for Recursive Self-Compression agents

The simulation is a TREE, not a singleton. Each node can contain agents
running their own nested simulations to predict futures.
"""

from .fractal_node import (
    SimulationStatus,
    SimulationConfig,
    WorldStateVector,
    RSCAgent,
    NeuralRSCAgent,
    RuleBasedRSCAgent,
    SimulationNode,
    RootSimulationNode,
    create_simulation,
)

__all__ = [
    'SimulationStatus',
    'SimulationConfig',
    'WorldStateVector',
    'RSCAgent',
    'NeuralRSCAgent',
    'RuleBasedRSCAgent',
    'SimulationNode',
    'RootSimulationNode',
    'create_simulation',
]
