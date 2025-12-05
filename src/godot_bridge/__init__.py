"""
Godot Physical Bridge - Symbol Grounding Layer

This module bridges the gap between the physical simulation (Godot 4.x)
and the semiotic knowledge graph, implementing Harnad's Symbol Grounding.

The Physical is Cognitive:
- Physical objects in Godot are not just geometry - they are semantic nodes
- Every perception triggers graph traversal through the taxonomies
- The bridge translates physics into meaning

Key Components:
- GodotBridge: WebSocket server for bidirectional communication
- SymbolGrounder: Maps Godot entities to semantic nodes
- PerceptionProcessor: Processes sensory input into CognitiveBlocks
- ActionExecutor: Translates intents into Godot commands

Theoretical Foundation:
- Symbol Grounding Problem (Harnad, 1990)
- Embodied Cognition (Varela, Thompson, Rosch)
- Sensorimotor Contingency Theory (O'Regan & Noë)

Author: ToM-NAS Project
"""

from .bridge import GodotBridge, BridgeConfig, ConnectionState
from .symbol_grounding import SymbolGrounder, GroundedSymbol, GroundingContext
from .perception import PerceptionProcessor, SensoryInput, PerceptualField
from .action import ActionExecutor, GodotAction, ActionResult, ActionType
from .protocol import (
    GodotMessage,
    EntityUpdate,
    AgentCommand,
    WorldState,
    MessageType,
    Vector3,
)

__all__ = [
    # Bridge
    "GodotBridge",
    "BridgeConfig",
    "ConnectionState",
    # Symbol Grounding
    "SymbolGrounder",
    "GroundedSymbol",
    "GroundingContext",
    # Perception
    "PerceptionProcessor",
    "SensoryInput",
    "PerceptualField",
    # Action
    "ActionExecutor",
    "GodotAction",
    "ActionResult",
    "ActionType",
    # Protocol
    "GodotMessage",
    "EntityUpdate",
    "AgentCommand",
    "WorldState",
    "MessageType",
    "Vector3",
]
