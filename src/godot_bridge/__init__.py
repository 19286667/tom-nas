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

from .action import ActionExecutor, ActionResult, GodotAction
from .bridge import BridgeConfig, ConnectionState, GodotBridge
from .perception import PerceptionProcessor, PerceptualField, SensoryInput
from .protocol import (
    AgentCommand,
    EntityUpdate,
    GodotMessage,
    MessageType,
    WorldState,
)
from .symbol_grounding import GroundedSymbol, GroundingContext, SymbolGrounder

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
    # Protocol
    "GodotMessage",
    "EntityUpdate",
    "AgentCommand",
    "WorldState",
    "MessageType",
]
