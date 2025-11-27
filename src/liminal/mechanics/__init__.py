"""
Game Mechanics for Liminal Architectures

This module implements the core gameplay systems:
- Soul Scanner: The HUD overlay for analyzing NPC Soul Maps
- Cognitive Hazards: Psychological interventions the player can use
- Ontological Instability: The "wanted" system based on reality disruption
"""

from .soul_scanner import SoulScanner, AnalysisResult
from .cognitive_hazards import CognitiveHazard, HAZARD_REGISTRY, apply_hazard
from .ontological_instability import OntologicalInstability, InstabilityLevel

__all__ = [
    'SoulScanner',
    'AnalysisResult',
    'CognitiveHazard',
    'HAZARD_REGISTRY',
    'apply_hazard',
    'OntologicalInstability',
    'InstabilityLevel',
]
