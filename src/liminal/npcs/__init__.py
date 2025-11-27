"""
NPC System for Liminal Architectures

This module provides:
- BaseNPC: Core NPC class with Soul Map, behaviors, and states
- Hero NPCs: Hand-crafted characters (Arthur, Agnes, Victoria, etc.)
- Archetypes: Templates for procedural NPC generation
"""

from .base_npc import BaseNPC, NPCState, NPCBehavior
from .heroes import HERO_NPCS, create_hero_npc
from .archetypes import ARCHETYPES, create_archetype_npc, populate_realm

__all__ = [
    'BaseNPC',
    'NPCState',
    'NPCBehavior',
    'HERO_NPCS',
    'create_hero_npc',
    'ARCHETYPES',
    'create_archetype_npc',
    'populate_realm',
]
