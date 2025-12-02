"""
Social world simulation module for ToM-NAS.

Contains the social simulation environment:
- SocialWorld4: Multi-agent society with 4 game types
- ZombieGame: ToM validation mechanism
- Agent: Agent data structure with resources, energy, and reputation
- WorldEvent: Event with observation tracking for information asymmetry (NEW)
"""

from .social_world import Agent, ZombieGame, SocialWorld4, WorldEvent

__all__ = [
    'Agent',
    'ZombieGame',
    'SocialWorld4',
    'WorldEvent',
]
