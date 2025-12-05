"""
Social world simulation module for ToM-NAS.

Contains the social simulation environment:
- SocialWorld4: Multi-agent society with 4 game types
- ZombieGame: ToM validation mechanism
- Agent: Agent data structure with resources, energy, and reputation
"""

from .social_world import Agent, SocialWorld4, ZombieGame

__all__ = [
    "Agent",
    "ZombieGame",
    "SocialWorld4",
]
