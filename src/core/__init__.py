"""
Core module for ToM-NAS.

Contains the foundational components:
- SoulMapOntology: 181-dimensional psychological state space
- RecursiveBeliefState: 5th-order belief modeling
- BeliefNetwork: Multi-agent belief management
- Events: Information asymmetry tracking for ToM scenarios
"""

from .beliefs import Belief, BeliefNetwork, RecursiveBeliefState
from .events import (
    AgentBeliefState,
    Event,
    EventType,
    InformationAsymmetryTracker,
    WorldState,
    create_sally_anne_scenario,
    verify_information_asymmetry,
)
from .ontology import OntologyDimension, SoulMapOntology

__all__ = [
    "SoulMapOntology",
    "OntologyDimension",
    "Belief",
    "RecursiveBeliefState",
    "BeliefNetwork",
    "Event",
    "EventType",
    "AgentBeliefState",
    "WorldState",
    "InformationAsymmetryTracker",
    "create_sally_anne_scenario",
    "verify_information_asymmetry",
]
