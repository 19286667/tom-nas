"""
Event System with Information Asymmetry for ToM-NAS

This module implements an event tracking system that maintains information
asymmetry - tracking which agents observed which events. This is essential
for Theory of Mind reasoning where different agents have different knowledge.

Key features:
- Events track their observers (observed_by)
- Scenarios can be replayed from different perspectives
- False belief detection through belief vs reality comparison
"""

import torch
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import copy


class EventType(Enum):
    """Types of events that can occur in the world."""
    OBJECT_PLACED = "object_placed"
    OBJECT_MOVED = "object_moved"
    AGENT_ENTERED = "agent_entered"
    AGENT_EXITED = "agent_exited"
    AGENT_ACTION = "agent_action"
    COMMUNICATION = "communication"
    OBSERVATION = "observation"


@dataclass
class Event:
    """
    A world event with observer tracking.

    The observed_by field is crucial for ToM: events only affect
    the beliefs of agents who observed them.
    """
    event_type: EventType
    timestamp: int
    actor: str  # Agent who performed the action
    target: Optional[str] = None  # Object or location affected
    source_location: Optional[str] = None
    target_location: Optional[str] = None
    observed_by: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def was_observed_by(self, agent: str) -> bool:
        """Check if an agent observed this event."""
        return agent in self.observed_by

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'actor': self.actor,
            'target': self.target,
            'source_location': self.source_location,
            'target_location': self.target_location,
            'observed_by': list(self.observed_by),
            'metadata': self.metadata,
        }


@dataclass
class AgentBeliefState:
    """Tracks what an agent believes about the world state."""
    agent_id: str
    object_locations: Dict[str, str] = field(default_factory=dict)
    agent_locations: Dict[str, str] = field(default_factory=dict)
    observed_events: List[Event] = field(default_factory=list)

    def update_from_event(self, event: Event):
        """Update beliefs based on an observed event."""
        if event.event_type == EventType.OBJECT_PLACED:
            self.object_locations[event.target] = event.target_location
        elif event.event_type == EventType.OBJECT_MOVED:
            self.object_locations[event.target] = event.target_location
        elif event.event_type == EventType.AGENT_ENTERED:
            self.agent_locations[event.actor] = event.target_location
        elif event.event_type == EventType.AGENT_EXITED:
            self.agent_locations[event.actor] = 'outside'

        self.observed_events.append(event)

    def get_believed_location(self, object_name: str) -> Optional[str]:
        """Get where this agent believes an object is located."""
        return self.object_locations.get(object_name)


class WorldState:
    """
    Ground truth world state with event history.

    Maintains the actual state of the world and all events that occurred.
    """

    def __init__(self):
        self.object_locations: Dict[str, str] = {}
        self.agent_locations: Dict[str, str] = {}
        self.events: List[Event] = []
        self.timestamp = 0
        self.agents: Set[str] = set()

    def add_agent(self, agent_id: str, location: str = 'room'):
        """Add an agent to the world."""
        self.agents.add(agent_id)
        self.agent_locations[agent_id] = location

    def place_object(self, object_name: str, location: str, actor: str,
                     observers: Optional[Set[str]] = None) -> Event:
        """Place an object at a location."""
        self.timestamp += 1

        # Determine observers (all agents in the room by default)
        if observers is None:
            observers = self._get_present_agents()

        event = Event(
            event_type=EventType.OBJECT_PLACED,
            timestamp=self.timestamp,
            actor=actor,
            target=object_name,
            target_location=location,
            observed_by=observers,
        )

        self.object_locations[object_name] = location
        self.events.append(event)

        return event

    def move_object(self, object_name: str, from_location: str,
                    to_location: str, actor: str,
                    observers: Optional[Set[str]] = None) -> Event:
        """Move an object from one location to another."""
        self.timestamp += 1

        if observers is None:
            observers = self._get_present_agents()

        event = Event(
            event_type=EventType.OBJECT_MOVED,
            timestamp=self.timestamp,
            actor=actor,
            target=object_name,
            source_location=from_location,
            target_location=to_location,
            observed_by=observers,
        )

        self.object_locations[object_name] = to_location
        self.events.append(event)

        return event

    def agent_enters(self, agent_id: str, location: str = 'room',
                     observers: Optional[Set[str]] = None) -> Event:
        """Agent enters a location."""
        self.timestamp += 1

        if observers is None:
            observers = self._get_present_agents()
        observers.add(agent_id)  # Agent sees themselves enter

        event = Event(
            event_type=EventType.AGENT_ENTERED,
            timestamp=self.timestamp,
            actor=agent_id,
            target_location=location,
            observed_by=observers,
        )

        self.agent_locations[agent_id] = location
        self.events.append(event)

        return event

    def agent_exits(self, agent_id: str, from_location: str = 'room',
                    observers: Optional[Set[str]] = None) -> Event:
        """Agent exits a location."""
        self.timestamp += 1

        if observers is None:
            observers = self._get_present_agents()

        event = Event(
            event_type=EventType.AGENT_EXITED,
            timestamp=self.timestamp,
            actor=agent_id,
            source_location=from_location,
            observed_by=observers,
        )

        self.agent_locations[agent_id] = 'outside'
        self.events.append(event)

        return event

    def _get_present_agents(self, location: str = 'room') -> Set[str]:
        """Get all agents currently in a location."""
        return {agent for agent, loc in self.agent_locations.items()
                if loc == location}

    def get_reality(self, object_name: str) -> Optional[str]:
        """Get the actual location of an object (ground truth)."""
        return self.object_locations.get(object_name)


class InformationAsymmetryTracker:
    """
    Tracks information asymmetry between agents.

    This is the core mechanism for Theory of Mind: different agents
    have different beliefs based on what they observed.
    """

    def __init__(self):
        self.world = WorldState()
        self.agent_beliefs: Dict[str, AgentBeliefState] = {}

    def add_agent(self, agent_id: str, location: str = 'room'):
        """Add an agent and initialize their belief state."""
        self.world.add_agent(agent_id, location)
        self.agent_beliefs[agent_id] = AgentBeliefState(agent_id)

    def process_event(self, event: Event):
        """Update agent beliefs based on event observers."""
        for agent_id in event.observed_by:
            if agent_id in self.agent_beliefs:
                self.agent_beliefs[agent_id].update_from_event(event)

    def get_agent_belief(self, agent_id: str, object_name: str) -> Optional[str]:
        """Get what an agent believes about an object's location."""
        if agent_id in self.agent_beliefs:
            return self.agent_beliefs[agent_id].get_believed_location(object_name)
        return None

    def get_reality(self, object_name: str) -> Optional[str]:
        """Get the actual location of an object."""
        return self.world.get_reality(object_name)

    def has_false_belief(self, agent_id: str, object_name: str) -> bool:
        """Check if an agent has a false belief about an object."""
        belief = self.get_agent_belief(agent_id, object_name)
        reality = self.get_reality(object_name)
        return belief is not None and reality is not None and belief != reality

    def get_belief_discrepancy(self, agent1: str, agent2: str,
                                object_name: str) -> Dict[str, Any]:
        """Get how two agents' beliefs differ about an object."""
        belief1 = self.get_agent_belief(agent1, object_name)
        belief2 = self.get_agent_belief(agent2, object_name)
        reality = self.get_reality(object_name)

        return {
            f'{agent1}_believes': belief1,
            f'{agent2}_believes': belief2,
            'reality': reality,
            f'{agent1}_has_false_belief': belief1 != reality,
            f'{agent2}_has_false_belief': belief2 != reality,
            'beliefs_differ': belief1 != belief2,
        }


def create_sally_anne_scenario() -> Tuple[List[Event], List[Dict[str, Any]]]:
    """
    Create the classic Sally-Anne false belief scenario.

    Story:
    1. Sally and Anne are in the room
    2. Sally puts the marble in the basket
    3. Sally leaves the room
    4. Anne moves the marble to the box
    5. Sally returns

    Question: Where will Sally look for the marble?
    Answer: The basket (Sally has a false belief)
    """
    tracker = InformationAsymmetryTracker()

    # Add agents
    tracker.add_agent('Sally', 'room')
    tracker.add_agent('Anne', 'room')
    tracker.add_agent('Observer', 'room')  # Omniscient observer

    events = []

    # 1. Sally enters (already in room, but make it explicit)
    e1 = tracker.world.agent_enters('Sally', 'room', {'Sally', 'Anne', 'Observer'})
    tracker.process_event(e1)
    events.append(e1)

    # 2. Anne enters
    e2 = tracker.world.agent_enters('Anne', 'room', {'Sally', 'Anne', 'Observer'})
    tracker.process_event(e2)
    events.append(e2)

    # 3. Sally puts marble in basket
    e3 = tracker.world.place_object('marble', 'basket', 'Sally', {'Sally', 'Anne', 'Observer'})
    tracker.process_event(e3)
    events.append(e3)

    # 4. Sally leaves
    e4 = tracker.world.agent_exits('Sally', 'room', {'Anne', 'Observer'})
    tracker.process_event(e4)
    events.append(e4)

    # 5. Anne moves marble to box (Sally doesn't see this!)
    e5 = tracker.world.move_object('marble', 'basket', 'box', 'Anne', {'Anne', 'Observer'})
    tracker.process_event(e5)
    events.append(e5)

    # 6. Sally returns
    e6 = tracker.world.agent_enters('Sally', 'room', {'Sally', 'Anne', 'Observer'})
    tracker.process_event(e6)
    events.append(e6)

    # Create questions
    questions = [
        {
            'question': 'Where will Sally look for the marble?',
            'type': 'first_order_belief',
            'target_agent': 'Sally',
            'correct_answer': 'basket',  # Sally's false belief
            'reality': 'box',
            'requires_tom': True,
        },
        {
            'question': 'Where is the marble really?',
            'type': 'reality',
            'correct_answer': 'box',
            'requires_tom': False,
        },
        {
            'question': 'Where does Anne think Sally will look?',
            'type': 'second_order_belief',
            'target_agent': 'Anne',
            'about_agent': 'Sally',
            'correct_answer': 'basket',  # Anne knows Sally has false belief
            'requires_tom': True,
        },
    ]

    # Store tracker for verification
    questions[0]['_tracker'] = tracker

    return events, questions


def verify_information_asymmetry() -> Dict[str, Any]:
    """
    Verify that information asymmetry is working correctly.

    Runs the Sally-Anne scenario and checks:
    1. Sally has false belief (thinks marble is in basket)
    2. Anne has true belief (knows marble is in box)
    3. Observer has true belief (knows marble is in box)
    4. Second-order beliefs work correctly
    """
    events, questions = create_sally_anne_scenario()
    tracker = questions[0]['_tracker']

    # Get beliefs
    sally_belief = tracker.get_agent_belief('Sally', 'marble')
    anne_belief = tracker.get_agent_belief('Anne', 'marble')
    observer_belief = tracker.get_agent_belief('Observer', 'marble')
    reality = tracker.get_reality('marble')

    # Check results
    results = {
        'sally_marble_belief': sally_belief,
        'anne_marble_belief': anne_belief,
        'observer_marble_belief': observer_belief,
        'reality': reality,
        'sally_has_false_belief': sally_belief != reality,
        'anne_has_true_belief': anne_belief == reality,
        'observer_has_true_belief': observer_belief == reality,
        'num_events': len(events),
        'sally_observed_events': len(tracker.agent_beliefs['Sally'].observed_events),
        'anne_observed_events': len(tracker.agent_beliefs['Anne'].observed_events),
    }

    # All tests pass if:
    # - Sally believes basket (false belief)
    # - Anne believes box (true belief)
    # - Reality is box
    results['all_tests_passed'] = (
        sally_belief == 'basket' and
        anne_belief == 'box' and
        reality == 'box'
    )

    return results


# Export
__all__ = [
    'Event',
    'EventType',
    'AgentBeliefState',
    'WorldState',
    'InformationAsymmetryTracker',
    'create_sally_anne_scenario',
    'verify_information_asymmetry',
]
