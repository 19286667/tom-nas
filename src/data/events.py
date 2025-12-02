"""
Event representation with observation tracking for Theory of Mind.

The key insight is that ToM requires information asymmetry - different agents
observe different events, leading to divergent beliefs about the world state.
"""

from dataclasses import dataclass, field
from typing import Set, Optional, List


@dataclass
class Event:
    """
    Single event in a scenario with observation tracking.

    This is the foundation for creating false beliefs - when an agent
    doesn't observe an event (not in observed_by), their belief state
    diverges from reality.

    Attributes:
        timestamp: When the event occurred (sequential ordering)
        actor: Who performed the action
        action: What was done ('enter', 'leave', 'put', 'move', 'take', 'say')
        target_location: Destination location (for put, move, enter)
        source_location: Origin location (for move, leave)
        object: Object being manipulated (if any)
        observed_by: Set of agents who witnessed this event
    """
    timestamp: int
    actor: str
    action: str  # 'enter', 'leave', 'put', 'move', 'take', 'say'
    target_location: Optional[str] = None
    source_location: Optional[str] = None
    object: Optional[str] = None
    observed_by: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Ensure actor always observes their own action."""
        self.observed_by = set(self.observed_by)
        self.observed_by.add(self.actor)

    def was_observed_by(self, agent: str) -> bool:
        """Check if a specific agent observed this event."""
        return agent in self.observed_by

    def __repr__(self) -> str:
        parts = [f"t={self.timestamp}", f"{self.actor}:{self.action}"]
        if self.object:
            parts.append(f"obj={self.object}")
        if self.target_location:
            parts.append(f"to={self.target_location}")
        if self.source_location:
            parts.append(f"from={self.source_location}")
        parts.append(f"obs={self.observed_by}")
        return f"Event({', '.join(parts)})"


@dataclass
class Scenario:
    """
    Complete scenario with events, question, and ground truth.

    A scenario represents a Sally-Anne style false belief test:
    1. A sequence of events with observation tracking
    2. A question about reality or beliefs
    3. The ground truth answer (computed from observations)

    Question types:
    - 'reality': What is the actual current state?
    - 'first_order_belief': What does agent X believe about the state?
    - 'second_order_belief': What does agent X believe agent Y believes?

    Attributes:
        events: Sequence of events that occurred
        question_type: Type of question being asked
        question_target_agent: Agent whose belief is being queried (for belief questions)
        question_about_agent: Second agent for second-order questions
        question_target_object: Object the question is about
        ground_truth_answer: Correct answer (computed from observations)
        difficulty: Optional difficulty rating
    """
    events: List[Event]
    question_type: str  # 'reality', 'first_order_belief', 'second_order_belief'
    question_target_agent: Optional[str] = None
    question_about_agent: Optional[str] = None  # For second-order: "What does X think Y believes?"
    question_target_object: Optional[str] = None
    ground_truth_answer: Optional[str] = None
    difficulty: Optional[str] = None  # 'easy', 'medium', 'hard'

    def get_agent_observations(self, agent: str) -> List[Event]:
        """
        Get only events this agent observed.

        This is key for computing what an agent believes -
        they only update their beliefs based on observed events.
        """
        return [e for e in self.events if agent in e.observed_by]

    def get_all_agents(self) -> Set[str]:
        """Get all agents mentioned in the scenario."""
        agents = set()
        for event in self.events:
            agents.add(event.actor)
            agents.update(event.observed_by)
        return agents

    def get_all_objects(self) -> Set[str]:
        """Get all objects mentioned in the scenario."""
        objects = set()
        for event in self.events:
            if event.object:
                objects.add(event.object)
        return objects

    def get_all_locations(self) -> Set[str]:
        """Get all locations mentioned in the scenario."""
        locations = set()
        for event in self.events:
            if event.target_location:
                locations.add(event.target_location)
            if event.source_location:
                locations.add(event.source_location)
        return locations

    def get_final_reality(self, obj: str) -> Optional[str]:
        """Get the actual final location of an object."""
        location = None
        for event in self.events:
            if event.object == obj and event.action in ('put', 'move'):
                location = event.target_location
            elif event.object == obj and event.action == 'take':
                location = None  # Object is being held
        return location

    def __repr__(self) -> str:
        return (f"Scenario(events={len(self.events)}, "
                f"q={self.question_type}, "
                f"answer={self.ground_truth_answer})")
