"""
Event System with Observation Tracking for Theory of Mind NAS

This module implements the event system that tracks which agents observed which events,
enabling proper information asymmetry for false belief reasoning.

The key insight is that an agent's beliefs are computed from ONLY the events they observed,
not from the complete event history. This creates the possibility of false beliefs when
an agent does not observe an event that changes the world state.
"""

import torch
import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    """Types of actions that can occur in Theory of Mind scenarios."""
    ENTER = "enter"          # Agent enters a location
    LEAVE = "leave"          # Agent leaves a location
    PUT = "put"              # Agent places object at location
    MOVE = "move"            # Agent moves object from one location to another
    TAKE = "take"            # Agent picks up object
    SAY = "say"              # Agent makes a statement (observable)
    THINK = "think"          # Agent has a thought (typically hidden)
    LOOK = "look"            # Agent looks at something
    OBSERVE = "observe"      # Agent observes the environment


@dataclass
class Event:
    """
    Represents a single event in a Theory of Mind scenario.

    The critical field is `observed_by`, which tracks which agents perceived this event.
    An agent's beliefs are computed from only the events they observed.

    Attributes:
        timestamp: Integer indicating when the event occurred (lower = earlier)
        actor: The agent who performed the action
        action: The type of action performed (ActionType enum or string)
        object: The object involved in the action, if any
        source_location: The location something came from (for move actions)
        target_location: The location something went to (for enter, put, move actions)
        content: The content of a say or think action
        observed_by: Set of agent names who observed this event
    """
    timestamp: int
    actor: str
    action: str  # Can be ActionType.value or string
    object: Optional[str] = None
    source_location: Optional[str] = None
    target_location: Optional[str] = None
    content: Optional[str] = None
    observed_by: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Ensure observed_by is a set."""
        if isinstance(self.observed_by, (list, tuple)):
            self.observed_by = set(self.observed_by)
        # The actor always observes their own action
        if self.actor:
            self.observed_by.add(self.actor)

    def was_observed_by(self, agent: str) -> bool:
        """Check if a specific agent observed this event."""
        return agent in self.observed_by

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'actor': self.actor,
            'action': self.action,
            'object': self.object,
            'source_location': self.source_location,
            'target_location': self.target_location,
            'content': self.content,
            'observed_by': list(self.observed_by)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            actor=data['actor'],
            action=data['action'],
            object=data.get('object'),
            source_location=data.get('source_location'),
            target_location=data.get('target_location'),
            content=data.get('content'),
            observed_by=set(data.get('observed_by', []))
        )


class AgentBeliefs:
    """
    Computes an agent's beliefs from their observed event sequence.

    This class filters events to only those observed by the agent, then computes
    beliefs about object locations and agent locations based on those observations.

    This is the core mechanism for creating false beliefs: if an agent didn't observe
    an event that changed an object's location, their belief remains at the last
    location they observed.
    """

    def __init__(self, agent_name: str, events: List[Event]):
        """
        Initialize beliefs from an event sequence.

        Args:
            agent_name: The name of the agent whose beliefs we're computing
            events: The complete event sequence (will be filtered to observed events)
        """
        self.agent_name = agent_name

        # Filter to only events this agent observed
        self.observed_events = [e for e in events if agent_name in e.observed_by]

        # Compute beliefs from observed events
        self._object_locations: Dict[str, Optional[str]] = {}
        self._agent_locations: Dict[str, Optional[str]] = {}
        self._agent_presence: Dict[str, Dict[str, bool]] = {}  # location -> agent -> present

        self._compute_beliefs()

    def _compute_beliefs(self):
        """Process observed events to build belief state."""
        for event in self.observed_events:
            action = event.action.lower() if isinstance(event.action, str) else event.action.value.lower()

            if action == "put":
                # Object is placed at target location
                if event.object and event.target_location:
                    self._object_locations[event.object] = event.target_location

            elif action == "move":
                # Object moved from source to target
                if event.object and event.target_location:
                    self._object_locations[event.object] = event.target_location

            elif action == "take":
                # Object taken - now with the actor
                if event.object:
                    self._object_locations[event.object] = f"with_{event.actor}"

            elif action == "enter":
                # Agent entered a location
                if event.target_location:
                    self._agent_locations[event.actor] = event.target_location
                    # Track presence at locations
                    if event.target_location not in self._agent_presence:
                        self._agent_presence[event.target_location] = {}
                    self._agent_presence[event.target_location][event.actor] = True

            elif action == "leave":
                # Agent left - location now unknown or None
                if event.source_location:
                    if event.source_location in self._agent_presence:
                        self._agent_presence[event.source_location][event.actor] = False
                self._agent_locations[event.actor] = None

    def get_object_location(self, object_name: str) -> Optional[str]:
        """
        Get the agent's belief about where an object is located.

        Returns None if the agent has no relevant observations.
        """
        return self._object_locations.get(object_name)

    def get_agent_location(self, agent_name: str) -> Optional[str]:
        """
        Get the agent's belief about where another agent is located.

        Returns None if location is unknown.
        """
        return self._agent_locations.get(agent_name)

    def get_agents_at_location(self, location: str) -> Set[str]:
        """Get all agents the believer thinks are at a specific location."""
        if location not in self._agent_presence:
            return set()
        return {agent for agent, present in self._agent_presence[location].items() if present}

    def get_belief_about_other(self, other_agent: str, object_name: str) -> Optional[str]:
        """
        Compute what this agent believes another agent believes about an object's location.

        This is second-order Theory of Mind: "Where does Anne think Sally thinks the marble is?"

        The computation:
        1. Track when the other agent was present based on our observations
        2. Filter our observations to those the other agent likely also observed
        3. Compute the other agent's beliefs from that filtered set
        """
        # Build a model of when the other agent was present at various locations
        other_agent_present_at: Dict[str, List[Tuple[int, int]]] = {}  # location -> [(start, end), ...]
        current_location = None
        enter_time = None

        for event in self.observed_events:
            action = event.action.lower() if isinstance(event.action, str) else event.action.value.lower()

            if event.actor == other_agent:
                if action == "enter":
                    current_location = event.target_location
                    enter_time = event.timestamp
                elif action == "leave":
                    if current_location and enter_time is not None:
                        if current_location not in other_agent_present_at:
                            other_agent_present_at[current_location] = []
                        other_agent_present_at[current_location].append((enter_time, event.timestamp))
                    current_location = None
                    enter_time = None

        # If other agent is still at a location, they're present until "now"
        if current_location and enter_time is not None:
            max_timestamp = max(e.timestamp for e in self.observed_events) if self.observed_events else 0
            if current_location not in other_agent_present_at:
                other_agent_present_at[current_location] = []
            other_agent_present_at[current_location].append((enter_time, max_timestamp + 1))

        # Filter events to those the other agent likely observed
        def other_likely_observed(event: Event) -> bool:
            """Check if the other agent was likely present to observe this event."""
            # If the other agent is the actor, they observed it
            if event.actor == other_agent:
                return True

            # Check if other agent was present at the event's location
            event_location = event.target_location or event.source_location
            if event_location and event_location in other_agent_present_at:
                for start, end in other_agent_present_at[event_location]:
                    if start <= event.timestamp <= end:
                        return True

            return False

        # Build filtered events
        filtered_events = [e for e in self.observed_events if other_likely_observed(e)]

        # Create beliefs for the other agent based on filtered events
        other_beliefs = AgentBeliefs.__new__(AgentBeliefs)
        other_beliefs.agent_name = other_agent
        other_beliefs.observed_events = filtered_events
        other_beliefs._object_locations = {}
        other_beliefs._agent_locations = {}
        other_beliefs._agent_presence = {}
        other_beliefs._compute_beliefs()

        return other_beliefs.get_object_location(object_name)


@dataclass
class QuestionType(Enum):
    """Types of Theory of Mind questions."""
    REALITY = "reality"                    # What is the actual state?
    MEMORY = "memory"                      # What was the state at time T?
    FIRST_ORDER_BELIEF = "first_order"     # What does X believe?
    SECOND_ORDER_BELIEF = "second_order"   # What does X think Y believes?
    HIGHER_ORDER_BELIEF = "higher_order"   # Deeper nesting


@dataclass
class Question:
    """
    A question about a Theory of Mind scenario.

    Attributes:
        question_type: The type of question (reality, belief, etc.)
        target_agent: The agent whose beliefs we're asking about (for belief questions)
        secondary_agent: For second-order: who target_agent is modeling
        target_object: The object we're asking about
        answer_choices: List of possible answers
        correct_answer: The ground truth answer
    """
    question_type: str  # QuestionType value
    target_object: str
    target_agent: Optional[str] = None
    secondary_agent: Optional[str] = None
    answer_choices: List[str] = field(default_factory=list)
    correct_answer: Optional[str] = None


class EventEncoder:
    """
    Encodes events into 181-dimensional Soul Map vectors.

    The encoding allocates specific dimension ranges for different feature types:
    - Dims 0-9: Agent encoding (one-hot)
    - Dims 10-19: Action encoding (one-hot)
    - Dims 20-39: Object encoding (one-hot)
    - Dims 40-49: Source location encoding (one-hot)
    - Dims 50-59: Target location encoding (one-hot)
    - Dims 60-69: Observer encoding (multi-hot)
    - Dim 70: Normalized timestamp
    - Dims 71-180: Available for additional features
    """

    # Dimension allocation
    AGENT_DIM_START = 0
    AGENT_DIM_END = 10
    ACTION_DIM_START = 10
    ACTION_DIM_END = 20
    OBJECT_DIM_START = 20
    OBJECT_DIM_END = 40
    SOURCE_LOC_DIM_START = 40
    SOURCE_LOC_DIM_END = 50
    TARGET_LOC_DIM_START = 50
    TARGET_LOC_DIM_END = 60
    OBSERVER_DIM_START = 60
    OBSERVER_DIM_END = 70
    TIMESTAMP_DIM = 70

    TOTAL_DIMS = 181

    def __init__(self):
        """Initialize with default vocabularies."""
        # Agent vocabulary (max 10 agents)
        self.agents_vocab: Dict[str, int] = {}
        self.agents_list: List[str] = []

        # Action vocabulary
        self.actions_vocab = {
            'enter': 0, 'leave': 1, 'put': 2, 'move': 3,
            'take': 4, 'say': 5, 'think': 6, 'look': 7, 'observe': 8
        }

        # Object vocabulary (max 20 objects)
        self.objects_vocab: Dict[str, int] = {}
        self.objects_list: List[str] = []

        # Location vocabulary (max 10 locations)
        self.locations_vocab: Dict[str, int] = {}
        self.locations_list: List[str] = []

    def register_agent(self, agent: str) -> int:
        """Register an agent and return its index."""
        if agent not in self.agents_vocab:
            if len(self.agents_list) >= (self.AGENT_DIM_END - self.AGENT_DIM_START):
                raise ValueError(f"Maximum number of agents ({self.AGENT_DIM_END - self.AGENT_DIM_START}) exceeded")
            idx = len(self.agents_list)
            self.agents_vocab[agent] = idx
            self.agents_list.append(agent)
        return self.agents_vocab[agent]

    def register_object(self, obj: str) -> int:
        """Register an object and return its index."""
        if obj not in self.objects_vocab:
            if len(self.objects_list) >= (self.OBJECT_DIM_END - self.OBJECT_DIM_START):
                raise ValueError(f"Maximum number of objects ({self.OBJECT_DIM_END - self.OBJECT_DIM_START}) exceeded")
            idx = len(self.objects_list)
            self.objects_vocab[obj] = idx
            self.objects_list.append(obj)
        return self.objects_vocab[obj]

    def register_location(self, loc: str) -> int:
        """Register a location and return its index."""
        if loc not in self.locations_vocab:
            if len(self.locations_list) >= (self.TARGET_LOC_DIM_END - self.TARGET_LOC_DIM_START):
                raise ValueError(f"Maximum number of locations ({self.TARGET_LOC_DIM_END - self.TARGET_LOC_DIM_START}) exceeded")
            idx = len(self.locations_list)
            self.locations_vocab[loc] = idx
            self.locations_list.append(loc)
        return self.locations_vocab[loc]

    def register_from_events(self, events: List[Event]):
        """Register all agents, objects, and locations from a list of events."""
        for event in events:
            # Register actor
            if event.actor:
                self.register_agent(event.actor)

            # Register all observers
            for observer in event.observed_by:
                self.register_agent(observer)

            # Register object
            if event.object:
                self.register_object(event.object)

            # Register locations
            if event.source_location:
                self.register_location(event.source_location)
            if event.target_location:
                self.register_location(event.target_location)

    def encode_event(self, event: Event, max_timestamp: int = 100) -> torch.Tensor:
        """
        Encode a single event into a 181-dimensional vector.

        Args:
            event: The event to encode
            max_timestamp: Maximum timestamp for normalization

        Returns:
            torch.Tensor of shape (181,)
        """
        vector = torch.zeros(self.TOTAL_DIMS)

        # Encode actor (one-hot)
        if event.actor and event.actor in self.agents_vocab:
            idx = self.AGENT_DIM_START + self.agents_vocab[event.actor]
            vector[idx] = 1.0

        # Encode action (one-hot)
        action_str = event.action.lower() if isinstance(event.action, str) else event.action.value.lower()
        if action_str in self.actions_vocab:
            idx = self.ACTION_DIM_START + self.actions_vocab[action_str]
            vector[idx] = 1.0

        # Encode object (one-hot)
        if event.object and event.object in self.objects_vocab:
            idx = self.OBJECT_DIM_START + self.objects_vocab[event.object]
            vector[idx] = 1.0

        # Encode source location (one-hot)
        if event.source_location and event.source_location in self.locations_vocab:
            idx = self.SOURCE_LOC_DIM_START + self.locations_vocab[event.source_location]
            vector[idx] = 1.0

        # Encode target location (one-hot)
        if event.target_location and event.target_location in self.locations_vocab:
            idx = self.TARGET_LOC_DIM_START + self.locations_vocab[event.target_location]
            vector[idx] = 1.0

        # Encode observers (multi-hot)
        for observer in event.observed_by:
            if observer in self.agents_vocab:
                idx = self.OBSERVER_DIM_START + self.agents_vocab[observer]
                vector[idx] = 1.0

        # Encode normalized timestamp
        vector[self.TIMESTAMP_DIM] = event.timestamp / max(max_timestamp, 1)

        return vector

    def encode_events(self, events: List[Event]) -> torch.Tensor:
        """
        Encode a sequence of events into a tensor.

        Args:
            events: List of events to encode

        Returns:
            torch.Tensor of shape (num_events, 181)
        """
        if not events:
            return torch.zeros(1, self.TOTAL_DIMS)

        max_timestamp = max(e.timestamp for e in events)
        encoded = [self.encode_event(e, max_timestamp) for e in events]
        return torch.stack(encoded)

    def get_location_vector(self, location: str) -> torch.Tensor:
        """Get a vector representation for a location (for answer decoding)."""
        vector = torch.zeros(self.TOTAL_DIMS)
        if location in self.locations_vocab:
            idx = self.TARGET_LOC_DIM_START + self.locations_vocab[location]
            vector[idx] = 1.0
        return vector


class ScenarioEncoder:
    """
    Encodes complete scenarios including events and questions.

    The question is encoded as the final "event" in the sequence, with special
    dimensions indicating the question type, target agent, and target object.
    """

    # Question encoding dimensions (within the Soul Map)
    QUESTION_TYPE_DIM_START = 170
    QUESTION_TYPE_DIM_END = 174
    TARGET_AGENT_DIM_START = 174
    TARGET_AGENT_DIM_END = 178
    TARGET_OBJECT_DIM_START = 178
    TARGET_OBJECT_DIM_END = 181

    # Question type encoding
    QUESTION_TYPES = {
        'reality': 0,
        'memory': 1,
        'first_order': 2,
        'second_order': 3
    }

    def __init__(self, event_encoder: Optional[EventEncoder] = None):
        """Initialize with an optional event encoder."""
        self.event_encoder = event_encoder or EventEncoder()

    def encode_question(self, question: Question) -> torch.Tensor:
        """
        Encode a question into a 181-dimensional vector.

        Args:
            question: The question to encode

        Returns:
            torch.Tensor of shape (181,)
        """
        vector = torch.zeros(EventEncoder.TOTAL_DIMS)

        # Encode question type (one-hot)
        q_type = question.question_type.lower()
        if q_type in self.QUESTION_TYPES:
            idx = self.QUESTION_TYPE_DIM_START + self.QUESTION_TYPES[q_type]
            vector[idx] = 1.0

        # Encode target agent
        if question.target_agent and question.target_agent in self.event_encoder.agents_vocab:
            idx = self.TARGET_AGENT_DIM_START + self.event_encoder.agents_vocab[question.target_agent]
            vector[idx] = 1.0

        # Encode target object
        if question.target_object and question.target_object in self.event_encoder.objects_vocab:
            # Map to the available dimensions
            obj_idx = self.event_encoder.objects_vocab[question.target_object]
            if obj_idx < (self.TARGET_OBJECT_DIM_END - self.TARGET_OBJECT_DIM_START):
                idx = self.TARGET_OBJECT_DIM_START + obj_idx
                vector[idx] = 1.0

        return vector

    def encode_scenario(self, events: List[Event], question: Question) -> torch.Tensor:
        """
        Encode a complete scenario (events + question) into a tensor.

        Args:
            events: List of events in the scenario
            question: The question about the scenario

        Returns:
            torch.Tensor of shape (num_events + 1, 181) where the last row is the question
        """
        # First register all entities from events
        self.event_encoder.register_from_events(events)

        # Encode events
        event_encodings = self.event_encoder.encode_events(events)

        # Encode question
        question_encoding = self.encode_question(question).unsqueeze(0)

        # Concatenate
        return torch.cat([event_encodings, question_encoding], dim=0)


class AnswerDecoder:
    """
    Decodes network outputs into predicted answers.

    For location questions, compares the output to location encodings
    and returns the most similar location.
    """

    def __init__(self, event_encoder: EventEncoder):
        """Initialize with the event encoder that knows the vocabularies."""
        self.event_encoder = event_encoder
        self._location_vectors: Optional[torch.Tensor] = None

    def _build_location_vectors(self):
        """Pre-compute location vectors for efficient comparison."""
        if not self.event_encoder.locations_list:
            self._location_vectors = None
            return

        vectors = []
        for loc in self.event_encoder.locations_list:
            vectors.append(self.event_encoder.get_location_vector(loc))
        self._location_vectors = torch.stack(vectors)

    def decode_location(self, output: torch.Tensor) -> Optional[str]:
        """
        Decode a network output to a predicted location.

        Args:
            output: Network output tensor of shape (181,) or (batch, 181)

        Returns:
            The predicted location string, or None if no locations registered
        """
        if not self.event_encoder.locations_list:
            return None

        if self._location_vectors is None:
            self._build_location_vectors()

        # Handle batched input
        if output.dim() == 1:
            output = output.unsqueeze(0)

        # Compute similarity with each location
        # Use dot product as similarity measure
        similarities = torch.matmul(output, self._location_vectors.T)

        # Get the most similar location
        best_idx = similarities.argmax(dim=-1)

        if output.shape[0] == 1:
            return self.event_encoder.locations_list[best_idx.item()]
        else:
            return [self.event_encoder.locations_list[idx.item()] for idx in best_idx]

    def decode_multiple_choice(self, output: torch.Tensor, choices: List[str]) -> int:
        """
        Decode to a multiple choice answer.

        Args:
            output: Network output tensor
            choices: List of answer choices

        Returns:
            Index of the predicted choice
        """
        # For multiple choice, we use the output dimensions that encode objects/locations
        # and compare to the choices

        # Simple approach: encode each choice and find most similar
        choice_vectors = []
        for choice in choices:
            vec = torch.zeros(EventEncoder.TOTAL_DIMS)
            # Check if it's a known location
            if choice in self.event_encoder.locations_vocab:
                idx = EventEncoder.TARGET_LOC_DIM_START + self.event_encoder.locations_vocab[choice]
                vec[idx] = 1.0
            # Check if it's a known object
            elif choice in self.event_encoder.objects_vocab:
                idx = EventEncoder.OBJECT_DIM_START + self.event_encoder.objects_vocab[choice]
                vec[idx] = 1.0
            choice_vectors.append(vec)

        if not choice_vectors:
            return 0

        choice_matrix = torch.stack(choice_vectors)

        if output.dim() == 1:
            similarities = torch.matmul(choice_matrix, output)
        else:
            similarities = torch.matmul(choice_matrix, output.squeeze())

        return similarities.argmax().item()


def compute_ground_truth(events: List[Event], question: Question) -> Optional[str]:
    """
    Compute the ground truth answer for a Theory of Mind question.

    Args:
        events: The complete event sequence
        question: The question to answer

    Returns:
        The correct answer as a string
    """
    q_type = question.question_type.lower()

    if q_type == 'reality':
        # Reality: What is the actual current state?
        # Process all events to get final state
        object_locations: Dict[str, str] = {}
        for event in events:
            action = event.action.lower() if isinstance(event.action, str) else event.action.value.lower()
            if action in ('put', 'move') and event.object and event.target_location:
                object_locations[event.object] = event.target_location
        return object_locations.get(question.target_object)

    elif q_type == 'first_order':
        # First-order belief: What does target_agent believe?
        if not question.target_agent:
            return None
        beliefs = AgentBeliefs(question.target_agent, events)
        return beliefs.get_object_location(question.target_object)

    elif q_type == 'second_order':
        # Second-order belief: What does target_agent think secondary_agent believes?
        if not question.target_agent or not question.secondary_agent:
            return None
        beliefs = AgentBeliefs(question.target_agent, events)
        return beliefs.get_belief_about_other(question.secondary_agent, question.target_object)

    elif q_type == 'memory':
        # Memory: What was the state at some earlier time?
        # This would require a timestamp parameter in the question
        # For now, return None
        return None

    return None


def create_sally_anne_scenario() -> Tuple[List[Event], List[Question]]:
    """
    Create the classic Sally-Anne false belief scenario.

    Returns:
        Tuple of (events, questions) for the scenario
    """
    events = [
        Event(
            timestamp=1,
            actor="Sally",
            action="enter",
            target_location="living_room",
            observed_by={"Sally", "Anne", "Observer"}
        ),
        Event(
            timestamp=2,
            actor="Anne",
            action="enter",
            target_location="living_room",
            observed_by={"Sally", "Anne", "Observer"}
        ),
        Event(
            timestamp=3,
            actor="Sally",
            action="put",
            object="marble",
            target_location="basket",
            observed_by={"Sally", "Anne", "Observer"}
        ),
        Event(
            timestamp=4,
            actor="Sally",
            action="leave",
            source_location="living_room",
            observed_by={"Anne", "Observer"}
        ),
        Event(
            timestamp=5,
            actor="Anne",
            action="move",
            object="marble",
            source_location="basket",
            target_location="box",
            observed_by={"Anne", "Observer"}
        ),
        Event(
            timestamp=6,
            actor="Sally",
            action="enter",
            target_location="living_room",
            observed_by={"Sally", "Anne", "Observer"}
        )
    ]

    questions = [
        Question(
            question_type="reality",
            target_object="marble",
            answer_choices=["basket", "box"],
            correct_answer="box"
        ),
        Question(
            question_type="first_order",
            target_agent="Sally",
            target_object="marble",
            answer_choices=["basket", "box"],
            correct_answer="basket"  # Sally has false belief!
        ),
        Question(
            question_type="first_order",
            target_agent="Anne",
            target_object="marble",
            answer_choices=["basket", "box"],
            correct_answer="box"  # Anne has true belief
        ),
        Question(
            question_type="second_order",
            target_agent="Anne",
            secondary_agent="Sally",
            target_object="marble",
            answer_choices=["basket", "box"],
            correct_answer="basket"  # Anne knows Sally has false belief
        )
    ]

    return events, questions


def verify_information_asymmetry() -> Dict[str, Any]:
    """
    Verification test that confirms the observation system creates false beliefs.

    Returns:
        Dictionary with verification results
    """
    events, questions = create_sally_anne_scenario()
    results = {}

    # Compute beliefs for each agent
    sally_beliefs = AgentBeliefs("Sally", events)
    anne_beliefs = AgentBeliefs("Anne", events)
    observer_beliefs = AgentBeliefs("Observer", events)

    results['sally_marble_belief'] = sally_beliefs.get_object_location("marble")
    results['anne_marble_belief'] = anne_beliefs.get_object_location("marble")
    results['observer_marble_belief'] = observer_beliefs.get_object_location("marble")

    # Verify false belief exists
    results['sally_has_false_belief'] = (
        results['sally_marble_belief'] == "basket" and
        results['observer_marble_belief'] == "box"
    )

    # Verify ground truth computation
    for question in questions:
        gt = compute_ground_truth(events, question)
        q_key = f"{question.question_type}_{question.target_agent or 'reality'}"
        results[f'ground_truth_{q_key}'] = gt
        results[f'correct_{q_key}'] = (gt == question.correct_answer)

    # Verify second-order belief
    anne_model_of_sally = anne_beliefs.get_belief_about_other("Sally", "marble")
    results['anne_thinks_sally_believes'] = anne_model_of_sally
    results['second_order_correct'] = (anne_model_of_sally == "basket")

    results['all_tests_passed'] = all(
        v for k, v in results.items()
        if k.startswith('correct_') or k.endswith('_correct') or k == 'sally_has_false_belief'
    )

    return results


if __name__ == "__main__":
    # Run verification tests
    print("=" * 60)
    print("INFORMATION ASYMMETRY VERIFICATION")
    print("=" * 60)

    results = verify_information_asymmetry()

    print("\n--- Agent Beliefs ---")
    print(f"Sally believes marble is in: {results['sally_marble_belief']}")
    print(f"Anne believes marble is in: {results['anne_marble_belief']}")
    print(f"Observer knows marble is in: {results['observer_marble_belief']}")

    print(f"\nSally has false belief: {results['sally_has_false_belief']}")

    print("\n--- Ground Truth Answers ---")
    for key, value in results.items():
        if key.startswith('ground_truth_'):
            print(f"  {key}: {value}")

    print("\n--- Second-Order Belief ---")
    print(f"Anne thinks Sally believes marble is in: {results['anne_thinks_sally_believes']}")
    print(f"Second-order correct: {results['second_order_correct']}")

    print("\n" + "=" * 60)
    print(f"ALL TESTS PASSED: {results['all_tests_passed']}")
    print("=" * 60)
