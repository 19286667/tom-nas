"""
Mentalese - The Language of Thought

Implements TypeScript-style typed cognitive blocks as the fundamental
atomic units of thought. These are the "conspicuous and freely available
syntactical objects" from which language and reasoning emerge.

Mentalese is:
- Unambiguous (unlike natural language)
- Compositional (complex thoughts from atomic parts)
- Type-shifting (Percept -> Hypothesis -> Belief -> Memory)
- Recursive (beliefs about beliefs about beliefs...)

Natural language is treated as a "lossy compression" of Mentalese,
used for inter-agent communication, introducing realistic ambiguity.

Theoretical Foundation:
- Language of Thought Hypothesis (Fodor)
- Mental Model Theory (Johnson-Laird)
- Conceptual Semantics (Jackendoff)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, TypeVar, Generic
import numpy as np
from abc import ABC, abstractmethod
import json
import hashlib


class BlockType(Enum):
    """The fundamental types of cognitive blocks."""
    PERCEPT = auto()      # Raw sensory input
    HYPOTHESIS = auto()   # Possible interpretation
    BELIEF = auto()       # Accepted proposition
    INTENT = auto()       # Goal or plan
    MEMORY = auto()       # Compressed archetype
    SIMULATION = auto()   # Nested simulation state
    INFERENCE = auto()    # Derived conclusion


class ConfidenceLevel(Enum):
    """Categorical confidence levels for beliefs."""
    CERTAIN = 0.95
    HIGHLY_CONFIDENT = 0.85
    CONFIDENT = 0.75
    MODERATE = 0.60
    UNCERTAIN = 0.45
    DOUBTFUL = 0.30
    SKEPTICAL = 0.15


class ModalityType(Enum):
    """Epistemic and deontic modalities."""
    ACTUAL = auto()       # Is the case
    POSSIBLE = auto()     # Could be the case
    NECESSARY = auto()    # Must be the case
    PROBABLE = auto()     # Likely the case
    OBLIGATORY = auto()   # Should be the case (deontic)
    PERMITTED = auto()    # May be the case (deontic)
    FORBIDDEN = auto()    # Must not be the case (deontic)


@dataclass
class Evidence:
    """Evidence supporting a cognitive block."""
    source_id: str                # ID of evidence source
    source_type: str              # "percept", "inference", "testimony", "memory"
    strength: float               # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    content: Optional[str] = None


@dataclass
class CognitiveBlock(ABC):
    """
    Base class for all cognitive blocks - atomic units of Mentalese.

    A CognitiveBlock is the fundamental unit of thought in the system.
    It can be composed with other blocks, transformed through type-shifting,
    and recursively nested (beliefs about beliefs).
    """
    block_type: BlockType
    block_id: str = field(default_factory=lambda: "")
    created_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0       # 0.0 to 1.0
    modality: ModalityType = ModalityType.ACTUAL
    evidence: List[Evidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.block_id:
            self.block_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique block ID based on content hash."""
        content = f"{self.block_type.name}_{self.created_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @abstractmethod
    def to_tensor(self) -> np.ndarray:
        """Convert block to tensor representation for neural processing."""
        pass

    @abstractmethod
    def to_natural_language(self) -> str:
        """Compress to lossy natural language representation."""
        pass

    def add_evidence(self, evidence: Evidence) -> None:
        """Add supporting evidence and update confidence."""
        self.evidence.append(evidence)
        # Bayesian-style confidence update
        self.confidence = min(1.0, self.confidence + (1 - self.confidence) * evidence.strength * 0.3)


@dataclass
class PerceptBlock(CognitiveBlock):
    """
    A raw perceptual input - the grounding of Mentalese in physics.

    Percepts are directly linked to Godot physics engine objects,
    providing symbol grounding (Harnad's Symbol Grounding Problem).
    """
    block_type: BlockType = field(default=BlockType.PERCEPT, init=False)

    # Perceptual content
    perceived_entity: str = ""           # Entity ID perceived
    godot_id: Optional[int] = None       # Godot physics ID
    position_3d: Optional[tuple] = None  # 3D position
    visual_features: Dict[str, float] = field(default_factory=dict)

    # Sensory modality
    modality_type: str = "visual"        # visual, auditory, tactile, etc.

    # Semantic expansion (from Indra's Net)
    activated_concepts: List[str] = field(default_factory=list)
    taxonomy_position: Optional[np.ndarray] = None

    def to_tensor(self) -> np.ndarray:
        """Convert percept to tensor for neural processing."""
        # Combine visual features with taxonomy position
        features = list(self.visual_features.values())
        if self.taxonomy_position is not None:
            features.extend(self.taxonomy_position.tolist())
        return np.array(features, dtype=np.float32)

    def to_natural_language(self) -> str:
        """Generate natural language description."""
        return f"I perceive {self.perceived_entity} at position {self.position_3d}"


@dataclass
class HypothesisBlock(CognitiveBlock):
    """
    A possible interpretation of percepts - intermediate state before belief.

    Hypotheses represent the "superposition of meanings" before
    context collapses them into definite beliefs.
    """
    block_type: BlockType = field(default=BlockType.HYPOTHESIS, init=False)

    # Hypothesis content
    proposition: str = ""                # What is being hypothesized
    subject: str = ""                    # Subject of proposition
    predicate: str = ""                  # Predicate being asserted

    # Alternative hypotheses
    alternatives: List[str] = field(default_factory=list)
    alternative_probs: List[float] = field(default_factory=list)

    # Source percepts
    source_percepts: List[str] = field(default_factory=list)  # Block IDs

    # Evaluation criteria
    testable_predictions: List[str] = field(default_factory=list)
    falsification_conditions: List[str] = field(default_factory=list)

    def to_tensor(self) -> np.ndarray:
        """Convert hypothesis to tensor."""
        # Encode proposition as embedding (simplified)
        # In full implementation, would use semantic embeddings
        return np.array([
            self.confidence,
            len(self.alternatives),
            len(self.source_percepts),
            len(self.testable_predictions),
        ], dtype=np.float32)

    def to_natural_language(self) -> str:
        """Generate natural language description."""
        conf_str = f"({self.confidence:.0%} confident)"
        return f"Maybe {self.subject} {self.predicate}. {conf_str}"


@dataclass
class BeliefBlock(CognitiveBlock):
    """
    An accepted proposition - the core unit of Theory of Mind.

    Beliefs can be:
    - First-order: I believe P
    - Second-order: I believe you believe P
    - Nth-order: Recursive nesting to arbitrary depth

    This is where transparent ToM happens - we can trace exactly
    what an agent believes about another agent's beliefs.
    """
    block_type: BlockType = field(default=BlockType.BELIEF, init=False)

    # Belief content
    proposition: str = ""                # The believed proposition
    subject: str = ""                    # Subject of belief
    predicate: str = ""                  # What is believed about subject

    # Belief target (for ToM)
    about_agent: Optional[str] = None    # If about another agent
    belief_order: int = 1                # 1 = first-order, 2 = second-order, etc.

    # Nested belief (for recursive ToM)
    nested_belief: Optional['BeliefBlock'] = None

    # Temporal aspects
    belief_start: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    decay_rate: float = 0.01             # How fast confidence decays

    # Source information
    source_type: str = "inference"       # perception, inference, testimony, simulation

    def to_tensor(self) -> np.ndarray:
        """Convert belief to tensor."""
        features = [
            self.confidence,
            self.belief_order,
            1.0 if self.about_agent else 0.0,
            1.0 if self.nested_belief else 0.0,
            self.decay_rate,
        ]
        return np.array(features, dtype=np.float32)

    def to_natural_language(self) -> str:
        """Generate natural language description."""
        if self.about_agent and self.nested_belief:
            nested_nl = self.nested_belief.to_natural_language()
            return f"I believe that {self.about_agent} believes: {nested_nl}"
        elif self.about_agent:
            return f"I believe that {self.about_agent} {self.predicate}"
        else:
            return f"I believe that {self.subject} {self.predicate}"

    def get_recursive_depth(self) -> int:
        """Get the recursive depth of this belief."""
        if self.nested_belief is None:
            return self.belief_order
        return self.belief_order + self.nested_belief.get_recursive_depth()


@dataclass
class IntentBlock(CognitiveBlock):
    """
    A goal or intention - drives action and explains behavior.

    Intents are crucial for ToM: understanding WHY an agent acts
    requires modeling their intentions, not just their actions.
    """
    block_type: BlockType = field(default=BlockType.INTENT, init=False)

    # Intent content
    goal: str = ""                       # The desired end state
    action_type: str = ""                # Type of action to achieve goal

    # Intent structure
    target_entity: Optional[str] = None  # What/who the intent targets
    target_agent: Optional[str] = None   # If intent is social

    # Preconditions and effects
    preconditions: List[str] = field(default_factory=list)
    expected_effects: List[str] = field(default_factory=list)

    # Intent about another's intent (ToM)
    about_agent: Optional[str] = None
    nested_intent: Optional['IntentBlock'] = None

    # Motivation
    motivation: str = ""                 # Why this intent exists
    urgency: float = 0.5                 # 0.0 to 1.0

    # Domain classification
    domain: str = "general"              # "EconomicTransaction", "SocialRelation", etc.

    def to_tensor(self) -> np.ndarray:
        """Convert intent to tensor."""
        features = [
            self.confidence,
            self.urgency,
            len(self.preconditions),
            len(self.expected_effects),
            1.0 if self.about_agent else 0.0,
            1.0 if self.nested_intent else 0.0,
        ]
        return np.array(features, dtype=np.float32)

    def to_natural_language(self) -> str:
        """Generate natural language description."""
        if self.about_agent and self.nested_intent:
            nested_nl = self.nested_intent.to_natural_language()
            return f"I believe {self.about_agent} intends: {nested_nl}"
        elif self.target_agent:
            return f"I intend to {self.action_type} regarding {self.target_agent}"
        else:
            return f"I intend to {self.action_type} to achieve: {self.goal}"


@dataclass
class MemoryBlock(CognitiveBlock):
    """
    A compressed archetype - the end product of cognitive compression.

    Memories are highly compressed representations that capture the
    essential pattern of experiences. They serve as priors for
    future perception and reasoning.
    """
    block_type: BlockType = field(default=BlockType.MEMORY, init=False)

    # Memory content
    archetype_label: str = ""            # High-level category
    compressed_features: Dict[str, float] = field(default_factory=dict)

    # Compression metadata
    original_block_ids: List[str] = field(default_factory=list)
    compression_ratio: float = 1.0       # How much compression occurred
    information_loss: float = 0.0        # Estimated information loss

    # Retrieval cues
    retrieval_cues: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0       # -1.0 to 1.0
    salience: float = 0.5                # How easily retrieved

    # Episodic vs semantic
    is_episodic: bool = False            # Specific event vs general knowledge
    temporal_context: Optional[str] = None

    def to_tensor(self) -> np.ndarray:
        """Convert memory to tensor."""
        features = list(self.compressed_features.values())
        features.extend([
            self.confidence,
            self.compression_ratio,
            self.information_loss,
            self.emotional_valence,
            self.salience,
        ])
        return np.array(features, dtype=np.float32)

    def to_natural_language(self) -> str:
        """Generate natural language description."""
        if self.is_episodic:
            return f"I remember when {self.archetype_label} happened"
        else:
            return f"I know that {self.archetype_label} is generally true"


@dataclass
class RecursiveBelief:
    """
    A recursive belief structure for N-th order Theory of Mind.

    Implements B_a(B_b(B_a(p))) style recursive beliefs where
    each level can be explicitly traced and reasoned about.
    """
    holder: str                          # Who holds this belief
    about: Optional[str] = None          # Who the belief is about (if ToM)
    content: Union[str, 'RecursiveBelief'] = ""  # Either proposition or nested belief
    confidence: float = 1.0
    order: int = 1                       # Belief order

    # Confidence decay with nesting (realistic uncertainty)
    CONFIDENCE_DECAY = 0.7               # Multiply by this for each level

    def get_effective_confidence(self) -> float:
        """Get confidence accounting for recursive decay."""
        return self.confidence * (self.CONFIDENCE_DECAY ** (self.order - 1))

    def to_notation(self) -> str:
        """Convert to formal notation: B_a(B_b(p))"""
        if isinstance(self.content, str):
            if self.about:
                return f"B_{self.holder}(about_{self.about}: {self.content})"
            return f"B_{self.holder}({self.content})"
        else:
            inner = self.content.to_notation()
            return f"B_{self.holder}({inner})"

    def flatten(self) -> List[tuple]:
        """Flatten recursive structure into list of (holder, about, proposition) tuples."""
        if isinstance(self.content, str):
            return [(self.holder, self.about, self.content)]
        else:
            inner = self.content.flatten()
            return [(self.holder, self.about, "nested")] + inner


@dataclass
class SimulationState(CognitiveBlock):
    """
    State of a recursive simulation being run by an agent.

    When Agent A simulates Agent B, this captures the state
    of that simulation, including B's simulated beliefs about A.
    """
    block_type: BlockType = field(default=BlockType.SIMULATION, init=False)

    # Simulation identity
    simulating_agent: str = ""           # Agent running the simulation
    simulated_agent: str = ""            # Agent being simulated

    # Simulated world state
    simulated_world_state: Dict[str, Any] = field(default_factory=dict)
    simulated_timestep: int = 0

    # Simulated agent's cognitive state
    simulated_beliefs: List[BeliefBlock] = field(default_factory=list)
    simulated_intents: List[IntentBlock] = field(default_factory=list)

    # Simulation depth
    recursion_depth: int = 1             # How deep we are
    max_depth: int = 3                   # Maximum allowed depth

    # Results
    predicted_action: Optional[str] = None
    prediction_confidence: float = 0.0

    def to_tensor(self) -> np.ndarray:
        """Convert simulation state to tensor."""
        return np.array([
            self.simulated_timestep,
            self.recursion_depth,
            len(self.simulated_beliefs),
            len(self.simulated_intents),
            self.prediction_confidence,
        ], dtype=np.float32)

    def to_natural_language(self) -> str:
        """Generate natural language description."""
        return (
            f"I am simulating {self.simulated_agent} at depth {self.recursion_depth}. "
            f"Predicted action: {self.predicted_action} "
            f"(confidence: {self.prediction_confidence:.0%})"
        )


# ==================== Block Operations ====================

@dataclass
class BlockTransition:
    """
    A transition between cognitive block types.

    Captures the type-shifting process that IS reasoning:
    Percept -> Hypothesis -> Belief -> Memory
    """
    source_block: CognitiveBlock
    target_block: CognitiveBlock
    transition_type: str                 # "elaboration", "compression", "revision", etc.
    transition_reason: str               # Why this transition happened
    information_delta: float = 0.0       # Information gained/lost

    def __post_init__(self):
        # Validate transition
        valid_transitions = {
            BlockType.PERCEPT: [BlockType.HYPOTHESIS, BlockType.BELIEF],
            BlockType.HYPOTHESIS: [BlockType.BELIEF, BlockType.HYPOTHESIS],
            BlockType.BELIEF: [BlockType.BELIEF, BlockType.MEMORY, BlockType.INTENT],
            BlockType.MEMORY: [BlockType.BELIEF, BlockType.HYPOTHESIS],
            BlockType.INTENT: [BlockType.INTENT, BlockType.BELIEF],
        }

        source_type = self.source_block.block_type
        target_type = self.target_block.block_type

        if target_type not in valid_transitions.get(source_type, []):
            raise ValueError(
                f"Invalid transition: {source_type.name} -> {target_type.name}"
            )


def compress_to_memory(
    blocks: List[CognitiveBlock],
    archetype_label: str,
    compression_method: str = "averaging"
) -> MemoryBlock:
    """
    Compress multiple cognitive blocks into a single memory archetype.

    This implements the cognitive compression that creates stable
    representations from transient experiences.

    Args:
        blocks: List of blocks to compress
        archetype_label: Label for the resulting archetype
        compression_method: "averaging", "prototype", or "exemplar"

    Returns:
        A compressed MemoryBlock
    """
    if not blocks:
        raise ValueError("Cannot compress empty block list")

    # Collect all features
    all_tensors = [b.to_tensor() for b in blocks]

    # Compute compressed features
    if compression_method == "averaging":
        mean_tensor = np.mean(all_tensors, axis=0)
        compressed_features = {f"dim_{i}": float(v) for i, v in enumerate(mean_tensor)}
    elif compression_method == "prototype":
        # Use median for prototype (more robust)
        median_tensor = np.median(all_tensors, axis=0)
        compressed_features = {f"dim_{i}": float(v) for i, v in enumerate(median_tensor)}
    else:
        # Exemplar: just store indices
        compressed_features = {"n_exemplars": float(len(blocks))}

    # Calculate information loss
    if len(all_tensors) > 1:
        variance = np.var(all_tensors, axis=0).mean()
        information_loss = float(variance)
    else:
        information_loss = 0.0

    # Compute emotional valence from source blocks
    valences = []
    for b in blocks:
        if hasattr(b, 'emotional_valence'):
            valences.append(b.emotional_valence)
    emotional_valence = np.mean(valences) if valences else 0.0

    return MemoryBlock(
        archetype_label=archetype_label,
        compressed_features=compressed_features,
        original_block_ids=[b.block_id for b in blocks],
        compression_ratio=len(blocks),
        information_loss=information_loss,
        confidence=np.mean([b.confidence for b in blocks]),
        emotional_valence=float(emotional_valence),
    )


def expand_from_memory(
    memory: MemoryBlock,
    context: Dict[str, Any]
) -> List[HypothesisBlock]:
    """
    Expand a memory into hypotheses for the current context.

    Memories serve as priors that generate expectations (hypotheses)
    about what might be true in the current situation.

    Args:
        memory: The memory to expand
        context: Current situational context

    Returns:
        List of hypothesis blocks generated from memory
    """
    hypotheses = []

    # Generate primary hypothesis from archetype
    primary = HypothesisBlock(
        proposition=f"Current situation matches {memory.archetype_label}",
        subject="current_situation",
        predicate=f"is_like_{memory.archetype_label}",
        confidence=memory.confidence * 0.8,  # Discount for memory uncertainty
        source_percepts=[memory.block_id],
    )
    hypotheses.append(primary)

    # Generate predictions based on compressed features
    for feature_name, feature_value in memory.compressed_features.items():
        if feature_value > 0.7:
            pred = HypothesisBlock(
                proposition=f"Expect high {feature_name}",
                subject="current_situation",
                predicate=f"has_high_{feature_name}",
                confidence=memory.confidence * feature_value,
                source_percepts=[memory.block_id],
            )
            hypotheses.append(pred)

    return hypotheses


def create_recursive_belief(
    holder: str,
    belief_chain: List[str],
    proposition: str,
    base_confidence: float = 0.9
) -> BeliefBlock:
    """
    Create a nested belief structure from a chain of agents.

    Example: create_recursive_belief("Alice", ["Bob", "Carol"], "door is open")
    Creates: Alice believes Bob believes Carol believes door is open

    Args:
        holder: The agent at the top level
        belief_chain: List of agents in the belief chain
        proposition: The base proposition
        base_confidence: Starting confidence

    Returns:
        A nested BeliefBlock structure
    """
    decay = 0.7  # Confidence decay per level

    if not belief_chain:
        # Base case: first-order belief
        return BeliefBlock(
            proposition=proposition,
            subject="",
            predicate=proposition,
            confidence=base_confidence,
            belief_order=1,
        )

    # Recursive case: build from inside out
    inner = create_recursive_belief(
        holder=belief_chain[0],
        belief_chain=belief_chain[1:],
        proposition=proposition,
        base_confidence=base_confidence * decay,
    )

    return BeliefBlock(
        proposition=f"{holder} believes about {belief_chain[0]}",
        subject=belief_chain[0],
        predicate="believes",
        about_agent=belief_chain[0],
        nested_belief=inner,
        confidence=base_confidence,
        belief_order=inner.belief_order + 1,
    )
