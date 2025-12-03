"""
Semiotic Knowledge Graph Schemas

Defines the fundamental data structures for the Indra's Net semantic substrate.
Every entity exists as a SemanticNode with typed edges to other nodes,
enabling the "background hum" of hyperlinked meaning that saturates cognition.

The schemas implement:
1. Semantic Prototype/Stereotype pairs (latent meanings collapsed by context)
2. Multi-dimensional taxonomy positioning (80 dimensions)
3. Activation spreading for concept retrieval
4. Context-dependent meaning collapse
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime
import numpy as np


class NodeType(Enum):
    """Types of nodes in the semantic web."""
    # Physical entities (grounded in Godot)
    OBJECT = auto()          # Physical objects (Chair, Hammer, Door)
    AGENT = auto()           # Embodied agents
    LOCATION = auto()        # Spatial locations (Room, Building)

    # Conceptual entities
    CONCEPT = auto()         # Abstract concepts (Justice, Trust)
    ARCHETYPE = auto()       # Jungian archetypes (Rebel, Sage, Hero)
    ROLE = auto()            # Social roles (Judge, Teacher, Parent)
    NORM = auto()            # Social norms and expectations
    RITUAL = auto()          # Ritualized behavioral sequences

    # Institutional entities
    INSTITUTION = auto()     # Formal institutions (Court, Church, Market)
    POWER_STRUCTURE = auto() # Power relationships and hierarchies

    # Aesthetic entities
    AESTHETIC_MODE = auto()  # Aesthetic categories (Gothic, Minimalist)
    EMOTIONAL_VALENCE = auto()  # Emotional associations

    # Action/Event entities
    ACTION = auto()          # Actions that can be performed
    EVENT = auto()           # Events that can occur
    UTTERANCE = auto()       # Speech acts and communication


class EdgeType(Enum):
    """Types of edges connecting semantic nodes."""
    # Taxonomic relations
    IS_A = auto()            # Hypernym/hyponym (Chair IS_A Furniture)
    PART_OF = auto()         # Meronymy (Leg PART_OF Chair)

    # Semantic relations
    ASSOCIATED_WITH = auto() # General semantic association
    STEREOTYPE_OF = auto()   # Links to stereotype activation
    PROTOTYPE_OF = auto()    # Links to prototype representation

    # Institutional relations
    OCCURS_IN = auto()       # Action/event occurs in institution
    ENFORCED_BY = auto()     # Norm enforced by institution
    ROLE_IN = auto()         # Role exists in institution

    # Power relations
    POWER_OVER = auto()      # Power differential
    DEFERS_TO = auto()       # Deference relationship

    # Causal/functional relations
    AFFORDS = auto()         # Object affords action (Chair AFFORDS Sitting)
    CAUSES = auto()          # Causal relationship
    PRECEDES = auto()        # Temporal precedence

    # Aesthetic relations
    EVOKES = auto()          # Evokes emotional/aesthetic response
    CONTRASTS_WITH = auto()  # Aesthetic contrast

    # Cognitive relations
    REMINDS_OF = auto()      # Associative memory link
    CONFLICTS_WITH = auto()  # Cognitive dissonance potential


class ActivationMode(Enum):
    """Modes of activation spreading through the semantic network."""
    SPREADING = auto()       # Standard spreading activation
    INHIBITORY = auto()      # Inhibits connected nodes
    PRIMING = auto()         # Primes for faster future activation
    CONTEXTUAL = auto()      # Context-dependent activation


@dataclass
class TaxonomyDimension:
    """
    A single dimension in the 80-Dimension Taxonomy.

    Each dimension represents a continuous spectrum upon which
    concepts and entities can be positioned.
    """
    id: int                          # Unique dimension ID (1-80)
    name: str                        # Human-readable name
    layer: str                       # Which taxonomy layer (Mundane, Institutional, Aesthetic)
    sublayer: str                    # Specific sublayer

    # Dimension semantics
    low_anchor: str                  # Low end description (e.g., "Chaotic")
    high_anchor: str                 # High end description (e.g., "Orderly")

    # Theoretical grounding
    theoretical_basis: str           # Reference to theoretical foundation
    psychological_correlates: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)


@dataclass
class ConceptualDomain:
    """
    A conceptual domain grouping related taxonomy dimensions.

    Domains represent coherent areas of human experience
    (e.g., Social Relations, Physical Space, Temporal Flow).
    """
    id: str                          # Unique domain ID
    name: str                        # Human-readable name
    description: str                 # What this domain covers

    dimensions: List[int] = field(default_factory=list)  # Dimension IDs in this domain
    parent_domain: Optional[str] = None
    child_domains: List[str] = field(default_factory=list)

    # Cross-domain mappings (Conceptual Metaphor Theory)
    source_mappings: Dict[str, str] = field(default_factory=dict)  # Maps to target domains

    def __hash__(self):
        return hash(self.id)


@dataclass
class SemanticNode:
    """
    A node in the Indra's Net semantic web.

    Every entity in the simulation exists as a SemanticNode with:
    - A position in the 80-dimension taxonomy space
    - Typed edges to other nodes
    - Activation state that spreads through the network
    - Context-dependent meaning collapse capabilities

    The node is the atomic unit of meaning in the semiotic substrate.
    """
    id: str                          # Unique identifier (e.g., "obj_chair_402")
    node_type: NodeType
    name: str                        # Human-readable label

    # Physical grounding (for OBJECT, AGENT, LOCATION types)
    godot_id: Optional[int] = None   # Links to Godot physics engine
    position_3d: Optional[Tuple[float, float, float]] = None

    # Taxonomy positioning (80-dimensional vector)
    taxonomy_position: np.ndarray = field(default_factory=lambda: np.zeros(80))

    # Prototype/Stereotype representation
    prototype_features: Dict[str, float] = field(default_factory=dict)
    stereotype_associations: Dict[str, float] = field(default_factory=dict)

    # Activation state
    activation_level: float = 0.0    # Current activation (0.0-1.0)
    activation_history: List[Tuple[datetime, float]] = field(default_factory=list)

    # Conceptual content
    properties: Dict[str, Any] = field(default_factory=dict)
    affordances: List[str] = field(default_factory=list)  # Actions this enables

    # Normative associations
    associated_norms: List[str] = field(default_factory=list)
    associated_roles: List[str] = field(default_factory=list)

    # Emotional/aesthetic valence
    emotional_valence: float = 0.0   # -1.0 (negative) to 1.0 (positive)
    arousal_level: float = 0.5       # 0.0 (calm) to 1.0 (exciting)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, SemanticNode):
            return self.id == other.id
        return False

    def get_taxonomy_slice(self, dimensions: List[int]) -> np.ndarray:
        """Extract specific dimensions from taxonomy position."""
        return self.taxonomy_position[dimensions]

    def update_activation(self, new_activation: float, timestamp: Optional[datetime] = None):
        """Update activation level and log history."""
        self.activation_level = np.clip(new_activation, 0.0, 1.0)
        self.activation_history.append(
            (timestamp or datetime.now(), self.activation_level)
        )
        self.last_accessed = timestamp or datetime.now()
        self.access_count += 1


@dataclass
class SemanticEdge:
    """
    A typed edge connecting two SemanticNodes.

    Edges carry semantic weight that influences activation spreading
    and enable typed traversal through the knowledge graph.
    """
    source_id: str                   # Source node ID
    target_id: str                   # Target node ID
    edge_type: EdgeType

    # Edge properties
    weight: float = 1.0              # Semantic weight (0.0-1.0)
    bidirectional: bool = False      # Whether edge works both ways

    # Context sensitivity
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    # Only active when conditions met (e.g., {"institution": "courtroom"})

    # Theoretical grounding
    theoretical_basis: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash((self.source_id, self.target_id, self.edge_type))

    @property
    def id(self) -> str:
        """Unique edge identifier."""
        return f"{self.source_id}--{self.edge_type.name}-->{self.target_id}"


@dataclass
class ActivationContext:
    """
    Context for activation spreading through the semantic network.

    The context collapses superposed meanings into definite interpretations.
    A "chair" in a "courtroom" activates different associations than
    a "chair" in a "living room".
    """
    # Spatial context
    current_location: Optional[str] = None
    perceiving_agent: Optional[str] = None

    # Institutional context
    active_institution: Optional[str] = None
    active_norms: List[str] = field(default_factory=list)
    active_roles: List[str] = field(default_factory=list)

    # Temporal context
    timestamp: datetime = field(default_factory=datetime.now)
    temporal_phase: str = "present"  # past, present, future, hypothetical

    # Cognitive context
    current_goal: Optional[str] = None
    emotional_state: Dict[str, float] = field(default_factory=dict)
    attention_focus: List[str] = field(default_factory=list)

    # Activation parameters
    spreading_decay: float = 0.7     # Decay factor per hop
    max_spread_depth: int = 3        # Maximum traversal depth
    activation_threshold: float = 0.1  # Minimum activation to propagate

    def matches_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Check if this context matches edge activation conditions."""
        for key, value in conditions.items():
            if key == "institution" and self.active_institution != value:
                return False
            if key == "location" and self.current_location != value:
                return False
            if key == "role" and value not in self.active_roles:
                return False
            if key == "norm" and value not in self.active_norms:
                return False
        return True


@dataclass
class SemanticActivation:
    """
    Result of a semantic query/activation through Indra's Net.

    When an agent perceives something, this represents the full
    "background hum" of activated associations, norms, and archetypes.
    """
    trigger_node: str                # Node that triggered activation
    context: ActivationContext       # Context that shaped activation

    # Activated content
    activated_nodes: Dict[str, float] = field(default_factory=dict)
    # node_id -> activation_level

    activated_edges: List[SemanticEdge] = field(default_factory=list)
    # Edges traversed during activation

    # Semantic dimensions activated
    taxonomy_dimensions_activated: List[int] = field(default_factory=list)

    # Emergent interpretations
    collapsed_meaning: Optional[str] = None  # The definite interpretation
    alternative_meanings: List[Tuple[str, float]] = field(default_factory=list)
    # (meaning, probability) pairs

    # Normative implications
    activated_norms: List[str] = field(default_factory=list)
    activated_roles: List[str] = field(default_factory=list)
    activated_archetypes: List[str] = field(default_factory=list)

    # Affective coloring
    aggregate_valence: float = 0.0
    aggregate_arousal: float = 0.5

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    traversal_depth: int = 0
    computation_time_ms: float = 0.0

    def get_top_activations(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the N most strongly activated nodes."""
        sorted_nodes = sorted(
            self.activated_nodes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_nodes[:n]

    def get_nodes_by_type(self, node_type: NodeType) -> Dict[str, float]:
        """Filter activated nodes by type."""
        # Note: Requires access to the graph to check types
        # This would be populated by the query engine
        return {}


@dataclass
class StereotypeDimension:
    """
    A dimension of stereotype content based on the Stereotype Content Model.

    Following Fiske et al., stereotypes are structured along
    Warmth (intentions) and Competence (capability) dimensions.
    """
    warmth: float = 0.5              # 0.0 (cold) to 1.0 (warm)
    competence: float = 0.5          # 0.0 (incompetent) to 1.0 (competent)

    # Extended dimensions
    status: float = 0.5              # Perceived social status
    competition: float = 0.5         # Perceived as competing for resources

    # Emotional predictions (based on BIAS map)
    predicted_emotions: Dict[str, float] = field(default_factory=dict)
    # e.g., {"admiration": 0.8, "contempt": 0.1}

    def get_quadrant(self) -> str:
        """Return the SCM quadrant (high/low warmth × high/low competence)."""
        warmth_level = "high" if self.warmth > 0.5 else "low"
        competence_level = "high" if self.competence > 0.5 else "low"
        return f"{warmth_level}_warmth_{competence_level}_competence"


@dataclass
class PrototypeRepresentation:
    """
    A prototype representation following Rosch's prototype theory.

    Prototypes are the "best examples" of a category, represented
    as weighted feature bundles.
    """
    category: str                    # The category this is a prototype of

    # Feature weights (feature -> typicality weight)
    features: Dict[str, float] = field(default_factory=dict)

    # Family resemblance structure
    central_tendency: Dict[str, float] = field(default_factory=dict)
    variance: Dict[str, float] = field(default_factory=dict)

    # Exemplar instances (for exemplar-based reasoning)
    exemplar_ids: List[str] = field(default_factory=list)

    def similarity_to(self, instance_features: Dict[str, float]) -> float:
        """Compute similarity of an instance to this prototype."""
        if not self.features or not instance_features:
            return 0.0

        common_features = set(self.features.keys()) & set(instance_features.keys())
        if not common_features:
            return 0.0

        similarity = 0.0
        for feature in common_features:
            # Weighted feature match
            proto_weight = self.features[feature]
            instance_weight = instance_features[feature]
            similarity += proto_weight * instance_weight

        return similarity / len(common_features)
