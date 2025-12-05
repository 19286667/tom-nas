"""
Symbol Grounding - Connecting Physics to Meaning

Implements Harnad's Symbol Grounding by linking physical objects in
Godot to their semantic representations in Indra's Net.

The Key Insight:
A "Chair" is not just an abstract concept - it is inextricably linked
to the specific Rigidbody3D with ID_992 in the Godot scene. An agent
cannot reason about a hammer it hasn't perceived or internalized.

This module:
1. Maps Godot entity IDs to semantic node IDs
2. Extracts visual/physical features for prototype matching
3. Creates grounded percepts that anchor cognition in physics

Theoretical Foundation:
- Symbol Grounding Problem (Harnad, 1990)
- Embodied Cognition (Varela, Thompson, Rosch)
- Affordance Theory (Gibson)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .protocol import EntityUpdate, Vector3


@dataclass
class GroundingContext:
    """
    Context for grounding symbols - affects interpretation.

    The same physical object may ground to different meanings
    depending on context (a chair in a courtroom vs. at home).
    """

    # Spatial context
    observer_position: Vector3 = field(default_factory=Vector3)
    observer_orientation: Vector3 = field(default_factory=Vector3)

    # Institutional context
    current_institution: Optional[str] = None
    active_norms: List[str] = field(default_factory=list)

    # Temporal context
    simulation_time: float = 0.0
    time_of_day: float = 12.0

    # Attention context
    attention_focus: Optional[int] = None  # Godot ID in focus
    recent_percepts: List[int] = field(default_factory=list)

    # Emotional context (affects perception)
    emotional_valence: float = 0.0
    arousal_level: float = 0.5


@dataclass
class VisualFeatures:
    """
    Visual features extracted from a Godot entity.

    These features enable prototype matching and categorization.
    """

    # Size features
    volume: float = 1.0  # Bounding box volume
    height: float = 1.0
    width: float = 1.0
    depth: float = 1.0

    # Color features (dominant color)
    hue: float = 0.0
    saturation: float = 0.5
    brightness: float = 0.5

    # Texture features
    roughness: float = 0.5
    metallic: float = 0.0

    # Shape features
    is_elongated: bool = False
    is_flat: bool = False
    is_spherical: bool = False
    has_handle: bool = False
    has_legs: bool = False

    # Material inference
    inferred_material: str = "unknown"  # wood, metal, fabric, etc.

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array(
            [
                self.volume / 10.0,  # Normalize
                self.height / 5.0,
                self.width / 5.0,
                self.depth / 5.0,
                self.hue,
                self.saturation,
                self.brightness,
                self.roughness,
                self.metallic,
                1.0 if self.is_elongated else 0.0,
                1.0 if self.is_flat else 0.0,
                1.0 if self.is_spherical else 0.0,
                1.0 if self.has_handle else 0.0,
                1.0 if self.has_legs else 0.0,
            ],
            dtype=np.float32,
        )


@dataclass
class GroundedSymbol:
    """
    A symbol grounded in physical reality.

    This is the fundamental unit linking Godot physics to meaning.
    """

    # Physical identity
    godot_id: int  # Godot node ID
    entity_type: str  # object, agent, location
    name: str  # Entity name

    # Semantic identity
    semantic_node_id: Optional[str] = None  # ID in Indra's Net
    category: str = "unknown"  # Inferred category

    # Physical state
    position: Vector3 = field(default_factory=Vector3)
    last_position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)

    # Visual features
    visual_features: VisualFeatures = field(default_factory=VisualFeatures)

    # Affordances (grounded in physics)
    physical_affordances: List[str] = field(default_factory=list)
    # e.g., ["can_sit_on", "can_pick_up", "can_open"]

    # Semantic associations (from Indra's Net)
    semantic_associations: Dict[str, float] = field(default_factory=dict)
    # e.g., {"furniture": 0.9, "status_object": 0.3}

    # Prototype match
    prototype_match: Optional[str] = None
    prototype_similarity: float = 0.0

    # Grounding metadata
    first_perceived: datetime = field(default_factory=datetime.now)
    last_perceived: datetime = field(default_factory=datetime.now)
    perception_count: int = 0

    # Stability (has it moved? changed?)
    is_stable: bool = True
    change_history: List[Dict[str, Any]] = field(default_factory=list)

    def update_position(self, new_position: Vector3):
        """Update position and track movement."""
        self.last_position = self.position
        self.position = new_position

        # Check if moved significantly
        distance = self.position.distance_to(self.last_position)
        if distance > 0.1:
            self.is_stable = False
            self.change_history.append(
                {
                    "type": "moved",
                    "distance": distance,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            self.is_stable = True

    def compute_velocity(self, dt: float):
        """Compute velocity from position change."""
        if dt > 0:
            dx = self.position.x - self.last_position.x
            dy = self.position.y - self.last_position.y
            dz = self.position.z - self.last_position.z
            self.velocity = Vector3(dx / dt, dy / dt, dz / dt)


class SymbolGrounder:
    """
    Main symbol grounding system.

    Maintains the mapping between Godot physical entities and
    semantic nodes in Indra's Net, enabling grounded cognition.
    """

    def __init__(self, knowledge_base=None):
        """
        Initialize the symbol grounder.

        Args:
            knowledge_base: The Indra's Net knowledge graph (optional)
        """
        self.knowledge_base = knowledge_base

        # Mapping: godot_id -> GroundedSymbol
        self.grounded_symbols: Dict[int, GroundedSymbol] = {}

        # Reverse mapping: semantic_node_id -> godot_id
        self.semantic_to_physical: Dict[str, int] = {}

        # Category prototypes (learned or predefined)
        self.category_prototypes: Dict[str, np.ndarray] = self._init_prototypes()

        # Affordance rules (what physical features enable what actions)
        self.affordance_rules: Dict[str, callable] = self._init_affordance_rules()

        # Statistics
        self.total_groundings = 0
        self.grounding_updates = 0

    def _init_prototypes(self) -> Dict[str, np.ndarray]:
        """Initialize category prototypes for matching."""
        # These would be learned or derived from training data
        # Simplified version with hand-coded prototypes
        prototypes = {
            "chair": np.array(
                [
                    0.05,  # volume (small-medium)
                    0.15,  # height (medium)
                    0.1,  # width
                    0.1,  # depth
                    0.1,  # hue (brown-ish)
                    0.4,  # saturation
                    0.5,  # brightness
                    0.6,  # roughness (wood)
                    0.1,  # metallic
                    0.0,  # not elongated
                    0.0,  # not flat
                    0.0,  # not spherical
                    0.0,  # no handle
                    1.0,  # has legs
                ],
                dtype=np.float32,
            ),
            "table": np.array(
                [
                    0.15,  # volume (larger)
                    0.15,  # height
                    0.3,  # width (wide)
                    0.2,  # depth
                    0.1,  # hue
                    0.3,  # saturation
                    0.5,  # brightness
                    0.5,  # roughness
                    0.1,  # metallic
                    0.0,  # not elongated
                    1.0,  # flat (surface)
                    0.0,  # not spherical
                    0.0,  # no handle
                    1.0,  # has legs
                ],
                dtype=np.float32,
            ),
            "hammer": np.array(
                [
                    0.01,  # volume (small)
                    0.1,  # height
                    0.02,  # width (narrow)
                    0.02,  # depth
                    0.0,  # hue (neutral)
                    0.3,  # saturation
                    0.4,  # brightness
                    0.4,  # roughness
                    0.7,  # metallic (head)
                    1.0,  # elongated
                    0.0,  # not flat
                    0.0,  # not spherical
                    1.0,  # has handle
                    0.0,  # no legs
                ],
                dtype=np.float32,
            ),
            "door": np.array(
                [
                    0.1,  # volume
                    0.4,  # height (tall)
                    0.2,  # width
                    0.01,  # depth (thin)
                    0.1,  # hue
                    0.3,  # saturation
                    0.5,  # brightness
                    0.5,  # roughness
                    0.2,  # metallic (handle)
                    0.0,  # not elongated
                    1.0,  # flat
                    0.0,  # not spherical
                    1.0,  # has handle
                    0.0,  # no legs
                ],
                dtype=np.float32,
            ),
            "person": np.array(
                [
                    0.1,  # volume
                    0.35,  # height (tall)
                    0.1,  # width
                    0.1,  # depth
                    0.15,  # hue (skin tones)
                    0.4,  # saturation
                    0.6,  # brightness
                    0.6,  # roughness
                    0.0,  # metallic
                    1.0,  # elongated (standing)
                    0.0,  # not flat
                    0.0,  # not spherical
                    0.0,  # no handle
                    1.0,  # has legs
                ],
                dtype=np.float32,
            ),
        }
        return prototypes

    def _init_affordance_rules(self) -> Dict[str, callable]:
        """Initialize rules for inferring affordances from features."""

        def can_sit_on(features: VisualFeatures) -> bool:
            return features.height < 1.5 and features.has_legs and not features.is_elongated

        def can_pick_up(features: VisualFeatures) -> bool:
            return features.volume < 0.1 and features.height < 1.0

        def can_open(features: VisualFeatures) -> bool:
            return features.has_handle

        def can_strike_with(features: VisualFeatures) -> bool:
            return features.is_elongated and features.has_handle

        def can_stand_on(features: VisualFeatures) -> bool:
            return features.is_flat and features.height < 0.5

        return {
            "can_sit_on": can_sit_on,
            "can_pick_up": can_pick_up,
            "can_open": can_open,
            "can_strike_with": can_strike_with,
            "can_stand_on": can_stand_on,
        }

    def ground_entity(self, entity: EntityUpdate, context: Optional[GroundingContext] = None) -> GroundedSymbol:
        """
        Ground a Godot entity as a semantic symbol.

        This is the core grounding operation: taking raw physics
        and producing grounded meaning.

        Args:
            entity: Entity update from Godot
            context: Grounding context (affects interpretation)

        Returns:
            GroundedSymbol linking physics to meaning
        """
        godot_id = entity.godot_id

        # Check if already grounded
        if godot_id in self.grounded_symbols:
            symbol = self.grounded_symbols[godot_id]
            self._update_grounded_symbol(symbol, entity)
            return symbol

        # Create new grounded symbol
        symbol = GroundedSymbol(
            godot_id=godot_id,
            entity_type=entity.entity_type,
            name=entity.name,
            position=entity.position,
        )

        # Extract visual features
        symbol.visual_features = self._extract_visual_features(entity)

        # Match to category prototype
        category, similarity = self._match_prototype(symbol.visual_features)
        symbol.category = category
        symbol.prototype_match = category
        symbol.prototype_similarity = similarity

        # Infer affordances from features
        symbol.physical_affordances = self._infer_affordances(symbol.visual_features)

        # Add affordances from Godot metadata
        symbol.physical_affordances.extend(entity.affordances)
        symbol.physical_affordances = list(set(symbol.physical_affordances))

        # Link to semantic node in knowledge base
        if self.knowledge_base:
            semantic_id = self._find_semantic_node(symbol, context)
            symbol.semantic_node_id = semantic_id
            if semantic_id:
                self.semantic_to_physical[semantic_id] = godot_id

                # Get semantic associations
                symbol.semantic_associations = self._get_semantic_associations(semantic_id, context)

        # Store grounded symbol
        self.grounded_symbols[godot_id] = symbol
        self.total_groundings += 1

        return symbol

    def _extract_visual_features(self, entity: EntityUpdate) -> VisualFeatures:
        """Extract visual features from entity update."""
        features = VisualFeatures()

        # Size from scale
        features.height = entity.scale.y
        features.width = entity.scale.x
        features.depth = entity.scale.z
        features.volume = features.height * features.width * features.depth

        # Color if available
        if entity.color:
            # Convert RGBA to HSV-like features
            r, g, b, a = entity.color
            features.brightness = (r + g + b) / 3.0
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            features.saturation = (max_c - min_c) / max_c if max_c > 0 else 0
            # Simplified hue
            if max_c == min_c:
                features.hue = 0
            elif max_c == r:
                features.hue = ((g - b) / (max_c - min_c)) % 6 / 6
            elif max_c == g:
                features.hue = ((b - r) / (max_c - min_c) + 2) / 6
            else:
                features.hue = ((r - g) / (max_c - min_c) + 4) / 6

        # Infer shape properties
        aspect_ratio = features.height / max(features.width, 0.01)
        features.is_elongated = aspect_ratio > 2.0
        features.is_flat = features.height < 0.3 * features.width

        # Check semantic tags for hints
        tags = entity.semantic_tags
        features.has_handle = any("handle" in t.lower() for t in tags)
        features.has_legs = any("leg" in t.lower() for t in tags)

        # Material inference from tags
        for tag in tags:
            if "wood" in tag.lower():
                features.inferred_material = "wood"
                features.roughness = 0.6
            elif "metal" in tag.lower():
                features.inferred_material = "metal"
                features.metallic = 0.8
            elif "fabric" in tag.lower():
                features.inferred_material = "fabric"
                features.roughness = 0.7

        return features

    def _match_prototype(self, features: VisualFeatures) -> Tuple[str, float]:
        """Match features to category prototypes."""
        feature_vector = features.to_vector()

        best_category = "unknown"
        best_similarity = 0.0

        for category, prototype in self.category_prototypes.items():
            # Cosine similarity
            dot = np.dot(feature_vector, prototype)
            norm_f = np.linalg.norm(feature_vector)
            norm_p = np.linalg.norm(prototype)

            if norm_f > 0 and norm_p > 0:
                similarity = dot / (norm_f * norm_p)
            else:
                similarity = 0.0

            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category

        return best_category, float(best_similarity)

    def _infer_affordances(self, features: VisualFeatures) -> List[str]:
        """Infer affordances from visual features."""
        affordances = []

        for affordance, rule in self.affordance_rules.items():
            if rule(features):
                affordances.append(affordance)

        return affordances

    def _find_semantic_node(self, symbol: GroundedSymbol, context: Optional[GroundingContext]) -> Optional[str]:
        """Find matching semantic node in knowledge base."""
        if not self.knowledge_base:
            return None

        # Try exact name match first
        node = self.knowledge_base.get_node(f"obj_{symbol.name.lower()}")
        if node:
            return node.id

        # Try category match
        node = self.knowledge_base.get_node(f"obj_{symbol.category}")
        if node:
            return node.id

        # Would do more sophisticated matching in full implementation
        return None

    def _get_semantic_associations(self, semantic_id: str, context: Optional[GroundingContext]) -> Dict[str, float]:
        """Get semantic associations from knowledge base."""
        if not self.knowledge_base:
            return {}

        associations = {}

        # Get neighbors in semantic graph
        neighbors = self.knowledge_base.get_neighbors(semantic_id)
        for neighbor_id, edge_type, weight in neighbors:
            associations[neighbor_id] = weight

        return associations

    def _update_grounded_symbol(self, symbol: GroundedSymbol, entity: EntityUpdate):
        """Update an existing grounded symbol with new data."""
        # Update position
        symbol.update_position(entity.position)

        # Update perception metadata
        symbol.last_perceived = datetime.now()
        symbol.perception_count += 1

        self.grounding_updates += 1

    def get_grounded_symbol(self, godot_id: int) -> Optional[GroundedSymbol]:
        """Get grounded symbol by Godot ID."""
        return self.grounded_symbols.get(godot_id)

    def get_by_semantic_id(self, semantic_id: str) -> Optional[GroundedSymbol]:
        """Get grounded symbol by semantic node ID."""
        godot_id = self.semantic_to_physical.get(semantic_id)
        if godot_id:
            return self.grounded_symbols.get(godot_id)
        return None

    def get_symbols_in_category(self, category: str) -> List[GroundedSymbol]:
        """Get all grounded symbols of a category."""
        return [s for s in self.grounded_symbols.values() if s.category == category]

    def get_symbols_with_affordance(self, affordance: str) -> List[GroundedSymbol]:
        """Get all grounded symbols with a specific affordance."""
        return [s for s in self.grounded_symbols.values() if affordance in s.physical_affordances]

    def get_nearby_symbols(self, position: Vector3, radius: float) -> List[GroundedSymbol]:
        """Get grounded symbols within radius of a position."""
        nearby = []
        for symbol in self.grounded_symbols.values():
            distance = symbol.position.distance_to(position)
            if distance <= radius:
                nearby.append(symbol)
        return nearby

    def clear_symbol(self, godot_id: int):
        """Remove a grounded symbol (entity destroyed)."""
        symbol = self.grounded_symbols.pop(godot_id, None)
        if symbol and symbol.semantic_node_id:
            self.semantic_to_physical.pop(symbol.semantic_node_id, None)

    def get_statistics(self) -> Dict[str, Any]:
        """Get grounding statistics."""
        categories = {}
        for symbol in self.grounded_symbols.values():
            categories[symbol.category] = categories.get(symbol.category, 0) + 1

        return {
            "total_groundings": self.total_groundings,
            "active_symbols": len(self.grounded_symbols),
            "grounding_updates": self.grounding_updates,
            "categories": categories,
            "semantic_links": len(self.semantic_to_physical),
        }
