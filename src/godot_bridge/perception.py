"""
Perception Processor - From Physics to Cognition

Processes raw sensory input from Godot into CognitiveBlocks that
can be used by the Mentalese-based reasoning system.

The Perceptual Pipeline:
1. Raw Godot data (EntityUpdate, AgentPerception)
2. Symbol Grounding (physics -> semantic nodes)
3. Semantic Expansion (Indra's Net activation)
4. CognitiveBlock creation (PerceptBlock)

This is where embodied cognition happens - perception is not
passive reception but active semantic construction.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


from .protocol import AgentPerception, EntityUpdate, Vector3
from .symbol_grounding import GroundedSymbol, GroundingContext, SymbolGrounder


@dataclass
class SensoryInput:
    """
    Raw sensory input from a single modality.
    """

    modality: str  # visual, auditory, tactile, etc.
    source_godot_id: Optional[int] = None
    intensity: float = 1.0
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Spatial information
    direction: Optional[Vector3] = None  # Direction to source
    distance: float = 0.0

    # Attention
    is_salient: bool = False  # Captured attention


@dataclass
class PerceptualField:
    """
    The complete perceptual field of an agent at a moment.

    Contains all sensory inputs, grounded symbols, and
    the resulting semantic activation.
    """

    # Agent identity
    agent_godot_id: int
    agent_name: str

    # Raw inputs by modality
    visual_inputs: List[SensoryInput] = field(default_factory=list)
    auditory_inputs: List[SensoryInput] = field(default_factory=list)
    tactile_inputs: List[SensoryInput] = field(default_factory=list)
    proprioceptive_inputs: List[SensoryInput] = field(default_factory=list)

    # Grounded symbols (after symbol grounding)
    grounded_symbols: List[GroundedSymbol] = field(default_factory=list)

    # Semantic activation (from Indra's Net)
    semantic_activation: Dict[str, float] = field(default_factory=dict)
    activated_norms: List[str] = field(default_factory=list)
    activated_roles: List[str] = field(default_factory=list)
    activated_archetypes: List[str] = field(default_factory=list)

    # Attention
    focal_object: Optional[int] = None  # Godot ID of attention focus
    peripheral_objects: List[int] = field(default_factory=list)

    # Context
    current_institution: Optional[str] = None
    current_location: Optional[str] = None

    # Temporal
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0  # How long to process

    def get_most_salient(self, n: int = 5) -> List[GroundedSymbol]:
        """Get the N most salient perceived symbols."""
        # Sort by perception count and prototype similarity
        sorted_symbols = sorted(
            self.grounded_symbols, key=lambda s: (s.perception_count, s.prototype_similarity), reverse=True
        )
        return sorted_symbols[:n]


class PerceptionProcessor:
    """
    Processes raw Godot perception into cognitive content.

    Pipeline:
    1. Receive AgentPerception from Godot
    2. Create SensoryInputs for each modality
    3. Ground each perceived entity as GroundedSymbol
    4. Trigger semantic activation through knowledge base
    5. Produce PerceptualField ready for cognition
    """

    def __init__(self, symbol_grounder: SymbolGrounder, knowledge_base=None):
        """
        Initialize perception processor.

        Args:
            symbol_grounder: Symbol grounding system
            knowledge_base: Indra's Net knowledge graph
        """
        self.grounder = symbol_grounder
        self.knowledge_base = knowledge_base

        # Perception history (for temporal integration)
        self.perception_history: Dict[int, List[PerceptualField]] = {}
        self.history_length = 10  # Keep last N perceptions

        # Attention parameters
        self.attention_decay = 0.9
        self.salience_threshold = 0.3

        # Statistics
        self.perceptions_processed = 0

    def process_perception(self, perception: AgentPerception) -> PerceptualField:
        """
        Process a complete perception from Godot.

        Args:
            perception: Raw perception data from Godot

        Returns:
            PerceptualField with grounded, semantically-enriched content
        """
        start_time = datetime.now()

        # Create perceptual field
        pfield = PerceptualField(
            agent_godot_id=perception.agent_godot_id,
            agent_name=perception.agent_name,
            current_institution=perception.current_institution,
        )

        # Create grounding context
        context = GroundingContext(
            observer_position=perception.own_position,
            observer_orientation=perception.own_orientation,
            current_institution=perception.current_institution,
            simulation_time=perception.timestamp,
        )

        # Process visual perception
        for entity in perception.visible_entities:
            # Create sensory input
            sensory = SensoryInput(
                modality="visual",
                source_godot_id=entity.godot_id,
                content={"entity": entity},
                direction=self._compute_direction(perception.own_position, entity.position),
                distance=perception.own_position.distance_to(entity.position),
            )

            # Compute salience
            sensory.is_salient = self._compute_salience(entity, context)
            pfield.visual_inputs.append(sensory)

            # Ground the entity
            grounded = self.grounder.ground_entity(entity, context)
            pfield.grounded_symbols.append(grounded)

        # Process auditory perception
        for utterance in perception.heard_utterances:
            sensory = SensoryInput(
                modality="auditory",
                source_godot_id=utterance.get("speaker_id"),
                content={"text": utterance.get("text", "")},
                intensity=utterance.get("volume", 1.0),
            )
            pfield.auditory_inputs.append(sensory)

        # Process proprioception
        proprio = SensoryInput(
            modality="proprioceptive",
            content={
                "position": perception.own_position,
                "velocity": perception.own_velocity,
                "energy": perception.energy_level,
                "held_object": perception.held_object,
            },
        )
        pfield.proprioceptive_inputs.append(proprio)

        # Determine attention focus
        pfield.focal_object = self._determine_attention_focus(pfield.visual_inputs, context)

        # Trigger semantic activation
        if self.knowledge_base and pfield.grounded_symbols:
            self._trigger_semantic_activation(pfield, context)

        # Record timing
        end_time = datetime.now()
        pfield.duration_ms = (end_time - start_time).total_seconds() * 1000
        pfield.timestamp = start_time

        # Store in history
        agent_id = perception.agent_godot_id
        if agent_id not in self.perception_history:
            self.perception_history[agent_id] = []
        self.perception_history[agent_id].append(pfield)
        if len(self.perception_history[agent_id]) > self.history_length:
            self.perception_history[agent_id].pop(0)

        self.perceptions_processed += 1

        return pfield

    def _compute_direction(self, observer: Vector3, target: Vector3) -> Vector3:
        """Compute direction vector from observer to target."""
        dx = target.x - observer.x
        dy = target.y - observer.y
        dz = target.z - observer.z

        # Normalize
        length = (dx * dx + dy * dy + dz * dz) ** 0.5
        if length > 0:
            return Vector3(dx / length, dy / length, dz / length)
        return Vector3(0, 0, 0)

    def _compute_salience(self, entity: EntityUpdate, context: GroundingContext) -> bool:
        """
        Determine if an entity is salient (attention-grabbing).

        Salience factors:
        - Movement (velocity)
        - Size (larger = more salient)
        - Distance (closer = more salient)
        - Novelty (new entities)
        - Relevance to current goals
        """
        salience_score = 0.0

        # Movement salience
        speed = (entity.velocity.x**2 + entity.velocity.y**2 + entity.velocity.z**2) ** 0.5
        if speed > 0.1:
            salience_score += 0.3

        # Size salience
        size = entity.scale.x * entity.scale.y * entity.scale.z
        if size > 1.0:
            salience_score += 0.2

        # Distance salience (closer = higher)
        if context.observer_position:
            distance = context.observer_position.distance_to(entity.position)
            if distance < 2.0:
                salience_score += 0.3
            elif distance < 5.0:
                salience_score += 0.1

        # Agent type is always salient
        if entity.entity_type == "agent":
            salience_score += 0.4

        return salience_score > self.salience_threshold

    def _determine_attention_focus(self, visual_inputs: List[SensoryInput], context: GroundingContext) -> Optional[int]:
        """Determine what object has attention focus."""
        # Prior attention has momentum
        prior_focus = context.attention_focus

        # Find most salient input
        salient_inputs = [v for v in visual_inputs if v.is_salient]

        if not salient_inputs:
            return prior_focus

        # If prior focus is still salient, keep it
        if prior_focus:
            for inp in salient_inputs:
                if inp.source_godot_id == prior_focus:
                    return prior_focus

        # Otherwise, switch to closest salient object
        closest = min(salient_inputs, key=lambda v: v.distance)
        return closest.source_godot_id

    def _trigger_semantic_activation(self, pfield: PerceptualField, context: GroundingContext):
        """
        Trigger semantic activation through Indra's Net.

        This is the "background hum" of cognition - perceiving
        an object activates associated concepts, norms, archetypes.
        """
        if not self.knowledge_base:
            return

        all_activations = {}

        for symbol in pfield.grounded_symbols:
            if symbol.semantic_node_id:
                # Create activation context
                from ..knowledge_base.schemas import ActivationContext as AC

                ac = AC(
                    current_location=pfield.current_location,
                    perceiving_agent=pfield.agent_name,
                    active_institution=context.current_institution,
                )

                # Spread activation
                activation = self.knowledge_base.spread_activation(symbol.semantic_node_id, ac)

                # Aggregate activations
                for node_id, level in activation.activated_nodes.items():
                    if node_id in all_activations:
                        all_activations[node_id] = max(all_activations[node_id], level)
                    else:
                        all_activations[node_id] = level

                # Collect norms, roles, archetypes
                pfield.activated_norms.extend(activation.activated_norms)
                pfield.activated_roles.extend(activation.activated_roles)
                pfield.activated_archetypes.extend(activation.activated_archetypes)

        pfield.semantic_activation = all_activations
        pfield.activated_norms = list(set(pfield.activated_norms))
        pfield.activated_roles = list(set(pfield.activated_roles))
        pfield.activated_archetypes = list(set(pfield.activated_archetypes))

    def get_recent_perceptions(self, agent_id: int, n: int = 5) -> List[PerceptualField]:
        """Get recent perceptions for an agent."""
        history = self.perception_history.get(agent_id, [])
        return history[-n:]

    def get_perception_delta(self, agent_id: int) -> Dict[str, Any]:
        """
        Get what changed since last perception.

        Returns new objects, disappeared objects, moved objects.
        """
        history = self.perception_history.get(agent_id, [])
        if len(history) < 2:
            return {"new": [], "gone": [], "moved": []}

        prev = history[-2]
        curr = history[-1]

        prev_ids = {s.godot_id for s in prev.grounded_symbols}
        curr_ids = {s.godot_id for s in curr.grounded_symbols}

        new_objects = curr_ids - prev_ids
        gone_objects = prev_ids - curr_ids

        # Check for moved objects
        moved = []
        for symbol in curr.grounded_symbols:
            if symbol.godot_id in prev_ids and not symbol.is_stable:
                moved.append(symbol.godot_id)

        return {
            "new": list(new_objects),
            "gone": list(gone_objects),
            "moved": moved,
        }

    def create_percept_block(self, pfield: PerceptualField):
        """
        Create a Mentalese PerceptBlock from a PerceptualField.

        This is the final step converting physical perception
        into cognitive content.
        """
        from ..cognition.mentalese import PerceptBlock

        # Get focal object
        focal_symbol = None
        for symbol in pfield.grounded_symbols:
            if symbol.godot_id == pfield.focal_object:
                focal_symbol = symbol
                break

        if focal_symbol:
            return PerceptBlock(
                perceived_entity=focal_symbol.name,
                godot_id=focal_symbol.godot_id,
                position_3d=focal_symbol.position.to_tuple(),
                visual_features={
                    f.name: getattr(focal_symbol.visual_features, f.name)
                    for f in focal_symbol.visual_features.__dataclass_fields__.values()
                    if isinstance(getattr(focal_symbol.visual_features, f.name), (int, float))
                },
                activated_concepts=list(pfield.semantic_activation.keys())[:10],
            )

        return None
