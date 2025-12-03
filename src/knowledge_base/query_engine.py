"""
Semantic Query Engine for Indra's Net

Implements sophisticated querying and activation spreading through
the semiotic knowledge graph. When an agent perceives an object,
the query engine triggers a Graph Traversal through the taxonomies,
activating associated concepts, norms, and archetypes.

This is the "background hum" of cognition - the saturation of hyperlinked meaning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import numpy as np

from .schemas import (
    SemanticNode,
    SemanticEdge,
    NodeType,
    EdgeType,
    ActivationContext,
    SemanticActivation,
    StereotypeDimension,
)
from .indras_net import IndrasNet
from .taxonomy import FullTaxonomy, TaxonomyLayer


@dataclass
class PerceptionResult:
    """
    Result of perceiving an object/entity in the simulation.

    When an agent perceives something, this captures the full semantic
    expansion - all the activated meanings, norms, and priors.
    """
    perceived_entity: str            # The entity that was perceived
    godot_id: Optional[int] = None   # Physical ID if applicable

    # Semantic expansion
    primary_activation: Optional[SemanticActivation] = None

    # Taxonomy positioning
    taxonomy_position: Optional[np.ndarray] = None
    dominant_layer: Optional[TaxonomyLayer] = None

    # Institutional context
    active_institution: Optional[str] = None
    applicable_norms: List[str] = field(default_factory=list)
    applicable_roles: List[str] = field(default_factory=list)

    # Stereotype priors
    stereotype_warmth: float = 0.5
    stereotype_competence: float = 0.5
    predicted_emotions: Dict[str, float] = field(default_factory=dict)

    # Archetype associations
    activated_archetypes: List[Tuple[str, float]] = field(default_factory=list)

    # Affordances
    available_actions: List[str] = field(default_factory=list)

    # Meta-information
    perception_time: datetime = field(default_factory=datetime.now)
    processing_ms: float = 0.0


@dataclass
class InstitutionalQuery:
    """
    Query for institutional context and associated norms/roles.
    """
    institution_id: str
    include_sub_institutions: bool = True
    include_roles: bool = True
    include_norms: bool = True
    include_power_structure: bool = True


@dataclass
class ArchetypeQuery:
    """
    Query for archetype matching based on taxonomy position.
    """
    taxonomy_position: np.ndarray
    max_distance: float = 0.5
    top_k: int = 3


class SemanticQueryEngine:
    """
    Query engine for traversing and querying Indra's Net.

    Provides high-level operations for:
    - Perception processing (object -> full semantic expansion)
    - Institutional context queries
    - Archetype matching
    - Stereotype retrieval
    - Norm applicability checking
    """

    def __init__(self, indras_net: IndrasNet):
        """
        Initialize the query engine.

        Args:
            indras_net: The semantic knowledge graph
        """
        self.net = indras_net
        self.taxonomy = indras_net.taxonomy

        # Cache for common queries
        self._institution_cache: Dict[str, Dict] = {}
        self._archetype_cache: Dict[str, np.ndarray] = {}

    def perceive(
        self,
        entity_id: str,
        context: ActivationContext,
        include_stereotypes: bool = True,
        include_archetypes: bool = True
    ) -> PerceptionResult:
        """
        Process perception of an entity through the semantic network.

        This is the core function that implements the "Physical is Cognitive"
        principle. When an agent perceives an object, this function:
        1. Retrieves the semantic node
        2. Spreads activation through the network
        3. Extracts applicable norms, roles, and archetypes
        4. Computes stereotype priors

        Args:
            entity_id: ID of the perceived entity
            context: The current activation context
            include_stereotypes: Whether to compute stereotype dimensions
            include_archetypes: Whether to match archetypes

        Returns:
            PerceptionResult with full semantic expansion
        """
        start_time = datetime.now()
        result = PerceptionResult(perceived_entity=entity_id)

        # Get the entity node
        node = self.net.get_node(entity_id)
        if not node:
            return result

        result.godot_id = node.godot_id
        result.taxonomy_position = node.taxonomy_position

        # Determine dominant taxonomy layer
        result.dominant_layer = self._compute_dominant_layer(node.taxonomy_position)

        # Spread activation through the network
        result.primary_activation = self.net.spread_activation(
            entity_id,
            context,
            initial_activation=1.0
        )

        # Extract institutional context
        if context.active_institution:
            inst_data = self._get_institution_data(context.active_institution)
            result.active_institution = context.active_institution
            result.applicable_norms = inst_data.get('norms', [])
            result.applicable_roles = inst_data.get('roles', [])

        # Add norms from activation
        result.applicable_norms.extend(result.primary_activation.activated_norms)
        result.applicable_roles.extend(result.primary_activation.activated_roles)

        # Deduplicate
        result.applicable_norms = list(set(result.applicable_norms))
        result.applicable_roles = list(set(result.applicable_roles))

        # Compute stereotype priors
        if include_stereotypes:
            stereotype = self._compute_stereotype(node, context)
            result.stereotype_warmth = stereotype.warmth
            result.stereotype_competence = stereotype.competence
            result.predicted_emotions = stereotype.predicted_emotions

        # Match archetypes
        if include_archetypes:
            result.activated_archetypes = self._match_archetypes(
                node.taxonomy_position
            )

        # Get available actions (affordances)
        result.available_actions = self._get_afforded_actions(entity_id, context)

        # Record timing
        end_time = datetime.now()
        result.processing_ms = (end_time - start_time).total_seconds() * 1000

        return result

    def query_institution(self, query: InstitutionalQuery) -> Dict[str, Any]:
        """
        Query institutional context, norms, roles, and power structures.

        Args:
            query: The institutional query specification

        Returns:
            Dict with institutional data
        """
        result = {
            'institution': query.institution_id,
            'exists': False,
            'norms': [],
            'roles': [],
            'power_structure': [],
            'sub_institutions': [],
        }

        # Get institution node
        inst_node = self.net.get_node(query.institution_id)
        if not inst_node:
            return result

        result['exists'] = True
        result['taxonomy_position'] = inst_node.taxonomy_position.tolist()

        # Get associated norms
        if query.include_norms:
            result['norms'] = self._get_institution_norms(query.institution_id)

        # Get associated roles
        if query.include_roles:
            result['roles'] = self._get_institution_roles(query.institution_id)

        # Get power structure
        if query.include_power_structure:
            result['power_structure'] = self._get_power_structure(query.institution_id)

        # Get sub-institutions
        if query.include_sub_institutions:
            result['sub_institutions'] = self._get_sub_institutions(query.institution_id)

        return result

    def match_archetypes(self, query: ArchetypeQuery) -> List[Tuple[str, float]]:
        """
        Find archetypes that best match a taxonomy position.

        Args:
            query: The archetype query specification

        Returns:
            List of (archetype_name, similarity) tuples
        """
        return self._match_archetypes(
            query.taxonomy_position,
            max_distance=query.max_distance,
            top_k=query.top_k
        )

    def check_norm_applicability(
        self,
        norm_id: str,
        agent_id: str,
        context: ActivationContext
    ) -> Dict[str, Any]:
        """
        Check if a norm applies to an agent in the current context.

        Args:
            norm_id: The norm to check
            agent_id: The agent to check against
            context: Current activation context

        Returns:
            Dict with applicability information
        """
        result = {
            'norm_id': norm_id,
            'applies': False,
            'enforcement_strength': 0.0,
            'agent_role': None,
            'violation_consequence': None,
        }

        norm_node = self.net.get_node(norm_id)
        if not norm_node:
            return result

        # Check if norm is in the current institution
        if context.active_institution:
            inst_norms = self._get_institution_norms(context.active_institution)
            if norm_id in [n['id'] for n in inst_norms]:
                result['applies'] = True
                result['enforcement_strength'] = norm_node.properties.get(
                    'enforcement_strength', 0.5
                )
                result['violation_consequence'] = norm_node.properties.get(
                    'violation_consequence', 'social_disapproval'
                )

        # Check if agent has a role that is subject to this norm
        agent_node = self.net.get_node(agent_id)
        if agent_node:
            for role_id in context.active_roles:
                role_node = self.net.get_node(role_id)
                if role_node and norm_id in role_node.associated_norms:
                    result['applies'] = True
                    result['agent_role'] = role_id

        return result

    def compute_power_differential(
        self,
        agent1_id: str,
        agent2_id: str,
        context: ActivationContext
    ) -> Dict[str, Any]:
        """
        Compute power differential between two agents in context.

        Args:
            agent1_id: First agent
            agent2_id: Second agent
            context: Current activation context

        Returns:
            Dict with power analysis
        """
        result = {
            'agent1': agent1_id,
            'agent2': agent2_id,
            'power_differential': 0.0,  # Positive = agent1 more powerful
            'sources': [],
        }

        # Get role-based power
        for role_id in context.active_roles:
            role_node = self.net.get_node(role_id)
            if role_node:
                power_level = role_node.properties.get('power_level', 0.5)
                # Check if either agent holds this role
                # (simplified - would need role assignment tracking)
                result['sources'].append({
                    'type': 'role',
                    'role': role_id,
                    'power_level': power_level,
                })

        # Get direct power edges
        power_edges = self.net.get_edges(
            source_id=agent1_id,
            target_id=agent2_id,
            edge_type=EdgeType.POWER_OVER
        )
        for edge in power_edges:
            result['power_differential'] += edge.weight
            result['sources'].append({
                'type': 'direct_power',
                'weight': edge.weight,
            })

        # Check deference edges (inverse)
        deference_edges = self.net.get_edges(
            source_id=agent1_id,
            target_id=agent2_id,
            edge_type=EdgeType.DEFERS_TO
        )
        for edge in deference_edges:
            result['power_differential'] -= edge.weight
            result['sources'].append({
                'type': 'deference',
                'weight': -edge.weight,
            })

        return result

    def find_conflicts(
        self,
        entity_id: str,
        context: ActivationContext
    ) -> List[Dict[str, Any]]:
        """
        Find potential conflicts (cognitive dissonance, norm violations).

        Args:
            entity_id: Entity to check for conflicts
            context: Current activation context

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        # Spread activation to find conflict edges
        activation = self.net.spread_activation(entity_id, context)

        for edge in activation.activated_edges:
            if edge.edge_type == EdgeType.CONFLICTS_WITH:
                conflicts.append({
                    'type': 'conceptual_conflict',
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'weight': edge.weight,
                })

        # Check for norm violations based on affordances
        entity_node = self.net.get_node(entity_id)
        if entity_node:
            for affordance in entity_node.affordances:
                action_node = self.net.get_node(f"action_{affordance}")
                if action_node:
                    for norm_id in context.active_norms:
                        # Check if action conflicts with norm
                        conflict_edges = self.net.get_edges(
                            source_id=action_node.id,
                            target_id=norm_id,
                            edge_type=EdgeType.CONFLICTS_WITH
                        )
                        for edge in conflict_edges:
                            conflicts.append({
                                'type': 'norm_violation_risk',
                                'action': affordance,
                                'norm': norm_id,
                                'weight': edge.weight,
                            })

        return conflicts

    def get_semantic_neighbors(
        self,
        entity_id: str,
        n: int = 10,
        filter_types: Optional[List[NodeType]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get semantically similar entities based on taxonomy position.

        Args:
            entity_id: Entity to find neighbors of
            n: Number of neighbors to return
            filter_types: Filter by node types

        Returns:
            List of (entity_id, similarity) tuples
        """
        node = self.net.get_node(entity_id)
        if not node:
            return []

        return self.net.find_by_taxonomy_position(
            node.taxonomy_position,
            max_distance=1.0,
            node_types=filter_types
        )[:n]

    # ==================== Private Helper Methods ====================

    def _compute_dominant_layer(self, taxonomy_position: np.ndarray) -> TaxonomyLayer:
        """Determine which taxonomy layer has the highest activation."""
        mundane_activation = np.sum(np.abs(taxonomy_position[0:27]))
        institutional_activation = np.sum(np.abs(taxonomy_position[27:54]))
        aesthetic_activation = np.sum(np.abs(taxonomy_position[54:80]))

        max_activation = max(mundane_activation, institutional_activation, aesthetic_activation)

        if max_activation == mundane_activation:
            return TaxonomyLayer.MUNDANE
        elif max_activation == institutional_activation:
            return TaxonomyLayer.INSTITUTIONAL
        else:
            return TaxonomyLayer.AESTHETIC

    def _get_institution_data(self, institution_id: str) -> Dict:
        """Get cached or compute institution data."""
        if institution_id in self._institution_cache:
            return self._institution_cache[institution_id]

        data = {
            'norms': [n['id'] for n in self._get_institution_norms(institution_id)],
            'roles': [r['id'] for r in self._get_institution_roles(institution_id)],
        }

        self._institution_cache[institution_id] = data
        return data

    def _get_institution_norms(self, institution_id: str) -> List[Dict]:
        """Get all norms associated with an institution."""
        norms = []

        # Get norms linked by ENFORCED_BY edges
        for source, edge_type, weight in self.net.get_neighbors(
            institution_id,
            edge_types=[EdgeType.ENFORCED_BY],
            direction="incoming"
        ):
            norm_node = self.net.get_node(source)
            if norm_node and norm_node.node_type == NodeType.NORM:
                norms.append({
                    'id': norm_node.id,
                    'name': norm_node.name,
                    'enforcement_strength': norm_node.properties.get('enforcement_strength', 0.5),
                })

        return norms

    def _get_institution_roles(self, institution_id: str) -> List[Dict]:
        """Get all roles associated with an institution."""
        roles = []

        # Get roles linked by ROLE_IN edges
        for source, edge_type, weight in self.net.get_neighbors(
            institution_id,
            edge_types=[EdgeType.ROLE_IN],
            direction="incoming"
        ):
            role_node = self.net.get_node(source)
            if role_node and role_node.node_type == NodeType.ROLE:
                roles.append({
                    'id': role_node.id,
                    'name': role_node.name,
                    'power_level': role_node.properties.get('power_level', 0.5),
                })

        return roles

    def _get_power_structure(self, institution_id: str) -> List[Dict]:
        """Get power relationships within an institution."""
        structure = []

        # Get all roles in institution
        roles = self._get_institution_roles(institution_id)
        role_ids = [r['id'] for r in roles]

        # Find power relationships between roles
        for role_id in role_ids:
            for target, edge_type, weight in self.net.get_neighbors(
                role_id,
                edge_types=[EdgeType.POWER_OVER, EdgeType.DEFERS_TO],
                direction="outgoing"
            ):
                if target in role_ids:
                    structure.append({
                        'source_role': role_id,
                        'target_role': target,
                        'relationship': edge_type.name,
                        'weight': weight,
                    })

        return structure

    def _get_sub_institutions(self, institution_id: str) -> List[str]:
        """Get sub-institutions (PART_OF relationships)."""
        subs = []

        for source, edge_type, weight in self.net.get_neighbors(
            institution_id,
            edge_types=[EdgeType.PART_OF],
            direction="incoming"
        ):
            node = self.net.get_node(source)
            if node and node.node_type == NodeType.INSTITUTION:
                subs.append(source)

        return subs

    def _compute_stereotype(
        self,
        node: SemanticNode,
        context: ActivationContext
    ) -> StereotypeDimension:
        """Compute stereotype dimensions for a node in context."""
        # Get base stereotype from node
        warmth = node.stereotype_associations.get('warmth', 0.5)
        competence = node.stereotype_associations.get('competence', 0.5)
        status = node.stereotype_associations.get('status', 0.5)

        # Adjust based on institutional context
        if context.active_institution:
            inst_node = self.net.get_node(context.active_institution)
            if inst_node:
                # Institutional context can shift perceptions
                inst_hierarchy = inst_node.taxonomy_position[33]  # Economic_Hierarchy dimension
                if inst_hierarchy > 0.7:  # High hierarchy institution
                    status *= 1.2
                    competence *= 1.1

        # Compute predicted emotions based on SCM/BIAS map
        predicted_emotions = self._compute_predicted_emotions(warmth, competence)

        return StereotypeDimension(
            warmth=np.clip(warmth, 0.0, 1.0),
            competence=np.clip(competence, 0.0, 1.0),
            status=np.clip(status, 0.0, 1.0),
            predicted_emotions=predicted_emotions,
        )

    def _compute_predicted_emotions(
        self,
        warmth: float,
        competence: float
    ) -> Dict[str, float]:
        """
        Compute predicted emotions based on Stereotype Content Model.

        High warmth + high competence -> admiration
        High warmth + low competence -> pity
        Low warmth + high competence -> envy
        Low warmth + low competence -> contempt
        """
        emotions = {}

        # Admiration (HW + HC)
        emotions['admiration'] = warmth * competence

        # Pity (HW + LC)
        emotions['pity'] = warmth * (1 - competence)

        # Envy (LW + HC)
        emotions['envy'] = (1 - warmth) * competence

        # Contempt (LW + LC)
        emotions['contempt'] = (1 - warmth) * (1 - competence)

        return emotions

    def _match_archetypes(
        self,
        taxonomy_position: np.ndarray,
        max_distance: float = 0.5,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Match a taxonomy position to archetypes."""
        matches = []

        archetype_nodes = self.net.get_nodes_by_type(NodeType.ARCHETYPE)

        for node in archetype_nodes:
            distance = self.taxonomy.compute_semantic_distance(
                taxonomy_position,
                node.taxonomy_position
            )

            if distance <= max_distance:
                # Convert distance to similarity
                similarity = 1.0 - (distance / max_distance)
                matches.append((node.name, similarity))

        # Sort by similarity and return top k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def _get_afforded_actions(
        self,
        entity_id: str,
        context: ActivationContext
    ) -> List[str]:
        """Get actions afforded by an entity in context."""
        actions = []

        # Get entity node
        node = self.net.get_node(entity_id)
        if not node:
            return actions

        # Get direct affordances
        actions.extend(node.affordances)

        # Get actions linked by AFFORDS edges
        for target, edge_type, weight in self.net.get_neighbors(
            entity_id,
            edge_types=[EdgeType.AFFORDS],
            direction="outgoing"
        ):
            action_node = self.net.get_node(target)
            if action_node and action_node.node_type == NodeType.ACTION:
                # Check if action is permitted in current institution
                if self._action_permitted_in_context(action_node, context):
                    actions.append(action_node.name)

        return list(set(actions))

    def _action_permitted_in_context(
        self,
        action_node: SemanticNode,
        context: ActivationContext
    ) -> bool:
        """Check if an action is permitted in the current context."""
        # By default, all actions are permitted
        # Check for norm conflicts
        for norm_id in context.active_norms:
            conflict_edges = self.net.get_edges(
                source_id=action_node.id,
                target_id=norm_id,
                edge_type=EdgeType.CONFLICTS_WITH
            )
            if conflict_edges:
                # Action conflicts with an active norm
                return False

        return True


def build_default_knowledge_base() -> Tuple[IndrasNet, SemanticQueryEngine]:
    """
    Build a default knowledge base with core concepts.

    Returns:
        Tuple of (IndrasNet, SemanticQueryEngine)
    """
    net = IndrasNet()

    # ==================== Add Core Archetypes ====================
    from .taxonomy import ARCHETYPE_TAXONOMY_POSITIONS

    for archetype_name, positions in ARCHETYPE_TAXONOMY_POSITIONS.items():
        net.create_archetype_node(
            archetype_id=f"archetype_{archetype_name.lower()}",
            name=archetype_name,
            preset=archetype_name,
        )

    # ==================== Add Core Institutions ====================
    from .taxonomy import INSTITUTION_TAXONOMY_POSITIONS

    for inst_name, positions in INSTITUTION_TAXONOMY_POSITIONS.items():
        net.create_institution_node(
            inst_id=f"inst_{inst_name.lower()}",
            name=inst_name,
            preset=inst_name,
        )

    # ==================== Add Common Objects ====================
    common_objects = [
        ("obj_chair", "Chair", {10: 0.7, 17: 0.6}, ["sit", "stand_on", "move"]),
        ("obj_table", "Table", {10: 0.7, 3: 0.8}, ["place_object", "eat_at", "work_at"]),
        ("obj_door", "Door", {7: 0.9, 8: 0.5}, ["open", "close", "knock"]),
        ("obj_book", "Book", {26: 0.9, 62: 0.8}, ["read", "give", "reference"]),
        ("obj_hammer", "Hammer", {11: 0.8, 14: 0.6}, ["strike", "build", "break"]),
        ("obj_key", "Key", {7: 0.7, 77: 0.6}, ["unlock", "lock", "give"]),
        ("obj_money", "Money", {24: 0.9, 32: 0.9}, ["pay", "receive", "count"]),
        ("obj_phone", "Phone", {20: 0.9, 23: 0.7}, ["call", "message", "browse"]),
    ]

    for obj_id, name, positions, affordances in common_objects:
        node = SemanticNode(
            id=obj_id,
            node_type=NodeType.OBJECT,
            name=name,
            taxonomy_position=net.taxonomy.create_taxonomy_vector(positions),
            affordances=affordances,
        )
        net.add_node(node)

    # ==================== Add Common Roles ====================
    common_roles = [
        ("role_judge", "Judge", "inst_courtroom", ["maintain_order", "render_verdict"], 0.9),
        ("role_defendant", "Defendant", "inst_courtroom", ["remain_silent", "testify"], 0.2),
        ("role_teacher", "Teacher", "inst_school", ["explain", "evaluate", "guide"], 0.7),
        ("role_student", "Student", "inst_school", ["learn", "ask", "submit_work"], 0.3),
        ("role_doctor", "Doctor", "inst_hospital", ["diagnose", "prescribe", "operate"], 0.8),
        ("role_patient", "Patient", "inst_hospital", ["comply", "report_symptoms"], 0.2),
        ("role_buyer", "Buyer", "inst_marketplace", ["negotiate", "pay", "receive"], 0.5),
        ("role_seller", "Seller", "inst_marketplace", ["advertise", "negotiate", "deliver"], 0.5),
    ]

    for role_id, name, inst_id, norms, power in common_roles:
        net.create_role_node(
            role_id=role_id,
            name=name,
            institution_id=inst_id,
            associated_norms=norms,
            power_level=power,
        )

    # ==================== Add Common Norms ====================
    common_norms = [
        ("norm_silence_court", "Silence in Court", "Maintain silence unless addressed", 0.9),
        ("norm_truth_oath", "Tell the Truth", "Speak only truth under oath", 0.95),
        ("norm_raise_hand", "Raise Hand to Speak", "Request permission before speaking", 0.7),
        ("norm_fair_price", "Fair Pricing", "Do not grossly overcharge", 0.6),
        ("norm_queue", "Queue Discipline", "Wait your turn in line", 0.7),
        ("norm_respect_elder", "Respect Elders", "Show deference to older individuals", 0.6),
    ]

    for norm_id, name, description, enforcement in common_norms:
        net.create_norm_node(
            norm_id=norm_id,
            name=name,
            description=description,
            enforcement_strength=enforcement,
        )

    # ==================== Add Semantic Edges ====================

    # Objects and their institutional associations
    net.add_edge(SemanticEdge(
        source_id="obj_book",
        target_id="inst_school",
        edge_type=EdgeType.OCCURS_IN,
        weight=0.9,
    ))

    net.add_edge(SemanticEdge(
        source_id="obj_money",
        target_id="inst_marketplace",
        edge_type=EdgeType.OCCURS_IN,
        weight=0.95,
    ))

    # Norms and their enforcement institutions
    net.add_edge(SemanticEdge(
        source_id="norm_silence_court",
        target_id="inst_courtroom",
        edge_type=EdgeType.ENFORCED_BY,
        weight=0.9,
    ))

    net.add_edge(SemanticEdge(
        source_id="norm_truth_oath",
        target_id="inst_courtroom",
        edge_type=EdgeType.ENFORCED_BY,
        weight=0.95,
    ))

    # Archetype associations
    net.add_edge(SemanticEdge(
        source_id="archetype_sage",
        target_id="role_teacher",
        edge_type=EdgeType.ASSOCIATED_WITH,
        weight=0.8,
    ))

    net.add_edge(SemanticEdge(
        source_id="archetype_hero",
        target_id="role_judge",
        edge_type=EdgeType.ASSOCIATED_WITH,
        weight=0.6,
    ))

    # Power relationships
    net.add_edge(SemanticEdge(
        source_id="role_judge",
        target_id="role_defendant",
        edge_type=EdgeType.POWER_OVER,
        weight=0.9,
    ))

    net.add_edge(SemanticEdge(
        source_id="role_teacher",
        target_id="role_student",
        edge_type=EdgeType.POWER_OVER,
        weight=0.7,
    ))

    net.add_edge(SemanticEdge(
        source_id="role_doctor",
        target_id="role_patient",
        edge_type=EdgeType.POWER_OVER,
        weight=0.6,
    ))

    # Create query engine
    engine = SemanticQueryEngine(net)

    return net, engine
