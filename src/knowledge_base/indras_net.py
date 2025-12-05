"""
Indra's Net - The Semiotic Knowledge Graph

Implements the omnipresent semantic web where every entity exists in a
superposition of latent meanings collapsed by context. Uses NetworkX as
the underlying graph database.

"In the heaven of Indra, there is said to be a network of pearls, so
arranged that if you look at one you see all the others reflected in it."
- Avatamsaka Sutra

Each node reflects all other nodes; meaning is distributed across the
entire network. Perception triggers cascading activation through the web.
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .schemas import (
    ActivationContext,
    EdgeType,
    NodeType,
    PrototypeRepresentation,
    SemanticActivation,
    SemanticEdge,
    SemanticNode,
    StereotypeDimension,
)
from .taxonomy import (
    ARCHETYPE_TAXONOMY_POSITIONS,
    INSTITUTION_TAXONOMY_POSITIONS,
    FullTaxonomy,
    TaxonomyLayer,
)


class IndrasNet:
    """
    The Semiotic Knowledge Graph implementing Indra's Net.

    A fully-connected semantic web where:
    - Every node reflects the entire network
    - Context collapses superposed meanings
    - Activation spreads through typed edges
    - Physical and cognitive are isomorphic

    The graph serves as the read-only substrate of meaning for
    the entire simulation.
    """

    def __init__(self, taxonomy: Optional[FullTaxonomy] = None):
        """
        Initialize Indra's Net.

        Args:
            taxonomy: The 80-Dimension taxonomy (creates default if None)
        """
        # Core graph structure
        self._graph = nx.MultiDiGraph()

        # Taxonomy reference
        self.taxonomy = taxonomy or FullTaxonomy()

        # Index structures for efficient lookup
        self._nodes_by_type: Dict[NodeType, Set[str]] = {t: set() for t in NodeType}
        self._nodes_by_godot_id: Dict[int, str] = {}
        self._prototypes: Dict[str, PrototypeRepresentation] = {}
        self._stereotypes: Dict[str, StereotypeDimension] = {}

        # Activation state
        self._activation_cache: Dict[str, float] = {}

        # Statistics
        self._creation_time = datetime.now()
        self._modification_count = 0

    # ==================== Node Operations ====================

    def add_node(self, node: SemanticNode) -> bool:
        """
        Add a semantic node to the graph.

        Args:
            node: The SemanticNode to add

        Returns:
            True if added successfully, False if already exists
        """
        if self._graph.has_node(node.id):
            return False

        # Add to NetworkX graph with all attributes
        self._graph.add_node(
            node.id,
            node_type=node.node_type,
            name=node.name,
            godot_id=node.godot_id,
            position_3d=node.position_3d,
            taxonomy_position=node.taxonomy_position,
            prototype_features=node.prototype_features,
            stereotype_associations=node.stereotype_associations,
            activation_level=node.activation_level,
            properties=node.properties,
            affordances=node.affordances,
            associated_norms=node.associated_norms,
            associated_roles=node.associated_roles,
            emotional_valence=node.emotional_valence,
            arousal_level=node.arousal_level,
            created_at=node.created_at,
            _node_object=node,  # Store full object for retrieval
        )

        # Update indices
        self._nodes_by_type[node.node_type].add(node.id)
        if node.godot_id is not None:
            self._nodes_by_godot_id[node.godot_id] = node.id

        self._modification_count += 1
        return True

    def get_node(self, node_id: str) -> Optional[SemanticNode]:
        """Retrieve a node by ID."""
        if not self._graph.has_node(node_id):
            return None
        return self._graph.nodes[node_id].get("_node_object")

    def get_node_by_godot_id(self, godot_id: int) -> Optional[SemanticNode]:
        """Retrieve a node by its Godot physics engine ID."""
        node_id = self._nodes_by_godot_id.get(godot_id)
        if node_id:
            return self.get_node(node_id)
        return None

    def get_nodes_by_type(self, node_type: NodeType) -> List[SemanticNode]:
        """Get all nodes of a specific type."""
        return [self.get_node(nid) for nid in self._nodes_by_type[node_type] if self.get_node(nid) is not None]

    def update_node(self, node: SemanticNode) -> bool:
        """Update an existing node."""
        if not self._graph.has_node(node.id):
            return False

        # Update all attributes
        nx_node = self._graph.nodes[node.id]
        nx_node["taxonomy_position"] = node.taxonomy_position
        nx_node["activation_level"] = node.activation_level
        nx_node["properties"] = node.properties
        nx_node["_node_object"] = node

        self._modification_count += 1
        return True

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if not self._graph.has_node(node_id):
            return False

        node = self.get_node(node_id)
        if node:
            self._nodes_by_type[node.node_type].discard(node_id)
            if node.godot_id is not None:
                self._nodes_by_godot_id.pop(node.godot_id, None)

        self._graph.remove_node(node_id)
        self._modification_count += 1
        return True

    # ==================== Edge Operations ====================

    def add_edge(self, edge: SemanticEdge) -> bool:
        """
        Add a semantic edge to the graph.

        Args:
            edge: The SemanticEdge to add

        Returns:
            True if added successfully
        """
        if not self._graph.has_node(edge.source_id):
            return False
        if not self._graph.has_node(edge.target_id):
            return False

        # Add edge with attributes
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.edge_type.name,
            edge_type=edge.edge_type,
            weight=edge.weight,
            bidirectional=edge.bidirectional,
            context_conditions=edge.context_conditions,
            theoretical_basis=edge.theoretical_basis,
            created_at=edge.created_at,
            _edge_object=edge,
        )

        # Add reverse edge if bidirectional
        if edge.bidirectional:
            reverse_edge = SemanticEdge(
                source_id=edge.target_id,
                target_id=edge.source_id,
                edge_type=edge.edge_type,
                weight=edge.weight,
                bidirectional=True,
                context_conditions=edge.context_conditions,
            )
            self._graph.add_edge(
                edge.target_id,
                edge.source_id,
                key=edge.edge_type.name,
                edge_type=edge.edge_type,
                weight=edge.weight,
                bidirectional=True,
                context_conditions=edge.context_conditions,
                _edge_object=reverse_edge,
            )

        self._modification_count += 1
        return True

    def get_edges(
        self, source_id: Optional[str] = None, target_id: Optional[str] = None, edge_type: Optional[EdgeType] = None
    ) -> List[SemanticEdge]:
        """
        Get edges matching the criteria.

        Args:
            source_id: Filter by source node
            target_id: Filter by target node
            edge_type: Filter by edge type

        Returns:
            List of matching edges
        """
        edges = []

        if source_id and target_id:
            # Specific edge lookup
            if self._graph.has_edge(source_id, target_id):
                for key, data in self._graph[source_id][target_id].items():
                    if edge_type is None or data.get("edge_type") == edge_type:
                        edge_obj = data.get("_edge_object")
                        if edge_obj:
                            edges.append(edge_obj)
        elif source_id:
            # All edges from source
            for target in self._graph.successors(source_id):
                for key, data in self._graph[source_id][target].items():
                    if edge_type is None or data.get("edge_type") == edge_type:
                        edge_obj = data.get("_edge_object")
                        if edge_obj:
                            edges.append(edge_obj)
        elif target_id:
            # All edges to target
            for source in self._graph.predecessors(target_id):
                for key, data in self._graph[source][target_id].items():
                    if edge_type is None or data.get("edge_type") == edge_type:
                        edge_obj = data.get("_edge_object")
                        if edge_obj:
                            edges.append(edge_obj)
        else:
            # All edges
            for source, target, data in self._graph.edges(data=True):
                if edge_type is None or data.get("edge_type") == edge_type:
                    edge_obj = data.get("_edge_object")
                    if edge_obj:
                        edges.append(edge_obj)

        return edges

    def get_neighbors(
        self, node_id: str, edge_types: Optional[List[EdgeType]] = None, direction: str = "both"
    ) -> List[Tuple[str, EdgeType, float]]:
        """
        Get neighboring nodes connected by specified edge types.

        Args:
            node_id: The node to find neighbors of
            edge_types: Filter by these edge types (None = all)
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of (neighbor_id, edge_type, weight) tuples
        """
        neighbors = []

        if direction in ["outgoing", "both"]:
            for target in self._graph.successors(node_id):
                for key, data in self._graph[node_id][target].items():
                    if edge_types is None or data.get("edge_type") in edge_types:
                        neighbors.append((target, data.get("edge_type"), data.get("weight", 1.0)))

        if direction in ["incoming", "both"]:
            for source in self._graph.predecessors(node_id):
                for key, data in self._graph[source][node_id].items():
                    if edge_types is None or data.get("edge_type") in edge_types:
                        neighbors.append((source, data.get("edge_type"), data.get("weight", 1.0)))

        return neighbors

    # ==================== Activation Spreading ====================

    def spread_activation(
        self, trigger_node_id: str, context: ActivationContext, initial_activation: float = 1.0
    ) -> SemanticActivation:
        """
        Spread activation through the network from a trigger node.

        This implements the core mechanism of Indra's Net: when one node
        is activated (perceived), activation spreads through typed edges
        to related concepts, collapsing superposed meanings based on context.

        Args:
            trigger_node_id: The node that triggered activation
            context: The context shaping activation spreading
            initial_activation: Starting activation level

        Returns:
            SemanticActivation with all activated content
        """
        start_time = datetime.now()

        # Initialize activation result
        result = SemanticActivation(
            trigger_node=trigger_node_id,
            context=context,
            activated_nodes={trigger_node_id: initial_activation},
        )

        # BFS-style activation spreading
        frontier = [(trigger_node_id, initial_activation, 0)]  # (node_id, activation, depth)
        visited = {trigger_node_id}

        while frontier:
            current_id, current_activation, depth = frontier.pop(0)

            # Stop if exceeded max depth
            if depth >= context.max_spread_depth:
                continue

            # Get current node
            current_node = self.get_node(current_id)
            if not current_node:
                continue

            # Spread to neighbors
            for neighbor_id, edge_type, weight in self.get_neighbors(current_id, direction="outgoing"):
                if neighbor_id in visited:
                    continue

                # Get edge for context checking
                edges = self.get_edges(current_id, neighbor_id, edge_type)
                if not edges:
                    continue

                edge = edges[0]

                # Check if edge is active in current context
                if edge.context_conditions and not context.matches_conditions(edge.context_conditions):
                    continue

                # Compute propagated activation
                propagated = current_activation * weight * context.spreading_decay

                # Check threshold
                if propagated < context.activation_threshold:
                    continue

                # Record activation
                visited.add(neighbor_id)
                result.activated_nodes[neighbor_id] = propagated
                result.activated_edges.append(edge)

                # Add to frontier for further spreading
                frontier.append((neighbor_id, propagated, depth + 1))

        # Post-process: identify activated norms, roles, archetypes
        for node_id, activation in result.activated_nodes.items():
            node = self.get_node(node_id)
            if not node:
                continue

            if node.node_type == NodeType.NORM:
                result.activated_norms.append(node.name)
            elif node.node_type == NodeType.ROLE:
                result.activated_roles.append(node.name)
            elif node.node_type == NodeType.ARCHETYPE:
                result.activated_archetypes.append(node.name)

        # Compute aggregate valence and arousal
        total_activation = sum(result.activated_nodes.values())
        if total_activation > 0:
            weighted_valence = 0.0
            weighted_arousal = 0.0
            for node_id, activation in result.activated_nodes.items():
                node = self.get_node(node_id)
                if node:
                    weighted_valence += node.emotional_valence * activation
                    weighted_arousal += node.arousal_level * activation
            result.aggregate_valence = weighted_valence / total_activation
            result.aggregate_arousal = weighted_arousal / total_activation

        # Record traversal depth
        result.traversal_depth = context.max_spread_depth

        # Record computation time
        end_time = datetime.now()
        result.computation_time_ms = (end_time - start_time).total_seconds() * 1000

        return result

    # ==================== Semantic Queries ====================

    def find_path(
        self, source_id: str, target_id: str, edge_types: Optional[List[EdgeType]] = None
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.

        Args:
            source_id: Starting node
            target_id: Ending node
            edge_types: Restrict to these edge types

        Returns:
            List of node IDs forming the path, or None if no path exists
        """
        try:
            if edge_types:
                # Create filtered view
                def edge_filter(u, v, key, data):
                    return data.get("edge_type") in edge_types

                view = nx.subgraph_view(self._graph, filter_edge=edge_filter)
                return nx.shortest_path(view, source_id, target_id)
            else:
                return nx.shortest_path(self._graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None

    def find_by_taxonomy_position(
        self, target_position: np.ndarray, max_distance: float = 0.5, node_types: Optional[List[NodeType]] = None
    ) -> List[Tuple[str, float]]:
        """
        Find nodes near a position in taxonomy space.

        Args:
            target_position: 80-dimensional target position
            max_distance: Maximum semantic distance
            node_types: Filter by node types

        Returns:
            List of (node_id, distance) tuples sorted by distance
        """
        results = []

        for node_id in self._graph.nodes():
            node = self.get_node(node_id)
            if not node:
                continue

            if node_types and node.node_type not in node_types:
                continue

            distance = self.taxonomy.compute_semantic_distance(node.taxonomy_position, target_position)

            if distance <= max_distance:
                results.append((node_id, distance))

        return sorted(results, key=lambda x: x[1])

    def find_by_stereotype(
        self, warmth_range: Tuple[float, float] = (0.0, 1.0), competence_range: Tuple[float, float] = (0.0, 1.0)
    ) -> List[str]:
        """
        Find nodes matching stereotype criteria (SCM dimensions).

        Args:
            warmth_range: (min, max) warmth values
            competence_range: (min, max) competence values

        Returns:
            List of matching node IDs
        """
        results = []

        for node_id in self._graph.nodes():
            node = self.get_node(node_id)
            if not node or not node.stereotype_associations:
                continue

            warmth = node.stereotype_associations.get("warmth", 0.5)
            competence = node.stereotype_associations.get("competence", 0.5)

            if (
                warmth_range[0] <= warmth <= warmth_range[1]
                and competence_range[0] <= competence <= competence_range[1]
            ):
                results.append(node_id)

        return results

    # ==================== Graph Construction Helpers ====================

    def create_object_node(
        self,
        object_id: str,
        name: str,
        godot_id: int,
        position_3d: Tuple[float, float, float],
        taxonomy_positions: Dict[int, float],
        affordances: List[str],
        properties: Optional[Dict[str, Any]] = None,
    ) -> SemanticNode:
        """
        Create and add a physical object node grounded in Godot.

        Args:
            object_id: Unique identifier (e.g., "obj_chair_402")
            name: Human-readable name
            godot_id: Godot physics engine ID
            position_3d: 3D position in world
            taxonomy_positions: Dict of dimension_id -> value
            affordances: Actions the object enables
            properties: Additional properties

        Returns:
            The created SemanticNode
        """
        node = SemanticNode(
            id=object_id,
            node_type=NodeType.OBJECT,
            name=name,
            godot_id=godot_id,
            position_3d=position_3d,
            taxonomy_position=self.taxonomy.create_taxonomy_vector(taxonomy_positions),
            affordances=affordances,
            properties=properties or {},
        )
        self.add_node(node)
        return node

    def create_institution_node(
        self,
        inst_id: str,
        name: str,
        preset: Optional[str] = None,
        taxonomy_positions: Optional[Dict[int, float]] = None,
        associated_roles: Optional[List[str]] = None,
        associated_norms: Optional[List[str]] = None,
    ) -> SemanticNode:
        """
        Create and add an institution node.

        Args:
            inst_id: Unique identifier
            name: Human-readable name
            preset: Use preset taxonomy position (e.g., "Courtroom")
            taxonomy_positions: Custom taxonomy positions (overrides preset)
            associated_roles: Roles existing in this institution
            associated_norms: Norms enforced by this institution

        Returns:
            The created SemanticNode
        """
        if preset and preset in INSTITUTION_TAXONOMY_POSITIONS:
            positions = INSTITUTION_TAXONOMY_POSITIONS[preset].copy()
            if taxonomy_positions:
                positions.update(taxonomy_positions)
        else:
            positions = taxonomy_positions or {}

        node = SemanticNode(
            id=inst_id,
            node_type=NodeType.INSTITUTION,
            name=name,
            taxonomy_position=self.taxonomy.create_taxonomy_vector(positions),
            associated_roles=associated_roles or [],
            associated_norms=associated_norms or [],
        )
        self.add_node(node)
        return node

    def create_archetype_node(
        self,
        archetype_id: str,
        name: str,
        preset: Optional[str] = None,
        taxonomy_positions: Optional[Dict[int, float]] = None,
        stereotype: Optional[StereotypeDimension] = None,
    ) -> SemanticNode:
        """
        Create and add an archetype node (Jungian archetypes).

        Args:
            archetype_id: Unique identifier
            name: Archetype name (e.g., "Hero", "Rebel")
            preset: Use preset taxonomy position
            taxonomy_positions: Custom positions
            stereotype: Stereotype content dimensions

        Returns:
            The created SemanticNode
        """
        if preset and preset in ARCHETYPE_TAXONOMY_POSITIONS:
            positions = ARCHETYPE_TAXONOMY_POSITIONS[preset].copy()
            if taxonomy_positions:
                positions.update(taxonomy_positions)
        else:
            positions = taxonomy_positions or {}

        stereotype_assoc = {}
        if stereotype:
            stereotype_assoc = {
                "warmth": stereotype.warmth,
                "competence": stereotype.competence,
                "status": stereotype.status,
            }

        node = SemanticNode(
            id=archetype_id,
            node_type=NodeType.ARCHETYPE,
            name=name,
            taxonomy_position=self.taxonomy.create_taxonomy_vector(positions),
            stereotype_associations=stereotype_assoc,
        )
        self.add_node(node)
        return node

    def create_role_node(
        self,
        role_id: str,
        name: str,
        institution_id: str,
        associated_norms: List[str],
        power_level: float = 0.5,
        stereotype: Optional[StereotypeDimension] = None,
    ) -> SemanticNode:
        """
        Create and add a social role node.

        Args:
            role_id: Unique identifier
            name: Role name (e.g., "Judge", "Teacher")
            institution_id: Institution this role exists in
            associated_norms: Norms associated with this role
            power_level: Power level (0.0-1.0)
            stereotype: Stereotype content

        Returns:
            The created SemanticNode
        """
        stereotype_assoc = {}
        if stereotype:
            stereotype_assoc = {
                "warmth": stereotype.warmth,
                "competence": stereotype.competence,
            }

        node = SemanticNode(
            id=role_id,
            node_type=NodeType.ROLE,
            name=name,
            associated_norms=associated_norms,
            properties={"power_level": power_level, "institution": institution_id},
            stereotype_associations=stereotype_assoc,
        )
        self.add_node(node)

        # Auto-link to institution
        if self._graph.has_node(institution_id):
            self.add_edge(
                SemanticEdge(
                    source_id=role_id,
                    target_id=institution_id,
                    edge_type=EdgeType.ROLE_IN,
                    weight=1.0,
                )
            )

        return node

    def create_norm_node(
        self,
        norm_id: str,
        name: str,
        description: str,
        enforcement_strength: float = 0.5,
        violation_consequence: str = "social_disapproval",
    ) -> SemanticNode:
        """
        Create and add a social norm node.

        Args:
            norm_id: Unique identifier
            name: Norm name
            description: What the norm prescribes
            enforcement_strength: How strongly enforced (0.0-1.0)
            violation_consequence: What happens on violation

        Returns:
            The created SemanticNode
        """
        node = SemanticNode(
            id=norm_id,
            node_type=NodeType.NORM,
            name=name,
            properties={
                "description": description,
                "enforcement_strength": enforcement_strength,
                "violation_consequence": violation_consequence,
            },
        )
        self.add_node(node)
        return node

    def create_action_node(
        self,
        action_id: str,
        name: str,
        required_objects: List[str],
        preconditions: Dict[str, Any],
        effects: Dict[str, Any],
    ) -> SemanticNode:
        """
        Create and add an action node.

        Args:
            action_id: Unique identifier
            name: Action name
            required_objects: Object IDs required for this action
            preconditions: Conditions that must be true
            effects: Changes caused by this action

        Returns:
            The created SemanticNode
        """
        node = SemanticNode(
            id=action_id,
            node_type=NodeType.ACTION,
            name=name,
            properties={
                "required_objects": required_objects,
                "preconditions": preconditions,
                "effects": effects,
            },
        )
        self.add_node(node)

        # Link to required objects
        for obj_id in required_objects:
            if self._graph.has_node(obj_id):
                obj_node = self.get_node(obj_id)
                if obj_node:
                    self.add_edge(
                        SemanticEdge(
                            source_id=obj_id,
                            target_id=action_id,
                            edge_type=EdgeType.AFFORDS,
                            weight=1.0,
                        )
                    )

        return node

    # ==================== Serialization ====================

    def save(self, filepath: Path) -> None:
        """Save the graph to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "graph": self._graph,
                    "nodes_by_type": self._nodes_by_type,
                    "nodes_by_godot_id": self._nodes_by_godot_id,
                    "prototypes": self._prototypes,
                    "stereotypes": self._stereotypes,
                },
                f,
            )

    def load(self, filepath: Path) -> None:
        """Load the graph from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self._graph = data["graph"]
            self._nodes_by_type = data["nodes_by_type"]
            self._nodes_by_godot_id = data["nodes_by_godot_id"]
            self._prototypes = data["prototypes"]
            self._stereotypes = data["stereotypes"]

    def export_to_json(self, filepath: Path) -> None:
        """Export graph structure to JSON (without numpy arrays)."""
        nodes = []
        for node_id in self._graph.nodes():
            node = self.get_node(node_id)
            if node:
                nodes.append(
                    {
                        "id": node.id,
                        "type": node.node_type.name,
                        "name": node.name,
                        "godot_id": node.godot_id,
                        "properties": node.properties,
                        "affordances": node.affordances,
                    }
                )

        edges = []
        for source, target, data in self._graph.edges(data=True):
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "type": data.get("edge_type", EdgeType.ASSOCIATED_WITH).name,
                    "weight": data.get("weight", 1.0),
                }
            )

        with open(filepath, "w") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, indent=2)

    # ==================== Statistics ====================

    @property
    def node_count(self) -> int:
        """Total number of nodes."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Total number of edges."""
        return self._graph.number_of_edges()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        stats = {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "nodes_by_type": {t.name: len(ids) for t, ids in self._nodes_by_type.items()},
            "creation_time": self._creation_time.isoformat(),
            "modification_count": self._modification_count,
        }

        # Compute connectivity metrics
        if self.node_count > 0:
            stats["avg_degree"] = self.edge_count / self.node_count
            stats["density"] = nx.density(self._graph)

        return stats

    def __repr__(self) -> str:
        return f"IndrasNet(nodes={self.node_count}, edges={self.edge_count})"
