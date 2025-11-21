"""
Recursive Belief Architecture for ToM-NAS - Full Bayesian Implementation
Supports 5th-order recursive beliefs with proper uncertainty propagation

Key features:
- Bayesian belief updates with evidence integration
- Confidence decay across recursion depths
- Belief trajectory extraction for transparency
- Divergence computation between belief states
- Cross-agent belief propagation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class BeliefNode:
    """
    Single belief node with Bayesian tracking.

    Represents: "Agent A believes X about target T"
    At order N: "A believes B believes ... (N times) ... X about T"
    """
    vec: torch.Tensor                    # Belief content (ontology-aligned)
    conf: float                          # Confidence [0,1]
    prior: Optional[torch.Tensor] = None # Prior belief before update
    evidence: List[torch.Tensor] = field(default_factory=list)
    timestamp: int = 0
    source: str = "inference"            # observation, inference, communication, prediction
    likelihood: float = 1.0              # P(evidence|belief) for Bayesian update
    order: int = 0                       # Recursion depth
    target_chain: List[int] = field(default_factory=list)  # Chain of targets

    def to_dict(self) -> Dict:
        """Serialize belief for logging/checkpointing."""
        return {
            'vec': self.vec.tolist() if isinstance(self.vec, torch.Tensor) else self.vec,
            'conf': self.conf,
            'timestamp': self.timestamp,
            'source': self.source,
            'order': self.order,
            'target_chain': self.target_chain,
            'likelihood': self.likelihood
        }

    @classmethod
    def from_dict(cls, data: Dict, device: str = 'cpu') -> 'BeliefNode':
        """Deserialize belief."""
        return cls(
            vec=torch.tensor(data['vec'], device=device),
            conf=data['conf'],
            timestamp=data['timestamp'],
            source=data['source'],
            order=data.get('order', 0),
            target_chain=data.get('target_chain', []),
            likelihood=data.get('likelihood', 1.0)
        )


class NestedBeliefStore:
    """
    Hierarchical storage for recursive beliefs.

    Structure:
    - Order 0: "I believe X about target T" - direct beliefs
    - Order 1: "I believe T believes X about T2" - 1st order ToM
    - Order 2: "I believe T1 believes T2 believes X" - 2nd order ToM
    - ...up to Order 5 (5th order ToM)

    Each order is indexed by target chain: [t1, t2, ..., tn]
    """

    def __init__(self, owner_id: int, ontology_dim: int, max_order: int = 5,
                 confidence_decay: float = 0.7, device: str = 'cpu'):
        self.owner_id = owner_id
        self.ontology_dim = ontology_dim
        self.max_order = max_order
        self.confidence_decay = confidence_decay
        self.device = device
        self.timestamp = 0

        # Nested dict: order -> target_key -> BeliefNode
        # target_key is tuple of target chain for hashability
        self.beliefs: Dict[int, Dict[Tuple[int, ...], BeliefNode]] = {
            order: {} for order in range(max_order + 1)
        }

        # History for trajectory extraction
        self.history: List[Dict] = []
        self.max_history = 1000

        # Bayesian priors
        self.prior_strength = 0.3  # Weight of prior vs evidence

    def _make_key(self, target_chain: List[int]) -> Tuple[int, ...]:
        """Convert target chain to hashable key."""
        return tuple(target_chain)

    def get_belief(self, order: int, target_chain: List[int]) -> Optional[BeliefNode]:
        """
        Retrieve belief at specific order and target chain.

        Examples:
        - get_belief(0, [5]) -> "What I believe about agent 5"
        - get_belief(1, [5, 3]) -> "What I believe agent 5 believes about agent 3"
        - get_belief(2, [5, 3, 7]) -> "What I believe 5 believes 3 believes about 7"
        """
        if order > self.max_order or order < 0:
            return None
        if len(target_chain) != order + 1:
            return None
        key = self._make_key(target_chain)
        return self.beliefs[order].get(key, None)

    def set_belief(self, order: int, target_chain: List[int],
                   content: torch.Tensor, confidence: float,
                   source: str = "inference", evidence: List[torch.Tensor] = None,
                   likelihood: float = 1.0) -> bool:
        """
        Set belief with confidence decay applied.

        Returns True if belief was set, False if invalid order/chain.
        """
        if order > self.max_order or order < 0:
            return False
        if len(target_chain) != order + 1:
            return False

        key = self._make_key(target_chain)

        # Apply confidence decay for higher orders
        decayed_conf = confidence * (self.confidence_decay ** order)

        # Get prior if exists
        prior = None
        if key in self.beliefs[order]:
            prior = self.beliefs[order][key].vec.clone()

        self.beliefs[order][key] = BeliefNode(
            vec=content.to(self.device),
            conf=decayed_conf,
            prior=prior,
            evidence=evidence or [],
            timestamp=self.timestamp,
            source=source,
            likelihood=likelihood,
            order=order,
            target_chain=list(target_chain)
        )

        # Record history
        self._record_update(order, target_chain, content, decayed_conf, source)

        return True

    def update_bayesian(self, order: int, target_chain: List[int],
                       observation: torch.Tensor,
                       observation_confidence: float = 1.0,
                       source: str = "observation") -> Optional[BeliefNode]:
        """
        Bayesian belief update:
        posterior = (prior * P(evidence|belief)) / P(evidence)

        Simplified as weighted combination:
        new_belief = alpha * prior + (1-alpha) * evidence
        new_conf = f(prior_conf, evidence_conf, likelihood)

        Args:
            order: Belief recursion order
            target_chain: Chain of target agents
            observation: New evidence tensor
            observation_confidence: Confidence in the observation
            source: Source of observation

        Returns:
            Updated BeliefNode or None if invalid
        """
        if order > self.max_order or order < 0:
            return None
        if len(target_chain) != order + 1:
            return None

        key = self._make_key(target_chain)

        # Get prior belief
        prior_node = self.beliefs[order].get(key, None)

        if prior_node is None:
            # No prior - use observation as initial belief
            return self._initialize_belief(order, target_chain, observation,
                                          observation_confidence, source)

        # Compute likelihood P(observation | prior_belief)
        # Using cosine similarity as proxy for likelihood
        prior_vec = prior_node.vec
        likelihood = self._compute_likelihood(prior_vec, observation)

        # Bayesian update
        # Weight by prior confidence and likelihood
        alpha = self.prior_strength * prior_node.conf
        beta = (1 - self.prior_strength) * observation_confidence * likelihood

        normalizer = alpha + beta + 1e-8
        alpha /= normalizer
        beta /= normalizer

        # Posterior belief vector
        posterior_vec = alpha * prior_vec + beta * observation

        # Posterior confidence
        # Increases if observation consistent with prior, decreases if inconsistent
        consistency = likelihood
        posterior_conf = (
            prior_node.conf * consistency +
            observation_confidence * (1 - consistency)
        ) * (self.confidence_decay ** order)
        posterior_conf = min(1.0, max(0.0, posterior_conf))

        # Update evidence list
        new_evidence = prior_node.evidence + [observation]
        if len(new_evidence) > 10:  # Keep only recent evidence
            new_evidence = new_evidence[-10:]

        self.beliefs[order][key] = BeliefNode(
            vec=posterior_vec,
            conf=posterior_conf,
            prior=prior_vec,
            evidence=new_evidence,
            timestamp=self.timestamp,
            source=source,
            likelihood=likelihood,
            order=order,
            target_chain=list(target_chain)
        )

        self._record_update(order, target_chain, posterior_vec, posterior_conf, source)

        return self.beliefs[order][key]

    def _compute_likelihood(self, prior: torch.Tensor, observation: torch.Tensor) -> float:
        """
        Compute P(observation | belief) using cosine similarity.
        Maps similarity from [-1, 1] to [0, 1] likelihood range.
        """
        prior_flat = prior.flatten()
        obs_flat = observation.flatten().to(prior.device)

        # Ensure same size
        min_size = min(len(prior_flat), len(obs_flat))
        prior_flat = prior_flat[:min_size]
        obs_flat = obs_flat[:min_size]

        # Cosine similarity
        dot = torch.dot(prior_flat, obs_flat)
        norm = (torch.norm(prior_flat) * torch.norm(obs_flat)) + 1e-8
        similarity = (dot / norm).item()

        # Map to [0, 1]
        likelihood = (similarity + 1) / 2
        return float(likelihood)

    def _initialize_belief(self, order: int, target_chain: List[int],
                          observation: torch.Tensor, confidence: float,
                          source: str) -> BeliefNode:
        """Initialize a new belief from observation."""
        key = self._make_key(target_chain)
        decayed_conf = confidence * (self.confidence_decay ** order)

        self.beliefs[order][key] = BeliefNode(
            vec=observation.to(self.device),
            conf=decayed_conf,
            prior=None,
            evidence=[observation],
            timestamp=self.timestamp,
            source=source,
            likelihood=1.0,
            order=order,
            target_chain=list(target_chain)
        )

        self._record_update(order, target_chain, observation, decayed_conf, source)
        return self.beliefs[order][key]

    def propagate_recursive(self, base_observation: torch.Tensor,
                           observed_agent: int,
                           max_propagation_order: int = None) -> Dict[int, List[BeliefNode]]:
        """
        Propagate belief update through recursive orders.

        When we observe agent A's state, we update:
        - Order 0: Our belief about A
        - Order 1: Our belief about what A believes about others
        - Order 2: Our belief about what A believes others believe
        - etc.

        Higher orders use increasingly uncertain projections.

        Args:
            base_observation: Observed state of agent
            observed_agent: ID of observed agent
            max_propagation_order: How deep to propagate (default: self.max_order)

        Returns:
            Dict mapping order -> list of updated BeliefNodes
        """
        if max_propagation_order is None:
            max_propagation_order = self.max_order

        updates = {order: [] for order in range(max_propagation_order + 1)}

        # Order 0: Direct observation
        node_0 = self.update_bayesian(
            order=0,
            target_chain=[observed_agent],
            observation=base_observation,
            observation_confidence=0.9,
            source="observation"
        )
        if node_0:
            updates[0].append(node_0)

        # Higher orders: Inferred beliefs
        for order in range(1, max_propagation_order + 1):
            # Generate target chains for this order
            # At order N, we need chains of length N+1
            target_chains = self._generate_target_chains(observed_agent, order)

            for chain in target_chains:
                # Project what this chain of agents might believe
                projected_belief = self._project_belief(base_observation, chain, order)

                # Confidence decreases with projection depth
                projection_conf = 0.9 * (0.7 ** order)

                node = self.update_bayesian(
                    order=order,
                    target_chain=chain,
                    observation=projected_belief,
                    observation_confidence=projection_conf,
                    source="inference"
                )
                if node:
                    updates[order].append(node)

        return updates

    def _generate_target_chains(self, start_agent: int, order: int,
                               max_chains: int = 5) -> List[List[int]]:
        """
        Generate plausible target chains for belief propagation.

        For order N starting from agent A, generates chains like:
        [A, B], [A, C], ... for order 1
        [A, B, C], [A, B, D], ... for order 2
        """
        if order == 0:
            return [[start_agent]]

        chains = []
        # Get existing beliefs to find known agents
        known_agents = set()
        for ord_beliefs in self.beliefs.values():
            for chain in ord_beliefs.keys():
                known_agents.update(chain)

        # Add start agent and some neighbors
        known_agents.add(start_agent)
        known_agents = list(known_agents)

        # Generate chains
        def generate_chain(current_chain: List[int], remaining_depth: int):
            if remaining_depth == 0:
                if len(current_chain) == order + 1:
                    chains.append(current_chain)
                return

            for agent in known_agents:
                if len(chains) >= max_chains:
                    return
                new_chain = current_chain + [agent]
                generate_chain(new_chain, remaining_depth - 1)

        generate_chain([start_agent], order)
        return chains[:max_chains]

    def _project_belief(self, base_observation: torch.Tensor,
                       target_chain: List[int], order: int) -> torch.Tensor:
        """
        Project what a chain of agents might believe.

        Uses existing beliefs and adds uncertainty based on order.
        """
        # Start with base observation
        projected = base_observation.clone()

        # Add increasing noise for higher-order projections
        noise_scale = 0.1 * order
        noise = torch.randn_like(projected) * noise_scale
        projected = projected + noise

        # If we have prior beliefs about intermediate agents, incorporate them
        for i in range(1, len(target_chain)):
            intermediate_chain = target_chain[:i+1]
            intermediate_order = len(intermediate_chain) - 1

            if intermediate_order <= order:
                prior = self.get_belief(intermediate_order, intermediate_chain)
                if prior is not None:
                    # Blend with prior knowledge
                    projected = 0.7 * projected + 0.3 * prior.vec

        return projected

    def extract_trajectory(self, order: int, target_chain: List[int],
                          num_steps: int = 10) -> List[Dict]:
        """
        Extract belief trajectory over time for visualization/analysis.

        Returns list of {timestamp, vec, conf, source} dicts.
        """
        key = self._make_key(target_chain)
        trajectory = []

        for record in self.history[-num_steps*10:]:  # Search recent history
            if record.get('order') == order and record.get('target_chain') == list(target_chain):
                trajectory.append({
                    'timestamp': record['timestamp'],
                    'vec': record['vec'],
                    'conf': record['conf'],
                    'source': record['source']
                })

        # Sort by timestamp and take most recent
        trajectory.sort(key=lambda x: x['timestamp'])
        return trajectory[-num_steps:]

    def compute_divergence(self, other: 'NestedBeliefStore',
                          order: int = None) -> Dict[str, float]:
        """
        Compute belief divergence between two agents' belief stores.

        Useful for:
        - Measuring belief alignment/disagreement
        - Detecting deception (large divergence in communicated vs actual beliefs)
        - Evaluating ToM accuracy (comparing inferred vs actual beliefs)

        Args:
            other: Another agent's belief store
            order: Specific order to compare (None = all orders)

        Returns:
            Dict with divergence metrics
        """
        divergences = {
            'total': 0.0,
            'by_order': {},
            'max_divergence': 0.0,
            'min_divergence': float('inf'),
            'num_compared': 0
        }

        orders_to_compare = [order] if order is not None else range(self.max_order + 1)

        for ord in orders_to_compare:
            order_div = []

            # Compare beliefs that exist in both stores
            for key in self.beliefs[ord].keys():
                if key in other.beliefs[ord]:
                    my_belief = self.beliefs[ord][key]
                    their_belief = other.beliefs[ord][key]

                    # Vector divergence (L2 distance)
                    vec_div = torch.norm(my_belief.vec - their_belief.vec).item()

                    # Confidence divergence
                    conf_div = abs(my_belief.conf - their_belief.conf)

                    # Combined divergence
                    combined = vec_div + conf_div
                    order_div.append(combined)

                    divergences['max_divergence'] = max(divergences['max_divergence'], combined)
                    divergences['min_divergence'] = min(divergences['min_divergence'], combined)
                    divergences['num_compared'] += 1

            if order_div:
                divergences['by_order'][ord] = np.mean(order_div)
                divergences['total'] += sum(order_div)

        if divergences['num_compared'] > 0:
            divergences['total'] /= divergences['num_compared']

        if divergences['min_divergence'] == float('inf'):
            divergences['min_divergence'] = 0.0

        return divergences

    def get_confidence_matrix(self, order: int) -> torch.Tensor:
        """Get confidence values for all beliefs at given order."""
        if order > self.max_order:
            return torch.zeros(1)

        beliefs = self.beliefs[order]
        if not beliefs:
            return torch.zeros(1)

        # Find max target ID for matrix size
        max_target = 0
        for chain in beliefs.keys():
            if chain:
                max_target = max(max_target, max(chain))

        conf_vec = torch.zeros(max_target + 1, device=self.device)
        for chain, node in beliefs.items():
            if chain:
                # Use last target in chain as index
                conf_vec[chain[-1]] = node.conf

        return conf_vec

    def get_all_beliefs_flat(self) -> List[BeliefNode]:
        """Get all beliefs as flat list for iteration."""
        all_beliefs = []
        for order_beliefs in self.beliefs.values():
            all_beliefs.extend(order_beliefs.values())
        return all_beliefs

    def _record_update(self, order: int, target_chain: List[int],
                      vec: torch.Tensor, conf: float, source: str):
        """Record belief update in history."""
        self.history.append({
            'timestamp': self.timestamp,
            'order': order,
            'target_chain': list(target_chain),
            'vec': vec.detach().cpu().numpy().tolist(),
            'conf': conf,
            'source': source
        })

        # Trim history if too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history//2:]

    def step(self):
        """Advance timestamp."""
        self.timestamp += 1

    def get_summary(self) -> Dict:
        """Get summary statistics of belief store."""
        total_beliefs = sum(len(b) for b in self.beliefs.values())
        avg_conf_by_order = {}

        for order in range(self.max_order + 1):
            if self.beliefs[order]:
                avg_conf_by_order[order] = np.mean([
                    n.conf for n in self.beliefs[order].values()
                ])
            else:
                avg_conf_by_order[order] = 0.0

        return {
            'owner_id': self.owner_id,
            'total_beliefs': total_beliefs,
            'beliefs_by_order': {o: len(b) for o, b in self.beliefs.items()},
            'avg_confidence_by_order': avg_conf_by_order,
            'timestamp': self.timestamp,
            'history_length': len(self.history)
        }


class RecursiveBeliefState(NestedBeliefStore):
    """
    Backward-compatible wrapper around NestedBeliefStore.
    Maintains the original API while using enhanced implementation.
    """

    def __init__(self, agent_id: int, ontology_dim: int, max_order: int = 5):
        super().__init__(
            owner_id=agent_id,
            ontology_dim=ontology_dim,
            max_order=max_order
        )
        self.agent_id = agent_id

    def update_belief(self, order: int, target: int, content: torch.Tensor,
                     confidence: float = 1.0, evidence: List = None,
                     source: str = "inference"):
        """Backward-compatible update method."""
        # Build target chain for the order (length must be order + 1)
        # For backward compatibility, create chain with target repeated
        target_chain = [target] * (order + 1)
        self.set_belief(order, target_chain, content, confidence, source, evidence)

    def get_belief(self, order: int, target) -> Optional[BeliefNode]:
        """
        Backward-compatible get_belief that accepts int or list target.
        """
        # Handle both int (old API) and list (new API)
        if isinstance(target, int):
            # Create chain with correct length for order
            target_chain = [target] * (order + 1)
        else:
            target_chain = list(target)
        return super().get_belief(order, target_chain)

    def query_recursive_belief(self, belief_path: List[int]) -> Optional[BeliefNode]:
        """Query using belief path (list of agent IDs)."""
        order = len(belief_path) - 1
        return self.get_belief(order, belief_path)


class BeliefNetwork:
    """
    Network of recursive belief states for multiple agents.

    Manages cross-agent belief propagation and provides
    network-wide analysis capabilities.
    """

    def __init__(self, num_agents: int, ontology_dim: int, max_order: int = 5,
                 device: str = 'cpu'):
        self.num_agents = num_agents
        self.ontology_dim = ontology_dim
        self.max_order = max_order
        self.device = device

        # Create belief store for each agent
        self.agent_beliefs: List[NestedBeliefStore] = [
            NestedBeliefStore(i, ontology_dim, max_order, device=device)
            for i in range(num_agents)
        ]

        # Track cross-agent communications
        self.communication_history: List[Dict] = []

    def get_agent_beliefs(self, agent_id: int) -> NestedBeliefStore:
        """Get belief store for specific agent."""
        if 0 <= agent_id < self.num_agents:
            return self.agent_beliefs[agent_id]
        raise ValueError(f"Invalid agent_id: {agent_id}")

    def propagate_observation(self, observer_id: int, observed_id: int,
                             observation: torch.Tensor,
                             max_order: int = None) -> Dict[int, List[BeliefNode]]:
        """
        Propagate an observation through an agent's belief hierarchy.

        When agent A observes agent B, updates:
        - A's 0th order belief about B
        - A's 1st order belief about B's beliefs
        - etc.
        """
        if observer_id >= self.num_agents or observed_id >= self.num_agents:
            return {}

        return self.agent_beliefs[observer_id].propagate_recursive(
            base_observation=observation,
            observed_agent=observed_id,
            max_propagation_order=max_order
        )

    def communicate_belief(self, sender_id: int, receiver_id: int,
                          order: int, target_chain: List[int],
                          honesty: float = 1.0) -> Optional[BeliefNode]:
        """
        Agent communicates a belief to another agent.

        Args:
            sender_id: Communicating agent
            receiver_id: Receiving agent
            order: Order of belief being communicated
            target_chain: Target chain of the belief
            honesty: How honestly the belief is communicated (1.0 = fully honest)

        Returns:
            Updated belief node in receiver's store
        """
        # Get sender's actual belief
        sender_belief = self.agent_beliefs[sender_id].get_belief(order, target_chain)
        if sender_belief is None:
            return None

        # Apply honesty transformation
        communicated_vec = sender_belief.vec.clone()
        if honesty < 1.0:
            # Add noise proportional to dishonesty
            noise = torch.randn_like(communicated_vec) * (1 - honesty) * 0.5
            communicated_vec = communicated_vec + noise

        # Receiver updates their belief based on communication
        # Lower confidence for communicated beliefs
        comm_confidence = sender_belief.conf * 0.7 * honesty

        updated = self.agent_beliefs[receiver_id].update_bayesian(
            order=order,
            target_chain=target_chain,
            observation=communicated_vec,
            observation_confidence=comm_confidence,
            source="communication"
        )

        # Record communication
        self.communication_history.append({
            'sender': sender_id,
            'receiver': receiver_id,
            'order': order,
            'target_chain': target_chain,
            'honesty': honesty,
            'timestamp': self.agent_beliefs[sender_id].timestamp
        })

        return updated

    def compute_tom_accuracy(self, agent_id: int, target_id: int,
                            order: int = 1) -> float:
        """
        Compute how accurately agent_id models target_id's beliefs.

        Compares:
        - What agent_id thinks target_id believes (order N from agent_id's perspective)
        - What target_id actually believes (order N-1 from target_id's perspective)

        Args:
            agent_id: Agent doing the modeling
            target_id: Agent being modeled
            order: Order of ToM to evaluate (1 = 1st order, etc.)

        Returns:
            Accuracy score [0, 1], higher is better
        """
        if order < 1:
            return 0.0

        # Get what agent thinks target believes
        agent_store = self.agent_beliefs[agent_id]
        target_store = self.agent_beliefs[target_id]

        accuracies = []

        # Compare beliefs that agent models vs target's actual beliefs
        for chain, inferred in agent_store.beliefs[order].items():
            if chain[0] != target_id:
                continue

            # Target's actual belief at order-1
            actual_chain = chain[1:]  # Remove first element (target itself)
            if len(actual_chain) != order:
                continue

            actual = target_store.get_belief(order - 1, list(actual_chain))
            if actual is None:
                continue

            # Compute similarity
            similarity = agent_store._compute_likelihood(inferred.vec, actual.vec)
            accuracies.append(similarity)

        return np.mean(accuracies) if accuracies else 0.5

    def get_network_summary(self) -> Dict:
        """Get summary of entire belief network."""
        return {
            'num_agents': self.num_agents,
            'ontology_dim': self.ontology_dim,
            'max_order': self.max_order,
            'agent_summaries': [
                store.get_summary() for store in self.agent_beliefs
            ],
            'total_communications': len(self.communication_history)
        }

    def compute_belief_consensus(self, target_chain: List[int],
                                 order: int) -> Dict[str, Any]:
        """
        Compute how much agents agree about a particular belief.

        Returns consensus metrics for the specified belief across all agents.
        """
        beliefs = []
        confidences = []

        for store in self.agent_beliefs:
            belief = store.get_belief(order, target_chain)
            if belief is not None:
                beliefs.append(belief.vec)
                confidences.append(belief.conf)

        if not beliefs:
            return {'consensus': 0.0, 'num_agents': 0}

        # Stack beliefs
        belief_matrix = torch.stack(beliefs)

        # Compute centroid
        centroid = belief_matrix.mean(dim=0)

        # Compute variance from centroid
        variance = torch.mean(torch.norm(belief_matrix - centroid, dim=1)).item()

        # Consensus is inverse of variance (normalized)
        consensus = 1.0 / (1.0 + variance)

        return {
            'consensus': consensus,
            'variance': variance,
            'centroid': centroid,
            'avg_confidence': np.mean(confidences),
            'num_agents': len(beliefs)
        }

    def step(self):
        """Advance time for all agent belief stores."""
        for store in self.agent_beliefs:
            store.step()

    def add_agent(self) -> int:
        """Add a new agent to the network."""
        new_id = self.num_agents
        self.agent_beliefs.append(
            NestedBeliefStore(new_id, self.ontology_dim, self.max_order,
                             device=self.device)
        )
        self.num_agents += 1
        return new_id

    def remove_agent(self, agent_id: int):
        """Remove an agent from the network (marks as inactive)."""
        if 0 <= agent_id < self.num_agents:
            # Clear the agent's beliefs but keep the slot
            self.agent_beliefs[agent_id] = NestedBeliefStore(
                agent_id, self.ontology_dim, self.max_order, device=self.device
            )


class BeliefEncoder(nn.Module):
    """
    Neural encoder for belief states.

    Converts raw observations into belief-space representations
    that can be stored in NestedBeliefStore.
    """

    def __init__(self, input_dim: int, ontology_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.ontology_dim = ontology_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ontology_dim),
            nn.Tanh()  # Bound output to [-1, 1]
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation into belief space.

        Returns:
            belief_vec: Ontology-aligned belief vector
            confidence: Confidence in the encoding
        """
        belief_vec = self.encoder(x)
        confidence = self.confidence_head(x)
        return belief_vec, confidence.squeeze(-1)


class BeliefDecoder(nn.Module):
    """
    Decode beliefs back to action/prediction space.
    """

    def __init__(self, ontology_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(ontology_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, belief: torch.Tensor) -> torch.Tensor:
        """Decode belief to output space."""
        return self.decoder(belief)
