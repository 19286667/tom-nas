"""
Recursive Belief Architecture for ToM-NAS - Supports 5th order beliefs

This module provides recursive belief modeling for Theory of Mind:
- 0th order: Direct observations (I see X)
- 1st order: Basic beliefs (I believe X)
- 2nd order: Meta-beliefs (I believe you believe X)
- 3rd-5th order: Higher-order nested beliefs
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class Belief:
    content: torch.Tensor
    confidence: float
    timestamp: int
    evidence: List[torch.Tensor]
    source: str


class RecursiveBeliefState:
    """Recursive belief structure supporting up to 5th-order ToM."""

    def __init__(self, agent_id: int, ontology_dim: int, max_order: int = 5):
        self.agent_id = agent_id
        self.ontology_dim = ontology_dim
        self.max_order = max_order
        self.confidence_decay = 0.7
        self.timestamp = 0
        self.beliefs = defaultdict(lambda: defaultdict(lambda: None))

    def update_belief(
        self,
        order: int,
        target: int,
        content: torch.Tensor,
        confidence: float = 1.0,
        evidence: List = None,
        source: str = "inference",
    ):
        if order > self.max_order:
            return
        decayed_confidence = confidence * (self.confidence_decay**order)
        self.beliefs[order][target] = Belief(
            content=content,
            confidence=decayed_confidence,
            timestamp=self.timestamp,
            evidence=evidence or [],
            source=source,
        )

    def get_belief(self, order: int, target: int) -> Optional[Belief]:
        if order > self.max_order:
            return None
        return self.beliefs[order].get(target, None)

    def query_recursive_belief(self, belief_path: List[int]) -> Optional[Belief]:
        order = len(belief_path) - 1
        if order > self.max_order or order < 0:
            return None
        target = belief_path[-1]
        return self.get_belief(order, target)

    def get_confidence_matrix(self, order: int) -> torch.Tensor:
        targets = list(self.beliefs[order].keys())
        if not targets:
            return torch.zeros(1, 1)
        conf_matrix = torch.zeros(max(targets) + 1)
        for target in targets:
            belief = self.beliefs[order][target]
            if belief:
                conf_matrix[target] = belief.confidence
        return conf_matrix


class BeliefNetwork:
    """Network of recursive belief states for multiple agents."""

    def __init__(self, num_agents: int, ontology_dim: int, max_order: int = 5):
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1")
        if ontology_dim < 1:
            raise ValueError("ontology_dim must be at least 1")
        if max_order < 0 or max_order > 10:
            raise ValueError("max_order must be between 0 and 10")

        self.num_agents = num_agents
        self.ontology_dim = ontology_dim
        self.max_order = max_order
        self.agent_beliefs = [RecursiveBeliefState(i, ontology_dim, max_order) for i in range(num_agents)]

    def get_agent_belief_state(self, agent_id: int) -> Optional[RecursiveBeliefState]:
        """Get the belief state for a specific agent."""
        if 0 <= agent_id < self.num_agents:
            return self.agent_beliefs[agent_id]
        return None

    def update_agent_belief(
        self,
        agent_id: int,
        order: int,
        target: int,
        content: torch.Tensor,
        confidence: float = 1.0,
        source: str = "inference",
    ) -> bool:
        """Update a specific agent's belief about a target."""
        if 0 <= agent_id < self.num_agents:
            self.agent_beliefs[agent_id].update_belief(order, target, content, confidence, source=source)
            return True
        return False

    def increment_all_timestamps(self):
        """Increment the timestamp for all agent belief states."""
        for belief_state in self.agent_beliefs:
            belief_state.timestamp += 1

    def get_all_beliefs_at_order(self, order: int) -> Dict[int, Dict]:
        """Get all beliefs at a specific order across all agents."""
        result = {}
        for agent_id, belief_state in enumerate(self.agent_beliefs):
            beliefs_at_order = {}
            for target in range(self.num_agents):
                belief = belief_state.get_belief(order, target)
                if belief is not None:
                    beliefs_at_order[target] = {
                        "confidence": belief.confidence,
                        "timestamp": belief.timestamp,
                        "source": belief.source,
                    }
            if beliefs_at_order:
                result[agent_id] = beliefs_at_order
        return result
