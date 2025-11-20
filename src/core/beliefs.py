"""
Recursive Belief Architecture for ToM-NAS - Supports 5th order beliefs
"""
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

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
        
    def update_belief(self, order: int, target: int, content: torch.Tensor,
                     confidence: float = 1.0, evidence: List = None, source: str = "inference"):
        if order > self.max_order:
            return
        decayed_confidence = confidence * (self.confidence_decay ** order)
        self.beliefs[order][target] = Belief(
            content=content, confidence=decayed_confidence, timestamp=self.timestamp,
            evidence=evidence or [], source=source
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
        self.num_agents = num_agents
        self.ontology_dim = ontology_dim
        self.max_order = max_order
        self.agent_beliefs = [
            RecursiveBeliefState(i, ontology_dim, max_order)
            for i in range(num_agents)
        ]
