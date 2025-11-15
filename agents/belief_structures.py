
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class NestedBelief:
    """Recursive belief structure"""
    order: int
    holder_id: int
    content: torch.Tensor
    confidence: float = 1.0
    nested: Optional["NestedBelief"] = None
    
    def to_string(self):
        return f"Agent {self.holder_id} belief (order {self.order}): confidence={self.confidence:.2f}"

class RecursiveBeliefTracker:
    """Tracks nested beliefs up to 5th order"""
    
    def __init__(self, n_agents=4, max_order=5, state_dim=55):
        self.n_agents = n_agents
        self.max_order = max_order
        self.state_dim = state_dim
        self.beliefs = {}
        
        # Initialize beliefs
        for agent in range(n_agents):
            for order in range(max_order + 1):
                key = (agent, order)
                self.beliefs[key] = torch.rand(state_dim)
    
    def update_belief(self, agent_id: int, order: int, observation: torch.Tensor):
        """Update belief based on observation"""
        key = (agent_id, order)
        if key in self.beliefs:
            # Simple momentum update
            old = self.beliefs[key]
            self.beliefs[key] = old * 0.9 + observation * 0.1
    
    def query_belief(self, agent_id: int, order: int) -> torch.Tensor:
        """Query belief at given order"""
        key = (agent_id, order)
        return self.beliefs.get(key, torch.zeros(self.state_dim))
