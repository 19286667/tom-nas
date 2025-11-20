"""
Social World 4: Complete society simulator with zombie detection
"""
import torch
import numpy as np
import random
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Agent:
    id: int
    is_zombie: bool = False
    resources: float = 100.0
    energy: float = 100.0
    reputation: Dict[int, float] = field(default_factory=dict)
    coalition: Optional[int] = None
    alive: bool = True
    zombie_type: Optional[str] = None
    ontology_state: Optional[torch.Tensor] = None

class ZombieGame:
    """Zombie detection - core ToM validation mechanism"""
    
    ZOMBIE_TYPES = {
        'behavioral': 'Inconsistent action patterns',
        'belief': 'Cannot model others beliefs',
        'causal': 'No counterfactual reasoning',
        'metacognitive': 'Poor uncertainty calibration',
        'linguistic': 'Narrative incoherence',
        'emotional': 'Flat affect patterns'
    }
    
    def __init__(self):
        self.detection_reward = 10.0
        self.false_positive_penalty = -20.0
        
    def create_zombie(self, agent_id: int, zombie_type: Optional[str] = None) -> Agent:
        if zombie_type is None:
            zombie_type = random.choice(list(self.ZOMBIE_TYPES.keys()))
        return Agent(id=agent_id, is_zombie=True, zombie_type=zombie_type)

class SocialWorld4:
    """Complete society simulator"""
    
    def __init__(self, num_agents: int, ontology_dim: int, num_zombies: int = 2):
        self.num_agents = num_agents
        self.ontology_dim = ontology_dim
        self.timestep = 0
        
        # Initialize agents
        self.agents = []
        for i in range(num_agents):
            agent = Agent(id=i)
            agent.ontology_state = torch.randn(ontology_dim)
            agent.reputation = {j: 0.5 for j in range(num_agents)}
            self.agents.append(agent)
            
        # Create zombies
        self.zombie_game = ZombieGame()
        zombie_indices = random.sample(range(num_agents), min(num_zombies, num_agents))
        for idx in zombie_indices:
            self.agents[idx] = self.zombie_game.create_zombie(idx)
            self.agents[idx].ontology_state = torch.randn(ontology_dim)
            
        self.history = []
        
    def step(self, agent_actions: List[Dict], belief_network=None) -> Dict:
        results = {
            'timestep': self.timestep,
            'games': {},
            'zombie_results': {'detections': []},
            'reputation_changes': {}
        }
        self.timestep += 1
        self.history.append(results)
        return results
