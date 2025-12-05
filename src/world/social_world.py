"""
Social World 4: Complete society simulator with zombie detection
"""
import torch
import numpy as np
import random
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math

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

class ZombieBehavior:
    """Generates zombie-specific behavioral patterns for detection tasks.

    Each zombie type exhibits distinct patterns that ToM agents should learn to detect:
    - Behavioral: Random/inconsistent actions that don't match context
    - Belief: Actions ignore others' perspectives
    - Causal: No planning or anticipation
    - Metacognitive: Overconfident or underconfident choices
    - Linguistic: Incoherent communication patterns
    - Emotional: Flat or inappropriate emotional responses
    """

    @staticmethod
    def generate_action(zombie_type: str, context: Dict) -> Dict:
        """Generate zombie-appropriate action based on type"""
        if zombie_type == 'behavioral':
            # Inconsistent random actions regardless of context
            return {
                'type': random.choice(['cooperate', 'defect', 'share', 'communicate']),
                'consistency_score': random.random() * 0.3,  # Low consistency
            }

        elif zombie_type == 'belief':
            # Always acts on own state, ignores others
            # Defects when should cooperate based on reciprocity
            partner_reputation = context.get('partner_reputation', 0.5)
            # Belief zombie ignores reputation, acts randomly
            return {
                'type': 'defect' if random.random() < 0.6 else 'cooperate',
                'ignores_other_beliefs': True,
            }

        elif zombie_type == 'causal':
            # No temporal reasoning - purely reactive
            # Doesn't anticipate consequences
            return {
                'type': 'share' if context.get('own_resources', 0) > 50 else 'defect',
                'amount': context.get('own_resources', 50) * 0.8,  # Shares too much
                'no_planning': True,
            }

        elif zombie_type == 'metacognitive':
            # Poor uncertainty calibration - always overconfident
            return {
                'type': 'detect_zombie',
                'suspect': random.randint(0, context.get('num_agents', 5) - 1),
                'confidence': 0.99,  # Always overconfident
            }

        elif zombie_type == 'linguistic':
            # Incoherent communication
            message = torch.randn(context.get('ontology_dim', 181)) * 0.5
            # Add noise that doesn't match true state
            return {
                'type': 'communicate',
                'message': message,
                'coherence_score': random.random() * 0.2,  # Low coherence
            }

        elif zombie_type == 'emotional':
            # Flat affect - no emotional variation
            return {
                'type': random.choice(['cooperate', 'defect']),
                'emotional_state': 0.5,  # Always flat
                'affect_variance': 0.01,  # No variation
            }

        return {'type': 'cooperate'}

    @staticmethod
    def get_behavioral_signature(zombie_type: str) -> Dict:
        """Get the behavioral signature patterns for zombie type"""
        signatures = {
            'behavioral': {
                'action_entropy': 0.95,  # High randomness
                'context_sensitivity': 0.1,  # Low response to context
                'temporal_consistency': 0.2,
            },
            'belief': {
                'perspective_taking': 0.0,  # Never considers others
                'reputation_response': 0.1,
                'reciprocity': 0.1,
            },
            'causal': {
                'planning_horizon': 0,
                'consequence_anticipation': 0.0,
                'delayed_gratification': 0.0,
            },
            'metacognitive': {
                'confidence_calibration': 0.1,  # Poorly calibrated
                'uncertainty_awareness': 0.1,
                'learning_from_mistakes': 0.1,
            },
            'linguistic': {
                'message_coherence': 0.1,
                'pragmatic_relevance': 0.1,
                'narrative_consistency': 0.1,
            },
            'emotional': {
                'affect_range': 0.05,
                'emotional_appropriateness': 0.1,
                'empathy_response': 0.0,
            },
        }
        return signatures.get(zombie_type, {})


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
        self.behavior_generator = ZombieBehavior()
        self.zombie_action_history = defaultdict(list)

    def create_zombie(self, agent_id: int, zombie_type: Optional[str] = None) -> Agent:
        if zombie_type is None:
            zombie_type = random.choice(list(self.ZOMBIE_TYPES.keys()))
        return Agent(id=agent_id, is_zombie=True, zombie_type=zombie_type)

    def get_zombie_action(self, agent: Agent, context: Dict) -> Dict:
        """Get action for a zombie agent based on its type"""
        if not agent.is_zombie or not agent.zombie_type:
            return {'type': 'cooperate'}

        action = ZombieBehavior.generate_action(agent.zombie_type, context)
        self.zombie_action_history[agent.id].append(action)
        return action

    def compute_detection_difficulty(self, zombie_type: str, num_observations: int) -> float:
        """Compute how difficult it should be to detect this zombie type"""
        base_difficulty = {
            'behavioral': 0.3,  # Easiest - obvious inconsistency
            'belief': 0.5,
            'causal': 0.6,
            'metacognitive': 0.7,
            'linguistic': 0.5,
            'emotional': 0.8,  # Hardest - subtle
        }
        difficulty = base_difficulty.get(zombie_type, 0.5)
        # More observations make detection easier
        observation_factor = 1.0 / (1.0 + num_observations * 0.1)
        return difficulty * observation_factor

class SocialWorld4:
    """Complete society simulator with 4 game types"""

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
            self.agents[idx].reputation = {j: 0.5 for j in range(num_agents)}

        self.history = []
        self.coalitions = {}  # coalition_id -> set of agent ids
        self.next_coalition_id = 0

        # Game configuration
        self.cooperation_reward = 3.0
        self.defection_reward = 5.0
        self.mutual_defection_reward = 1.0
        self.resource_transfer_cost = 0.1

    def play_cooperation_game(self, agent1_id: int, agent2_id: int,
                             action1: str, action2: str) -> Dict:
        """Prisoner's Dilemma style cooperation game"""
        payoffs = {
            ('cooperate', 'cooperate'): (self.cooperation_reward, self.cooperation_reward),
            ('cooperate', 'defect'): (0.0, self.defection_reward),
            ('defect', 'cooperate'): (self.defection_reward, 0.0),
            ('defect', 'defect'): (self.mutual_defection_reward, self.mutual_defection_reward)
        }

        payoff1, payoff2 = payoffs.get((action1, action2), (0.0, 0.0))

        # Update resources
        self.agents[agent1_id].resources += payoff1
        self.agents[agent2_id].resources += payoff2

        # Update reputations based on actions
        if action1 == 'cooperate':
            self.agents[agent2_id].reputation[agent1_id] = min(1.0,
                self.agents[agent2_id].reputation[agent1_id] + 0.1)
        else:
            self.agents[agent2_id].reputation[agent1_id] = max(0.0,
                self.agents[agent2_id].reputation[agent1_id] - 0.1)

        if action2 == 'cooperate':
            self.agents[agent1_id].reputation[agent2_id] = min(1.0,
                self.agents[agent1_id].reputation[agent2_id] + 0.1)
        else:
            self.agents[agent1_id].reputation[agent2_id] = max(0.0,
                self.agents[agent1_id].reputation[agent2_id] - 0.1)

        return {
            'game_type': 'cooperation',
            'players': (agent1_id, agent2_id),
            'actions': (action1, action2),
            'payoffs': (payoff1, payoff2)
        }

    def play_communication_game(self, sender_id: int, receiver_id: int,
                                message: torch.Tensor, true_state: torch.Tensor) -> Dict:
        """Communication game - sender describes state, receiver infers"""
        # Measure communication accuracy
        message_quality = 1.0 - torch.mean(torch.abs(message - true_state)).item()

        # Reward accurate communication
        reward = message_quality * 2.0
        self.agents[sender_id].resources += reward * 0.5
        self.agents[receiver_id].resources += reward * 0.5

        # Update reputation based on communication honesty
        self.agents[receiver_id].reputation[sender_id] = (
            0.9 * self.agents[receiver_id].reputation[sender_id] +
            0.1 * message_quality
        )

        return {
            'game_type': 'communication',
            'sender': sender_id,
            'receiver': receiver_id,
            'message_quality': message_quality,
            'reward': reward
        }

    def play_resource_sharing_game(self, giver_id: int, receiver_id: int,
                                   amount: float) -> Dict:
        """Resource sharing with cost"""
        amount = max(0.0, min(amount, self.agents[giver_id].resources))

        if amount > 0:
            cost = amount * self.resource_transfer_cost
            self.agents[giver_id].resources -= (amount + cost)
            self.agents[receiver_id].resources += amount

            # Reputation boost for generosity
            self.agents[receiver_id].reputation[giver_id] = min(1.0,
                self.agents[receiver_id].reputation[giver_id] + amount * 0.01)

        return {
            'game_type': 'resource_sharing',
            'giver': giver_id,
            'receiver': receiver_id,
            'amount': amount,
            'cost': amount * self.resource_transfer_cost if amount > 0 else 0.0
        }

    def attempt_zombie_detection(self, detector_id: int, suspect_id: int) -> Dict:
        """Attempt to detect if an agent is a zombie"""
        is_zombie = self.agents[suspect_id].is_zombie

        # Simple detection based on reputation and observation
        # In a full implementation, this would use ToM reasoning
        detection_accuracy = self.agents[detector_id].reputation.get(suspect_id, 0.5)
        detected_correctly = (is_zombie and detection_accuracy < 0.3) or \
                           (not is_zombie and detection_accuracy > 0.7)

        if detected_correctly and is_zombie:
            reward = self.zombie_game.detection_reward
            self.agents[detector_id].resources += reward
            result = 'correct_detection'
        elif not is_zombie and detection_accuracy < 0.5:
            # False positive
            penalty = self.zombie_game.false_positive_penalty
            self.agents[detector_id].resources += penalty
            result = 'false_positive'
        else:
            reward = 0.0
            result = 'no_detection'

        return {
            'game_type': 'zombie_detection',
            'detector': detector_id,
            'suspect': suspect_id,
            'is_zombie': is_zombie,
            'result': result,
            'reward': reward if 'reward' in locals() else 0.0
        }

    def form_coalition(self, member_ids: List[int]) -> int:
        """Form a new coalition"""
        coalition_id = self.next_coalition_id
        self.next_coalition_id += 1
        self.coalitions[coalition_id] = set(member_ids)

        for agent_id in member_ids:
            self.agents[agent_id].coalition = coalition_id

        return coalition_id

    def leave_coalition(self, agent_id: int):
        """Agent leaves their current coalition"""
        coalition_id = self.agents[agent_id].coalition
        if coalition_id is not None and coalition_id in self.coalitions:
            self.coalitions[coalition_id].discard(agent_id)
            if len(self.coalitions[coalition_id]) == 0:
                del self.coalitions[coalition_id]
        self.agents[agent_id].coalition = None

    def update_agent_states(self):
        """Update agent psychological states based on resources and reputation"""
        for agent in self.agents:
            if not agent.alive:
                continue

            # Decay energy
            agent.energy = max(0.0, agent.energy - 1.0)

            # Resources affect ontology state
            if agent.ontology_state is not None:
                # Update resource-related dimensions (simplified)
                resource_level = min(1.0, agent.resources / 200.0)
                energy_level = agent.energy / 100.0

                # Average reputation
                avg_reputation = np.mean(list(agent.reputation.values())) if agent.reputation else 0.5

                # Update state (first few dimensions related to resources/energy)
                agent.ontology_state[0] = resource_level
                agent.ontology_state[1] = energy_level
                if len(agent.ontology_state) > 2:
                    agent.ontology_state[2] = avg_reputation

    def step(self, agent_actions: List[Dict], belief_network=None) -> Dict:
        """Execute one timestep of the social world"""
        results = {
            'timestep': self.timestep,
            'games': [],
            'zombie_detections': [],
            'reputation_changes': {},
            'coalition_changes': [],
            'agent_states': []
        }

        # Process actions
        for i, action_dict in enumerate(agent_actions):
            if i >= len(self.agents) or not self.agents[i].alive:
                continue

            action_type = action_dict.get('type', 'none')

            if action_type == 'cooperate' or action_type == 'defect':
                # Cooperation game with random partner
                partner = action_dict.get('partner')
                if partner is None:
                    partner = random.choice([j for j in range(self.num_agents)
                                           if j != i and self.agents[j].alive])
                partner_action = agent_actions[partner].get('type', 'defect')
                game_result = self.play_cooperation_game(i, partner, action_type, partner_action)
                results['games'].append(game_result)

            elif action_type == 'communicate':
                receiver = action_dict.get('receiver', (i + 1) % self.num_agents)
                message = action_dict.get('message', torch.zeros(self.ontology_dim))
                true_state = self.agents[i].ontology_state
                game_result = self.play_communication_game(i, receiver, message, true_state)
                results['games'].append(game_result)

            elif action_type == 'share':
                receiver = action_dict.get('receiver', (i + 1) % self.num_agents)
                amount = action_dict.get('amount', 10.0)
                game_result = self.play_resource_sharing_game(i, receiver, amount)
                results['games'].append(game_result)

            elif action_type == 'detect_zombie':
                suspect = action_dict.get('suspect', (i + 1) % self.num_agents)
                detection_result = self.attempt_zombie_detection(i, suspect)
                results['zombie_detections'].append(detection_result)

            elif action_type == 'form_coalition':
                members = action_dict.get('members', [i])
                coalition_id = self.form_coalition(members)
                results['coalition_changes'].append({
                    'action': 'form',
                    'coalition_id': coalition_id,
                    'members': members
                })

            elif action_type == 'leave_coalition':
                self.leave_coalition(i)
                results['coalition_changes'].append({
                    'action': 'leave',
                    'agent_id': i
                })

        # Update all agent states
        self.update_agent_states()

        # Record current agent states
        for agent in self.agents:
            results['agent_states'].append({
                'id': agent.id,
                'resources': agent.resources,
                'energy': agent.energy,
                'coalition': agent.coalition,
                'is_zombie': agent.is_zombie,
                'alive': agent.alive
            })

        self.timestep += 1
        results['timestep'] = self.timestep  # Update to reflect new timestep
        self.history.append(results)
        return results

    def get_observation(self, agent_id: int) -> Dict:
        """Get what an agent can observe about the world"""
        agent = self.agents[agent_id]

        # Observe other agents (with noise for non-coalition members)
        observations = []
        for other in self.agents:
            if other.id == agent_id:
                continue

            # More accurate observation of coalition members
            noise_level = 0.1 if other.coalition == agent.coalition else 0.3

            obs = {
                'id': other.id,
                'estimated_resources': other.resources + np.random.randn() * 20 * noise_level,
                'estimated_energy': other.energy + np.random.randn() * 20 * noise_level,
                'reputation': agent.reputation.get(other.id, 0.5),
                'in_same_coalition': other.coalition == agent.coalition,
            }
            observations.append(obs)

        return {
            'agent_id': agent_id,
            'own_resources': agent.resources,
            'own_energy': agent.energy,
            'own_coalition': agent.coalition,
            'observations': observations,
            'timestep': self.timestep
        }

    def get_statistics(self) -> Dict:
        """Get world statistics"""
        alive_agents = [a for a in self.agents if a.alive]

        return {
            'timestep': self.timestep,
            'num_alive': len(alive_agents),
            'total_resources': sum(a.resources for a in alive_agents),
            'avg_resources': np.mean([a.resources for a in alive_agents]) if alive_agents else 0.0,
            'avg_energy': np.mean([a.energy for a in alive_agents]) if alive_agents else 0.0,
            'num_coalitions': len(self.coalitions),
            'zombies_remaining': sum(1 for a in alive_agents if a.is_zombie)
        }
