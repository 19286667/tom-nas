"""
Social World 4: Complete society simulator with zombie detection

This module provides a multi-agent social simulation with:
- Cooperation, communication, and resource sharing games
- Zombie detection (agents with impaired ToM)
- Coalition formation
- Observation tracking for information asymmetry (NEW)

The observation tracking enables proper Theory of Mind testing:
agents can only update beliefs based on events they observed.
"""
import torch
import numpy as np
import random
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class WorldEvent:
    """
    Event in the social world with observation tracking.

    This enables information asymmetry - different agents observe
    different events, leading to divergent beliefs about world state.
    """
    timestamp: int
    event_type: str  # 'move_object', 'agent_enter', 'agent_leave', 'resource_transfer', etc.
    actor_id: int
    details: Dict[str, Any] = field(default_factory=dict)
    observed_by: Set[int] = field(default_factory=set)

    def __post_init__(self):
        """Actor always observes their own action."""
        self.observed_by = set(self.observed_by)
        self.observed_by.add(self.actor_id)


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
    location: Optional[str] = None  # Current location for observation tracking
    beliefs: Dict[str, Any] = field(default_factory=dict)  # Agent's beliefs about world

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
    """Complete society simulator with 4 game types"""

    def __init__(self, num_agents: int, ontology_dim: int, num_zombies: int = 2):
        self.num_agents = num_agents
        self.ontology_dim = ontology_dim
        self.num_zombies = num_zombies  # Store for reset
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
        # Increment timestep first so result reflects completed step number
        self.timestep += 1
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

    def reset(self):
        """Reset the world to initial state with new agents and zombies"""
        self.timestep = 0
        self.history = []
        self.coalitions = {}
        self.next_coalition_id = 0

        # Reinitialize agents
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(id=i)
            agent.ontology_state = torch.randn(self.ontology_dim)
            agent.reputation = {j: 0.5 for j in range(self.num_agents)}
            self.agents.append(agent)

        # Create new zombies using stored count
        zombie_indices = random.sample(range(self.num_agents), min(self.num_zombies, self.num_agents))
        for idx in zombie_indices:
            self.agents[idx] = self.zombie_game.create_zombie(idx)
            self.agents[idx].ontology_state = torch.randn(self.ontology_dim)
            self.agents[idx].reputation = {j: 0.5 for j in range(self.num_agents)}

    # =========================================================================
    # Observation Tracking Methods (NEW)
    # These enable proper Theory of Mind testing with information asymmetry
    # =========================================================================

    def get_present_agents(self, location: str = 'main') -> Set[int]:
        """
        Get set of agent IDs currently at a location.

        Used to determine who observes events at that location.
        """
        return {a.id for a in self.agents if a.alive and
                (a.location == location or a.location is None)}

    def create_event(self, event_type: str, actor_id: int,
                     details: Dict[str, Any] = None,
                     location: str = 'main') -> WorldEvent:
        """
        Create an event with proper observation tracking.

        Only agents present at the location observe the event.
        """
        present = self.get_present_agents(location)
        event = WorldEvent(
            timestamp=self.timestep,
            event_type=event_type,
            actor_id=actor_id,
            details=details or {},
            observed_by=present
        )

        # Store in history for later querying
        if not hasattr(self, 'event_log'):
            self.event_log: List[WorldEvent] = []
        self.event_log.append(event)

        return event

    def move_object(self, object_name: str, new_location: str,
                    actor_id: int, observed_by: Set[int] = None) -> WorldEvent:
        """
        Move object with observation tracking.

        Creates a world event that only specified agents observe.
        This is key for creating false beliefs - agents who don't
        observe the move still believe the object is at the old location.
        """
        if not hasattr(self, 'objects'):
            self.objects: Dict[str, str] = {}

        old_location = self.objects.get(object_name, 'unknown')
        self.objects[object_name] = new_location

        # Create event with observation tracking
        if observed_by is None:
            observed_by = self.get_present_agents()

        event = WorldEvent(
            timestamp=self.timestep,
            event_type='move_object',
            actor_id=actor_id,
            details={
                'object': object_name,
                'from': old_location,
                'to': new_location
            },
            observed_by=observed_by
        )

        if not hasattr(self, 'event_log'):
            self.event_log = []
        self.event_log.append(event)

        # Update agent beliefs based on who observed
        for agent in self.agents:
            if agent.id in observed_by:
                # Agent saw the move - update their belief
                if 'object_locations' not in agent.beliefs:
                    agent.beliefs['object_locations'] = {}
                agent.beliefs['object_locations'][object_name] = new_location
            # Else: agent's belief unchanged (potential false belief!)

        return event

    def agent_enters(self, agent_id: int, location: str = 'main') -> WorldEvent:
        """
        Agent enters a location.

        Other present agents observe this entry.
        """
        if 0 <= agent_id < len(self.agents):
            self.agents[agent_id].location = location

        present = self.get_present_agents(location)
        event = WorldEvent(
            timestamp=self.timestep,
            event_type='agent_enter',
            actor_id=agent_id,
            details={'location': location},
            observed_by=present | {agent_id}
        )

        if not hasattr(self, 'event_log'):
            self.event_log = []
        self.event_log.append(event)

        return event

    def agent_leaves(self, agent_id: int, location: str = 'main') -> WorldEvent:
        """
        Agent leaves a location.

        Present agents observe the departure.
        """
        present = self.get_present_agents(location)

        if 0 <= agent_id < len(self.agents):
            self.agents[agent_id].location = None

        event = WorldEvent(
            timestamp=self.timestep,
            event_type='agent_leave',
            actor_id=agent_id,
            details={'location': location},
            observed_by=present  # Actor is leaving, so they also observe
        )

        if not hasattr(self, 'event_log'):
            self.event_log = []
        self.event_log.append(event)

        return event

    def get_agent_observations(self, agent_id: int) -> List[WorldEvent]:
        """
        Get all events this agent observed.

        Useful for reconstructing an agent's belief state.
        """
        if not hasattr(self, 'event_log'):
            return []
        return [e for e in self.event_log if agent_id in e.observed_by]

    def get_agent_belief_about_object(self, agent_id: int, object_name: str) -> Optional[str]:
        """
        Get what an agent believes about an object's location.

        Based on events they observed.
        """
        if 0 <= agent_id < len(self.agents):
            agent = self.agents[agent_id]
            return agent.beliefs.get('object_locations', {}).get(object_name)
        return None

    def compute_belief_accuracy(self, agent_id: int) -> float:
        """
        Compute how accurate an agent's beliefs are vs reality.

        Returns value between 0 and 1.
        """
        if not hasattr(self, 'objects') or not self.objects:
            return 1.0

        correct = 0
        total = 0

        if 0 <= agent_id < len(self.agents):
            agent = self.agents[agent_id]
            obj_beliefs = agent.beliefs.get('object_locations', {})

            for obj, actual_loc in self.objects.items():
                believed_loc = obj_beliefs.get(obj)
                if believed_loc is not None:
                    total += 1
                    if believed_loc == actual_loc:
                        correct += 1

        return correct / total if total > 0 else 1.0

    def create_false_belief_scenario(self, observer_id: int, actor_id: int,
                                     object_name: str = 'ball',
                                     loc1: str = 'basket',
                                     loc2: str = 'box') -> List[WorldEvent]:
        """
        Create a classic Sally-Anne style false belief scenario.

        1. Both agents enter
        2. Actor puts object in loc1 (both observe)
        3. Observer leaves (actor remains)
        4. Actor moves object to loc2 (only actor observes)
        5. Observer returns

        After this, observer believes object is in loc1,
        but it's actually in loc2.
        """
        events = []

        # Initialize objects dict if needed
        if not hasattr(self, 'objects'):
            self.objects = {}

        # 1. Both agents enter
        events.append(self.agent_enters(observer_id))
        events.append(self.agent_enters(actor_id))

        # 2. Actor puts object in loc1 (both observe)
        self.objects[object_name] = loc1
        e = WorldEvent(
            timestamp=self.timestep,
            event_type='put_object',
            actor_id=actor_id,
            details={'object': object_name, 'location': loc1},
            observed_by={observer_id, actor_id}
        )
        if not hasattr(self, 'event_log'):
            self.event_log = []
        self.event_log.append(e)
        events.append(e)

        # Update both agents' beliefs
        for aid in [observer_id, actor_id]:
            if 0 <= aid < len(self.agents):
                if 'object_locations' not in self.agents[aid].beliefs:
                    self.agents[aid].beliefs['object_locations'] = {}
                self.agents[aid].beliefs['object_locations'][object_name] = loc1

        # 3. Observer leaves
        self.timestep += 1
        events.append(self.agent_leaves(observer_id))

        # 4. Actor moves object to loc2 (only actor observes!)
        self.timestep += 1
        move_event = self.move_object(
            object_name, loc2, actor_id,
            observed_by={actor_id}  # Observer is NOT present!
        )
        events.append(move_event)

        # 5. Observer returns
        self.timestep += 1
        events.append(self.agent_enters(observer_id))

        return events

    def validate_false_belief_test(self, observer_id: int, object_name: str,
                                   expected_belief: str) -> Dict[str, Any]:
        """
        Validate a false belief test scenario.

        Returns:
            - belief_correct: Does observer have the expected false belief?
            - reality: What is the actual location?
            - belief: What does observer believe?
        """
        if not hasattr(self, 'objects'):
            return {'error': 'No objects tracked'}

        reality = self.objects.get(object_name)
        belief = self.get_agent_belief_about_object(observer_id, object_name)

        return {
            'reality': reality,
            'belief': belief,
            'expected_belief': expected_belief,
            'belief_correct': belief == expected_belief,
            'has_false_belief': belief != reality,
            'valid_test': belief == expected_belief and reality != expected_belief
        }
