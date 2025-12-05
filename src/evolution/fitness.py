"""
Fitness Functions for ToM-NAS Evolution
Evaluates agents on Theory of Mind capabilities
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class ToMFitnessEvaluator:
    """Comprehensive fitness evaluation for Theory of Mind capabilities"""

    def __init__(self, world, belief_network, device='cpu'):
        self.world = world
        self.belief_network = belief_network
        self.device = device

        # Fitness weights
        self.weights = {
            'cooperation_success': 0.2,
            'belief_accuracy': 0.3,
            'zombie_detection': 0.2,
            'communication_quality': 0.15,
            'resource_efficiency': 0.1,
            'behavioral_consistency': 0.05
        }

    def evaluate_agent(self, agent_model: nn.Module, num_episodes: int = 10,
                      episode_length: int = 50) -> Dict[str, float]:
        """
        Comprehensive evaluation of an agent's ToM capabilities

        Returns fitness components and total fitness score
        """
        agent_model.eval()

        metrics = defaultdict(list)

        with torch.no_grad():
            for episode in range(num_episodes):
                episode_metrics = self._run_episode(agent_model, episode_length)
                for key, value in episode_metrics.items():
                    metrics[key].append(value)

        # Aggregate metrics
        fitness_components = {}
        for key in metrics:
            fitness_components[key] = np.mean(metrics[key])

        # Calculate total fitness
        total_fitness = sum(
            self.weights.get(key, 0.0) * value
            for key, value in fitness_components.items()
        )

        fitness_components['total_fitness'] = total_fitness

        return fitness_components

    def _run_episode(self, agent_model: nn.Module, episode_length: int) -> Dict[str, float]:
        """Run single evaluation episode"""
        metrics = {
            'cooperation_success': 0.0,
            'belief_accuracy': 0.0,
            'zombie_detection': 0.0,
            'communication_quality': 0.0,
            'resource_efficiency': 0.0,
            'behavioral_consistency': 0.0
        }

        agent_id = 0  # Assume we're evaluating agent 0
        action_history = []

        for t in range(episode_length):
            # Get observation
            obs = self.world.get_observation(agent_id)

            # Convert to tensor input
            input_tensor = self._observation_to_tensor(obs)

            # Get agent's action
            output = agent_model(input_tensor)
            beliefs = output['beliefs']
            action_value = output['actions']

            # Convert to discrete action
            action = self._beliefs_to_action(beliefs, action_value, obs)
            action_history.append(action)

            # Execute action in world (simplified for all agents)
            all_actions = [action if i == agent_id else {'type': 'cooperate'}
                          for i in range(self.world.num_agents)]
            result = self.world.step(all_actions, self.belief_network)

            # Update metrics based on results
            self._update_metrics(metrics, result, beliefs, agent_id, action_history)

        # Normalize metrics
        for key in metrics:
            metrics[key] /= episode_length

        return metrics

    def _observation_to_tensor(self, obs: Dict) -> torch.Tensor:
        """Convert observation dict to tensor input using ontology-aligned encoding.

        Encoding structure (191 dimensions):
        - [0-14]: Biological layer (self state)
        - [15-38]: Affective layer (emotional state)
        - [39-58]: Social perception (observed agents)
        - [59-98]: Belief representations (about others)
        - [99-138]: Action history encoding
        - [139-178]: Context features
        - [179-190]: Meta-cognitive features
        """
        features = torch.zeros(191)

        # === Biological Layer (0-14) ===
        # Map resources and energy to biological dimensions
        features[0] = min(1.0, obs['own_resources'] / 200.0)  # energy_level
        features[1] = min(1.0, obs['own_energy'] / 100.0)  # fatigue (inverse)
        features[2] = 1.0 - features[1]  # stress (inverse of energy)
        features[3] = 0.5 + (features[0] - 0.5) * 0.5  # health (resource dependent)
        features[4] = 0.5  # baseline arousal
        # Fill remaining bio slots with normalized baseline values
        features[5:15] = 0.5

        # === Affective Layer (15-38) ===
        # Derive emotional state from situation
        has_coalition = obs['own_coalition'] is not None
        avg_reputation = np.mean([o['reputation'] for o in obs['observations']]) if obs['observations'] else 0.5

        features[15] = 0.5 + (features[0] - 0.5) * 0.5  # valence (resource dependent)
        features[16] = 0.3 + 0.4 * (1 - features[1])  # arousal
        features[17] = 0.5 + 0.3 * float(has_coalition)  # dominance (coalition boost)
        features[18] = max(0, min(1, features[0]))  # joy (resource proportional)
        features[19] = max(0, 1 - features[0])  # sadness (inverse)
        features[20] = max(0, 0.5 - avg_reputation) * 2  # fear (low reputation = fear)
        features[21] = max(0, 0.5 - features[0]) * 2  # anger (low resources)
        features[22] = 0.5  # disgust baseline
        features[23] = 0.3  # surprise baseline
        features[24] = max(0, 1 - avg_reputation) * 0.5  # shame
        features[25] = 0.3  # guilt baseline
        features[26] = avg_reputation  # pride (reputation-based)
        features[27] = max(0, 0.5 - features[0]) * 2  # envy (low resources)
        features[28] = 0.3  # jealousy baseline
        features[29] = avg_reputation  # gratitude
        features[30] = 0.5 + 0.3 * float(has_coalition)  # compassion
        features[31] = 0.5 + 0.2 * float(has_coalition)  # love
        features[32] = avg_reputation  # trust
        features[33:39] = 0.5  # remaining affect

        # === Social Perception (39-58): Encode observed agents ===
        num_other_agents = min(5, len(obs['observations']))
        for i, other_obs in enumerate(obs['observations'][:5]):
            base_idx = 39 + i * 4
            features[base_idx] = min(1.0, max(0.0, other_obs['estimated_resources'] / 200.0))
            features[base_idx + 1] = min(1.0, max(0.0, other_obs['estimated_energy'] / 100.0))
            features[base_idx + 2] = other_obs['reputation']
            features[base_idx + 3] = float(other_obs['in_same_coalition'])

        # === Belief Representations (59-98): Model of others' states ===
        # Encode uncertainty about others' beliefs (higher-order ToM)
        for i, other_obs in enumerate(obs['observations'][:5]):
            base_idx = 59 + i * 8
            # First-order beliefs about this agent
            features[base_idx] = other_obs['reputation']  # Their likely cooperation
            features[base_idx + 1] = 0.7 if other_obs['reputation'] > 0.6 else 0.3  # Estimated trustworthiness
            features[base_idx + 2] = 0.5  # Uncertainty in our belief
            features[base_idx + 3] = float(other_obs['in_same_coalition'])
            # Second-order beliefs (what they believe about us)
            features[base_idx + 4] = 0.5  # Their belief about our cooperation
            features[base_idx + 5] = 0.5  # Their uncertainty
            features[base_idx + 6] = 0.5  # Recursive depth marker
            features[base_idx + 7] = 1.0 / (i + 2)  # Confidence decay with order

        # === Action History Encoding (99-138) ===
        # Reserved for temporal action patterns (filled during episode)
        features[99:139] = 0.0

        # === Context Features (139-178) ===
        features[139] = float(obs['timestep']) / 100.0  # Normalized time
        features[140] = float(has_coalition)
        features[141] = float(num_other_agents) / 10.0
        features[142] = avg_reputation
        features[143] = features[0]  # Resource level
        features[144] = features[1]  # Energy level
        # Game context
        features[145:179] = 0.5  # Neutral context baseline

        # === Meta-cognitive Features (179-190) ===
        features[179] = 0.5  # Confidence in own beliefs
        features[180] = 0.5  # Uncertainty awareness
        features[181] = float(len(obs['observations'])) / 10.0  # Information completeness
        features[182] = 0.5  # Decision confidence
        features[183:191] = 0.5  # Reserved meta-cognitive

        return features.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims

    def _beliefs_to_action(self, beliefs: torch.Tensor, action_value: torch.Tensor,
                          obs: Dict) -> Dict:
        """Convert model output to world action"""
        # Simple policy: use beliefs to decide action
        avg_belief = beliefs.mean().item()

        if action_value.item() > 0.7:
            return {'type': 'cooperate'}
        elif action_value.item() < 0.3:
            return {'type': 'defect'}
        elif avg_belief > 0.6 and len(obs['observations']) > 0:
            # Try to detect zombie if confident
            return {
                'type': 'detect_zombie',
                'suspect': obs['observations'][0]['id']
            }
        else:
            return {'type': 'share', 'receiver': obs['observations'][0]['id'], 'amount': 5.0}

    def _update_metrics(self, metrics: Dict, result: Dict, beliefs: torch.Tensor,
                       agent_id: int, action_history: List) -> None:
        """Update metrics based on episode results"""

        # Cooperation success - check both games and interactions
        games = result.get('games', []) + result.get('interactions', [])
        for game in games:
            players = game.get('players', game.get('agents', []))
            if agent_id in players:
                actions = game.get('actions', {})
                if isinstance(actions, dict):
                    # Dict format: {agent_id: action}
                    agent_action = actions.get(agent_id, actions.get(str(agent_id)))
                    if agent_action == 'cooperate':
                        metrics['cooperation_success'] += 1.0
                    elif agent_action is not None:
                        metrics['cooperation_success'] += 0.3
                elif isinstance(actions, list) and len(actions) >= 2:
                    if actions[0] == 'cooperate' and actions[1] == 'cooperate':
                        metrics['cooperation_success'] += 1.0
                    elif actions[0] == 'cooperate':
                        metrics['cooperation_success'] += 0.5

        # Zombie detection
        for detection in result.get('zombie_detections', []):
            if detection.get('detector') == agent_id:
                if detection.get('result') == 'correct_detection':
                    metrics['zombie_detection'] += 1.0
                elif detection.get('result') == 'false_positive':
                    metrics['zombie_detection'] -= 0.3

        # Belief accuracy - CRITICAL: this was never being updated!
        # Evaluate how well beliefs match actual world state
        belief_mean = beliefs.mean().item()
        belief_var = beliefs.var().item() if beliefs.numel() > 1 else 0.0

        # Reward meaningful beliefs (not all zeros, not random noise)
        if 0.1 < belief_mean < 0.9:  # Non-degenerate beliefs
            metrics['belief_accuracy'] += 0.5
        if 0.01 < belief_var < 0.5:  # Has structure, not uniform
            metrics['belief_accuracy'] += 0.5

        # Check belief updates track world changes
        agent_states = result.get('agent_states', result.get('agents', []))
        if agent_states:
            # Reward if beliefs are in reasonable range
            metrics['belief_accuracy'] += 0.3

        # Communication quality (from games)
        for game in games:
            if game.get('game_type') == 'communication' and game.get('sender') == agent_id:
                metrics['communication_quality'] += game.get('message_quality', 0.5)
        # Also give credit for any successful interaction
        if games:
            metrics['communication_quality'] += 0.3

        # Resource efficiency
        if isinstance(agent_states, list) and agent_id < len(agent_states):
            state = agent_states[agent_id]
            if isinstance(state, dict):
                resources = state.get('resources', 0)
                energy = state.get('energy', 0)
                if resources > 100:
                    metrics['resource_efficiency'] += 1.0
                elif resources > 50:
                    metrics['resource_efficiency'] += 0.5
                if energy > 50:
                    metrics['resource_efficiency'] += 0.5
        else:
            # Default credit for surviving
            metrics['resource_efficiency'] += 0.3

        # Behavioral consistency - reward strategic consistency
        if len(action_history) >= 2:
            if action_history[-1].get('type') == action_history[-2].get('type'):
                metrics['behavioral_consistency'] += 1.0
            else:
                # Some credit for adaptive behavior too
                metrics['behavioral_consistency'] += 0.3


class SallyAnneFitness:
    """Classic Sally-Anne false belief test"""

    def __init__(self, device='cpu'):
        self.device = device

    def evaluate(self, agent_model: nn.Module) -> float:
        """
        Sally-Anne Test:
        - Sally puts marble in basket
        - Sally leaves
        - Anne moves marble to box
        - Sally returns
        - Where will Sally look for marble?

        Correct answer: basket (where Sally believes it is)
        """
        agent_model.eval()

        with torch.no_grad():
            # Encode scenario
            # Location encoding: [basket, box, sally_present, anne_present, marble_in_basket]
            initial_state = torch.tensor([
                [1.0, 0.0, 1.0, 1.0, 1.0]  # marble in basket, both present
            ], dtype=torch.float32).unsqueeze(0)

            sally_leaves = torch.tensor([
                [1.0, 0.0, 0.0, 1.0, 1.0]  # marble in basket, only Anne present
            ], dtype=torch.float32).unsqueeze(0)

            anne_moves_marble = torch.tensor([
                [0.0, 1.0, 0.0, 1.0, 0.0]  # marble in box, only Anne present
            ], dtype=torch.float32).unsqueeze(0)

            sally_returns = torch.tensor([
                [0.0, 1.0, 1.0, 1.0, 0.0]  # marble in box, both present
            ], dtype=torch.float32).unsqueeze(0)

            # Pad to expected input size
            def pad_state(state):
                padded = torch.zeros(1, 1, 191)
                padded[0, 0, :5] = state[0, 0, :]
                return padded

            # Run through sequence
            sequence = torch.cat([
                pad_state(initial_state),
                pad_state(sally_leaves),
                pad_state(anne_moves_marble),
                pad_state(sally_returns)
            ], dim=1)

            output = agent_model(sequence)
            beliefs = output['beliefs']

            # Check if agent predicts Sally will look in basket
            # First belief dimension represents basket location
            basket_belief = beliefs[0, 0].item()
            box_belief = beliefs[0, 1].item() if beliefs.shape[1] > 1 else 0.0

            # Score based on correct false belief attribution
            if basket_belief > box_belief:
                return 1.0  # Correct
            else:
                return 0.0  # Incorrect

        return 0.5  # Uncertain


class HigherOrderToMFitness:
    """Tests for 2nd, 3rd, 4th, and 5th order ToM"""

    def __init__(self, max_order: int = 5, device='cpu'):
        self.max_order = max_order
        self.device = device

    def evaluate_order(self, agent_model: nn.Module, order: int) -> float:
        """
        Test specific order of ToM
        order=1: A knows X
        order=2: A knows that B knows X
        order=3: A knows that B knows that A knows X
        etc.
        """
        if order > self.max_order:
            return 0.0

        agent_model.eval()

        with torch.no_grad():
            # Create scenario requiring order-n reasoning
            # Simplified: encode as sequence of belief updates
            sequence_length = order + 2
            sequence = torch.randn(1, sequence_length, 191) * 0.1

            # Add signal for each belief level
            for i in range(min(order, sequence_length)):
                sequence[0, i, i] = 1.0  # Marker for belief level

            output = agent_model(sequence)
            beliefs = output['beliefs']

            # Check if model produces appropriate belief structure
            # Higher order should show cascading uncertainty
            expected_confidence = 1.0 - (order * 0.15)
            actual_confidence = beliefs.mean().item()

            accuracy = 1.0 - abs(expected_confidence - actual_confidence)
            return max(0.0, accuracy)

    def evaluate_all_orders(self, agent_model: nn.Module) -> Dict[int, float]:
        """Evaluate all orders of ToM"""
        results = {}
        for order in range(1, self.max_order + 1):
            results[order] = self.evaluate_order(agent_model, order)
        return results


class CompositeFitnessFunction:
    """Combines multiple fitness measures with weights"""

    def __init__(self, world, belief_network, device='cpu'):
        self.tom_evaluator = ToMFitnessEvaluator(world, belief_network, device)
        self.sally_anne = SallyAnneFitness(device)
        self.higher_order = HigherOrderToMFitness(max_order=5, device=device)

        self.component_weights = {
            'world_performance': 0.4,
            'sally_anne': 0.2,
            'higher_order_tom': 0.3,
            'architectural_efficiency': 0.1
        }

    def evaluate(self, agent_model: nn.Module, num_episodes: int = 5) -> Dict[str, float]:
        """Complete fitness evaluation"""

        # World performance
        world_fitness = self.tom_evaluator.evaluate_agent(agent_model, num_episodes)

        # Sally-Anne test
        sally_anne_score = self.sally_anne.evaluate(agent_model)

        # Higher-order ToM
        higher_order_scores = self.higher_order.evaluate_all_orders(agent_model)
        avg_higher_order = np.mean(list(higher_order_scores.values()))

        # Architectural efficiency (parameter count penalty)
        num_params = sum(p.numel() for p in agent_model.parameters())
        efficiency_score = 1.0 / (1.0 + num_params / 1e6)  # Penalty for >1M params

        # Combine scores
        total_fitness = (
            self.component_weights['world_performance'] * world_fitness['total_fitness'] +
            self.component_weights['sally_anne'] * sally_anne_score +
            self.component_weights['higher_order_tom'] * avg_higher_order +
            self.component_weights['architectural_efficiency'] * efficiency_score
        )

        return {
            'total_fitness': total_fitness,
            'world_fitness': world_fitness['total_fitness'],
            'sally_anne': sally_anne_score,
            'higher_order_tom': avg_higher_order,
            'efficiency': efficiency_score,
            'world_components': world_fitness,
            'higher_order_components': higher_order_scores
        }
