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
        """Convert observation dict to tensor input"""
        # Simplified conversion - in full version would be more sophisticated
        features = [
            obs['own_resources'] / 200.0,
            obs['own_energy'] / 100.0,
            float(obs['own_coalition'] is not None)
        ]

        # Add features from other agents
        for other_obs in obs['observations'][:5]:  # Limit to 5 neighbors
            features.extend([
                other_obs['estimated_resources'] / 200.0,
                other_obs['estimated_energy'] / 100.0,
                other_obs['reputation'],
                float(other_obs['in_same_coalition'])
            ])

        # Pad to fixed size
        while len(features) < 191:
            features.append(0.0)
        features = features[:191]

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

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

        # Cooperation success
        for game in result.get('games', []):
            if agent_id in game.get('players', []):
                if game['actions'][0] == 'cooperate' and game['actions'][1] == 'cooperate':
                    metrics['cooperation_success'] += 1.0
                elif game['actions'][0] == 'cooperate':
                    metrics['cooperation_success'] += 0.5

        # Zombie detection
        for detection in result.get('zombie_detections', []):
            if detection['detector'] == agent_id:
                if detection['result'] == 'correct_detection':
                    metrics['zombie_detection'] += 1.0
                elif detection['result'] == 'false_positive':
                    metrics['zombie_detection'] -= 0.5

        # Communication quality (from games)
        for game in result.get('games', []):
            if game.get('game_type') == 'communication' and game.get('sender') == agent_id:
                metrics['communication_quality'] += game.get('message_quality', 0.0)

        # Resource efficiency
        agent_states = result.get('agent_states', [])
        if agent_states and agent_id < len(agent_states):
            state = agent_states[agent_id]
            if state['resources'] > 100:
                metrics['resource_efficiency'] += 0.1
            if state['energy'] > 50:
                metrics['resource_efficiency'] += 0.05

        # Behavioral consistency
        if len(action_history) >= 2:
            if action_history[-1].get('type') == action_history[-2].get('type'):
                metrics['behavioral_consistency'] += 0.1


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
