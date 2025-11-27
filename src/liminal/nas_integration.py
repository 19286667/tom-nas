"""
NAS Integration - Bridge to ToM-NAS Coevolution System

This module provides the integration layer between the Liminal game environment
and the existing ToM-NAS evolutionary system. It enables:

1. Using evolved NAS agents to control NPC behavior
2. Evaluating NAS agents in the Liminal environment
3. Generating training data from Liminal gameplay
4. Fitness evaluation based on ToM performance in-game
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .game_environment import LiminalEnvironment, ActionType, Observation
from .soul_map import SoulMap
from .npcs.base_npc import BaseNPC


@dataclass
class LiminalFitnessConfig:
    """Configuration for Liminal-based fitness evaluation."""

    # Episode settings
    episodes_per_eval: int = 5
    max_episode_length: int = 500

    # Fitness component weights
    prediction_accuracy_weight: float = 0.35
    intervention_success_weight: float = 0.20
    social_navigation_weight: float = 0.15
    stability_maintenance_weight: float = 0.15
    exploration_weight: float = 0.10
    survival_weight: float = 0.05

    # Reward scaling
    reward_scale: float = 1.0


class LiminalAgentAdapter:
    """
    Adapter to use ToM-NAS agents in the Liminal environment.

    Takes a NAS-evolved model and wraps it for use as either:
    1. The player agent (controlling actions)
    2. An NPC brain (controlling NPC behavior)
    """

    def __init__(
        self,
        model: nn.Module,
        input_dim: int = 191,
        output_dim: int = 181,
        device: str = "cpu",
    ):
        """
        Initialize the adapter.

        Args:
            model: A ToM-NAS evolved model (TRN, RSAN, Transformer, or Hybrid)
            input_dim: Expected input dimension
            output_dim: Expected output dimension
            device: Device to run model on
        """
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.model.to(device)
        self.model.eval()

        # Action mapping
        self.action_map = {
            0: ActionType.WAIT,
            1: ActionType.MOVE,
            2: ActionType.ANALYZE,
            3: ActionType.INTERVENE,
            4: ActionType.PREDICT,
            5: ActionType.INTERACT,
        }

    def observation_to_model_input(self, observation: Observation,
                                   seq_len: int = 20) -> torch.Tensor:
        """
        Convert Liminal observation to model input format.

        The model expects [batch, seq_len, input_dim].
        We pad/transform the observation to fit.
        """
        # Get the full observation tensor
        obs_tensor = observation.full_tensor

        # Pad or truncate to match input_dim
        if obs_tensor.shape[0] < self.input_dim:
            padding = torch.zeros(self.input_dim - obs_tensor.shape[0])
            obs_tensor = torch.cat([obs_tensor, padding])
        elif obs_tensor.shape[0] > self.input_dim:
            obs_tensor = obs_tensor[:self.input_dim]

        # Create sequence (repeat observation for now - could use history)
        sequence = obs_tensor.unsqueeze(0).repeat(seq_len, 1)

        # Add batch dimension
        return sequence.unsqueeze(0).to(self.device)

    def model_output_to_action(self, output: Dict[str, torch.Tensor],
                               game_state: Any) -> Dict[str, Any]:
        """
        Convert model output to Liminal action.

        Args:
            output: Model output dict with 'beliefs' and 'actions'
            game_state: Current game state for context

        Returns:
            Action dict for environment step
        """
        # Get beliefs and actions
        beliefs = output.get('beliefs', torch.zeros(1, self.output_dim))
        actions = output.get('actions', torch.zeros(1))

        # Interpret action value
        action_value = actions[0].item()

        # Map to action type
        action_idx = int(action_value * len(self.action_map)) % len(self.action_map)
        action_type = self.action_map[action_idx]

        # Build action dict based on type
        action = {"type": action_type}

        # Add action-specific parameters
        if action_type == ActionType.ANALYZE:
            # Choose target from nearby NPCs
            if game_state.nearby_npcs:
                action["target_id"] = game_state.nearby_npcs[0]
                action["depth"] = "moderate"

        elif action_type == ActionType.INTERVENE:
            if game_state.nearby_npcs:
                action["target_id"] = game_state.nearby_npcs[0]
                # Choose hazard based on beliefs
                action["hazard"] = self._select_hazard(beliefs)
                action["intensity"] = min(1.0, abs(beliefs[0, 0].item()) + 0.5)

        elif action_type == ActionType.PREDICT:
            if game_state.nearby_npcs:
                action["target_id"] = game_state.nearby_npcs[0]
                action["prediction"] = self._generate_prediction(beliefs)

        elif action_type == ActionType.MOVE:
            # Generate movement from beliefs
            action["target"] = self._generate_movement(beliefs, game_state)

        return action

    def _select_hazard(self, beliefs: torch.Tensor) -> str:
        """Select appropriate hazard based on belief state."""
        # Simple heuristic based on belief values
        hazards = [
            "doubt", "fear", "validation", "reassurance",
            "curiosity", "clarity", "empathy", "paradox"
        ]

        # Use belief tensor to weight selection
        idx = int(abs(beliefs[0, :8].sum().item() * 10)) % len(hazards)
        return hazards[idx]

    def _generate_prediction(self, beliefs: torch.Tensor) -> Dict[str, Any]:
        """Generate prediction from beliefs."""
        action_types = ["flee", "avoid", "observe", "approach", "interact", "continue"]
        idx = int(abs(beliefs[0, 0].item() * 10)) % len(action_types)
        return {"action_type": action_types[idx]}

    def _generate_movement(self, beliefs: torch.Tensor,
                          game_state: Any) -> Tuple[float, float]:
        """Generate movement target from beliefs."""
        current = game_state.player_position

        # Use beliefs to determine direction
        dx = beliefs[0, 0].item() * 10
        dy = beliefs[0, 1].item() * 10

        return (current[0] + dx, current[1] + dy)

    def select_action(self, observation: Observation,
                      game_state: Any) -> Dict[str, Any]:
        """
        Main method to select an action given an observation.

        Args:
            observation: Current environment observation
            game_state: Current game state

        Returns:
            Action dict for environment step
        """
        # Convert observation to model input
        model_input = self.observation_to_model_input(observation)

        # Run model
        with torch.no_grad():
            output = self.model(model_input)

        # Convert to action
        return self.model_output_to_action(output, game_state)


class LiminalFitnessEvaluator:
    """
    Evaluates ToM-NAS agents in the Liminal environment.

    This provides fitness scores for the evolutionary algorithm
    based on performance in the game environment.
    """

    def __init__(self, config: Optional[LiminalFitnessConfig] = None):
        """
        Initialize the evaluator.

        Args:
            config: Fitness evaluation configuration
        """
        self.config = config or LiminalFitnessConfig()

    def evaluate_agent(self, model: nn.Module,
                       seed: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate a ToM-NAS agent in the Liminal environment.

        Args:
            model: The neural network model to evaluate
            seed: Random seed for reproducibility

        Returns:
            Dict with fitness scores for different aspects
        """
        # Create environment
        env = LiminalEnvironment(
            population_size=100,
            include_heroes=True,
            max_episode_length=self.config.max_episode_length,
            seed=seed,
        )

        # Create adapter
        adapter = LiminalAgentAdapter(model)

        # Run episodes
        all_scores = []

        for ep in range(self.config.episodes_per_eval):
            episode_seed = seed + ep if seed else None
            scores = self._run_episode(env, adapter, episode_seed)
            all_scores.append(scores)

        # Average scores across episodes
        avg_scores = {}
        for key in all_scores[0].keys():
            avg_scores[key] = sum(s[key] for s in all_scores) / len(all_scores)

        # Calculate total fitness
        avg_scores["total_fitness"] = (
            avg_scores["prediction_accuracy"] * self.config.prediction_accuracy_weight +
            avg_scores["intervention_success"] * self.config.intervention_success_weight +
            avg_scores["social_navigation"] * self.config.social_navigation_weight +
            avg_scores["stability_maintenance"] * self.config.stability_maintenance_weight +
            avg_scores["exploration"] * self.config.exploration_weight +
            avg_scores["survival"] * self.config.survival_weight
        ) * self.config.reward_scale

        return avg_scores

    def _run_episode(self, env: LiminalEnvironment,
                     adapter: LiminalAgentAdapter,
                     seed: Optional[int]) -> Dict[str, float]:
        """Run a single episode and return scores."""
        observation = env.reset(seed)
        game_state = env._get_game_state()

        total_reward = 0.0
        predictions_correct = 0
        predictions_made = 0
        interventions_success = 0
        interventions_made = 0
        realms_visited = {env.current_realm}
        steps_stable = 0
        total_steps = 0

        done = False
        while not done:
            # Select action
            action = adapter.select_action(observation, game_state)

            # Take step
            result = env.step(action)

            observation = result.observation
            game_state = result.game_state
            total_reward += result.reward
            done = result.done
            total_steps += 1

            # Track metrics
            action_info = result.info.get("action_info", {})

            # Prediction tracking
            if action_info.get("accuracy") is not None:
                predictions_made += 1
                if action_info["accuracy"] > 0.5:
                    predictions_correct += 1

            # Intervention tracking
            if "stability_change" in action_info:
                interventions_made += 1
                if abs(action_info["stability_change"]) > 0.05:
                    interventions_success += 1

            # Realm tracking
            realms_visited.add(game_state.current_realm)

            # Stability tracking
            if game_state.instability_level.value <= 1:
                steps_stable += 1

        # Calculate scores
        return {
            "prediction_accuracy": (predictions_correct / predictions_made
                                   if predictions_made > 0 else 0.0),
            "intervention_success": (interventions_success / interventions_made
                                    if interventions_made > 0 else 0.0),
            "social_navigation": total_reward / max(1, total_steps) + 0.5,
            "stability_maintenance": steps_stable / max(1, total_steps),
            "exploration": len(realms_visited) / len(list(env.npcs.values())[0].current_realm.__class__),
            "survival": 1.0 if not result.info.get("collapse", False) else 0.0,
            "total_reward": total_reward,
            "steps": total_steps,
        }


class LiminalDataGenerator:
    """
    Generates training data from Liminal gameplay.

    Creates datasets of (observation, action, outcome) tuples
    that can be used for supervised learning of ToM behaviors.
    """

    def __init__(self, env: LiminalEnvironment):
        """
        Initialize the data generator.

        Args:
            env: The Liminal environment to generate data from
        """
        self.env = env
        self.collected_data: List[Dict[str, Any]] = []

    def collect_episode(self, policy: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Collect data from one episode.

        Args:
            policy: Optional policy function (obs -> action). If None, uses random.

        Returns:
            List of (observation, action, outcome) tuples
        """
        episode_data = []
        observation = self.env.reset()
        game_state = self.env._get_game_state()
        done = False

        while not done:
            # Select action
            if policy:
                action = policy(observation, game_state)
            else:
                action = self._random_action(game_state)

            # Record pre-step state
            pre_state = {
                "observation": observation.full_tensor.clone(),
                "game_state": game_state,
                "action": action,
            }

            # Take step
            result = self.env.step(action)

            # Record outcome
            pre_state["reward"] = result.reward
            pre_state["next_observation"] = result.observation.full_tensor.clone()
            pre_state["done"] = result.done
            pre_state["info"] = result.info

            episode_data.append(pre_state)
            self.collected_data.append(pre_state)

            observation = result.observation
            game_state = result.game_state
            done = result.done

        return episode_data

    def _random_action(self, game_state: Any) -> Dict[str, Any]:
        """Generate a random valid action."""
        import random

        action_types = list(ActionType)
        action_type = random.choice(action_types)

        action = {"type": action_type}

        if action_type in [ActionType.ANALYZE, ActionType.INTERVENE,
                          ActionType.PREDICT, ActionType.INTERACT]:
            if game_state.nearby_npcs:
                action["target_id"] = random.choice(game_state.nearby_npcs)

        if action_type == ActionType.INTERVENE:
            hazards = ["doubt", "fear", "validation", "reassurance", "curiosity"]
            action["hazard"] = random.choice(hazards)

        return action

    def get_training_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get a batch of training data.

        Args:
            batch_size: Number of samples

        Returns:
            Dict with 'observations', 'actions', 'rewards', 'next_observations'
        """
        if len(self.collected_data) < batch_size:
            return {}

        import random
        samples = random.sample(self.collected_data, batch_size)

        observations = torch.stack([s["observation"] for s in samples])
        next_observations = torch.stack([s["next_observation"] for s in samples])
        rewards = torch.tensor([s["reward"] for s in samples])

        return {
            "observations": observations,
            "next_observations": next_observations,
            "rewards": rewards,
        }


def integrate_with_coevolution(
    liminal_env: LiminalEnvironment,
    coevolution_trainer: Any,  # CoevolutionaryTrainer from train_coevolution.py
    fitness_weight: float = 0.3,
) -> None:
    """
    Integrate Liminal fitness into the coevolution training loop.

    This adds Liminal-based evaluation as a component of the overall
    fitness function used in coevolutionary training.

    Args:
        liminal_env: The Liminal game environment
        coevolution_trainer: The CoevolutionaryTrainer instance
        fitness_weight: How much weight to give Liminal fitness (0-1)
    """
    liminal_evaluator = LiminalFitnessEvaluator()

    # Store original evaluate_agent method
    original_evaluate = coevolution_trainer.evaluate_agent

    def enhanced_evaluate(agent, *args, **kwargs):
        """Enhanced evaluation including Liminal fitness."""
        # Get original fitness
        original_fitness = original_evaluate(agent, *args, **kwargs)

        # Get Liminal fitness
        try:
            liminal_scores = liminal_evaluator.evaluate_agent(agent.model)
            liminal_fitness = liminal_scores["total_fitness"]
        except Exception as e:
            print(f"Liminal evaluation failed: {e}")
            liminal_fitness = 0.0

        # Combine fitnesses
        combined_fitness = (
            original_fitness * (1 - fitness_weight) +
            liminal_fitness * fitness_weight
        )

        # Update agent fitness
        agent.fitness = combined_fitness
        agent.liminal_scores = liminal_scores if liminal_fitness > 0 else {}

        return combined_fitness

    # Replace method
    coevolution_trainer.evaluate_agent = enhanced_evaluate


# Export
__all__ = [
    'LiminalFitnessConfig',
    'LiminalAgentAdapter',
    'LiminalFitnessEvaluator',
    'LiminalDataGenerator',
    'integrate_with_coevolution',
]
