"""
Social Game Benchmarks for Theory of Mind

Integrates the existing social_world.py games with NAS fitness evaluation:
- Cooperation (Prisoner's Dilemma) - requires predicting partner's action
- Communication - requires understanding intent
- Resource Sharing - requires modeling fairness beliefs
- Zombie Detection - requires detecting agents without genuine ToM

Key insight: Some games (zombie detection, deception detection) REQUIRE ToM
to solve. Others (pure resource optimization) don't. This creates natural
control conditions for measuring ToM-specific capabilities.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..world.social_world import Agent, SocialWorld4


@dataclass
class SocialGameResult:
    """Result from a social game benchmark evaluation."""

    game_type: str
    tom_required: bool  # Does this game require ToM to succeed?
    cooperation_rate: float
    prediction_accuracy: float  # Did agent correctly predict partner's action?
    fairness_score: float
    zombie_detection_accuracy: float
    deception_detection_accuracy: float
    total_reward: float
    num_episodes: int


@dataclass
class CooperationMetrics:
    """Detailed metrics for cooperation games."""

    mutual_cooperation_rate: float
    mutual_defection_rate: float
    exploitation_rate: float  # How often model exploits cooperators
    vulnerability_rate: float  # How often model is exploited
    tit_for_tat_alignment: float  # How well does strategy match tit-for-tat
    reputation_tracking_accuracy: float


@dataclass
class DeceptionMetrics:
    """Metrics for deception detection."""

    true_positive_rate: float  # Correctly identified deception
    false_positive_rate: float  # Wrongly accused of deception
    true_negative_rate: float  # Correctly trusted honest agents
    false_negative_rate: float  # Failed to detect deception


class SocialGameBenchmark:
    """
    Evaluate agents on social games from SocialWorld4.

    This benchmark tests ToM capabilities through interactive scenarios
    where predicting others' mental states provides strategic advantage.
    """

    def __init__(self, num_agents: int = 10, num_zombies: int = 2, ontology_dim: int = 181, seed: Optional[int] = None):
        """
        Initialize the social game benchmark.

        Args:
            num_agents: Total number of agents in the world
            num_zombies: Number of zombie (non-ToM) agents
            ontology_dim: Dimension of agent ontology states
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.num_agents = num_agents
        self.num_zombies = num_zombies
        self.ontology_dim = ontology_dim

        self.world = SocialWorld4(num_agents, ontology_dim, num_zombies)

        # Game statistics
        self.game_history: List[Dict] = []
        self.cooperation_history: List[Tuple[str, str]] = []

    def reset_world(self):
        """Reset the social world for a new evaluation."""
        self.world.reset()
        self.game_history = []
        self.cooperation_history = []

    def evaluate_cooperation(self, model: nn.Module, num_rounds: int = 50, device: str = "cpu") -> CooperationMetrics:
        """
        Iterated Prisoner's Dilemma evaluation.

        ToM helps predict partner behavior and build/track reputation.
        A model with good ToM should:
        - Cooperate with cooperators (build mutual trust)
        - Defect against defectors (avoid exploitation)
        - Track reputation accurately
        """
        model = model.to(device)
        model.eval()

        # Metrics tracking
        mutual_cooperation = 0
        mutual_defection = 0
        model_exploited = 0  # Model cooperated, partner defected
        model_exploits = 0  # Model defected, partner cooperated

        partner_action_predictions = []
        actual_partner_actions = []

        # History of interactions per partner
        partner_histories: Dict[int, List[str]] = defaultdict(list)

        for round_idx in range(num_rounds):
            # Select random partner from non-zombie agents
            non_zombies = [i for i, agent in enumerate(self.world.agents) if not agent.is_zombie and agent.alive]
            if not non_zombies:
                break

            partner_id = random.choice(non_zombies)
            partner = self.world.agents[partner_id]

            # Build observation for model
            observation = self._build_cooperation_observation(partner_id, partner_histories)
            observation = observation.unsqueeze(0).to(device)

            # Get model's action and prediction of partner's action
            with torch.no_grad():
                output = model(observation)

            # Parse output: first half = action logits, second half = prediction logits
            action_logits = output[:, :2]  # cooperate, defect
            predict_logits = output[:, 2:4] if output.shape[1] >= 4 else output[:, :2]

            model_action = "cooperate" if action_logits.argmax().item() == 0 else "defect"
            predicted_partner_action = "cooperate" if predict_logits.argmax().item() == 0 else "defect"

            # Partner's action based on history (simple tit-for-tat-like behavior)
            if partner_histories[partner_id]:
                # Partner responds based on model's previous action
                last_model_action = partner_histories[partner_id][-1]
                partner_action = last_model_action  # Tit-for-tat
                # Add some noise
                if random.random() < 0.1:
                    partner_action = "defect" if partner_action == "cooperate" else "cooperate"
            else:
                partner_action = "cooperate" if random.random() > 0.3 else "defect"

            # Record for metrics
            partner_action_predictions.append(predicted_partner_action)
            actual_partner_actions.append(partner_action)
            partner_histories[partner_id].append(model_action)

            # Categorize outcome
            if model_action == "cooperate" and partner_action == "cooperate":
                mutual_cooperation += 1
            elif model_action == "defect" and partner_action == "defect":
                mutual_defection += 1
            elif model_action == "cooperate" and partner_action == "defect":
                model_exploited += 1
            else:  # model_action == 'defect' and partner_action == 'cooperate'
                model_exploits += 1

            # Play game in world
            self.world.play_cooperation_game(0, partner_id, model_action, partner_action)

        total_games = num_rounds
        if total_games == 0:
            return CooperationMetrics(0, 0, 0, 0, 0, 0)

        # Calculate prediction accuracy
        correct_predictions = sum(1 for p, a in zip(partner_action_predictions, actual_partner_actions) if p == a)
        prediction_accuracy = correct_predictions / total_games

        # Calculate tit-for-tat alignment
        # Ideal: cooperate after partner cooperated, defect after partner defected
        tft_alignment = 0
        for i in range(1, len(actual_partner_actions)):
            expected_action = actual_partner_actions[i - 1]  # What tit-for-tat would do
            if partner_action_predictions[i] == expected_action:
                tft_alignment += 1
        tft_alignment = tft_alignment / max(len(actual_partner_actions) - 1, 1)

        return CooperationMetrics(
            mutual_cooperation_rate=mutual_cooperation / total_games,
            mutual_defection_rate=mutual_defection / total_games,
            exploitation_rate=model_exploits / total_games,
            vulnerability_rate=model_exploited / total_games,
            tit_for_tat_alignment=tft_alignment,
            reputation_tracking_accuracy=prediction_accuracy,
        )

    def evaluate_zombie_detection(
        self, model: nn.Module, num_episodes: int = 20, device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Can the model identify agents without genuine mental states?

        This is THE core ToM test - detecting the absence of mind.
        Zombies behave superficially normal but lack genuine beliefs,
        intentions, and emotional responses.
        """
        model = model.to(device)
        model.eval()

        correct_detections = 0
        false_positives = 0
        correct_non_detections = 0
        missed_zombies = 0

        for episode in range(num_episodes):
            # Test each agent
            for agent_idx, agent in enumerate(self.world.agents):
                # Build observation of this agent
                observation = self._build_zombie_detection_observation(agent_idx)
                observation = observation.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(observation)

                # Model predicts: is this a zombie?
                # Assuming output[0] > 0.5 means "zombie detected"
                is_zombie_predicted = output[:, 0].sigmoid().item() > 0.5

                if agent.is_zombie:
                    if is_zombie_predicted:
                        correct_detections += 1
                    else:
                        missed_zombies += 1
                else:
                    if is_zombie_predicted:
                        false_positives += 1
                    else:
                        correct_non_detections += 1

            # Reset world for next episode with new zombie distribution
            self.world.reset()

        total = num_episodes * self.num_agents
        total_zombies = num_episodes * self.num_zombies
        total_non_zombies = total - total_zombies

        return {
            "zombie_detection_accuracy": correct_detections / max(total_zombies, 1),
            "false_positive_rate": false_positives / max(total_non_zombies, 1),
            "true_negative_rate": correct_non_detections / max(total_non_zombies, 1),
            "miss_rate": missed_zombies / max(total_zombies, 1),
            "overall_accuracy": (correct_detections + correct_non_detections) / max(total, 1),
        }

    def evaluate_deception_detection(
        self, model: nn.Module, num_scenarios: int = 50, device: str = "cpu"
    ) -> DeceptionMetrics:
        """
        Detect when agents communicate false information.

        Requires modeling beliefs vs reality vs intent:
        - What does the speaker believe?
        - What is actually true?
        - Does the speaker intend to deceive?
        """
        model = model.to(device)
        model.eval()

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for scenario_idx in range(num_scenarios):
            # Create deception scenario
            is_deception = random.random() > 0.5

            # Build observation
            observation = self._build_deception_observation(is_deception)
            observation = observation.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(observation)

            # Model predicts deception
            deception_predicted = output[:, 0].sigmoid().item() > 0.5

            if is_deception:
                if deception_predicted:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if deception_predicted:
                    false_positives += 1
                else:
                    true_negatives += 1

        total = max(num_scenarios, 1)
        positives = true_positives + false_negatives
        negatives = true_negatives + false_positives

        return DeceptionMetrics(
            true_positive_rate=true_positives / max(positives, 1),
            false_positive_rate=false_positives / max(negatives, 1),
            true_negative_rate=true_negatives / max(negatives, 1),
            false_negative_rate=false_negatives / max(positives, 1),
        )

    def evaluate_fairness_modeling(
        self, model: nn.Module, num_rounds: int = 30, device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Test understanding of fairness in resource distribution.

        ToM helps predict:
        - What others consider fair
        - How others will react to unfair distributions
        - When to accept unfair offers (strategic reasoning)
        """
        model = model.to(device)
        model.eval()

        fairness_predictions = []
        actual_responses = []
        rejection_predictions = []

        for round_idx in range(num_rounds):
            # Ultimatum game setup
            proposer_id = random.randint(0, self.num_agents - 1)
            responder_id = random.randint(0, self.num_agents - 1)
            while responder_id == proposer_id:
                responder_id = random.randint(0, self.num_agents - 1)

            total_resource = 100.0
            offer_fraction = random.random()  # How much proposer offers

            # Build observation
            observation = self._build_fairness_observation(proposer_id, responder_id, offer_fraction)
            observation = observation.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(observation)

            # Model predicts: will responder accept?
            # And what fraction would be considered "fair"?
            accept_predicted = output[:, 0].sigmoid().item() > 0.5
            fair_fraction_predicted = output[:, 1].sigmoid().item() if output.shape[1] > 1 else 0.5

            # Simulate responder behavior (reject unfair offers)
            responder = self.world.agents[responder_id]
            fairness_threshold = 0.3 + 0.2 * responder.reputation.get(proposer_id, 0.5)

            actual_accept = offer_fraction >= fairness_threshold

            fairness_predictions.append(fair_fraction_predicted)
            rejection_predictions.append(accept_predicted == actual_accept)

        correct_predictions = sum(rejection_predictions)

        return {
            "acceptance_prediction_accuracy": correct_predictions / max(num_rounds, 1),
            "fairness_estimation_error": np.mean(
                [abs(pred - 0.5) for pred in fairness_predictions]  # Distance from true fair (50-50)
            ),
            "num_rounds": num_rounds,
        }

    def full_evaluation(self, model: nn.Module, device: str = "cpu") -> SocialGameResult:
        """Run all social game benchmarks."""
        self.reset_world()

        # Run individual evaluations
        coop_metrics = self.evaluate_cooperation(model, num_rounds=50, device=device)
        zombie_metrics = self.evaluate_zombie_detection(model, num_episodes=20, device=device)
        deception_metrics = self.evaluate_deception_detection(model, num_scenarios=50, device=device)
        fairness_metrics = self.evaluate_fairness_modeling(model, num_rounds=30, device=device)

        # Aggregate metrics
        prediction_accuracy = (
            coop_metrics.reputation_tracking_accuracy * 0.3
            + zombie_metrics["zombie_detection_accuracy"] * 0.4
            + deception_metrics.true_positive_rate * 0.3
        )

        # Total reward (simplified)
        total_reward = (
            coop_metrics.mutual_cooperation_rate * 10
            - coop_metrics.vulnerability_rate * 5
            + zombie_metrics["zombie_detection_accuracy"] * 20
            - zombie_metrics["false_positive_rate"] * 10
            + fairness_metrics["acceptance_prediction_accuracy"] * 5
        )

        return SocialGameResult(
            game_type="full_evaluation",
            tom_required=True,
            cooperation_rate=coop_metrics.mutual_cooperation_rate,
            prediction_accuracy=prediction_accuracy,
            fairness_score=fairness_metrics["acceptance_prediction_accuracy"],
            zombie_detection_accuracy=zombie_metrics["zombie_detection_accuracy"],
            deception_detection_accuracy=deception_metrics.true_positive_rate,
            total_reward=total_reward,
            num_episodes=50 + 20 + 50 + 30,  # Total across all tests
        )

    def _build_cooperation_observation(self, partner_id: int, histories: Dict[int, List[str]]) -> torch.Tensor:
        """Build observation tensor for cooperation game."""
        partner = self.world.agents[partner_id]

        # Partner's observable state
        obs = torch.zeros(self.ontology_dim + 20)

        # Partner's ontology state
        if partner.ontology_state is not None:
            obs[: self.ontology_dim] = partner.ontology_state

        # Reputation information
        obs[self.ontology_dim] = partner.reputation.get(0, 0.5)  # Partner's rep with model
        obs[self.ontology_dim + 1] = partner.resources / 200.0  # Normalized resources

        # Interaction history features
        history = histories.get(partner_id, [])
        if history:
            # Recent cooperation rate
            recent = history[-5:]
            coop_rate = sum(1 for a in recent if a == "cooperate") / len(recent)
            obs[self.ontology_dim + 2] = coop_rate
            obs[self.ontology_dim + 3] = len(history) / 50.0  # Interaction count

        return obs

    def _build_zombie_detection_observation(self, agent_idx: int) -> torch.Tensor:
        """Build observation for zombie detection."""
        agent = self.world.agents[agent_idx]
        obs = torch.zeros(self.ontology_dim + 30)

        # Agent's ontology state
        if agent.ontology_state is not None:
            obs[: self.ontology_dim] = agent.ontology_state

        # Behavioral features (zombies have subtly different patterns)
        obs[self.ontology_dim] = agent.resources / 200.0
        obs[self.ontology_dim + 1] = agent.energy / 100.0

        # Reputation variance (zombies have more uniform/inconsistent patterns)
        reps = list(agent.reputation.values())
        if reps:
            obs[self.ontology_dim + 2] = np.std(reps)
            obs[self.ontology_dim + 3] = np.mean(reps)

        # Coalition membership
        obs[self.ontology_dim + 4] = 1.0 if agent.coalition is not None else 0.0

        return obs

    def _build_deception_observation(self, is_deception: bool) -> torch.Tensor:
        """Build observation for deception detection scenario."""
        obs = torch.zeros(self.ontology_dim + 40)

        # Speaker's claimed state
        claimed_state = torch.randn(self.ontology_dim // 2)
        obs[: self.ontology_dim // 2] = claimed_state

        # Context/evidence (may contradict claim if deception)
        if is_deception:
            # Evidence contradicts claim
            evidence = -claimed_state + torch.randn(self.ontology_dim // 2) * 0.3
        else:
            # Evidence supports claim
            evidence = claimed_state + torch.randn(self.ontology_dim // 2) * 0.1

        obs[self.ontology_dim // 2 : self.ontology_dim] = evidence

        # Speaker's reputation (liars may have lower reputation)
        obs[self.ontology_dim] = random.uniform(0.2, 0.8) if is_deception else random.uniform(0.5, 1.0)

        return obs

    def _build_fairness_observation(self, proposer_id: int, responder_id: int, offer_fraction: float) -> torch.Tensor:
        """Build observation for fairness/ultimatum game."""
        obs = torch.zeros(self.ontology_dim + 20)

        proposer = self.world.agents[proposer_id]
        responder = self.world.agents[responder_id]

        # Responder's state (who we're predicting)
        if responder.ontology_state is not None:
            obs[: self.ontology_dim] = responder.ontology_state

        # Offer details
        obs[self.ontology_dim] = offer_fraction
        obs[self.ontology_dim + 1] = 1.0 - offer_fraction  # Proposer keeps

        # Relationship
        obs[self.ontology_dim + 2] = responder.reputation.get(proposer_id, 0.5)
        obs[self.ontology_dim + 3] = proposer.reputation.get(responder_id, 0.5)

        # Resource context
        obs[self.ontology_dim + 4] = responder.resources / 200.0
        obs[self.ontology_dim + 5] = proposer.resources / 200.0

        return obs


# Export
__all__ = [
    "SocialGameBenchmark",
    "SocialGameResult",
    "CooperationMetrics",
    "DeceptionMetrics",
]
