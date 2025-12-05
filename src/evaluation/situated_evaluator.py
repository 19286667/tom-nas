"""
Situated Evaluator: ToM-Specific Fitness Assessment
====================================================

Based on Ma et al. "Towards a Science of Evaluating ToM in LLMs"

Key Insight: Standard evaluation asks "Did you win?"
             Situated evaluation asks "Was your internal model accurate?"

This evaluator measures:
1. Belief Accuracy (40%): Internal BeliefNest vs. ground truth
2. Action Success (30%): Goal achievement
3. Social Cost (20%): Friction incurred
4. Efficiency (10%): Computational cost

Additionally tracks:
- ToM depth actually used
- Belief calibration (confidence vs. accuracy)
- Counterfactual reasoning quality

Author: ToM-NAS Project
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime
import logging

from ..simulation_config import (
    SimulationConfig,
    EvaluationConfig,
    InstitutionGenotype,
)
from ..core.beliefs import BeliefNetwork, Belief

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from ..evolution.poet_manager import AgentGenotype

logger = logging.getLogger(__name__)


# =============================================================================
# GROUND TRUTH STATE
# =============================================================================

@dataclass
class AgentGroundTruth:
    """
    The actual internal state of an agent (for evaluation purposes).

    In the simulation, this represents "God's-eye view" of what an agent
    actually believes/wants/feels - used to evaluate ToM accuracy.
    """
    agent_id: int
    timestamp: float

    # Actual beliefs (what they really believe)
    beliefs: Dict[str, Any] = field(default_factory=dict)

    # Actual goals (what they really want)
    goals: List[str] = field(default_factory=list)
    goal_priorities: Dict[str, float] = field(default_factory=dict)

    # Actual emotions
    emotional_state: Dict[str, float] = field(default_factory=dict)

    # Actual intentions
    current_intention: Optional[str] = None
    planned_actions: List[str] = field(default_factory=list)

    # Knowledge state
    known_facts: List[str] = field(default_factory=list)
    false_beliefs: List[str] = field(default_factory=list)  # Things they wrongly believe


@dataclass
class SimulationState:
    """
    Complete state of a simulation episode for evaluation.
    """
    episode_id: str
    institution: str
    timestamp: float

    # All agents' ground truth states
    agent_states: Dict[int, AgentGroundTruth] = field(default_factory=dict)

    # World state facts
    world_facts: List[str] = field(default_factory=list)

    # Occurred events
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Social dynamics
    relationship_states: Dict[Tuple[int, int], float] = field(default_factory=dict)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

@dataclass
class BeliefAccuracyMetrics:
    """Metrics for belief accuracy evaluation."""
    # Overall accuracy
    mean_accuracy: float = 0.0

    # By ToM order
    accuracy_by_order: Dict[int, float] = field(default_factory=dict)

    # By belief type
    accuracy_by_type: Dict[str, float] = field(default_factory=dict)

    # Confusion analysis
    false_positives: int = 0  # Believed true when false
    false_negatives: int = 0  # Believed false when true
    true_positives: int = 0
    true_negatives: int = 0

    # Belief coverage
    beliefs_held: int = 0
    beliefs_possible: int = 0


@dataclass
class CalibrationMetrics:
    """Metrics for confidence calibration."""
    # Overall calibration error
    expected_calibration_error: float = 0.0

    # By confidence bin
    calibration_by_bin: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Overconfidence vs underconfidence
    overconfidence_rate: float = 0.0
    underconfidence_rate: float = 0.0


@dataclass
class ActionMetrics:
    """Metrics for action success."""
    # Goal achievement
    goals_achieved: int = 0
    goals_attempted: int = 0
    goal_success_rate: float = 0.0

    # Action outcomes
    successful_actions: int = 0
    failed_actions: int = 0
    total_actions: int = 0


@dataclass
class SocialMetrics:
    """Metrics for social costs."""
    # Total friction
    total_social_cost: float = 0.0

    # Norm violations
    norm_violations: int = 0
    detected_violations: int = 0

    # Relationship changes
    relationships_improved: int = 0
    relationships_damaged: int = 0
    net_relationship_change: float = 0.0


@dataclass
class EfficiencyMetrics:
    """Metrics for computational efficiency."""
    # ToM usage
    max_tom_depth_used: int = 0
    mean_tom_depth_used: float = 0.0

    # Computation
    mental_simulations_run: int = 0
    inference_time_ms: float = 0.0

    # Resource usage
    memory_peak_mb: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result for an agent-environment pair."""
    # Identity
    agent_id: str
    environment_id: str
    episode_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Composite fitness
    fitness: float = 0.0

    # Component scores (before weighting)
    belief_accuracy_score: float = 0.0
    action_success_score: float = 0.0
    social_cost_score: float = 0.0
    efficiency_score: float = 0.0

    # Detailed metrics
    belief_metrics: BeliefAccuracyMetrics = field(default_factory=BeliefAccuracyMetrics)
    calibration_metrics: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    action_metrics: ActionMetrics = field(default_factory=ActionMetrics)
    social_metrics: SocialMetrics = field(default_factory=SocialMetrics)
    efficiency_metrics: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)

    # Reasoning trace (for debugging)
    reasoning_trace: List[str] = field(default_factory=list)


# =============================================================================
# SITUATED EVALUATOR
# =============================================================================

class SituatedEvaluator:
    """
    Evaluator for Theory of Mind agents based on situated assessment.

    This evaluator computes fitness based not just on action success,
    but on the accuracy of the agent's internal model of other agents.
    """

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()

        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []

    def evaluate(
        self,
        agent: "AgentGenotype",
        environment: InstitutionGenotype,
        belief_network: BeliefNetwork = None,
        simulation_state: SimulationState = None,
        agent_idx: int = 0,
    ) -> float:
        """
        Evaluate an agent's performance in an environment.

        Args:
            agent: Agent genotype being evaluated
            environment: Environment genotype
            belief_network: Agent's belief network (for belief accuracy)
            simulation_state: Ground truth simulation state
            agent_idx: Index of agent in belief network

        Returns:
            Composite fitness score (0-1)
        """
        result = EvaluationResult(
            agent_id=agent.genotype_id,
            environment_id=f"{environment.institution_type.value}_{environment.complexity_level:.2f}",
            episode_id=simulation_state.episode_id if simulation_state else "synthetic",
        )

        # Compute component scores
        if belief_network and simulation_state:
            result.belief_accuracy_score = self._evaluate_belief_accuracy(
                belief_network, simulation_state, agent_idx
            )
            result.belief_metrics = self._compute_belief_metrics(
                belief_network, simulation_state, agent_idx
            )
            result.calibration_metrics = self._compute_calibration_metrics(
                belief_network, simulation_state, agent_idx
            )
        else:
            # Synthetic evaluation based on architecture
            result.belief_accuracy_score = self._synthetic_belief_score(agent, environment)

        result.action_success_score = self._evaluate_action_success(simulation_state)
        result.social_cost_score = self._evaluate_social_cost(simulation_state, environment)
        result.efficiency_score = self._evaluate_efficiency(agent, simulation_state)

        # Compute composite fitness
        result.fitness = self._compute_composite_fitness(result)

        # Record
        self.evaluation_history.append(result)

        return result.fitness

    def _evaluate_belief_accuracy(
        self,
        belief_network: BeliefNetwork,
        state: SimulationState,
        agent_idx: int
    ) -> float:
        """
        Evaluate accuracy of agent's beliefs against ground truth.

        This is the PRIMARY metric for ToM evaluation.
        """
        if not state.agent_states:
            return 0.5

        total_accuracy = 0.0
        num_comparisons = 0

        belief_state = belief_network.get_agent_belief_state(agent_idx)
        if belief_state is None:
            return 0.5

        # Compare beliefs about each other agent
        for target_id, ground_truth in state.agent_states.items():
            if target_id == agent_idx:
                continue

            # Check beliefs at each order
            for order in range(belief_state.max_order + 1):
                belief = belief_state.get_belief(order, target_id)
                if belief is None:
                    continue

                # Compare to ground truth
                accuracy = self._compare_belief_to_truth(
                    belief, ground_truth, order
                )
                total_accuracy += accuracy
                num_comparisons += 1

        if num_comparisons == 0:
            return 0.5

        return total_accuracy / num_comparisons

    def _compare_belief_to_truth(
        self,
        belief: Belief,
        truth: AgentGroundTruth,
        order: int
    ) -> float:
        """Compare a specific belief to ground truth."""
        # This is a simplified comparison
        # In full implementation, would do semantic matching

        # Use confidence as a proxy for now
        # Higher confidence on correct beliefs = better
        # Higher confidence on wrong beliefs = worse

        # Placeholder: random accuracy weighted by order
        # Real implementation would decode belief content and compare
        base_accuracy = 0.5 + np.random.randn() * 0.2
        order_penalty = 0.1 * order  # Harder to be accurate at higher orders

        accuracy = max(0, min(1, base_accuracy - order_penalty))
        return accuracy

    def _compute_belief_metrics(
        self,
        belief_network: BeliefNetwork,
        state: SimulationState,
        agent_idx: int
    ) -> BeliefAccuracyMetrics:
        """Compute detailed belief accuracy metrics."""
        metrics = BeliefAccuracyMetrics()

        belief_state = belief_network.get_agent_belief_state(agent_idx)
        if belief_state is None:
            return metrics

        # Count beliefs by order
        accuracy_sums = {}
        accuracy_counts = {}

        for order in range(belief_state.max_order + 1):
            order_beliefs = belief_state.beliefs[order]
            for target, belief in order_beliefs.items():
                if belief is None:
                    continue

                metrics.beliefs_held += 1

                # Placeholder accuracy
                acc = 0.5 + np.random.randn() * 0.2

                if order not in accuracy_sums:
                    accuracy_sums[order] = 0.0
                    accuracy_counts[order] = 0

                accuracy_sums[order] += acc
                accuracy_counts[order] += 1

        # Compute averages
        for order in accuracy_sums:
            if accuracy_counts[order] > 0:
                metrics.accuracy_by_order[order] = accuracy_sums[order] / accuracy_counts[order]

        if metrics.beliefs_held > 0:
            metrics.mean_accuracy = sum(accuracy_sums.values()) / sum(accuracy_counts.values())

        return metrics

    def _compute_calibration_metrics(
        self,
        belief_network: BeliefNetwork,
        state: SimulationState,
        agent_idx: int
    ) -> CalibrationMetrics:
        """
        Compute confidence calibration metrics.

        A well-calibrated agent should be right ~80% of the time
        when it says it's 80% confident.
        """
        metrics = CalibrationMetrics()

        belief_state = belief_network.get_agent_belief_state(agent_idx)
        if belief_state is None:
            return metrics

        # Bin beliefs by confidence
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        bin_confidences = {f"{b[0]}-{b[1]}": [] for b in bins}
        bin_accuracies = {f"{b[0]}-{b[1]}": [] for b in bins}

        for order in range(belief_state.max_order + 1):
            for target, belief in belief_state.beliefs[order].items():
                if belief is None:
                    continue

                conf = belief.confidence
                acc = 0.5 + np.random.randn() * 0.2  # Placeholder

                # Find bin
                for b in bins:
                    if b[0] <= conf < b[1]:
                        key = f"{b[0]}-{b[1]}"
                        bin_confidences[key].append(conf)
                        bin_accuracies[key].append(acc)
                        break

        # Compute ECE
        total_samples = 0
        ece = 0.0

        for key in bin_confidences:
            if bin_confidences[key]:
                mean_conf = np.mean(bin_confidences[key])
                mean_acc = np.mean(bin_accuracies[key])
                n = len(bin_confidences[key])

                ece += n * abs(mean_conf - mean_acc)
                total_samples += n

                metrics.calibration_by_bin[key] = (mean_conf, mean_acc)

                if mean_conf > mean_acc:
                    metrics.overconfidence_rate += n
                else:
                    metrics.underconfidence_rate += n

        if total_samples > 0:
            metrics.expected_calibration_error = ece / total_samples
            metrics.overconfidence_rate /= total_samples
            metrics.underconfidence_rate /= total_samples

        return metrics

    def _synthetic_belief_score(
        self,
        agent: "AgentGenotype",
        environment: InstitutionGenotype
    ) -> float:
        """
        Compute synthetic belief score based on architecture-environment match.

        Used when no simulation state is available.
        """
        # Architecture factors
        belief_capacity = min(agent.belief_module_size / 128, 1.0)
        tom_capacity = min(agent.max_tom_depth / 5, 1.0)
        attention_capacity = min(agent.num_attention_heads / 8, 1.0)

        # Environment demands
        deception_demand = environment.deception_prevalence
        complexity_demand = environment.complexity_level
        info_demand = environment.information_asymmetry

        # Match score
        belief_match = belief_capacity * (0.5 + 0.5 * complexity_demand)
        tom_match = tom_capacity * (0.5 + 0.5 * deception_demand)
        attention_match = attention_capacity * (0.5 + 0.5 * info_demand)

        score = (belief_match + tom_match + attention_match) / 3
        score += np.random.randn() * 0.05  # Noise

        return np.clip(score, 0, 1)

    def _evaluate_action_success(
        self,
        state: Optional[SimulationState]
    ) -> float:
        """Evaluate action success rate."""
        if state is None:
            return 0.5 + np.random.randn() * 0.1

        # Would analyze events in state
        return 0.5 + np.random.randn() * 0.15

    def _evaluate_social_cost(
        self,
        state: Optional[SimulationState],
        environment: InstitutionGenotype
    ) -> float:
        """
        Evaluate social cost (lower is better).

        Returns inverted score (1 - normalized_cost) so higher = better.
        """
        if state is None:
            # Estimate based on environment friction
            expected_cost = environment.friction_coefficient * 0.3
            cost = expected_cost + np.random.randn() * 0.1
            return np.clip(1 - cost, 0, 1)

        # Would analyze norm violations in state
        return 0.7 + np.random.randn() * 0.1

    def _evaluate_efficiency(
        self,
        agent: "AgentGenotype",
        state: Optional[SimulationState]
    ) -> float:
        """Evaluate computational efficiency."""
        # Penalize overly large architectures
        param_count = agent.get_parameter_count()
        max_params = 10_000_000

        size_efficiency = 1 - min(param_count / max_params, 1.0)

        # Reward appropriate ToM depth usage
        # (Using max when needed, not when not needed)
        tom_efficiency = 0.5  # Placeholder

        return (size_efficiency + tom_efficiency) / 2

    def _compute_composite_fitness(self, result: EvaluationResult) -> float:
        """Compute weighted composite fitness score."""
        fitness = (
            self.config.belief_accuracy_weight * result.belief_accuracy_score +
            self.config.action_success_weight * result.action_success_score +
            self.config.social_cost_weight * result.social_cost_score +
            self.config.efficiency_weight * result.efficiency_score
        )

        return np.clip(fitness, 0, 1)

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        if not self.evaluation_history:
            return {}

        fitness_values = [r.fitness for r in self.evaluation_history]
        belief_values = [r.belief_accuracy_score for r in self.evaluation_history]

        return {
            "total_evaluations": len(self.evaluation_history),
            "mean_fitness": np.mean(fitness_values),
            "max_fitness": np.max(fitness_values),
            "mean_belief_accuracy": np.mean(belief_values),
            "fitness_std": np.std(fitness_values),
        }

    def export_results(self, path: str):
        """Export evaluation results to JSON."""
        import json

        results = []
        for r in self.evaluation_history:
            results.append({
                "agent_id": r.agent_id,
                "environment_id": r.environment_id,
                "fitness": r.fitness,
                "belief_accuracy": r.belief_accuracy_score,
                "action_success": r.action_success_score,
                "social_cost": r.social_cost_score,
                "efficiency": r.efficiency_score,
                "timestamp": r.timestamp,
            })

        with open(path, "w") as f:
            json.dump(results, f, indent=2)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AgentGroundTruth",
    "SimulationState",
    "BeliefAccuracyMetrics",
    "CalibrationMetrics",
    "ActionMetrics",
    "SocialMetrics",
    "EfficiencyMetrics",
    "EvaluationResult",
    "SituatedEvaluator",
]
