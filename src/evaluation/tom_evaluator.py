"""
Evaluate models on Theory of Mind tasks with proper ground truth.

This module provides rigorous evaluation of ToM capability by:
1. Using scenarios with information asymmetry
2. Computing ground truth from observation tracking
3. Separately measuring reality vs belief question performance
4. Comparing against baselines that don't require ToM
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

try:
    # When imported as part of src package
    from ..data.events import Scenario
    from ..data.encoding import ScenarioEncoder
    from ..data.beliefs import BeliefComputer
except ImportError:
    # When imported directly (e.g., from scripts)
    from data.events import Scenario
    from data.encoding import ScenarioEncoder
    from data.beliefs import BeliefComputer


class ToMEvaluator:
    """
    Evaluate Theory of Mind capability with proper ground truth.

    The key insight: a model truly demonstrates ToM if it can:
    1. Answer reality questions correctly (knows actual state)
    2. Answer belief questions correctly (knows agents' beliefs differ)

    A model that only tracks reality will fail belief questions
    when agents have false beliefs.
    """

    def __init__(self, encoder: ScenarioEncoder, device: str = 'cpu'):
        """
        Initialize evaluator.

        Args:
            encoder: ScenarioEncoder for encoding scenarios
            device: Device to run evaluation on
        """
        self.encoder = encoder
        self.belief_computer = BeliefComputer()
        self.device = device

    def evaluate(self, model: nn.Module,
                 scenarios: List[Scenario],
                 batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate model on scenarios, broken down by question type.

        Args:
            model: PyTorch model to evaluate
            scenarios: List of scenarios to test on
            batch_size: Batch size for evaluation

        Returns:
            Dict with accuracy metrics broken down by question type
        """
        model.eval()
        model.to(self.device)

        results = {
            'reality_correct': 0,
            'reality_total': 0,
            'first_order_correct': 0,
            'first_order_total': 0,
            'second_order_correct': 0,
            'second_order_total': 0,
        }

        predictions = []
        ground_truths = []

        with torch.no_grad():
            for i in range(0, len(scenarios), batch_size):
                batch = scenarios[i:i + batch_size]

                for scenario in batch:
                    # Skip scenarios with no ground truth
                    if scenario.ground_truth_answer is None:
                        continue

                    # Encode scenario
                    encoded = self.encoder.encode_scenario(scenario)
                    encoded = encoded.unsqueeze(0).to(self.device)

                    # Get model prediction
                    output = model(encoded)

                    # Handle different output formats
                    pred_vec = self._extract_prediction(output)

                    # Decode to location
                    predicted = self.encoder.decode_location(pred_vec)
                    predictions.append(predicted)
                    ground_truths.append(scenario.ground_truth_answer)

                    # Check against ground truth
                    correct = (predicted == scenario.ground_truth_answer)

                    # Track by question type
                    q_type = scenario.question_type
                    if q_type == 'reality':
                        results['reality_total'] += 1
                        if correct:
                            results['reality_correct'] += 1
                    elif q_type == 'first_order_belief':
                        results['first_order_total'] += 1
                        if correct:
                            results['first_order_correct'] += 1
                    elif q_type == 'second_order_belief':
                        results['second_order_total'] += 1
                        if correct:
                            results['second_order_correct'] += 1

        # Compute accuracies
        for prefix in ['reality', 'first_order', 'second_order']:
            total = results[f'{prefix}_total']
            if total > 0:
                results[f'{prefix}_accuracy'] = (
                    results[f'{prefix}_correct'] / total
                )
            else:
                results[f'{prefix}_accuracy'] = 0.0

        # Overall ToM accuracy (excluding reality questions)
        tom_correct = (results['first_order_correct'] +
                      results['second_order_correct'])
        tom_total = (results['first_order_total'] +
                    results['second_order_total'])

        results['tom_accuracy'] = tom_correct / tom_total if tom_total > 0 else 0.0
        results['overall_accuracy'] = (
            (results['reality_correct'] + tom_correct) /
            (results['reality_total'] + tom_total)
            if (results['reality_total'] + tom_total) > 0 else 0.0
        )

        # Additional metrics
        results['predictions'] = predictions
        results['ground_truths'] = ground_truths

        return results

    def _extract_prediction(self, output) -> torch.Tensor:
        """Extract prediction vector from various output formats."""
        if isinstance(output, dict):
            # Try common keys
            for key in ['beliefs', 'output', 'logits', 'hidden_states']:
                if key in output:
                    output = output[key]
                    break
            else:
                output = list(output.values())[0]

        if isinstance(output, tuple):
            output = output[0]

        # Get last timestep
        if output.dim() == 3:
            output = output[0, -1]  # batch=0, last timestep
        elif output.dim() == 2:
            output = output[0]  # batch=0

        return output

    def evaluate_with_confusion(self, model: nn.Module,
                                scenarios: List[Scenario]) -> Dict[str, Any]:
        """
        Evaluate with detailed confusion matrix.

        Returns accuracy plus confusion matrix for error analysis.
        """
        results = self.evaluate(model, scenarios)

        # Build confusion matrix for locations
        num_locs = len(self.encoder.locations)
        confusion = torch.zeros(num_locs, num_locs)

        for pred, truth in zip(results['predictions'], results['ground_truths']):
            pred_idx = self.encoder.get_location_index(pred)
            truth_idx = self.encoder.get_location_index(truth)
            if pred_idx >= 0 and truth_idx >= 0:
                confusion[truth_idx, pred_idx] += 1

        results['confusion_matrix'] = confusion
        results['location_names'] = self.encoder.locations

        return results


class BaselineEvaluator:
    """
    Reality-only baseline that ignores observations.

    This baseline always predicts the ACTUAL final location,
    ignoring any information asymmetry. It should:
    - Score high on reality questions
    - Score LOW on belief questions (when beliefs differ from reality)

    If it scores high on belief questions too, then our scenarios
    don't actually require Theory of Mind!
    """

    def __init__(self, encoder: ScenarioEncoder):
        """
        Initialize baseline evaluator.

        Args:
            encoder: ScenarioEncoder for location vocabulary
        """
        self.encoder = encoder

    def evaluate(self, scenarios: List[Scenario]) -> Dict[str, float]:
        """
        Evaluate the 'cheating' reality-only baseline.

        This baseline ignores who observed what and always predicts
        the actual current location of objects.
        """
        results = {
            'reality_correct': 0, 'reality_total': 0,
            'first_order_correct': 0, 'first_order_total': 0,
            'second_order_correct': 0, 'second_order_total': 0,
        }

        for scenario in scenarios:
            if scenario.ground_truth_answer is None:
                continue

            # Compute reality (final object location) from ALL events
            reality = self._compute_reality(scenario)

            # This baseline ALWAYS predicts reality
            correct = (reality == scenario.ground_truth_answer)

            q_type = scenario.question_type
            if q_type == 'reality':
                results['reality_total'] += 1
                if correct:
                    results['reality_correct'] += 1
            elif q_type == 'first_order_belief':
                results['first_order_total'] += 1
                if correct:
                    results['first_order_correct'] += 1
            elif q_type == 'second_order_belief':
                results['second_order_total'] += 1
                if correct:
                    results['second_order_correct'] += 1

        # Compute accuracies
        for prefix in ['reality', 'first_order', 'second_order']:
            total = results[f'{prefix}_total']
            if total > 0:
                results[f'{prefix}_accuracy'] = results[f'{prefix}_correct'] / total
            else:
                results[f'{prefix}_accuracy'] = 0.0

        # The key test: baseline should ACE reality but FAIL beliefs
        results['tom_accuracy'] = (
            (results['first_order_correct'] + results['second_order_correct']) /
            (results['first_order_total'] + results['second_order_total'])
            if (results['first_order_total'] + results['second_order_total']) > 0 else 0.0
        )

        return results

    def _compute_reality(self, scenario: Scenario) -> Optional[str]:
        """Compute actual object location from all events."""
        obj = scenario.question_target_object
        location = None
        for event in scenario.events:
            if event.object == obj and event.action in ('put', 'move'):
                location = event.target_location
        return location


class RandomBaselineEvaluator:
    """Random baseline for sanity checking."""

    def __init__(self, encoder: ScenarioEncoder, seed: int = 42):
        self.encoder = encoder
        self.rng = torch.Generator().manual_seed(seed)

    def evaluate(self, scenarios: List[Scenario]) -> Dict[str, float]:
        """Evaluate random predictions."""
        results = {
            'reality_correct': 0, 'reality_total': 0,
            'first_order_correct': 0, 'first_order_total': 0,
        }

        for scenario in scenarios:
            if scenario.ground_truth_answer is None:
                continue

            # Random location
            random_idx = torch.randint(0, len(self.encoder.locations), (1,),
                                      generator=self.rng).item()
            predicted = self.encoder.locations[random_idx]
            correct = (predicted == scenario.ground_truth_answer)

            q_type = scenario.question_type
            if q_type == 'reality':
                results['reality_total'] += 1
                if correct:
                    results['reality_correct'] += 1
            elif q_type == 'first_order_belief':
                results['first_order_total'] += 1
                if correct:
                    results['first_order_correct'] += 1

        for prefix in ['reality', 'first_order']:
            total = results[f'{prefix}_total']
            results[f'{prefix}_accuracy'] = results[f'{prefix}_correct'] / total if total > 0 else 0.0

        return results


def compute_tom_improvement(model_results: Dict, baseline_results: Dict) -> Dict[str, float]:
    """
    Compute how much a model improves over the reality baseline.

    Positive improvement on belief questions indicates the model
    is actually learning ToM rather than just tracking reality.
    """
    improvement = {}

    for key in ['reality_accuracy', 'first_order_accuracy', 'second_order_accuracy', 'tom_accuracy']:
        if key in model_results and key in baseline_results:
            improvement[f'{key}_improvement'] = model_results[key] - baseline_results[key]

    # Critical metric: improvement on ToM questions
    improvement['learns_tom'] = improvement.get('tom_accuracy_improvement', 0) > 0.1

    return improvement
