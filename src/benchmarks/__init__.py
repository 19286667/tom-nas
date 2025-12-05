"""
ToM-NAS Benchmark Suite

Unified benchmarks for evaluating Theory of Mind capabilities:
- ToMi: False belief scenarios (Sally-Anne style)
- SocialIQA: Naturalistic social reasoning
- Social Games: Interactive multi-agent scenarios

The UnifiedBenchmark class provides a single interface for running
all benchmarks and computing aggregate ToM scores.
"""

import torch.nn as nn
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .tomi_loader import (
    ToMiDataset,
    ToMiParser,
    ToMiEvaluator,
    ToMiExample,
    ToMiQuestion,
    EventEncoder,
)
from .social_games import (
    SocialGameBenchmark,
    SocialGameResult,
    CooperationMetrics,
    DeceptionMetrics,
)
from .socialIQA_loader import (
    SocialIQADataset,
    SocialIQAExample,
    SocialIQAEncoder,
    SocialIQAEvaluator,
    SocialIQAMetrics,
)


@dataclass
class UnifiedBenchmarkResult:
    """Results from unified benchmark evaluation."""
    # ToMi Results
    tomi_tom_accuracy: float
    tomi_control_accuracy: float
    tomi_specificity: float
    tomi_first_order: float
    tomi_second_order: float

    # SocialIQA Results
    social_iqa_accuracy: float
    social_iqa_intent: float
    social_iqa_emotion: float
    social_iqa_motivation: float

    # Social Games Results
    social_games_cooperation: float
    social_games_prediction: float
    social_games_zombie_detection: float
    social_games_deception_detection: float

    # Aggregate Scores
    tom_aggregate: float
    control_aggregate: float
    tom_specificity: float  # tom_aggregate - control_aggregate

    # Metadata
    total_examples: int
    model_name: str = ""


class UnifiedBenchmark:
    """
    Unified benchmark suite combining all ToM tests.

    Components:
    - ToMi: False belief scenarios (Sally-Anne style)
    - SocialIQA: Naturalistic social reasoning
    - Social Games: Interactive multi-agent scenarios

    Usage:
        benchmark = UnifiedBenchmark()
        results = benchmark.full_evaluation(model, device='cuda')
        print(f"ToM Aggregate Score: {results['tom_aggregate']:.3f}")
    """

    def __init__(self, tomi_data_dir: Optional[str] = None,
                 social_iqa_data_dir: Optional[str] = None,
                 num_social_agents: int = 10,
                 seed: Optional[int] = None):
        """
        Initialize all benchmark components.

        Args:
            tomi_data_dir: Path to ToMi dataset (uses synthetic if None)
            social_iqa_data_dir: Path to SocialIQA dataset (uses synthetic if None)
            num_social_agents: Number of agents in social games
            seed: Random seed for reproducibility
        """
        # Initialize datasets
        self.tomi = ToMiDataset(tomi_data_dir)
        self.tomi_evaluator = ToMiEvaluator(self.tomi)

        self.social_iqa = SocialIQADataset(social_iqa_data_dir)
        self.social_iqa_evaluator = SocialIQAEvaluator(self.social_iqa)

        self.social_games = SocialGameBenchmark(
            num_agents=num_social_agents,
            num_zombies=2,
            seed=seed
        )

        # Configuration
        self.weights = {
            'tomi': 0.4,
            'social_iqa': 0.3,
            'social_games': 0.3,
        }

    def evaluate_tomi(self, model: nn.Module,
                      num_samples: int = 100,
                      device: str = 'cpu') -> Dict[str, float]:
        """
        Evaluate model on ToMi false belief scenarios.

        Returns:
            Dictionary with tom_accuracy, control_accuracy, specificity,
            first_order_accuracy, second_order_accuracy
        """
        results = self.tomi_evaluator.evaluate(model, num_samples)
        return results

    def evaluate_social_iqa(self, model: nn.Module,
                            num_samples: int = 100,
                            device: str = 'cpu') -> Dict[str, float]:
        """
        Evaluate model on SocialIQA questions.

        Returns:
            Dictionary with overall accuracy and per-type accuracies
        """
        metrics = self.social_iqa_evaluator.evaluate(model, num_samples, device)
        return {
            'accuracy': metrics.accuracy,
            'intent_accuracy': metrics.intent_accuracy,
            'emotion_accuracy': metrics.emotion_accuracy,
            'motivation_accuracy': metrics.motivation_accuracy,
            'subsequent_accuracy': metrics.subsequent_accuracy,
            'prerequisite_accuracy': metrics.prerequisite_accuracy,
            'num_examples': metrics.num_examples,
        }

    def evaluate_social_games(self, model: nn.Module,
                              device: str = 'cpu') -> SocialGameResult:
        """
        Evaluate model on interactive social games.

        Returns:
            SocialGameResult with cooperation, prediction, and detection metrics
        """
        return self.social_games.full_evaluation(model, device)

    def full_evaluation(self, model: nn.Module,
                        device: str = 'cpu',
                        model_name: str = "") -> UnifiedBenchmarkResult:
        """
        Run all benchmarks and return unified scores.

        Args:
            model: Neural network model to evaluate
            device: Device to run evaluation on
            model_name: Optional name for logging

        Returns:
            UnifiedBenchmarkResult with all metrics and aggregate scores
        """
        # Run all evaluations
        tomi_results = self.evaluate_tomi(model, num_samples=100, device=device)
        social_iqa_results = self.evaluate_social_iqa(model, num_samples=100, device=device)
        social_games_results = self.evaluate_social_games(model, device=device)

        # Compute aggregate ToM score
        # Weight by importance: false belief is core ToM, others are supporting
        tom_aggregate = (
            tomi_results['tom_accuracy'] * self.weights['tomi'] +
            social_iqa_results['accuracy'] * self.weights['social_iqa'] +
            social_games_results.prediction_accuracy * self.weights['social_games']
        )

        # Control score (tasks that don't require ToM)
        control_aggregate = (
            tomi_results['control_accuracy'] * self.weights['tomi'] +
            social_iqa_results['subsequent_accuracy'] * self.weights['social_iqa'] +
            social_games_results.cooperation_rate * self.weights['social_games']
        )

        # ToM specificity: how much better on ToM vs control
        tom_specificity = tom_aggregate - control_aggregate

        total_examples = (
            tomi_results.get('total_examples', 100) +
            social_iqa_results.get('num_examples', 100) +
            social_games_results.num_episodes
        )

        return UnifiedBenchmarkResult(
            # ToMi
            tomi_tom_accuracy=tomi_results['tom_accuracy'],
            tomi_control_accuracy=tomi_results['control_accuracy'],
            tomi_specificity=tomi_results['specificity'],
            tomi_first_order=tomi_results['first_order_accuracy'],
            tomi_second_order=tomi_results['second_order_accuracy'],

            # SocialIQA
            social_iqa_accuracy=social_iqa_results['accuracy'],
            social_iqa_intent=social_iqa_results['intent_accuracy'],
            social_iqa_emotion=social_iqa_results['emotion_accuracy'],
            social_iqa_motivation=social_iqa_results['motivation_accuracy'],

            # Social Games
            social_games_cooperation=social_games_results.cooperation_rate,
            social_games_prediction=social_games_results.prediction_accuracy,
            social_games_zombie_detection=social_games_results.zombie_detection_accuracy,
            social_games_deception_detection=social_games_results.deception_detection_accuracy,

            # Aggregates
            tom_aggregate=tom_aggregate,
            control_aggregate=control_aggregate,
            tom_specificity=tom_specificity,

            # Metadata
            total_examples=total_examples,
            model_name=model_name,
        )

    def quick_evaluation(self, model: nn.Module,
                         device: str = 'cpu') -> Dict[str, float]:
        """
        Fast evaluation using fewer samples.

        Good for NAS fitness evaluation where speed matters.
        """
        tomi_results = self.evaluate_tomi(model, num_samples=20, device=device)

        # Skip full social games for speed
        tom_score = tomi_results['tom_accuracy']
        control_score = tomi_results['control_accuracy']

        return {
            'tom_score': tom_score,
            'control_score': control_score,
            'specificity': tom_score - control_score,
            'first_order': tomi_results['first_order_accuracy'],
        }

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of available benchmarks."""
        return {
            'tomi': {
                'name': 'Theory of Mind Inventory',
                'examples': len(self.tomi),
                'question_types': ['first_order', 'second_order', 'reality', 'memory'],
                'requires_tom': True,
            },
            'social_iqa': {
                'name': 'Social Intelligence QA',
                'examples': len(self.social_iqa),
                'question_types': ['intent', 'emotion', 'motivation', 'subsequent', 'prerequisite'],
                'requires_tom': True,
            },
            'social_games': {
                'name': 'Interactive Social Games',
                'game_types': ['cooperation', 'zombie_detection', 'deception', 'fairness'],
                'num_agents': self.social_games.num_agents,
                'requires_tom': True,
            },
            'weights': self.weights,
        }


# Convenience function for quick evaluation
def evaluate_tom_model(model: nn.Module,
                       device: str = 'cpu',
                       quick: bool = False) -> Dict[str, float]:
    """
    Convenience function to evaluate a model on ToM benchmarks.

    Args:
        model: Model to evaluate
        device: Device to run on
        quick: If True, run fast evaluation with fewer samples

    Returns:
        Dictionary of evaluation metrics
    """
    benchmark = UnifiedBenchmark()

    if quick:
        return benchmark.quick_evaluation(model, device)
    else:
        results = benchmark.full_evaluation(model, device)
        return {
            'tom_aggregate': results.tom_aggregate,
            'control_aggregate': results.control_aggregate,
            'tom_specificity': results.tom_specificity,
            'tomi_accuracy': results.tomi_tom_accuracy,
            'social_iqa_accuracy': results.social_iqa_accuracy,
            'social_games_prediction': results.social_games_prediction,
        }


# Export all
__all__ = [
    # ToMi
    'ToMiDataset',
    'ToMiParser',
    'ToMiEvaluator',
    'ToMiExample',
    'ToMiQuestion',
    'EventEncoder',

    # Social Games
    'SocialGameBenchmark',
    'SocialGameResult',
    'CooperationMetrics',
    'DeceptionMetrics',

    # SocialIQA
    'SocialIQADataset',
    'SocialIQAExample',
    'SocialIQAEncoder',
    'SocialIQAEvaluator',
    'SocialIQAMetrics',

    # Unified
    'UnifiedBenchmark',
    'UnifiedBenchmarkResult',
    'evaluate_tom_model',
]
