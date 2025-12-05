"""
Theory of Mind Specific Fitness Function for Neural Architecture Search

This module implements fitness evaluation specifically designed for Theory of Mind
capabilities, incorporating:
- ToM accuracy on belief questions
- Control accuracy on reality questions
- ToM specificity (ToM accuracy - control accuracy)
- Efficiency penalty for parameter count

The key insight is that high ToM accuracy alone is insufficient - we need to verify
that models are actually tracking beliefs rather than exploiting shortcuts.
ToM specificity measures whether the model performs better on belief questions
than on reality questions, indicating genuine mental state reasoning.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..benchmarks.tomi_loader import ToMiDataset, ToMiEvaluator


@dataclass
class ToMFitnessResult:
    """Results from ToM fitness evaluation."""

    total_fitness: float
    tom_accuracy: float
    control_accuracy: float
    tom_specificity: float
    first_order_accuracy: float
    second_order_accuracy: float
    reality_accuracy: float
    efficiency_score: float
    param_count: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_fitness": self.total_fitness,
            "tom_accuracy": self.tom_accuracy,
            "control_accuracy": self.control_accuracy,
            "tom_specificity": self.tom_specificity,
            "first_order_accuracy": self.first_order_accuracy,
            "second_order_accuracy": self.second_order_accuracy,
            "reality_accuracy": self.reality_accuracy,
            "efficiency_score": self.efficiency_score,
            "param_count": self.param_count,
        }


class ToMSpecificFitness:
    """
    Fitness function specifically designed for Theory of Mind evaluation.

    The fitness function combines:
    - tom_accuracy: Performance on first and second order belief questions
    - control_accuracy: Performance on reality and memory questions
    - tom_specificity: tom_accuracy - control_accuracy (reward genuine ToM)
    - efficiency: Penalty for excessive parameters

    Formula:
    fitness = w1 * tom_accuracy + w2 * control_accuracy + w3 * tom_specificity + w4 * efficiency

    Default weights:
    - w1 = 0.5 (primary focus on ToM capability)
    - w2 = 0.2 (maintain basic competence)
    - w3 = 0.2 (reward genuine ToM over shortcuts)
    - w4 = 0.1 (encourage efficient architectures)
    """

    def __init__(
        self,
        dataset: Optional[ToMiDataset] = None,
        target_params: int = 500000,
        device: str = "cpu",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the ToM fitness evaluator.

        Args:
            dataset: ToMi dataset for evaluation
            target_params: Target parameter count for efficiency calculation
            device: Device for model evaluation
            weights: Optional custom weights for fitness components
        """
        self.device = device
        self.target_params = target_params

        # Create or use provided dataset
        if dataset is None:
            self.dataset = ToMiDataset()
            self.dataset.generate_synthetic(num_examples=500)
            self.dataset.split()
        else:
            self.dataset = dataset

        self.evaluator = ToMiEvaluator(self.dataset)

        # Fitness weights
        self.weights = weights or {
            "tom_accuracy": 0.5,
            "control_accuracy": 0.2,
            "tom_specificity": 0.2,
            "efficiency": 0.1,
        }

    def evaluate(self, model: nn.Module, num_examples: Optional[int] = None, split: str = "val") -> ToMFitnessResult:
        """
        Evaluate a model's Theory of Mind fitness.

        Args:
            model: Neural network model to evaluate
            num_examples: Optional limit on examples
            split: Dataset split to use ('train', 'val', 'test')

        Returns:
            ToMFitnessResult with all fitness components
        """
        model = model.to(self.device)

        # Get ToMi accuracies
        accuracies = self.evaluator.evaluate(model, split=split, num_examples=num_examples)

        # Extract components
        first_order = accuracies.get("first_order_accuracy", 0.0)
        second_order = accuracies.get("second_order_accuracy", 0.0)
        reality = accuracies.get("reality_accuracy", 0.0)
        memory = accuracies.get("memory_accuracy", 0.0)

        # Compute aggregate metrics
        tom_accuracy = accuracies.get("tom_accuracy", (first_order + second_order) / 2)
        control_accuracy = accuracies.get("control_accuracy", (reality + memory) / 2)
        tom_specificity = accuracies.get("tom_specificity", tom_accuracy - control_accuracy)

        # Compute efficiency
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        efficiency_score = self._compute_efficiency(param_count)

        # Compute total fitness
        total_fitness = (
            self.weights["tom_accuracy"] * tom_accuracy
            + self.weights["control_accuracy"] * control_accuracy
            + self.weights["tom_specificity"] * max(tom_specificity, 0)  # Don't penalize negative
            + self.weights["efficiency"] * efficiency_score
        )

        return ToMFitnessResult(
            total_fitness=total_fitness,
            tom_accuracy=tom_accuracy,
            control_accuracy=control_accuracy,
            tom_specificity=tom_specificity,
            first_order_accuracy=first_order,
            second_order_accuracy=second_order,
            reality_accuracy=reality,
            efficiency_score=efficiency_score,
            param_count=param_count,
        )

    def _compute_efficiency(self, param_count: int) -> float:
        """
        Compute efficiency score based on parameter count.

        Returns 1.0 for models at or under target, decreasing for larger models.
        """
        if param_count <= self.target_params:
            return 1.0

        # Logarithmic penalty for exceeding target
        ratio = param_count / self.target_params
        penalty = np.log2(ratio)  # 2x params -> 0 penalty, 4x -> -1, etc.

        return max(0.0, 1.0 - 0.5 * penalty)


class AdversarialToMFitness:
    """
    Fitness evaluation with adversarial variants to detect shortcut exploitation.

    Tests whether models exploit spurious correlations by:
    1. Shuffling location order
    2. Using novel agent names
    3. Adding irrelevant events
    4. Modifying sentence structure
    """

    def __init__(self, base_fitness: ToMSpecificFitness, adversarial_weight: float = 0.3):
        self.base_fitness = base_fitness
        self.adversarial_weight = adversarial_weight

    def evaluate(self, model: nn.Module, num_examples: Optional[int] = 100) -> Dict[str, Any]:
        """
        Evaluate with both standard and adversarial examples.
        """
        # Standard evaluation
        standard_result = self.base_fitness.evaluate(model, num_examples)

        # Generate and evaluate adversarial variants
        adversarial_accuracy = self._evaluate_adversarial(model, num_examples)

        # Combine scores
        combined_fitness = (
            1 - self.adversarial_weight
        ) * standard_result.total_fitness + self.adversarial_weight * adversarial_accuracy

        return {
            "total_fitness": combined_fitness,
            "standard_fitness": standard_result.total_fitness,
            "adversarial_accuracy": adversarial_accuracy,
            "standard_result": standard_result.to_dict(),
            "robustness_gap": standard_result.tom_accuracy - adversarial_accuracy,
        }

    def _evaluate_adversarial(self, model: nn.Module, num_examples: int) -> float:
        """Evaluate on adversarial variants."""
        # For now, use standard evaluation with added noise
        # Full implementation would generate adversarial scenarios

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(min(num_examples, len(self.base_fitness.dataset.examples))):
                # Get random example
                example = np.random.choice(self.base_fitness.dataset.examples)

                for q_idx, question in enumerate(example.questions):
                    if question.question_type not in ["first_order", "second_order"]:
                        continue

                    inp, tgt, correct_idx = self.base_fitness.dataset.encode_example(example, q_idx)

                    # Add noise as simple adversarial perturbation
                    inp = inp + 0.1 * torch.randn_like(inp)

                    output = model(inp.unsqueeze(0).to(self.base_fitness.device))

                    if isinstance(output, dict):
                        beliefs = output.get("beliefs")
                        if beliefs is not None:
                            pred_idx = beliefs.argmax(dim=-1).item()
                            if pred_idx == correct_idx:
                                correct += 1

                    total += 1

        return correct / total if total > 0 else 0.0


class EvolutionaryFitnessCache:
    """
    Caches fitness evaluations to avoid redundant computation.

    Uses architecture hash to identify previously evaluated configurations.
    """

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, ToMFitnessResult] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_model(self, model: nn.Module) -> str:
        """Create hash from model architecture."""
        # Simple hash based on parameter shapes
        shapes = []
        for name, param in model.named_parameters():
            shapes.append(f"{name}:{tuple(param.shape)}")
        return hash(tuple(shapes)).__str__()

    def get(self, model: nn.Module) -> Optional[ToMFitnessResult]:
        """Get cached fitness if available."""
        key = self._hash_model(model)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, model: nn.Module, result: ToMFitnessResult):
        """Cache a fitness result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entries (simple LRU approximation)
            to_remove = list(self.cache.keys())[: len(self.cache) // 2]
            for key in to_remove:
                del self.cache[key]

        key = self._hash_model(model)
        self.cache[key] = result

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }


class EarlyTerminationFitness:
    """
    Fitness evaluation with early termination for clearly bad architectures.

    If accuracy on a small sample is below threshold, skip full evaluation.
    This significantly speeds up evolutionary search.
    """

    def __init__(
        self,
        base_fitness: ToMSpecificFitness,
        sample_size: int = 50,
        threshold: float = 0.25,  # Must be better than random (1/4 for 4 locations)
    ):
        self.base_fitness = base_fitness
        self.sample_size = sample_size
        self.threshold = threshold

        self.early_terminated = 0
        self.fully_evaluated = 0

    def evaluate(self, model: nn.Module) -> Tuple[ToMFitnessResult, bool]:
        """
        Evaluate with potential early termination.

        Returns:
            (result, was_early_terminated)
        """
        # Quick evaluation on small sample
        quick_result = self.base_fitness.evaluate(model, num_examples=self.sample_size, split="train")

        # Check if worth full evaluation
        if quick_result.tom_accuracy < self.threshold:
            self.early_terminated += 1
            # Return pessimistic estimate
            return quick_result, True

        # Full evaluation
        self.fully_evaluated += 1
        full_result = self.base_fitness.evaluate(model, split="val")

        return full_result, False

    def get_stats(self) -> Dict[str, Any]:
        """Get early termination statistics."""
        total = self.early_terminated + self.fully_evaluated
        return {
            "early_terminated": self.early_terminated,
            "fully_evaluated": self.fully_evaluated,
            "early_termination_rate": self.early_terminated / total if total > 0 else 0.0,
        }


class CombinedToMFitness:
    """
    Combined fitness function integrating all components.

    Uses caching, early termination, and ToM-specific evaluation.
    """

    def __init__(self, device: str = "cpu", target_params: int = 500000):
        # Initialize components
        self.dataset = ToMiDataset()
        self.dataset.generate_synthetic(num_examples=1000)
        self.dataset.split()

        self.base_fitness = ToMSpecificFitness(dataset=self.dataset, target_params=target_params, device=device)

        self.early_termination = EarlyTerminationFitness(self.base_fitness)
        self.cache = EvolutionaryFitnessCache()

        self.device = device

    def evaluate(self, model: nn.Module) -> ToMFitnessResult:
        """
        Evaluate model fitness with caching and early termination.
        """
        # Check cache
        cached = self.cache.get(model)
        if cached is not None:
            return cached

        # Evaluate with early termination
        result, early_terminated = self.early_termination.evaluate(model)

        # Cache result
        self.cache.put(model, result)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {"cache": self.cache.get_stats(), "early_termination": self.early_termination.get_stats()}


def test_tom_fitness():
    """Test ToM fitness evaluation."""
    print("=" * 60)
    print("TOM FITNESS TEST")
    print("=" * 60)

    # Create simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(181, 128), nn.ReLU(), nn.Linear(128, 181))

        def forward(self, x):
            if x.dim() == 3:
                x = x[:, -1, :]  # Take last timestep
            out = torch.sigmoid(self.net(x))
            return {"beliefs": out, "actions": out.mean(dim=-1)}

    model = SimpleModel()

    # Create fitness evaluator
    fitness = CombinedToMFitness()

    print("\nEvaluating model...")
    result = fitness.evaluate(model)

    print(f"\n--- Results ---")
    print(f"Total Fitness: {result.total_fitness:.4f}")
    print(f"ToM Accuracy: {result.tom_accuracy:.4f}")
    print(f"Control Accuracy: {result.control_accuracy:.4f}")
    print(f"ToM Specificity: {result.tom_specificity:.4f}")
    print(f"First-order: {result.first_order_accuracy:.4f}")
    print(f"Second-order: {result.second_order_accuracy:.4f}")
    print(f"Reality: {result.reality_accuracy:.4f}")
    print(f"Efficiency: {result.efficiency_score:.4f}")
    print(f"Parameters: {result.param_count:,}")

    # Test caching
    print("\n--- Testing Cache ---")
    result2 = fitness.evaluate(model)  # Should hit cache
    stats = fitness.get_stats()
    print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
    print(f"Early termination rate: {stats['early_termination']['early_termination_rate']:.2%}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_tom_fitness()
