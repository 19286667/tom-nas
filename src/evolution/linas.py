"""
LINAS: Lightweight Iterative Neural Architecture Search

This module implements predictor-guided architecture search that combines:
1. Supernet weight sharing for efficient evaluation
2. Learned predictors that estimate fitness from architecture features
3. Iterative refinement to focus search on promising candidates

The key insight is that after evaluating some architectures, we can train
a predictor to estimate fitness, then use that predictor to filter candidates
before expensive full evaluation.

DyNAS-T (Intel Labs) achieves ~4x greater sample efficiency than methods
that evaluate every candidate fully.
"""

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ArchitectureFeatures:
    """
    Feature representation of an architecture for the predictor.

    Encodes architectural choices as a fixed-size vector that can be
    used as input to the fitness predictor.
    """

    arch_type: str  # 'trn', 'rsan', 'transformer'
    num_layers: int
    hidden_dim: int
    num_heads: int
    dropout: float
    use_skip_connections: bool

    # Computed features
    param_estimate: int = 0
    depth_ratio: float = 0.0  # num_layers / max_layers
    width_ratio: float = 0.0  # hidden_dim / max_hidden

    def to_vector(self, max_layers: int = 5, max_hidden: int = 256, max_heads: int = 8) -> torch.Tensor:
        """
        Convert to feature vector for predictor input.

        Returns a 15-dimensional vector:
        - [0-2]: One-hot architecture type
        - [3]: Normalized num_layers
        - [4]: Normalized hidden_dim
        - [5]: Normalized num_heads
        - [6]: Dropout rate
        - [7]: Skip connections (0 or 1)
        - [8]: Estimated parameters (log-normalized)
        - [9-14]: Architecture-specific interaction features
        """
        vec = torch.zeros(15)

        # One-hot arch type
        arch_idx = {"trn": 0, "rsan": 1, "transformer": 2}.get(self.arch_type.lower(), 0)
        vec[arch_idx] = 1.0

        # Normalized hyperparameters
        vec[3] = self.num_layers / max_layers
        vec[4] = self.hidden_dim / max_hidden
        vec[5] = self.num_heads / max_heads
        vec[6] = self.dropout
        vec[7] = float(self.use_skip_connections)

        # Parameter estimate (log-normalized)
        if self.param_estimate > 0:
            vec[8] = np.log10(self.param_estimate) / 7  # Normalize to ~0-1
        else:
            vec[8] = self._estimate_params() / 1e6

        # Interaction features
        vec[9] = vec[3] * vec[4]  # depth-width interaction
        vec[10] = vec[4] * vec[5]  # width-heads interaction
        vec[11] = vec[3] * vec[7]  # depth-skip interaction
        vec[12] = float(self.arch_type.lower() == "trn") * vec[3]  # TRN depth
        vec[13] = float(self.arch_type.lower() != "trn") * vec[5]  # Attention heads
        vec[14] = vec[3] * vec[4] * vec[5]  # Three-way interaction

        return vec

    def _estimate_params(self) -> float:
        """Rough parameter count estimate."""
        hidden = self.hidden_dim
        layers = self.num_layers
        base = 181 * hidden + hidden  # Input projection

        if self.arch_type.lower() == "trn":
            base += layers * (3 * (hidden * 2) * hidden)
        else:
            base += layers * (4 * hidden * hidden + 8 * hidden * hidden)

        base += hidden * 181 + 181 + hidden + 1  # Output heads
        return base

    @classmethod
    def from_config(cls, config: Any) -> "ArchitectureFeatures":
        """Create from SubnetConfig or similar."""
        return cls(
            arch_type=config.arch_type,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            num_heads=getattr(config, "num_heads", 4),
            dropout=getattr(config, "dropout", 0.1),
            use_skip_connections=getattr(config, "use_skip_connections", True),
        )


class FitnessPredictor(nn.Module):
    """
    Neural network that predicts fitness from architecture features.

    This is a small MLP that learns to map architecture features to
    expected fitness scores based on evaluation data.
    """

    def __init__(self, input_dim: int = 15, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict fitness from feature vector."""
        return self.network(x).squeeze(-1)

    def predict_fitness(self, features: ArchitectureFeatures) -> float:
        """Convenience method to predict fitness for a single architecture."""
        self.eval()
        with torch.no_grad():
            vec = features.to_vector().unsqueeze(0)
            return self.forward(vec).item()


@dataclass
class EvaluationRecord:
    """Record of an architecture evaluation."""

    features: ArchitectureFeatures
    fitness: float
    accuracy: float
    param_count: int
    evaluation_method: str  # 'full', 'supernet', 'proxy'


class PredictorTrainer:
    """
    Trains and updates the fitness predictor.
    """

    def __init__(self, predictor: FitnessPredictor, lr: float = 1e-3):
        self.predictor = predictor
        self.optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
        self.records: List[EvaluationRecord] = []
        self.training_losses: List[float] = []

    def add_record(self, record: EvaluationRecord):
        """Add an evaluation record to the training data."""
        self.records.append(record)

    def add_records(self, records: List[EvaluationRecord]):
        """Add multiple evaluation records."""
        self.records.extend(records)

    def train(self, epochs: int = 50, batch_size: int = 32) -> float:
        """
        Train the predictor on accumulated records.

        Returns:
            Final training loss
        """
        if len(self.records) < 10:
            return float("inf")  # Not enough data

        self.predictor.train()

        # Prepare data
        X = torch.stack([r.features.to_vector() for r in self.records])
        y = torch.tensor([r.fitness for r in self.records])

        # Normalize targets
        y_mean = y.mean()
        y_std = y.std() + 1e-6
        y_norm = (y - y_mean) / y_std

        for epoch in range(epochs):
            # Shuffle
            indices = torch.randperm(len(self.records))
            X = X[indices]
            y_norm = y_norm[indices]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(self.records), batch_size):
                batch_X = X[i : i + batch_size]
                batch_y = y_norm[i : i + batch_size]

                self.optimizer.zero_grad()
                pred = self.predictor(batch_X)
                loss = F.mse_loss(pred, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.training_losses.append(avg_loss)

        return self.training_losses[-1] if self.training_losses else float("inf")

    def get_prediction_stats(self) -> Dict[str, float]:
        """Get statistics about predictor performance."""
        if len(self.records) < 10:
            return {"correlation": 0.0, "mse": float("inf")}

        self.predictor.eval()
        with torch.no_grad():
            X = torch.stack([r.features.to_vector() for r in self.records])
            y_true = torch.tensor([r.fitness for r in self.records])
            y_pred = self.predictor(X)

        # Correlation
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()

        if y_true_np.std() > 0 and y_pred_np.std() > 0:
            correlation = np.corrcoef(y_true_np, y_pred_np)[0, 1]
        else:
            correlation = 0.0

        mse = F.mse_loss(y_pred, y_true).item()

        return {"correlation": correlation, "mse": mse, "num_records": len(self.records)}


class LINASSearch:
    """
    LINAS: Lightweight Iterative Neural Architecture Search.

    Combines supernet evaluation with predictor-guided search:
    1. Train supernet once
    2. Evaluate sample architectures to build predictor
    3. Use predictor to score candidates
    4. Evaluate top candidates more thoroughly
    5. Update predictor and repeat
    """

    def __init__(
        self, supernet, supernet_evaluator, device: str = "cpu"  # ToMSupernet instance  # SupernetEvaluator instance
    ):
        self.supernet = supernet
        self.evaluator = supernet_evaluator
        self.device = device

        # Predictor and trainer
        self.predictor = FitnessPredictor()
        self.trainer = PredictorTrainer(self.predictor)

        # Search state
        self.iteration = 0
        self.best_architectures: List[Tuple[ArchitectureFeatures, float]] = []
        self.all_evaluations: List[EvaluationRecord] = []

    def generate_candidates(
        self, num_candidates: int, arch_types: Optional[List[str]] = None
    ) -> List[ArchitectureFeatures]:
        """Generate random candidate architectures."""
        if arch_types is None:
            arch_types = ["trn", "rsan", "transformer"]

        candidates = []
        for _ in range(num_candidates):
            features = ArchitectureFeatures(
                arch_type=random.choice(arch_types),
                num_layers=random.randint(1, 5),
                hidden_dim=random.choice([64, 96, 128, 160, 192, 224, 256]),
                num_heads=random.choice([2, 4, 6, 8]),
                dropout=random.choice([0.0, 0.1, 0.2]),
                use_skip_connections=random.random() > 0.3,
            )
            features.param_estimate = int(features._estimate_params())
            candidates.append(features)

        return candidates

    def evaluate_architecture(
        self,
        features: ArchitectureFeatures,
        eval_data: List[Tuple[torch.Tensor, torch.Tensor]],
        fine_tune_epochs: int = 2,
        method: str = "supernet",
    ) -> EvaluationRecord:
        """
        Evaluate a single architecture.

        Args:
            features: Architecture features
            eval_data: Evaluation data
            fine_tune_epochs: Epochs for fine-tuning
            method: Evaluation method label

        Returns:
            EvaluationRecord with results
        """
        from .supernet import SubnetConfig

        config = SubnetConfig(
            arch_type=features.arch_type,
            num_layers=features.num_layers,
            hidden_dim=features.hidden_dim,
            num_heads=features.num_heads,
            dropout=features.dropout,
            use_skip_connections=features.use_skip_connections,
        )

        results = self.evaluator.evaluate_config(config, eval_data, fine_tune_epochs)

        record = EvaluationRecord(
            features=features,
            fitness=results["accuracy"],  # Use accuracy as fitness
            accuracy=results["accuracy"],
            param_count=results["param_count"],
            evaluation_method=method,
        )

        return record

    def predictor_score_candidates(
        self, candidates: List[ArchitectureFeatures]
    ) -> List[Tuple[ArchitectureFeatures, float]]:
        """Score candidates using the predictor."""
        self.predictor.eval()
        scored = []

        with torch.no_grad():
            for features in candidates:
                vec = features.to_vector().unsqueeze(0)
                score = self.predictor(vec).item()
                scored.append((features, score))

        # Sort by predicted fitness (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def search_iteration(
        self,
        eval_data: List[Tuple[torch.Tensor, torch.Tensor]],
        num_candidates: int = 100,
        num_to_evaluate: int = 10,
        fine_tune_epochs: int = 2,
    ) -> Dict[str, Any]:
        """
        Perform one iteration of LINAS search.

        Args:
            eval_data: Evaluation data
            num_candidates: Candidates to generate
            num_to_evaluate: Candidates to fully evaluate
            fine_tune_epochs: Fine-tuning epochs per evaluation

        Returns:
            Iteration results
        """
        self.iteration += 1

        # Generate candidates
        candidates = self.generate_candidates(num_candidates)

        # Score with predictor (if trained)
        if len(self.trainer.records) >= 10:
            # Train predictor on accumulated data
            self.trainer.train(epochs=30)

            # Score and rank candidates
            scored = self.predictor_score_candidates(candidates)
            top_candidates = [f for f, s in scored[:num_to_evaluate]]
        else:
            # Not enough data, use random selection
            top_candidates = random.sample(candidates, min(num_to_evaluate, len(candidates)))

        # Evaluate top candidates
        new_records = []
        for features in top_candidates:
            record = self.evaluate_architecture(features, eval_data, fine_tune_epochs, method="linas")
            new_records.append(record)
            self.all_evaluations.append(record)
            self.trainer.add_record(record)

        # Update best architectures
        for record in new_records:
            self.best_architectures.append((record.features, record.fitness))

        # Sort and keep top 20
        self.best_architectures.sort(key=lambda x: x[1], reverse=True)
        self.best_architectures = self.best_architectures[:20]

        return {
            "iteration": self.iteration,
            "num_evaluated": len(new_records),
            "best_fitness": self.best_architectures[0][1] if self.best_architectures else 0,
            "avg_fitness": np.mean([r.fitness for r in new_records]),
            "predictor_stats": self.trainer.get_prediction_stats(),
        }

    def run_search(
        self,
        eval_data: List[Tuple[torch.Tensor, torch.Tensor]],
        num_iterations: int = 10,
        candidates_per_iteration: int = 100,
        evaluations_per_iteration: int = 10,
    ) -> Dict[str, Any]:
        """
        Run complete LINAS search.

        Args:
            eval_data: Evaluation data
            num_iterations: Number of search iterations
            candidates_per_iteration: Candidates generated per iteration
            evaluations_per_iteration: Full evaluations per iteration

        Returns:
            Search results including best architectures
        """
        results = []

        for i in range(num_iterations):
            iter_result = self.search_iteration(
                eval_data, num_candidates=candidates_per_iteration, num_to_evaluate=evaluations_per_iteration
            )
            results.append(iter_result)

            print(
                f"Iteration {i+1}/{num_iterations}: "
                f"Best={iter_result['best_fitness']:.4f}, "
                f"Avg={iter_result['avg_fitness']:.4f}"
            )

        return {
            "best_architectures": self.best_architectures[:10],
            "total_evaluations": len(self.all_evaluations),
            "final_predictor_stats": self.trainer.get_prediction_stats(),
            "iteration_results": results,
        }


class EfficientNASPipeline:
    """
    Complete efficient NAS pipeline combining all strategies.

    Stage 1: Zero-cost proxy filtering
    Stage 2: Supernet training
    Stage 3: LINAS predictor-guided search
    Stage 4: Final validation
    """

    def __init__(self, input_dim: int = 181, output_dim: int = 181, device: str = "cpu", param_budget: int = 500000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.param_budget = param_budget

        # Components (initialized lazily)
        self.proxy_evaluator = None
        self.architecture_filter = None
        self.supernet = None
        self.supernet_trainer = None
        self.supernet_evaluator = None
        self.linas_search = None

    def stage1_proxy_filter(self, num_candidates: int = 5000, top_k: int = 500) -> List[ArchitectureFeatures]:
        """
        Stage 1: Generate candidates and filter with zero-cost proxies.
        """
        print("Stage 1: Zero-cost proxy filtering")
        from ..agents.architectures import RecursiveSelfAttention, TransformerToMAgent, TransparentRNN
        from .zero_cost_proxies import ArchitectureFilter, ZeroCostProxy

        self.proxy_evaluator = ZeroCostProxy(
            input_dim=self.input_dim, device=self.device, target_params=self.param_budget
        )

        # Generate candidate architectures
        candidates = []
        features_list = []

        for _ in range(num_candidates):
            arch_type = random.choice(["trn", "rsan", "transformer"])
            num_layers = random.randint(1, 5)
            hidden_dim = random.choice([64, 96, 128, 160, 192, 224, 256])
            num_heads = random.choice([2, 4, 6, 8])

            features = ArchitectureFeatures(
                arch_type=arch_type,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=random.choice([0.0, 0.1, 0.2]),
                use_skip_connections=random.random() > 0.3,
            )

            # Create actual model for proxy evaluation
            if arch_type == "trn":
                model = TransparentRNN(self.input_dim, hidden_dim, self.output_dim, num_layers)
            elif arch_type == "rsan":
                model = RecursiveSelfAttention(self.input_dim, hidden_dim, self.output_dim, num_heads, num_layers)
            else:
                model = TransformerToMAgent(self.input_dim, hidden_dim, self.output_dim, num_layers, num_heads)

            candidates.append(model)
            features_list.append(features)

        # Filter
        self.architecture_filter = ArchitectureFilter(self.proxy_evaluator, param_budget=self.param_budget)

        filtered = self.architecture_filter.filter_architectures(candidates, top_k=top_k)

        # Return features for filtered architectures
        filtered_features = [features_list[idx] for idx, _, _ in filtered]

        print(f"  Generated {num_candidates} candidates, kept {len(filtered_features)} after filtering")
        stats = self.architecture_filter.get_statistics()
        print(f"  Avg evaluation time: {stats['avg_evaluation_time_ms']:.1f}ms")

        return filtered_features

    def stage2_train_supernet(self, train_data, num_epochs: int = 20):  # DataLoader or list
        """
        Stage 2: Train the supernet.
        """
        print("Stage 2: Supernet training")
        from .supernet import SupernetEvaluator, SupernetTrainer, ToMSupernet

        self.supernet = ToMSupernet(input_dim=self.input_dim, output_dim=self.output_dim).to(self.device)

        self.supernet_trainer = SupernetTrainer(self.supernet, device=self.device)

        # Train
        if hasattr(train_data, "__iter__"):
            result = self.supernet_trainer.progressive_shrinking(train_data, num_epochs=num_epochs, phases=4)
        else:
            print("  Warning: train_data should be iterable")
            result = {"final_loss": 0.0}

        self.supernet_evaluator = SupernetEvaluator(self.supernet, device=self.device)

        print(f"  Final loss: {result['final_loss']:.4f}")

    def stage3_linas_search(
        self, eval_data: List[Tuple[torch.Tensor, torch.Tensor]], num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Stage 3: LINAS predictor-guided search.
        """
        print("Stage 3: LINAS search")

        self.linas_search = LINASSearch(self.supernet, self.supernet_evaluator, device=self.device)

        results = self.linas_search.run_search(
            eval_data, num_iterations=num_iterations, candidates_per_iteration=100, evaluations_per_iteration=10
        )

        print(f"  Total evaluations: {results['total_evaluations']}")
        print(f"  Best accuracy: {results['best_architectures'][0][1]:.4f}")

        return results

    def stage4_validate(
        self, train_fn: Callable, top_k: int = 5  # Function(features) -> trained_accuracy
    ) -> List[Tuple[ArchitectureFeatures, float]]:
        """
        Stage 4: Validate top architectures with full training.
        """
        print("Stage 4: Final validation")

        if not self.linas_search or not self.linas_search.best_architectures:
            print("  No architectures to validate")
            return []

        top_architectures = self.linas_search.best_architectures[:top_k]
        validated = []

        for features, linas_score in top_architectures:
            print(
                f"  Validating {features.arch_type} " f"(layers={features.num_layers}, hidden={features.hidden_dim})..."
            )

            final_accuracy = train_fn(features)
            validated.append((features, final_accuracy))

            print(f"    LINAS score: {linas_score:.4f}, Final accuracy: {final_accuracy:.4f}")

        # Sort by final accuracy
        validated.sort(key=lambda x: x[1], reverse=True)

        return validated


def test_linas():
    """Test LINAS search."""
    print("=" * 60)
    print("LINAS SEARCH TEST")
    print("=" * 60)

    # Create dummy evaluation data
    eval_data = [(torch.randn(8, 10, 181), torch.rand(8, 181)) for _ in range(5)]

    # Create supernet and evaluator
    from .supernet import SupernetEvaluator, ToMSupernet

    supernet = ToMSupernet(181, 181)
    evaluator = SupernetEvaluator(supernet)

    # Run LINAS search
    search = LINASSearch(supernet, evaluator)

    results = search.run_search(eval_data, num_iterations=3, candidates_per_iteration=20, evaluations_per_iteration=5)

    print("\nFinal results:")
    print(f"  Total evaluations: {results['total_evaluations']}")
    print("  Best architectures:")
    for features, fitness in results["best_architectures"][:3]:
        print(
            f"    {features.arch_type}: layers={features.num_layers}, "
            f"hidden={features.hidden_dim}, fitness={fitness:.4f}"
        )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_linas()
