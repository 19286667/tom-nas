"""
Reinforced Mutation Controller for Neural Architecture Search

This module implements a learned mutation controller inspired by the RENAS paper.
The controller learns which mutations are likely to be beneficial based on the
history of mutations and their outcomes during evolution.

Key features:
- Observes parent architecture and proposed mutation
- Predicts fitness improvement
- Guides mutation selection toward beneficial changes
- Learns architecture-specific preferences

The controller provides insights about what architectural features matter
for Theory of Mind capability.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MutationRecord:
    """Record of a mutation and its outcome."""

    parent_encoding: torch.Tensor
    mutation_encoding: torch.Tensor
    fitness_improvement: float
    parent_fitness: float
    child_fitness: float


class ArchitectureEncoder(nn.Module):
    """
    Encodes architecture configuration into a fixed-size vector.

    The encoding captures:
    - Architecture type (one-hot)
    - Continuous hyperparameters (normalized)
    - Type-specific features
    """

    ARCH_TYPES = ["trn", "rsan", "transformer"]
    CELL_TYPES = ["gru", "lstm", "srn"]

    OUTPUT_DIM = 25

    def __init__(self):
        super().__init__()
        # Learnable embeddings for discrete features
        self.arch_type_embed = nn.Embedding(len(self.ARCH_TYPES), 4)
        self.cell_type_embed = nn.Embedding(len(self.CELL_TYPES), 3)

    def encode(self, config: Dict[str, Any]) -> torch.Tensor:
        """
        Encode an architecture configuration.

        Args:
            config: Dictionary with architecture parameters
                - arch_type: 'trn', 'rsan', or 'transformer'
                - num_layers: int
                - hidden_dim: int
                - num_heads: int (for attention)
                - dropout: float
                - use_skip_connections: bool
                - cell_type: str (for TRN)

        Returns:
            Tensor of shape (OUTPUT_DIM,)
        """
        vec = torch.zeros(self.OUTPUT_DIM)

        # Architecture type (one-hot) - dims 0-2
        arch_type = config.get("arch_type", "transformer").lower()
        arch_idx = self.ARCH_TYPES.index(arch_type) if arch_type in self.ARCH_TYPES else 0
        vec[arch_idx] = 1.0

        # Normalized hyperparameters - dims 3-10
        vec[3] = config.get("num_layers", 2) / 5.0  # Max 5 layers
        vec[4] = config.get("hidden_dim", 128) / 256.0  # Max 256
        vec[5] = config.get("num_heads", 4) / 8.0  # Max 8 heads
        vec[6] = config.get("dropout", 0.1)
        vec[7] = float(config.get("use_skip_connections", True))
        vec[8] = config.get("recursion_depth", 3) / 5.0  # For RSAN
        vec[9] = config.get("learning_rate", 0.001) * 1000  # Normalize
        vec[10] = np.log10(config.get("param_count", 100000)) / 7.0  # Log-normalized

        # Cell type (for TRN) - dims 11-13
        cell_type = config.get("cell_type", "gru").lower()
        cell_idx = self.CELL_TYPES.index(cell_type) if cell_type in self.CELL_TYPES else 0
        vec[11 + cell_idx] = 1.0

        # Architecture-specific features - dims 14-19
        if arch_type == "trn":
            vec[14] = 1.0
            vec[15] = vec[3] * vec[4]  # depth-width interaction for TRN
        elif arch_type == "rsan":
            vec[16] = 1.0
            vec[17] = vec[8] * vec[5]  # recursion-heads interaction
        else:  # transformer
            vec[18] = 1.0
            vec[19] = vec[3] * vec[5]  # layers-heads interaction

        # Derived features - dims 20-24
        vec[20] = vec[3] * vec[4]  # General depth-width
        vec[21] = vec[4] * vec[5]  # width-heads
        vec[22] = vec[3] * vec[7]  # depth-skip
        vec[23] = (1 - vec[6]) * vec[4]  # anti-dropout * width
        vec[24] = vec[3] * vec[4] * vec[5]  # three-way

        return vec


class MutationEncoder(nn.Module):
    """
    Encodes a mutation specification.

    Mutations are characterized by:
    - Which parameter changed
    - Direction of change (increase/decrease)
    - Magnitude of change
    """

    PARAM_NAMES = [
        "num_layers",
        "hidden_dim",
        "num_heads",
        "dropout",
        "use_skip_connections",
        "cell_type",
        "recursion_depth",
        "learning_rate",
        "arch_type",
    ]

    OUTPUT_DIM = 15

    def encode(self, param_name: str, old_value: Any, new_value: Any) -> torch.Tensor:
        """
        Encode a mutation.

        Args:
            param_name: Name of the mutated parameter
            old_value: Value before mutation
            new_value: Value after mutation

        Returns:
            Tensor of shape (OUTPUT_DIM,)
        """
        vec = torch.zeros(self.OUTPUT_DIM)

        # Parameter identity (one-hot) - dims 0-8
        param_idx = self.PARAM_NAMES.index(param_name) if param_name in self.PARAM_NAMES else 0
        vec[param_idx] = 1.0

        # Compute delta for numeric parameters
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            delta = new_value - old_value

            # Normalized delta - dim 9
            if param_name == "hidden_dim":
                vec[9] = delta / 256.0
            elif param_name == "num_layers":
                vec[9] = delta / 5.0
            elif param_name == "num_heads":
                vec[9] = delta / 8.0
            elif param_name == "dropout":
                vec[9] = delta
            elif param_name == "learning_rate":
                vec[9] = delta * 1000
            else:
                vec[9] = delta

            # Direction - dims 10-11
            vec[10] = float(delta > 0)  # Increase
            vec[11] = float(delta < 0)  # Decrease

            # Magnitude - dim 12
            vec[12] = abs(delta) / (abs(old_value) + 1e-6) if old_value != 0 else abs(delta)

        # Boolean toggle - dim 13
        if isinstance(old_value, bool) and isinstance(new_value, bool):
            vec[13] = float(old_value != new_value)

        # Categorical change - dim 14
        if isinstance(old_value, str) and isinstance(new_value, str):
            vec[14] = float(old_value != new_value)

        return vec


class MutationController(nn.Module):
    """
    Neural network that predicts fitness improvement from mutations.

    Takes architecture encoding and mutation encoding, outputs predicted
    fitness improvement.
    """

    def __init__(self, arch_dim: int = 25, mutation_dim: int = 15, hidden_dim: int = 64):
        super().__init__()
        self.arch_dim = arch_dim
        self.mutation_dim = mutation_dim

        combined_dim = arch_dim + mutation_dim

        self.network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        self.arch_encoder = ArchitectureEncoder()
        self.mutation_encoder = MutationEncoder()

    def forward(self, arch_encoding: torch.Tensor, mutation_encoding: torch.Tensor) -> torch.Tensor:
        """
        Predict fitness improvement.

        Args:
            arch_encoding: Architecture encoding (batch, arch_dim)
            mutation_encoding: Mutation encoding (batch, mutation_dim)

        Returns:
            Predicted fitness improvement (batch,)
        """
        combined = torch.cat([arch_encoding, mutation_encoding], dim=-1)
        return self.network(combined).squeeze(-1)

    def predict_improvement(self, config: Dict[str, Any], param_name: str, old_value: Any, new_value: Any) -> float:
        """
        Convenience method to predict improvement for a single mutation.
        """
        self.eval()
        with torch.no_grad():
            arch_enc = self.arch_encoder.encode(config).unsqueeze(0)
            mut_enc = self.mutation_encoder.encode(param_name, old_value, new_value).unsqueeze(0)
            return self.forward(arch_enc, mut_enc).item()


class MutationDataset:
    """
    Dataset of mutation records for training the controller.
    """

    def __init__(self):
        self.records: List[MutationRecord] = []
        self.arch_encoder = ArchitectureEncoder()
        self.mutation_encoder = MutationEncoder()

    def add_example(
        self,
        parent_config: Dict[str, Any],
        param_name: str,
        old_value: Any,
        new_value: Any,
        parent_fitness: float,
        child_fitness: float,
    ):
        """Add a mutation example to the dataset."""
        arch_enc = self.arch_encoder.encode(parent_config)
        mut_enc = self.mutation_encoder.encode(param_name, old_value, new_value)
        improvement = child_fitness - parent_fitness

        record = MutationRecord(
            parent_encoding=arch_enc,
            mutation_encoding=mut_enc,
            fitness_improvement=improvement,
            parent_fitness=parent_fitness,
            child_fitness=child_fitness,
        )
        self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx) -> MutationRecord:
        return self.records[idx]

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all data as tensors."""
        arch_encs = torch.stack([r.parent_encoding for r in self.records])
        mut_encs = torch.stack([r.mutation_encoding for r in self.records])
        improvements = torch.tensor([r.fitness_improvement for r in self.records])
        return arch_encs, mut_encs, improvements


class ControllerTrainer:
    """
    Trainer for the mutation controller.
    """

    def __init__(self, controller: MutationController, lr: float = 1e-3):
        self.controller = controller
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
        self.dataset = MutationDataset()
        self.losses: List[float] = []

    def add_mutation_record(
        self,
        parent_config: Dict[str, Any],
        param_name: str,
        old_value: Any,
        new_value: Any,
        parent_fitness: float,
        child_fitness: float,
    ):
        """Record a mutation and its outcome."""
        self.dataset.add_example(parent_config, param_name, old_value, new_value, parent_fitness, child_fitness)

    def train(self, epochs: int = 50, batch_size: int = 32) -> float:
        """
        Train the controller on accumulated data.

        Returns:
            Final training loss
        """
        if len(self.dataset) < 20:
            return float("inf")  # Not enough data

        self.controller.train()

        arch_encs, mut_encs, improvements = self.dataset.get_tensors()

        # Normalize targets
        imp_mean = improvements.mean()
        imp_std = improvements.std() + 1e-6
        imp_norm = (improvements - imp_mean) / imp_std

        for epoch in range(epochs):
            # Shuffle
            indices = torch.randperm(len(self.dataset))
            arch_encs = arch_encs[indices]
            mut_encs = mut_encs[indices]
            imp_norm = imp_norm[indices]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(self.dataset), batch_size):
                batch_arch = arch_encs[i : i + batch_size]
                batch_mut = mut_encs[i : i + batch_size]
                batch_imp = imp_norm[i : i + batch_size]

                self.optimizer.zero_grad()
                pred = self.controller(batch_arch, batch_mut)
                loss = F.mse_loss(pred, batch_imp)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.losses.append(avg_loss)

        return self.losses[-1] if self.losses else float("inf")


class GuidedMutator:
    """
    Applies guided mutations using the learned controller.
    """

    # Mutation options for each parameter
    MUTATION_OPTIONS = {
        "num_layers": [-2, -1, 1, 2],
        "hidden_dim": [-64, -32, 32, 64],
        "num_heads": [-2, -1, 1, 2],
        "dropout": [-0.1, -0.05, 0.05, 0.1],
        "use_skip_connections": ["toggle"],
        "recursion_depth": [-1, 1],
        "learning_rate": [-0.0005, -0.0001, 0.0001, 0.0005],
    }

    # Parameter bounds
    PARAM_BOUNDS = {
        "num_layers": (1, 5),
        "hidden_dim": (64, 256),
        "num_heads": (2, 8),
        "dropout": (0.0, 0.3),
        "recursion_depth": (1, 5),
        "learning_rate": (0.0001, 0.01),
    }

    def __init__(self, controller: MutationController, exploration_rate: float = 0.2):
        self.controller = controller
        self.exploration_rate = exploration_rate

    def generate_mutations(self, config: Dict[str, Any]) -> List[Tuple[str, Any, Any, float]]:
        """
        Generate candidate mutations with predicted improvements.

        Args:
            config: Current architecture configuration

        Returns:
            List of (param_name, old_value, new_value, predicted_improvement)
        """
        candidates = []

        for param_name, deltas in self.MUTATION_OPTIONS.items():
            if param_name not in config:
                continue

            old_value = config[param_name]

            for delta in deltas:
                # Compute new value
                if delta == "toggle" and isinstance(old_value, bool):
                    new_value = not old_value
                elif isinstance(old_value, (int, float)):
                    new_value = old_value + delta

                    # Apply bounds
                    if param_name in self.PARAM_BOUNDS:
                        lo, hi = self.PARAM_BOUNDS[param_name]
                        new_value = max(lo, min(hi, new_value))

                        if new_value == old_value:
                            continue
                else:
                    continue

                # Predict improvement
                pred = self.controller.predict_improvement(config, param_name, old_value, new_value)

                candidates.append((param_name, old_value, new_value, pred))

        return candidates

    def select_mutation(self, config: Dict[str, Any]) -> Optional[Tuple[str, Any, Any]]:
        """
        Select a mutation to apply.

        Uses the controller to rank mutations, then selects probabilistically
        with higher probability for higher predicted improvement.

        Args:
            config: Current architecture configuration

        Returns:
            (param_name, old_value, new_value) or None if no valid mutation
        """
        candidates = self.generate_mutations(config)

        if not candidates:
            return None

        # Exploration: random mutation
        if random.random() < self.exploration_rate:
            choice = random.choice(candidates)
            return (choice[0], choice[1], choice[2])

        # Exploitation: use controller predictions
        # Convert to probabilities using softmax
        predictions = [c[3] for c in candidates]
        max_pred = max(predictions)
        exp_preds = [np.exp(p - max_pred) for p in predictions]
        total = sum(exp_preds)
        probs = [e / total for e in exp_preds]

        # Sample
        idx = np.random.choice(len(candidates), p=probs)
        choice = candidates[idx]

        return (choice[0], choice[1], choice[2])

    def apply_mutation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a guided mutation to produce a new configuration.

        Args:
            config: Parent configuration

        Returns:
            Mutated child configuration
        """
        mutation = self.select_mutation(config)

        if mutation is None:
            return config.copy()

        param_name, old_value, new_value = mutation

        child = config.copy()
        child[param_name] = new_value

        return child


class ControllerAnalyzer:
    """
    Analyzes the learned controller to understand mutation preferences.
    """

    def __init__(self, controller: MutationController):
        self.controller = controller

    def analyze_parameter_preferences(
        self, arch_types: List[str] = None
    ) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
        """
        Analyze which parameters the controller prefers to mutate.

        Returns:
            Dictionary mapping arch_type -> param_name -> [(delta, predicted_improvement)]
        """
        if arch_types is None:
            arch_types = ["trn", "rsan", "transformer"]

        results = {}

        for arch_type in arch_types:
            # Create template config
            template = {
                "arch_type": arch_type,
                "num_layers": 2,
                "hidden_dim": 128,
                "num_heads": 4,
                "dropout": 0.1,
                "use_skip_connections": True,
                "recursion_depth": 3,
                "learning_rate": 0.001,
                "cell_type": "gru",
            }

            arch_results = {}

            for param_name, deltas in GuidedMutator.MUTATION_OPTIONS.items():
                if param_name not in template:
                    continue

                old_value = template[param_name]
                param_results = []

                for delta in deltas:
                    if delta == "toggle" and isinstance(old_value, bool):
                        new_value = not old_value
                        effective_delta = 1 if new_value else -1
                    elif isinstance(old_value, (int, float)):
                        new_value = old_value + delta
                        effective_delta = delta
                    else:
                        continue

                    pred = self.controller.predict_improvement(template, param_name, old_value, new_value)
                    param_results.append((effective_delta, pred))

                arch_results[param_name] = param_results

            results[arch_type] = arch_results

        return results

    def get_top_mutations(self, config: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, Any, Any, float]]:
        """
        Get the top predicted mutations for a configuration.
        """
        mutator = GuidedMutator(self.controller, exploration_rate=0.0)
        candidates = mutator.generate_mutations(config)

        # Sort by predicted improvement
        candidates.sort(key=lambda x: x[3], reverse=True)

        return candidates[:top_k]

    def print_analysis(self):
        """Print analysis of controller preferences."""
        print("=" * 60)
        print("MUTATION CONTROLLER ANALYSIS")
        print("=" * 60)

        preferences = self.analyze_parameter_preferences()

        for arch_type, arch_prefs in preferences.items():
            print(f"\n--- {arch_type.upper()} ---")

            for param_name, results in arch_prefs.items():
                # Find best direction
                if results:
                    best = max(results, key=lambda x: x[1])
                    worst = min(results, key=lambda x: x[1])

                    direction = "increase" if best[0] > 0 else "decrease"
                    print(f"  {param_name}: best to {direction} " f"(pred={best[1]:.4f}), worst pred={worst[1]:.4f}")


def test_mutation_controller():
    """Test the mutation controller."""
    print("=" * 60)
    print("MUTATION CONTROLLER TEST")
    print("=" * 60)

    # Create controller
    controller = MutationController()
    trainer = ControllerTrainer(controller)

    # Add synthetic training data
    for _ in range(100):
        config = {
            "arch_type": random.choice(["trn", "rsan", "transformer"]),
            "num_layers": random.randint(1, 5),
            "hidden_dim": random.choice([64, 128, 192, 256]),
            "num_heads": random.choice([2, 4, 6, 8]),
            "dropout": random.choice([0.0, 0.1, 0.2]),
            "use_skip_connections": random.random() > 0.5,
            "recursion_depth": random.randint(1, 5),
            "learning_rate": random.choice([0.0001, 0.001, 0.01]),
        }

        param_name = random.choice(["num_layers", "hidden_dim", "num_heads"])
        old_value = config[param_name]

        if param_name == "num_layers":
            new_value = max(1, min(5, old_value + random.choice([-1, 1])))
        elif param_name == "hidden_dim":
            new_value = max(64, min(256, old_value + random.choice([-32, 32])))
        else:
            new_value = max(2, min(8, old_value + random.choice([-2, 2])))

        # Synthetic fitness: prefer larger hidden dim, moderate layers
        parent_fitness = 0.5 + 0.001 * config["hidden_dim"] - 0.05 * abs(config["num_layers"] - 3)

        child_config = config.copy()
        child_config[param_name] = new_value
        child_fitness = 0.5 + 0.001 * child_config["hidden_dim"] - 0.05 * abs(child_config["num_layers"] - 3)

        trainer.add_mutation_record(config, param_name, old_value, new_value, parent_fitness, child_fitness)

    # Train
    print("\nTraining controller...")
    final_loss = trainer.train(epochs=50)
    print(f"Final loss: {final_loss:.4f}")

    # Analyze
    print("\nAnalyzing learned preferences...")
    analyzer = ControllerAnalyzer(controller)
    analyzer.print_analysis()

    # Test guided mutation
    print("\n--- Guided Mutation Test ---")
    test_config = {
        "arch_type": "transformer",
        "num_layers": 2,
        "hidden_dim": 128,
        "num_heads": 4,
        "dropout": 0.1,
        "use_skip_connections": True,
    }

    top_mutations = analyzer.get_top_mutations(test_config)
    print(f"Top predicted mutations for transformer config:")
    for param, old, new, pred in top_mutations:
        print(f"  {param}: {old} -> {new} (pred improvement: {pred:.4f})")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_mutation_controller()
