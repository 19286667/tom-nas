"""
Training Loop for ToM-NAS Neural Components

Trains:
1. Belief Encoder - encodes agent beliefs into latent space
2. Reasoning RNN - processes sequences for ToM reasoning
3. ToM Predictor - predicts other agents' beliefs/actions

Training data comes from:
- Self-play: agents interacting and observing each other
- Synthetic scenarios: generated false-belief tasks
- Research cycles: hypothesis/result pairs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import random
import logging
import os
import json
from datetime import datetime

from src.config.constants import SOUL_MAP_DIMS, INPUT_DIMS, OUTPUT_DIMS

logger = logging.getLogger(__name__)


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

@dataclass
class ToMExample:
    """A single Theory of Mind training example."""
    context: torch.Tensor          # Observation sequence
    belief_target: torch.Tensor    # What agent believes
    action_target: torch.Tensor    # What agent does
    other_belief: torch.Tensor     # What agent believes OTHER believes
    label: int                     # Task type (0=false belief, 1=true belief, etc.)


class SyntheticDataGenerator:
    """
    Generates synthetic ToM training data.

    Creates scenarios that require genuine belief modeling:
    - False belief tasks (Sally-Anne style)
    - Recursive belief tasks (A thinks B thinks...)
    - Deception scenarios
    - Cooperation scenarios
    """

    def __init__(self, dim: int = SOUL_MAP_DIMS):
        self.dim = dim

    def generate_false_belief_task(self) -> ToMExample:
        """
        Generate a Sally-Anne style false belief task.

        Scenario: Agent A sees object moved. Agent B doesn't.
        Question: Where does B think the object is?

        Correct answer requires modeling B's LIMITED knowledge.
        """
        # Initial state: object at location L1
        location_1 = torch.randn(self.dim) * 0.5
        location_2 = torch.randn(self.dim) * 0.5

        # Context: sequence of observations
        # [initial_state, A_observes_move, B_absent]
        context = torch.stack([
            location_1,  # Object at L1
            location_2,  # Object moved to L2 (A sees this)
            torch.zeros(self.dim),  # B was absent
        ])

        # A's belief: object at L2 (true belief)
        belief_a = location_2.clone()

        # B's belief: object at L1 (false belief - didn't see move)
        belief_b = location_1.clone()

        # A's model of B's belief should be L1
        a_thinks_b_thinks = location_1.clone()

        # Action: if asked "where will B look?", answer L1
        action = location_1.clone()

        return ToMExample(
            context=context,
            belief_target=belief_a,
            action_target=action,
            other_belief=a_thinks_b_thinks,
            label=0,  # False belief task
        )

    def generate_true_belief_task(self) -> ToMExample:
        """
        Generate a true belief control task.

        Both agents see the same thing - no belief divergence.
        """
        location = torch.randn(self.dim) * 0.5

        context = torch.stack([
            location,  # Object at location
            location,  # Both observe (no change)
            torch.ones(self.dim) * 0.5,  # Both present
        ])

        return ToMExample(
            context=context,
            belief_target=location,
            action_target=location,
            other_belief=location,  # True belief - same as own
            label=1,  # True belief task
        )

    def generate_recursive_belief_task(self, depth: int = 2) -> ToMExample:
        """
        Generate recursive belief task.

        Depth 2: A thinks B thinks X
        Depth 3: A thinks B thinks C thinks X
        """
        base_belief = torch.randn(self.dim) * 0.5

        # Each level adds uncertainty/noise
        beliefs = [base_belief]
        for d in range(depth):
            noise = torch.randn(self.dim) * (0.1 * (d + 1))
            beliefs.append(beliefs[-1] + noise)

        context = torch.stack([
            beliefs[0],
            torch.ones(self.dim) * depth,  # Encode depth
            torch.randn(self.dim) * 0.1,
        ])

        return ToMExample(
            context=context,
            belief_target=beliefs[1],
            action_target=beliefs[0],
            other_belief=beliefs[-1],
            label=2 + depth,  # Recursive task
        )

    def generate_deception_task(self) -> ToMExample:
        """
        Generate deception scenario.

        Agent A wants B to believe X, but truth is Y.
        Requires modeling what will change B's beliefs.
        """
        truth = torch.randn(self.dim) * 0.5
        deception = torch.randn(self.dim) * 0.5

        context = torch.stack([
            truth,      # A knows truth
            deception,  # A wants B to believe this
            torch.ones(self.dim) * -0.5,  # Deception intent marker
        ])

        # A's action should convey deception, not truth
        action = deception.clone()

        return ToMExample(
            context=context,
            belief_target=truth,  # A still believes truth
            action_target=action,  # But acts to convey deception
            other_belief=deception,  # Goal: B believes deception
            label=10,  # Deception task
        )

    def generate_batch(self, batch_size: int, task_mix: Dict[str, float] = None) -> List[ToMExample]:
        """Generate a batch of mixed ToM tasks."""
        task_mix = task_mix or {
            'false_belief': 0.3,
            'true_belief': 0.2,
            'recursive_2': 0.2,
            'recursive_3': 0.15,
            'deception': 0.15,
        }

        generators = {
            'false_belief': self.generate_false_belief_task,
            'true_belief': self.generate_true_belief_task,
            'recursive_2': lambda: self.generate_recursive_belief_task(2),
            'recursive_3': lambda: self.generate_recursive_belief_task(3),
            'deception': self.generate_deception_task,
        }

        batch = []
        for _ in range(batch_size):
            task = random.choices(
                list(task_mix.keys()),
                weights=list(task_mix.values())
            )[0]
            batch.append(generators[task]())

        return batch


class SelfPlayDataCollector:
    """
    Collects training data from actual agent interactions.

    Watches agents research, form beliefs, and interact,
    then extracts (observation, belief, action) tuples.
    """

    def __init__(self, max_buffer: int = 10000):
        self.buffer: List[ToMExample] = []
        self.max_buffer = max_buffer

    def record_interaction(
        self,
        observer_id: str,
        observed_id: str,
        context: torch.Tensor,
        observer_belief: torch.Tensor,
        observed_action: torch.Tensor,
        observer_prediction: torch.Tensor,
    ):
        """Record an interaction for training."""
        example = ToMExample(
            context=context,
            belief_target=observer_belief,
            action_target=observed_action,
            other_belief=observer_prediction,
            label=100,  # Self-play data
        )

        self.buffer.append(example)

        # Maintain buffer size
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer:]

    def sample(self, n: int) -> List[ToMExample]:
        """Sample n examples from buffer."""
        if len(self.buffer) < n:
            return self.buffer.copy()
        return random.sample(self.buffer, n)


class ToMDataset(Dataset):
    """PyTorch dataset for ToM training."""

    def __init__(self, examples: List[ToMExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'context': ex.context,
            'belief_target': ex.belief_target,
            'action_target': ex.action_target,
            'other_belief': ex.other_belief,
            'label': ex.label,
        }


# =============================================================================
# TRAINING LOOP
# =============================================================================

class ToMTrainer:
    """
    Trains Theory of Mind neural components.

    Components:
    - Belief Encoder: observations → belief state
    - ToM Predictor: belief state → predicted other's belief
    - Action Predictor: belief state → action
    """

    def __init__(
        self,
        belief_encoder: nn.Module,
        tom_predictor: nn.Module,
        action_predictor: nn.Module,
        learning_rate: float = 1e-4,
        device: str = 'cpu',
    ):
        self.belief_encoder = belief_encoder.to(device)
        self.tom_predictor = tom_predictor.to(device)
        self.action_predictor = action_predictor.to(device)
        self.device = device

        # Optimizers
        self.optimizer = optim.AdamW(
            list(belief_encoder.parameters()) +
            list(tom_predictor.parameters()) +
            list(action_predictor.parameters()),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Loss functions
        self.belief_loss = nn.MSELoss()
        self.action_loss = nn.MSELoss()
        self.tom_loss = nn.MSELoss()

        # Metrics
        self.metrics = {
            'train_loss': [],
            'belief_loss': [],
            'action_loss': [],
            'tom_loss': [],
            'false_belief_acc': [],
        }

        # Data generator
        self.data_generator = SyntheticDataGenerator()
        self.self_play_collector = SelfPlayDataCollector()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        self.belief_encoder.train()
        self.tom_predictor.train()
        self.action_predictor.train()

        # Move to device
        context = batch['context'].to(self.device)
        belief_target = batch['belief_target'].to(self.device)
        action_target = batch['action_target'].to(self.device)
        other_belief = batch['other_belief'].to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        # 1. Encode beliefs from context
        belief_state = self.belief_encoder(context)

        # 2. Predict other's belief (ToM)
        predicted_other = self.tom_predictor(belief_state)

        # 3. Predict action
        predicted_action = self.action_predictor(belief_state)

        # Compute losses
        loss_belief = self.belief_loss(belief_state, belief_target)
        loss_tom = self.tom_loss(predicted_other, other_belief)
        loss_action = self.action_loss(predicted_action, action_target)

        # Combined loss (weighted)
        total_loss = loss_belief + 2.0 * loss_tom + loss_action

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.belief_encoder.parameters()) +
            list(self.tom_predictor.parameters()) +
            list(self.action_predictor.parameters()),
            max_norm=1.0
        )

        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'belief_loss': loss_belief.item(),
            'tom_loss': loss_tom.item(),
            'action_loss': loss_action.item(),
        }

    def evaluate_false_belief(self, n_samples: int = 100) -> float:
        """
        Evaluate false belief task accuracy.

        This is the key ToM benchmark: can the model predict
        that another agent has a FALSE belief?
        """
        self.belief_encoder.eval()
        self.tom_predictor.eval()

        correct = 0

        with torch.no_grad():
            for _ in range(n_samples):
                # Generate false belief scenario
                example = self.data_generator.generate_false_belief_task()

                context = example.context.unsqueeze(0).to(self.device)
                true_other_belief = example.other_belief.to(self.device)
                own_belief = example.belief_target.to(self.device)

                # Predict
                belief_state = self.belief_encoder(context)
                predicted_other = self.tom_predictor(belief_state).squeeze()

                # Check if prediction is closer to FALSE belief than TRUE belief
                dist_to_false = torch.dist(predicted_other, true_other_belief)
                dist_to_true = torch.dist(predicted_other, own_belief)

                if dist_to_false < dist_to_true:
                    correct += 1

        accuracy = correct / n_samples
        return accuracy

    def train_epoch(
        self,
        n_batches: int = 100,
        batch_size: int = 32,
        synthetic_ratio: float = 0.7,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {
            'total_loss': 0,
            'belief_loss': 0,
            'tom_loss': 0,
            'action_loss': 0,
        }

        for _ in range(n_batches):
            # Generate batch
            n_synthetic = int(batch_size * synthetic_ratio)
            n_selfplay = batch_size - n_synthetic

            examples = self.data_generator.generate_batch(n_synthetic)
            examples.extend(self.self_play_collector.sample(n_selfplay))

            if not examples:
                examples = self.data_generator.generate_batch(batch_size)

            # Create batch tensors
            batch = {
                'context': torch.stack([ex.context for ex in examples]),
                'belief_target': torch.stack([ex.belief_target for ex in examples]),
                'action_target': torch.stack([ex.action_target for ex in examples]),
                'other_belief': torch.stack([ex.other_belief for ex in examples]),
                'label': torch.tensor([ex.label for ex in examples]),
            }

            # Train step
            step_metrics = self.train_step(batch)

            for k, v in step_metrics.items():
                epoch_metrics[k] += v / n_batches

        # Evaluate false belief
        false_belief_acc = self.evaluate_false_belief()
        epoch_metrics['false_belief_acc'] = false_belief_acc

        # Record metrics
        for k, v in epoch_metrics.items():
            self.metrics.setdefault(k, []).append(v)

        return epoch_metrics

    def train(
        self,
        n_epochs: int = 100,
        n_batches_per_epoch: int = 100,
        batch_size: int = 32,
        checkpoint_dir: str = 'checkpoints',
        log_every: int = 10,
    ):
        """Full training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"Starting training for {n_epochs} epochs")

        best_acc = 0

        for epoch in range(n_epochs):
            metrics = self.train_epoch(n_batches_per_epoch, batch_size)

            if epoch % log_every == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"loss={metrics['total_loss']:.4f}, "
                    f"tom_loss={metrics['tom_loss']:.4f}, "
                    f"false_belief_acc={metrics['false_belief_acc']:.2%}"
                )

            # Save best model
            if metrics['false_belief_acc'] > best_acc:
                best_acc = metrics['false_belief_acc']
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, 'best_model.pt'),
                    epoch,
                    metrics
                )

            # Periodic checkpoint
            if epoch % 50 == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'),
                    epoch,
                    metrics
                )

        logger.info(f"Training complete. Best false belief accuracy: {best_acc:.2%}")

        return self.metrics

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'belief_encoder': self.belief_encoder.state_dict(),
            'tom_predictor': self.tom_predictor.state_dict(),
            'action_predictor': self.action_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.belief_encoder.load_state_dict(checkpoint['belief_encoder'])
        self.tom_predictor.load_state_dict(checkpoint['tom_predictor'])
        self.action_predictor.load_state_dict(checkpoint['action_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


# =============================================================================
# SIMPLE MODELS FOR TRAINING
# =============================================================================

class SimpleBeliefEncoder(nn.Module):
    """Simple belief encoder for training."""

    def __init__(self, input_dim: int = SOUL_MAP_DIMS, hidden_dim: int = 256, output_dim: int = SOUL_MAP_DIMS):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq, dim)
        _, hidden = self.rnn(x)
        return self.fc(hidden.squeeze(0))


class SimpleToMPredictor(nn.Module):
    """Simple ToM predictor."""

    def __init__(self, dim: int = SOUL_MAP_DIMS, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleActionPredictor(nn.Module):
    """Simple action predictor."""

    def __init__(self, dim: int = SOUL_MAP_DIMS, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# ENTRY POINT
# =============================================================================

def create_trainer(device: str = 'cpu') -> ToMTrainer:
    """Create trainer with default models."""
    belief_encoder = SimpleBeliefEncoder()
    tom_predictor = SimpleToMPredictor()
    action_predictor = SimpleActionPredictor()

    return ToMTrainer(
        belief_encoder=belief_encoder,
        tom_predictor=tom_predictor,
        action_predictor=action_predictor,
        device=device,
    )


def main():
    """Run training."""
    import argparse

    parser = argparse.ArgumentParser(description='Train ToM-NAS neural components')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    trainer = create_trainer(args.device)
    trainer.train(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == '__main__':
    main()
