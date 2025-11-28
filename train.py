#!/usr/bin/env python
"""
ToM-NAS Training Pipeline
Complete training script with checkpointing, logging, and evaluation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork
from src.world.social_world import SocialWorld4
from src.evaluation.benchmarks import BenchmarkSuite
from src.evaluation.metrics import MetricsTracker
from src.utils import observation_to_tensor, create_model


class ToMTrainer:
    """Complete training pipeline for ToM agents"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))

        # Initialize components
        self.ontology = SoulMapOntology()
        self.belief_network = BeliefNetwork(
            num_agents=config['num_agents'],
            ontology_dim=config['ontology_dim'],
            max_order=config['max_belief_order']
        )
        self.world = SocialWorld4(
            num_agents=config['num_agents'],
            ontology_dim=config['ontology_dim'],
            num_zombies=config.get('num_zombies', 2)
        )

        # Create agent
        self.agent = self._create_agent(config)
        self.agent.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Loss functions
        self.belief_criterion = nn.MSELoss()
        self.action_criterion = nn.BCELoss()

        # Metrics
        self.metrics = MetricsTracker()
        self.benchmark_suite = BenchmarkSuite(self.device)

        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_score = 0.0
        self.epochs_without_improvement = 0

    def _create_agent(self, config: Dict) -> nn.Module:
        """Create agent based on configuration"""
        return create_model(
            arch_type=config.get('architecture', 'TRN'),
            input_dim=config.get('input_dim', 191),
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=config.get('ontology_dim', 181),
            num_layers=config.get('num_layers', 2),
            num_heads=config.get('num_heads', 4)
        )

    def generate_training_batch(self, batch_size: int = 32,
                               sequence_length: int = 20) -> Dict:
        """Generate a batch of training data from social world"""
        batch_observations = []
        batch_target_beliefs = []
        batch_target_actions = []

        for _ in range(batch_size):
            # Reset world
            self.world = SocialWorld4(
                self.config['num_agents'],
                self.config['ontology_dim'],
                self.config.get('num_zombies', 2)
            )

            observations = []
            target_beliefs = []
            target_actions = []

            for t in range(sequence_length):
                # Get observation for agent 0
                obs = self.world.get_observation(0)

                # Convert observation to tensor
                obs_tensor = self._obs_to_tensor(obs)
                observations.append(obs_tensor)

                # Generate random actions for all agents
                actions = [
                    {'type': np.random.choice(['cooperate', 'defect', 'share'])}
                    for _ in range(self.config['num_agents'])
                ]

                # Step world
                result = self.world.step(actions, self.belief_network)

                # Create target beliefs (ground truth from world state)
                target_belief = torch.zeros(self.config['ontology_dim'])
                if len(self.world.agents) > 0:
                    target_belief[:3] = torch.tensor([
                        self.world.agents[0].resources / 200.0,
                        self.world.agents[0].energy / 100.0,
                        0.5  # Placeholder for other features
                    ])

                target_beliefs.append(target_belief)

                # Target action (cooperative or not)
                target_action = torch.tensor([0.7])  # Cooperative by default
                target_actions.append(target_action)

            # Stack into sequences
            obs_seq = torch.stack(observations)
            belief_seq = torch.stack(target_beliefs)
            action_seq = torch.stack(target_actions)

            batch_observations.append(obs_seq)
            batch_target_beliefs.append(belief_seq[-1])  # Last timestep
            batch_target_actions.append(action_seq[-1])

        return {
            'observations': torch.stack(batch_observations).to(self.device),
            'target_beliefs': torch.stack(batch_target_beliefs).to(self.device),
            'target_actions': torch.stack(batch_target_actions).to(self.device)
        }

    def _obs_to_tensor(self, obs: Dict) -> torch.Tensor:
        """Convert observation to tensor"""
        return observation_to_tensor(obs)

    def train_epoch(self, num_batches: int = 50) -> Dict:
        """Train for one epoch"""
        self.agent.train()

        epoch_loss = 0.0
        epoch_belief_loss = 0.0
        epoch_action_loss = 0.0

        for batch_idx in range(num_batches):
            # Generate batch
            batch = self.generate_training_batch(
                batch_size=self.config.get('batch_size', 32),
                sequence_length=self.config.get('sequence_length', 20)
            )

            # Forward pass
            self.optimizer.zero_grad()
            output = self.agent(batch['observations'])

            # Calculate losses
            belief_loss = self.belief_criterion(
                output['beliefs'],
                batch['target_beliefs']
            )

            action_loss = self.action_criterion(
                output['actions'].unsqueeze(-1),
                batch['target_actions']
            )

            # Combined loss
            total_loss = belief_loss + 0.5 * action_loss

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track losses
            epoch_loss += total_loss.item()
            epoch_belief_loss += belief_loss.item()
            epoch_action_loss += action_loss.item()

        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_belief_loss = epoch_belief_loss / num_batches
        avg_action_loss = epoch_action_loss / num_batches

        return {
            'total_loss': avg_loss,
            'belief_loss': avg_belief_loss,
            'action_loss': avg_action_loss
        }

    def evaluate(self) -> Dict:
        """Evaluate agent on benchmark suite"""
        self.agent.eval()

        results = self.benchmark_suite.run_full_suite(self.agent)

        return results

    def train(self, num_epochs: int):
        """Complete training loop"""
        print("\n" + "="*80)
        print("ToM-NAS Training Started")
        print("="*80)
        print(f"Architecture:  {self.config.get('architecture', 'TRN')}")
        print(f"Epochs:        {num_epochs}")
        print(f"Batch size:    {self.config.get('batch_size', 32)}")
        print(f"Learning rate: {self.config.get('learning_rate', 0.001)}")
        print(f"Device:        {self.device}")
        print("="*80)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-"*80)

            # Train
            train_results = self.train_epoch(
                num_batches=self.config.get('batches_per_epoch', 50)
            )

            print(f"  Loss:        {train_results['total_loss']:.4f}")
            print(f"  Belief Loss: {train_results['belief_loss']:.4f}")
            print(f"  Action Loss: {train_results['action_loss']:.4f}")

            # Log metrics
            self.metrics.log_training(
                epoch=epoch,
                loss=train_results['total_loss'],
                belief_accuracy=1.0 - train_results['belief_loss'],
                action_accuracy=1.0 - train_results['action_loss'],
                learning_rate=self.optimizer.param_groups[0]['lr']
            )

            # Evaluate periodically
            if (epoch + 1) % self.config.get('eval_interval', 10) == 0:
                print("\n  Running evaluation...")
                eval_results = self.evaluate()

                score = eval_results['percentage']
                print(f"  Benchmark Score: {score:.2f}%")

                # Save if best
                if score > self.best_score:
                    self.best_score = score
                    self.epochs_without_improvement = 0
                    self.save_checkpoint('best_model.pt')
                    print(f"  New best score! Model saved.")
                else:
                    self.epochs_without_improvement += 1

                # Log evaluation metrics
                self.metrics.log_evaluation(
                    benchmark_scores={'overall': score},
                    tom_order_scores={},
                    cooperation_score=0.0,
                    zombie_detection_score=0.0,
                    overall_score=score
                )

            # Checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 20) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

            # Early stopping
            if self.config.get('early_stopping', False):
                patience = self.config.get('patience', 20)
                if self.epochs_without_improvement >= patience:
                    print(f"\nEarly stopping after {patience} epochs without improvement")
                    break

        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        print(f"Best Score: {self.best_score:.2f}%")

        # Save final metrics
        self.metrics.save_to_file(os.path.join(self.checkpoint_dir, 'metrics.json'))

        return self.best_score

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_score': self.best_score,
            'metrics_summary': self.metrics.get_training_summary()
        }

        torch.save(checkpoint, filepath)
        print(f"  Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_score = checkpoint.get('best_score', 0.0)

        print(f"Checkpoint loaded: {filepath}")
        print(f"Best score: {self.best_score:.2f}%")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ToM-NAS Agent')

    parser.add_argument('--architecture', type=str, default='TRN',
                       choices=['TRN', 'RSAN', 'Transformer'],
                       help='Architecture to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--num-agents', type=int, default=6,
                       help='Number of agents in world')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--eval-interval', type=int, default=10,
                       help='Evaluate every N epochs')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping')

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    config = {
        'architecture': args.architecture,
        'num_agents': args.num_agents,
        'ontology_dim': 181,
        'input_dim': 191,
        'hidden_dim': args.hidden_dim,
        'max_belief_order': 5,
        'num_zombies': 2,
        'batch_size': args.batch_size,
        'sequence_length': 20,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'device': args.device,
        'checkpoint_dir': args.checkpoint_dir,
        'batches_per_epoch': 50,
        'eval_interval': args.eval_interval,
        'checkpoint_interval': 20,
        'early_stopping': args.early_stopping,
        'patience': 20
    }

    # Create trainer
    trainer = ToMTrainer(config)

    # Train
    best_score = trainer.train(args.epochs)

    print(f"\nFinal best score: {best_score:.2f}%")


if __name__ == "__main__":
    main()
