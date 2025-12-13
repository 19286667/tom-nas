#!/usr/bin/env python3
"""
ToM-NAS Training and Benchmarking Script

Usage:
    python train_and_benchmark.py --train --epochs 100
    python train_and_benchmark.py --benchmark
    python train_and_benchmark.py --train --benchmark --epochs 50
"""

import argparse
import logging
import os
import sys
import torch

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training import (
    ToMTrainer,
    SimpleBeliefEncoder,
    SimpleToMPredictor,
    SimpleActionPredictor,
    create_trainer,
)
from src.benchmarks import UnifiedBenchmark, evaluate_tom_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(args):
    """Run training."""
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    logger.info(f"Using device: {device}")

    trainer = create_trainer(device=device)

    metrics = trainer.train(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
    )

    logger.info("Training complete!")
    logger.info(f"Final false belief accuracy: {metrics['false_belief_acc'][-1]:.2%}")

    return trainer


def benchmark(args, trainer=None):
    """Run benchmarks."""
    logger.info("=" * 60)
    logger.info("RUNNING BENCHMARKS")
    logger.info("=" * 60)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    # Create or load model
    if trainer:
        model = trainer.belief_encoder
    else:
        model = SimpleBeliefEncoder()
        if args.checkpoint:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['belief_encoder'])

    model = model.to(device)
    model.eval()

    # Run benchmarks
    if args.quick:
        logger.info("Running quick evaluation...")
        results = evaluate_tom_model(model, device=device, quick=True)
    else:
        logger.info("Running full evaluation...")
        benchmark_suite = UnifiedBenchmark()
        full_results = benchmark_suite.full_evaluation(model, device=device)
        results = {
            'tom_aggregate': full_results.tom_aggregate,
            'control_aggregate': full_results.control_aggregate,
            'tom_specificity': full_results.tom_specificity,
            'tomi_accuracy': full_results.tomi_tom_accuracy,
            'tomi_first_order': full_results.tomi_first_order,
            'tomi_second_order': full_results.tomi_second_order,
            'social_iqa': full_results.social_iqa_accuracy,
            'social_games': full_results.social_games_prediction,
        }

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)

    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2%}")
        else:
            logger.info(f"  {key}: {value}")

    # Check if above chance
    tom_score = results.get('tom_aggregate', results.get('tom_score', 0))
    if tom_score > 0.6:
        logger.info("")
        logger.info("✓ Model shows above-chance ToM performance!")
    else:
        logger.info("")
        logger.info("✗ Model at or below chance level")

    return results


def main():
    parser = argparse.ArgumentParser(description='ToM-NAS Training and Benchmarking')

    # Mode
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')

    # Training args
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-every', type=int, default=10, help='Log every N epochs')

    # Benchmark args
    parser.add_argument('--checkpoint', type=str, help='Load checkpoint for benchmarking')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks')

    # General
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    args = parser.parse_args()

    if not args.train and not args.benchmark:
        logger.info("No mode specified, running both training and benchmarks")
        args.train = True
        args.benchmark = True

    trainer = None
    if args.train:
        trainer = train(args)

    if args.benchmark:
        benchmark(args, trainer)


if __name__ == '__main__':
    main()
