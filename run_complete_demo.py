#!/usr/bin/env python
"""
ToM-NAS Complete Demonstration
Showcases all components working together in a full pipeline
"""
import torch
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork
from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent
from src.world.social_world import SocialWorld4
from src.evolution.nas_engine import NASEngine, EvolutionConfig
from src.evaluation.benchmarks import BenchmarkSuite
from train import ToMTrainer


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def demo_1_components():
    """Demonstrate individual component functionality"""
    print_section("DEMO 1: Core Components")

    # Ontology
    print("\n>>> Soul Map Ontology")
    ontology = SoulMapOntology()
    print(f"  Dimensions: {ontology.total_dims}")
    print(f"  Default state shape: {ontology.get_default_state().shape}")
    print("  ✓ Ontology working")

    # Beliefs
    print("\n>>> Recursive Belief System")
    beliefs = BeliefNetwork(num_agents=6, ontology_dim=181, max_order=5)
    print(f"  Agents: {beliefs.num_agents}")
    print(f"  Max belief order: {beliefs.max_order}")
    print("  ✓ Belief network working")

    # Architectures
    print("\n>>> Neural Architectures")
    trn = TransparentRNN(191, 128, 181)
    rsan = RecursiveSelfAttention(191, 128, 181)
    transformer = TransformerToMAgent(191, 128, 181)

    test_input = torch.randn(1, 10, 191)

    trn_out = trn(test_input)
    rsan_out = rsan(test_input)
    trans_out = transformer(test_input)

    print(f"  TRN output: {trn_out['beliefs'].shape}")
    print(f"  RSAN output: {rsan_out['beliefs'].shape}")
    print(f"  Transformer output: {trans_out['beliefs'].shape}")
    print("  ✓ All architectures working")

    # Social World
    print("\n>>> Social World 4")
    world = SocialWorld4(num_agents=10, ontology_dim=181, num_zombies=2)
    print(f"  Agents: {world.num_agents}")
    print(f"  Zombies: {sum(1 for a in world.agents if a.is_zombie)}")

    # Run simulation
    for _ in range(5):
        actions = [{'type': 'cooperate'} for _ in range(10)]
        world.step(actions)

    print(f"  Simulated 5 timesteps")
    stats = world.get_statistics()
    print(f"  Average resources: {stats['avg_resources']:.1f}")
    print("  ✓ Social world working")


def demo_2_training():
    """Demonstrate training pipeline"""
    print_section("DEMO 2: Training Pipeline")

    print("\n>>> Quick Training Demo (5 epochs)")

    config = {
        'architecture': 'TRN',
        'num_agents': 6,
        'ontology_dim': 181,
        'input_dim': 191,
        'hidden_dim': 128,
        'max_belief_order': 5,
        'num_zombies': 2,
        'batch_size': 16,
        'sequence_length': 10,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'device': 'cpu',
        'checkpoint_dir': 'demo_checkpoints',
        'batches_per_epoch': 10,
        'eval_interval': 5,
        'early_stopping': False
    }

    trainer = ToMTrainer(config)

    print("\nTraining TRN agent...")
    best_score = trainer.train(num_epochs=5)

    print(f"\n  Best score achieved: {best_score:.2f}%")
    print("  ✓ Training pipeline working")


def demo_3_evaluation():
    """Demonstrate evaluation and benchmarks"""
    print_section("DEMO 3: Evaluation & Benchmarks")

    print("\n>>> Running Benchmark Suite")

    agent = TransparentRNN(191, 128, 181)
    benchmark_suite = BenchmarkSuite(device='cpu')

    results = benchmark_suite.run_full_suite(agent)

    print(f"\n  Final Score: {results['percentage']:.1f}%")
    print(f"  Tests Passed: {results['passed_count']}/{results['total_tests']}")
    print("  ✓ Benchmarks working")


def demo_4_evolution():
    """Demonstrate evolution/NAS"""
    print_section("DEMO 4: Evolution/NAS (Quick Demo)")

    print("\n>>> Quick Evolution Demo (5 generations, 10 individuals)")

    world = SocialWorld4(num_agents=6, ontology_dim=181, num_zombies=2)
    beliefs = BeliefNetwork(num_agents=6, ontology_dim=181, max_order=5)

    config = EvolutionConfig(
        population_size=10,
        num_generations=5,
        elite_size=2,
        tournament_size=3,
        mutation_rate=0.1,
        crossover_rate=0.7,
        fitness_episodes=2,  # Reduced for demo
        device='cpu',
        checkpoint_interval=5
    )

    nas_engine = NASEngine(config, world, beliefs)
    best_individual = nas_engine.run(num_generations=5)

    print(f"\n  Best fitness: {best_individual.fitness:.4f}")
    print(f"  Best architecture: {best_individual.gene.gene_dict['arch_type']}")
    print("  ✓ Evolution working")


def demo_5_full_pipeline():
    """Demonstrate complete integrated pipeline"""
    print_section("DEMO 5: Complete Integration")

    print("\n>>> Full ToM-NAS Pipeline")

    # 1. Initialize
    print("\n1. Initializing components...")
    ontology = SoulMapOntology()
    beliefs = BeliefNetwork(6, 181, 5)
    world = SocialWorld4(6, 181, 2)
    agent = TransparentRNN(191, 128, 181)
    print("   ✓ All components initialized")

    # 2. Generate data from world
    print("\n2. Generating training data from social world...")
    observations = []
    for t in range(10):
        obs = world.get_observation(0)
        observations.append(obs)
        actions = [{'type': 'cooperate'} for _ in range(6)]
        world.step(actions, beliefs)
    print(f"   ✓ Generated {len(observations)} observations")

    # 3. Forward pass
    print("\n3. Running agent forward pass...")
    test_input = torch.randn(1, 10, 191)
    output = agent(test_input)
    print(f"   ✓ Agent produced beliefs: {output['beliefs'].shape}")
    print(f"   ✓ Agent produced actions: {output['actions'].shape}")

    # 4. Evaluate
    print("\n4. Evaluating on benchmarks...")
    benchmark_suite = BenchmarkSuite()
    results = benchmark_suite.run_full_suite(agent)
    print(f"   ✓ Benchmark score: {results['percentage']:.1f}%")

    # 5. Update beliefs
    print("\n5. Updating belief network...")
    agent_0 = beliefs.agent_beliefs[0]
    for order in range(3):
        for target in range(3):
            content = torch.randn(181)
            agent_0.update_belief(order, target, content)
    print(f"   ✓ Updated recursive beliefs up to order 3")

    # 6. World statistics
    print("\n6. World statistics...")
    stats = world.get_statistics()
    print(f"   Timestep: {stats['timestep']}")
    print(f"   Average resources: {stats['avg_resources']:.1f}")
    print(f"   Zombies remaining: {stats['zombies_remaining']}")
    print("   ✓ World simulation complete")

    print("\n>>> ✓ COMPLETE PIPELINE WORKING")


def main():
    """Run complete demonstration"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  ToM-NAS: Complete System Demonstration".center(78) + "█")
    print("█" + "  Theory of Mind Neural Architecture Search".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: CPU")

    try:
        # Run all demos
        demo_1_components()
        demo_2_training()
        demo_3_evaluation()
        demo_4_evolution()
        demo_5_full_pipeline()

        # Final summary
        print_section("DEMONSTRATION COMPLETE")

        print("\n✓ ALL COMPONENTS VERIFIED:")
        print("  • Soul Map Ontology (181 dimensions)")
        print("  • Recursive Belief System (5th-order)")
        print("  • Three Neural Architectures (TRN, RSAN, Transformer)")
        print("  • Social World 4 (Full simulation)")
        print("  • Evolution/NAS Engine")
        print("  • Benchmark Suite")
        print("  • Training Pipeline")
        print("  • Complete Integration")

        print("\n✓ SYSTEM STATUS: FULLY OPERATIONAL")

        print("\n📚 Next Steps:")
        print("  1. Run full training: python train.py --epochs 100")
        print("  2. Run evolution: python experiment_runner.py --experiment evolution")
        print("  3. Run tests: python test_comprehensive.py")
        print("  4. Generate visualizations: python visualize.py --all")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "█"*80)
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "█"*80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
