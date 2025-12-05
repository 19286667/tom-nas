#!/usr/bin/env python
"""
Complete ToM-NAS System Runner

Usage:
    python scripts/run_full_system.py --mode gui     # Launch Streamlit app
    python scripts/run_full_system.py --mode cli     # Run CLI experiment
    python scripts/run_full_system.py --mode demo    # Run Liminal demo
    python scripts/run_full_system.py --mode test    # Run all tests
    python scripts/run_full_system.py --mode verify  # Verify core systems
"""

import argparse
import subprocess
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_gui():
    """Launch the Streamlit visualization app."""
    print("=" * 60)
    print("Launching ToM-NAS Visualization")
    print("=" * 60)
    print()
    print("Starting Streamlit server...")
    print("The browser should open automatically.")
    print("If not, navigate to http://localhost:8501")
    print()

    app_path = os.path.join(project_root, "src", "visualization", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


def run_demo():
    """Run the Liminal demo."""
    print("=" * 60)
    print("Running Liminal Demo")
    print("=" * 60)
    print()

    demo_path = os.path.join(project_root, "run_liminal_demo.py")
    if os.path.exists(demo_path):
        subprocess.run([sys.executable, demo_path, "--demo", "all"])
    else:
        print("Liminal demo not found. Running built-in demo...")
        run_builtin_demo()


def run_builtin_demo():
    """Run a built-in demo of the system."""
    print("\n[1] Initializing Liminal Environment...")

    from src.liminal import LiminalEnvironment, RealmType

    env = LiminalEnvironment(population_size=100, include_heroes=True)
    print(f"    Created environment with {len(env.npcs)} NPCs")

    # Get stats
    stats = env.get_statistics()
    print(f"    Hero NPCs: {stats['hero_count']}")
    print(f"    Zombie NPCs: {stats['zombie_count']}")

    print("\n[2] NPCs per Realm:")
    for realm, count in stats['npcs_per_realm'].items():
        print(f"    {realm}: {count} NPCs")

    print("\n[3] Running Sally-Anne Scenario...")

    from src.core.events import create_sally_anne_scenario, verify_information_asymmetry

    events, questions = create_sally_anne_scenario()
    results = verify_information_asymmetry()

    print(f"    Events created: {len(events)}")
    print(f"    Sally believes marble is in: {results['sally_marble_belief']}")
    print(f"    Reality: {results['reality']}")
    print(f"    Sally has false belief: {results['sally_has_false_belief']}")

    if results['all_tests_passed']:
        print("    [PASS] Information asymmetry working correctly!")
    else:
        print("    [FAIL] Information asymmetry test failed")

    print("\n[4] Running Benchmark Sample...")

    from src.benchmarks import ToMiDataset

    dataset = ToMiDataset()
    print(f"    ToMi examples: {len(dataset)}")

    batch_events, batch_targets, batch_examples = dataset.get_batch(4)
    print(f"    Batch shape: {batch_events.shape}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running ToM-NAS Tests")
    print("=" * 60)
    print()

    # Run pytest
    tests_dir = os.path.join(project_root, "tests")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", tests_dir, "-v", "--tb=short"],
        cwd=project_root
    )

    return result.returncode


def verify_systems():
    """Verify that core systems are working."""
    print("=" * 60)
    print("ToM-NAS System Verification")
    print("=" * 60)
    print()

    all_passed = True

    # Test 1: Information Asymmetry
    print("[1] Verifying information asymmetry...")
    try:
        from src.core.events import verify_information_asymmetry
        results = verify_information_asymmetry()

        if results['all_tests_passed']:
            print("    [PASS] Sally has false belief")
            print("    [PASS] Anne has true belief")
            print("    [PASS] Information asymmetry working")
        else:
            print("    [FAIL] Information asymmetry test failed")
            all_passed = False
    except Exception as e:
        print(f"    [ERROR] {e}")
        all_passed = False

    # Test 2: Belief Network
    print("\n[2] Verifying belief network...")
    try:
        from src.core.beliefs import BeliefNetwork
        import torch

        network = BeliefNetwork(num_agents=5, ontology_dim=181, max_order=5)
        content = torch.randn(181)
        success = network.update_agent_belief(0, order=1, target=1, content=content)

        if success:
            print("    [PASS] Belief network functional")
        else:
            print("    [FAIL] Belief update failed")
            all_passed = False
    except Exception as e:
        print(f"    [ERROR] {e}")
        all_passed = False

    # Test 3: ToMi Benchmark
    print("\n[3] Verifying ToMi benchmark...")
    try:
        from src.benchmarks import ToMiDataset

        dataset = ToMiDataset()
        events, targets, examples = dataset.get_batch(4)

        if events.shape[0] == 4 and targets.shape[0] == 4:
            print("    [PASS] ToMi dataset functional")
        else:
            print("    [FAIL] Batch generation failed")
            all_passed = False
    except Exception as e:
        print(f"    [ERROR] {e}")
        all_passed = False

    # Test 4: Supernet
    print("\n[4] Verifying elastic supernet...")
    try:
        from src.evolution.supernet import ElasticLSTMCell
        import torch

        cell = ElasticLSTMCell(input_dim=64, max_hidden_dim=256)
        x = torch.randn(2, 64)
        h = torch.zeros(2, 256)
        c = torch.zeros(2, 256)

        # Test with reduced hidden dim (the bug fix)
        h_new, _ = cell(x, (h, c), active_hidden_dim=128)

        if h_new.shape == torch.Size([2, 256]):
            print("    [PASS] Elastic LSTM working with variable dimensions")
        else:
            print("    [FAIL] Dimension mismatch")
            all_passed = False
    except RuntimeError as e:
        if "normalized_shape" in str(e):
            print("    [FAIL] LayerNorm dimension bug not fixed")
            all_passed = False
        else:
            raise
    except Exception as e:
        print(f"    [ERROR] {e}")
        all_passed = False

    # Test 5: Liminal Environment
    print("\n[5] Verifying Liminal environment...")
    try:
        from src.liminal import LiminalEnvironment

        env = LiminalEnvironment(population_size=50, include_heroes=True)
        obs = env.reset()

        if obs is not None and len(env.npcs) > 0:
            print(f"    [PASS] Environment created with {len(env.npcs)} NPCs")
        else:
            print("    [FAIL] Environment creation failed")
            all_passed = False
    except Exception as e:
        print(f"    [ERROR] {e}")
        all_passed = False

    # Test 6: Social Games
    print("\n[6] Verifying social games benchmark...")
    try:
        from src.benchmarks import SocialGameBenchmark

        benchmark = SocialGameBenchmark(num_agents=10, num_zombies=2)
        zombies = sum(1 for a in benchmark.world.agents if a.is_zombie)

        if zombies == 2:
            print("    [PASS] Social games functional")
        else:
            print("    [FAIL] Zombie creation incorrect")
            all_passed = False
    except Exception as e:
        print(f"    [ERROR] {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SYSTEMS VERIFIED SUCCESSFULLY")
    else:
        print("SOME SYSTEMS FAILED - See above for details")
    print("=" * 60)

    return 0 if all_passed else 1


def run_cli_experiment(generations: int = 50):
    """Run a CLI-based evolution experiment."""
    print("=" * 60)
    print("ToM-NAS: Theory of Mind Neural Architecture Search")
    print("=" * 60)
    print()

    # First verify systems
    print("Verifying systems before starting...")
    from src.core.events import verify_information_asymmetry

    results = verify_information_asymmetry()
    if not results['all_tests_passed']:
        print("[ERROR] Core systems not working. Run --mode verify for details.")
        return 1

    print("[OK] Core systems verified")
    print()

    # Setup
    print("Setting up evolution...")

    from src.world.social_world import SocialWorld4
    from src.core.beliefs import BeliefNetwork
    from src.evolution import NASEngine, EvolutionConfig
    from src.benchmarks import UnifiedBenchmark

    # Create world and belief network
    world = SocialWorld4(num_agents=20, ontology_dim=181, num_zombies=3)
    belief_network = BeliefNetwork(num_agents=20, ontology_dim=181, max_order=5)

    # Configure evolution
    config = EvolutionConfig(
        population_size=20,
        num_generations=generations,
        elite_size=2,
        tournament_size=3,
        mutation_rate=0.1,
        crossover_rate=0.7,
        fitness_episodes=3,
        device='cpu',
        input_dim=191,
        output_dim=181,
    )

    # Create NAS engine
    print(f"Creating NAS engine (population={config.population_size})...")
    engine = NASEngine(config, world, belief_network)

    # Run evolution
    print(f"\nStarting evolution for {generations} generations...")
    print("=" * 60)

    try:
        best = engine.run(num_generations=generations)

        print("\nEvolution complete!")
        print(f"Best fitness: {best.fitness:.4f}")
        print(f"Best architecture: {best.gene.gene_dict['arch_type']}")

        # Save results
        engine.save_checkpoint("final_checkpoint.pt")
        print("\nCheckpoint saved to final_checkpoint.pt")

    except KeyboardInterrupt:
        print("\n\nEvolution interrupted by user")
        print("Saving partial results...")
        engine.save_checkpoint("interrupted_checkpoint.pt")

    return 0


def main():
    parser = argparse.ArgumentParser(description="ToM-NAS System Runner")
    parser.add_argument(
        '--mode',
        choices=['gui', 'cli', 'demo', 'test', 'verify'],
        default='gui',
        help='Mode to run: gui (visualization), cli (experiment), demo, test, verify'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=50,
        help='Number of generations for CLI mode'
    )

    args = parser.parse_args()

    if args.mode == 'gui':
        run_gui()
    elif args.mode == 'demo':
        run_demo()
    elif args.mode == 'test':
        return run_tests()
    elif args.mode == 'verify':
        return verify_systems()
    elif args.mode == 'cli':
        return run_cli_experiment(args.generations)


if __name__ == "__main__":
    sys.exit(main() or 0)
