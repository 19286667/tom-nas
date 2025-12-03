"""
ToM-NAS Simulation - Main Entry Point

Run with: python -m src.simulation
"""

import sys
import argparse

from .menu import MenuSystem
from .shell import ToMNASShell
from .visualization import demo_visualization
from .liminal_integration import demo_integration


def main():
    """Main entry point for ToM-NAS simulation."""
    parser = argparse.ArgumentParser(
        description='ToM-NAS: Theory of Mind Neural Architecture Search Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.simulation              # Launch interactive menu
  python -m src.simulation --shell      # Launch command shell
  python -m src.simulation --demo       # Run quick demo
  python -m src.simulation --viz        # Test visualization
        """
    )

    parser.add_argument(
        '--shell', '-s',
        action='store_true',
        help='Launch interactive command shell'
    )

    parser.add_argument(
        '--menu', '-m',
        action='store_true',
        help='Launch graphical menu (default)'
    )

    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run a quick demonstration'
    )

    parser.add_argument(
        '--viz', '-v',
        action='store_true',
        help='Test visualization system'
    )

    parser.add_argument(
        '--integration', '-i',
        action='store_true',
        help='Test liminal integration'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Determine mode
    if args.shell:
        print("Launching ToM-NAS Shell...")
        shell = ToMNASShell()
        shell.cmdloop()

    elif args.demo:
        print("Running ToM-NAS Demo...")
        run_demo(args.seed)

    elif args.viz:
        print("Testing Visualization System...")
        demo_visualization()

    elif args.integration:
        print("Testing Liminal Integration...")
        demo_integration()

    else:
        # Default: launch menu
        print("Launching ToM-NAS Menu...")
        menu = MenuSystem()
        menu.run()


def run_demo(seed=None):
    """Run a quick demonstration of the simulation."""
    from .shell import SimulationConfig, SimulationRunner

    print("\n" + "=" * 60)
    print(" ToM-NAS Quick Demo")
    print("=" * 60)

    # Create configuration
    config = SimulationConfig(
        world_size=(30, 30, 1),
        num_agents=8,
        num_zombies=2,
        tom_levels=[0, 1, 2, 3],
        steps_per_run=20,
        benchmark_frequency=5,
        seed=seed or 42,
    )

    print(f"\nConfiguration:")
    print(f"  World: {config.world_size[0]}x{config.world_size[1]}")
    print(f"  Agents: {config.num_agents} ({config.num_zombies} zombies)")
    print(f"  ToM levels: {config.tom_levels}")
    print(f"  Steps: {config.steps_per_run}")

    # Create runner
    runner = SimulationRunner(config)

    print("\nInitializing simulation...")
    runner.initialize()

    print(f"Created {len(runner.agents)} agents")

    # Run simulation
    print("\nRunning simulation...")
    for step in range(config.steps_per_run):
        runner.step()

        # Show progress every 5 steps
        if (step + 1) % 5 == 0:
            stats = runner.get_statistics()
            print(f"  Step {step + 1}: Avg fitness = {stats['avg_fitness']:.3f}")

    # Final statistics
    print("\n" + "-" * 40)
    print("Final Statistics:")
    stats = runner.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Show top agents
    print("\nTop Performing Agents:")
    sorted_agents = sorted(
        runner.agents.values(),
        key=lambda a: a.success.compute_fitness(a.profile.get_success_weights()),
        reverse=True
    )
    for i, agent in enumerate(sorted_agents[:3]):
        fitness = agent.success.compute_fitness(agent.profile.get_success_weights())
        tom_level = agent.tom_reasoner.k_level
        zombie = " (ZOMBIE)" if agent.is_zombie else ""
        print(f"  {i+1}. Agent {agent.id} (L{tom_level}){zombie}: {fitness:.3f}")

    print("\n" + "=" * 60)
    print(" Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
