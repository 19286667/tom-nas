#!/usr/bin/env python
"""
ToM-NAS Evolution Runner with Rich Visualization
Runs full evolutionary experiments with interpretable output
"""
import torch
import numpy as np
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better visualization: pip install rich")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import ToM-NAS components
from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork
from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent, create_architecture
from src.world.social_world import SocialWorld4
from src.evolution.nas_engine import NASEngine, EvolutionConfig
from src.evaluation.zombie_detection import ZombieDetectionSuite, ZombieType
from src.evaluation.tom_benchmarks import ToMBenchmarkSuite
from src.training.curriculum import CurriculumManager, CurriculumStage

console = Console() if RICH_AVAILABLE else None


def print_header():
    """Print fancy header."""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]Theory of Mind Neural Architecture Search[/bold cyan]\n"
            "[dim]Evolving genuine ToM through coevolutionary pressure[/dim]",
            border_style="cyan"
        ))
    else:
        print("="*70)
        print("Theory of Mind Neural Architecture Search (ToM-NAS)")
        print("Evolving genuine ToM through coevolutionary pressure")
        print("="*70)


def create_stats_table(generation: int, stats: dict) -> Table:
    """Create a rich table for generation statistics."""
    table = Table(title=f"Generation {generation} Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Best Fitness", f"{stats['best_fitness']:.4f}")
    table.add_row("Avg Fitness", f"{stats['avg_fitness']:.4f}")
    table.add_row("Diversity", f"{stats['diversity']:.4f}")
    table.add_row("Species Count", str(stats.get('species_count', 1)))
    table.add_row("Mutation Rate", f"{stats['mutation_rate']:.4f}")

    return table


def create_population_table(population: list) -> Table:
    """Create table showing population overview."""
    table = Table(title="Population Overview", show_header=True)
    table.add_column("#", style="dim")
    table.add_column("Architecture", style="cyan")
    table.add_column("Fitness", style="green")
    table.add_column("Generation", style="yellow")

    for i, ind in enumerate(sorted(population, key=lambda x: x.fitness or 0, reverse=True)[:10]):
        arch = ind.gene.gene_dict.get('arch_type', 'Unknown')
        fitness = f"{ind.fitness:.4f}" if ind.fitness else "N/A"
        table.add_row(str(i+1), arch, fitness, str(ind.generation))

    return table


def create_tom_analysis_table(tom_results: dict) -> Table:
    """Create table showing ToM benchmark results."""
    table = Table(title="Theory of Mind Analysis", show_header=True)
    table.add_column("Order", style="cyan")
    table.add_column("Sally-Anne", style="green")
    table.add_column("Status", style="yellow")

    progression = tom_results.get('sally_anne_progression', [])
    for order, score in enumerate(progression):
        status = "[green]PASS[/green]" if score > 0.5 else "[red]FAIL[/red]"
        table.add_row(f"Order {order}", f"{score:.3f}", status)

    return table


def create_zombie_analysis_table(zombie_results: dict) -> Table:
    """Create table showing zombie detection results."""
    table = Table(title="Zombie Detection Analysis", show_header=True)
    table.add_column("Test Type", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Status", style="yellow")

    for test_name, result in zombie_results.get('test_results', {}).items():
        score = result.get('score', 0)
        passed = result.get('passed', False)
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(test_name.replace('_', ' ').title(), f"{score:.3f}", status)

    return table


def run_evolution(num_generations: int = 10, population_size: int = 10,
                  show_plots: bool = True, verbose: bool = True):
    """
    Run ToM-NAS evolution with rich output.

    Args:
        num_generations: Number of generations to evolve
        population_size: Size of population
        show_plots: Whether to show matplotlib plots at end
        verbose: Whether to show detailed output
    """
    print_header()

    # Initialize components
    if RICH_AVAILABLE:
        console.print("\n[bold]Initializing System Components...[/bold]")
    else:
        print("\nInitializing System Components...")

    # Create world and belief network
    ontology = SoulMapOntology()
    world = SocialWorld4(num_agents=10, ontology_dim=181, num_zombies=2)
    belief_net = BeliefNetwork(num_agents=10, ontology_dim=181, max_order=5)

    # Create evolution config
    config = EvolutionConfig(
        population_size=population_size,
        num_generations=num_generations,
        elite_size=2,
        tournament_size=3,
        mutation_rate=0.1,
        crossover_rate=0.7,
        fitness_episodes=3,
        use_speciation=True,
        use_coevolution=True
    )

    # Create engine
    engine = NASEngine(config, world, belief_net)

    # Initialize evaluation suites
    zombie_suite = ZombieDetectionSuite()
    tom_suite = ToMBenchmarkSuite(input_dim=191)

    if RICH_AVAILABLE:
        console.print(f"  [green]✓[/green] Ontology: {ontology.total_dims} dimensions")
        console.print(f"  [green]✓[/green] World: {world.num_agents} agents ({sum(1 for a in world.agents if a.is_zombie)} zombies)")
        console.print(f"  [green]✓[/green] Beliefs: {belief_net.max_order}th-order ToM")
        console.print(f"  [green]✓[/green] Population: {population_size} individuals")
        console.print(f"  [green]✓[/green] Generations: {num_generations}")
    else:
        print(f"  - Ontology: {ontology.total_dims} dimensions")
        print(f"  - World: {world.num_agents} agents")
        print(f"  - Population: {population_size}")

    # Initialize population
    if RICH_AVAILABLE:
        console.print("\n[bold]Initializing Population...[/bold]")
    engine.initialize_population()

    # Track history for plotting
    history = {
        'best_fitness': [],
        'avg_fitness': [],
        'diversity': [],
        'tom_scores': [],
        'zombie_scores': []
    }

    # Evolution loop
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]Starting Evolution ({num_generations} generations)[/bold cyan]\n")
    else:
        print(f"\nStarting Evolution ({num_generations} generations)\n")

    start_time = time.time()

    for gen in range(num_generations):
        gen_start = time.time()

        # Evolve one generation
        engine.evolve_generation()

        # Get stats
        fitnesses = [ind.fitness for ind in engine.population if ind.fitness is not None]
        stats = {
            'best_fitness': max(fitnesses) if fitnesses else 0,
            'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
            'diversity': engine._calculate_diversity(),
            'mutation_rate': engine.adaptive_mutation.get_rate(),
            'species_count': engine.species_manager.get_species_count() if engine.species_manager else 1
        }

        # Record history
        history['best_fitness'].append(stats['best_fitness'])
        history['avg_fitness'].append(stats['avg_fitness'])
        history['diversity'].append(stats['diversity'])

        # Evaluate best individual with ToM and zombie tests
        if engine.best_individual:
            tom_results = tom_suite.run_full_evaluation(engine.best_individual.model)
            zombie_results = zombie_suite.run_full_evaluation(
                engine.best_individual.model, {'input_dim': 191}
            )
            history['tom_scores'].append(tom_results['overall_score'])
            history['zombie_scores'].append(1 - zombie_results['zombie_probability'])

        gen_time = time.time() - gen_start

        # Display output
        if RICH_AVAILABLE:
            console.print(f"\n[bold]Generation {gen + 1}/{num_generations}[/bold] ({gen_time:.1f}s)")
            console.print(create_stats_table(gen + 1, stats))

            if verbose and (gen + 1) % 5 == 0:
                console.print(create_population_table(engine.population))
                if engine.best_individual:
                    console.print(create_tom_analysis_table(tom_results))
                    console.print(create_zombie_analysis_table(zombie_results))
        else:
            print(f"\nGeneration {gen + 1}: Best={stats['best_fitness']:.4f}, "
                  f"Avg={stats['avg_fitness']:.4f}, Diversity={stats['diversity']:.4f}")

    total_time = time.time() - start_time

    # Final summary
    if RICH_AVAILABLE:
        console.print("\n" + "="*70)
        console.print(Panel.fit(
            f"[bold green]Evolution Complete![/bold green]\n\n"
            f"Total Time: {total_time:.1f}s\n"
            f"Best Fitness: {engine.best_individual.fitness:.4f}\n"
            f"Best Architecture: {engine.best_individual.gene.gene_dict['arch_type']}\n"
            f"Final Diversity: {stats['diversity']:.4f}",
            title="Results",
            border_style="green"
        ))
    else:
        print("\n" + "="*70)
        print("Evolution Complete!")
        print(f"Total Time: {total_time:.1f}s")
        print(f"Best Fitness: {engine.best_individual.fitness:.4f}")
        print(f"Best Architecture: {engine.best_individual.gene.gene_dict['arch_type']}")

    # Final evaluation of best individual
    if RICH_AVAILABLE:
        console.print("\n[bold]Final Evaluation of Best Individual[/bold]")

    best = engine.best_individual
    final_tom = tom_suite.run_full_evaluation(best.model)
    final_zombie = zombie_suite.run_full_evaluation(best.model, {'input_dim': 191})

    if RICH_AVAILABLE:
        console.print(create_tom_analysis_table(final_tom))
        console.print(create_zombie_analysis_table(final_zombie))

        # Architecture details
        arch_table = Table(title="Best Architecture Configuration")
        arch_table.add_column("Parameter", style="cyan")
        arch_table.add_column("Value", style="green")
        for key, value in best.gene.gene_dict.items():
            arch_table.add_row(key, str(value))
        console.print(arch_table)
    else:
        print(f"\nToM Score: {final_tom['overall_score']:.4f}")
        print(f"Max ToM Order: {final_tom['max_tom_order']}")
        print(f"Zombie Probability: {final_zombie['zombie_probability']:.4f}")

    # Show plots if available
    if show_plots and MATPLOTLIB_AVAILABLE and len(history['best_fitness']) > 1:
        create_evolution_plots(history, num_generations)

    return engine, history


def create_evolution_plots(history: dict, num_generations: int):
    """Create matplotlib plots of evolution progress."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ToM-NAS Evolution Progress', fontsize=14, fontweight='bold')

    generations = range(1, len(history['best_fitness']) + 1)

    # Fitness plot
    ax1 = axes[0, 0]
    ax1.plot(generations, history['best_fitness'], 'b-', label='Best', linewidth=2)
    ax1.plot(generations, history['avg_fitness'], 'g--', label='Average', linewidth=1.5)
    ax1.fill_between(generations, history['avg_fitness'], history['best_fitness'], alpha=0.3)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Diversity plot
    ax2 = axes[0, 1]
    ax2.plot(generations, history['diversity'], 'r-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Diversity')
    ax2.set_title('Population Diversity')
    ax2.grid(True, alpha=0.3)

    # ToM scores
    ax3 = axes[1, 0]
    if history['tom_scores']:
        ax3.plot(generations, history['tom_scores'], 'purple', linewidth=2, marker='o', markersize=4)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('ToM Score')
    ax3.set_title('Theory of Mind Performance')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Zombie detection (inverted - higher = less zombie-like)
    ax4 = axes[1, 1]
    if history['zombie_scores']:
        ax4.plot(generations, history['zombie_scores'], 'orange', linewidth=2, marker='s', markersize=4)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Human-likeness Score')
    ax4.set_title('Zombie Detection (1 = Human-like)')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = 'evolution_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    # Try to display
    try:
        plt.show()
    except:
        pass


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run ToM-NAS Evolution')
    parser.add_argument('--generations', '-g', type=int, default=10,
                        help='Number of generations (default: 10)')
    parser.add_argument('--population', '-p', type=int, default=10,
                        help='Population size (default: 10)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable matplotlib plots')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Less verbose output')

    args = parser.parse_args()

    engine, history = run_evolution(
        num_generations=args.generations,
        population_size=args.population,
        show_plots=not args.no_plots,
        verbose=not args.quiet
    )

    print("\nTo run again with different settings:")
    print(f"  python run_evolution.py -g {args.generations} -p {args.population}")


if __name__ == "__main__":
    main()
