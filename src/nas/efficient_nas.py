"""
Efficient Neural Architecture Search pipeline.

Combines zero-cost proxy pre-filtering with lightweight training
evaluation and evolutionary search for GPU-efficient NAS.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Callable, Optional, Any
import random
import copy
from dataclasses import dataclass, field
from .zero_cost import ZeroCostProxies, rank_architectures


@dataclass
class NASConfig:
    """Configuration for efficient NAS pipeline."""
    num_candidates: int = 1000      # Initial candidates to generate
    num_to_train: int = 50          # Candidates to actually train
    generations: int = 20           # Evolution generations
    population_size: int = 20       # Population per generation
    tournament_size: int = 3        # Tournament selection size
    mutation_rate: float = 0.1      # Weight mutation probability
    architecture_mutation_rate: float = 0.2  # Structural mutation probability
    max_params: int = 500000        # Maximum parameters allowed
    training_epochs: int = 5        # Epochs for lightweight training
    input_dim: int = 181            # Input dimension
    device: str = 'cpu'             # Device to use


class EfficientNAS:
    """
    Efficient Neural Architecture Search combining:
    1. Zero-cost proxy pre-filtering (fast)
    2. Lightweight training evaluation (medium)
    3. Evolutionary search (refinement)

    This pipeline minimizes GPU time by:
    - Filtering 95% of candidates with zero-cost metrics
    - Using only 5 epochs for initial training
    - Evolving the best architectures found
    """

    def __init__(self,
                 architecture_generator: Callable[[], nn.Module],
                 fitness_evaluator: Callable[[nn.Module, int], float],
                 config: Optional[NASConfig] = None):
        """
        Initialize NAS pipeline.

        Args:
            architecture_generator: Callable that returns a new random architecture
            fitness_evaluator: Callable(model, epochs) -> fitness score
            config: NAS configuration
        """
        self.arch_generator = architecture_generator
        self.fitness_eval = fitness_evaluator
        self.config = config or NASConfig()
        self.proxies = ZeroCostProxies(input_dim=self.config.input_dim)

        # History for analysis
        self.eval_history: List[Dict] = []
        self.generation_history: List[Dict] = []

    def search(self) -> List[Dict[str, Any]]:
        """
        Main search pipeline.

        Returns:
            List of top 10 architectures with their metadata
        """
        print("=" * 60)
        print("EFFICIENT NAS PIPELINE")
        print("=" * 60)

        # Stage 1: Generate candidates
        candidates = self._stage1_generate()

        # Stage 2: Zero-cost proxy ranking
        top_candidates = self._stage2_filter(candidates)

        # Stage 3: Lightweight training evaluation
        evaluated = self._stage3_train(top_candidates)

        # Stage 4: Evolutionary refinement
        final_evaluated = self._stage4_evolve(evaluated)

        print("\n" + "=" * 60)
        print("SEARCH COMPLETE")
        print("=" * 60)
        print(f"Best fitness: {final_evaluated[0]['fitness']:.4f}")
        print(f"Total architectures evaluated: {len(self.eval_history)}")

        return final_evaluated[:10]

    def _stage1_generate(self) -> List[nn.Module]:
        """Stage 1: Generate candidate architectures."""
        print(f"\n[Stage 1] Generating {self.config.num_candidates} candidate architectures...")

        candidates = []
        rejected = 0

        for i in range(self.config.num_candidates):
            try:
                arch = self.arch_generator()
                param_count = sum(p.numel() for p in arch.parameters())

                if param_count <= self.config.max_params:
                    candidates.append(arch)
                else:
                    rejected += 1
            except Exception as e:
                rejected += 1

        print(f"  Generated {len(candidates)} valid candidates")
        print(f"  Rejected {rejected} (over parameter budget or invalid)")

        return candidates

    def _stage2_filter(self, candidates: List[nn.Module]) -> List[nn.Module]:
        """Stage 2: Filter candidates using zero-cost proxies."""
        print(f"\n[Stage 2] Computing zero-cost proxy scores...")

        ranked = rank_architectures(candidates, self.proxies, verbose=False)

        # Report score distribution
        scores = [score for _, score, _ in ranked]
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"  Mean score: {sum(scores)/len(scores):.4f}")

        # Take top candidates for training
        top_indices = [idx for idx, score, _ in ranked[:self.config.num_to_train]]
        top_candidates = [candidates[i] for i in top_indices]

        print(f"  Selected top {len(top_candidates)} for training")

        return top_candidates

    def _stage3_train(self, candidates: List[nn.Module]) -> List[Dict[str, Any]]:
        """Stage 3: Lightweight training evaluation."""
        print(f"\n[Stage 3] Training and evaluating top candidates...")

        evaluated = []

        for i, arch in enumerate(candidates):
            print(f"  Training {i+1}/{len(candidates)}...", end=' ', flush=True)

            try:
                fitness = self.fitness_eval(arch, self.config.training_epochs)

                result = {
                    'architecture': arch,
                    'fitness': fitness,
                    'proxy_rank': i,
                    'generation': 0,
                    'encoding': self._encode_architecture(arch)
                }
                evaluated.append(result)
                self.eval_history.append(result)

                print(f"fitness={fitness:.4f}")
            except Exception as e:
                print(f"failed: {e}")

        # Sort by fitness
        evaluated.sort(key=lambda x: x['fitness'], reverse=True)

        return evaluated

    def _stage4_evolve(self, evaluated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 4: Evolutionary refinement."""
        print(f"\n[Stage 4] Evolutionary refinement ({self.config.generations} generations)...")

        # Initialize population with best from training
        population = [e['architecture'] for e in evaluated[:self.config.population_size]]
        fitness_map = {id(e['architecture']): e['fitness'] for e in evaluated}

        best_fitness = evaluated[0]['fitness'] if evaluated else 0

        for gen in range(self.config.generations):
            # Tournament selection for parents
            parents = self._tournament_select(population, fitness_map)

            # Create offspring through mutation
            offspring = []
            for parent in parents[:self.config.population_size // 2]:
                child = self._mutate(parent)
                offspring.append(child)

            # Evaluate offspring
            for child in offspring:
                try:
                    fitness = self.fitness_eval(child, self.config.training_epochs)

                    result = {
                        'architecture': child,
                        'fitness': fitness,
                        'generation': gen + 1,
                        'encoding': self._encode_architecture(child)
                    }
                    evaluated.append(result)
                    self.eval_history.append(result)
                    fitness_map[id(child)] = fitness

                except Exception as e:
                    continue

            # Select next generation (elitism + offspring)
            all_architectures = list(set(population + offspring))
            all_with_fitness = [(a, fitness_map.get(id(a), 0)) for a in all_architectures]
            all_with_fitness.sort(key=lambda x: x[1], reverse=True)
            population = [a for a, _ in all_with_fitness[:self.config.population_size]]

            # Track best
            gen_best = all_with_fitness[0][1] if all_with_fitness else 0
            if gen_best > best_fitness:
                best_fitness = gen_best

            self.generation_history.append({
                'generation': gen + 1,
                'best_fitness': gen_best,
                'mean_fitness': sum(f for _, f in all_with_fitness[:self.config.population_size])
                               / self.config.population_size
            })

            print(f"  Generation {gen+1}: best={gen_best:.4f}, overall_best={best_fitness:.4f}")

        # Sort all evaluated by fitness
        evaluated.sort(key=lambda x: x['fitness'], reverse=True)

        return evaluated

    def _tournament_select(self, population: List[nn.Module],
                          fitness_map: Dict[int, float]) -> List[nn.Module]:
        """Tournament selection for parents."""
        selected = []

        for _ in range(len(population)):
            tournament = random.sample(population, min(self.config.tournament_size, len(population)))
            winner = max(tournament, key=lambda x: fitness_map.get(id(x), 0))
            selected.append(winner)

        return selected

    def _mutate(self, model: nn.Module) -> nn.Module:
        """Mutate architecture."""
        child = copy.deepcopy(model)

        # Weight mutation
        with torch.no_grad():
            for param in child.parameters():
                if random.random() < self.config.mutation_rate:
                    noise = torch.randn_like(param) * 0.1
                    param.add_(noise)

        return child

    def _encode_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Encode architecture to feature dict for analysis."""
        features = {
            'num_linear': sum(1 for m in model.modules() if isinstance(m, nn.Linear)),
            'num_lstm': sum(1 for m in model.modules() if isinstance(m, nn.LSTM)),
            'num_gru': sum(1 for m in model.modules() if isinstance(m, nn.GRU)),
            'num_attention': sum(1 for m in model.modules() if isinstance(m, nn.MultiheadAttention)),
            'num_conv': sum(1 for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))),
            'total_params': sum(p.numel() for p in model.parameters()),
            'num_layers': len(list(model.modules()))
        }
        return features

    def get_analysis(self) -> Dict[str, Any]:
        """Get analysis of search results."""
        if not self.eval_history:
            return {}

        # Best architecture
        best = max(self.eval_history, key=lambda x: x['fitness'])

        # Fitness progression
        by_gen = {}
        for e in self.eval_history:
            gen = e.get('generation', 0)
            if gen not in by_gen:
                by_gen[gen] = []
            by_gen[gen].append(e['fitness'])

        gen_stats = {
            gen: {
                'mean': sum(fits) / len(fits),
                'max': max(fits),
                'min': min(fits)
            }
            for gen, fits in by_gen.items()
        }

        # Correlation between architecture features and fitness
        # (simplified - just track which features are common in top architectures)
        top_20 = sorted(self.eval_history, key=lambda x: x['fitness'], reverse=True)[:20]
        feature_sums = {}
        for e in top_20:
            for k, v in e.get('encoding', {}).items():
                if k not in feature_sums:
                    feature_sums[k] = 0
                feature_sums[k] += v

        return {
            'best_fitness': best['fitness'],
            'best_architecture_features': best.get('encoding', {}),
            'total_evaluations': len(self.eval_history),
            'generation_stats': gen_stats,
            'top_20_avg_features': {k: v/20 for k, v in feature_sums.items()}
        }


def create_default_generator(input_dim: int = 181,
                             output_dim: int = 181,
                             hidden_range: Tuple[int, int] = (64, 256),
                             layers_range: Tuple[int, int] = (1, 4)) -> Callable[[], nn.Module]:
    """
    Create a default architecture generator.

    Generates random LSTM-based architectures for ToM tasks.
    """
    def generator() -> nn.Module:
        # Pick hidden dim from range, then round to multiple of 8 for attention compatibility
        raw_hidden = random.randint(*hidden_range)
        hidden_dim = ((raw_hidden + 7) // 8) * 8  # Round up to nearest multiple of 8

        num_layers = random.randint(*layers_range)
        dropout = random.uniform(0, 0.3)
        use_attention = random.random() > 0.5
        bidirectional = random.random() > 0.7

        class GeneratedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                )
                lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

                if use_attention:
                    # Calculate num_heads that divides evenly
                    num_heads = 4
                    while lstm_out_dim % num_heads != 0 and num_heads > 1:
                        num_heads //= 2
                    self.attention = nn.MultiheadAttention(lstm_out_dim, num_heads=num_heads, batch_first=True)
                else:
                    self.attention = None

                self.output = nn.Linear(lstm_out_dim, output_dim)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                if self.attention is not None:
                    lstm_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                return self.output(lstm_out)

        return GeneratedModel()

    return generator
