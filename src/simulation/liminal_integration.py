"""
Liminal Environment Integration

Bridges the ToM-NAS simulation module with the existing Liminal Architectures
game environment, enabling:
- Agent translation between systems
- World state synchronization
- Benchmark embedding in Liminal context
- Unified POET co-evolution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import random
import numpy as np

# Import from our new simulation module
from .integrated_agent import IntegratedAgent, AgentMemory, NestedBelief
from .world import SimulationWorld, Action, ActionType, Location
from .benchmark_embedding import EmbeddedBenchmark, ToMBenchEmbed, SOTOPIAEmbed
from .poet_engine import POETEngine, AgentArchitectureGenome
from .visualization import TerminalRenderer, create_renderer

# Import from existing liminal module
try:
    from ..liminal import (
        LiminalEnvironment,
        GameState as LiminalGameState,
        SoulMap,
        RealmType,
        REALMS,
    )
    from ..liminal.npcs.base_npc import BaseNPC, NPCState
    from ..liminal.psychosocial_coevolution import (
        PsychosocialCoevolutionEngine,
        SocialNetwork,
    )
    from ..liminal.narrative_emergence import NarrativeEmergenceSystem
    LIMINAL_AVAILABLE = True
except ImportError:
    LIMINAL_AVAILABLE = False


# Import from taxonomy module
try:
    from ..taxonomy import (
        PsychosocialProfile,
        SuccessState,
        InstitutionalContext,
    )
    TAXONOMY_AVAILABLE = True
except ImportError:
    TAXONOMY_AVAILABLE = False


class IntegrationMode(Enum):
    """Modes for environment integration."""
    SIMULATION_ONLY = "simulation"      # Use only new simulation
    LIMINAL_ONLY = "liminal"            # Use only liminal environment
    HYBRID = "hybrid"                   # Use both with sync
    LIMINAL_EMBEDDED = "embedded"       # Embed simulation in liminal


@dataclass
class IntegrationConfig:
    """Configuration for environment integration."""
    mode: IntegrationMode = IntegrationMode.HYBRID
    sync_frequency: int = 10           # Steps between syncs
    share_npcs: bool = True            # Share NPCs between systems
    unified_time: bool = True          # Use unified tick counter
    benchmark_in_liminal: bool = True  # Embed benchmarks in liminal realms
    coevolution_engine: str = "both"   # Which POET engine to use


@dataclass
class UnifiedAgent:
    """Agent representation that works in both systems."""
    id: int
    name: str

    # Simulation module representation
    integrated_agent: Optional[IntegratedAgent] = None

    # Liminal representation
    liminal_npc: Optional[Any] = None  # BaseNPC when available
    soul_map: Optional[Any] = None     # SoulMap when available

    # Shared state
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    realm: Optional[str] = None
    is_zombie: bool = False

    # Performance tracking
    tom_accuracy: float = 0.0
    fitness: float = 0.5

    def sync_to_liminal(self):
        """Sync state from integrated agent to liminal NPC."""
        if self.integrated_agent and self.liminal_npc:
            # Sync position
            self.liminal_npc.position = (self.position[0], self.position[1])

            # Sync emotional state from psychosocial profile
            if hasattr(self.integrated_agent, 'profile'):
                profile = self.integrated_agent.profile
                if hasattr(profile, 'layer1'):
                    emotional_range = getattr(profile.layer1, 'emotional_range', 0.5)
                    self.liminal_npc.emotional_state = (
                        "calm" if emotional_range < 0.3 else
                        "neutral" if emotional_range < 0.7 else
                        "excited"
                    )

            # Sync zombie status
            self.liminal_npc.is_zombie = self.is_zombie

    def sync_from_liminal(self):
        """Sync state from liminal NPC to integrated agent."""
        if self.liminal_npc and self.integrated_agent:
            # Sync position
            pos = self.liminal_npc.position
            self.position = (pos[0], pos[1], 0.0)

            # Sync soul map to psychosocial profile
            if hasattr(self.liminal_npc, 'soul_map') and self.soul_map:
                soul_map = self.liminal_npc.soul_map
                if hasattr(self.integrated_agent, 'profile'):
                    self._translate_soul_map_to_profile(soul_map)

    def _translate_soul_map_to_profile(self, soul_map):
        """Translate SoulMap dimensions to PsychosocialProfile."""
        if not TAXONOMY_AVAILABLE:
            return

        profile = self.integrated_agent.profile

        # Map cognitive dimensions (soul_map indices 0-14)
        if hasattr(profile, 'layer3'):
            profile.layer3.fluid_intelligence = soul_map.get_dimension(0)  # reasoning
            profile.layer3.working_memory = soul_map.get_dimension(1)     # attention

        # Map emotional dimensions (soul_map indices 15-29)
        if hasattr(profile, 'layer1'):
            profile.layer1.emotional_range = soul_map.get_dimension(15)   # intensity
            profile.layer1.emotional_stability = soul_map.get_dimension(17)  # stability

        # Map motivational dimensions (soul_map indices 30-44)
        if hasattr(profile, 'layer2'):
            profile.layer2.intrinsic_motivation = soul_map.get_dimension(30)  # drives


@dataclass
class UnifiedWorld:
    """World representation that bridges both systems."""
    simulation_world: Optional[SimulationWorld] = None
    liminal_env: Optional[Any] = None  # LiminalEnvironment when available

    # Unified state
    tick: int = 0
    agents: Dict[int, UnifiedAgent] = field(default_factory=dict)
    active_benchmarks: List[EmbeddedBenchmark] = field(default_factory=list)

    # Realm mapping
    realm_to_zone: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize realm-to-zone mapping."""
        if not self.realm_to_zone:
            # Map liminal realms to simulation world zones
            self.realm_to_zone = {
                'peregrine': (0, 0, 25, 25),      # Top-left quarter
                'ministry': (25, 0, 50, 25),      # Top-right quarter
                'spleen_towns': (0, 25, 25, 50),  # Bottom-left quarter
                'city': (25, 25, 50, 50),         # Bottom-right quarter
                'hollow': (10, 10, 40, 40),       # Center (overlapping)
                'nothing': (45, 45, 50, 50),      # Corner (dangerous)
            }


class EnvironmentBridge:
    """
    Bridge between ToM-NAS simulation and Liminal Architectures.

    Enables running experiments that leverage both:
    - The detailed 3D simulation with benchmarks and POET
    - The rich psychological game environment with NPCs and mechanics
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()

        # Initialize components based on mode
        self.simulation_world: Optional[SimulationWorld] = None
        self.liminal_env: Optional[Any] = None
        self.unified_world = UnifiedWorld()

        # Co-evolution engines
        self.simulation_poet: Optional[POETEngine] = None
        self.liminal_coevo: Optional[Any] = None

        # Benchmark systems
        self.tombench = ToMBenchEmbed()
        self.sotopia = SOTOPIAEmbed()

        # Visualization
        self.renderer = create_renderer('standard')

        # Narrative tracking
        self.narrative_system: Optional[Any] = None

        # Statistics
        self.integration_stats = {
            'syncs': 0,
            'agent_transfers': 0,
            'benchmark_embeddings': 0,
            'realm_transitions': 0,
        }

    def initialize(self,
                   world_size: Tuple[int, int, int] = (50, 50, 1),
                   population_size: int = 20,
                   num_zombies: int = 2,
                   include_heroes: bool = True,
                   seed: Optional[int] = None) -> bool:
        """Initialize the integrated environment."""

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize simulation world
        self.simulation_world = SimulationWorld(
            width=world_size[0],
            height=world_size[1],
            depth=world_size[2],
        )

        # Initialize liminal environment if available
        if LIMINAL_AVAILABLE and self.config.mode != IntegrationMode.SIMULATION_ONLY:
            self.liminal_env = LiminalEnvironment(
                population_size=population_size,
                include_heroes=include_heroes,
                seed=seed,
            )

            # Initialize narrative system
            self.narrative_system = NarrativeEmergenceSystem()

            # Initialize co-evolution
            self.liminal_coevo = PsychosocialCoevolutionEngine()

        # Create unified agents
        self._create_unified_agents(population_size, num_zombies, include_heroes)

        # Initialize POET engine
        self.simulation_poet = POETEngine(population_size=10)

        return True

    def _create_unified_agents(self,
                               population_size: int,
                               num_zombies: int,
                               include_heroes: bool):
        """Create agents that exist in both systems."""
        from ..taxonomy.sampling import AgentSampler

        sampler = AgentSampler()

        for i in range(population_size):
            is_zombie = i < num_zombies

            # Create integrated agent
            integrated = sampler.sample_agent(i)
            integrated.is_zombie = is_zombie

            # Create unified wrapper
            unified = UnifiedAgent(
                id=i,
                name=f"Agent_{i}",
                integrated_agent=integrated,
                is_zombie=is_zombie,
                position=integrated.position if hasattr(integrated, 'position') else (0, 0, 0),
            )

            # Link to liminal NPC if available
            if LIMINAL_AVAILABLE and self.liminal_env:
                npc_list = list(self.liminal_env.npcs.values())
                if i < len(npc_list):
                    npc = npc_list[i]
                    unified.liminal_npc = npc
                    unified.soul_map = npc.soul_map
                    unified.realm = npc.current_realm.value

            self.unified_world.agents[i] = unified

    def step(self, actions: Optional[Dict[int, Any]] = None) -> Dict[str, Any]:
        """Execute one step in the unified environment."""
        self.unified_world.tick += 1
        results = {
            'tick': self.unified_world.tick,
            'simulation_results': None,
            'liminal_results': None,
            'benchmark_results': [],
            'narrative_events': [],
        }

        # Step simulation world
        if self.simulation_world:
            sim_actions = self._translate_actions_to_simulation(actions)

            # Build agent dict for simulation
            sim_agents = {
                aid: ua.integrated_agent
                for aid, ua in self.unified_world.agents.items()
                if ua.integrated_agent is not None
            }

            results['simulation_results'] = self.simulation_world.step(sim_actions)

        # Step liminal environment
        if LIMINAL_AVAILABLE and self.liminal_env:
            lim_action = self._translate_actions_to_liminal(actions)
            lim_result = self.liminal_env.step(lim_action or {'type': 'wait'})
            results['liminal_results'] = {
                'tick': lim_result.game_state.tick,
                'reward': lim_result.reward,
                'done': lim_result.done,
                'instability': lim_result.game_state.instability,
            }

            # Check for narrative events
            if self.narrative_system:
                events = self.narrative_system.detect_patterns(
                    self.liminal_env.get_all_npcs(),
                    self.unified_world.tick
                )
                results['narrative_events'] = events

        # Sync if needed
        if self.config.mode == IntegrationMode.HYBRID:
            if self.unified_world.tick % self.config.sync_frequency == 0:
                self._sync_agents()
                self.integration_stats['syncs'] += 1

        # Check active benchmarks
        for benchmark in self.unified_world.active_benchmarks:
            if benchmark.is_active():
                bench_result = benchmark.evaluate(self.unified_world.agents)
                if bench_result:
                    results['benchmark_results'].append(bench_result)

        return results

    def _translate_actions_to_simulation(self,
                                         actions: Optional[Dict[int, Any]]) -> Dict[int, Action]:
        """Translate unified actions to simulation format."""
        if not actions:
            return {}

        sim_actions = {}
        for agent_id, action in actions.items():
            if isinstance(action, Action):
                sim_actions[agent_id] = action
            elif isinstance(action, dict):
                action_type = ActionType(action.get('type', 'wait'))
                sim_actions[agent_id] = Action(
                    agent_id=agent_id,
                    action_type=action_type,
                    target_id=action.get('target_id'),
                    target_location=action.get('location'),
                    payload=action.get('payload', {}),
                )

        return sim_actions

    def _translate_actions_to_liminal(self,
                                      actions: Optional[Dict[int, Any]]) -> Optional[Dict]:
        """Translate unified actions to liminal format."""
        if not actions:
            return None

        # Use first action as player action (liminal is single-player)
        if actions:
            first_action = next(iter(actions.values()))
            if isinstance(first_action, dict):
                return first_action
            elif hasattr(first_action, 'action_type'):
                return {
                    'type': first_action.action_type.value,
                    'target_id': first_action.target_id,
                }

        return {'type': 'wait'}

    def _sync_agents(self):
        """Synchronize agent states between systems."""
        for unified_agent in self.unified_world.agents.values():
            if self.config.mode == IntegrationMode.HYBRID:
                # Bidirectional sync
                unified_agent.sync_from_liminal()
                unified_agent.sync_to_liminal()
            elif self.config.mode == IntegrationMode.LIMINAL_EMBEDDED:
                # Liminal is primary
                unified_agent.sync_from_liminal()

    def embed_benchmark(self,
                        benchmark_type: str = 'false_belief',
                        realm: Optional[str] = None) -> EmbeddedBenchmark:
        """Embed a benchmark scenario in the unified world."""

        # Select agents for benchmark
        available_agents = [
            ua.integrated_agent
            for ua in self.unified_world.agents.values()
            if ua.integrated_agent is not None
        ]

        if len(available_agents) < 2:
            raise ValueError("Need at least 2 agents for benchmark")

        # Create benchmark
        if benchmark_type in ['false_belief', 'faux_pas', 'second_order']:
            scenario = self.tombench.create_scenario(
                benchmark_type,
                available_agents[:3]
            )
        elif benchmark_type in ['cooperation', 'competition', 'negotiation']:
            scenario = self.sotopia.create_scenario(
                benchmark_type,
                available_agents[:4],
                complexity=0.5
            )
        else:
            scenario = self.tombench.create_scenario(
                'false_belief',
                available_agents[:2]
            )

        embedded = self.tombench.embed_scenario(
            scenario,
            self.simulation_world,
            realm_zone=self._get_realm_zone(realm),
        )

        self.unified_world.active_benchmarks.append(embedded)
        self.integration_stats['benchmark_embeddings'] += 1

        return embedded

    def _get_realm_zone(self, realm: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
        """Get simulation zone for a realm."""
        if realm:
            return self.unified_world.realm_to_zone.get(realm)
        return None

    def run_poet_generation(self) -> Dict[str, Any]:
        """Run one generation of POET co-evolution."""
        results = {
            'simulation_poet': None,
            'liminal_coevo': None,
        }

        # Run simulation POET
        if self.simulation_poet:
            def evaluator(genome):
                # Create agent from genome and evaluate
                return self._evaluate_genome(genome)

            results['simulation_poet'] = self.simulation_poet.evolve_generation(evaluator)

        # Run liminal co-evolution
        if LIMINAL_AVAILABLE and self.liminal_coevo:
            # Build social network from current agents
            network = self._build_social_network()
            coevo_result = self.liminal_coevo.evolve(
                network,
                epochs=1
            )
            results['liminal_coevo'] = coevo_result

        return results

    def _evaluate_genome(self, genome: AgentArchitectureGenome) -> float:
        """Evaluate an agent architecture genome."""
        # Create test agents from genome
        test_agents = []
        for i in range(5):
            agent = genome.instantiate(i)
            test_agents.append(agent)

        # Run quick evaluation
        total_score = 0.0
        for agent in test_agents:
            if hasattr(agent, 'success') and hasattr(agent, 'profile'):
                weights = agent.profile.get_success_weights()
                total_score += agent.success.compute_fitness(weights)

        return total_score / len(test_agents)

    def _build_social_network(self):
        """Build social network from current agents."""
        if not LIMINAL_AVAILABLE:
            return None

        network = SocialNetwork()

        for unified in self.unified_world.agents.values():
            if unified.integrated_agent:
                network.add_node(
                    unified.id,
                    unified.integrated_agent.profile if hasattr(unified.integrated_agent, 'profile') else None
                )

        # Add edges based on recent interactions
        for ua1 in self.unified_world.agents.values():
            if ua1.integrated_agent and hasattr(ua1.integrated_agent, 'memory'):
                memory = ua1.integrated_agent.memory
                if hasattr(memory, 'relational'):
                    for other_id, relationship in memory.relational.items():
                        if other_id in self.unified_world.agents:
                            network.add_edge(ua1.id, other_id, relationship)

        return network

    def get_visualization(self, mode: str = 'standard') -> str:
        """Get visual representation of current state."""
        if mode != self.renderer.__class__.__name__.lower().replace('renderer', ''):
            self.renderer = create_renderer(mode)

        # Build agent dict for renderer
        agents_for_render = {}
        for aid, ua in self.unified_world.agents.items():
            if ua.integrated_agent:
                ua.integrated_agent.position = ua.position
                agents_for_render[aid] = ua.integrated_agent

        return self.renderer.render_world(
            self.simulation_world,
            agents_for_render,
            step=self.unified_world.tick,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            'unified': {
                'tick': self.unified_world.tick,
                'total_agents': len(self.unified_world.agents),
                'zombies': sum(1 for ua in self.unified_world.agents.values() if ua.is_zombie),
                'active_benchmarks': len(self.unified_world.active_benchmarks),
            },
            'integration': self.integration_stats,
        }

        # Add simulation stats
        if self.simulation_world:
            stats['simulation'] = {
                'world_size': (
                    self.simulation_world.width,
                    self.simulation_world.height,
                    self.simulation_world.depth,
                ),
                'resources': len(self.simulation_world.resources),
            }

        # Add liminal stats
        if LIMINAL_AVAILABLE and self.liminal_env:
            stats['liminal'] = self.liminal_env.get_statistics()

        # Agent performance
        tom_scores = []
        fitness_scores = []
        for ua in self.unified_world.agents.values():
            if ua.integrated_agent:
                if hasattr(ua.integrated_agent, 'tom_reasoner'):
                    tom_scores.append(ua.integrated_agent.tom_reasoner.k_level)
                if hasattr(ua.integrated_agent, 'success'):
                    weights = ua.integrated_agent.profile.get_success_weights()
                    fitness_scores.append(ua.integrated_agent.success.compute_fitness(weights))

        if tom_scores:
            stats['performance'] = {
                'avg_tom_level': sum(tom_scores) / len(tom_scores),
                'max_tom_level': max(tom_scores),
                'avg_fitness': sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0,
            }

        return stats

    def save_state(self, filepath: str):
        """Save current state to file."""
        import json

        state = {
            'tick': self.unified_world.tick,
            'config': {
                'mode': self.config.mode.value,
                'sync_frequency': self.config.sync_frequency,
            },
            'agents': {
                str(aid): {
                    'name': ua.name,
                    'position': ua.position,
                    'is_zombie': ua.is_zombie,
                    'realm': ua.realm,
                    'fitness': ua.fitness,
                }
                for aid, ua in self.unified_world.agents.items()
            },
            'integration_stats': self.integration_stats,
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load state from file."""
        import json

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.unified_world.tick = state['tick']
        self.integration_stats = state.get('integration_stats', self.integration_stats)

        # Restore agent positions
        for aid_str, agent_data in state.get('agents', {}).items():
            aid = int(aid_str)
            if aid in self.unified_world.agents:
                ua = self.unified_world.agents[aid]
                ua.position = tuple(agent_data['position'])
                ua.is_zombie = agent_data['is_zombie']
                ua.realm = agent_data.get('realm')
                ua.fitness = agent_data.get('fitness', 0.5)


def create_integrated_environment(
    mode: str = 'hybrid',
    world_size: Tuple[int, int, int] = (50, 50, 1),
    population_size: int = 20,
    num_zombies: int = 2,
    seed: Optional[int] = None
) -> EnvironmentBridge:
    """
    Factory function to create an integrated environment.

    Args:
        mode: 'simulation', 'liminal', 'hybrid', or 'embedded'
        world_size: (width, height, depth) of simulation world
        population_size: Number of agents
        num_zombies: Number of zombie agents
        seed: Random seed

    Returns:
        Configured EnvironmentBridge
    """
    config = IntegrationConfig(
        mode=IntegrationMode(mode),
    )

    bridge = EnvironmentBridge(config)
    bridge.initialize(
        world_size=world_size,
        population_size=population_size,
        num_zombies=num_zombies,
        seed=seed,
    )

    return bridge


# Demo function
def demo_integration():
    """Demonstrate the environment integration."""
    print("Creating integrated environment...")

    bridge = create_integrated_environment(
        mode='simulation',  # Use simulation only for demo
        world_size=(30, 30, 1),
        population_size=10,
        num_zombies=2,
        seed=42,
    )

    print(f"Initialized with {len(bridge.unified_world.agents)} agents")

    # Run a few steps
    for i in range(5):
        results = bridge.step()
        print(f"Step {results['tick']}: sim={results['simulation_results'] is not None}")

    # Embed a benchmark
    try:
        benchmark = bridge.embed_benchmark('false_belief')
        print(f"Embedded benchmark: {benchmark.scenario.name}")
    except Exception as e:
        print(f"Benchmark embedding: {e}")

    # Get visualization
    viz = bridge.get_visualization()
    print("\nVisualization:")
    print(viz)

    # Get statistics
    stats = bridge.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    demo_integration()
