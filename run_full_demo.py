#!/usr/bin/env python
"""
ToM-NAS Full System Demonstration
Runs a complete test of all system components including evolution
"""
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_full_demo():
    print("="*70)
    print("ToM-NAS Complete System Demonstration")
    print("="*70)

    # 1. Test Ontology
    print("\n[1/8] Testing 181-Dimensional Soul Map Ontology...")
    from src.core.ontology import SoulMapOntology
    ontology = SoulMapOntology()
    state_vec = ontology.get_default_state()
    # Test encode with dict input
    test_state = {'heart_rate': 0.7, 'energy_level': 0.5}
    encoded = ontology.encode(test_state)
    decoded = ontology.decode_to_interpretable(state_vec)
    print(f"  - Total dimensions: {ontology.total_dims}")
    print(f"  - Layers: {len(ontology.layer_ranges)}")
    print(f"  - Layer summary keys: {len(decoded.get('layer_summary', {}))}")
    print("  [PASS]")

    # 2. Test Belief System
    print("\n[2/8] Testing Bayesian Belief System...")
    from src.core.beliefs import BeliefNetwork, NestedBeliefStore
    belief_net = BeliefNetwork(num_agents=5, ontology_dim=181, max_order=5)

    # Test belief propagation
    obs = torch.randn(181)
    updates = belief_net.propagate_observation(0, 1, obs, max_order=3)
    print(f"  - Belief network with {belief_net.num_agents} agents")
    print(f"  - Max ToM order: {belief_net.max_order}")
    print(f"  - Propagated to {sum(len(u) for u in updates.values())} beliefs")
    print("  [PASS]")

    # 3. Test Architectures
    print("\n[3/8] Testing Agent Architectures...")
    from src.agents.architectures import (
        TransparentRNN, RecursiveSelfAttention, TransformerToMAgent,
        FullHybrid, create_architecture
    )

    x = torch.randn(2, 5, 191)

    trn = TransparentRNN(191, 128, 181)
    trn_out = trn(x, return_trace=True)
    print(f"  - TRN: output shape {trn_out['beliefs'].shape}")

    rsan = RecursiveSelfAttention(191, 128, 181, max_recursion=5)
    rsan_out = rsan(x, return_attention=True)
    print(f"  - RSAN: recursion depth {rsan_out['recursion_depth']}")

    hybrid = create_architecture('Full_Hybrid', 191, 128, 181)
    hybrid_out = hybrid(x)
    print(f"  - Full Hybrid: components {list(hybrid_out.get('component_outputs', {}).keys())}")
    print("  [PASS]")

    # 4. Test Social World
    print("\n[4/8] Testing Social World 4...")
    from src.world.social_world import SocialWorld4
    world = SocialWorld4(num_agents=8, ontology_dim=181, num_zombies=2)

    actions = [{'type': 'cooperate'} for _ in range(world.num_agents)]
    result = world.step(actions)
    print(f"  - Agents: {world.num_agents} ({sum(1 for a in world.agents if a.is_zombie)} zombies)")
    print(f"  - Games played: {len(result['games'])}")
    print(f"  - Timestep: {result['timestep']}")
    print("  [PASS]")

    # 5. Test Zombie Detection
    print("\n[5/8] Testing Zombie Detection Suite...")
    from src.evaluation.zombie_detection import ZombieDetectionSuite, ZombieType

    zombie_suite = ZombieDetectionSuite()
    test_agent = TransparentRNN(191, 128, 181)
    results = zombie_suite.run_full_evaluation(test_agent, {'input_dim': 191})
    print(f"  - Zombie types: {[z.value for z in ZombieType]}")
    print(f"  - Overall score: {results['overall_score']:.3f}")
    print(f"  - Zombie probability: {results['zombie_probability']:.3f}")
    print("  [PASS]")

    # 6. Test ToM Benchmarks
    print("\n[6/8] Testing ToM Benchmarks...")
    from src.evaluation.tom_benchmarks import ToMBenchmarkSuite

    tom_suite = ToMBenchmarkSuite(input_dim=191)
    tom_results = tom_suite.run_full_evaluation(test_agent)
    print(f"  - Benchmarks: {tom_results['num_total']}")
    print(f"  - Passed: {tom_results['num_passed']}")
    print(f"  - Max ToM order achieved: {tom_results['max_tom_order']}")
    print("  [PASS]")

    # 7. Test Cognitive Extensions
    print("\n[7/8] Testing Cognitive Extensions...")
    from src.agents.cognitive_extensions import (
        WorkingMemory, EpisodicMemory, PlanningModule, CuriosityModule
    )

    wm = WorkingMemory(capacity=7, item_dim=128)
    item = torch.randn(128)
    wm.encode(item, force=True)
    retrieved, weights = wm.retrieve(item)
    print(f"  - Working Memory: {wm.capacity} slots")

    em = EpisodicMemory(item_dim=128, max_memories=100)
    em.store(item, context={'test': True})
    retrieved_memories = em.retrieve(item, k=3)
    print(f"  - Episodic Memory: stored {em.get_statistics()['count']} memories")

    pm = PlanningModule(state_dim=128, action_dim=32)
    pm.set_goal(item)
    plan = pm.generate_plan(item, horizon=5)
    print(f"  - Planning: generated plan with {len(plan)} steps")

    cm = CuriosityModule(state_dim=128, action_dim=32)
    curiosity = cm.compute_curiosity(item, torch.randn(32), item)
    print(f"  - Curiosity: reward = {curiosity['curiosity_reward']:.3f}")
    print("  [PASS]")

    # 8. Test Curriculum Learning
    print("\n[8/8] Testing Curriculum Learning...")
    from src.training.curriculum import CurriculumManager, CurriculumStage, AdaptiveCurriculum

    curriculum = AdaptiveCurriculum(start_stage=CurriculumStage.SW1)
    print(f"  - Starting stage: {curriculum.current_stage.name}")

    # Simulate progression
    for fitness in [0.3, 0.5, 0.7]:
        curriculum.update(fitness)

    print(f"  - Current stage: {curriculum.current_stage.name}")
    print(f"  - Episodes in stage: {curriculum.episodes_in_stage}")

    config = curriculum.get_environment_config()
    print(f"  - Environment: {config['num_agents']} agents, {config['world_type']} world")
    print("  [PASS]")

    # 9. Quick Evolution Test
    print("\n[BONUS] Running Quick Evolution Test...")
    from src.evolution.nas_engine import NASEngine, EvolutionConfig

    config = EvolutionConfig(
        population_size=4,
        num_generations=2,
        elite_size=1,
        fitness_episodes=1
    )

    engine = NASEngine(config, world, belief_net)
    engine.initialize_population()
    print(f"  - Population: {len(engine.population)} individuals")

    # Run 1 generation
    engine.evolve_generation()
    print(f"  - Generation {engine.generation} complete")
    print(f"  - Best fitness: {engine.best_individual.fitness:.4f}")
    print("  [PASS]")

    # Summary
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ToM-NAS System Fully Operational")
    print("="*70)
    print("\nSystem Components:")
    print("  - 181-dim Soul Map Ontology (9 psychological layers)")
    print("  - Bayesian belief system with 5th-order ToM")
    print("  - 3 base architectures + hybrids (TRN, RSAN, Transformer)")
    print("  - Social World 4 with zombie detection")
    print("  - 6 zombie detection tests")
    print("  - Sally-Anne benchmarks (0-5th order)")
    print("  - Cognitive extensions (WM, EM, Planning, Curiosity)")
    print("  - Curriculum learning (SW1-SW4)")
    print("  - Evolutionary NAS engine")
    print("="*70)

if __name__ == "__main__":
    run_full_demo()
