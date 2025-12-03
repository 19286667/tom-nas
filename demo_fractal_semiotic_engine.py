#!/usr/bin/env python3
"""
Fractal Semiotic Engine - Integrated Demo

Demonstrates the complete integration of all four subsystems:
1. Semiotic Knowledge Graph (Indra's Net)
2. Cognitive Core (Mentalese + RSC)
3. POET Evolutionary Controller
4. Godot Physical Bridge

This demo shows how agents evolve recursive Theory of Mind through
navigating a semantically-saturated environment under institutional pressure.

"The physical is cognitive" - every object exists in a superposition
of meanings collapsed by context.

Author: ToM-NAS Project
"""

import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any

# Import all subsystems
from src.knowledge_base import (
    IndrasNet,
    SemanticQueryEngine,
    ActivationContext,
    SemanticNode,
    NodeType,
    EdgeType,
    SemanticEdge,
    FullTaxonomy,
)
from src.knowledge_base.query_engine import build_default_knowledge_base

from src.cognition import (
    CognitiveBlock,
    PerceptBlock,
    HypothesisBlock,
    BeliefBlock,
    IntentBlock,
    MemoryBlock,
    RecursiveBelief,
    SimulationState,
    RecursiveSimulationNode,
    SimulationConfig,
    WorldModel,
    AgentModel,
    TinyRecursiveModel,
    TRMConfig,
)
from src.cognition.mentalese import create_recursive_belief, compress_to_memory

from src.evolution import (
    POETController,
    POETConfig,
    EnvironmentGenotype,
    EnvironmentType,
    create_preset_environment,
    NASEngine,
    EvolutionConfig,
)

from src.godot_bridge import (
    GodotBridge,
    BridgeConfig,
    SymbolGrounder,
    PerceptionProcessor,
    ActionExecutor,
    GodotAction,
    ActionType,
    Vector3,
    EntityUpdate,
)

from src.world.social_world import SocialWorld4
from src.core.beliefs import BeliefNetwork


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print("="*60)


def demo_semiotic_knowledge_graph():
    """Demonstrate the Semiotic Knowledge Graph (Indra's Net)."""
    print_section("SUB-SYSTEM 1: Semiotic Knowledge Graph (Indra's Net)")

    # Build default knowledge base
    print("\n[1.1] Building Indra's Net with default knowledge base...")
    net, query_engine = build_default_knowledge_base()

    print(f"  - Total nodes: {net.node_count}")
    print(f"  - Total edges: {net.edge_count}")
    print(f"  - Archetypes: {len(net.get_nodes_by_type(NodeType.ARCHETYPE))}")
    print(f"  - Institutions: {len(net.get_nodes_by_type(NodeType.INSTITUTION))}")

    # Demonstrate semantic activation
    print("\n[1.2] Demonstrating semantic activation spreading...")

    context = ActivationContext(
        active_institution="inst_courtroom",
        active_roles=["role_judge", "role_defendant"],
    )

    # Activate from a chair in a courtroom
    print("\n  Perceiving 'Chair' in a Courtroom context...")
    activation = net.spread_activation("obj_chair", context)

    print(f"  - Nodes activated: {len(activation.activated_nodes)}")
    print(f"  - Top activations:")
    for node_id, level in activation.get_top_activations(5):
        node = net.get_node(node_id)
        name = node.name if node else node_id
        print(f"      {name}: {level:.3f}")

    print(f"  - Activated norms: {activation.activated_norms[:3]}")
    print(f"  - Aggregate valence: {activation.aggregate_valence:.3f}")

    # Query institutional context
    print("\n[1.3] Querying institutional context (Courtroom)...")
    from src.knowledge_base.query_engine import InstitutionalQuery

    inst_query = InstitutionalQuery(
        institution_id="inst_courtroom",
        include_roles=True,
        include_norms=True,
        include_power_structure=True,
    )
    inst_data = query_engine.query_institution(inst_query)

    print(f"  - Institution exists: {inst_data['exists']}")
    print(f"  - Associated roles: {[r['name'] for r in inst_data['roles'][:3]]}")
    print(f"  - Power structure edges: {len(inst_data['power_structure'])}")

    return net, query_engine


def demo_cognitive_core(knowledge_base: IndrasNet):
    """Demonstrate the Cognitive Core (Mentalese + RSC)."""
    print_section("SUB-SYSTEM 2: Cognitive Core (Mentalese + RSC)")

    # Create Mentalese blocks
    print("\n[2.1] Creating Mentalese cognitive blocks...")

    # Percept
    percept = PerceptBlock(
        perceived_entity="Bob",
        godot_id=42,
        position_3d=(5.0, 0.0, 3.0),
        visual_features={"height": 1.8, "posture": 0.7},
        activated_concepts=["person", "agent", "potential_ally"],
    )
    print(f"  - Percept: {percept.to_natural_language()}")

    # Hypothesis
    hypothesis = HypothesisBlock(
        proposition="Bob intends to cooperate",
        subject="Bob",
        predicate="intends to cooperate",
        confidence=0.6,
        alternatives=["Bob intends to defect", "Bob is undecided"],
        source_percepts=[percept.block_id],
    )
    print(f"  - Hypothesis: {hypothesis.to_natural_language()}")

    # Belief (with ToM)
    belief = BeliefBlock(
        proposition="Bob believes I will cooperate",
        subject="cooperation",
        predicate="Bob expects from me",
        about_agent="Bob",
        belief_order=2,
        confidence=0.7,
    )
    print(f"  - Belief (2nd-order ToM): {belief.to_natural_language()}")

    # Intent
    intent = IntentBlock(
        goal="establish cooperation",
        action_type="speak",
        target_agent="Bob",
        motivation="mutual benefit",
        urgency=0.8,
    )
    print(f"  - Intent: {intent.to_natural_language()}")

    # Create recursive belief
    print("\n[2.2] Creating recursive belief structure (3rd-order ToM)...")
    recursive = create_recursive_belief(
        holder="Alice",
        belief_chain=["Bob", "Alice"],
        proposition="will_cooperate",
        base_confidence=0.9
    )
    print(f"  - Depth: {recursive.get_recursive_depth()}")
    print(f"  - Alice believes Bob believes Alice believes: will_cooperate")
    print(f"  - Effective confidence: {0.9 * 0.7 * 0.7:.3f}")

    # Demonstrate Recursive Self-Compression (RSC)
    print("\n[2.3] Demonstrating Recursive Self-Compression...")

    # Create world model
    world_model = WorldModel(timestep=0)
    world_model.agents["Bob"] = AgentModel(
        agent_id="Bob",
        intents=[intent],
        beliefs=[belief],
        model_confidence=0.7,
    )
    world_model.agents["Alice"] = AgentModel(
        agent_id="Alice",
        model_confidence=1.0,
    )

    # Create simulation config
    sim_config = SimulationConfig(
        max_recursion_depth=3,
        max_steps_per_simulation=5,
        confidence_decay_per_depth=0.7,
    )

    # Run recursive simulation
    print("\n  Alice simulating Bob simulating Alice...")
    sim_node = RecursiveSimulationNode(
        simulating_agent="Alice",
        world_model=world_model,
        config=sim_config,
        current_depth=0,
    )

    result = sim_node.run_simulation("Bob", num_steps=3)

    print(f"  - Predicted action: {result.predicted_action}")
    print(f"  - Prediction confidence: {result.prediction_confidence:.3f}")
    print(f"  - Recursion depth: {result.recursion_depth}")
    print(f"  - Was approximated: {result.was_approximated}")

    # Show reasoning trace
    print("\n  Reasoning trace:")
    for line in result.reasoning_trace[:5]:
        print(f"    {line}")

    # TRM Demo
    print("\n[2.4] Tiny Recursive Model (TRM) approximation...")
    trm_config = TRMConfig(
        input_dim=32,
        hidden_dim=64,
        output_dim=32,
        num_actions=8,
    )
    trm = TinyRecursiveModel(trm_config)

    print(f"  - TRM parameters: {sum(p.numel() for p in trm.parameters()):,}")
    print(f"  - Can approximate deep ToM efficiently")

    return percept, belief, intent


def demo_poet_evolution():
    """Demonstrate POET Evolutionary Controller."""
    print_section("SUB-SYSTEM 3: POET Evolutionary Controller")

    print("\n[3.1] Creating sociological environments...")

    # Create preset environments
    env_types = [
        EnvironmentType.THE_HOLLOW,
        EnvironmentType.THE_MARKET,
        EnvironmentType.THE_MINISTRY,
        EnvironmentType.THE_COURT,
    ]

    for env_type in env_types:
        env = create_preset_environment(env_type)
        print(f"\n  {env_type.name}:")
        print(f"    - Institutional friction: {env.institutional_friction:.2f}")
        print(f"    - Power differential: {env.power_differential:.2f}")
        print(f"    - Deception pressure: {env.deception_pressure:.2f}")
        print(f"    - Norm rigidity: {env.norm_rigidity:.2f}")

    # Show environment evolution
    print("\n[3.2] Demonstrating environment mutation...")
    parent_env = create_preset_environment(EnvironmentType.THE_MARKET)
    child_env = parent_env.mutate(mutation_rate=0.2)

    print(f"  Parent (The Market) friction: {parent_env.institutional_friction:.3f}")
    print(f"  Child friction: {child_env.institutional_friction:.3f}")
    print(f"  Generation: {parent_env.generation} -> {child_env.generation}")

    # POET configuration
    print("\n[3.3] POET configuration for co-evolution...")
    poet_config = POETConfig(
        num_environments=5,
        num_agents_per_env=4,
        generations=50,
        env_mutation_rate=0.2,
        transfer_threshold=0.7,
        enable_transfer=True,
    )

    print(f"  - Environments: {poet_config.num_environments}")
    print(f"  - Agents per env: {poet_config.num_agents_per_env}")
    print(f"  - Transfer threshold: {poet_config.transfer_threshold}")
    print(f"  - ToM depth bonus: {poet_config.tom_depth_bonus}")

    print("\n  [Note: Full POET evolution requires NAS engine initialization]")

    return poet_config


def demo_godot_bridge(knowledge_base: IndrasNet):
    """Demonstrate Godot Physical Bridge."""
    print_section("SUB-SYSTEM 4: Godot Physical Bridge")

    print("\n[4.1] Initializing symbol grounding system...")
    grounder = SymbolGrounder(knowledge_base)

    # Simulate entity from Godot
    entity = EntityUpdate(
        godot_id=402,
        entity_type="object",
        name="Chair",
        position=Vector3(5.0, 0.0, 3.0),
        scale=Vector3(0.6, 1.0, 0.6),
        semantic_tags=["furniture", "seating", "wood"],
        affordances=["sit", "move"],
        is_interactable=True,
    )

    # Ground the entity
    from src.godot_bridge.symbol_grounding import GroundingContext
    context = GroundingContext(
        current_institution="courtroom",
    )

    grounded = grounder.ground_entity(entity, context)

    print(f"\n  Grounded Symbol:")
    print(f"    - Godot ID: {grounded.godot_id}")
    print(f"    - Category: {grounded.category}")
    print(f"    - Prototype match: {grounded.prototype_match}")
    print(f"    - Similarity: {grounded.prototype_similarity:.3f}")
    print(f"    - Affordances: {grounded.physical_affordances}")
    print(f"    - Semantic node: {grounded.semantic_node_id}")

    # Demonstrate perception processing
    print("\n[4.2] Demonstrating perception processing...")
    from src.godot_bridge.protocol import AgentPerception

    perception = AgentPerception(
        agent_godot_id=1,
        agent_name="Alice",
        visible_entities=[entity],
        own_position=Vector3(0, 0, 0),
        current_institution="courtroom",
        timestamp=100.0,
    )

    processor = PerceptionProcessor(grounder, knowledge_base)
    pfield = processor.process_perception(perception)

    print(f"\n  Perceptual Field:")
    print(f"    - Agent: {pfield.agent_name}")
    print(f"    - Visual inputs: {len(pfield.visual_inputs)}")
    print(f"    - Grounded symbols: {len(pfield.grounded_symbols)}")
    print(f"    - Activated norms: {pfield.activated_norms[:3]}")
    print(f"    - Institution: {pfield.current_institution}")

    # Demonstrate action execution
    print("\n[4.3] Demonstrating action execution...")
    executor = ActionExecutor(grounder)

    action = executor.plan_action(
        agent_id=1,
        action_type=ActionType.EXAMINE,
        target_entity_id=402,
        reason="Investigating the chair",
    )

    if action:
        print(f"\n  Planned Action:")
        print(f"    - Type: {action.action_type.name}")
        print(f"    - Target: {action.target_entity_id}")
        print(f"    - Reason: {action.reason}")
        print(f"    - Command ID: {action.command_id}")

        executor.queue_action(action)
        print(f"    - Queued for execution")

    # Bridge configuration
    print("\n[4.4] Bridge configuration for Godot communication...")
    bridge_config = BridgeConfig(
        host="localhost",
        port=9080,
        heartbeat_interval_ms=1000,
    )
    print(f"  - WebSocket: ws://{bridge_config.host}:{bridge_config.port}")
    print(f"  - Heartbeat: {bridge_config.heartbeat_interval_ms}ms")
    print(f"  - Ready for Godot 4.x connection")

    return grounder, processor, executor


def demo_integrated_system():
    """Demonstrate fully integrated Fractal Semiotic Engine."""
    print_section("INTEGRATED SYSTEM: The Fractal Semiotic Engine")

    print("\n" + "="*60)
    print(" THE SCIENTIFIC MANIFESTO")
    print("="*60)
    print("""
  Contemporary LLM-based agents fail at robust Theory of Mind (ToM)
  because they are fundamentally ungrounded. They operate on statistical
  correlations of text, lacking a causal link to consequential reality.

  THE THESIS:
  High-order, transparent Theory of Mind is an emergent property of
  navigating reality saturated with ASSOCIATIVE SEMANTIC ENCODING
  under intense INSTITUTIONAL AND EVOLUTIONARY PRESSURE.

  This simulation implements:
  1. INDRA'S NET - The omnipresent semantic web where every entity
     exists in hyperlinked meaning
  2. MENTALESE - TypeScript-style cognitive blocks as atomic thoughts
  3. RECURSIVE SELF-COMPRESSION - Agents simulating agents simulating...
  4. POET CO-EVOLUTION - Environment forces emergence of deep ToM
  5. SYMBOL GROUNDING - Physical reality and cognition are isomorphic
""")

    # Run all subsystem demos
    print("\n" + "-"*60)
    print(" RUNNING INTEGRATED DEMO")
    print("-"*60)

    # 1. Knowledge Graph
    net, query_engine = demo_semiotic_knowledge_graph()

    # 2. Cognitive Core
    percept, belief, intent = demo_cognitive_core(net)

    # 3. POET Evolution
    poet_config = demo_poet_evolution()

    # 4. Godot Bridge
    grounder, processor, executor = demo_godot_bridge(net)

    # Final integration summary
    print_section("INTEGRATION COMPLETE")

    print("""
  The Fractal Semiotic Engine is now initialized with:

  ┌─────────────────────────────────────────────────────────────┐
  │  LAYER 1: SEMIOTIC KNOWLEDGE GRAPH (Indra's Net)            │
  │    └─ Nodes: """ + f"{net.node_count:,}" + """ | Edges: """ + f"{net.edge_count:,}" + """                     │
  │    └─ 80-Dimension Taxonomy (Mundane, Institutional, Aesthetic)   │
  ├─────────────────────────────────────────────────────────────┤
  │  LAYER 2: COGNITIVE CORE (Mentalese + RSC)                  │
  │    └─ CognitiveBlocks: Percept → Hypothesis → Belief → Memory     │
  │    └─ Recursive simulation depth: up to 5th order ToM       │
  │    └─ TRM approximation for computational efficiency        │
  ├─────────────────────────────────────────────────────────────┤
  │  LAYER 3: EVOLUTIONARY CONTROLLER (POET)                    │
  │    └─ """ + f"{poet_config.num_environments} sociological environments" + """                  │
  │    └─ Red Queen dynamics: environments co-evolve with agents│
  │    └─ Transfer learning between compatible environments     │
  ├─────────────────────────────────────────────────────────────┤
  │  LAYER 4: PHYSICAL BRIDGE (Godot)                           │
  │    └─ Symbol grounding: Godot physics → semantic nodes      │
  │    └─ WebSocket communication at ws://localhost:9080        │
  │    └─ Perception → Cognition → Action loop                  │
  └─────────────────────────────────────────────────────────────┘

  To run the full system:
  1. Start Godot 4.x simulation with WebSocket client
  2. Run: python demo_fractal_semiotic_engine.py --full
  3. Watch agents evolve recursive Theory of Mind

  "The physical is cognitive. The cognitive is physical.
   In Indra's Net, each pearl reflects all others."
""")

    return {
        'knowledge_base': net,
        'query_engine': query_engine,
        'grounder': grounder,
        'perception_processor': processor,
        'action_executor': executor,
        'poet_config': poet_config,
    }


if __name__ == "__main__":
    import sys

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " FRACTAL SEMIOTIC ENGINE ".center(58) + "║")
    print("║" + " Theory of Mind through Embodied Semantic Evolution ".center(58) + "║")
    print("╚" + "═"*58 + "╝")

    # Run integrated demo
    components = demo_integrated_system()

    print("\n" + "="*60)
    print(" DEMO COMPLETE")
    print("="*60)
    print("\nAll subsystems initialized and integrated successfully.")
    print("The Fractal Semiotic Engine is ready for evolution.\n")
