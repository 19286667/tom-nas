# Fractal Semiotic Engine

## A Scientific Mandate for Evolving Theory of Mind

This document describes the Fractal Semiotic Engine - a system where physical reality and cognitive processes are isomorphic, with **meaning** as the fundamental physics.

## The Scientific Manifesto

**The Failure of Current AI:** Contemporary LLM-based agents fail at robust Theory of Mind (ToM) because they are fundamentally ungrounded. They operate on statistical correlations of text, lacking a causal link to a consequential reality.

**The Thesis:** High-order, transparent Theory of Mind—the ability to recursively model the minds of others to N degrees (B_a(B_b(B_a(p))))—is an emergent property of navigating a reality saturated with **associative semantic encoding (Indra's Net)** under intense **institutional and evolutionary pressure**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   FRACTAL SEMIOTIC ENGINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         SUB-SYSTEM 1: SEMIOTIC KNOWLEDGE GRAPH          │   │
│  │                      (Indra's Net)                      │   │
│  │                                                         │   │
│  │  "In the heaven of Indra, there is said to be a        │   │
│  │   network of pearls, so arranged that if you look      │   │
│  │   at one you see all the others reflected in it."      │   │
│  │                                                         │   │
│  │  • 80-Dimension Taxonomy (Mundane/Institutional/Aesthetic) │
│  │  • SemanticNodes with typed edges                      │   │
│  │  • Activation spreading through semantic web           │   │
│  │  • Context-dependent meaning collapse                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          SUB-SYSTEM 2: COGNITIVE CORE                   │   │
│  │            (Mentalese + RSC Engine)                     │   │
│  │                                                         │   │
│  │  CognitiveBlocks (Type-Shifting):                      │   │
│  │  Percept → Hypothesis → Belief → Memory                │   │
│  │                                                         │   │
│  │  Recursive Self-Compression:                           │   │
│  │  Agent A simulates Agent B simulating Agent A...       │   │
│  │  (Up to 5th order Theory of Mind)                      │   │
│  │                                                         │   │
│  │  TRM: Neural approximation for deep recursion          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│              ┌─────────────┴─────────────┐                     │
│              ▼                           ▼                     │
│  ┌──────────────────────┐   ┌──────────────────────────────┐  │
│  │  SUB-SYSTEM 3: POET  │   │  SUB-SYSTEM 4: GODOT BRIDGE  │  │
│  │  Evolution Controller│   │  (Symbol Grounding)          │  │
│  │                      │   │                              │  │
│  │ Environment Genotypes│   │ Godot ID → Semantic Node     │  │
│  │ • The Hollow         │   │ Physics → Cognition          │  │
│  │ • The Market         │   │ Cognition → Physics          │  │
│  │ • The Ministry       │   │                              │  │
│  │ • The Court          │   │ WebSocket @ ws://localhost:9080│
│  │ • The Temple         │   │                              │  │
│  │                      │   │ "A chair is not just an      │  │
│  │ Red Queen Dynamics   │   │  abstract concept - it is    │  │
│  │ (Co-evolution)       │   │  Rigidbody3D ID_992"         │  │
│  └──────────────────────┘   └──────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## The 80-Dimension Taxonomy

### Layer 1: Mundane Life (Dimensions 1-27)
- **Sustenance** (1-3): Nourishment, Gustatory Pleasure, Commensality
- **Bodily Care** (4-6): Hygiene, Health Maintenance, Appearance
- **Mobility** (7-9): Locomotion, Spatial Range, Transport Autonomy
- **Domestic Labor** (10-12): Space Maintenance, Object Care, Burden
- **Work** (13-16): Productive Engagement, Skill, Autonomy, Return
- **Leisure** (17-19): Engagement, Activity Type, Social Dimension
- **Social** (20-23): Frequency, Depth, Status, Communication Mode
- **Commerce** (24-25): Exchange Activity, Transaction Complexity
- **Learning** (26): Learning Mode
- **Rest** (27): Restoration State

### Layer 2: Institutional (Dimensions 28-54)
- **Governance** (28-31): Authority, Legitimacy, Participation, Transparency
- **Economy** (32-35): Market Integration, Property, Hierarchy, Exchange
- **Law** (36-39): Formality, Enforcement, Justice, Rights
- **Religion** (40-43): Sacred Presence, Ritual, Orthodoxy, Transcendence
- **Education** (44-47): Knowledge Access, Pedagogy, Credentials, Canon
- **Family** (48-51): Kinship, Authority, Marital Norms, Obligation
- **Healthcare** (52-54): Care Access, Medical Authority, Body Autonomy

### Layer 3: Aesthetic (Dimensions 55-80)
- **Visual** (55-58): Chromatic Intensity, Formal Complexity, Scale, Light
- **Temporal** (59-61): Pace, Rhythm, Duration
- **Symbolic** (62-65): Meaning Saturation, Reference, Ambiguity, Irony
- **Emotional** (66-69): Intensity, Valence, Tension, Wonder
- **Cultural** (70-72): Temporal Orientation, Specificity, Innovation
- **Authenticity** (73-75): Genuineness, Craft Evidence, Patina
- **Status** (76-78): Signal, Exclusivity, Taste Marker
- **Narrative** (79-80): Coherence, Genre Adherence

## Mentalese: The Language of Thought

```python
# A cognitive block representing a belief about another's intent
class BeliefBlock(CognitiveBlock):
    proposition: str           # "Bob intends to deceive"
    about_agent: str          # "Bob"
    belief_order: int         # 2 (second-order ToM)
    confidence: float         # 0.75
    nested_belief: BeliefBlock  # What Bob believes about me
    evidence: List[Evidence]  # Supporting percepts
```

**Type-Shifting as Reasoning:**
```
Percept → Hypothesis → Belief → Memory
   ↑           ↓
   └───────────┘ (Revision)
```

## Recursive Self-Compression (RSC)

When Agent A needs to predict Agent B:

1. **Snapshot**: A compresses current perception into a Dynamic Block
2. **Instantiation**: A spins up lightweight internal simulation (A')
3. **Population**: A places models of itself and B into A'
4. **Execution**: A runs simulation forward N steps
5. **Recursion**: If B needs to think about A, it spins up A''

```python
# Example: 3rd-order Theory of Mind
sim = RecursiveSimulationNode(
    simulating_agent="Alice",
    world_model=world,
    config=SimulationConfig(max_recursion_depth=3)
)

# Alice simulates Bob simulating Alice
result = sim.run_simulation("Bob")
# result.predicted_action: What Bob will do
# result.recursion_depth: 3 (Alice → Bob → Alice)
```

## POET with Sociological Genotypes

Unlike standard POET (which evolves terrain difficulty), we evolve **institutional difficulty**:

| Environment | Friction | Power | Deception | Norms |
|-------------|----------|-------|-----------|-------|
| The Hollow  | 0.2      | 0.3   | 0.2       | 0.2   |
| The Market  | 0.5      | 0.7   | 0.7       | 0.4   |
| The Ministry| 0.9      | 0.9   | 0.5       | 0.9   |
| The Court   | 0.8      | 0.9   | 0.6       | 0.95  |
| The Temple  | 0.7      | 0.6   | 0.3       | 0.9   |

**Red Queen Dynamics**: As agents solve environments, environments mutate to become harder, forcing deeper ToM.

## Symbol Grounding

Solves Harnad's Symbol Grounding Problem by anchoring cognition in physics:

```python
# Godot entity
entity = EntityUpdate(
    godot_id=402,
    name="Chair",
    position=Vector3(5, 0, 3),
    semantic_tags=["furniture", "wood"],
    affordances=["sit", "move"]
)

# Grounded symbol
symbol = grounder.ground_entity(entity, context)
# symbol.category: "chair" (prototype match)
# symbol.semantic_node_id: "obj_chair" (in Indra's Net)
# symbol.physical_affordances: ["can_sit_on", "can_pick_up"]
```

## Quick Start

```bash
# Run the integrated demo
python demo_fractal_semiotic_engine.py

# Start the Godot bridge (for live simulation)
python -c "
from src.godot_bridge import GodotBridge, BridgeConfig
bridge = GodotBridge(BridgeConfig())
bridge.start(blocking=True)  # ws://localhost:9080
"
```

## Module Structure

```
src/
├── knowledge_base/          # SUB-SYSTEM 1: Indra's Net
│   ├── schemas.py          # SemanticNode, SemanticEdge, etc.
│   ├── taxonomy.py         # 80-Dimension Taxonomy
│   ├── indras_net.py       # Graph database
│   └── query_engine.py     # Semantic traversal
│
├── cognition/              # SUB-SYSTEM 2: Mentalese + RSC
│   ├── mentalese.py        # CognitiveBlock types
│   ├── recursive_simulation.py  # RSC engine
│   └── trm.py              # Tiny Recursive Model
│
├── evolution/              # SUB-SYSTEM 3: POET
│   ├── nas_engine.py       # Neural Architecture Search
│   ├── poet_controller.py  # POET with sociological genotypes
│   └── fitness.py          # ToM fitness evaluation
│
└── godot_bridge/           # SUB-SYSTEM 4: Symbol Grounding
    ├── protocol.py         # WebSocket messages
    ├── symbol_grounding.py # Physics → Meaning
    ├── perception.py       # Sensory processing
    ├── action.py           # Intent execution
    └── bridge.py           # WebSocket server
```

## Theoretical Foundations

- **Symbol Grounding Problem** (Harnad, 1990)
- **Language of Thought Hypothesis** (Fodor, 1975)
- **Simulation Theory of Mind** (Goldman, 2006)
- **POET Open-Ended Evolution** (Wang et al., 2019)
- **Conceptual Metaphor Theory** (Lakoff & Johnson, 1980)
- **New Institutional Economics** (North, 1990)
- **Stereotype Content Model** (Fiske et al., 2002)

---

*"The physical is cognitive. The cognitive is physical. In Indra's Net, each pearl reflects all others."*
