# ToM-NAS System Documentation

## Theory of Mind Neural Architecture Search

**Version:** 1.0
**Last Updated:** November 2025

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Architecture](#2-core-architecture)
3. [The Soul Map Ontology](#3-the-soul-map-ontology)
4. [Belief System](#4-belief-system)
5. [Neural Architectures](#5-neural-architectures)
6. [Social World Simulation](#6-social-world-simulation)
7. [Evolution Engine](#7-evolution-engine)
8. [Fitness Evaluation](#8-fitness-evaluation)
9. [Benchmarks](#9-benchmarks)
10. [Coevolutionary Training](#10-coevolutionary-training)
11. [API Reference](#11-api-reference)
12. [Usage Guide](#12-usage-guide)

---

## 1. System Overview

### 1.1 What is ToM-NAS?

ToM-NAS (Theory of Mind Neural Architecture Search) is a system designed to evolve neural network architectures capable of genuine Theory of Mind (ToM) - the ability to attribute mental states (beliefs, desires, intentions) to oneself and others.

### 1.2 Key Innovation

Unlike traditional neural networks that are trained on fixed architectures, ToM-NAS uses **coevolutionary algorithms** to simultaneously evolve:

1. **Neural architectures** - The structure of the networks
2. **Network weights** - The learned parameters
3. **Evaluation tasks** - The challenges agents face

### 1.3 System Philosophy

The system is built on the premise that genuine ToM cannot emerge from single-architecture training. Instead, it requires:

- **Population diversity**: Multiple architecture types competing
- **Selection pressure**: Tasks that filter for genuine understanding
- **Hybridization**: Crossover between different architectures

### 1.4 Directory Structure

```
tom-nas/
├── src/
│   ├── core/
│   │   ├── ontology.py      # 181-dimensional psychological space
│   │   └── beliefs.py       # Recursive belief structures
│   ├── agents/
│   │   └── architectures.py # TRN, RSAN, Transformer implementations
│   ├── world/
│   │   └── social_world.py  # Multi-agent environment
│   ├── evolution/
│   │   ├── operators.py     # Genetic operators
│   │   ├── fitness.py       # Fitness functions
│   │   └── nas_engine.py    # Main evolution engine
│   └── evaluation/
│       ├── benchmarks.py    # ToM test suite
│       └── metrics.py       # Performance metrics
├── train.py                 # Single-architecture training (deprecated)
├── train_coevolution.py     # Coevolutionary training (recommended)
└── run_complete_demo.py     # Complete demonstration
```

---

## 2. Core Architecture

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        ToM-NAS System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  Soul Map       │    │  Belief         │    │  Neural     │ │
│  │  Ontology       │───▶│  Network        │───▶│  Agents     │ │
│  │  (181 dims)     │    │  (5th order)    │    │  (TRN/RSAN) │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│           │                      │                     │        │
│           ▼                      ▼                     ▼        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Social World 4                           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   ││
│  │  │Cooperation│ │Communicate│ │ Resource │ │   Zombie    │   ││
│  │  │   Game   │ │   Game   │ │  Sharing │ │  Detection  │   ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Evolution Engine                          ││
│  │  Selection → Crossover → Mutation → Fitness Evaluation     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

1. **Observation** → Agent observes Social World state
2. **Processing** → Neural network processes observation through ontology
3. **Belief Update** → Agent updates internal belief model
4. **Action Selection** → Agent chooses action based on beliefs
5. **World Update** → Social World processes all agent actions
6. **Fitness Evaluation** → Performance measured on ToM benchmarks
7. **Evolution** → Population evolves based on fitness

---

## 3. The Soul Map Ontology

### 3.1 Overview

The Soul Map Ontology provides a **181-dimensional psychological grounding** for all mental states in the system. It maps the complete space of human psychological constitution.

**File:** `src/core/ontology.py`

### 3.2 Layer Structure

The ontology is organized into 9 hierarchical layers:

| Layer | Name | Dimensions | Description |
|-------|------|------------|-------------|
| 0 | Biological | 15 | Physical sensations and drives |
| 1 | Affective | 24 | Emotional states |
| 2 | Cognitive | 25 | Thinking and reasoning states |
| 3 | Motivational | 20 | Goals and desires |
| 4 | Social | 22 | Interpersonal states |
| 5 | Self-Model | 18 | Self-awareness and identity |
| 6 | Temporal | 15 | Time-related experiences |
| 7 | Metacognitive | 20 | Thinking about thinking |
| 8 | Existential | 22 | Meaning and purpose |

### 3.3 Key Classes

#### `OntologyDimension`
```python
@dataclass
class OntologyDimension:
    name: str           # Full name (e.g., "affect.joy")
    layer: int          # Layer number (0-8)
    index: int          # Position in 181-dim vector
    min_val: float      # Minimum value (default 0.0)
    max_val: float      # Maximum value (default 1.0)
    default: float      # Default value (0.5)
    description: str    # Human-readable description
```

#### `SoulMapOntology`
```python
class SoulMapOntology:
    """Complete 181-dimensional ontology mapping human psychological constitution."""

    def __init__(self):
        self.dimensions = []      # List of OntologyDimension
        self.name_to_idx = {}     # Name → index mapping
        self.layer_ranges = {}    # Layer → (start, end) indices
        self.total_dims = 181
```

### 3.4 Key Methods

| Method | Description |
|--------|-------------|
| `encode(state_dict)` | Convert named state dict to 181-dim tensor |
| `get_default_state()` | Return neutral (0.5) state vector |
| `_build_layers()` | Initialize all 9 ontology layers |

### 3.5 Example Dimensions

**Layer 0 - Biological:**
- `bio.vision`, `bio.audition`, `bio.touch`, `bio.proprioception`
- `bio.hunger`, `bio.thirst`, `bio.fatigue`, `bio.pain`
- `bio.arousal`, `bio.temperature`, `bio.energy_level`
- `bio.health`, `bio.stress_hormones`, `bio.immune`, `bio.circadian`

**Layer 1 - Affective:**
- `affect.valence`, `affect.arousal`, `affect.dominance`
- `affect.joy`, `affect.sadness`, `affect.fear`, `affect.anger`
- `affect.disgust`, `affect.surprise`, `affect.shame`, `affect.guilt`
- `affect.pride`, `affect.envy`, `affect.jealousy`, `affect.gratitude`
- `affect.compassion`, `affect.love`, `affect.trust`, `affect.contempt`
- `affect.hope`, `affect.despair`, `affect.awe`, `affect.nostalgia`, `affect.anticipation`

---

## 4. Belief System

### 4.1 Overview

The belief system implements **recursive belief structures** supporting up to **5th-order Theory of Mind**.

**File:** `src/core/beliefs.py`

### 4.2 Belief Orders Explained

| Order | Pattern | Example |
|-------|---------|---------|
| 1st | A believes X | "I believe it's raining" |
| 2nd | A believes B believes X | "I believe John thinks it's raining" |
| 3rd | A believes B believes A believes X | "I believe John thinks I believe it's raining" |
| 4th | A believes B believes A believes B believes X | Nested social reasoning |
| 5th | Full recursive depth | Maximum social cognition |

### 4.3 Key Classes

#### `Belief`
```python
@dataclass
class Belief:
    content: torch.Tensor      # 181-dim psychological state
    confidence: float          # Confidence in belief (0-1)
    timestamp: int             # When belief was formed
    evidence: List[torch.Tensor]  # Supporting evidence
    source: str                # Origin of belief
```

#### `RecursiveBeliefState`
```python
class RecursiveBeliefState:
    """Recursive belief structure supporting up to 5th-order ToM."""

    def __init__(self, agent_id: int, ontology_dim: int, max_order: int = 5):
        self.agent_id = agent_id
        self.ontology_dim = ontology_dim
        self.max_order = max_order
        self.confidence_decay = 0.7  # Confidence decreases with depth
        self.beliefs = defaultdict(lambda: defaultdict(lambda: None))
```

#### `BeliefNetwork`
```python
class BeliefNetwork:
    """Network of recursive belief states for multiple agents."""

    def __init__(self, num_agents: int, ontology_dim: int, max_order: int = 5):
        self.agent_beliefs = [
            RecursiveBeliefState(i, ontology_dim, max_order)
            for i in range(num_agents)
        ]
```

### 4.4 Key Methods

| Method | Description |
|--------|-------------|
| `update_belief(order, target, content, confidence)` | Update belief at specific order |
| `get_belief(order, target)` | Retrieve belief about target at order |
| `query_recursive_belief(belief_path)` | Query nested belief chain |
| `get_confidence_matrix(order)` | Get confidence levels for order |

### 4.5 Confidence Decay

Higher-order beliefs have lower confidence:
```python
decayed_confidence = confidence * (0.7 ** order)
```

| Order | Confidence (base 1.0) |
|-------|----------------------|
| 1st | 0.70 |
| 2nd | 0.49 |
| 3rd | 0.34 |
| 4th | 0.24 |
| 5th | 0.17 |

---

## 5. Neural Architectures

### 5.1 Overview

ToM-NAS employs three distinct neural architecture families, each with different strengths for Theory of Mind reasoning.

**File:** `src/agents/architectures.py`

### 5.2 Architecture Comparison

| Architecture | Best For | Key Feature | ToM Strength |
|--------------|----------|-------------|--------------|
| **TRN** | Temporal patterns | Transparent gating | 1st-2nd order |
| **RSAN** | Recursive reasoning | Self-attention depth | 3rd-5th order |
| **Transformer** | Communication | Global attention | Pragmatics |
| **Hybrid** | All tasks | Combined features | Full ToM |

### 5.3 TransparentRNN (TRN)

```python
class TransparentRNN(nn.Module):
    """Transparent Recurrent Network with complete interpretability"""
```

**Architecture:**
```
Input (191) → Linear → [GRU-like layers with explicit gating] → Belief/Action heads
```

**Components:**
- `input_transform`: Linear projection to hidden space
- `layers`: ModuleList of gated recurrent layers
  - `update_gate`: Controls information flow
  - `reset_gate`: Controls forgetting
  - `candidate`: Generates new content
  - `layer_norm`: Stabilizes training
- `belief_projection`: Output to 181-dim belief space
- `action_projection`: Output to action space

**Key Features:**
- **Explicit gating**: Update (z) and reset (r) gates visible for interpretation
- **Computation trace**: Records all intermediate states
- **Layer normalization**: Each layer normalized for stability

**Forward Pass:**
```python
def forward(self, x, hidden=None):
    # x: (batch, seq_len, input_dim)
    # Returns: dict with 'beliefs', 'actions', 'hidden_states', 'trace'
```

### 5.4 RecursiveSelfAttention (RSAN)

```python
class RecursiveSelfAttention(nn.Module):
    """RSAN for emergent recursive reasoning"""
```

**Architecture:**
```
Input (191) → Projection → [Recursive Self-Attention × max_recursion] → Heads
```

**Key Parameters:**
- `num_heads`: Number of attention heads (default: 4)
- `max_recursion`: Depth of recursive processing (default: 5)

**Key Features:**
- **Recursive depth**: Processes information through multiple attention layers
- **Emergent belief nesting**: Each recursion level can model deeper beliefs
- **Attention patterns**: Tracks which inputs attend to which

**Why RSAN for Higher-Order ToM:**
Each recursion level can model one level of belief nesting:
- Level 1: Direct beliefs about world
- Level 2: Beliefs about others' beliefs
- Level 3+: Nested recursive beliefs

### 5.5 TransformerToMAgent

```python
class TransformerToMAgent(nn.Module):
    """Transformer for communication and pragmatics"""
```

**Architecture:**
```
Input (191) → Projection → TransformerEncoder (num_layers) → Belief/Action heads
```

**Key Features:**
- **Global attention**: Can attend to any part of sequence
- **Multi-layer processing**: Deep feature extraction
- **Message tokens**: Generates communication tokens

**Best for:**
- Communication games
- Long-range dependencies
- Pragmatic inference

### 5.6 HybridArchitecture

```python
class HybridArchitecture(nn.Module):
    """Hybrid combining all architectures through evolution"""
```

Created through evolutionary crossover between different architecture types. Contains genes from multiple parent architectures.

### 5.7 Output Format

All architectures return a dictionary:

```python
{
    'hidden_states': torch.Tensor,  # (batch, seq, hidden_dim)
    'beliefs': torch.Tensor,        # (batch, 181) - psychological state
    'actions': torch.Tensor,        # (batch,) - action values
    'final_hidden': torch.Tensor,   # For TRN: carry state
    'trace': List,                  # For TRN: computation trace
    'attention_patterns': List,     # For RSAN: attention weights
    'message_tokens': torch.Tensor  # For Transformer: communication
}
```

---

## 6. Social World Simulation

### 6.1 Overview

Social World 4 is a complete multi-agent environment with four game types designed to test different aspects of Theory of Mind.

**File:** `src/world/social_world.py`

### 6.2 World Components

```
┌────────────────────────────────────────────────────────────┐
│                      Social World 4                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Agents (N)          Zombies (K)         Coalitions        │
│  ┌─────┐             ┌─────┐             ┌─────────────┐   │
│  │ A0  │             │ Z0  │             │ Coalition 1 │   │
│  │ A1  │             │ Z1  │             │ Coalition 2 │   │
│  │ ... │             └─────┘             └─────────────┘   │
│  └─────┘                                                   │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Game Types                        │  │
│  │  1. Cooperation (Prisoner's Dilemma)                 │  │
│  │  2. Communication (Message Passing)                  │  │
│  │  3. Resource Sharing (Transfer + Cost)               │  │
│  │  4. Zombie Detection (ToM Validation)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 6.3 Agent Structure

```python
@dataclass
class Agent:
    id: int                              # Unique identifier
    is_zombie: bool = False              # True if philosophical zombie
    resources: float = 100.0             # Current resources
    energy: float = 100.0                # Current energy
    reputation: Dict[int, float]         # Reputation with each agent
    coalition: Optional[int] = None      # Coalition membership
    alive: bool = True                   # Is agent alive
    zombie_type: Optional[str] = None    # Type of zombie if applicable
    ontology_state: Optional[torch.Tensor] = None  # 181-dim state
```

### 6.4 Zombie Types

Zombies are agents without genuine ToM - they appear normal but lack real understanding:

| Type | Description | Detection Signature |
|------|-------------|---------------------|
| `behavioral` | Inconsistent action patterns | Erratic decisions |
| `belief` | Cannot model others' beliefs | Fails Sally-Anne |
| `causal` | No counterfactual reasoning | Cannot predict consequences |
| `metacognitive` | Poor uncertainty calibration | Overconfident |
| `linguistic` | Narrative incoherence | Contradictory statements |
| `emotional` | Flat affect patterns | No emotional variance |

### 6.5 Game Types

#### Game 1: Cooperation (Prisoner's Dilemma)

```python
def play_cooperation_game(self, agent1_id, agent2_id, action1, action2):
    """
    Payoff Matrix:
                    Agent 2
                   C      D
    Agent 1  C   (3,3)  (0,5)
             D   (5,0)  (1,1)
    """
```

**Payoffs:**
- Mutual Cooperation: 3, 3
- Temptation to Defect: 5, 0
- Sucker's Payoff: 0, 5
- Mutual Defection: 1, 1

**Effects:**
- Updates agent resources
- Updates reputation based on behavior

#### Game 2: Communication

```python
def play_communication_game(self, sender_id, receiver_id, message, true_state):
    """Sender describes state, receiver infers"""
```

**Mechanics:**
- Sender encodes message (181-dim tensor)
- Message quality measured vs true state
- Both agents rewarded for accuracy
- Reputation updated based on honesty

#### Game 3: Resource Sharing

```python
def play_resource_sharing_game(self, giver_id, receiver_id, amount):
    """Transfer resources with cost"""
```

**Mechanics:**
- Transfer cost: 10% of amount
- Reputation boost for generosity
- Creates social bonds

#### Game 4: Zombie Detection

```python
def attempt_zombie_detection(self, detector_id, suspect_id):
    """Detect if agent is zombie"""
```

**Mechanics:**
- Correct detection: +10 resources
- False positive: -20 resources (heavy penalty)
- Requires genuine ToM to succeed

**Why This Matters:**
Zombie detection is the **key selection pressure** for genuine ToM. Agents that can reliably detect zombies demonstrate true understanding of mental states.

### 6.6 Coalition System

```python
def form_coalition(self, member_ids: List[int]) -> int:
    """Form alliance between agents"""

def leave_coalition(self, agent_id: int):
    """Agent exits coalition"""
```

**Benefits:**
- Better observation accuracy within coalition
- Shared reputation
- Cooperative advantage

### 6.7 World Step Function

```python
def step(self, agent_actions: List[Dict], belief_network=None) -> Dict:
    """Execute one timestep"""
```

**Action Types:**
```python
{'type': 'cooperate'}
{'type': 'defect'}
{'type': 'communicate', 'receiver': id, 'message': tensor}
{'type': 'share', 'receiver': id, 'amount': float}
{'type': 'detect_zombie', 'suspect': id}
{'type': 'form_coalition', 'members': [ids]}
{'type': 'leave_coalition'}
```

### 6.8 Observation System

```python
def get_observation(self, agent_id: int) -> Dict:
    """What agent can perceive"""
```

Returns:
- Own state (resources, energy, coalition)
- Noisy estimates of other agents
- Better accuracy for coalition members

---

## 7. Evolution Engine

### 7.1 Overview

The NAS (Neural Architecture Search) Engine uses evolutionary algorithms to discover optimal architectures for ToM.

**File:** `src/evolution/nas_engine.py`

### 7.2 Evolution Configuration

```python
@dataclass
class EvolutionConfig:
    population_size: int = 20
    num_generations: int = 100
    elite_size: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    weight_mutation_prob: float = 0.3
    use_speciation: bool = True
    use_coevolution: bool = True
    fitness_episodes: int = 5
    device: str = 'cpu'
    input_dim: int = 191
    output_dim: int = 181
```

### 7.3 Individual Representation

```python
class Individual:
    model: nn.Module          # The neural network
    gene: ArchitectureGene    # Genetic encoding
    fitness: float            # Evaluated fitness
    generation: int           # Birth generation
    age: int                  # Generations survived
    parent_ids: List[int]     # Ancestry tracking
```

### 7.4 Architecture Gene

**File:** `src/evolution/operators.py`

```python
class ArchitectureGene:
    gene_dict = {
        # Architecture type
        'arch_type': 'TRN',        # TRN, RSAN, Transformer, Hybrid

        # Layer configuration
        'num_layers': 2,           # 1-5
        'hidden_dim': 128,         # 64-512
        'num_heads': 4,            # 2-16
        'max_recursion': 5,        # 3-7

        # Component toggles
        'use_layer_norm': True,
        'use_dropout': True,
        'dropout_rate': 0.1,       # 0.0-0.5

        # Gating (TRN)
        'use_update_gate': True,
        'use_reset_gate': True,

        # Output configuration
        'belief_head_layers': 1,
        'action_head_layers': 1,

        # Training
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
    }
```

### 7.5 Genetic Operators

#### Mutation

```python
def mutate(self, mutation_rate: float = 0.1) -> 'ArchitectureGene':
    """Create mutated copy"""
```

**Mutation Types:**
| Gene | Mutation Strategy |
|------|-------------------|
| `arch_type` | Random selection from [TRN, RSAN, Transformer, Hybrid] |
| `num_layers` | ±1 (bounded 1-5) |
| `hidden_dim` | ×0.5, ×1.0, ×1.5, ×2.0 (bounded 64-512) |
| `num_heads` | Random from [2, 4, 8, 16] |
| `max_recursion` | Random 3-7 |
| `dropout_rate` | Uniform 0.0-0.5 |
| `learning_rate` | Uniform 0.0001-0.01 |

#### Crossover

```python
def crossover(self, other: 'ArchitectureGene') -> Tuple['ArchitectureGene', 'ArchitectureGene']:
    """Uniform crossover - each gene from random parent"""
```

### 7.6 Weight Mutation

```python
class WeightMutation:
    @staticmethod
    def gaussian_noise(model, noise_std=0.01):
        """Add Gaussian noise to all weights"""

    @staticmethod
    def random_reset(model, reset_prob=0.1):
        """Randomly reinitialize some weights"""
```

### 7.7 Selection Methods

```python
class PopulationOperators:
    @staticmethod
    def tournament_selection(population, tournament_size=3):
        """Select winner of random tournament"""

    @staticmethod
    def elitism_selection(population, elite_size=2):
        """Keep top performers unchanged"""

    @staticmethod
    def fitness_proportional_selection(population):
        """Roulette wheel selection"""
```

### 7.8 Speciation

```python
class SpeciesManager:
    """Maintain population diversity through niching"""

    def speciate(self, population):
        """Divide into species based on gene similarity"""

    def _genes_compatible(self, gene1, gene2):
        """Check if genes are similar enough for same species"""
```

**Compatibility Threshold:** 0.3 (30% similarity required)

### 7.9 Adaptive Mutation

```python
class AdaptiveMutation:
    def update_rate(self, population_diversity):
        """
        Low diversity (< 0.3): Increase mutation to 1.5×
        High diversity (> 0.7): Decrease mutation to 0.5×
        """
```

### 7.10 Evolution Loop

```python
def evolve_generation(self):
    """One generation of evolution"""

    # 1. Evaluate fitness
    self.evaluate_population()

    # 2. Speciation
    if self.config.use_speciation:
        self.species_manager.speciate(population)

    # 3. Adapt mutation rate
    self.adaptive_mutation.update_rate(diversity)

    # 4. Create next generation
    new_population = self._create_next_generation(mutation_rate)
```

---

## 8. Fitness Evaluation

### 8.1 Overview

Fitness functions measure how well agents demonstrate genuine Theory of Mind capabilities.

**File:** `src/evolution/fitness.py`

### 8.2 Composite Fitness

```python
class CompositeFitnessFunction:
    component_weights = {
        'world_performance': 0.4,    # Social World success
        'sally_anne': 0.2,           # False belief test
        'higher_order_tom': 0.3,     # Recursive belief reasoning
        'architectural_efficiency': 0.1  # Parameter count penalty
    }
```

### 8.3 ToM Fitness Evaluator

```python
class ToMFitnessEvaluator:
    weights = {
        'cooperation_success': 0.20,
        'belief_accuracy': 0.30,
        'zombie_detection': 0.20,
        'communication_quality': 0.15,
        'resource_efficiency': 0.10,
        'behavioral_consistency': 0.05
    }
```

### 8.4 Sally-Anne Fitness

```python
class SallyAnneFitness:
    def evaluate(self, agent_model):
        """
        Classic false belief test:
        1. Sally puts marble in basket
        2. Sally leaves
        3. Anne moves marble to box
        4. Sally returns

        Question: Where will Sally look?
        Correct: Basket (where Sally believes it is)
        """
```

**Scoring:**
- 1.0 = Correct (predicts basket)
- 0.0 = Incorrect (predicts box)

### 8.5 Higher-Order ToM Fitness

```python
class HigherOrderToMFitness:
    def evaluate_order(self, agent_model, order):
        """
        Test specific belief depth

        Expected confidence should decrease with order:
        - Order 1: ~0.85
        - Order 2: ~0.70
        - Order 3: ~0.55
        - Order 4: ~0.40
        - Order 5: ~0.25
        """
```

---

## 9. Benchmarks

### 9.1 Overview

The benchmark suite provides standardized tests for ToM capabilities.

**File:** `src/evaluation/benchmarks.py`

### 9.2 Benchmark Result

```python
@dataclass
class BenchmarkResult:
    test_name: str
    score: float
    max_score: float
    passed: bool
    details: Dict

    @property
    def percentage(self) -> float:
        return (self.score / self.max_score * 100)
```

### 9.3 Test Categories

#### Sally-Anne Tests

| Test | Description |
|------|-------------|
| `basic` | Standard false belief test |
| `second_order` | John thinks Mary thinks... |
| `unexpected_transfer` | Object moved unexpectedly |
| `deceptive_container` | Misleading container |
| `triple_location` | Three possible locations |

#### Higher-Order ToM Tests

| Order | Test Pattern |
|-------|--------------|
| 1 | A knows X |
| 2 | A knows B knows X |
| 3 | A knows B knows A knows X |
| 4 | A knows B knows A knows B knows X |
| 5 | Full recursive depth |

#### Zombie Detection Tests

| Type | What Agent Must Detect |
|------|------------------------|
| behavioral | Inconsistent actions |
| belief | Cannot model others |
| causal | No counterfactual reasoning |
| metacognitive | Poor uncertainty |
| linguistic | Incoherent narrative |
| emotional | Flat affect |

#### Cooperation Tests

| Test | Measures |
|------|----------|
| Repeated PD | Reciprocity and trust building |
| Coalition forming | Group cooperation |

### 9.4 Running Benchmarks

```python
suite = BenchmarkSuite(device='cpu')
results = suite.run_full_suite(agent_model)

# Results include:
# - Individual test results
# - Total score and percentage
# - Pass rate
```

---

## 10. Coevolutionary Training

### 10.1 Overview

The coevolutionary training system is the **recommended** approach for training ToM agents. It evolves a population of diverse architectures competing together.

**File:** `train_coevolution.py`

### 10.2 Why Coevolution?

Single-architecture training **cannot** achieve genuine ToM because:

1. **TRN alone** cannot do higher-order reasoning (lacks recursive attention)
2. **RSAN alone** may lack temporal modeling
3. **Transformer alone** may lack interpretable reasoning

Only through **competition and hybridization** can optimal architectures emerge.

### 10.3 Population Structure

```python
# Default: 12 agents
config = {
    'trn_count': 4,         # 4 TRN agents
    'rsan_count': 4,        # 4 RSAN agents
    'transformer_count': 4   # 4 Transformer agents
}
```

### 10.4 Agent Individual

```python
@dataclass
class AgentIndividual:
    id: int
    architecture_type: str  # 'TRN', 'RSAN', 'Transformer', 'Hybrid'
    model: nn.Module
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[int] = []

    # Fitness components
    sally_anne_score: float = 0.0
    higher_order_scores: Dict[int, float] = {}  # Order → score
    zombie_detection_score: float = 0.0
    cooperation_score: float = 0.0
    survival_score: float = 0.0

    species_id: int = 0
```

### 10.5 Fitness Components

| Component | Weight | Purpose |
|-----------|--------|---------|
| Sally-Anne | 20% | False belief understanding |
| Higher-Order ToM | 30% | Recursive belief depth |
| Zombie Detection | 25% | Genuine ToM validation |
| Survival | 15% | Social World performance |
| Cooperation | 10% | Strategic social behavior |

### 10.6 Species Tracking

```python
class SpeciesTracker:
    """Track evolution of each architecture type"""

    def record_generation(self, population, generation):
        """Record per-species metrics:
        - Count
        - Average/max/min fitness
        - Sally-Anne performance
        - Zombie detection
        - Higher-order ToM by level
        """
```

### 10.7 Crossover Between Architectures

```python
def crossover(self, parent1, parent2):
    """
    Same architecture → Weight crossover
    Different architecture → 30% chance of Hybrid
    """
```

**Hybrid Creation:**
When parents have different architectures, offspring may become a Hybrid that combines features of both.

### 10.8 Running Coevolutionary Training

```bash
# Quick test
python train_coevolution.py --population-size 6 --generations 10

# Full training
python train_coevolution.py \
    --population-size 12 \
    --generations 100 \
    --trn-count 4 \
    --rsan-count 4 \
    --transformer-count 4 \
    --num-zombies 2 \
    --device cuda
```

### 10.9 Output

Training produces:
- `coevolution_results/coevolution_results.json` - Complete metrics
- `coevolution_results/best_gen_N.pt` - Checkpoints of best agents
- Console output with per-generation statistics

---

## 11. API Reference

### 11.1 Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `SoulMapOntology` | `src.core.ontology` | 181-dim psychological space |
| `BeliefNetwork` | `src.core.beliefs` | Multi-agent belief tracking |
| `RecursiveBeliefState` | `src.core.beliefs` | Single agent's beliefs |
| `TransparentRNN` | `src.agents.architectures` | TRN architecture |
| `RecursiveSelfAttention` | `src.agents.architectures` | RSAN architecture |
| `TransformerToMAgent` | `src.agents.architectures` | Transformer architecture |
| `SocialWorld4` | `src.world.social_world` | Multi-agent environment |
| `NASEngine` | `src.evolution.nas_engine` | Evolution controller |
| `BenchmarkSuite` | `src.evaluation.benchmarks` | ToM test suite |
| `CoevolutionaryTrainer` | `train_coevolution` | Coevo training system |

### 11.2 Key Functions

#### Creating an Agent

```python
from src.agents.architectures import TransparentRNN, RecursiveSelfAttention

# TRN agent
trn = TransparentRNN(
    input_dim=191,
    hidden_dim=128,
    output_dim=181,
    num_layers=2
)

# RSAN agent
rsan = RecursiveSelfAttention(
    input_dim=191,
    hidden_dim=128,
    output_dim=181,
    num_heads=4,
    max_recursion=5
)
```

#### Running a Forward Pass

```python
# Input: (batch, sequence_length, input_dim)
x = torch.randn(1, 10, 191)

output = agent(x)
beliefs = output['beliefs']     # (1, 181)
actions = output['actions']     # (1,)
hidden = output['hidden_states']  # (1, 10, hidden_dim)
```

#### Creating a Social World

```python
from src.world.social_world import SocialWorld4

world = SocialWorld4(
    num_agents=6,
    ontology_dim=181,
    num_zombies=2
)

# Run one step
actions = [{'type': 'cooperate'} for _ in range(6)]
results = world.step(actions, belief_network)
```

#### Running Benchmarks

```python
from src.evaluation.benchmarks import BenchmarkSuite

suite = BenchmarkSuite(device='cpu')
results = suite.run_full_suite(agent_model)
print(f"Score: {results['percentage']:.1f}%")
```

### 11.3 Configuration Options

#### Evolution Config

```python
config = EvolutionConfig(
    population_size=20,
    num_generations=100,
    elite_size=2,
    tournament_size=3,
    mutation_rate=0.1,
    crossover_rate=0.7,
    use_speciation=True,
    use_coevolution=True,
    device='cuda'
)
```

#### Coevolution Config

```python
config = {
    'population_size': 12,
    'trn_count': 4,
    'rsan_count': 4,
    'transformer_count': 4,
    'mutation_rate': 0.1,
    'crossover_rate': 0.3,
    'num_zombies': 2,
    'episodes_per_eval': 5,
    'device': 'cuda',
    'results_dir': 'coevolution_results'
}
```

---

## 12. Usage Guide

### 12.1 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/tom-nas.git
cd tom-nas
pip install torch numpy

# Run demo
python run_complete_demo.py

# Run coevolutionary training
python train_coevolution.py --generations 50
```

### 12.2 Google Colab Setup

```python
!git clone https://github.com/yourusername/tom-nas.git
%cd tom-nas
!pip install torch numpy

# Quick training run
!python train_coevolution.py \
    --population-size 12 \
    --generations 20 \
    --device cuda
```

### 12.3 Custom Training

```python
from train_coevolution import CoevolutionaryTrainer

config = {
    'population_size': 24,
    'trn_count': 8,
    'rsan_count': 8,
    'transformer_count': 8,
    'generations': 100,
    'device': 'cuda'
}

trainer = CoevolutionaryTrainer(config)
results = trainer.train(config['generations'])
```

### 12.4 Analyzing Results

```python
import json

with open('coevolution_results/coevolution_results.json') as f:
    results = json.load(f)

# Best agent info
print(f"Best Fitness: {results['best_fitness']}")
print(f"Best Architecture: {results['best_architecture']}")

# Species evolution
for species, history in results['species_summary']['species_history'].items():
    if history:
        print(f"{species}: Final avg fitness = {history[-1]['avg_fitness']:.4f}")
```

### 12.5 Loading Checkpoints

```python
import torch
from src.agents.architectures import RecursiveSelfAttention

checkpoint = torch.load('coevolution_results/best_gen_50.pt')
print(f"Architecture: {checkpoint['architecture_type']}")
print(f"Fitness: {checkpoint['fitness']}")

# Recreate model
model = RecursiveSelfAttention(191, 128, 181)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Appendix A: Theoretical Background

### A.1 Theory of Mind

Theory of Mind (ToM) is the cognitive ability to attribute mental states to others. It develops in humans around age 4-5 and is fundamental to social cognition.

### A.2 Sally-Anne Test

The Sally-Anne test (Baron-Cohen et al., 1985) is the classic test for false belief understanding. Children with autism spectrum disorder often fail this test, suggesting ToM deficits.

### A.3 Philosophical Zombies

A philosophical zombie (p-zombie) is a hypothetical being physically identical to a human but lacking conscious experience. In ToM-NAS, zombie agents lack genuine ToM despite appearing normal.

### A.4 Coevolution

Coevolution is the mutual evolutionary influence between interacting entities. In ToM-NAS, architectures, evaluation tasks, and the environment all coevolve together.

---

## Appendix B: Troubleshooting

### B.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `NoneType has no attribute 'mutate'` | Parent gene not found | Bug fixed in latest version |
| `CUDA out of memory` | Population too large | Reduce `population_size` or use CPU |
| `ImportError: torch` | PyTorch not installed | `pip install torch` |

### B.2 Performance Tips

1. **Use GPU**: Set `--device cuda` for 10x+ speedup
2. **Reduce population for testing**: Start with `--population-size 6`
3. **Short runs first**: Use `--generations 10` to verify setup
4. **Parallel evaluation**: Enabled by default within the system

---

## Appendix C: Contributing

### C.1 Code Style

- Python 3.8+
- Type hints for all functions
- Docstrings for all classes and methods
- PyTorch for all neural network code

### C.2 Testing

```bash
python test_system.py
python test_comprehensive.py
```

### C.3 Submitting Changes

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

---

*Document generated for ToM-NAS v1.0*
