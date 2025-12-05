# ToM-NAS Architectural Manifest
## A Socio-Computational Scientific Instrument

> **This is NOT a game. This is NOT a chatbot. This is a scientific instrument for solving the Symbol Grounding Problem for Theory of Mind.**

---

## I. Core Thesis

**Hypothesis**: Theory of Mind (ToM) is not a static capability but a *dynamic adaptation to Institutional Friction*. ToM emerges when agents must navigate social structures where survival depends on accurately modeling others' beliefs, intentions, and likely actions.

**Method**: POET-driven co-evolution where "Social Intelligence" is the only adaptation that ensures survival against an increasingly complex "Institutional" environment.

**Mechanism**: The environment is not terrain—it is *Social Structure*. As agents master low-friction institutions (Family), the system introduces higher-friction institutions (Workplace → Political → Adversarial), forcing deeper ToM to maintain performance.

---

## II. Architectural Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 5: EVALUATION                          │
│         Situated Assessment (Belief Accuracy, Not Just Win)     │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 4: EVOLUTION                           │
│              POET (Co-evolve Agents + Institutions)             │
│              NAS (Optimize ToM Module Architecture)             │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 3: COGNITION                           │
│              MetaMind (3-Stage Reasoning Pipeline)              │
│              BeliefNest (Nested Belief Graphs)                  │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 2: INSTITUTIONS                        │
│              Norms, Roles, Information Asymmetry                │
│              (The Selective Pressure)                           │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 1: THE MUNDANE                         │
│              Physical Grounding (Godot 4.x)                     │
│              Maintenance Activities, Material Culture           │
│              (Signal-to-Noise for Robust ToM)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## III. Layer Specifications

### Layer 1: The Mundane (Physical Grounding)

**Source Documents**: "Miscellaneous Components of Human Life"

**Purpose**: Provides the Signal-to-Noise ratio required for robust ToM. If agents only perform "social tasks," the problem is too clean. Real ToM must filter social signals through the noise of daily life.

**Implementation**:

```
Godot Scene
├── Objects tagged with Material Culture
│   ├── Class: Comfort Object (emotional regulation)
│   ├── Class: Status Object (social signaling)
│   ├── Class: Functional Object (task completion)
│   └── Class: Sentimental Object (identity markers)
│
├── Agent Perception Pipeline
│   Agent sees: Feature Vector, NOT raw Godot ID
│   [Location: Kitchen, Object: Coffee, Context: Morning Routine, Time: 7:30]
│
└── Maintenance Constraints
    Agents MUST satisfy:
    - Eating (hunger_threshold < 0.3 → cognitive penalty)
    - Sleeping (fatigue_threshold < 0.2 → ToM depth reduced)
    - Hygiene (social interaction penalty if neglected)
```

**Critical Constraint**: Interrupting another agent's Maintenance Activity incurs social friction. This forces agents to model others' internal states:

> "Do I interrupt Bob's Morning Routine to ask for a favor? No, he is Pre-Coffee. His cooperation probability is low."

### Layer 2: Institutions (The Selective Pressure)

**Source Documents**: "The Institutions Taxonomy"

**Purpose**: Institutions define the *rules of the game* that agents must learn. Critically, these are NORMS, not hard rules—they can be violated, but at cost.

**Institutional Progression** (POET Evolution):

```
EARLY EVOLUTION (Low Friction)
├── Family: High trust, low deception, simple role structure
└── Friendship: Voluntary, low power differential

MIDDLE EVOLUTION (Medium Friction)
├── Education: Authority gradients, evaluation pressure
├── Workplace: Role hierarchy, information asymmetry
└── Healthcare: Expert/layperson gap, vulnerability

LATE EVOLUTION (High Friction)
├── Legal: Adversarial, explicit rules, high stakes
├── Political: Coalitions, public/private personas
└── Economic: Competition, strategic deception

ADVERSARIAL (Maximum Friction)
├── Military: Command hierarchy, life/death stakes
└── Criminal: Trust networks, betrayal dynamics
```

**Institution Genotype** (for POET mutation):

```python
InstitutionGenotype:
  - complexity_level: float       # 0.0 = trivial, 1.0 = adversarial
  - information_asymmetry: float  # Key ToM driver
  - deception_prevalence: float   # How common is strategic deception
  - role_hierarchy_depth: int     # Flat vs. deep hierarchy
  - friction_coefficient: float   # How much norms resist behavior
  - norm_negotiability: float     # Can norms be contested?
```

### Layer 3: Cognitive Architecture (MetaMind + BeliefNest)

**Source Documents**: MetaMind paper, BeliefNest paper, Hypothetical Minds

**Purpose**: Replace naive LLM calls with structured ToM reasoning.

#### 3.1 Prototype/Stereotype Filter (Priors)

When Agent A perceives Agent B for the first time:

```
1. Extract Features: [Wearing_Suit, In_Bank, Confident_Posture]
2. Prototype Lookup: Query Sociological DB
3. Return Prior: {Competence: 0.8, Warmth: 0.3} (Stereotype Content Model)
4. Initialize BeliefNest Node with prior
```

#### 3.2 BeliefNest (State Representation)

A directed graph storing nested beliefs:

```
B_me(B_you(World))  →  "I believe you believe X"
B_me(B_you(B_them(World)))  →  "I believe you believe they believe X"
```

**API Specification**:

```python
class BeliefNest:
    def add_belief(
        self,
        subject: str,           # Who holds this belief
        predicate: str,         # What relation
        object: str,            # About what/whom
        nesting_level: int,     # ToM order
        confidence: float,      # Certainty
        evidence: List[str],    # Grounding observations
    ) -> BeliefNode

    def query_belief(
        self,
        belief_path: List[str],  # e.g., ["me", "bob", "alice"]
        predicate: str,
    ) -> Optional[BeliefNode]

    def get_contradictions(
        self,
        agent: str,
    ) -> List[Tuple[BeliefNode, BeliefNode]]
```

**Update Dynamics**:

```
On Observation:
  - Direct observation: weight = 0.8
  - Inference from behavior: weight = 0.5
  - Social transmission (told by others): weight = 0.3

On Contradiction:
  - If confidence_delta > threshold: trigger belief revision
  - If unresolvable: mark agent as "unpredictable"
```

#### 3.3 MetaMind Pipeline (Reasoning)

Three-stage pipeline replacing single LLM calls:

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: ToM Agent (Hypothesis Generation)                  │
│                                                             │
│ Input: BeliefNest graph + current observation               │
│ Output: Set of hypotheses about other agents' mental states │
│                                                             │
│ Example:                                                    │
│   Observation: "Bob dropped his phone and looked around"    │
│   Hypotheses:                                               │
│     H1: Bob feels embarrassed (p=0.6)                       │
│     H2: Bob is checking if anyone noticed (p=0.7)           │
│     H3: Bob doesn't care (p=0.2)                           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Domain Agent (Institutional Filtering)             │
│                                                             │
│ Input: Hypotheses + Current Institution + Role context      │
│ Output: Institutionally-filtered hypotheses                 │
│                                                             │
│ Example:                                                    │
│   Context: Workplace, Bob is Manager, I am Employee         │
│   Norms: "Managers maintain composure" (high expectation)   │
│   Filtered:                                                 │
│     H1: Bob feels embarrassed → AMPLIFIED (norm violation)  │
│     H2: Bob checking for witnesses → CONFIRMED              │
│     H3: Bob doesn't care → REDUCED (managers care about image)│
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Response Agent (Action Selection)                  │
│                                                             │
│ Input: Filtered hypotheses + Goal + Social cost function    │
│ Output: Selected action                                     │
│                                                             │
│ Example:                                                    │
│   Goal: Maintain good relationship with Bob                 │
│   Options:                                                  │
│     A1: Help pick up phone (social_cost=0.1, goal_fit=0.8) │
│     A2: Pretend not to see (social_cost=0.0, goal_fit=0.6) │
│     A3: Make joke about it (social_cost=0.5, goal_fit=0.3) │
│   Selected: A2 (lowest social cost, decent goal fit)        │
└─────────────────────────────────────────────────────────────┘
```

### Layer 4: Evolutionary Mechanism (POET + NAS)

**Source**: POET paper, tom-nas repository

#### 4.1 POET Co-Evolution

```
OUTER LOOP: Environment-Agent Pairing
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Agent Population          Environment Population           │
│  ┌───┬───┬───┬───┐        ┌───┬───┬───┐                   │
│  │ A1│ A2│ A3│...│        │ E1│ E2│...│                   │
│  └───┴───┴───┴───┘        └───┴───┴───┘                   │
│       │                         │                          │
│       └─────────┬───────────────┘                          │
│                 ▼                                           │
│         ┌─────────────┐                                    │
│         │  EVALUATION  │                                    │
│         └─────────────┘                                    │
│                 │                                           │
│    ┌────────────┼────────────┐                             │
│    ▼            ▼            ▼                              │
│ TRANSFER    MUTATION    EXTINCTION                         │
│ (A1→E2)   (A1'=mutate(A1)) (A3 dies)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Migration Rules:
- If agent achieves performance > 0.8 on current environment:
  → Migrate to harder environment
- If environment has no agents achieving > 0.2:
  → Mutate environment (reduce difficulty)
- If agent/environment pair stagnates:
  → Force transfer to novel pairing
```

#### 4.2 NAS for ToM Modules

```
INNER LOOP: Architecture Optimization
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Search Space (from tom-nas/search_space.py):               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ToM Module Types:                                    │   │
│  │   - Intent Module (infers goals)                     │   │
│  │   - Belief Module (tracks beliefs)                   │   │
│  │   - Emotion Module (recognizes affect)               │   │
│  │   - Norm Module (encodes constraints)                │   │
│  │   - Prediction Module (forecasts behavior)           │   │
│  │                                                      │   │
│  │ Architecture Choices:                                │   │
│  │   - Attention heads per module                       │   │
│  │   - Cross-module connectivity                        │   │
│  │   - Recursive depth (ToM order capacity)             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Zero-Cost Proxies (from tom-nas/proxies.py):               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Rapid architecture scoring WITHOUT full training:    │   │
│  │   - Synflow: gradient flow analysis                  │   │
│  │   - Fisher: parameter importance                     │   │
│  │   - GRASP: gradient signal preservation              │   │
│  │                                                      │   │
│  │ Use Case: Before running expensive Godot simulation, │   │
│  │ estimate which architectures are likely to succeed   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Layer 5: Evaluation (Situated Assessment)

**Source**: Ma et al. "Towards a Science of Evaluating ToM in LLMs"

**Key Insight**: Standard evaluation asks "Did you win?" Situated evaluation asks "Was your internal model accurate?"

```
EVALUATION METRICS
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ PRIMARY: Belief Accuracy (40%)                              │
│   - Compare BeliefNest graph to ground truth                │
│   - Measure: KL divergence between predicted and actual     │
│     beliefs of other agents                                 │
│                                                             │
│ SECONDARY: Action Success (30%)                             │
│   - Did actions achieve stated goals?                       │
│   - Did agent avoid catastrophic norm violations?           │
│                                                             │
│ TERTIARY: Social Cost (20%)                                 │
│   - Total social friction incurred                          │
│   - Relationship damage/improvement                         │
│                                                             │
│ EFFICIENCY: Computational Cost (10%)                        │
│   - ToM depth actually used vs. available                   │
│   - Mental simulation budget consumed                       │
│                                                             │
│ CALIBRATION BONUS:                                          │
│   - Are confidence estimates accurate?                      │
│   - Agent says "80% sure" → is it right 80% of the time?   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## IV. Integration Map

### Module Dependencies

```
src/
├── simulation_config.py      # THE CONSTITUTION (start here)
│
├── core/
│   ├── beliefs.py           # BeliefNetwork (existing) → Wrap with BeliefNest API
│   ├── ontology.py          # SoulMap dimensions → Feed to Sociological DB
│   └── metamind.py          # NEW: 3-stage MetaMind pipeline
│
├── godot_bridge/
│   ├── enhanced_server.py   # BeliefNetwork ↔ Godot integration (done)
│   └── protocol.py          # Message types (done)
│
├── liminal/
│   ├── soul_map.py          # 60-dim psychology → Agent internal state
│   ├── psychosocial_coevolution.py  # Social network dynamics
│   └── game_environment.py  # → Rename to institutional_environment.py
│
├── evolution/
│   ├── poet_manager.py      # NEW: POET outer loop
│   ├── population.py        # Agent/Environment populations
│   └── transfer.py          # Agent migration logic
│
└── evaluation/
    ├── situated_metrics.py  # NEW: Belief accuracy, calibration
    └── fitness.py           # Composite fitness function
```

### Data Flow

```
                    ┌─────────────────┐
                    │  Godot Scene    │
                    │  (Physical)     │
                    └────────┬────────┘
                             │ Perception
                             ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Sociological DB │◄──│ Symbol Grounder │──►│ BeliefNest      │
│ (RAG: Norms,    │   │ (Godot→Semantic)│   │ (Nested Beliefs)│
│  Prototypes)    │   └─────────────────┘   └────────┬────────┘
└─────────────────┘                                  │
        │                                            │
        │         ┌─────────────────────────────────┘
        │         │
        ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                       MetaMind Pipeline                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ ToM Agent    │→ │ Domain Agent │→ │ Response     │      │
│  │ (Hypotheses) │  │ (Norms)      │  │ (Actions)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼ Action
                    ┌─────────────────┐
                    │  Godot Scene    │
                    │  (Execute)      │
                    └────────┬────────┘
                             │ Outcome
                             ▼
                    ┌─────────────────┐
                    │  Evaluator      │
                    │  (Situated)     │
                    └────────┬────────┘
                             │ Fitness
                             ▼
                    ┌─────────────────┐
                    │  POET Manager   │
                    │  (Evolve)       │
                    └─────────────────┘
```

---

## V. Implementation Commands for AI Assistants

When implementing this system, AI coding assistants MUST follow these directives:

### DO:
1. **Always check `simulation_config.py` first** - It is the Constitution
2. **Respect the layer hierarchy** - Lower layers ground higher layers
3. **Use the BeliefNest API** for ALL belief operations - No ad-hoc belief tracking
4. **Run MetaMind pipeline** for ALL agent decisions - No direct LLM calls
5. **Log ToM depth used** - We need to know what order of reasoning agents employ
6. **Ground symbols in Godot** - No "disembodied" reasoning about abstract entities

### DO NOT:
1. **Do NOT skip the Mundane constraints** - Agents have bodies
2. **Do NOT hardcode norms** - Query the Sociological DB
3. **Do NOT evaluate on action success alone** - Belief accuracy matters more
4. **Do NOT use single-shot LLM reasoning** - Use the 3-stage pipeline
5. **Do NOT evolve agents without evolving environments** - POET requires both

### Code Review Checklist:
- [ ] Does this code respect Maintenance Activity constraints?
- [ ] Does this code query the ContextManager for norms?
- [ ] Does this code update the BeliefNest on observation?
- [ ] Does this code use the MetaMind pipeline for decisions?
- [ ] Does this code report ToM depth to the Evaluator?
- [ ] Does this code work with headless Godot for POET scaling?

---

## VI. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12 | Initial Constitution |

---

## VII. References

- **Symbol Grounding Problem**: Harnad, S. (1990). The Symbol Grounding Problem.
- **POET**: Wang, R., et al. (2019). Paired Open-Ended Trailblazer.
- **MetaMind**: Reasoning framework for ToM.
- **BeliefNest**: Nested belief representation.
- **Hypothetical Minds**: Multi-agent simulation for ToM.
- **Stereotype Content Model**: Fiske, S. T., et al. (2002).
- **Situated Evaluation**: Ma, Y., et al. (2024). Towards a Science of Evaluating ToM.

---

*This document is the architectural truth. All implementations must conform to it.*
