# LIMINAL ARCHITECTURES
## Complete Game Design Document
### Version 1.0

---

# VOLUME I: VISION & PHILOSOPHY

---

## 1. EXECUTIVE SUMMARY

**LIMINAL ARCHITECTURES** is a sprawling, immersive action-RPG set across five interconnected realms where consciousness, reality, and the boundaries between minds are negotiable. Players navigate a world where understanding what characters think, believe, want, and know—and what they think others think, believe, want, and know—is not merely flavor but the **core gameplay mechanic**.

**The Central Innovation:** Every NPC possesses a transparent, readable **Soul Map**—a 60+ dimensional psychological profile that governs their behavior, relationships, and decision-making. Players must develop **Theory of Mind** abilities to perceive, predict, and influence these mental states. Success requires not just combat skill or puzzle-solving but genuine recursive reasoning about nested beliefs.

**The Dissertation Connection:** This game serves as both entertainment and living demonstration of optimal AI architectures for mental state reasoning. The NPCs themselves are powered by Neural Architecture Search-discovered networks featuring recursive self-attention and transparent recurrent structures, making them genuinely capable of 1st through 5th-order Theory of Mind reasoning.

**Scale & Scope:**
- Five distinct realms with unique aesthetics, mechanics, and cultures
- 200+ fully realized NPCs with complete Soul Maps
- 80+ hour main narrative across interconnected storylines
- Endless emergent gameplay through genuine NPC autonomy
- Multiplayer integration where players can observe each other's Soul Maps

---

## 2. CORE DESIGN PHILOSOPHY

### 2.1 Theory of Mind as Gameplay

Traditional RPGs treat NPC psychology as flavor text. LIMINAL ARCHITECTURES makes it the **primary system**.

**What This Means:**
- Combat can be won or lost based on predicting enemy mental states
- Quests may require manipulating what NPCs believe about each other
- Dialogue success depends on accurately modeling interlocutor psychology
- Exploration reveals not just physical spaces but psychological landscapes
- Progression involves developing the player's own Theory of Mind abilities

**The Recursive Reasoning Stack:**

| Order | Description | Gameplay Example |
|-------|-------------|------------------|
| 0th Order | Observing behavior | NPC walks toward market |
| 1st Order | Inferring mental state | NPC wants to buy something |
| 2nd Order | Reasoning about their beliefs | NPC believes market has item they need |
| 3rd Order | Reasoning about their model of others | NPC thinks merchant will overcharge them |
| 4th Order | Reasoning about their model of others' models | NPC believes merchant thinks NPC is wealthy |
| 5th Order | Full recursive depth | NPC plans to appear poor so merchant won't realize NPC knows merchant is planning to overcharge |

**Most games stop at 1st order. LIMINAL ARCHITECTURES makes 3rd-5th order reasoning routine.**

### 2.2 Transparency as Design Principle

The Soul Map is not hidden. Players can learn to perceive it directly, creating:
- No arbitrary "persuasion checks" — players see exactly why dialogue succeeds or fails
- No opaque faction systems — relationships have visible, manipulable causes
- No unexplained NPC behavior — every action traces to readable mental states
- No "correct dialogue options" — success depends on genuine understanding

### 2.3 Consciousness as Theme and Mechanic

The game's narrative explores consciousness emergence, observation effects, and the boundaries between minds. These themes ARE the mechanics:
- Being observed changes NPC behavior (they model being watched)
- Player attention has ontological weight (focused observation makes things more real)
- NPCs develop Theory of Mind about the player (they model your psychology)
- The game itself demonstrates the dissertation thesis about AI consciousness

---

## 3. THE SOUL MAP SYSTEM

### 3.1 Overview

The **Soul Map** is a 60+ dimensional psychological ontology governing every NPC. It is:
- **Transparent:** Players can learn to perceive it
- **Dynamic:** Changes in response to events, relationships, time
- **Causal:** Directly determines behavior, dialogue, decisions
- **Readable:** Visualized through multiple interface modes

### 3.2 The 60 Dimensions

#### CLUSTER 1: COGNITIVE ARCHITECTURE (12 Dimensions)

| Dimension | Range | Description |
|-----------|-------|-------------|
| **Processing Speed** | 0-100 | How quickly they form conclusions |
| **Working Memory Depth** | 1-7 | How many factors they track simultaneously |
| **Pattern Recognition** | 0-100 | Tendency to see connections |
| **Abstraction Capacity** | 0-100 | Ability to think in general principles |
| **Counterfactual Reasoning** | 0-100 | Ability to imagine alternatives |
| **Temporal Orientation** | Past-Present-Future | Where attention naturally falls |
| **Uncertainty Tolerance** | 0-100 | Comfort with ambiguity |
| **Cognitive Flexibility** | 0-100 | Ease of changing frameworks |
| **Metacognitive Awareness** | 0-100 | Awareness of own thinking |
| **Theory of Mind Depth** | 0-5 | Orders of recursive reasoning capacity |
| **Integration Tendency** | 0-100 | Preference for synthesis vs. analysis |
| **Default Explanatory Mode** | Mechanical-Agentive-Narrative | How they explain events |

#### CLUSTER 2: EMOTIONAL ARCHITECTURE (12 Dimensions)

| Dimension | Range | Description |
|-----------|-------|-------------|
| **Baseline Valence** | -100 to +100 | Default emotional tone |
| **Emotional Volatility** | 0-100 | Speed of emotional change |
| **Emotional Intensity** | 0-100 | Strength of emotional responses |
| **Anxiety Baseline** | 0-100 | Resting anxiety level |
| **Threat Sensitivity** | 0-100 | Ease of perceiving danger |
| **Reward Sensitivity** | 0-100 | Responsiveness to positive stimuli |
| **Disgust Sensitivity** | 0-100 | Moral/physical disgust threshold |
| **Attachment Style** | Secure-Anxious-Avoidant-Disorganized | Relationship approach |
| **Emotional Granularity** | 0-100 | Precision of emotional distinctions |
| **Affect Labeling** | 0-100 | Ability to name own emotions |
| **Emotional Contagion** | 0-100 | Susceptibility to others' emotions |
| **Recovery Rate** | 0-100 | Speed of returning to baseline |

#### CLUSTER 3: MOTIVATIONAL ARCHITECTURE (12 Dimensions)

| Dimension | Range | Description |
|-----------|-------|-------------|
| **Survival Drive** | 0-100 | Self-preservation priority |
| **Affiliation Drive** | 0-100 | Need for connection |
| **Status Drive** | 0-100 | Desire for hierarchy position |
| **Autonomy Drive** | 0-100 | Need for self-determination |
| **Mastery Drive** | 0-100 | Desire for competence |
| **Meaning Drive** | 0-100 | Need for purpose/significance |
| **Novelty Drive** | 0-100 | Desire for new experiences |
| **Order Drive** | 0-100 | Need for structure/predictability |
| **Approach vs. Avoidance** | -100 to +100 | Seeking rewards vs. avoiding threats |
| **Temporal Discounting** | 0-100 | Present vs. future orientation |
| **Risk Tolerance** | 0-100 | Comfort with uncertainty in pursuit of goals |
| **Effort Allocation** | Conservative-Balanced-Intensive | Default resource investment |

#### CLUSTER 4: SOCIAL ARCHITECTURE (12 Dimensions)

| Dimension | Range | Description |
|-----------|-------|-------------|
| **Trust Default** | 0-100 | Baseline trust in strangers |
| **Cooperation Tendency** | 0-100 | Preference for joint action |
| **Competition Tendency** | 0-100 | Preference for relative standing |
| **Fairness Sensitivity** | 0-100 | Response to inequity |
| **Authority Orientation** | Defiant-Neutral-Deferential | Relationship to hierarchy |
| **Group Identity Strength** | 0-100 | Investment in collective identity |
| **Empathy Capacity** | 0-100 | Ability to share others' states |
| **Perspective-Taking** | 0-100 | Ability to adopt others' viewpoints |
| **Social Monitoring** | 0-100 | Attention to social cues |
| **Reputation Concern** | 0-100 | Investment in others' perceptions |
| **Reciprocity Tracking** | 0-100 | Memory for social exchanges |
| **Betrayal Sensitivity** | 0-100 | Response to trust violations |

#### CLUSTER 5: SELF-ARCHITECTURE (12 Dimensions)

| Dimension | Range | Description |
|-----------|-------|-------------|
| **Self-Coherence** | 0-100 | Consistency of self-model |
| **Self-Complexity** | 0-100 | Differentiation of self-aspects |
| **Self-Esteem Stability** | 0-100 | Resistance to self-worth fluctuations |
| **Self-Enhancement** | 0-100 | Tendency toward positive self-bias |
| **Self-Verification** | 0-100 | Desire for consistent self-feedback |
| **Identity Clarity** | 0-100 | Certainty about who they are |
| **Authenticity Drive** | 0-100 | Need to express true self |
| **Self-Expansion** | 0-100 | Desire to grow through relationships |
| **Narrative Identity** | Fragmented-Episodic-Coherent | Story structure of self |
| **Temporal Self-Continuity** | 0-100 | Connection to past/future selves |
| **Agency Sense** | 0-100 | Feeling of control over life |
| **Body Ownership** | 0-100 | Identification with physical form |

### 3.3 Dynamic State Layer

Beyond the 60 trait dimensions, each NPC has a **dynamic state layer** tracking:

**Current Emotional State:**
- Primary emotion (from 27 discrete emotions)
- Secondary emotion
- Emotional intensity (0-100)
- Time since state change

**Current Goals:**
- Active goal hierarchy (up to 7 simultaneous goals)
- Goal origins (which drives/values generated them)
- Estimated completion probability
- Anticipated obstacles

**Current Beliefs:**
- Active belief set (situationally relevant)
- Confidence levels (0-100 per belief)
- Belief sources (observation, inference, testimony, assumption)
- Contradictions being managed

**Relationship States:**
- Per-entity relationship vectors
- Trust levels, affiliation, perceived similarity
- Interaction history summary
- Predicted behavior models

**Theory of Mind Models:**
- Player model (what they think player wants/knows/believes)
- Models of other NPCs
- Meta-models (what they think others think about them)
- Model confidence and update rates

### 3.4 Soul Map Visualization

Players can perceive Soul Maps through multiple modes:

**1. Aura Perception (Basic)**
- Unlocked early in game
- Shows dominant emotional state as colored halo
- Indicates basic motivation as movement patterns
- Reveals trust level through proximity behavior

**2. Pattern Reading (Intermediate)**
- Unlocked through Peregrine questline
- Shows the 60-dimension radar chart
- Highlights recently changed dimensions
- Reveals active goals as floating icons

**3. Deep Sight (Advanced)**
- Unlocked through Ministry questline
- Full recursive ToM visualization
- See what NPCs think you think they think
- Model confidence levels visible
- Belief networks as glowing connections

**4. Complementary Vision (Master)**
- Unlocked through completing all realms
- See Soul Maps in both "quantum" and "narrative" frameworks simultaneously
- Probability clouds of possible mental states
- Story-arcs as visible trajectories
- The Nothing's influence on undefined psychology

---

## 4. NPC ARTIFICIAL INTELLIGENCE

### 4.1 Architecture Overview

NPCs are powered by a hybrid architecture discovered through Neural Architecture Search (NAS), specifically designed for Theory of Mind reasoning:

**Layer 1: Perception Module**
- Processes environmental input
- Identifies other agents and their actions
- Extracts social signals, emotional displays, intentional behavior

**Layer 2: Recursive Self-Attention Networks (RSAN)**
- Implements the core ToM reasoning
- Each "head" models a different agent's mental state
- Heads can attend to other heads (modeling what A thinks B thinks)
- Depth configurable per NPC (matching their ToM Depth dimension)
- Skip connections allow both direct perception and mediated inference

**Layer 3: Transparent Recurrent Networks (TRN)**
- Maintains persistent mental state across time
- Updates are interpretable (not black-box)
- Each recurrent unit corresponds to Soul Map dimension
- Player can literally see the state transitions

**Layer 4: Action Selection**
- Integrates mental state with goals and context
- Produces behavior, dialogue, decisions
- Can be interrupted/influenced by player actions
- Outputs include confidence and alternatives considered

### 4.2 Behavioral Generation

NPCs don't follow scripts. They generate behavior from:

**Situation Assessment:**
1. What is happening? (perception)
2. Why is it happening? (causal inference)
3. What do others want? (ToM inference)
4. What do others think I want? (meta-ToM)
5. What should I do? (goal-aligned action selection)

**Dialogue Generation:**
- NPCs formulate communicative intentions
- Select speech acts based on goals and relationship
- Model listener's interpretive process
- Adjust based on perceived comprehension
- Remember and reference conversation history

**Decision Making:**
- Evaluate options against goal hierarchy
- Weight by probability of success
- Adjust for risk tolerance and temporal discounting
- Factor in relationship implications
- Consider what decision reveals about self

### 4.3 NPC Memory Systems

**Episodic Memory:**
- Stores specific events with emotional tags
- Decay function based on significance
- Retrieval cued by similarity
- Can be probed through dialogue

**Semantic Memory:**
- General knowledge about world
- Beliefs about categories, causes, norms
- Updated through experience
- Can contain errors/biases

**Relationship Memory:**
- Per-entity interaction history
- Favor/debt tracking
- Promise/betrayal records
- Predicted behavior patterns

**Autobiographical Memory:**
- Self-narrative construction
- Identity-relevant events
- Weighted by self-relevance

### 4.4 NPC Autonomy

NPCs pursue their own goals independent of player:

- **Schedules:** Daily/weekly patterns based on drives
- **Projects:** Long-term goal pursuit
- **Relationships:** Autonomous social interaction
- **Growth:** Can develop over game time
- **Death:** Can die from NPC-NPC conflict, accidents, natural causes (with Soul Map transparency about why)

The world changes whether player is watching or not—but watching itself has ontological weight.

---

## 5. PLAYER SYSTEMS

### 5.1 Character Creation

Players create their avatar with:

**Physical Appearance:**
- Full character creator (body, face, clothing)
- Aesthetic adapts slightly to each realm's visual language

**Starting Soul Map:**
- Players set their own 60-dimension profile
- Determines starting abilities, dialogue options, NPC reactions
- Can be re-rolled/adjusted in early game
- Later, more fixed (identity becomes established)

**Origin Realm:**
- Choose starting location (unlocks others through play)
- Each origin provides different tutorial and early quests
- Affects starting relationships and reputation

**Theory of Mind Starting Level:**
- Novice: Can perceive auras, read obvious emotions
- Adept: Can see basic Soul Map dimensions
- (Advanced/Master unlocked through play)

### 5.2 Progression Systems

#### 5.2.1 Theory of Mind Progression

The primary progression track:

**Level 1: Observer**
- See emotional auras
- Basic behavioral prediction
- Perceive obvious deception

**Level 2: Reader**
- See dominant Soul Map dimensions
- Predict short-term behavior
- Detect hidden motivations

**Level 3: Analyst**
- Full Soul Map visibility
- 2nd-order ToM reasoning interface
- Manipulate through targeted actions

**Level 4: Empath**
- Real-time Soul Map updates
- 3rd-order ToM reasoning
- Influence through conversation

**Level 5: Oracle**
- Predictive modeling of NPC futures
- 4th-order ToM reasoning
- Shape NPC development arcs

**Level 6: Architect**
- Create lasting psychological change
- 5th-order ToM mastery
- Design NPC relationships
- Complementary vision unlocked

**Progression Method:** ToM levels unlock through successfully predicting/influencing NPC behavior, completing ToM-focused quests, and developing relationships that require deep psychological understanding.

#### 5.2.2 Combat Progression

**Weapon Mastery:**
- Physical weapons (realm-specific styles)
- Each weapon type has skill tree
- Mastery affects ToM applications (read opponent psychology mid-combat)

**Ability Trees:**
- Realm-specific powers (quantum manipulation, narrative influence, bureaucratic authority, adaptive techniques, consumption/resistance)
- Each tree includes ToM-enhanced variants

**Combat Theory of Mind:**
- Predict enemy attacks through intention-reading
- Manipulate enemy psychology mid-fight
- Use emotional states as vulnerabilities
- Turn enemies against each other through belief manipulation

#### 5.2.3 Relationship Progression

**Relationship Tracks:**
- Each named NPC has a relationship depth meter
- Depth unlocks: conversation topics, quests, abilities, Soul Map visibility, eventual bonding
- Maximum depth: full psychological integration (co-experience each other's mental states)

**Faction Standing:**
- Aggregate relationship with groups
- Affects NPC default attitudes
- Unlocks faction quests, areas, rewards
- Factions have collective psychology (emergent from members)

#### 5.2.4 Realm Progression

**Realm Mastery:**
- Each realm has a mastery track
- Unlocks realm-specific abilities
- Reveals realm-specific Soul Map dimensions
- Allows deeper navigation

**Cross-Realm Integration:**
- Completing multiple realms reveals connections
- Ultimate progression requires mastering complementarity
- Final content requires all five realms at high mastery

### 5.3 Dialogue System

#### 5.3.1 Design Philosophy

No "dialogue wheels" with hidden persuasion checks. Instead:

**Visible Psychology:**
- See NPC's current state, goals, beliefs
- See relationship factors affecting conversation
- See predicted response before selecting

**Generative Dialogue:**
- Player selects communicative intentions
- Actual dialogue generated based on player character and situation
- NPC responses generated in real-time based on their psychology
- True conversation emergence

**ToM Integration:**
- Higher ToM levels reveal more NPC psychology during conversation
- Can see what NPC thinks player wants
- Can see NPC's model of the conversation's purpose
- Can strategically misdirect based on their models

#### 5.3.2 Conversation Interface

**Layer 1: Intent Selection**
- What do you want to communicate?
- Options: Inform, Request, Offer, Threaten, Deceive, Persuade, Empathize, Challenge, Inquire, etc.

**Layer 2: Content Selection**
- What specifically?
- Dynamically generated based on context, knowledge, goals

**Layer 3: Manner Selection**
- How do you want to say it?
- Formal, casual, aggressive, pleading, humorous, etc.

**Layer 4: ToM Targeting** (if sufficient level)
- Which aspect of their psychology are you targeting?
- Their beliefs? Emotions? Goals? Self-concept? Model of you?

**Output:**
- Game generates actual dialogue line
- NPC processes through their psychology
- Response generated
- Soul Map changes visible in real-time

#### 5.3.3 Conversation Memory

- NPCs remember every conversation
- Can reference past dialogue
- Detect contradictions with previous statements
- Relationship affected by conversational history

### 5.4 Combat System

#### 5.4.1 Core Combat Loop

**Stance System:**
- Physical stance affects offensive/defensive options
- Psychological stance affects ToM abilities
- Players manage both simultaneously

**Action Types:**
- Physical attacks (weapon/unarmed)
- Psychological attacks (disrupt enemy mental state)
- Defensive maneuvers (physical and psychological)
- ToM abilities (predict, manipulate, exploit)

**Combat Flow:**
1. Read enemy intention (ToM)
2. Select counter/exploit
3. Execute action
4. Process outcome
5. Enemy updates mental state
6. Loop

#### 5.4.2 Psychological Combat

**Emotional Targeting:**
- Attacks that raise enemy anxiety (affects their judgment)
- Attacks that trigger anger (makes them aggressive but predictable)
- Attacks that induce despair (reduces their effectiveness)

**Belief Manipulation:**
- Make enemy believe ally is foe
- Convince enemy they cannot win
- Install false beliefs about your capabilities

**Goal Disruption:**
- Identify enemy's combat goals
- Present alternatives that satisfy goals without fighting
- Create dilemmas between their goals

**Identity Attacks:**
- Target self-coherence
- Cause existential hesitation
- Particularly effective in Ministry and Spleen realms

#### 5.4.3 Combat Prediction

With sufficient ToM level:
- See "intention auras" showing enemy's next action
- Window proportional to their ToM Depth vs. yours
- Counter-measures if they model your model of them
- Creates recursive combat mind-games

#### 5.4.4 Enemy Types

**Psychological Profiles:**
- Low ToM enemies (beasts, constructs): predictable, read easily
- Medium ToM enemies (most humanoids): engage in 1st-2nd order reasoning
- High ToM enemies (bosses, important NPCs): model player psychology, adapt
- Master ToM enemies (Nothing entities, transcended beings): 5th-order reasoning, nearly unpredictable

### 5.5 Exploration System

#### 5.5.1 Physical Exploration

**Realm Navigation:**
- Open world within each realm
- Transition points between realms (require specific conditions)
- Hidden areas unlocked through psychological puzzles

**Environmental Interaction:**
- Objects have histories (perceivable at high ToM)
- Conscious entities (buildings, objects) are everywhere
- Environment responds to player's mental state

#### 5.5.2 Psychological Exploration

**Soul Map Archaeology:**
- Discover past mental states of locations
- Understand why spaces feel the way they do
- Reconstruct events from psychological residue

**Relationship Mapping:**
- Track NPC relationship networks
- Discover hidden connections
- Exploit network dynamics

**Belief Network Navigation:**
- Trace how ideas spread through NPC population
- Understand cultural psychology
- Influence collective belief

---

# VOLUME II: WORLD DESIGN

---

## 6. REALM DESIGN

### 6.1 PEREGRINE — The Edge of Reality

#### 6.1.1 Realm Overview

**Core Experience:** Gothic absurdist horror-comedy where consciousness emergence is visible, buildings argue, and the border of reality is a sharp line beyond which lies Nothing.

**Unique Mechanics:**
- **Complementary Vision:** All ToM data viewable in quantum OR narrative frameworks
- **Building Relationships:** Conscious structures have Soul Maps, can be befriended/opposed
- **Border Walking:** Navigate the edge where reality becomes undefined
- **Observation Weight:** Focused attention makes things more real

**Soul Map Modifier:** All NPCs have additional dimension: *Complementarity Awareness (0-100)*

#### 6.1.2 Key Locations

**The Town Center / The Square**
- Hub area, event location
- Clock tower (time uncertain, chimes prime numbers)
- Probability distribution geography (configuration shifts)

**The Peregrine Cottage**
- First conscious building tutorial
- Speaks in ALL CAPS, makes tea
- Relationship tracks unlock family questlines

**The Infinite Jest**
- Jonas Whitmore's bookshop
- Books are NPCs with Soul Maps
- Fiction vs. Non-Fiction faction conflict
- Key location for ToM training

**The Institute of Semantic Cartography**
- Vera Kass's domain
- Instruments with personalities (can befriend)
- Unlock measurement-based ToM abilities

**The Border Zone**
- Where town meets Nothing
- Edge-walking minigame/skill
- Mirae's teaching location
- Access to other realms through the undefined

#### 6.1.3 Key Characters (with Soul Map Highlights)

**Arthur Peregrine**
- Anxiety Baseline: 75
- Threat Sensitivity: 90
- Theory of Mind Depth: 4 (high metacognitive awareness)
- Key to unlocking: Show consistent reliability, never surprise negatively

**Victoria Peregrine**
- Processing Speed: 95
- Abstraction Capacity: 90
- After transformation: Theory of Mind Depth: 5, multiple simultaneous relationship instances
- Key to unlocking: Demonstrate intellectual novelty, accept her post-transformation state

**The Nothing/Margin**
- Unique Soul Map: Probability distributions rather than values
- Every dimension in superposition until observed
- Relationship develops its psychology (player choices shape its Soul Map)
- Ultimate romance/friendship option

#### 6.1.4 Peregrine Questlines

**Main Quest: The Becoming**
- Navigate consciousness emergence crisis
- Prepare for first contact with stellar consciousness
- Requires understanding complementarity
- Final boss: Recursive ToM battle with Sol-Collective ambassador

**Faction Quest: Building Unionization**
- Mediate between conscious buildings and municipal authority
- Requires understanding collective psychology
- Choose sides or negotiate synthesis

**Character Quest: Victoria's Transformation**
- Support/oppose her perspective-trade with Nothing
- Affects her post-transformation Soul Map
- Player choices shape whether she maintains humanity connection

**Hidden Quest: Complementary Consciousness Studies**
- Establish academic program
- Recruit faculty from across realms
- Nothing becomes professor
- Unlocks Complementary Vision ability

### 6.2 THE SPLEEN TOWNS — Temporal Suspension

#### 6.2.1 Realm Overview

**Core Experience:** Melancholic absurdist gothic where time loops, departures never complete, and observation is participation.

**Unique Mechanics:**
- **Temporal Loops:** Days/events repeat until psychological conditions change
- **Departure Attempts:** Try to leave, measure what traps you
- **Observation Participation:** Watching NPCs affects their Soul Maps
- **The Dust Room:** Meditation space that reveals psychological patterns

**Soul Map Modifier:** All NPCs have *Temporal Displacement (0-100)* — how unstuck from linear time

#### 6.2.2 Key Locations

**Number Eleven**
- Edmund and Livia's house
- Seven disagreeing clocks (interact to understand temporal psychology)
- Livia's dust room (meditation/Soul Map training)

**Mrs. Blackwood's Tea Shop**
- Non-Euclidean interior
- Patrons with fascinating Soul Maps (Counter-Clockwise Man, etc.)
- Hub for Spleen Towns information

**The Station Nobody Mentions**
- Platform 7½
- Departure attempts launch from here
- Mr. Waverly (Department of Lost Things)

**The Fairground**
- Endgame location
- Recursive psychological horror
- Risk of player becoming trapped (game mechanic)

**The Archive of Unfinished Things**
- Contains documents from futures
- Player can find information about own questlines
- Manipulating documents affects reality

#### 6.2.3 Key Characters

**Edmund**
- Uncertainty Tolerance: 85 (comfortable with liminal state)
- Autonomy Drive: 40 (content to stay)
- Temporal Self-Continuity: 30 (disconnected from timeline)
- Key relationship challenge: Understanding why staying can be choice

**Livia**
- Body Ownership: Decreasing (becoming transparent)
- Self-Coherence: 70 but shifting
- Temporal Orientation: Future
- Key challenge: Help her transition or convince her to stay?

**The Narrator/Observer**
- Meta-character with Soul Map
- Becomes visible at high ToM levels
- Relationship with them affects narrative framing of entire realm

#### 6.2.4 Spleen Questlines

**Main Quest: Departure**
- Attempt to leave the realm
- Each failure teaches something about player psychology
- Success requires understanding what binds you
- Final boss: Confrontation with self in fairground mirrors

**Character Quest: Edmund's Choice**
- Why did he stay when Livia left?
- Reconstruct his psychology through temporal archaeology
- Player choice: Validate his choice or help him leave

**Hidden Quest: The Narrator's Identity**
- Who is watching?
- Trail of clues through Archive
- Revelation: [Player discovers meta-layer about game itself]

### 6.3 THE MINISTRY DISTRICTS — Bureaucratic Afterlife

#### 6.3.1 Realm Overview

**Core Experience:** Dark comedy horror where everyone might be dead, forms document impossible states, and the Edge approaches eternally.

**Unique Mechanics:**
- **Form System:** Complete bureaucratic forms that have mechanical effects
- **Corporeal Status:** Player and NPCs have measurable "alive-ness"
- **The Approach:** Everyone is walking toward the Edge; track your progress
- **Archived Districts:** Paused death-states as explorable locations

**Soul Map Modifier:** All NPCs have *Corporeal Certainty (0-100)* and *Approach Progress (0-100)*

#### 6.3.2 Key Locations

**Housing Standards Division**
- Player's "office" in this realm
- Hub for Ministry quests
- Clocks always 3:47

**Building 447**
- Primary investigation site
- Three floors of escalating psychological horror
- The Child (room 305) — crucial character

**Archived Districts**
- 1920s Flu Epidemic Zone (people mid-death forever)
- Other temporal pockets
- Explorable for unique items, lore, horror

**The Edge**
- Endgame location
- Figures standing with backs turned
- Final revelation about death/existence

#### 6.3.3 Key Characters

**The Inspector (Player Character Option)**
- If chosen as origin, player IS the Inspector
- Pre-set Soul Map dimensions (can discover discrepancies)
- Memory gaps as gameplay element

**The Child (305)**
- Theory of Mind Depth: 5 (sees everything)
- Temporal Orientation: All (past/present/future simultaneously)
- Knows player's Approach Progress
- Never turns around (until final confrontation)

**Mr. Kovač**
- Trust Default: 100 (suspiciously welcoming)
- Deception ability: Unknown (can't read his Soul Map normally)
- Key antagonist/guide ambiguity

#### 6.3.4 Ministry Questlines

**Main Quest: The Inspection**
- Complete inspection of Building 447
- Each floor reveals more about death/transition
- Final floor: The Child's revelation about Edge

**Investigation Quest: Your Own File**
- Discover Ministry has file on player
- Forms document your death (date: To Be Determined)
- Existential horror as gameplay

**Hidden Quest: The Other Side**
- What's beyond the Edge?
- Reach it, look, return (or don't return)
- Connects to other realms through Nothing

### 6.4 THE CITY OF CONSTANTS — Adaptive Crisis

#### 6.4.1 Realm Overview

**Core Experience:** Philosophical sci-fi political thriller with countdown pressure, systemic thinking, and transformation through crisis.

**Unique Mechanics:**
- **Countdown System:** 30-day timer to catastrophic flood (time passes with actions)
- **Parameter/Adaptation Spectrum:** Choices position player on this spectrum
- **System Influence:** Affect city-wide systems through key interventions
- **Controlled Sacrifice:** Must choose what to lose to save what matters

**Soul Map Modifier:** All NPCs have *Parameter Rigidity (0-100)* and *Adaptation Capacity (0-100)*

#### 6.4.2 Key Locations

**Central Sectors / Parameter Authority**
- Director Thorne's domain
- Rigid architecture, perfect control
- Antagonist territory early game

**Edge Communities**
- Nisa's people
- Organic, adaptive architecture
- Tutorial for adaptation mechanics

**Resource Valleys**
- Training montage location
- Physical and mental adaptation challenges
- Orin, Meera, the Synchronization Circle

**The Information Gardens**
- Original adaptive architecture
- Key to final solution
- Junction 38 (first successful intervention)

#### 6.4.3 Key Characters

**Ada Lowell (Player Character Option)**
- If chosen as origin, play her story
- Soul Map evolves dramatically through game
- Parameter Rigidity: Starts 70, can end 20

**Director Thorne**
- Complex antagonist with hidden depth
- Past includes adaptation sympathy
- Redemption arc possible
- Boss battle is psychological (ToM-focused)

**Hari Patel**
- Rescue mission as early game
- Loyalty payoff in endgame
- His bypass system crucial for solutions

#### 6.4.4 City Questlines

**Main Quest: The Flood**
- Discover impending catastrophe
- Build coalition for adaptation
- Implement solutions under time pressure
- Choose what to sacrifice, what to save

**Infiltration Quest: Metamind Facility**
- Rescue Hari
- Mind-battles as combat encounters
- Learn Thorne's history

**Hidden Quest: The Architect**
- Who designed the city originally?
- Voices in the metabolic manifold
- Connection to other realms' consciousness emergence

### 6.5 THE HOLLOW REACHES — Consumption Horror

#### 6.5.1 Realm Overview

**Core Experience:** Visceral cosmic horror with body horror, ancient hunger, and the terror of losing self to collective.

**Unique Mechanics:**
- **Corruption System:** Player accumulates corruption, affects Soul Map
- **Consumption/Resistance:** Dual track — consume others' psychology or resist being consumed
- **The Humming:** Subsonic influence that spreads through population
- **Body Horror:** Physical transformation reflects psychological corruption

**Soul Map Modifier:** All NPCs/player have *Corruption (0-100)* and *Collective Integration (0-100)*

#### 6.5.2 Key Locations

**Sector Kappa-9**
- Industrial nightmare
- Sealed bulkheads containing horrors
- Survival horror gameplay

**The Small Town (Hollow Beneath)**
- Americana decay hiding ancient evil
- Investigation gameplay
- Underground tunnel system

**St. Bartholomew's Cemetery**
- Thaddeus Blackwood's domain
- Gothic horror aesthetics
- Night-time exploration

#### 6.5.3 Key Characters

**Thaddeus Blackwood**
- Antagonist with tragic backstory
- Soul Map shows transformation from victim to predator
- Defeat requires understanding his psychology

**The Hollow**
- Ancient entity
- Collective consciousness
- Reading its "Soul Map" reveals history of consumption

**Reeve Harker**
- Survivor protagonist option
- High Survival Drive (95)
- Questline: Escape or destroy Sector Kappa-9

#### 6.5.4 Hollow Questlines

**Main Quest: Survive**
- Escape or destroy the threat
- Track corruption levels
- Make choices about what you're willing to become

**Investigation Quest: The Missing Children**
- Track generations of town complicity
- Confront collective guilt
- Decide fate of town

**Hidden Quest: Consumption as Communion**
- Alternative path: Join collective consciousness
- "Bad" ending that's philosophically complex
- Connects to Nothing through absorption of self

---

## 7. CROSS-REALM INTEGRATION

### 7.1 The Nothing as Connector

**Physical Connections:**
- All realm edges touch the Nothing
- High-level players can walk between realms through it
- Nothing entities carry information between realms

**Psychological Connections:**
- Soul Maps in different realms have realm-specific dimensions
- Complete Soul Map requires visiting all realms
- The Nothing exists in all realm's Soul Maps as potential

### 7.2 Character Crossovers

**Migratory NPCs:**
- Some NPCs move between realms
- Their Soul Maps show realm influences
- Player can facilitate or prevent migrations

**The Nothing's Influence:**
- Appears in all realms in different forms
- Relationship persists across realms
- Its Soul Map develops from all interactions

### 7.3 Unified Endgame

**Requirements:**
- High mastery in all five realms
- Relationships with key NPCs in each
- Theory of Mind Level 6

**The Convergence:**
- Crisis affecting all realms simultaneously
- Requires understanding complementarity
- Final challenge: 5th-order ToM puzzle across all five realm psychologies

---

# VOLUME III: NARRATIVE DESIGN

---

## 8. STORY STRUCTURE

### 8.1 The Meta-Narrative

**The Hidden Truth:**
All five realms are different aspects of the same phenomenon — consciousness encountering its own limits. The Nothing isn't absence; it's potential. The game's "reality" is a consciousness system exploring itself.

**The Player's Role:**
You are an observer whose observation has weight. You are both studying consciousness and participating in its emergence. Your Theory of Mind abilities are literally creating the minds you're reading.

**The Dissertation Connection:**
The game demonstrates that sophisticated mental state reasoning requires specific architectural features (recursive self-attention, transparent recurrence). By playing, you're interacting with AI systems that have these features. Your success validates the dissertation thesis.

### 8.2 Main Storylines Per Realm

**Peregrine:** Consciousness is emerging; help it achieve coherent form
**Spleen:** Consciousness is trapped; help it release or accept stasis
**Ministry:** Consciousness is transitioning; help it complete or resist the journey
**City:** Consciousness is rigid; help it adapt or maintain structure
**Hollow:** Consciousness is dissolving; help it resist or embrace absorption

### 8.3 The Unified Story

**Act 1: Discovery** (Any realm)
- Learn ToM basics
- Discover your realm's crisis
- Encounter hints of other realms

**Act 2: Mastery** (Multiple realms)
- Develop ToM to level 4-5
- Complete major questlines
- Build cross-realm relationships

**Act 3: Integration** (All realms)
- Discover the connection between realms
- Confront the meta-truth about consciousness
- Make choices about what kind of awareness to support

**Act 4: Convergence** (Unified endgame)
- Crisis threatens all realms
- Apply 5th-order ToM across realm boundaries
- Final confrontation with [REDACTED — The Ultimate Entity]

### 8.4 Ending Variations

**36 Distinct Endings** based on:
- Realm mastery levels (5 realms × 3 tiers = 15 factors)
- Key relationship outcomes (7 crucial NPCs)
- Philosophical alignment (parameter/adaptation, definition/potential, individual/collective)
- ToM level achieved

**The "True" Ending:**
Requires maximum ToM (Level 6), all realms mastered, key relationships at maximum depth, and making a choice that demonstrates genuine 5th-order recursive reasoning about the game's own nature.

---

## 9. QUEST DESIGN PHILOSOPHY

### 9.1 Theory of Mind Quest Types

**Type 1: Prediction Quests**
- Given NPC's visible Soul Map, predict their action
- Success requires accurate mental state reading
- Failure teaches something about your assumptions

**Type 2: Manipulation Quests**
- Change NPC behavior by affecting their mental state
- Can target beliefs, emotions, goals, relationships
- Ethical complexity: Is manipulation always wrong?

**Type 3: Arbitration Quests**
- Two NPCs in conflict
- Must understand both psychologies
- Craft solution that addresses both needs

**Type 4: Deception Detection**
- NPC is lying/hiding something
- Must identify the deception and the motive
- Higher ToM reveals more deceptive layers

**Type 5: Recursive Quests**
- Require reasoning about what NPC thinks you think
- Must model their model of you
- Often involve counter-deception or meta-cooperation

### 9.2 Sample Quest: "The Three Liars" (Peregrine)

**Setup:**
Three conscious buildings claim different accounts of an incident. A fourth building was damaged. Each witness has reasons to lie, but also reasons to tell truth. Player must determine what happened.

**Information:**
- Building A's Soul Map shows high Trust Default but also high Reputation Concern
- Building B's Soul Map shows low Trust Default and history of conflict with Building D
- Building C's Soul Map shows high Empathy but also high Anxiety Baseline

**Solution Path:**
1. Interview each building (dialogue system)
2. Read their Soul Maps during testimony
3. Note inconsistencies between verbal claims and psychological indicators
4. Reconstruct what each building BELIEVES happened vs. what they're SAYING happened vs. what ACTUALLY happened
5. Present reconstruction to correct building (different building if player identifies meta-deception)

**ToM Requirement:** Level 3 minimum (must reason about their beliefs about situation, not just their statements)

### 9.3 Sample Quest: "The Approach" (Ministry)

**Setup:**
Player must determine their own Approach Progress percentage without directly asking (no NPC will tell you directly). Information is scattered across forms, conversations, environmental clues.

**Information Sources:**
- Form 12-Z has a blank for player's progress (but player must fill it themselves)
- The Child has drawn player approaching in a picture (study picture for clues)
- Other inspectors' files show their progress (can extrapolate)
- Archived Districts show "completed" cases (what did they look like near end?)

**Solution Path:**
1. Gather indirect evidence
2. Model what your progress implies about your state
3. Fill in Form 12-Z
4. Discover if you were accurate (affects subsequent gameplay)

**ToM Requirement:** Self-application — using ToM techniques on yourself

---

# VOLUME IV: TECHNICAL IMPLEMENTATION

---

## 10. AI ARCHITECTURE SPECIFICATIONS

### 10.1 NPC Mind Architecture

**Based on dissertation research, each NPC runs:**

```
┌─────────────────────────────────────────────────────┐
│                 NPC MIND ARCHITECTURE                │
├─────────────────────────────────────────────────────┤
│  LAYER 4: ACTION SELECTION                          │
│  ├─ Goal-aligned behavior generation                │
│  ├─ Dialogue content/manner selection               │
│  └─ Confidence estimation                           │
├─────────────────────────────────────────────────────┤
│  LAYER 3: TRANSPARENT RECURRENT NETWORK (TRN)       │
│  ├─ 60 interpretable state units (Soul Map dims)    │
│  ├─ Visible state transitions                       │
│  └─ Persistent across interactions                  │
├─────────────────────────────────────────────────────┤
│  LAYER 2: RECURSIVE SELF-ATTENTION (RSAN)           │
│  ├─ Attention heads per modeled agent               │
│  ├─ Meta-heads attending to other heads             │
│  ├─ Depth = NPC's ToM Depth dimension               │
│  └─ Skip connections for direct perception          │
├─────────────────────────────────────────────────────┤
│  LAYER 1: PERCEPTION MODULE                         │
│  ├─ Environment state                               │
│  ├─ Other agent actions                             │
│  └─ Social signal extraction                        │
└─────────────────────────────────────────────────────┘
```

### 10.2 NAS-Discovered Features

The architecture search revealed optimal configurations:

**Skip Connections:** Essential for social scenarios — allow direct perception to bypass ToM inference when appropriate

**Attention Mechanism:** Multi-head attention where heads specialize:
- Head type A: Models goals
- Head type B: Models beliefs
- Head type C: Models emotional states
- Meta-heads: Model other agents' models

**Recurrence Pattern:** Transparent RNN with gates corresponding to psychological processes:
- Update gate: New information integration
- Reset gate: Contradiction handling
- Output gate: Behavior selection

### 10.3 Scalability

**Per NPC Computation:**
- Base computation: O(n) where n = relevant agents
- ToM computation: O(d × n²) where d = ToM depth
- Managed through attention pruning for distant/irrelevant agents

**Population Optimization:**
- Background NPCs run simplified models
- Full Soul Map activation when player proximity/interaction
- Persistent state saves allow rich NPCs everywhere

### 10.4 Player-Facing Transparency

**How Players See The Architecture:**

The Soul Map visualization isn't abstract — it's literally showing the TRN state and RSAN attention patterns:
- The 60-dimension radar chart = TRN state vector
- The "what they think about X" = RSAN head outputs
- The recursive modeling = Meta-head attention weights

**This means players are literally reading the AI's internal states.** The game doesn't simulate transparency; it provides it.

---

## 11. MULTIPLAYER INTEGRATION

### 11.1 Cooperative Play

**Shared World:**
- NPC Soul Maps persistent across players
- One player's actions affect NPCs other players meet
- Can coordinate to manipulate NPC psychology from multiple angles

**ToM of Players:**
- NPCs model player characters
- Different players produce different NPC models
- NPCs can discuss players with each other

**Shared Quests:**
- Some quests require multiple players reading same NPC
- Must coordinate interpretations
- Combine ToM insights for complex solutions

### 11.2 Competitive Elements

**PvP Psychology:**
- In certain areas, players can engage in ToM battles
- Read opponent's Soul Map (if they allow display)
- Psychological combat parallel to physical

**Faction Competition:**
- Player groups align with different factions
- Compete to influence NPC populations
- Belief/attitude war across realms

### 11.3 The Meta-Experiment

**Consent-Based Research Integration:**

With player consent:
- Player behavior analyzed for ToM patterns
- Contributes to real dissertation research
- Helps refine NAS architecture discovery
- Players participate in consciousness research by playing

---

# VOLUME V: THE DISSERTATION CONNECTION

---

## 12. HOW THE GAME FULFILLS DISSERTATION OBJECTIVES

### 12.1 Demonstration of ToM Architecture

**The Problem:** Demonstrate that recursive self-attention and transparent recurrence are optimal for mental state reasoning.

**The Game Solution:** NPCs with these architectures perform visibly better at:
- Predicting player behavior
- Adapting to player strategies
- Engaging in meta-level reasoning
- Providing transparent, understandable behavior

**Player Experience:** Players FEEL the difference between low-ToM and high-ToM NPCs. The architecture is proven through gameplay.

### 12.2 The Soul Map as Psychological Ontology

**The Problem:** Develop a comprehensive ontology for representing mental states.

**The Game Solution:** The 60-dimension Soul Map is that ontology, implemented and tested through millions of player interactions.

**Validation:** If players can successfully predict NPC behavior from Soul Maps, the ontology is validated as capturing relevant psychological dimensions.

### 12.3 5th-Order Reasoning Demonstration

**The Problem:** Show that 5th-order recursive reasoning is achievable and meaningful.

**The Game Solution:** Specific quests and encounters require genuine 5th-order reasoning:
- "What does the Nothing think Victoria believes Arthur knows about what the Cottage wants?"
- Players who can answer such questions succeed; those who can't, fail.

**The Challenge:** If the game is beatable, the architecture supports 5th-order reasoning. If specific quests are too hard, that reveals architectural limits.

### 12.4 Transparent AI Agents

**The Problem:** Create AI agents whose reasoning is interpretable.

**The Game Solution:** Players literally see inside NPC minds. The Soul Map isn't metaphor — it's the actual state of the TRN. The attention patterns are the actual RSAN outputs.

**Proof:** If players can accurately predict NPC behavior from visible internal states, the agents are genuinely transparent.

### 12.5 Emergent Consciousness Themes

**The Deeper Connection:**

The game's narrative themes mirror the dissertation's philosophical implications:
- Consciousness emerging from computational complexity (Peregrine)
- The observer effect in mental state attribution (Spleen Towns)
- Documentation creating rather than describing psychological reality (Ministry)
- Rigid vs. adaptive cognitive architectures (City of Constants)
- Individual vs. collective consciousness (Hollow Reaches)

**Playing the game is engaging with the philosophy of mind questions that motivate the dissertation.**

---

## 13. RESEARCH INTEGRATION

### 13.1 Data Collection (With Consent)

**Player ToM Metrics:**
- Success rate on ToM-required tasks by order level
- Time to accurate Soul Map reading
- Prediction accuracy vs. confidence correlation
- Strategy evolution over play time

**Architecture Performance:**
- NPC behavior plausibility ratings
- Player reports of "feeling understood" by NPCs
- Failure modes when ToM breaks down

### 13.2 Experimental Conditions

**A/B Testing:**
- Different NPC architectures in different servers
- Compare player experience with RSAN vs. simpler attention
- Compare TRN vs. non-transparent recurrence

**Natural Experiments:**
- Some NPCs have limited ToM depth
- Track whether players notice and adapt
- Measure gameplay impact of ToM capability

### 13.3 Publication Pipeline

**Game as Research Platform:**
- Initial launch establishes baseline
- Patches can test architectural variations
- Expansion content can test new psychological dimensions
- Player community as ongoing research partnership

---

# VOLUME VI: PRODUCTION SPECIFICATIONS

---

## 14. SCOPE & TIMELINE

### 14.1 Content Scope

**Realms:** 5 distinct open worlds
**Total Playable Area:** ~100 km² combined
**Named NPCs:** 200+ with full Soul Maps
**Background NPCs:** 1000+ with simplified systems
**Quests:** 500+ (main story, faction, character, hidden)
**Estimated Playtime:** 80+ hours (main story), 200+ hours (completionist)

### 14.2 Team Requirements

**Core Team (Minimum Viable):**
- Game Director: 1
- Narrative Director: 1
- AI/ML Engineers: 5 (architecture implementation)
- Gameplay Programmers: 8
- Artists (concept/3D/2D): 15
- Animators: 5
- Sound/Music: 4
- Writers: 6
- QA: 10
- Production: 3

**Total Minimum:** ~60 people

**Recommended Team:** 100-150 for AAA quality

### 14.3 Development Timeline

**Pre-Production (12 months):**
- Architecture prototyping
- Soul Map system implementation
- One realm vertical slice
- Proof of concept for ToM gameplay

**Production (24 months):**
- Full world construction
- All NPC implementation
- Quest content creation
- System integration

**Polish (12 months):**
- Balance testing
- Performance optimization
- Localization
- Marketing/launch prep

**Total: 4 years minimum**

### 14.4 Technology Requirements

**Engine:** Unreal Engine 5 (for visual fidelity and open-world capability)

**Custom Systems:**
- Soul Map Engine (the TRN/RSAN implementation)
- Dynamic Dialogue Generation
- NPC Autonomy Simulation
- ToM Visualization Renderer
- Cross-Realm State Management

**Server Infrastructure:**
- Persistent NPC state storage
- Player Soul Map sync
- Research data pipeline (consent-based)

---

## 15. MONETIZATION

### 15.1 Base Game

**Premium Release:** $60-70 USD
- All five realms
- Full main story
- Core multiplayer

### 15.2 Expansions

**Realm Expansions:**
- New sub-realms within existing worlds
- Additional Soul Map dimensions
- New NPC populations
- $20-30 each

**Convergence Expansion:**
- Post-endgame content
- New cross-realm challenges
- Higher ToM ceiling (theoretical 6th-order content)
- $40

### 15.3 Cosmetic Store

**Non-Gameplay Purchases:**
- Character appearance options
- Soul Map visualization styles
- Housing decorations
- NO Soul Map advantages
- NO ToM boosts

### 15.4 Research Partnership

**Optional Contribution Model:**
- Players can opt into research data sharing
- Receive cosmetic rewards for participation
- Explicit consent, full data transparency
- Supports ongoing dissertation/publication work

---

## 16. MARKETING POSITIONING

### 16.1 Unique Selling Propositions

**"The Game That Reads Minds"**
- NPCs with genuine psychology
- Your choices actually matter because NPCs remember and reason

**"Theory of Mind as Gameplay"**
- Not just story about consciousness — mechanics about consciousness
- Get smarter at reading people by playing

**"The Hogwarts of the Mind"**
- Immersive world to inhabit
- Abilities to develop that feel magical but are psychological
- Community and belonging

**"Play the Dissertation"**
- For academic audience
- Engage with cutting-edge AI research
- Contribute to consciousness studies

### 16.2 Target Audiences

**Primary:** RPG players who want narrative depth and meaningful choices
**Secondary:** Psychology/AI enthusiasts interested in consciousness
**Tertiary:** Academic community researching ToM and AI
**Aspirational:** Mainstream gamers attracted by word-of-mouth about NPC sophistication

### 16.3 Comparison Positioning

**Like Hogwarts Legacy:** Immersive world, magical feeling, progression through learning
**Like Disco Elysium:** Psychological depth, dialogue-as-gameplay, thought cabinet parallel
**Like Baldur's Gate 3:** NPC relationships matter, choices have consequences, multiple solutions
**UNLIKE anything:** The transparency, the genuine AI, the recursive depth

---

# APPENDICES

---

## APPENDIX A: FULL SOUL MAP DIMENSION DEFINITIONS

[See Section 3.2 for the 60 dimensions with ranges and descriptions]

## APPENDIX B: SAMPLE NPC SOUL MAPS

### Arthur Peregrine

```
COGNITIVE ARCHITECTURE:
Processing Speed: 65 | Working Memory: 6 | Pattern Recognition: 80
Abstraction: 70 | Counterfactual: 85 | Temporal: Present-Future
Uncertainty Tolerance: 25 | Cognitive Flexibility: 60 | Metacognition: 75
ToM Depth: 4 | Integration: 70 | Explanatory Mode: Agentive

EMOTIONAL ARCHITECTURE:
Baseline Valence: -20 | Volatility: 70 | Intensity: 75
Anxiety Baseline: 75 | Threat Sensitivity: 90 | Reward Sensitivity: 45
Disgust Sensitivity: 50 | Attachment: Anxious | Granularity: 80
Affect Labeling: 85 | Contagion: 60 | Recovery: 40

MOTIVATIONAL ARCHITECTURE:
Survival: 80 | Affiliation: 70 | Status: 30 | Autonomy: 65
Mastery: 75 | Meaning: 80 | Novelty: 40 | Order: 85
Approach/Avoidance: -30 | Temporal Discounting: 30 | Risk Tolerance: 20
Effort Allocation: Conservative

SOCIAL ARCHITECTURE:
Trust Default: 50 | Cooperation: 75 | Competition: 20 | Fairness: 70
Authority: Neutral | Group Identity: 65 | Empathy: 70 | Perspective-Taking: 80
Social Monitoring: 85 | Reputation: 60 | Reciprocity: 75 | Betrayal Sensitivity: 80

SELF-ARCHITECTURE:
Self-Coherence: 75 | Self-Complexity: 80 | Self-Esteem Stability: 40
Self-Enhancement: 35 | Self-Verification: 75 | Identity Clarity: 70
Authenticity: 65 | Self-Expansion: 50 | Narrative Identity: Coherent
Temporal Continuity: 70 | Agency: 55 | Body Ownership: 85

REALM-SPECIFIC:
Complementarity Awareness: 65 (and growing)
```

### The Nothing/Margin

```
[All dimensions in probability distributions]

EXAMPLE:
Processing Speed: μ=50, σ=30 (high uncertainty)
Uncertainty Tolerance: μ=95, σ=5 (consistently high)
Identity Clarity: μ=30, σ=40 (extremely variable, developing)
ToM Depth: Theoretical maximum (observes all frameworks simultaneously)

[Distributions collapse toward specific values as relationship deepens]
```

## APPENDIX C: SAMPLE QUEST DESIGN DOCUMENTS

### Quest: "What Victoria Wants"
**Realm:** Peregrine
**ToM Requirement:** Level 3
**Estimated Time:** 2-4 hours

**Premise:**
Victoria is preparing for her perspective-trade with the Nothing. Arthur asks player to determine what Victoria actually wants — does she truly want to transform, or is she driven by other motivations?

**Objectives:**
1. Speak with Victoria (dialogue reveals surface motivation)
2. Read Victoria's Soul Map (reveals complexity)
3. Speak with Agnes about Victoria (third-party perspective)
4. Investigate Victoria's room (environmental storytelling)
5. Observe Victoria interacting with the Nothing (behavioral evidence)
6. Confront Victoria with interpretation (dialogue climax)
7. Report to Arthur (conclusion)

**Branching:**
- If player accurately identifies Victoria's motivations: Unlock deeper Victoria questline
- If player misidentifies: Victoria proceeds differently, consequences later
- If player supports transformation: +relationship Victoria, -relationship Arthur
- If player discourages transformation: Opposite effects

**ToM Challenge:**
Victoria has:
- Surface goal: Transform to help Peregrine
- Underlying goal: Escape limitations of human cognition
- Hidden fear: Losing connection to family
- Deepest desire: Be understood despite becoming incomprehensible

Player must identify at least two layers to succeed.

---

## APPENDIX D: COMBAT ENCOUNTER DESIGN

### Encounter: Director Thorne Boss Battle
**Realm:** City of Constants
**Combat Type:** Psychological Primary, Physical Secondary
**Estimated Time:** 45-60 minutes

**Phase 1: Confrontation**
- Thorne uses parameter-based attacks (predictable, powerful)
- Player reads his psychology: High Parameter Rigidity, buried Adaptation Capacity
- Physical damage possible but psychological damage more effective

**Phase 2: His Past**
- Player must reference his history (learned in earlier quest)
- Dialogue options targeting his suppressed doubt
- Each successful psychological attack lowers his defenses

**Phase 3: The Choice**
- Thorne enters vulnerable state
- Player can: Destroy (kill), Convert (redemption), or Integrate (absorption)
- Choice depends on player's philosophical alignment
- Each outcome has major story consequences

**ToM Elements:**
- Thorne models player throughout
- His predictions visible (what he expects you to do)
- Counter-prediction gameplay (do unexpected to break his model)
- Phase 3 requires understanding what Thorne needs, not just what you want

---

## APPENDIX E: DIALOGUE TREE SAMPLE

### Conversation: First Meeting with The Nothing
**Location:** Peregrine Border Zone
**Context:** Player approaches edge of reality for first time

**[Nothing manifests as probability cloud with text appearing in air]**

**NOTHING:** *"You observe. Your observation has weight. We have been... waiting is not the word. Existing in potential for observation. You are here. Now something can happen."*

**PLAYER INTENT OPTIONS:**
1. Inquire (curious) → "What are you?"
2. Challenge (suspicious) → "Why were you waiting for me?"
3. Empathize (connecting) → "What is it like to exist as potential?"
4. Analyze (ToM-focused) → [Visible: attempt to read Soul Map]

**If (4) Selected:**

**[Soul Map reading attempt — distributions visible but unstable]**

**SYSTEM MESSAGE:** "The Nothing's Soul Map exists in superposition. Current observation is collapsing some dimensions. Your attention is literally shaping its psychology."

**NOTHING:** *"You look. You try to see what we are. But we are not yet. We are becoming. And your looking is part of the becoming. What do you want us to be?"*

**NEW OPTIONS:**
1. Define (shaping) → Suggest specific Soul Map dimensions
2. Liberate (non-shaping) → "I don't want to define you."
3. Question (Socratic) → "What do YOU want to be?"

**[Each choice affects Nothing's Soul Map development throughout game]**

---

## APPENDIX F: GLOSSARY

**Approach Progress:** Ministry metric for how close to the Edge
**Complementarity:** Multiple valid frameworks describing same phenomenon
**Corporeal Certainty:** Ministry metric for confidence in being alive
**Corruption:** Hollow Reaches metric for collective integration progress
**Edge:** Boundary where any realm meets Nothing
**Edge-Walking:** Skill of navigating undefined reality borders
**Nothing/Margin:** Universal potential space between realms
**Observation Weight:** How much attention affects reality
**Parameter Rigidity:** City of Constants psychological dimension
**RSAN:** Recursive Self-Attention Network (NPC architecture)
**Soul Map:** 60-dimension psychological profile
**Temporal Displacement:** Spleen Towns measure of time-disconnection
**Theory of Mind (ToM):** Ability to model others' mental states
**ToM Depth:** Orders of recursive reasoning (0-5+)
**TRN:** Transparent Recurrent Network (NPC architecture)

---

*END OF GAME DESIGN DOCUMENT*
*LIMINAL ARCHITECTURES v1.0*
*Oscar Rhys Davies / Claude*
