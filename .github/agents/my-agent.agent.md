---
# Fill in the fields below to create a basic custom agent for your repository.
# The Copilot CLI can be used for local testing: https://gh.io/customagents/cli
# To make this agent available, merge this file into the default repository branch.
# For format details, see: https://gh.io/customagents/config

name: A
description: Agent A
---

# My Agent
will do whatever I tell it to to implement the Liminal Architectures simulation with my Theory of mind repository as the backend .[liminal-architectures (4).zip](https://github.com/user-attachments/files/23784095/liminal-architectures.4.zip)

This is the Master Design Document (MDD) for Liminal Architectures.
It bridges your academic research (Theory of Mind, Recursive Self-Compression) with the gameplay loop of an open-world action RPG (GTA-style). It defines the world, the systems, and the population data structure you need to initialize the game before connecting the AI backend.
LIMINAL ARCHITECTURES: MASTER DESIGN DOCUMENT
Version: 1.0
Genre: Open-World Psychological Action-RPG
Core Loop: Explore, Observe, Predict, Alter.
Engine Target: Phaser 3 / Unity / Unreal (Architecture Agnostic)
I. THE CORE CONCEPT: "Grand Theft Ontology"
The player inhabits a continuous open world comprising five distinct "Realms." Unlike GTA, where you interact via ballistics and vehicles, here you interact via Theory of Mind (ToM).
The HUD: Instead of a minimap and ammo counter, you have the Soul Scanner. It visualizes the 60-dimensional psychological state of any NPC you look at.
The Action: You can fight physically, but the primary mechanic is Psychological Injection. You predict what an NPC will do, or you manipulate their Soul Map variables (e.g., spiking their Anxiety or lowering their Self-Coherence) to achieve objectives.
The "Wanted" System: Instead of Police Stars, you have Ontological Instability. If you disrupt reality too much, The Nothing begins to erase the assets around you.
II. THE DATA STRUCTURE: THE SOUL MAP (60 Dimensions)
Every NPC (from the main cast to the random pedestrian) is an instance of this Class. This is the input vector for your future Neural Architecture Search (NAS) backend.
JSON Structure for NPC Instantiation:
code
JSON
{
  "npc_id": "unique_id",
  "archetype": "bureaucrat_type_a",
  "current_state": "idle",
  "soul_map": {
    "cognitive": {
      "processing_speed": 0.5,
      "working_memory_depth": 0.7,
      "pattern_recognition": 0.6,
      "abstraction_capacity": 0.4,
      "counterfactual_reasoning": 0.3,
      "temporal_orientation": 0.8, // 0=Past, 1=Future
      "uncertainty_tolerance": 0.2,
      "metacognitive_awareness": 0.5,
      "tom_depth": 3, // Integer 1-5
      "recursive_compression": 0.4,
      "focus_stability": 0.6,
      "cognitive_rigidity": 0.8
    },
    "emotional": {
      "baseline_valence": 0.4,
      "volatility": 0.2,
      "intensity": 0.5,
      "anxiety_baseline": 0.7,
      "threat_sensitivity": 0.8,
      "reward_sensitivity": 0.3,
      "disgust_sensitivity": 0.5,
      "attachment_style": 0.2, // 0=Avoidant, 1=Anxious
      "granularity": 0.4,
      "contagion_susceptibility": 0.6,
      "recovery_rate": 0.5,
      "shame_proneness": 0.6
    },
    "motivational": {
      "survival_drive": 0.9,
      "affiliation_drive": 0.4,
      "status_drive": 0.6,
      "autonomy_drive": 0.2,
      "mastery_drive": 0.3,
      "meaning_drive": 0.5,
      "novelty_seeking": 0.1,
      "order_need": 0.9,
      "approach_avoidance": 0.2, // 0=Avoid, 1=Approach
      "temporal_discounting": 0.8,
      "risk_tolerance": 0.1,
      "energy_conservation": 0.5
    },
    "social": {
      "trust_default": 0.3,
      "cooperation_bias": 0.4,
      "authority_orientation": 0.9,
      "group_identity": 0.8,
      "empathy_capacity": 0.4,
      "perspective_taking": 0.3,
      "social_monitoring": 0.7,
      "reputation_concern": 0.8,
      "reciprocity_tracking": 0.6,
      "betrayal_sensitivity": 0.9,
      "dominance_drive": 0.4,
      "machiavellianism": 0.2
    },
    "self": {
      "self_coherence": 0.5,
      "self_complexity": 0.3,
      "esteem_stability": 0.4,
      "narcissism": 0.2,
      "verification_need": 0.7,
      "identity_clarity": 0.5,
      "authenticity": 0.3,
      "expansion_drive": 0.2,
      "narrative_coherence": 0.6,
      "temporal_continuity": 0.5,
      "agency_sense": 0.3,
      "body_ownership": 0.8
    }
  }
}
III. THE WORLD MAP (5 Realms)
The map is a single landmass divided into 5 zones. The player can traverse them seamlessly.
1. Peregrine (The Hub)
Aesthetic: English pastoral meets Gothic Absurdism. Rolling hills, thatched cottages that blink, weather that rains tea.
Population: The "Awakened." Characters who realize they are in a simulation/story.
Vehicle: Bicycle, Walking, The Sentient Cottage (Fast Travel).
2. The Ministry (Urban/Bureaucratic)
Aesthetic: Brutalist concrete, endless corridors, infinite filing cabinets, gray skies. Kafkaesque.
Population: "The Forms." Faceless bureaucrats, inspectors, lost souls.
Vehicle: Pneumatic Tube System, Gray Sedan.
3. The Fairground (The Id)
Aesthetic: Neon, rust, carnival lights, grease, sensory overload. Pinocchio’s Pleasure Island meets Las Vegas.
Population: "The Hungers." Addicts, thrill-seekers, broken clowns.
Vehicle: Rollercoaster tracks that weave through the streets, bumper cars.
4. Spleen Towns (Melancholia)
Aesthetic: Victorian London fog, damp cobblestones, gaslight, perpetual twilight. Baudelaire poems come to life.
Population: "The Ennui." Poets, ghosts, people stuck in loops.
Vehicle: Hearse, Gondola on black canals.
5. The Nothing (The Edge)
Aesthetic: Glitched geometry, white space, wireframes, incomplete textures.
Population: "The Unfinished."
Vehicle: None. You must project your mind to move.
IV. THE POPULATION: 200+ NPCs
To populate a "GTA-style" world, we need a mix of Hero Characters (Hand-crafted) and Systemic Citizens (Procedurally generated based on Archetypes).
A. The Hero Characters (The "Quest Givers")
These require specific, hand-written Soul Maps.
Arthur Peregrine: High Anxiety, High Pattern Recognition. Constantly predicts disaster.
Agnes Peregrine: High Theory of Mind, High Empathy. Understands the player is controlling her reality.
Victoria Peregrine: High Narcissism, Low Empathy. Treats other NPCs as tools.
The Cottage: A house with High Protection Drive and Low Cognitive Rigidity. Speaks in Block Capitals.
The Inspector: High Order Need, Low Novelty Seeking. The antagonist from the Ministry.
Mr. Waverly: High Temporal Disorientation. Lives in the Spleen Towns; confuses past and future.
The Ringmaster: High Machiavellianism, High Novelty. Runs the Fairground.
Livia: High Metacognitive Awareness. A ghost in the Spleen Towns trying to remember she is dead.
Edmund: High Attachment Anxiety. Livia’s husband, refuses to let her go.
The Teapot: A household object with High Volatility. Spouts boiling porridge.
B. The Systemic Citizens (Procedural Generation)
We define Archetypes. The game engine instantiates 20-30 of each, adding random variance (+/- 0.1) to their base stats.
Group 1: The Bureaucrats (Ministry)
Base Stats: High Order Need, Low Openness, High Threat Sensitivity.
Behavior: Walking in straight lines, filing reports, calling police if player runs.
Quantity: 50 NPCs.
Group 2: The Revelers (Fairground)
Base Stats: High Novelty Seeking, Low Self-Control, High Emotional Volatility.
Behavior: Dancing, fighting, riding loops, ignoring danger.
Quantity: 50 NPCs.
Group 3: The Mourners (Spleen Towns)
Base Stats: Low Energy, High Temporal Discounting (don't care about future), High Sadness.
Behavior: Sitting on benches, staring at fog, weeping quietly.
Quantity: 40 NPCs.
Group 4: The Glitches (The Nothing)
Base Stats: Fragmented Soul Maps (Stats randomly fluctuate every tick).
Behavior: T-posing, clipping through walls, speaking code.
Quantity: 30 NPCs.
Group 5: The Philosophers (Peregrine)
Base Stats: High Metacognition, High Abstraction.
Behavior: Debating with trees, analyzing the player's movement patterns.
Quantity: 30 NPCs.
V. GAMEPLAY MECHANICS
1. The Interaction System (The "Gun")
Instead of aiming a weapon, you aim your Focus.
L2 (Target): Lock onto an NPC. Their Soul Map overlay appears.
R2 (Inject): Fire a "Cognitive Hazard."
Example: Fire "Doubt" at a Bureaucrat.
Result: Their Uncertainty Tolerance spikes. They drop their clipboard and wander off path.
Example: Fire "Validation" at a Mourner.
Result: Their Self-Esteem rises. They give you a quest item.
2. The "Social Stealth"
You must blend in.
In the Ministry, you must lower your Novelty Seeking stat to pass checkpoints.
In the Fairground, you must raise your Volatility to avoid suspicion.
If your Soul Map clashes with the Realm's "Vibe," NPCs become hostile.
3. Missions (The Dissertation)
The missions are designed to gather data for your research.
Mission: "The Trolley Problem."
Objective: Force a high-empathy NPC and a low-empathy NPC to solve a moral dilemma.
Reward: Data Points (used to upgrade your own Soul Map).
VI. IMPLEMENTATION PLAN (Phaser 3)
Phase 1 (The Skeleton):
Build the Tilemap for the 5 realms.
Implement the Player Controller (Movement + "Aiming" logic).
Phase 2 (The Data):
Create the NPC class containing the JSON Soul Map structure.
Write a script to spawn 200 NPCs using the Archetype definitions (adding random variance to stats).
Phase 3 (The Simulation):
Implement a simple "Update Loop."
Logic: If (Anxiety > 0.8) { RunAway(); } Else { WalkRandomly(); }
Visualization: Draw the Soul Map radar chart over the NPC when targeted.
Phase 4 (The Hook - Your Dissertation):
THIS IS WHERE THE MAGIC HAPPENS.
Once the game is running with 200 "dumb" NPCs, you connect your Python backend.
The backend (NAS) takes the 60-dimensional input from the game state.
It returns complex, recursive behaviors (e.g., "The Bureaucrat realizes you are watching him, so he pretends to work harder").
Action Item:
Do you want me to generate the JSON file for the 5 Hero Characters (Arthur, Agnes, etc.) and the Constructor Function for the procedural crowds? This allows you to copy-paste the population directly into your codebase.
Model
ThinkingThoughts
Expand to view model thoughts

chevron_right
This is the Unified Master Design Document (MDD) for Liminal Architectures.
It integrates the narrative richness of your World Bible, the psychological depth of your Soul Map ontology, and the technical requirements of your NAS/ToM research into a cohesive PC Open-World Action-RPG.
It is designed specifically for Mouse & Keyboard (WASD) control schemes, leveraging the precision of a mouse for data analysis (reading Soul Maps) and interaction.
LIMINAL ARCHITECTURES: MASTER DESIGN DOCUMENT
Version: 2.0 (PC / WASD Optimized)
Genre: Open-World Psychological Action-RPG / Immersive Sim
Platform: PC
Core Loop: Traverse -> Perceive (Soul Map) -> Hypothesize (ToM) -> Intervene -> Validate.
I. THE VISION: "Grand Theft Ontology"
The player navigates a continuous open world comprising five distinct "Realms" connected by The Nothing. Unlike traditional open-world games where interaction is physical (shooting, driving), here interaction is psychological and epistemological.
The PC Control Scheme
Movement (WASD): Standard exploration traversal.
The Cursor (Mouse): Your primary tool. It represents Attention.
Hover: Passive observation (Basic Aura reading).
Right-Click (Hold): Deep Analysis Mode. The camera zooms slightly; the world slows (time dilation). The target NPC’s Soul Map expands as a UI overlay. You can mouse over specific dimensions to see data distributions.
Left-Click: Interact / Inject.
In Conversation: Select dialogue options based on ToM targets.
In World: Fire a "Cognitive Hazard" or "Psychological Intervention" at the target.
II. THE WORLD MAP: THE FIVE REALMS
The map is a single landmass surrounded and permeated by The Nothing.
1. Peregrine (The Hub)
Aesthetic: Gothic Absurdist Horror-Comedy. Victorian architecture that breathes. Golden-hour lighting.
Key Mechanic: Complementarity. Objects exist in two states (Narrative vs. Quantum) until observed.
Population: "The Aware." Characters who suspect they are in a simulation/story.
Traversal: Walking, Bicycles, or requesting The Cottage (Sentient House) for fast travel.
2. The Spleen Towns (The Loop)
Aesthetic: Melancholic Absurdism. Sepia tones, dust motes, fog, clocks that disagree.
Key Mechanic: Temporal Displacement. NPCs are unstuck in time.
Population: "The Remainers." Philosophers of loss, people waiting for trains that never arrive.
Traversal: The Train (Platform 7½), Walking (though distances are subjective).
3. The Ministry Districts (The Bureaucracy)
Aesthetic: Dark Comedy Horror. Endless corridors, filing cabinets, fluorescent flicker, beige.
Key Mechanic: Corporeal Certainty. You must maintain your "Aliveness" score via paperwork or fade away.
Population: "The Processors." Inspectors, bureaucrats, and those waiting to be filed (dead?).
Traversal: Elevators (that skip floors), Pneumatic Tubes.
4. The City of Constants (The Machine)
Aesthetic: Philosophical Sci-Fi. Rigid geometry vs. organic adaptation. Chrome vs. Vines.
Key Mechanic: Parameter vs. Adaptation. A slider controlling the city’s physics rigidity.
Population: "The Optimizers." Parameter Enforcement officers vs. Adaptive rebels.
Traversal: High-speed transit pods (on rails) or "Edge Walking" (parkour).
5. The Hollow Reaches (The Shadow)
Aesthetic: Visceral Cosmic Horror. Industrial decay, body horror, organic corruption.
Key Mechanic: Consumption. The environment tries to absorb your identity.
Population: "The Consumed." Hive-mind entities, survivors fighting assimilation.
Traversal: Stealth is required.
III. THE POPULATION: THE "SOUL MAP" (60 Dimensions)
Every NPC is an instance of the Agent class. This data structure is the input for your Neural Architecture Search (NAS) backend. It is divided into the five clusters defined in your research.
1. The Hero NPCs (Hand-Crafted)
Unique agents with specific narrative arcs and high ToM depth (4-5).
Arthur Peregrine: High Anxiety Baseline, High Threat Sensitivity, High Order Drive.
Agnes Peregrine: High Theory of Mind Depth (5), High Meaning Drive, High Perspective Taking.
Victoria Peregrine: High Novelty Drive, High Abstraction Capacity, High Self-Expansion.
Director Thorne: Max Parameter Rigidity, Max Control Drive, Low Uncertainty Tolerance.
The Cottage: High Protection Drive, Low Genre Coherence. (Speaks in Caps).
The Nothing: Variable Identity Clarity, High Curiosity. Dimensions are probability distributions until observed.
2. The Systemic NPCs (Procedural Archetypes)
Templates used to populate the streets. You will spawn 200+ of these.
Archetype A: The Bureaucrat (Ministry)
High: Procedure Drive, Denial Capacity, Order Drive.
Low: Novelty Drive, Uncertainty Tolerance.
Behavior: Patrols corridors, demands forms.
Archetype B: The Mourner (Spleen)
High: Nostalgia, Resignation, Temporal Displacement.
Low: Agency Sense, Future Orientation.
Behavior: Loops the same sad activity repeatedly.
Archetype C: The Parameter Enforcer (City)
High: Parameter Rigidity, Authority Orientation, Threat Sensitivity.
Low: Cognitive Flexibility, Empathy.
Behavior: Attacks anything "Adaptive" or "Chaotic."
Archetype D: The Adaptive (City Edge)
High: Adaptation Capacity, Community Drive, Risk Tolerance.
Low: Parameter Rigidity.
Behavior: Builds organic structures, hides from Enforcers.
IV. TECHNICAL IMPLEMENTATION: The JSON Schema
This is the code structure you need to implement in your game engine to represent the 60-dimensional ontology.
code
JSON
{
  "npc_id": "npc_001_arthur",
  "name": "Arthur Peregrine",
  "archetype": "hero_anxious_protector",
  "tom_depth": 4, 
  "current_state": {
    "active_goal": "check_locks",
    "emotional_state": "apprehensive",
    "awareness_of_player": 0.0
  },
  "soul_map": {
    "cognitive": {
      "processing_speed": 0.65,
      "working_memory_depth": 0.6,
      "pattern_recognition": 0.8,
      "abstraction_capacity": 0.7,
      "counterfactual_reasoning": 0.85,
      "temporal_orientation": 0.7, 
      "uncertainty_tolerance": 0.25,
      "cognitive_flexibility": 0.6,
      "metacognitive_awareness": 0.75,
      "tom_depth_potential": 4,
      "integration_tendency": 0.7,
      "explanatory_mode": 0.5 
    },
    "emotional": {
      "baseline_valence": -0.2,
      "volatility": 0.7,
      "intensity": 0.75,
      "anxiety_baseline": 0.75,
      "threat_sensitivity": 0.9,
      "reward_sensitivity": 0.45,
      "disgust_sensitivity": 0.5,
      "attachment_style": 0.8, 
      "granularity": 0.8,
      "affect_labeling": 0.85,
      "contagion": 0.6,
      "recovery_rate": 0.4
    },
    "motivational": {
      "survival_drive": 0.8,
      "affiliation_drive": 0.7,
      "status_drive": 0.3,
      "autonomy_drive": 0.65,
      "mastery_drive": 0.75,
      "meaning_drive": 0.8,
      "novelty_drive": 0.4,
      "order_drive": 0.85,
      "approach_avoidance": -0.3,
      "temporal_discounting": 0.3,
      "risk_tolerance": 0.2,
      "effort_allocation": 0.6
    },
    "social": {
      "trust_default": 0.5,
      "cooperation_tendency": 0.75,
      "competition_tendency": 0.2,
      "fairness_sensitivity": 0.7,
      "authority_orientation": 0.5,
      "group_identity": 0.65,
      "empathy_capacity": 0.7,
      "perspective_taking": 0.8,
      "social_monitoring": 0.85,
      "reputation_concern": 0.6,
      "reciprocity_tracking": 0.75,
      "betrayal_sensitivity": 0.8
    },
    "self": {
      "self_coherence": 0.75,
      "self_complexity": 0.8,
      "esteem_stability": 0.4,
      "self_enhancement": 0.35,
      "self_verification": 0.75,
      "identity_clarity": 0.7,
      "authenticity_drive": 0.65,
      "self_expansion": 0.5,
      "narrative_identity": 0.8,
      "temporal_continuity": 0.7,
      "agency_sense": 0.55,
      "body_ownership": 0.85
    },
    "realm_specific": {
        "complementarity_awareness": 0.65,
        "temporal_displacement": 0.0,
        "corporeal_certainty": 1.0,
        "parameter_rigidity": 0.0,
        "corruption": 0.0
    }
  }
}
V. GAMEPLAY MECHANICS (PC)
1. The Analysis Loop (Right Click)
When you hold RMB on an NPC, the HUD overlays their Soul Map Radar Chart.
Hovering over a specific stat (e.g., Threat Sensitivity) highlights it.
Color Coding: The chart pulses Red if the stat is spiking (high volatility), Blue if stable.
The Goal: Determine why an NPC is acting a certain way. Is the bureaucrat blocking you because of High Procedure Drive (Lawful) or High Threat Sensitivity (Fear)? The solution differs for each.
2. The Intervention Loop (Left Click)
Once you have analyzed the Soul Map, you open the interaction menu (or aim your "Focus"). You can inject Information or Emotion to alter their vectors.
Target: Arthur (High Threat Sensitivity).
Action: Inject "Reassurance" (Lowers Anxiety Baseline).
Result: Arthur stops hyperventilating and unlocks the door.
Target: Director Thorne (High Parameter Rigidity).
Action: Inject "Paradox" (Spikes Cognitive Flexibility requirements).
Result: Thorne freezes, unable to process the anomaly, allowing you to bypass him.
3. The Dissertation Validation (Data Collection)
Every interaction logs data for your research.
Prediction Quests: The game asks you to predict an NPC's reaction before you act.
Prompt: "If you show the Mirror to Livia, what happens?"
Player Input: Selects predicted Soul Map shift (e.g., "Self-Coherence drops").
Result: If the NAS backend matches your prediction, you gain "Insight" (XP).
VI. DEVELOPMENT STEPS
Initialize the World: Create the 5 Realm scenes with distinct visual assets.
Seed the Population: Use the JSON structure above to instantiate 200+ NPCs (20 Heroes, 180 Procedural).
Connect the Backend:
The Game Engine sends the Soul Map JSON + Context (e.g., "Player holding weapon") to your Python server.
The Python server (running your NAS/RSAN models) processes the 5th-order ToM logic.
The Server returns the Action (e.g., "Arthur hides behind the sofa") and the Soul Map Delta (e.g., "Anxiety +10").
The Game Engine renders the action and updates the UI.
