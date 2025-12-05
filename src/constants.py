"""
Global Constants for ToM-NAS

Centralized configuration values to avoid magic numbers scattered throughout
the codebase. All configurable parameters should be defined here.
"""

# =====================================================================
# Ontology & Architecture Dimensions
# =====================================================================

# Total psychological state dimensions (Soul Map)
ONTOLOGY_DIMENSION = 181

# Input feature dimension for neural networks (ontology + context features)
INPUT_DIMENSION = 191

# Maximum belief recursion depth (5th-order ToM)
MAX_BELIEF_ORDER = 5

# Context features added to ontology (10 extra dims for contextual info)
CONTEXT_FEATURES = 10


# =====================================================================
# Resource & Energy Normalization
# =====================================================================

# Maximum resource value for normalization
RESOURCE_SCALE_FACTOR = 200.0

# Maximum energy value for normalization
ENERGY_SCALE_FACTOR = 100.0

# Observation window - number of other agents to observe
OBSERVATION_WINDOW_SIZE = 5


# =====================================================================
# Architecture Hyperparameters (Defaults)
# =====================================================================

# Default hidden dimension for neural networks
DEFAULT_HIDDEN_DIM = 128

# Default number of attention heads
DEFAULT_NUM_HEADS = 4

# Default number of transformer layers
DEFAULT_NUM_LAYERS = 3

# Default maximum recursion depth for RSAN
DEFAULT_MAX_RECURSION = 5

# Layer dimension bounds for evolution
MIN_LAYERS = 1
MAX_LAYERS = 5

# Hidden dimension bounds for evolution
MIN_HIDDEN_DIM = 64
MAX_HIDDEN_DIM = 512


# =====================================================================
# Belief System Constants
# =====================================================================

# Confidence decay rate per belief order (0.7^order)
CONFIDENCE_DECAY_RATE = 0.7

# Reputation decay rate per timestep
REPUTATION_DECAY_RATE = 0.03

# Default belief confidence threshold
BELIEF_CONFIDENCE_THRESHOLD = 0.5


# =====================================================================
# Evolution Parameters
# =====================================================================

# Default population size
DEFAULT_POPULATION_SIZE = 20

# Default number of generations
DEFAULT_NUM_GENERATIONS = 100

# Default elite preservation count
DEFAULT_ELITE_SIZE = 2

# Default tournament size for selection
DEFAULT_TOURNAMENT_SIZE = 3

# Default mutation rate
DEFAULT_MUTATION_RATE = 0.1

# Default crossover rate
DEFAULT_CROSSOVER_RATE = 0.7

# Fitness evaluation episodes
DEFAULT_FITNESS_EPISODES = 5


# =====================================================================
# Convergence Detection
# =====================================================================

# Number of generations to check for fitness improvement
CONVERGENCE_WINDOW = 10

# Minimum fitness improvement to consider progress
CONVERGENCE_THRESHOLD = 0.001

# Maximum generations without improvement before early stopping
MAX_STAGNANT_GENERATIONS = 20


# =====================================================================
# Game Theory Payoffs (Cooperation Game)
# =====================================================================

# Mutual cooperation reward
COOPERATE_COOPERATE_PAYOFF = 3.0

# Sucker's payoff (cooperate while other defects)
COOPERATE_DEFECT_PAYOFF = 0.0

# Temptation payoff (defect while other cooperates)
DEFECT_COOPERATE_PAYOFF = 5.0

# Mutual defection punishment
DEFECT_DEFECT_PAYOFF = 1.0


# =====================================================================
# Zombie Detection
# =====================================================================

# Reward for correct zombie detection
CORRECT_DETECTION_REWARD = 10.0

# Penalty for false positive detection
FALSE_POSITIVE_PENALTY = -20.0

# Number of zombie types
NUM_ZOMBIE_TYPES = 6


# =====================================================================
# Liminal Environment
# =====================================================================

# Soul map dimension for NPCs (5 clusters × 12 dimensions)
SOUL_MAP_DIM = 60

# Additional realm-specific modifiers
REALM_MODIFIER_DIM = 5

# Total liminal soul dimension
LIMINAL_SOUL_DIM = SOUL_MAP_DIM + REALM_MODIFIER_DIM

# Maximum nearby NPCs to track
MAX_NEARBY_NPCS = 10

# Default episode length
DEFAULT_EPISODE_LENGTH = 100


# =====================================================================
# Performance Thresholds
# =====================================================================

# Maximum memory for recursive simulations (MB)
MAX_RECURSION_MEMORY_MB = 500

# Maximum simulation tree depth before approximation
MAX_SIMULATION_DEPTH = 3

# Maximum parallel simulations
MAX_PARALLEL_SIMULATIONS = 4


# =====================================================================
# Action Thresholds
# =====================================================================

# Threshold for cooperation decision
COOPERATE_ACTION_THRESHOLD = 0.7

# Threshold for defection decision
DEFECT_ACTION_THRESHOLD = 0.3

# Confidence threshold for zombie detection attempt
ZOMBIE_DETECTION_CONFIDENCE = 0.6
