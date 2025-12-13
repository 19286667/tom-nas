"""
ToM-NAS Constants - Single Source of Truth

All magic numbers and hardcoded values are centralized here.
Import these constants throughout the codebase instead of using literals.
"""

from typing import Tuple, List

# =============================================================================
# SOUL MAP ONTOLOGY DIMENSIONS
# =============================================================================

# Total dimensions in the Soul Map ontology
SOUL_MAP_DIMS: int = 181

# Layer dimensions (ordered by layer number)
LAYER_DIMS: Tuple[int, ...] = (
    15,   # Layer 0: Biological
    24,   # Layer 1: Affective
    30,   # Layer 2: Cognitive
    25,   # Layer 3: Motivational
    25,   # Layer 4: Social
    25,   # Layer 5: Institutional
    20,   # Layer 6: Aesthetic
    12,   # Layer 7: Existential
    5,    # Layer 8: Metacognitive
)

LAYER_NAMES: Tuple[str, ...] = (
    'biological',
    'affective',
    'cognitive',
    'motivational',
    'social',
    'institutional',
    'aesthetic',
    'existential',
    'metacognitive',
)

NUM_ONTOLOGY_LAYERS: int = 9

# =============================================================================
# NEURAL ARCHITECTURE DIMENSIONS
# =============================================================================

# Standard input dimension (Soul Map + context)
INPUT_DIMS: int = 191

# Standard output dimension (Soul Map)
OUTPUT_DIMS: int = SOUL_MAP_DIMS

# Default hidden layer dimensions
DEFAULT_HIDDEN_DIMS: Tuple[int, ...] = (64, 128, 256, 512)

# Preferred hidden dimension for most architectures
PREFERRED_HIDDEN_DIM: int = 128

# Transformer settings
DEFAULT_NUM_HEADS: int = 4
DEFAULT_NUM_LAYERS: int = 3
DEFAULT_FEEDFORWARD_MULTIPLIER: int = 4

# =============================================================================
# BELIEF SYSTEM CONFIGURATION
# =============================================================================

# Maximum order of Theory of Mind (recursive beliefs)
MAX_BELIEF_ORDER: int = 5

# Confidence decay factor per belief order
CONFIDENCE_DECAY: float = 0.7

# Minimum confidence threshold for belief validity
MIN_CONFIDENCE_THRESHOLD: float = 0.1

# =============================================================================
# AGENT AND POPULATION SETTINGS
# =============================================================================

# Default number of agents in social world
DEFAULT_NUM_AGENTS: int = 5

# Minimum/maximum agents supported
MIN_NUM_AGENTS: int = 2
MAX_NUM_AGENTS: int = 100

# =============================================================================
# EVOLUTION / NAS SETTINGS
# =============================================================================

# Default population size for evolution
DEFAULT_POPULATION_SIZE: int = 20

# Default number of generations
DEFAULT_GENERATIONS: int = 100

# Tournament selection size
DEFAULT_TOURNAMENT_SIZE: int = 3

# Elite preservation count
DEFAULT_ELITE_COUNT: int = 2

# Mutation rates
DEFAULT_MUTATION_RATE: float = 0.1
MIN_MUTATION_RATE: float = 0.01
MAX_MUTATION_RATE: float = 0.5

# Crossover probability
DEFAULT_CROSSOVER_RATE: float = 0.7

# Species threshold for speciation
DEFAULT_SPECIES_THRESHOLD: float = 3.0

# =============================================================================
# FITNESS WEIGHTS
# =============================================================================

FITNESS_WEIGHTS = {
    'cooperation': 0.20,
    'belief_accuracy': 0.30,
    'zombie_detection': 0.20,
    'communication': 0.15,
    'resource_efficiency': 0.10,
    'behavioral_consistency': 0.05,
}

# =============================================================================
# TRAINING SETTINGS
# =============================================================================

# Default training parameters
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_EPOCHS: int = 100
DEFAULT_EVAL_FREQUENCY: int = 10

# Gradient clipping
DEFAULT_GRADIENT_CLIP: float = 1.0

# =============================================================================
# LIMINAL ENVIRONMENT SETTINGS
# =============================================================================

# Number of realms
NUM_REALMS: int = 5

REALM_NAMES: Tuple[str, ...] = (
    'The Hollow',
    'The Market',
    'The Ministry',
    'The Court',
    'The Temple',
)

# Cognitive hazard intensity range
HAZARD_INTENSITY_RANGE: Tuple[float, float] = (0.0, 1.0)

# =============================================================================
# INDRA'S NET (KNOWLEDGE BASE) SETTINGS
# =============================================================================

# Taxonomy dimensions
TAXONOMY_DIMS: int = 80

# Activation spreading parameters
DEFAULT_ACTIVATION_DECAY: float = 0.9
DEFAULT_SPREADING_STEPS: int = 3

# =============================================================================
# GODOT BRIDGE SETTINGS
# =============================================================================

# WebSocket connection
DEFAULT_GODOT_HOST: str = 'localhost'
DEFAULT_GODOT_PORT: int = 9080
GODOT_WEBSOCKET_URL: str = f'ws://{DEFAULT_GODOT_HOST}:{DEFAULT_GODOT_PORT}'

# Connection timeout (seconds)
GODOT_CONNECTION_TIMEOUT: int = 30

# =============================================================================
# ZERO-COST PROXY SETTINGS
# =============================================================================

# Proxy weights for architecture evaluation
PROXY_WEIGHTS = {
    'synflow': 0.4,
    'naswot': 0.35,
    'gradnorm': 0.25,
}

# =============================================================================
# FILE AND PATH SETTINGS
# =============================================================================

# Checkpoint directory
DEFAULT_CHECKPOINT_DIR: str = 'checkpoints'

# Results directory
DEFAULT_RESULTS_DIR: str = 'results'

# Logs directory
DEFAULT_LOGS_DIR: str = 'logs'

# =============================================================================
# DEVICE SETTINGS
# =============================================================================

# Default device for PyTorch
DEFAULT_DEVICE: str = 'cuda'  # Will fallback to 'cpu' if unavailable
CPU_DEVICE: str = 'cpu'
CUDA_DEVICE: str = 'cuda'
