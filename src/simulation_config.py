"""
ToM-NAS Simulation Configuration: The Constitution
====================================================

This module defines the master configuration for the Socio-Computational
Scientific Instrument. It is NOT a game configuration. It operationalizes:

1. The Symbol Grounding Problem for Theory of Mind
2. POET-driven co-evolution where Social Intelligence is survival-critical
3. Institutional Friction as the selective pressure
4. MetaMind + BeliefNest as the cognitive architecture

ARCHITECTURAL INVARIANTS (Do Not Violate):
- Agents MUST satisfy Mundane Maintenance to maintain cognitive performance
- Institutions provide NORMS, not rules - violation is possible but costly
- BeliefNest depth determines ToM order (not hardcoded reasoning)
- POET mutates BOTH agent architectures AND institutional complexity

Author: ToM-NAS Project
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import json


# =============================================================================
# LAYER 1: THE MUNDANE - Physical & Grounding Layer
# =============================================================================

class MundaneCategory(Enum):
    """
    Categories from 'Miscellaneous Components of Human Life'
    These provide Signal-to-Noise ratio for robust ToM.
    """
    # Maintenance Activities (Survival Baseline)
    EATING = "eating"
    SLEEPING = "sleeping"
    HYGIENE = "hygiene"
    FINANCIAL_MAINTENANCE = "financial_maintenance"
    HEALTH_MAINTENANCE = "health_maintenance"

    # Social Rhythms
    MORNING_ROUTINE = "morning_routine"
    EVENING_ROUTINE = "evening_routine"
    WEEKEND_PATTERN = "weekend_pattern"

    # Material Culture
    COMFORT_OBJECTS = "comfort_objects"
    STATUS_OBJECTS = "status_objects"
    FUNCTIONAL_OBJECTS = "functional_objects"
    SENTIMENTAL_OBJECTS = "sentimental_objects"

    # Spaces
    PRIVATE_SPACE = "private_space"
    SEMI_PRIVATE_SPACE = "semi_private_space"
    PUBLIC_SPACE = "public_space"
    LIMINAL_SPACE = "liminal_space"


class ObjectClass(Enum):
    """Material Culture classification for Godot objects."""
    COMFORT = "comfort"          # Provides psychological comfort
    STATUS = "status"            # Signals social position
    FUNCTIONAL = "functional"    # Serves practical purpose
    SENTIMENTAL = "sentimental"  # Emotional attachment
    INSTITUTIONAL = "institutional"  # Belongs to institution, not person


@dataclass
class MundaneConstraints:
    """
    Constraints from Mundane Life that affect cognitive performance.
    Agents cannot do pure social reasoning - they have bodies.
    """
    # Maintenance thresholds (0.0 = critical, 1.0 = satisfied)
    hunger_threshold: float = 0.3
    fatigue_threshold: float = 0.2
    hygiene_threshold: float = 0.4

    # Cognitive penalties when below threshold
    tom_depth_penalty: int = 1  # Reduce ToM order when stressed
    attention_penalty: float = 0.3  # Reduce attention capacity

    # Time costs (simulation ticks)
    eating_duration: int = 30
    sleeping_duration: int = 480  # 8 hours in minutes
    hygiene_duration: int = 20

    # Social costs of maintenance
    interruption_cost: float = 0.2  # Social friction from interrupting routines


# =============================================================================
# LAYER 2: INSTITUTIONS - The Selective Pressure
# =============================================================================

class InstitutionType(Enum):
    """
    From your Institutions taxonomy.
    Ordered roughly by ToM complexity required.
    """
    # Low friction (early evolution)
    FAMILY = "family"
    FRIENDSHIP = "friendship"

    # Medium friction
    EDUCATION = "education"
    WORKPLACE = "workplace"
    HEALTHCARE = "healthcare"
    RELIGIOUS = "religious"

    # High friction (late evolution)
    LEGAL = "legal"
    POLITICAL = "political"
    ECONOMIC_MARKET = "economic_market"
    MEDIA = "media"

    # Maximum friction (adversarial)
    MILITARY = "military"
    CRIMINAL = "criminal"


@dataclass
class InstitutionalNorm:
    """
    A norm within an institution. Norms are NOT rules - they can be violated.
    """
    name: str
    description: str
    violation_cost: float  # Social cost of violating (0.0 - 1.0)
    detection_probability: float  # How likely violation is noticed
    applies_to_roles: List[str]  # Which roles this norm constrains
    context_modifiers: Dict[str, float] = field(default_factory=dict)


@dataclass
class InstitutionGenotype:
    """
    The genetic representation of an Institution for POET evolution.
    """
    institution_type: InstitutionType
    complexity_level: float  # 0.0 = trivial, 1.0 = adversarial

    # Information asymmetry (key ToM driver)
    information_asymmetry: float  # 0.0 = all public, 1.0 = all private
    deception_prevalence: float   # How common is strategic deception

    # Role structure
    role_hierarchy_depth: int  # Flat (1) to deep hierarchy (5+)
    role_power_differential: float  # 0.0 = equal, 1.0 = extreme inequality

    # Norm density
    explicit_norms: List[InstitutionalNorm] = field(default_factory=list)
    implicit_norms: List[InstitutionalNorm] = field(default_factory=list)

    # Interaction patterns (from your Institutional Interaction Patterns)
    friction_coefficient: float = 0.5  # How much norms resist behavior
    reinforcement_cascade: bool = False  # Do violations compound?
    norm_negotiability: float = 0.3  # Can norms be contested?

    def mutate(self, mutation_rate: float = 0.1) -> 'InstitutionGenotype':
        """Mutate this institution for POET evolution."""
        import random
        new = InstitutionGenotype(
            institution_type=self.institution_type,
            complexity_level=min(1.0, self.complexity_level + random.gauss(0, mutation_rate)),
            information_asymmetry=max(0, min(1, self.information_asymmetry + random.gauss(0, mutation_rate))),
            deception_prevalence=max(0, min(1, self.deception_prevalence + random.gauss(0, mutation_rate))),
            role_hierarchy_depth=max(1, self.role_hierarchy_depth + random.randint(-1, 1)),
            role_power_differential=max(0, min(1, self.role_power_differential + random.gauss(0, mutation_rate))),
            friction_coefficient=max(0, min(1, self.friction_coefficient + random.gauss(0, mutation_rate))),
            norm_negotiability=max(0, min(1, self.norm_negotiability + random.gauss(0, mutation_rate))),
        )
        new.explicit_norms = self.explicit_norms.copy()
        new.implicit_norms = self.implicit_norms.copy()
        new.reinforcement_cascade = random.random() < 0.1 if not self.reinforcement_cascade else True
        return new


# =============================================================================
# LAYER 3: COGNITIVE ARCHITECTURE - MetaMind + BeliefNest
# =============================================================================

class StereotypeDimension(Enum):
    """
    From Stereotype Content Model (Fiske et al.)
    Used for prototype-based belief initialization.
    """
    WARMTH = "warmth"        # Intent: friendly vs. hostile
    COMPETENCE = "competence"  # Capability: able vs. unable


@dataclass
class SocialPrototype:
    """
    A prototype for initializing beliefs about unknown agents.
    From your Semantic Prototype Theory integration.
    """
    name: str
    warmth_prior: float  # -1.0 to 1.0
    competence_prior: float  # -1.0 to 1.0
    feature_triggers: List[str]  # Features that activate this prototype
    confidence: float = 0.5  # How certain is this prior


@dataclass
class BeliefNestConfig:
    """
    Configuration for the BeliefNest graph structure.
    """
    max_nesting_depth: int = 5  # Maximum ToM order supported
    belief_decay_rate: float = 0.1  # How quickly old beliefs fade
    contradiction_threshold: float = 0.3  # When beliefs conflict too much

    # Pruning for computational efficiency
    max_beliefs_per_agent: int = 100
    relevance_threshold: float = 0.1  # Prune beliefs below this relevance

    # Update dynamics
    observation_weight: float = 0.8  # How much direct observation matters
    inference_weight: float = 0.5  # How much inferred beliefs matter
    social_transmission_weight: float = 0.3  # How much others' claims matter


@dataclass
class MetaMindConfig:
    """
    Configuration for the MetaMind 3-stage pipeline.
    """
    # Stage 1: Hypothesis Generation (ToM Agent)
    max_hypotheses: int = 5
    hypothesis_diversity_pressure: float = 0.3  # Force diverse explanations

    # Stage 2: Institutional Filtering (Domain Agent)
    norm_weight: float = 0.7  # How much norms constrain responses
    role_weight: float = 0.5  # How much role expectations matter

    # Stage 3: Response Selection (Response Agent)
    action_temperature: float = 0.3  # Exploration vs. exploitation
    social_cost_weight: float = 0.6  # How much to avoid social friction

    # Integration
    simulation_budget: int = 10  # How many mental simulations per decision
    time_horizon: int = 5  # How far ahead to simulate consequences


# =============================================================================
# LAYER 4: EVOLUTIONARY MECHANISM - POET + NAS
# =============================================================================

@dataclass
class POETConfig:
    """
    Configuration for Paired Open-Ended Trailblazer.
    Co-evolves agents and institutional environments.
    """
    # Population sizes
    agent_population_size: int = 50
    environment_population_size: int = 20

    # Evolution parameters
    generations_per_epoch: int = 100
    migration_threshold: float = 0.8  # Performance to trigger migration
    extinction_threshold: float = 0.2  # Performance below = extinction

    # Novelty search (prevents convergence to local optima)
    novelty_weight: float = 0.3
    archive_size: int = 500  # Behavioral archive for novelty

    # Transfer learning
    enable_agent_transfer: bool = True  # Can agents move between environments?
    enable_environment_sharing: bool = True  # Can environments be shared?

    # Mutation rates
    agent_mutation_rate: float = 0.1
    environment_mutation_rate: float = 0.05


@dataclass
class NASConfig:
    """
    Neural Architecture Search configuration for ToM modules.
    Uses your tom-nas repo's search space.
    """
    # Search space (from your search_space.py)
    search_modules: List[str] = field(default_factory=lambda: [
        "intent_module",      # Infers agent intentions
        "belief_module",      # Tracks agent beliefs
        "emotion_module",     # Recognizes emotional states
        "norm_module",        # Encodes institutional norms
        "prediction_module",  # Predicts future states
    ])

    # Zero-cost proxies for efficiency (from your proxies.py)
    use_zero_cost_proxies: bool = True
    proxy_types: List[str] = field(default_factory=lambda: [
        "synflow",
        "fisher",
        "grasp",
    ])

    # Architecture constraints
    max_parameters: int = 10_000_000
    max_flops: int = 1_000_000_000
    min_tom_depth: int = 2  # Minimum ToM order the architecture must support


# =============================================================================
# LAYER 5: EVALUATION - Situated Assessment
# =============================================================================

@dataclass
class EvaluationConfig:
    """
    Evaluation based on Situated Evaluation (Ma et al.)
    Not just "did you win" but "was your model of others accurate?"
    """
    # Primary metrics
    belief_accuracy_weight: float = 0.4  # Internal model vs. ground truth
    action_success_weight: float = 0.3   # Did actions achieve goals
    social_cost_weight: float = 0.2      # Social friction incurred
    efficiency_weight: float = 0.1       # Computational cost

    # ToM-specific metrics
    tom_depth_achieved: bool = True      # Track max ToM depth used
    belief_calibration: bool = True      # Are confidence estimates accurate?
    counterfactual_accuracy: bool = True  # "What if" reasoning quality

    # Mundane integration
    maintenance_satisfaction: bool = True  # Did agent meet bodily needs?
    routine_disruption_cost: bool = True   # Cost of disrupting others' routines


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """
    The Master Configuration - The Constitution.

    This binds all layers into a coherent system. When you instantiate
    this config, you are defining the entire experimental apparatus.
    """
    # Experiment metadata
    experiment_name: str = "tom_nas_situated_evolution"
    experiment_version: str = "1.0.0"
    random_seed: int = 42

    # Layer configurations
    mundane: MundaneConstraints = field(default_factory=MundaneConstraints)
    belief_nest: BeliefNestConfig = field(default_factory=BeliefNestConfig)
    metamind: MetaMindConfig = field(default_factory=MetaMindConfig)
    poet: POETConfig = field(default_factory=POETConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Initial institution (starting evolutionary pressure)
    initial_institution: InstitutionType = InstitutionType.FAMILY

    # Godot integration
    godot_port: int = 9080
    godot_headless: bool = False  # Run without rendering for speed
    physics_ticks_per_second: int = 30

    # Sociological database (RAG system)
    taxonomy_db_path: str = "data/sociological_db"
    enable_rag_lookup: bool = True

    # Logging and checkpointing
    log_level: str = "INFO"
    checkpoint_interval: int = 100  # generations
    checkpoint_path: str = "checkpoints/"

    # Computational constraints
    max_parallel_simulations: int = 4
    simulation_timeout_seconds: int = 300
    memory_limit_gb: float = 16.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        return {
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "random_seed": self.random_seed,
            "initial_institution": self.initial_institution.value,
            "godot_port": self.godot_port,
            "godot_headless": self.godot_headless,
            "physics_ticks_per_second": self.physics_ticks_per_second,
            "taxonomy_db_path": self.taxonomy_db_path,
            "log_level": self.log_level,
            "checkpoint_interval": self.checkpoint_interval,
            # Nested configs would be expanded here
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SimulationConfig':
        """Deserialize from dictionary."""
        config = cls()
        config.experiment_name = d.get("experiment_name", config.experiment_name)
        config.experiment_version = d.get("experiment_version", config.experiment_version)
        config.random_seed = d.get("random_seed", config.random_seed)
        config.initial_institution = InstitutionType(d.get("initial_institution", "family"))
        config.godot_port = d.get("godot_port", config.godot_port)
        config.godot_headless = d.get("godot_headless", config.godot_headless)
        return config

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SimulationConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def create_minimal_config() -> SimulationConfig:
    """Minimal config for testing."""
    return SimulationConfig(
        experiment_name="minimal_test",
        godot_headless=True,
        poet=POETConfig(
            agent_population_size=10,
            environment_population_size=5,
            generations_per_epoch=10,
        ),
    )


def create_family_scenario_config() -> SimulationConfig:
    """Configuration for Family institution scenario (low friction)."""
    return SimulationConfig(
        experiment_name="family_baseline",
        initial_institution=InstitutionType.FAMILY,
        belief_nest=BeliefNestConfig(max_nesting_depth=3),
        metamind=MetaMindConfig(norm_weight=0.3),  # Low norm pressure
    )


def create_workplace_scenario_config() -> SimulationConfig:
    """Configuration for Workplace institution (medium friction)."""
    return SimulationConfig(
        experiment_name="workplace_evolution",
        initial_institution=InstitutionType.WORKPLACE,
        belief_nest=BeliefNestConfig(max_nesting_depth=4),
        metamind=MetaMindConfig(norm_weight=0.7, role_weight=0.8),
    )


def create_adversarial_scenario_config() -> SimulationConfig:
    """Configuration for high-friction adversarial scenarios."""
    return SimulationConfig(
        experiment_name="adversarial_tom",
        initial_institution=InstitutionType.POLITICAL,
        belief_nest=BeliefNestConfig(max_nesting_depth=5),
        metamind=MetaMindConfig(
            norm_weight=0.5,  # Norms are negotiable
            action_temperature=0.5,  # More exploration
            simulation_budget=20,  # More mental simulation
        ),
        poet=POETConfig(
            novelty_weight=0.5,  # Push for novel strategies
        ),
    )


def create_full_poet_config() -> SimulationConfig:
    """Full POET configuration for serious experiments."""
    return SimulationConfig(
        experiment_name="full_poet_coevolution",
        initial_institution=InstitutionType.FAMILY,
        poet=POETConfig(
            agent_population_size=100,
            environment_population_size=50,
            generations_per_epoch=500,
            enable_agent_transfer=True,
            enable_environment_sharing=True,
        ),
        nas=NASConfig(
            use_zero_cost_proxies=True,
        ),
        max_parallel_simulations=8,
    )


# =============================================================================
# SOCIOLOGICAL DATABASE SCHEMA
# =============================================================================

@dataclass
class TaxonomyEntry:
    """
    Entry in the sociological database.
    Used for RAG-based norm/context lookup.
    """
    category: str  # e.g., "institution", "mundane", "norm"
    name: str
    description: str
    embedding: Optional[List[float]] = None  # For vector search
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Relational links
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)


class ContextManager:
    """
    Manager for sociological context lookup.
    Interfaces with the vector database containing taxonomies.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cache: Dict[str, TaxonomyEntry] = {}

    def get_norms(
        self,
        location: str,
        role: Optional[str] = None,
        institution: Optional[str] = None
    ) -> List[InstitutionalNorm]:
        """
        Retrieve applicable norms for a given context.

        Example:
            norms = context_manager.get_norms(
                location="Bank",
                role="Teller",
                institution="economic"
            )
        """
        # This would query the vector DB in production
        # Placeholder implementation
        return []

    def get_prototypes(
        self,
        features: List[str]
    ) -> List[SocialPrototype]:
        """
        Retrieve social prototypes matching observed features.

        Example:
            prototypes = context_manager.get_prototypes(
                features=["wearing_suit", "in_office", "confident_posture"]
            )
        """
        return []

    def get_mundane_constraints(
        self,
        time_of_day: float,
        location_type: str
    ) -> Dict[str, Any]:
        """
        Get mundane activity expectations for context.

        Example:
            constraints = context_manager.get_mundane_constraints(
                time_of_day=7.5,  # 7:30 AM
                location_type="kitchen"
            )
            # Returns: {"expected_activity": "morning_routine", "interruptibility": 0.2}
        """
        return {}


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    # Enums
    "MundaneCategory",
    "ObjectClass",
    "InstitutionType",
    "StereotypeDimension",
    # Data classes
    "MundaneConstraints",
    "InstitutionalNorm",
    "InstitutionGenotype",
    "SocialPrototype",
    "BeliefNestConfig",
    "MetaMindConfig",
    "POETConfig",
    "NASConfig",
    "EvaluationConfig",
    "SimulationConfig",
    "TaxonomyEntry",
    # Classes
    "ContextManager",
    # Factory functions
    "create_minimal_config",
    "create_family_scenario_config",
    "create_workplace_scenario_config",
    "create_adversarial_scenario_config",
    "create_full_poet_config",
]
