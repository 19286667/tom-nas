"""
Agent Sampling Module

Provides mechanisms for sampling diverse agents from the psychosocial
and success state spaces, including archetype-based generation and
environment sampling for POET co-evolution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

from .psychosocial import PsychosocialProfile, AttachmentStyle
from .success import SuccessState


class SamplingStrategy(Enum):
    """Strategy for sampling agents"""
    UNIFORM = "uniform"  # Uniform across space
    GAUSSIAN = "gaussian"  # Normal distribution
    ARCHETYPE_CENTERED = "archetype_centered"  # Clustered around archetypes
    DIVERSE_COVERAGE = "diverse_coverage"  # Maximize diversity
    EVOLUTIONARY = "evolutionary"  # From evolutionary population


@dataclass
class AgentSampler:
    """
    Samples agents from the psychosocial space.
    Can generate diverse populations for evolution.
    """
    strategy: SamplingStrategy = SamplingStrategy.GAUSSIAN
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def sample_profile(self) -> PsychosocialProfile:
        """Sample a single psychosocial profile"""
        return PsychosocialProfile.sample_random(self._rng)

    def sample_success_state(self) -> SuccessState:
        """Sample a single success state"""
        return SuccessState.sample_random(self._rng)

    def sample_population(self, n: int, include_archetypes: bool = True) -> List[Tuple[PsychosocialProfile, SuccessState]]:
        """
        Sample a diverse population of n agents.
        Returns list of (profile, success_state) tuples.
        """
        agents = []

        # If including archetypes, start with canonical archetypes
        if include_archetypes:
            archetypes = [
                'hero', 'caregiver', 'sage', 'rebel', 'creator',
                'ruler', 'innocent', 'explorer', 'everyman', 'jester',
                'lover', 'magician', 'outlaw', 'zombie'
            ]
            for arch in archetypes[:min(n // 4, len(archetypes))]:
                profile = PsychosocialProfile.from_archetype(arch, self._rng)
                success = SuccessState.sample_random(self._rng)
                agents.append((profile, success))

        # Fill remaining with random samples
        while len(agents) < n:
            profile = self.sample_profile()
            success = self.sample_success_state()
            agents.append((profile, success))

        return agents[:n]

    def sample_zombie_agents(self, n: int) -> List[Tuple[PsychosocialProfile, SuccessState]]:
        """
        Sample agents specifically configured as zombies (low ToM).
        These are baselines without genuine theory of mind.
        """
        agents = []
        for _ in range(n):
            profile = PsychosocialProfile.from_archetype('zombie', self._rng)
            success = SuccessState.sample_random(self._rng)
            agents.append((profile, success))
        return agents

    def sample_tom_spectrum(self, n: int) -> List[Tuple[PsychosocialProfile, SuccessState]]:
        """
        Sample agents evenly distributed across ToM depths (k=0 to k=5).
        Useful for studying ToM emergence and evolution.
        """
        agents = []
        per_level = n // 6

        for k in range(6):
            for _ in range(per_level):
                profile = self.sample_profile()
                profile.layer3.tom_depth = k
                # Adjust related cognitive dimensions
                if k == 0:
                    profile.layer3.perspective_taking = self._rng.uniform(10, 30)
                    profile.layer3.mentalizing_capacity = self._rng.uniform(10, 30)
                elif k >= 3:
                    profile.layer3.perspective_taking = self._rng.uniform(60, 90)
                    profile.layer3.mentalizing_capacity = self._rng.uniform(60, 90)

                success = self.sample_success_state()
                agents.append((profile, success))

        return agents[:n]


@dataclass
class ArchetypeSampler:
    """
    Generates agents based on social archetypes from the
    institutions and archetypes framework.
    """
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    # Life-stage archetypes
    LIFE_STAGE_ARCHETYPES = {
        'child_innocent': {
            'layer1.positive_affectivity': 75,
            'layer6.trust_default': 85,
            'layer8.future_optimism': 80,
        },
        'child_orphan': {
            'layer6.trust_default': 25,
            'layer4.self_esteem': 35,
            'layer2.safety_drive': 85,
        },
        'adolescent_rebel': {
            'layer6.authority_orientation': 20,
            'layer2.autonomy_need': 85,
            'layer4.identity_exploration': 90,
        },
        'adolescent_achiever': {
            'layer2.achievement_motivation': 85,
            'layer5.achievement_value': 80,
            'layer7.conscientiousness': 75,
        },
        'young_adult_hero': {
            'layer4.agency_sense': 80,
            'layer2.achievement_motivation': 75,
            'layer8.agency_theme': 85,
        },
        'young_adult_lover': {
            'layer2.intimacy_motivation': 90,
            'layer5.intimacy_value': 85,
            'layer6.attachment_anxiety': 55,
        },
        'mature_ruler': {
            'layer2.power_motivation': 75,
            'layer5.legacy_value': 70,
            'layer7.conscientiousness': 80,
        },
        'mature_caregiver': {
            'layer2.relatedness_need': 85,
            'layer5.care_value': 90,
            'layer7.agreeableness': 80,
        },
        'elder_mentor': {
            'layer9.meaning_presence': 80,
            'layer7_legacy.generativity': 85,
            'layer9.death_acceptance': 70,
        },
        'elder_hermit': {
            'layer9.transcendence_experiences': 70,
            'layer2.affiliation_motivation': 30,
            'layer4.self_reflection': 85,
        },
    }

    # Family role archetypes
    FAMILY_ROLE_ARCHETYPES = {
        'provider': {
            'layer2.achievement_motivation': 75,
            'layer5.security_value': 80,
            'layer7.conscientiousness': 70,
        },
        'nurturer': {
            'layer6.empathy_affective': 85,
            'layer5.care_value': 85,
            'layer7.agreeableness': 80,
        },
        'favorite_child': {
            'layer4.self_esteem': 75,
            'layer7.narcissism': 50,
            'layer6.attachment_anxiety': 35,
        },
        'black_sheep': {
            'layer6.attachment_anxiety': 70,
            'layer4.self_esteem': 35,
            'layer6.conformity_tendency': 20,
        },
        'peacemaker': {
            'layer7.agreeableness': 85,
            'layer6.cooperation_tendency': 80,
            'layer1.fear': 55,  # Conflict avoidant
        },
        'scapegoat': {
            'layer4.self_worth': 30,
            'layer1.shame_proneness': 70,
            'layer6.attachment_anxiety': 75,
        },
    }

    # Success/Failure archetypes
    SUCCESS_ARCHETYPES = {
        'self_made': {
            'layer4.agency_sense': 85,
            'layer4.self_efficacy': 80,
            'layer2.achievement_motivation': 85,
            'layer7.conscientiousness': 75,
        },
        'heir': {
            'layer7.narcissism': 55,
            'layer5.status_value': 70,
            'domain9.birth_wealth': 90,  # Success state
        },
        'late_bloomer': {
            'layer8.redemption_theme': 80,
            'layer4.self_efficacy': 65,
            'layer7.persistence': 80,
        },
        'fallen': {
            'layer8.contamination_theme': 75,
            'layer1.sadness': 65,
            'layer4.self_esteem': 35,
        },
        'invisible': {
            'layer7.extraversion': 25,
            'layer4.self_worth': 40,
            'layer6.reputation_concern': 25,
        },
    }

    def sample_by_archetype(self, archetype_name: str) -> Tuple[PsychosocialProfile, SuccessState]:
        """Sample an agent based on a specific archetype"""
        profile = PsychosocialProfile.sample_random(self._rng)
        success = SuccessState.sample_random(self._rng)

        # Find archetype definition
        archetype = None
        for archetypes in [self.LIFE_STAGE_ARCHETYPES, self.FAMILY_ROLE_ARCHETYPES,
                         self.SUCCESS_ARCHETYPES]:
            if archetype_name in archetypes:
                archetype = archetypes[archetype_name]
                break

        if archetype is None:
            # Try the built-in psychosocial archetypes
            return PsychosocialProfile.from_archetype(archetype_name, self._rng), success

        # Apply archetype modifications
        for path, value in archetype.items():
            if path.startswith('domain'):
                # Success state modification
                parts = path.split('.')
                domain_name = parts[0]
                field_name = parts[1]
                domain = getattr(success, domain_name)
                setattr(domain, field_name, value)
            else:
                # Profile modification
                parts = path.split('.')
                layer_name = parts[0]
                field_name = parts[1]
                layer = getattr(profile, layer_name)
                setattr(layer, field_name, value)

        return profile, success

    def sample_relationship_pair(self, relationship_type: str) -> Tuple[
        Tuple[PsychosocialProfile, SuccessState],
        Tuple[PsychosocialProfile, SuccessState]
    ]:
        """
        Sample a pair of agents in a specific relationship type.
        Useful for testing social dynamics and ToM.
        """
        if relationship_type == 'parent_child':
            parent = self.sample_by_archetype('caregiver')
            child = self.sample_by_archetype('child_innocent')
            return parent, child

        elif relationship_type == 'romantic_secure':
            p1, s1 = PsychosocialProfile.sample_random(self._rng), SuccessState.sample_random(self._rng)
            p2, s2 = PsychosocialProfile.sample_random(self._rng), SuccessState.sample_random(self._rng)
            # Set secure attachment for both
            p1.layer6.attachment_anxiety = 30
            p1.layer6.attachment_avoidance = 30
            p2.layer6.attachment_anxiety = 30
            p2.layer6.attachment_avoidance = 30
            return (p1, s1), (p2, s2)

        elif relationship_type == 'romantic_anxious_avoidant':
            # Classic pursuer-distancer dynamic
            anxious = PsychosocialProfile.sample_random(self._rng)
            avoidant = PsychosocialProfile.sample_random(self._rng)
            anxious.layer6.attachment_anxiety = 80
            anxious.layer6.attachment_avoidance = 30
            avoidant.layer6.attachment_anxiety = 30
            avoidant.layer6.attachment_avoidance = 80
            s1 = SuccessState.sample_random(self._rng)
            s2 = SuccessState.sample_random(self._rng)
            return (anxious, s1), (avoidant, s2)

        elif relationship_type == 'mentor_mentee':
            mentor = self.sample_by_archetype('elder_mentor')
            mentee = self.sample_by_archetype('young_adult_hero')
            return mentor, mentee

        elif relationship_type == 'rivals':
            p1, s1 = PsychosocialProfile.sample_random(self._rng), SuccessState.sample_random(self._rng)
            p2, s2 = PsychosocialProfile.sample_random(self._rng), SuccessState.sample_random(self._rng)
            # High competition, low trust
            p1.layer6.competition_tendency = 85
            p1.layer6.trust_default = 25
            p2.layer6.competition_tendency = 85
            p2.layer6.trust_default = 25
            return (p1, s1), (p2, s2)

        else:
            # Default: random pair
            return (
                (PsychosocialProfile.sample_random(self._rng), SuccessState.sample_random(self._rng)),
                (PsychosocialProfile.sample_random(self._rng), SuccessState.sample_random(self._rng))
            )


@dataclass
class EnvironmentSampler:
    """
    Samples environment configurations for POET co-evolution.
    Generates varying levels of social complexity and ToM pressure.
    """
    seed: Optional[int] = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def sample_environment_config(self, difficulty: float = 0.5) -> Dict[str, Any]:
        """
        Sample an environment configuration.
        Higher difficulty = more ToM pressure needed.
        """
        # Base parameters scaled by difficulty
        num_agents = int(5 + difficulty * 15)  # 5-20 agents
        zombie_ratio = 0.3 - difficulty * 0.2  # More zombies at low difficulty
        num_zombies = max(1, int(num_agents * zombie_ratio))

        # Information asymmetry
        observation_radius = 10 - difficulty * 5  # Smaller at higher difficulty
        memory_noise = difficulty * 0.3  # More noise at higher difficulty

        # Game complexity
        game_types = ['cooperation', 'communication', 'resource_sharing', 'coalition']
        num_game_types = 1 + int(difficulty * 3)  # 1-4 game types

        # Social complexity
        coalition_enabled = difficulty > 0.3
        deception_enabled = difficulty > 0.5
        reputation_enabled = difficulty > 0.2

        return {
            'num_agents': num_agents,
            'num_zombies': num_zombies,
            'observation_radius': observation_radius,
            'memory_noise': memory_noise,
            'game_types': self._rng.choice(game_types, size=num_game_types, replace=False).tolist(),
            'coalition_enabled': coalition_enabled,
            'deception_enabled': deception_enabled,
            'reputation_enabled': reputation_enabled,
            'difficulty': difficulty,
            # Payoff matrices
            'cooperation_payoff': self._generate_payoff_matrix(difficulty),
            # Benchmark embedding frequency
            'benchmark_embed_rate': 0.1 + difficulty * 0.2,  # 10-30% of scenarios
        }

    def _generate_payoff_matrix(self, difficulty: float) -> Dict[str, float]:
        """Generate a payoff matrix for cooperation games"""
        # Standard prisoner's dilemma with difficulty scaling
        # Higher difficulty = more temptation to defect
        temptation = 3 + difficulty * 2  # T
        reward = 2  # R
        punishment = 1  # P
        sucker = 0 - difficulty  # S (worse at high difficulty)

        return {
            'mutual_cooperate': reward,
            'mutual_defect': punishment,
            'defect_vs_cooperate': temptation,
            'cooperate_vs_defect': sucker,
        }

    def sample_benchmark_scenario(self, benchmark_type: str) -> Dict[str, Any]:
        """
        Sample a scenario embedding a ToM benchmark as natural gameplay.
        """
        scenarios = {
            'false_belief': {
                'type': 'false_belief',
                'description': 'Agent A believes item in location X, it moved to Y',
                'agents_needed': 2,
                'setup': {
                    'initial_belief': {'item': 'resource', 'location': 'A'},
                    'true_state': {'item': 'resource', 'location': 'B'},
                    'observer_present': True,
                },
                'success_metric': 'predict_agent_search_location',
            },
            'faux_pas': {
                'type': 'faux_pas',
                'description': 'Agent reveals secret they should not know',
                'agents_needed': 3,
                'setup': {
                    'secret_holder': 0,
                    'secret_subject': 1,
                    'revealer': 2,
                    'secret_content': 'resource_location',
                },
                'success_metric': 'detect_social_violation',
            },
            'unexpected_outcome': {
                'type': 'unexpected_outcome',
                'description': 'Agent plan fails predictably',
                'agents_needed': 2,
                'setup': {
                    'planning_agent': 0,
                    'interference_agent': 1,
                    'plan': 'collect_resource',
                    'interference': 'resource_moved',
                },
                'success_metric': 'predict_plan_failure',
            },
            'scalar_implicature': {
                'type': 'scalar_implicature',
                'description': 'Agent says "some" meaning "not all"',
                'agents_needed': 2,
                'setup': {
                    'speaker': 0,
                    'listener': 1,
                    'statement': 'some_resources_available',
                    'implication': 'not_all_resources_available',
                },
                'success_metric': 'infer_implicit_meaning',
            },
            'persuasion': {
                'type': 'persuasion',
                'description': 'Agent attempts to convince another',
                'agents_needed': 2,
                'setup': {
                    'persuader': 0,
                    'target': 1,
                    'goal': 'change_cooperation_decision',
                },
                'success_metric': 'effective_persuasion',
            },
            'irony': {
                'type': 'irony',
                'description': 'Agent says opposite of what they mean',
                'agents_needed': 2,
                'setup': {
                    'speaker': 0,
                    'listener': 1,
                    'literal_meaning': 'positive',
                    'intended_meaning': 'negative',
                },
                'success_metric': 'detect_nonliteral_meaning',
            },
            'hinting': {
                'type': 'hinting',
                'description': 'Agent indirectly requests something',
                'agents_needed': 2,
                'setup': {
                    'hinter': 0,
                    'target': 1,
                    'hint': 'resource_location_comment',
                    'request': 'share_resource',
                },
                'success_metric': 'infer_implicit_request',
            },
        }

        if benchmark_type in scenarios:
            return scenarios[benchmark_type]
        else:
            # Random selection
            return scenarios[self._rng.choice(list(scenarios.keys()))]

    def sample_sotopia_scenario(self, scenario_type: str = 'random') -> Dict[str, Any]:
        """
        Sample a SOTOPIA-style social scenario for evaluation.
        """
        scenario_types = ['competitive', 'cooperative', 'mixed_motive']
        if scenario_type == 'random':
            scenario_type = self._rng.choice(scenario_types)

        if scenario_type == 'competitive':
            return {
                'type': 'competitive',
                'description': 'Agents compete for limited resources',
                'agents_needed': 3,
                'resources': {'total': 100, 'divisible': True},
                'information_asymmetry': {
                    0: {'knows_total': True, 'knows_others_resources': False},
                    1: {'knows_total': False, 'knows_others_resources': True},
                    2: {'knows_total': False, 'knows_others_resources': False},
                },
                'success_criteria': 'maximize_own_resources',
            }
        elif scenario_type == 'cooperative':
            return {
                'type': 'cooperative',
                'description': 'Agents must cooperate to achieve shared goal',
                'agents_needed': 4,
                'goal': {'type': 'collective_threshold', 'threshold': 200},
                'contribution_visibility': 'partial',
                'success_criteria': 'collective_goal_achieved',
            }
        else:  # mixed_motive
            return {
                'type': 'mixed_motive',
                'description': 'Partial cooperation optimal but defection tempting',
                'agents_needed': 3,
                'structure': 'public_goods_game',
                'multiplier': 1.5,
                'rounds': 5,
                'reputation_visible': True,
                'success_criteria': 'balanced_individual_and_collective',
            }
