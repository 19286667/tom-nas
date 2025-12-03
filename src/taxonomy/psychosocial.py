"""
Psychosocial Taxonomy - 10-Layer Hierarchical Model of Human Constitution

Based on the comprehensive "Hierarchical Semantic Model of Human Constitution"
integrating biological, cognitive, affective, personality, self, relational,
social, cultural, temporal, and existential dimensions.

Total: 10 layers, 80+ dimensions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class AttachmentStyle(Enum):
    """Attachment patterns from developmental psychology"""
    SECURE = "secure"
    ANXIOUS_PREOCCUPIED = "anxious_preoccupied"
    DISMISSIVE_AVOIDANT = "dismissive_avoidant"
    FEARFUL_AVOIDANT = "fearful_avoidant"


class RegulatoryFocus(Enum):
    """Promotion vs prevention orientation"""
    PROMOTION = "promotion"  # Approach goals, gains
    PREVENTION = "prevention"  # Avoidance goals, losses
    BALANCED = "balanced"


class TemporalOrientation(Enum):
    """Past, present, or future focused"""
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    BALANCED = "balanced"


class CopingStyle(Enum):
    """Primary coping mechanism preference"""
    PROBLEM_FOCUSED = "problem_focused"
    EMOTION_FOCUSED = "emotion_focused"
    AVOIDANT = "avoidant"
    SOCIAL_SUPPORT = "social_support"
    MIXED = "mixed"


@dataclass
class Layer0_Biological:
    """
    Physical-Material Substrate
    Sensory processing, physiological state, embodiment
    """
    # Sensory sensitivities (0-100)
    visual_sensitivity: float = 50.0
    auditory_sensitivity: float = 50.0
    tactile_sensitivity: float = 50.0
    olfactory_sensitivity: float = 50.0
    gustatory_sensitivity: float = 50.0
    proprioceptive_sensitivity: float = 50.0
    interoceptive_sensitivity: float = 50.0
    vestibular_sensitivity: float = 50.0

    # Physiological state
    baseline_energy: float = 50.0
    fatigue_resistance: float = 50.0
    pain_threshold: float = 50.0
    stress_reactivity: float = 50.0  # HPA axis
    autonomic_balance: float = 50.0  # Sympathetic vs parasympathetic

    # Chronotype and rhythms
    chronotype: float = 50.0  # 0=early bird, 100=night owl
    circadian_stability: float = 50.0

    @property
    def sensory_sensitivity(self) -> float:
        """Overall sensory processing sensitivity"""
        return np.mean([
            self.visual_sensitivity,
            self.auditory_sensitivity,
            self.tactile_sensitivity,
            self.olfactory_sensitivity,
            self.gustatory_sensitivity,
        ])

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.visual_sensitivity, self.auditory_sensitivity,
            self.tactile_sensitivity, self.olfactory_sensitivity,
            self.gustatory_sensitivity, self.proprioceptive_sensitivity,
            self.interoceptive_sensitivity, self.vestibular_sensitivity,
            self.baseline_energy, self.fatigue_resistance,
            self.pain_threshold, self.stress_reactivity,
            self.autonomic_balance, self.chronotype, self.circadian_stability
        ]) / 100.0


@dataclass
class Layer1_Affective:
    """
    Affective-Motivational System
    Emotions, moods, and their regulation
    """
    # Baseline affect
    positive_affectivity: float = 50.0
    negative_affectivity: float = 50.0

    # Specific emotion dispositions (0-100)
    joy: float = 50.0
    sadness: float = 50.0
    fear: float = 50.0
    anger: float = 50.0
    disgust: float = 50.0
    surprise: float = 50.0

    # Self-conscious emotions
    shame_proneness: float = 50.0
    guilt_proneness: float = 50.0
    pride: float = 50.0
    embarrassment_proneness: float = 50.0

    # Emotion characteristics
    emotional_intensity: float = 50.0
    emotional_reactivity: float = 50.0
    emotional_recovery: float = 50.0  # Return to baseline speed
    emotion_differentiation: float = 50.0  # Granularity

    # Regulation
    emotion_regulation_capacity: float = 50.0
    suppression_tendency: float = 50.0
    reappraisal_tendency: float = 50.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.positive_affectivity, self.negative_affectivity,
            self.joy, self.sadness, self.fear, self.anger,
            self.disgust, self.surprise, self.shame_proneness,
            self.guilt_proneness, self.pride, self.embarrassment_proneness,
            self.emotional_intensity, self.emotional_reactivity,
            self.emotional_recovery, self.emotion_differentiation,
            self.emotion_regulation_capacity, self.suppression_tendency,
            self.reappraisal_tendency
        ]) / 100.0


@dataclass
class Layer2_Motivational:
    """
    Motivational Systems
    Drives, needs, goals
    """
    # Basic drives (0-100)
    survival_drive: float = 50.0
    safety_drive: float = 50.0

    # Psychological needs (Self-Determination Theory)
    autonomy_need: float = 50.0
    competence_need: float = 50.0
    relatedness_need: float = 50.0

    # Social motivations
    status_motivation: float = 50.0
    affiliation_motivation: float = 50.0
    intimacy_motivation: float = 50.0
    achievement_motivation: float = 50.0
    power_motivation: float = 50.0

    # Existential motivations
    meaning_motivation: float = 50.0
    self_actualization: float = 50.0
    transcendence_motivation: float = 50.0
    legacy_motivation: float = 50.0
    authenticity_drive: float = 50.0

    # Approach/Avoidance
    approach_tendency: float = 50.0
    avoidance_tendency: float = 50.0
    risk_tolerance: float = 50.0
    novelty_seeking: float = 50.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.survival_drive, self.safety_drive,
            self.autonomy_need, self.competence_need, self.relatedness_need,
            self.status_motivation, self.affiliation_motivation,
            self.intimacy_motivation, self.achievement_motivation,
            self.power_motivation, self.meaning_motivation,
            self.self_actualization, self.transcendence_motivation,
            self.legacy_motivation, self.authenticity_drive,
            self.approach_tendency, self.avoidance_tendency,
            self.risk_tolerance, self.novelty_seeking
        ]) / 100.0


@dataclass
class Layer3_Cognitive:
    """
    Cognitive Architecture
    Processing, memory, reasoning, theory of mind
    """
    # Processing characteristics
    processing_speed: float = 50.0
    working_memory_capacity: float = 50.0
    cognitive_flexibility: float = 50.0
    attention_span: float = 50.0
    attention_control: float = 50.0

    # Reasoning styles
    analytical_thinking: float = 50.0
    intuitive_thinking: float = 50.0
    abstract_thinking: float = 50.0
    concrete_thinking: float = 50.0

    # Theory of Mind (CRITICAL)
    tom_depth: int = 2  # k-ToM level (0-5)
    perspective_taking: float = 50.0
    mentalizing_capacity: float = 50.0
    empathy_cognitive: float = 50.0

    # Metacognition
    metacognitive_awareness: float = 50.0
    uncertainty_tolerance: float = 50.0
    cognitive_need: float = 50.0  # Need for cognition
    cognitive_closure_need: float = 50.0

    # Learning
    learning_rate: float = 50.0
    pattern_recognition: float = 50.0
    counterfactual_reasoning: float = 50.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.processing_speed, self.working_memory_capacity,
            self.cognitive_flexibility, self.attention_span,
            self.attention_control, self.analytical_thinking,
            self.intuitive_thinking, self.abstract_thinking,
            self.concrete_thinking, self.tom_depth / 5.0 * 100,
            self.perspective_taking, self.mentalizing_capacity,
            self.empathy_cognitive, self.metacognitive_awareness,
            self.uncertainty_tolerance, self.cognitive_need,
            self.cognitive_closure_need, self.learning_rate,
            self.pattern_recognition, self.counterfactual_reasoning
        ]) / 100.0


@dataclass
class Layer4_Self:
    """
    Self-System
    Identity, self-concept, agency
    """
    # Self-concept structure
    self_coherence: float = 50.0
    self_complexity: float = 50.0
    self_clarity: float = 50.0

    # Self-evaluation
    self_esteem: float = 50.0
    self_efficacy: float = 50.0
    self_worth: float = 50.0

    # Self processes
    self_awareness: float = 50.0
    self_reflection: float = 50.0
    self_compassion: float = 50.0
    self_criticism: float = 50.0

    # Agency
    agency_sense: float = 50.0
    locus_of_control: float = 50.0  # Internal (0) vs External (100)
    perceived_control: float = 50.0

    # Identity
    identity_stability: float = 50.0
    identity_exploration: float = 50.0
    authenticity: float = 50.0

    # Temporal self
    past_self_connection: float = 50.0
    future_self_connection: float = 50.0
    narrative_coherence: float = 50.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.self_coherence, self.self_complexity, self.self_clarity,
            self.self_esteem, self.self_efficacy, self.self_worth,
            self.self_awareness, self.self_reflection, self.self_compassion,
            self.self_criticism, self.agency_sense, self.locus_of_control,
            self.perceived_control, self.identity_stability,
            self.identity_exploration, self.authenticity,
            self.past_self_connection, self.future_self_connection,
            self.narrative_coherence
        ]) / 100.0


@dataclass
class Layer5_Values:
    """
    Value System - SUCCESS DOMAIN WEIGHTS (PRIVATE)
    These determine what the agent optimizes for.
    """
    # Physical/Health values
    health_value: float = 50.0
    longevity_value: float = 50.0
    vitality_value: float = 50.0

    # Economic values
    wealth_value: float = 50.0
    security_value: float = 50.0
    status_value: float = 50.0

    # Professional values
    achievement_value: float = 50.0
    mastery_value: float = 50.0
    impact_value: float = 50.0

    # Relational values
    intimacy_value: float = 50.0
    family_value: float = 50.0
    friendship_value: float = 50.0
    community_value: float = 50.0

    # Psychological values
    happiness_value: float = 50.0
    growth_value: float = 50.0
    peace_value: float = 50.0

    # Meaning values
    purpose_value: float = 50.0
    spirituality_value: float = 50.0
    creativity_value: float = 50.0

    # Legacy values
    legacy_value: float = 50.0
    generativity_value: float = 50.0

    # Moral values
    honesty_value: float = 50.0
    fairness_value: float = 50.0
    care_value: float = 50.0
    loyalty_value: float = 50.0
    authority_value: float = 50.0
    sanctity_value: float = 50.0

    def get_success_weights(self) -> Dict[str, float]:
        """
        Get normalized weights for the 9 success domains.
        These are PRIVATE - other agents cannot see these directly.
        """
        raw = {
            'physical': (self.health_value + self.longevity_value + self.vitality_value) / 3,
            'economic': (self.wealth_value + self.security_value + self.status_value) / 3,
            'professional': (self.achievement_value + self.mastery_value + self.impact_value) / 3,
            'relational': (self.intimacy_value + self.family_value + self.friendship_value + self.community_value) / 4,
            'psychological': (self.happiness_value + self.growth_value + self.peace_value) / 3,
            'meaning': (self.purpose_value + self.spirituality_value + self.creativity_value) / 3,
            'legacy': (self.legacy_value + self.generativity_value) / 2,
            '21st_century': 50.0,  # Default
            'structural': 50.0,  # Default
        }
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.health_value, self.longevity_value, self.vitality_value,
            self.wealth_value, self.security_value, self.status_value,
            self.achievement_value, self.mastery_value, self.impact_value,
            self.intimacy_value, self.family_value, self.friendship_value,
            self.community_value, self.happiness_value, self.growth_value,
            self.peace_value, self.purpose_value, self.spirituality_value,
            self.creativity_value, self.legacy_value, self.generativity_value,
            self.honesty_value, self.fairness_value, self.care_value,
            self.loyalty_value, self.authority_value, self.sanctity_value
        ]) / 100.0


@dataclass
class Layer6_Social:
    """
    Social Cognition and Relational Patterns
    """
    # Trust and cooperation
    trust_default: float = 50.0
    cooperation_tendency: float = 50.0
    competition_tendency: float = 50.0
    betrayal_sensitivity: float = 50.0

    # Social perception
    empathy_affective: float = 50.0
    social_monitoring: float = 50.0
    reputation_concern: float = 50.0
    fairness_sensitivity: float = 50.0

    # Group dynamics
    group_identity_strength: float = 50.0
    ingroup_favoritism: float = 50.0
    authority_orientation: float = 50.0
    conformity_tendency: float = 50.0

    # Reciprocity
    reciprocity_tracking: float = 50.0
    forgiveness_tendency: float = 50.0
    revenge_tendency: float = 50.0

    # Attachment style (encoded as dimensions)
    attachment_anxiety: float = 50.0
    attachment_avoidance: float = 50.0

    # Social skills
    social_competence: float = 50.0
    assertiveness: float = 50.0

    @property
    def attachment_style(self) -> AttachmentStyle:
        """Derive attachment style from anxiety and avoidance"""
        if self.attachment_anxiety < 50 and self.attachment_avoidance < 50:
            return AttachmentStyle.SECURE
        elif self.attachment_anxiety >= 50 and self.attachment_avoidance < 50:
            return AttachmentStyle.ANXIOUS_PREOCCUPIED
        elif self.attachment_anxiety < 50 and self.attachment_avoidance >= 50:
            return AttachmentStyle.DISMISSIVE_AVOIDANT
        else:
            return AttachmentStyle.FEARFUL_AVOIDANT

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.trust_default, self.cooperation_tendency,
            self.competition_tendency, self.betrayal_sensitivity,
            self.empathy_affective, self.social_monitoring,
            self.reputation_concern, self.fairness_sensitivity,
            self.group_identity_strength, self.ingroup_favoritism,
            self.authority_orientation, self.conformity_tendency,
            self.reciprocity_tracking, self.forgiveness_tendency,
            self.revenge_tendency, self.attachment_anxiety,
            self.attachment_avoidance, self.social_competence,
            self.assertiveness
        ]) / 100.0


@dataclass
class Layer7_Behavioral:
    """
    Behavioral Patterns and Personality
    Big Five + additional traits
    """
    # Big Five (0-100)
    openness: float = 50.0
    conscientiousness: float = 50.0
    extraversion: float = 50.0
    agreeableness: float = 50.0
    neuroticism: float = 50.0

    # HEXACO Honesty-Humility
    honesty_humility: float = 50.0

    # Dark Triad (clinical, not moral judgment)
    narcissism: float = 25.0  # Default low
    machiavellianism: float = 25.0
    psychopathy: float = 25.0

    # Behavioral style
    impulsivity: float = 50.0
    deliberation: float = 50.0
    persistence: float = 50.0

    # Coping
    coping_style: CopingStyle = CopingStyle.MIXED
    stress_resilience: float = 50.0

    # Regulatory focus
    promotion_focus: float = 50.0
    prevention_focus: float = 50.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.openness, self.conscientiousness, self.extraversion,
            self.agreeableness, self.neuroticism, self.honesty_humility,
            self.narcissism, self.machiavellianism, self.psychopathy,
            self.impulsivity, self.deliberation, self.persistence,
            self.stress_resilience, self.promotion_focus, self.prevention_focus
        ]) / 100.0


@dataclass
class Layer8_Narrative:
    """
    Narrative Identity and Life Story
    """
    # Life themes
    agency_theme: float = 50.0  # Self-determination in story
    communion_theme: float = 50.0  # Connection in story
    redemption_theme: float = 50.0  # Suffering → growth narrative
    contamination_theme: float = 50.0  # Good → bad narrative

    # Narrative structure
    narrative_coherence: float = 50.0
    narrative_complexity: float = 50.0
    meaning_making_tendency: float = 50.0

    # Life chapters
    current_life_chapter: str = "young_adult"
    chapter_satisfaction: float = 50.0

    # Turning points (count)
    turning_points_positive: int = 0
    turning_points_negative: int = 0

    # Future story
    future_optimism: float = 50.0
    goal_clarity: float = 50.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.agency_theme, self.communion_theme,
            self.redemption_theme, self.contamination_theme,
            self.narrative_coherence, self.narrative_complexity,
            self.meaning_making_tendency, self.chapter_satisfaction,
            self.turning_points_positive / 10.0 * 100,
            self.turning_points_negative / 10.0 * 100,
            self.future_optimism, self.goal_clarity
        ]) / 100.0


@dataclass
class Layer9_Existential:
    """
    Existential-Spiritual Dimension
    Ultimate concerns, meaning, transcendence
    """
    # Existential givens (Yalom)
    death_awareness: float = 50.0
    death_anxiety: float = 50.0
    death_acceptance: float = 50.0

    freedom_awareness: float = 50.0
    responsibility_acceptance: float = 50.0

    isolation_awareness: float = 50.0
    connection_seeking: float = 50.0

    meaninglessness_awareness: float = 50.0
    meaning_presence: float = 50.0

    # Spiritual
    transcendence_experiences: float = 50.0
    spiritual_practices: float = 50.0
    religious_commitment: float = 50.0

    # Worldview
    worldview_coherence: float = 50.0
    cosmic_significance: float = 50.0

    # Temporal orientation
    temporal_orientation: TemporalOrientation = TemporalOrientation.BALANCED

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.death_awareness, self.death_anxiety, self.death_acceptance,
            self.freedom_awareness, self.responsibility_acceptance,
            self.isolation_awareness, self.connection_seeking,
            self.meaninglessness_awareness, self.meaning_presence,
            self.transcendence_experiences, self.spiritual_practices,
            self.religious_commitment, self.worldview_coherence,
            self.cosmic_significance
        ]) / 100.0


@dataclass
class PsychosocialProfile:
    """
    Complete 10-Layer Psychosocial Profile
    Represents the full psychological constitution of an agent
    """
    layer0: Layer0_Biological = field(default_factory=Layer0_Biological)
    layer1: Layer1_Affective = field(default_factory=Layer1_Affective)
    layer2: Layer2_Motivational = field(default_factory=Layer2_Motivational)
    layer3: Layer3_Cognitive = field(default_factory=Layer3_Cognitive)
    layer4: Layer4_Self = field(default_factory=Layer4_Self)
    layer5: Layer5_Values = field(default_factory=Layer5_Values)
    layer6: Layer6_Social = field(default_factory=Layer6_Social)
    layer7: Layer7_Behavioral = field(default_factory=Layer7_Behavioral)
    layer8: Layer8_Narrative = field(default_factory=Layer8_Narrative)
    layer9: Layer9_Existential = field(default_factory=Layer9_Existential)

    def get_success_weights(self) -> Dict[str, float]:
        """Get the private success domain weights from Layer 5"""
        return self.layer5.get_success_weights()

    def to_vector(self) -> np.ndarray:
        """Convert entire profile to a single vector"""
        return np.concatenate([
            self.layer0.to_vector(),
            self.layer1.to_vector(),
            self.layer2.to_vector(),
            self.layer3.to_vector(),
            self.layer4.to_vector(),
            self.layer5.to_vector(),
            self.layer6.to_vector(),
            self.layer7.to_vector(),
            self.layer8.to_vector(),
            self.layer9.to_vector(),
        ])

    @property
    def total_dimensions(self) -> int:
        """Total number of dimensions across all layers"""
        return len(self.to_vector())

    @classmethod
    def sample_random(cls, rng: Optional[np.random.Generator] = None) -> 'PsychosocialProfile':
        """Sample a random profile from the psychosocial space"""
        if rng is None:
            rng = np.random.default_rng()

        def rand_val(mean: float = 50.0, std: float = 15.0) -> float:
            """Generate a random value with soft bounds"""
            val = rng.normal(mean, std)
            return float(np.clip(val, 0, 100))

        profile = cls()

        # Randomize Layer 0
        for field_name in ['visual_sensitivity', 'auditory_sensitivity', 'tactile_sensitivity',
                          'olfactory_sensitivity', 'gustatory_sensitivity', 'proprioceptive_sensitivity',
                          'interoceptive_sensitivity', 'vestibular_sensitivity', 'baseline_energy',
                          'fatigue_resistance', 'pain_threshold', 'stress_reactivity',
                          'autonomic_balance', 'chronotype', 'circadian_stability']:
            setattr(profile.layer0, field_name, rand_val())

        # Randomize Layer 1
        for field_name in ['positive_affectivity', 'negative_affectivity', 'joy', 'sadness',
                          'fear', 'anger', 'disgust', 'surprise', 'shame_proneness',
                          'guilt_proneness', 'pride', 'embarrassment_proneness',
                          'emotional_intensity', 'emotional_reactivity', 'emotional_recovery',
                          'emotion_differentiation', 'emotion_regulation_capacity',
                          'suppression_tendency', 'reappraisal_tendency']:
            setattr(profile.layer1, field_name, rand_val())

        # Continue for other layers...
        # Layer 2
        for field_name in ['survival_drive', 'safety_drive', 'autonomy_need', 'competence_need',
                          'relatedness_need', 'status_motivation', 'affiliation_motivation',
                          'intimacy_motivation', 'achievement_motivation', 'power_motivation',
                          'meaning_motivation', 'self_actualization', 'transcendence_motivation',
                          'legacy_motivation', 'authenticity_drive', 'approach_tendency',
                          'avoidance_tendency', 'risk_tolerance', 'novelty_seeking']:
            setattr(profile.layer2, field_name, rand_val())

        # Layer 3
        for field_name in ['processing_speed', 'working_memory_capacity', 'cognitive_flexibility',
                          'attention_span', 'attention_control', 'analytical_thinking',
                          'intuitive_thinking', 'abstract_thinking', 'concrete_thinking',
                          'perspective_taking', 'mentalizing_capacity', 'empathy_cognitive',
                          'metacognitive_awareness', 'uncertainty_tolerance', 'cognitive_need',
                          'cognitive_closure_need', 'learning_rate', 'pattern_recognition',
                          'counterfactual_reasoning']:
            setattr(profile.layer3, field_name, rand_val())
        profile.layer3.tom_depth = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.05, 0.15, 0.35, 0.25, 0.15, 0.05]))

        # Layer 4
        for field_name in ['self_coherence', 'self_complexity', 'self_clarity', 'self_esteem',
                          'self_efficacy', 'self_worth', 'self_awareness', 'self_reflection',
                          'self_compassion', 'self_criticism', 'agency_sense', 'locus_of_control',
                          'perceived_control', 'identity_stability', 'identity_exploration',
                          'authenticity', 'past_self_connection', 'future_self_connection',
                          'narrative_coherence']:
            setattr(profile.layer4, field_name, rand_val())

        # Layer 5
        for field_name in ['health_value', 'longevity_value', 'vitality_value', 'wealth_value',
                          'security_value', 'status_value', 'achievement_value', 'mastery_value',
                          'impact_value', 'intimacy_value', 'family_value', 'friendship_value',
                          'community_value', 'happiness_value', 'growth_value', 'peace_value',
                          'purpose_value', 'spirituality_value', 'creativity_value',
                          'legacy_value', 'generativity_value', 'honesty_value', 'fairness_value',
                          'care_value', 'loyalty_value', 'authority_value', 'sanctity_value']:
            setattr(profile.layer5, field_name, rand_val())

        # Layer 6
        for field_name in ['trust_default', 'cooperation_tendency', 'competition_tendency',
                          'betrayal_sensitivity', 'empathy_affective', 'social_monitoring',
                          'reputation_concern', 'fairness_sensitivity', 'group_identity_strength',
                          'ingroup_favoritism', 'authority_orientation', 'conformity_tendency',
                          'reciprocity_tracking', 'forgiveness_tendency', 'revenge_tendency',
                          'attachment_anxiety', 'attachment_avoidance', 'social_competence',
                          'assertiveness']:
            setattr(profile.layer6, field_name, rand_val())

        # Layer 7
        for field_name in ['openness', 'conscientiousness', 'extraversion', 'agreeableness',
                          'neuroticism', 'honesty_humility', 'narcissism', 'machiavellianism',
                          'psychopathy', 'impulsivity', 'deliberation', 'persistence',
                          'stress_resilience', 'promotion_focus', 'prevention_focus']:
            if field_name in ['narcissism', 'machiavellianism', 'psychopathy']:
                setattr(profile.layer7, field_name, rand_val(25, 15))  # Dark triad lower by default
            else:
                setattr(profile.layer7, field_name, rand_val())
        profile.layer7.coping_style = rng.choice(list(CopingStyle))

        # Layer 8
        for field_name in ['agency_theme', 'communion_theme', 'redemption_theme',
                          'contamination_theme', 'narrative_coherence', 'narrative_complexity',
                          'meaning_making_tendency', 'chapter_satisfaction', 'future_optimism',
                          'goal_clarity']:
            setattr(profile.layer8, field_name, rand_val())
        profile.layer8.turning_points_positive = int(rng.integers(0, 6))
        profile.layer8.turning_points_negative = int(rng.integers(0, 4))

        # Layer 9
        for field_name in ['death_awareness', 'death_anxiety', 'death_acceptance',
                          'freedom_awareness', 'responsibility_acceptance', 'isolation_awareness',
                          'connection_seeking', 'meaninglessness_awareness', 'meaning_presence',
                          'transcendence_experiences', 'spiritual_practices', 'religious_commitment',
                          'worldview_coherence', 'cosmic_significance']:
            setattr(profile.layer9, field_name, rand_val())
        profile.layer9.temporal_orientation = rng.choice(list(TemporalOrientation))

        return profile

    @classmethod
    def from_archetype(cls, archetype: str, rng: Optional[np.random.Generator] = None) -> 'PsychosocialProfile':
        """
        Create a profile from a named archetype.
        Archetypes provide meaningful starting configurations.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Start with random
        profile = cls.sample_random(rng)

        # Override based on archetype
        archetypes = {
            'hero': {
                'layer2.achievement_motivation': 80,
                'layer4.agency_sense': 85,
                'layer4.self_efficacy': 75,
                'layer7.conscientiousness': 70,
                'layer7.extraversion': 65,
                'layer8.agency_theme': 85,
            },
            'caregiver': {
                'layer2.relatedness_need': 85,
                'layer5.care_value': 90,
                'layer6.trust_default': 70,
                'layer6.cooperation_tendency': 80,
                'layer6.empathy_affective': 85,
                'layer7.agreeableness': 80,
                'layer8.communion_theme': 85,
            },
            'sage': {
                'layer3.processing_speed': 75,
                'layer3.abstract_thinking': 85,
                'layer3.tom_depth': 4,
                'layer5.mastery_value': 80,
                'layer7.openness': 85,
                'layer9.meaning_presence': 80,
            },
            'rebel': {
                'layer2.autonomy_need': 90,
                'layer5.authenticity_drive': 85,
                'layer6.authority_orientation': 20,
                'layer6.conformity_tendency': 20,
                'layer7.openness': 75,
                'layer7.agreeableness': 35,
            },
            'creator': {
                'layer3.cognitive_flexibility': 80,
                'layer5.creativity_value': 90,
                'layer7.openness': 90,
                'layer2.novelty_seeking': 80,
            },
            'ruler': {
                'layer2.power_motivation': 80,
                'layer2.status_motivation': 75,
                'layer5.authority_value': 80,
                'layer6.authority_orientation': 75,
                'layer7.conscientiousness': 75,
            },
            'innocent': {
                'layer1.positive_affectivity': 80,
                'layer4.self_esteem': 70,
                'layer6.trust_default': 85,
                'layer8.future_optimism': 85,
            },
            'explorer': {
                'layer2.novelty_seeking': 90,
                'layer5.growth_value': 80,
                'layer7.openness': 85,
                'layer2.risk_tolerance': 75,
            },
            'everyman': {
                # All dimensions close to 50 - default profile
            },
            'jester': {
                'layer1.joy': 80,
                'layer1.positive_affectivity': 75,
                'layer7.extraversion': 80,
                'layer6.social_competence': 75,
            },
            'lover': {
                'layer2.intimacy_motivation': 90,
                'layer5.intimacy_value': 85,
                'layer6.empathy_affective': 80,
                'layer6.attachment_anxiety': 60,
            },
            'magician': {
                'layer3.counterfactual_reasoning': 85,
                'layer9.transcendence_experiences': 80,
                'layer5.spirituality_value': 75,
            },
            'outlaw': {
                'layer7.agreeableness': 30,
                'layer6.conformity_tendency': 20,
                'layer2.autonomy_need': 85,
                'layer7.machiavellianism': 60,
            },
            'zombie': {
                # Special archetype: Low ToM, reactive
                'layer3.tom_depth': 0,
                'layer3.perspective_taking': 20,
                'layer3.mentalizing_capacity': 20,
                'layer3.empathy_cognitive': 25,
                'layer6.empathy_affective': 25,
                'layer4.self_awareness': 30,
                'layer3.metacognitive_awareness': 25,
            }
        }

        mods = archetypes.get(archetype.lower(), {})
        for path, value in mods.items():
            parts = path.split('.')
            layer_name = parts[0]
            field_name = parts[1]
            layer = getattr(profile, layer_name)
            setattr(layer, field_name, value)

        return profile

    def describe(self) -> str:
        """Generate a human-readable description of this profile"""
        lines = [
            "=== Psychosocial Profile ===",
            f"Total Dimensions: {self.total_dimensions}",
            "",
            "--- Key Characteristics ---",
            f"ToM Depth: {self.layer3.tom_depth} (k-level)",
            f"Attachment Style: {self.layer6.attachment_style.value}",
            f"Big Five: O={self.layer7.openness:.0f} C={self.layer7.conscientiousness:.0f} "
            f"E={self.layer7.extraversion:.0f} A={self.layer7.agreeableness:.0f} N={self.layer7.neuroticism:.0f}",
            f"Self-Esteem: {self.layer4.self_esteem:.0f}",
            f"Trust Default: {self.layer6.trust_default:.0f}",
            f"Cooperation: {self.layer6.cooperation_tendency:.0f}",
            f"Meaning Presence: {self.layer9.meaning_presence:.0f}",
            "",
            "--- Top Values ---",
        ]

        # Get top 5 values
        weights = self.get_success_weights()
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for domain, weight in sorted_weights:
            lines.append(f"  {domain}: {weight:.2%}")

        return '\n'.join(lines)
