"""
Success/Failure Taxonomy - 9-Domain Life Outcome Framework

Based on the comprehensive "Exhaustive Taxonomy of Human Success & Failure"
covering physical, economic, professional, relational, psychological,
meaning, legacy, 21st-century, and structural dimensions.

Total: 9 domains, 120+ dimensions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class HealthStatus(Enum):
    """Physical health outcome levels"""
    THRIVING = "thriving"
    GOOD = "good"
    MANAGING = "managing"
    STRUGGLING = "struggling"
    FAILING = "failing"
    TERMINAL = "terminal"


class WealthLevel(Enum):
    """Economic outcome levels"""
    EXTREME_WEALTH = "extreme_wealth"  # Top 0.1%
    HIGH_WEALTH = "high_wealth"  # Top 1-10%
    UPPER_MIDDLE = "upper_middle"  # Top 10-30%
    MIDDLE = "middle"  # 30-70%
    LOWER_MIDDLE = "lower_middle"  # 70-90%
    POOR = "poor"  # Bottom 10-20%
    EXTREME_POVERTY = "extreme_poverty"


class CareerStatus(Enum):
    """Career trajectory types"""
    ASCENDING = "ascending"
    PLATEAU_SATISFIED = "plateau_satisfied"
    PLATEAU_STUCK = "plateau_stuck"
    DESCENDING = "descending"
    ABSENT = "absent"


class RelationshipStatus(Enum):
    """Partnership outcome levels"""
    THRIVING = "thriving"
    SATISFACTORY = "satisfactory"
    STRUGGLING = "struggling"
    FAILED = "failed"
    ABSENT_CHOSEN = "absent_chosen"
    ABSENT_UNCHOSEN = "absent_unchosen"


class WellbeingLevel(Enum):
    """Subjective wellbeing levels"""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    SUFFERING = "suffering"


class MeaningLevel(Enum):
    """Life meaning outcome levels"""
    DEEP = "deep"
    ADEQUATE = "adequate"
    STRUGGLING = "struggling"
    CRISIS = "crisis"


@dataclass
class Domain1_Physical:
    """
    Physical/Health Domain
    Biological vitality and longevity outcomes
    """
    # Overall health (0-100)
    health: float = 50.0
    functional_capacity: float = 50.0
    energy_vitality: float = 50.0

    # Specific health dimensions
    chronic_conditions_count: int = 0
    pain_level: float = 20.0  # Lower is better
    mobility: float = 80.0
    sensory_function: float = 80.0

    # Health behaviors
    exercise_regularity: float = 50.0
    nutrition_quality: float = 50.0
    sleep_quality: float = 50.0
    substance_use: float = 20.0  # Lower is better

    # Mental health
    mental_health: float = 50.0
    depression_level: float = 20.0  # Lower is better
    anxiety_level: float = 20.0  # Lower is better

    # Longevity indicators
    biological_age_offset: float = 0.0  # Negative = younger than chronological
    life_expectancy_adjustment: float = 0.0

    @property
    def status(self) -> HealthStatus:
        """Derive health status from metrics"""
        avg = (self.health + self.functional_capacity + self.energy_vitality) / 3
        if avg >= 80:
            return HealthStatus.THRIVING
        elif avg >= 65:
            return HealthStatus.GOOD
        elif avg >= 50:
            return HealthStatus.MANAGING
        elif avg >= 30:
            return HealthStatus.STRUGGLING
        else:
            return HealthStatus.FAILING

    def compute_score(self) -> float:
        """Compute normalized domain score (0-1)"""
        positive = np.mean([
            self.health, self.functional_capacity, self.energy_vitality,
            self.mobility, self.sensory_function, self.exercise_regularity,
            self.nutrition_quality, self.sleep_quality, self.mental_health
        ])
        negative = np.mean([
            self.pain_level, self.substance_use, self.depression_level,
            self.anxiety_level
        ])
        return (positive - negative / 2) / 100.0


@dataclass
class Domain2_Economic:
    """
    Economic/Material Domain
    Wealth, income, and financial security outcomes
    """
    # Wealth metrics
    net_worth: float = 50.0  # Normalized 0-100
    income: float = 50.0
    income_stability: float = 50.0
    income_growth: float = 50.0

    # Security
    emergency_fund_months: float = 1.0
    debt_to_asset_ratio: float = 0.5
    insurance_coverage: float = 50.0

    # Housing
    housing_security: float = 50.0
    housing_quality: float = 50.0
    housing_ownership: float = 0.0  # 0=renting, 100=owned outright

    # Basic needs
    food_security: float = 80.0
    healthcare_access: float = 50.0
    transportation_access: float = 60.0

    # Financial trajectory
    wealth_trajectory: float = 50.0  # Growing/stable/declining
    retirement_readiness: float = 30.0

    @property
    def wealth_level(self) -> WealthLevel:
        """Derive wealth level from metrics"""
        if self.net_worth >= 95:
            return WealthLevel.EXTREME_WEALTH
        elif self.net_worth >= 80:
            return WealthLevel.HIGH_WEALTH
        elif self.net_worth >= 65:
            return WealthLevel.UPPER_MIDDLE
        elif self.net_worth >= 40:
            return WealthLevel.MIDDLE
        elif self.net_worth >= 20:
            return WealthLevel.LOWER_MIDDLE
        elif self.net_worth >= 10:
            return WealthLevel.POOR
        else:
            return WealthLevel.EXTREME_POVERTY

    def compute_score(self) -> float:
        """Compute normalized domain score (0-1)"""
        positive = np.mean([
            self.net_worth, self.income, self.income_stability,
            self.income_growth, self.emergency_fund_months * 10,  # Scale months
            self.insurance_coverage, self.housing_security,
            self.housing_quality, self.food_security,
            self.healthcare_access, self.transportation_access,
            self.wealth_trajectory, self.retirement_readiness
        ])
        negative = self.debt_to_asset_ratio * 50  # Scale ratio
        return max(0, (positive - negative) / 100.0)


@dataclass
class Domain3_Professional:
    """
    Career/Professional Domain
    Work achievement, satisfaction, and trajectory
    """
    # Employment
    employment_status: float = 50.0  # 0=unemployed, 100=thriving employed
    job_security: float = 50.0
    underemployment: float = 30.0  # Lower is better

    # Position
    position_level: float = 50.0  # Hierarchy position
    scope_responsibility: float = 50.0
    autonomy: float = 50.0

    # Compensation
    compensation_satisfaction: float = 50.0
    benefits_quality: float = 50.0

    # Growth
    skill_development: float = 50.0
    career_advancement: float = 50.0
    career_trajectory: float = 50.0

    # Quality
    work_engagement: float = 50.0
    job_satisfaction: float = 50.0
    work_meaning: float = 50.0
    work_life_balance: float = 50.0

    # Recognition
    professional_reputation: float = 50.0
    recognition_received: float = 50.0

    # Burnout (lower is better)
    burnout_level: float = 30.0

    @property
    def status(self) -> CareerStatus:
        """Derive career status from metrics"""
        if self.career_trajectory >= 65 and self.employment_status >= 50:
            return CareerStatus.ASCENDING
        elif self.career_trajectory >= 40 and self.job_satisfaction >= 50:
            return CareerStatus.PLATEAU_SATISFIED
        elif self.career_trajectory >= 40:
            return CareerStatus.PLATEAU_STUCK
        elif self.employment_status < 30:
            return CareerStatus.ABSENT
        else:
            return CareerStatus.DESCENDING

    def compute_score(self) -> float:
        """Compute normalized domain score (0-1)"""
        positive = np.mean([
            self.employment_status, self.job_security, self.position_level,
            self.scope_responsibility, self.autonomy, self.compensation_satisfaction,
            self.benefits_quality, self.skill_development, self.career_advancement,
            self.career_trajectory, self.work_engagement, self.job_satisfaction,
            self.work_meaning, self.work_life_balance, self.professional_reputation,
            self.recognition_received
        ])
        negative = np.mean([self.underemployment, self.burnout_level])
        return max(0, (positive - negative / 2) / 100.0)


@dataclass
class Domain4_Relational:
    """
    Relational Domain
    Partnerships, family, friendships, community
    """
    # Partnership
    partnership_satisfaction: float = 50.0
    partnership_stability: float = 50.0
    intimacy_quality: float = 50.0
    partnership_support: float = 50.0

    # Family
    family_connection: float = 50.0
    parenting_satisfaction: float = 50.0  # If applicable
    family_support_given: float = 50.0
    family_support_received: float = 50.0

    # Friendship
    close_friend_count: int = 3
    friendship_satisfaction: float = 50.0
    social_support_network: float = 50.0

    # Community
    community_belonging: float = 50.0
    social_integration: float = 50.0
    civic_participation: float = 30.0

    # Relational quality
    loneliness: float = 30.0  # Lower is better
    social_isolation: float = 30.0  # Lower is better
    relationship_conflict: float = 30.0  # Lower is better

    @property
    def status(self) -> RelationshipStatus:
        """Derive relationship status from metrics"""
        if self.partnership_satisfaction >= 75 and self.partnership_stability >= 70:
            return RelationshipStatus.THRIVING
        elif self.partnership_satisfaction >= 50:
            return RelationshipStatus.SATISFACTORY
        elif self.partnership_satisfaction >= 30:
            return RelationshipStatus.STRUGGLING
        elif self.partnership_satisfaction > 0:
            return RelationshipStatus.FAILED
        elif self.loneliness < 40:
            return RelationshipStatus.ABSENT_CHOSEN
        else:
            return RelationshipStatus.ABSENT_UNCHOSEN

    def compute_score(self) -> float:
        """Compute normalized domain score (0-1)"""
        positive = np.mean([
            self.partnership_satisfaction, self.partnership_stability,
            self.intimacy_quality, self.partnership_support,
            self.family_connection, self.parenting_satisfaction,
            self.family_support_given, self.family_support_received,
            min(100, self.close_friend_count * 15),  # Scale count
            self.friendship_satisfaction, self.social_support_network,
            self.community_belonging, self.social_integration,
            self.civic_participation
        ])
        negative = np.mean([self.loneliness, self.social_isolation, self.relationship_conflict])
        return max(0, (positive - negative) / 100.0)


@dataclass
class Domain5_Psychological:
    """
    Psychological/Wellbeing Domain
    Subjective wellbeing, identity, and growth
    """
    # Subjective wellbeing
    life_satisfaction: float = 50.0
    positive_affect: float = 50.0
    negative_affect: float = 30.0  # Lower is better
    happiness: float = 50.0

    # Eudaimonic wellbeing
    autonomy: float = 50.0
    environmental_mastery: float = 50.0
    personal_growth: float = 50.0
    positive_relations: float = 50.0
    purpose_in_life: float = 50.0
    self_acceptance: float = 50.0

    # Identity
    identity_clarity: float = 50.0
    identity_security: float = 50.0
    self_esteem: float = 50.0
    self_efficacy: float = 50.0

    # Growth
    learning_growth: float = 50.0
    wisdom_development: float = 50.0

    # Regulation
    emotional_stability: float = 50.0
    stress_management: float = 50.0
    resilience: float = 50.0

    @property
    def level(self) -> WellbeingLevel:
        """Derive wellbeing level from metrics"""
        avg = (self.life_satisfaction + self.happiness +
               self.positive_affect - self.negative_affect / 2) / 3
        if avg >= 70:
            return WellbeingLevel.HIGH
        elif avg >= 45:
            return WellbeingLevel.MODERATE
        elif avg >= 25:
            return WellbeingLevel.LOW
        else:
            return WellbeingLevel.SUFFERING

    def compute_score(self) -> float:
        """Compute normalized domain score (0-1)"""
        positive = np.mean([
            self.life_satisfaction, self.positive_affect, self.happiness,
            self.autonomy, self.environmental_mastery, self.personal_growth,
            self.positive_relations, self.purpose_in_life, self.self_acceptance,
            self.identity_clarity, self.identity_security, self.self_esteem,
            self.self_efficacy, self.learning_growth, self.wisdom_development,
            self.emotional_stability, self.stress_management, self.resilience
        ])
        negative = self.negative_affect
        return max(0, (positive - negative / 3) / 100.0)


@dataclass
class Domain6_Meaning:
    """
    Meaning/Purpose Domain
    Existential fulfillment, values, and significance
    """
    # Meaning presence
    meaning_presence: float = 50.0
    purpose_clarity: float = 50.0
    life_coherence: float = 50.0
    significance_feeling: float = 50.0

    # Values alignment
    values_clarity: float = 50.0
    values_living: float = 50.0  # Acting according to values
    integrity: float = 50.0

    # Transcendence
    spiritual_wellbeing: float = 50.0
    transcendent_experiences: float = 30.0
    connection_to_larger: float = 50.0

    # Achievement
    goal_progress: float = 50.0
    accomplishment_sense: float = 50.0
    impact_feeling: float = 50.0

    # Existential
    death_acceptance: float = 40.0
    existential_anxiety: float = 40.0  # Lower is better (mostly)
    meaninglessness: float = 30.0  # Lower is better

    @property
    def level(self) -> MeaningLevel:
        """Derive meaning level from metrics"""
        avg = (self.meaning_presence + self.purpose_clarity +
               self.life_coherence + self.significance_feeling) / 4
        if avg >= 70:
            return MeaningLevel.DEEP
        elif avg >= 45:
            return MeaningLevel.ADEQUATE
        elif avg >= 25:
            return MeaningLevel.STRUGGLING
        else:
            return MeaningLevel.CRISIS

    def compute_score(self) -> float:
        """Compute normalized domain score (0-1)"""
        positive = np.mean([
            self.meaning_presence, self.purpose_clarity, self.life_coherence,
            self.significance_feeling, self.values_clarity, self.values_living,
            self.integrity, self.spiritual_wellbeing, self.transcendent_experiences,
            self.connection_to_larger, self.goal_progress, self.accomplishment_sense,
            self.impact_feeling, self.death_acceptance
        ])
        negative = np.mean([self.existential_anxiety, self.meaninglessness])
        return max(0, (positive - negative / 2) / 100.0)


@dataclass
class Domain7_Legacy:
    """
    Legacy/Generativity Domain
    Impact beyond self and lifespan
    """
    # Generativity
    generativity: float = 50.0
    mentoring: float = 30.0
    teaching_impact: float = 30.0
    care_for_next_generation: float = 50.0

    # Creation
    creative_output: float = 30.0
    lasting_works: float = 20.0
    contribution_to_field: float = 30.0

    # Impact
    lives_positively_affected: float = 30.0  # Scale
    community_contribution: float = 40.0
    social_impact: float = 30.0

    # Recognition
    remembered_likelihood: float = 30.0
    reputation_legacy: float = 30.0

    # Transmission
    values_transmitted: float = 40.0
    wisdom_shared: float = 40.0
    family_legacy: float = 40.0

    def compute_score(self) -> float:
        """Compute normalized domain score (0-1)"""
        return np.mean([
            self.generativity, self.mentoring, self.teaching_impact,
            self.care_for_next_generation, self.creative_output,
            self.lasting_works, self.contribution_to_field,
            self.lives_positively_affected, self.community_contribution,
            self.social_impact, self.remembered_likelihood,
            self.reputation_legacy, self.values_transmitted,
            self.wisdom_shared, self.family_legacy
        ]) / 100.0


@dataclass
class Domain8_21stCentury:
    """
    21st Century Specific Domain
    Digital, environmental, adaptability, epistemic
    """
    # Digital
    digital_fluency: float = 50.0
    technology_enhancement: float = 50.0  # Tech helping vs hurting life
    digital_wellbeing: float = 50.0
    online_reputation: float = 50.0
    digital_addiction: float = 30.0  # Lower is better

    # Environmental
    environmental_footprint: float = 50.0  # Lower is better for planet
    sustainability_practices: float = 40.0
    climate_anxiety: float = 40.0  # Lower is better

    # Adaptability
    change_adaptability: float = 50.0
    uncertainty_navigation: float = 50.0
    continuous_learning: float = 50.0
    skill_relevance: float = 50.0

    # Epistemic
    information_diet_quality: float = 50.0
    critical_thinking: float = 50.0
    misinformation_resistance: float = 50.0
    worldview_accuracy: float = 50.0

    # Global
    cross_cultural_competence: float = 40.0
    global_awareness: float = 50.0

    def compute_score(self) -> float:
        """Compute normalized domain score (0-1)"""
        positive = np.mean([
            self.digital_fluency, self.technology_enhancement,
            self.digital_wellbeing, self.online_reputation,
            self.sustainability_practices, self.change_adaptability,
            self.uncertainty_navigation, self.continuous_learning,
            self.skill_relevance, self.information_diet_quality,
            self.critical_thinking, self.misinformation_resistance,
            self.worldview_accuracy, self.cross_cultural_competence,
            self.global_awareness
        ])
        negative = np.mean([
            self.digital_addiction, self.climate_anxiety,
            self.environmental_footprint
        ])
        return max(0, (positive - negative / 2) / 100.0)


@dataclass
class Domain9_Structural:
    """
    Structural Factors Domain
    Background advantages/disadvantages, luck
    """
    # Birth circumstances
    birth_wealth: float = 50.0
    birth_country_advantage: float = 50.0
    family_stability_childhood: float = 50.0
    early_education_quality: float = 50.0

    # Demographic factors
    majority_group_membership: float = 50.0
    discrimination_exposure: float = 30.0  # Lower is better
    geographic_advantage: float = 50.0

    # Health lottery
    genetic_health_advantage: float = 50.0
    appearance_advantage: float = 50.0

    # Social capital
    network_quality: float = 50.0
    cultural_capital: float = 50.0

    # Luck
    circumstantial_luck: float = 50.0
    disaster_avoidance: float = 70.0

    # Current access
    opportunity_access: float = 50.0
    social_mobility: float = 50.0
    safety_net_availability: float = 50.0

    def compute_score(self) -> float:
        """Compute normalized domain score (0-1)"""
        positive = np.mean([
            self.birth_wealth, self.birth_country_advantage,
            self.family_stability_childhood, self.early_education_quality,
            self.majority_group_membership, self.geographic_advantage,
            self.genetic_health_advantage, self.appearance_advantage,
            self.network_quality, self.cultural_capital,
            self.circumstantial_luck, self.disaster_avoidance,
            self.opportunity_access, self.social_mobility,
            self.safety_net_availability
        ])
        negative = self.discrimination_exposure
        return max(0, (positive - negative / 2) / 100.0)


@dataclass
class SuccessState:
    """
    Complete Success/Failure State across all 9 domains
    """
    alive: bool = True

    domain1: Domain1_Physical = field(default_factory=Domain1_Physical)
    domain2: Domain2_Economic = field(default_factory=Domain2_Economic)
    domain3: Domain3_Professional = field(default_factory=Domain3_Professional)
    domain4: Domain4_Relational = field(default_factory=Domain4_Relational)
    domain5: Domain5_Psychological = field(default_factory=Domain5_Psychological)
    domain6: Domain6_Meaning = field(default_factory=Domain6_Meaning)
    domain7: Domain7_Legacy = field(default_factory=Domain7_Legacy)
    domain8: Domain8_21stCentury = field(default_factory=Domain8_21stCentury)
    domain9: Domain9_Structural = field(default_factory=Domain9_Structural)

    def compute_fitness(self, weights: Dict[str, float]) -> float:
        """
        Compute overall fitness using provided weights.
        Weights come from the agent's PRIVATE value system (Layer 5).
        """
        if not self.alive:
            return 0.0

        domain_scores = {
            'physical': self.domain1.compute_score(),
            'economic': self.domain2.compute_score(),
            'professional': self.domain3.compute_score(),
            'relational': self.domain4.compute_score(),
            'psychological': self.domain5.compute_score(),
            'meaning': self.domain6.compute_score(),
            'legacy': self.domain7.compute_score(),
            '21st_century': self.domain8.compute_score(),
            'structural': self.domain9.compute_score(),
        }

        total = 0.0
        for domain, score in domain_scores.items():
            weight = weights.get(domain, 1.0 / 9.0)
            total += weight * score

        return total

    def get_domain_scores(self) -> Dict[str, float]:
        """Get scores for all domains"""
        return {
            'physical': self.domain1.compute_score(),
            'economic': self.domain2.compute_score(),
            'professional': self.domain3.compute_score(),
            'relational': self.domain4.compute_score(),
            'psychological': self.domain5.compute_score(),
            'meaning': self.domain6.compute_score(),
            'legacy': self.domain7.compute_score(),
            '21st_century': self.domain8.compute_score(),
            'structural': self.domain9.compute_score(),
        }

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for neural network input"""
        scores = self.get_domain_scores()
        return np.array([scores[k] for k in sorted(scores.keys())])

    @classmethod
    def sample_random(cls, rng: Optional[np.random.Generator] = None) -> 'SuccessState':
        """Sample a random success state"""
        if rng is None:
            rng = np.random.default_rng()

        def rand_val(mean: float = 50.0, std: float = 15.0) -> float:
            return float(np.clip(rng.normal(mean, std), 0, 100))

        state = cls()

        # Domain 1 - Physical
        for field_name in ['health', 'functional_capacity', 'energy_vitality',
                          'mobility', 'sensory_function', 'exercise_regularity',
                          'nutrition_quality', 'sleep_quality', 'mental_health',
                          'biological_age_offset', 'life_expectancy_adjustment']:
            if hasattr(state.domain1, field_name):
                setattr(state.domain1, field_name, rand_val())
        state.domain1.pain_level = rand_val(30, 15)
        state.domain1.substance_use = rand_val(25, 15)
        state.domain1.depression_level = rand_val(30, 15)
        state.domain1.anxiety_level = rand_val(35, 15)
        state.domain1.chronic_conditions_count = int(rng.poisson(0.5))

        # Domain 2 - Economic
        for field_name in ['net_worth', 'income', 'income_stability', 'income_growth',
                          'insurance_coverage', 'housing_security', 'housing_quality',
                          'housing_ownership', 'food_security', 'healthcare_access',
                          'transportation_access', 'wealth_trajectory', 'retirement_readiness']:
            setattr(state.domain2, field_name, rand_val())
        state.domain2.emergency_fund_months = max(0, rng.normal(2, 2))
        state.domain2.debt_to_asset_ratio = max(0, rng.normal(0.4, 0.3))

        # Domain 3 - Professional
        for field_name in ['employment_status', 'job_security', 'position_level',
                          'scope_responsibility', 'autonomy', 'compensation_satisfaction',
                          'benefits_quality', 'skill_development', 'career_advancement',
                          'career_trajectory', 'work_engagement', 'job_satisfaction',
                          'work_meaning', 'work_life_balance', 'professional_reputation',
                          'recognition_received']:
            setattr(state.domain3, field_name, rand_val())
        state.domain3.underemployment = rand_val(30, 15)
        state.domain3.burnout_level = rand_val(35, 15)

        # Domain 4 - Relational
        for field_name in ['partnership_satisfaction', 'partnership_stability',
                          'intimacy_quality', 'partnership_support', 'family_connection',
                          'parenting_satisfaction', 'family_support_given',
                          'family_support_received', 'friendship_satisfaction',
                          'social_support_network', 'community_belonging',
                          'social_integration', 'civic_participation']:
            setattr(state.domain4, field_name, rand_val())
        state.domain4.close_friend_count = max(0, int(rng.poisson(3)))
        state.domain4.loneliness = rand_val(35, 15)
        state.domain4.social_isolation = rand_val(30, 15)
        state.domain4.relationship_conflict = rand_val(30, 15)

        # Domain 5 - Psychological
        for field_name in ['life_satisfaction', 'positive_affect', 'happiness',
                          'autonomy', 'environmental_mastery', 'personal_growth',
                          'positive_relations', 'purpose_in_life', 'self_acceptance',
                          'identity_clarity', 'identity_security', 'self_esteem',
                          'self_efficacy', 'learning_growth', 'wisdom_development',
                          'emotional_stability', 'stress_management', 'resilience']:
            setattr(state.domain5, field_name, rand_val())
        state.domain5.negative_affect = rand_val(35, 15)

        # Domain 6 - Meaning
        for field_name in ['meaning_presence', 'purpose_clarity', 'life_coherence',
                          'significance_feeling', 'values_clarity', 'values_living',
                          'integrity', 'spiritual_wellbeing', 'transcendent_experiences',
                          'connection_to_larger', 'goal_progress', 'accomplishment_sense',
                          'impact_feeling', 'death_acceptance']:
            setattr(state.domain6, field_name, rand_val())
        state.domain6.existential_anxiety = rand_val(40, 15)
        state.domain6.meaninglessness = rand_val(30, 15)

        # Domain 7 - Legacy
        for field_name in ['generativity', 'mentoring', 'teaching_impact',
                          'care_for_next_generation', 'creative_output', 'lasting_works',
                          'contribution_to_field', 'lives_positively_affected',
                          'community_contribution', 'social_impact', 'remembered_likelihood',
                          'reputation_legacy', 'values_transmitted', 'wisdom_shared',
                          'family_legacy']:
            setattr(state.domain7, field_name, rand_val(40, 20))

        # Domain 8 - 21st Century
        for field_name in ['digital_fluency', 'technology_enhancement', 'digital_wellbeing',
                          'online_reputation', 'sustainability_practices', 'change_adaptability',
                          'uncertainty_navigation', 'continuous_learning', 'skill_relevance',
                          'information_diet_quality', 'critical_thinking',
                          'misinformation_resistance', 'worldview_accuracy',
                          'cross_cultural_competence', 'global_awareness']:
            setattr(state.domain8, field_name, rand_val())
        state.domain8.digital_addiction = rand_val(35, 15)
        state.domain8.climate_anxiety = rand_val(40, 15)
        state.domain8.environmental_footprint = rand_val(55, 15)

        # Domain 9 - Structural
        for field_name in ['birth_wealth', 'birth_country_advantage',
                          'family_stability_childhood', 'early_education_quality',
                          'majority_group_membership', 'geographic_advantage',
                          'genetic_health_advantage', 'appearance_advantage',
                          'network_quality', 'cultural_capital', 'circumstantial_luck',
                          'disaster_avoidance', 'opportunity_access', 'social_mobility',
                          'safety_net_availability']:
            setattr(state.domain9, field_name, rand_val())
        state.domain9.discrimination_exposure = rand_val(30, 20)

        return state

    def describe(self) -> str:
        """Generate human-readable description"""
        scores = self.get_domain_scores()
        lines = [
            "=== Success State ===",
            f"Alive: {self.alive}",
            "",
            "--- Domain Scores ---",
        ]
        for domain, score in sorted(scores.items()):
            bar_len = int(score * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            lines.append(f"{domain:15s} [{bar}] {score:.2f}")

        lines.extend([
            "",
            "--- Status Summary ---",
            f"Health: {self.domain1.status.value}",
            f"Wealth: {self.domain2.wealth_level.value}",
            f"Career: {self.domain3.status.value}",
            f"Relationship: {self.domain4.status.value}",
            f"Wellbeing: {self.domain5.level.value}",
            f"Meaning: {self.domain6.level.value}",
        ])

        return '\n'.join(lines)
