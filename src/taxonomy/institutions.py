"""
Institutions and Social Archetypes Module

Represents the institutional structures and social patterns that
shape agent behavior and outcomes. Based on the comprehensive
"Institutions and Social Archetypes/Givens" framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np


class InstitutionType(Enum):
    """Types of social institutions"""
    FAMILY = "family"
    EDUCATION = "education"
    ECONOMIC = "economic"
    POLITICAL = "political"
    HEALTHCARE = "healthcare"
    RELIGIOUS = "religious"
    MEDIA = "media"


class FamilyForm(Enum):
    """Family structure types"""
    NUCLEAR_TRADITIONAL = "nuclear_traditional"
    NUCLEAR_DUAL_INCOME = "nuclear_dual_income"
    SINGLE_PARENT = "single_parent"
    BLENDED = "blended"
    EXTENDED = "extended"
    SAME_SEX = "same_sex"
    CHOSEN = "chosen"
    CHILDLESS = "childless"


class SocioeconomicClass(Enum):
    """Socioeconomic strata"""
    UPPER = "upper"
    UPPER_MIDDLE = "upper_middle"
    MIDDLE = "middle"
    LOWER_MIDDLE = "lower_middle"
    WORKING = "working"
    POOR = "poor"


@dataclass
class Institution:
    """
    Base class for institutional structures.
    Institutions mediate between individual action and life outcomes.
    """
    type: InstitutionType
    name: str
    quality: float = 50.0  # 0-100, institutional quality/access
    influence: float = 50.0  # How much this institution affects agent

    def get_success_modifier(self) -> float:
        """Return modifier for success outcomes based on institutional quality"""
        return (self.quality / 100.0) * (self.influence / 100.0)

    def get_behavior_constraints(self) -> Dict[str, float]:
        """Return behavioral constraints imposed by this institution"""
        return {}


@dataclass
class FamilyInstitution(Institution):
    """
    Family as institution - primary socialization, attachment, resource transmission
    """
    form: FamilyForm = FamilyForm.NUCLEAR_TRADITIONAL
    stability: float = 50.0
    attachment_quality: float = 50.0
    resource_level: float = 50.0
    cultural_capital: float = 50.0
    social_capital: float = 50.0

    def __post_init__(self):
        self.type = InstitutionType.FAMILY
        self.name = f"Family ({self.form.value})"

    def get_attachment_style_probability(self) -> Dict[str, float]:
        """Return probability of different attachment styles"""
        if self.attachment_quality >= 70 and self.stability >= 60:
            return {'secure': 0.6, 'anxious': 0.15, 'avoidant': 0.15, 'disorganized': 0.1}
        elif self.attachment_quality >= 50:
            return {'secure': 0.35, 'anxious': 0.25, 'avoidant': 0.25, 'disorganized': 0.15}
        else:
            return {'secure': 0.15, 'anxious': 0.30, 'avoidant': 0.30, 'disorganized': 0.25}

    def get_success_path_modifier(self) -> Dict[str, float]:
        """Modifiers for different success domains"""
        return {
            'relational': self.attachment_quality / 100.0,
            'psychological': (self.attachment_quality + self.stability) / 200.0,
            'economic': (self.resource_level + self.cultural_capital) / 200.0,
            'professional': (self.cultural_capital + self.social_capital) / 200.0,
        }


@dataclass
class EducationInstitution(Institution):
    """
    Education as institution - knowledge, credentials, socialization
    """
    level: str = "secondary"  # primary, secondary, tertiary, graduate
    quality_tier: str = "public"  # public, private, elite
    accessibility: float = 50.0
    credential_value: float = 50.0
    socialization_quality: float = 50.0
    critical_thinking_emphasis: float = 50.0

    def __post_init__(self):
        self.type = InstitutionType.EDUCATION
        self.name = f"Education ({self.level}, {self.quality_tier})"

    def get_success_path_modifier(self) -> Dict[str, float]:
        """Modifiers for success domains"""
        return {
            'professional': (self.quality + self.credential_value) / 200.0,
            'economic': self.credential_value / 100.0,
            '21st_century': self.critical_thinking_emphasis / 100.0,
        }


@dataclass
class EconomicInstitution(Institution):
    """
    Economic institutions - labor markets, financial systems, welfare
    """
    labor_market_access: float = 50.0
    financial_system_access: float = 50.0
    safety_net_strength: float = 50.0
    mobility_opportunity: float = 50.0
    exploitation_risk: float = 30.0  # Lower is better

    def __post_init__(self):
        self.type = InstitutionType.ECONOMIC
        self.name = "Economic System"

    def get_success_path_modifier(self) -> Dict[str, float]:
        """Modifiers for success domains"""
        return {
            'economic': (self.labor_market_access + self.financial_system_access) / 200.0,
            'professional': self.mobility_opportunity / 100.0,
            'structural': (self.safety_net_strength - self.exploitation_risk) / 100.0,
        }


@dataclass
class PoliticalInstitution(Institution):
    """
    Political institutions - government, legal system, rights
    """
    democratic_quality: float = 50.0
    rule_of_law: float = 50.0
    rights_protection: float = 50.0
    corruption_level: float = 30.0  # Lower is better
    voice_access: float = 50.0

    def __post_init__(self):
        self.type = InstitutionType.POLITICAL
        self.name = "Political System"

    def get_success_path_modifier(self) -> Dict[str, float]:
        """Modifiers for success domains"""
        return {
            'structural': (self.rights_protection + self.rule_of_law - self.corruption_level) / 200.0,
            'psychological': self.voice_access / 100.0,
        }


@dataclass
class HealthcareInstitution(Institution):
    """
    Healthcare institutions - access, quality, coverage
    """
    access_level: float = 50.0
    quality_of_care: float = 50.0
    coverage_breadth: float = 50.0
    affordability: float = 50.0
    mental_health_access: float = 40.0

    def __post_init__(self):
        self.type = InstitutionType.HEALTHCARE
        self.name = "Healthcare System"

    def get_success_path_modifier(self) -> Dict[str, float]:
        """Modifiers for success domains"""
        return {
            'physical': (self.access_level + self.quality_of_care) / 200.0,
            'psychological': self.mental_health_access / 100.0,
            'economic': self.affordability / 100.0,  # Medical bankruptcy prevention
        }


@dataclass
class ReligiousInstitution(Institution):
    """
    Religious/Spiritual institutions - meaning, community, moral framework
    """
    engagement_level: float = 50.0  # 0=secular, 100=deeply religious
    community_strength: float = 50.0
    meaning_provision: float = 50.0
    moral_framework_clarity: float = 50.0
    pluralism: float = 50.0  # Tolerance of other views

    def __post_init__(self):
        self.type = InstitutionType.RELIGIOUS
        self.name = "Religious/Spiritual"

    def get_success_path_modifier(self) -> Dict[str, float]:
        """Modifiers for success domains"""
        return {
            'meaning': (self.meaning_provision * self.engagement_level) / 10000.0,
            'relational': (self.community_strength * self.engagement_level) / 10000.0,
            'psychological': self.moral_framework_clarity / 100.0 * 0.3,
        }


@dataclass
class MediaInstitution(Institution):
    """
    Media institutions - information, attention, discourse
    """
    information_quality: float = 50.0
    pluralism: float = 50.0
    misinformation_prevalence: float = 40.0  # Lower is better
    attention_capture: float = 60.0
    digital_literacy_support: float = 40.0

    def __post_init__(self):
        self.type = InstitutionType.MEDIA
        self.name = "Media Environment"

    def get_success_path_modifier(self) -> Dict[str, float]:
        """Modifiers for success domains"""
        epistemic_quality = (self.information_quality + self.pluralism -
                           self.misinformation_prevalence + self.digital_literacy_support) / 400.0
        return {
            '21st_century': epistemic_quality,
            'psychological': -self.attention_capture / 200.0,  # Attention capture is negative
        }


@dataclass
class InstitutionalContext:
    """
    Complete institutional context for an agent.
    Represents the full set of institutions that shape their life.
    """
    family: FamilyInstitution = field(default_factory=FamilyInstitution)
    education: EducationInstitution = field(default_factory=EducationInstitution)
    economic: EconomicInstitution = field(default_factory=EconomicInstitution)
    political: PoliticalInstitution = field(default_factory=PoliticalInstitution)
    healthcare: HealthcareInstitution = field(default_factory=HealthcareInstitution)
    religious: ReligiousInstitution = field(default_factory=ReligiousInstitution)
    media: MediaInstitution = field(default_factory=MediaInstitution)

    # Socioeconomic position
    socioeconomic_class: SocioeconomicClass = SocioeconomicClass.MIDDLE

    def get_combined_modifiers(self) -> Dict[str, float]:
        """Get combined success modifiers from all institutions"""
        modifiers = {}
        institutions = [
            self.family, self.education, self.economic,
            self.political, self.healthcare, self.religious, self.media
        ]

        for inst in institutions:
            inst_mods = inst.get_success_path_modifier()
            for domain, mod in inst_mods.items():
                if domain in modifiers:
                    modifiers[domain] += mod
                else:
                    modifiers[domain] = mod

        # Normalize
        n_institutions = len(institutions)
        return {k: v / n_institutions for k, v in modifiers.items()}

    def get_structural_advantage(self) -> float:
        """Calculate overall structural advantage (0-1)"""
        class_values = {
            SocioeconomicClass.UPPER: 1.0,
            SocioeconomicClass.UPPER_MIDDLE: 0.8,
            SocioeconomicClass.MIDDLE: 0.5,
            SocioeconomicClass.LOWER_MIDDLE: 0.35,
            SocioeconomicClass.WORKING: 0.2,
            SocioeconomicClass.POOR: 0.1,
        }
        base = class_values[self.socioeconomic_class]
        modifiers = self.get_combined_modifiers()
        structural_mod = modifiers.get('structural', 0.5)
        return (base + structural_mod) / 2

    @classmethod
    def sample_random(cls, rng: Optional[np.random.Generator] = None) -> 'InstitutionalContext':
        """Sample a random institutional context"""
        if rng is None:
            rng = np.random.default_rng()

        def rand_val(mean: float = 50.0, std: float = 15.0) -> float:
            return float(np.clip(rng.normal(mean, std), 0, 100))

        # Sample socioeconomic class with realistic distribution
        classes = list(SocioeconomicClass)
        probs = [0.05, 0.15, 0.30, 0.25, 0.15, 0.10]  # Approximate distribution
        soc_class = rng.choice(classes, p=probs)

        # Class affects institutional access
        class_mod = {
            SocioeconomicClass.UPPER: 30,
            SocioeconomicClass.UPPER_MIDDLE: 15,
            SocioeconomicClass.MIDDLE: 0,
            SocioeconomicClass.LOWER_MIDDLE: -10,
            SocioeconomicClass.WORKING: -20,
            SocioeconomicClass.POOR: -30,
        }[soc_class]

        ctx = cls()
        ctx.socioeconomic_class = soc_class

        # Family
        ctx.family = FamilyInstitution(
            form=rng.choice(list(FamilyForm)),
            quality=rand_val(50 + class_mod),
            influence=rand_val(70),
            stability=rand_val(55 + class_mod * 0.5),
            attachment_quality=rand_val(50),
            resource_level=rand_val(50 + class_mod),
            cultural_capital=rand_val(50 + class_mod),
            social_capital=rand_val(50 + class_mod * 0.7),
        )

        # Education
        ctx.education = EducationInstitution(
            level=rng.choice(['primary', 'secondary', 'tertiary', 'graduate'],
                           p=[0.1, 0.3, 0.4, 0.2]),
            quality_tier=rng.choice(['public', 'private', 'elite'],
                                   p=[0.6, 0.3, 0.1]),
            quality=rand_val(50 + class_mod),
            influence=rand_val(60),
            accessibility=rand_val(60 + class_mod * 0.5),
            credential_value=rand_val(50 + class_mod),
            socialization_quality=rand_val(50),
            critical_thinking_emphasis=rand_val(45),
        )

        # Economic
        ctx.economic = EconomicInstitution(
            quality=rand_val(50),
            influence=rand_val(70),
            labor_market_access=rand_val(50 + class_mod),
            financial_system_access=rand_val(50 + class_mod),
            safety_net_strength=rand_val(40),
            mobility_opportunity=rand_val(45),
            exploitation_risk=rand_val(40 - class_mod * 0.3),
        )

        # Political
        ctx.political = PoliticalInstitution(
            quality=rand_val(50),
            influence=rand_val(40),
            democratic_quality=rand_val(55),
            rule_of_law=rand_val(55),
            rights_protection=rand_val(55 + class_mod * 0.3),
            corruption_level=rand_val(35),
            voice_access=rand_val(50 + class_mod * 0.5),
        )

        # Healthcare
        ctx.healthcare = HealthcareInstitution(
            quality=rand_val(50 + class_mod * 0.5),
            influence=rand_val(50),
            access_level=rand_val(50 + class_mod),
            quality_of_care=rand_val(50 + class_mod * 0.7),
            coverage_breadth=rand_val(45),
            affordability=rand_val(40 - class_mod * 0.3),
            mental_health_access=rand_val(35),
        )

        # Religious
        ctx.religious = ReligiousInstitution(
            quality=rand_val(50),
            influence=rand_val(40),
            engagement_level=rand_val(40),
            community_strength=rand_val(50),
            meaning_provision=rand_val(50),
            moral_framework_clarity=rand_val(55),
            pluralism=rand_val(50),
        )

        # Media
        ctx.media = MediaInstitution(
            quality=rand_val(45),
            influence=rand_val(60),
            information_quality=rand_val(45),
            pluralism=rand_val(50),
            misinformation_prevalence=rand_val(45),
            attention_capture=rand_val(65),
            digital_literacy_support=rand_val(40),
        )

        return ctx

    def describe(self) -> str:
        """Generate human-readable description"""
        lines = [
            "=== Institutional Context ===",
            f"Socioeconomic Class: {self.socioeconomic_class.value}",
            f"Structural Advantage: {self.get_structural_advantage():.2f}",
            "",
            "--- Institutions ---",
            f"Family: {self.family.form.value} (quality={self.family.quality:.0f})",
            f"Education: {self.education.level} {self.education.quality_tier} (quality={self.education.quality:.0f})",
            f"Economic: access={self.economic.labor_market_access:.0f}, safety_net={self.economic.safety_net_strength:.0f}",
            f"Political: democracy={self.political.democratic_quality:.0f}, rights={self.political.rights_protection:.0f}",
            f"Healthcare: access={self.healthcare.access_level:.0f}, quality={self.healthcare.quality_of_care:.0f}",
            f"Religious: engagement={self.religious.engagement_level:.0f}",
            f"Media: info_quality={self.media.information_quality:.0f}, attention_capture={self.media.attention_capture:.0f}",
            "",
            "--- Combined Modifiers ---",
        ]
        mods = self.get_combined_modifiers()
        for domain, mod in sorted(mods.items()):
            lines.append(f"  {domain}: {mod:+.3f}")

        return '\n'.join(lines)
