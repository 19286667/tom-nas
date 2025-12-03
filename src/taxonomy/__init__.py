"""
ToM-NAS Taxonomy Module

Comprehensive psychological and social taxonomies for agent modeling.
Integrates:
- 10-layer Psychosocial Profile (80+ dimensions)
- 9-domain Success/Failure State (120+ dimensions)
- Institutional and Social Archetypes
- Prototype/Stereotype Meaning Framework
"""

from .psychosocial import (
    PsychosocialProfile,
    Layer0_Biological,
    Layer1_Affective,
    Layer2_Motivational,
    Layer3_Cognitive,
    Layer4_Self,
    Layer5_Values,
    Layer6_Social,
    Layer7_Behavioral,
    Layer8_Narrative,
    Layer9_Existential,
)

from .success import (
    SuccessState,
    Domain1_Physical,
    Domain2_Economic,
    Domain3_Professional,
    Domain4_Relational,
    Domain5_Psychological,
    Domain6_Meaning,
    Domain7_Legacy,
    Domain8_21stCentury,
    Domain9_Structural,
)

from .sampling import (
    AgentSampler,
    ArchetypeSampler,
    EnvironmentSampler,
)

from .institutions import (
    Institution,
    InstitutionalContext,
    FamilyInstitution,
    EducationInstitution,
    EconomicInstitution,
    PoliticalInstitution,
    HealthcareInstitution,
    ReligiousInstitution,
    MediaInstitution,
)

__all__ = [
    # Psychosocial
    'PsychosocialProfile',
    'Layer0_Biological',
    'Layer1_Affective',
    'Layer2_Motivational',
    'Layer3_Cognitive',
    'Layer4_Self',
    'Layer5_Values',
    'Layer6_Social',
    'Layer7_Behavioral',
    'Layer8_Narrative',
    'Layer9_Existential',
    # Success
    'SuccessState',
    'Domain1_Physical',
    'Domain2_Economic',
    'Domain3_Professional',
    'Domain4_Relational',
    'Domain5_Psychological',
    'Domain6_Meaning',
    'Domain7_Legacy',
    'Domain8_21stCentury',
    'Domain9_Structural',
    # Sampling
    'AgentSampler',
    'ArchetypeSampler',
    'EnvironmentSampler',
    # Institutions
    'Institution',
    'InstitutionalContext',
    'FamilyInstitution',
    'EducationInstitution',
    'EconomicInstitution',
    'PoliticalInstitution',
    'HealthcareInstitution',
    'ReligiousInstitution',
    'MediaInstitution',
]
