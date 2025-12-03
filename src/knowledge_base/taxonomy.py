"""
The 80-Dimension Taxonomy

Implements the full taxonomy structure organizing human experience into
Mundane Life, Institutional, and Aesthetic layers. Each layer contains
dimensions that position concepts in semantic space.

Theoretical Foundation:
- Mundane Life: Based on activity theory and practice theory
- Institutions: Based on new institutional economics and sociological institutionalism
- Aesthetics: Based on aesthetic philosophy and cognitive aesthetics

The taxonomy is the coordinate system of Indra's Net.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import numpy as np

from .schemas import TaxonomyDimension, ConceptualDomain


class TaxonomyLayer(Enum):
    """The three main layers of the taxonomy."""
    MUNDANE = auto()       # Layer 1: Everyday life activities
    INSTITUTIONAL = auto()  # Layer 2: Formal social structures
    AESTHETIC = auto()     # Layer 3: Aesthetic and symbolic dimensions


@dataclass
class MundaneTaxonomy:
    """
    Layer 1: The Mundane Life Taxonomy

    Dimensions capturing everyday human activities and practices.
    Organized into sublayers covering the full spectrum of daily life.

    Sublayers (10 sublayers, ~27 dimensions):
    1.1 Sustenance - Eating, drinking, food preparation
    1.2 Bodily Care - Hygiene, health, grooming
    1.3 Mobility - Movement, transportation
    1.4 Domestic Labor - Cleaning, maintenance
    1.5 Work/Occupation - Productive activities
    1.6 Leisure - Recreation, entertainment
    1.7 Social Interaction - Communication, relationships
    1.8 Commerce - Buying, selling, trading
    1.9 Learning - Education, skill acquisition
    1.10 Rest - Sleep, relaxation
    """

    dimensions: List[TaxonomyDimension] = field(default_factory=list)
    domains: List[ConceptualDomain] = field(default_factory=list)

    def __post_init__(self):
        self._build_mundane_dimensions()
        self._build_mundane_domains()

    def _build_mundane_dimensions(self):
        """Build the 27 mundane life dimensions."""
        mundane_dims = [
            # 1.1 Sustenance (dims 1-3)
            (1, "Nourishment", "Sustenance", "Hunger", "Satiation",
             "Basic metabolic needs driving eating behavior"),
            (2, "Gustatory_Pleasure", "Sustenance", "Aversive", "Pleasurable",
             "Hedonic quality of food experience"),
            (3, "Commensality", "Sustenance", "Solitary", "Communal",
             "Social dimension of eating (alone vs. with others)"),

            # 1.2 Bodily Care (dims 4-6)
            (4, "Hygiene_State", "Bodily Care", "Unclean", "Pristine",
             "Physical cleanliness and self-care"),
            (5, "Health_Maintenance", "Bodily Care", "Neglected", "Optimized",
             "Active health management behaviors"),
            (6, "Appearance_Investment", "Bodily Care", "Minimal", "Elaborate",
             "Effort invested in physical presentation"),

            # 1.3 Mobility (dims 7-9)
            (7, "Locomotion_Mode", "Mobility", "Stationary", "Mobile",
             "Current movement state"),
            (8, "Spatial_Range", "Mobility", "Proximal", "Distal",
             "Distance of typical movements"),
            (9, "Transport_Autonomy", "Mobility", "Dependent", "Independent",
             "Self-sufficiency in movement"),

            # 1.4 Domestic Labor (dims 10-12)
            (10, "Space_Maintenance", "Domestic Labor", "Chaotic", "Orderly",
             "State of living space organization"),
            (11, "Object_Care", "Domestic Labor", "Neglected", "Maintained",
             "Maintenance of possessions"),
            (12, "Domestic_Burden", "Domestic Labor", "Light", "Heavy",
             "Weight of household responsibilities"),

            # 1.5 Work/Occupation (dims 13-16)
            (13, "Productive_Engagement", "Work", "Idle", "Industrious",
             "Level of productive activity"),
            (14, "Skill_Application", "Work", "Unskilled", "Expert",
             "Complexity of skills employed"),
            (15, "Work_Autonomy", "Work", "Directed", "Self-directed",
             "Control over work activities"),
            (16, "Economic_Return", "Work", "Unrewarded", "Lucrative",
             "Material compensation received"),

            # 1.6 Leisure (dims 17-19)
            (17, "Leisure_Engagement", "Leisure", "Bored", "Absorbed",
             "Quality of recreational experience"),
            (18, "Activity_Type", "Leisure", "Passive", "Active",
             "Physical engagement in leisure"),
            (19, "Social_Dimension", "Leisure", "Solitary", "Collective",
             "Social nature of leisure activities"),

            # 1.7 Social Interaction (dims 20-23)
            (20, "Social_Frequency", "Social", "Isolated", "Connected",
             "Frequency of social contact"),
            (21, "Relationship_Depth", "Social", "Superficial", "Intimate",
             "Depth of social bonds"),
            (22, "Social_Status", "Social", "Marginal", "Central",
             "Position in social network"),
            (23, "Communication_Mode", "Social", "Tacit", "Explicit",
             "Directness of communication"),

            # 1.8 Commerce (dims 24-25)
            (24, "Exchange_Activity", "Commerce", "Consumer", "Producer",
             "Role in economic exchange"),
            (25, "Transaction_Complexity", "Commerce", "Simple", "Complex",
             "Complexity of commercial interactions"),

            # 1.9 Learning (dim 26)
            (26, "Learning_Mode", "Learning", "Receiving", "Discovering",
             "Approach to knowledge acquisition"),

            # 1.10 Rest (dim 27)
            (27, "Restoration_State", "Rest", "Depleted", "Restored",
             "Energy restoration through rest"),
        ]

        for dim_tuple in mundane_dims:
            dim = TaxonomyDimension(
                id=dim_tuple[0],
                name=dim_tuple[1],
                layer="Mundane",
                sublayer=dim_tuple[2],
                low_anchor=dim_tuple[3],
                high_anchor=dim_tuple[4],
                theoretical_basis=dim_tuple[5],
            )
            self.dimensions.append(dim)

    def _build_mundane_domains(self):
        """Build conceptual domains for mundane life."""
        domains = [
            ConceptualDomain(
                id="mundane_sustenance",
                name="Sustenance",
                description="All activities related to eating, drinking, and nourishment",
                dimensions=[1, 2, 3],
            ),
            ConceptualDomain(
                id="mundane_body",
                name="Bodily Care",
                description="Personal care, hygiene, and health maintenance",
                dimensions=[4, 5, 6],
            ),
            ConceptualDomain(
                id="mundane_mobility",
                name="Mobility",
                description="Movement through physical space",
                dimensions=[7, 8, 9],
            ),
            ConceptualDomain(
                id="mundane_domestic",
                name="Domestic Labor",
                description="Household maintenance and care",
                dimensions=[10, 11, 12],
            ),
            ConceptualDomain(
                id="mundane_work",
                name="Work/Occupation",
                description="Productive and economic activities",
                dimensions=[13, 14, 15, 16],
            ),
            ConceptualDomain(
                id="mundane_leisure",
                name="Leisure",
                description="Recreation and entertainment",
                dimensions=[17, 18, 19],
            ),
            ConceptualDomain(
                id="mundane_social",
                name="Social Interaction",
                description="Interpersonal relationships and communication",
                dimensions=[20, 21, 22, 23],
            ),
            ConceptualDomain(
                id="mundane_commerce",
                name="Commerce",
                description="Economic exchange and transactions",
                dimensions=[24, 25],
            ),
            ConceptualDomain(
                id="mundane_learning",
                name="Learning",
                description="Knowledge and skill acquisition",
                dimensions=[26],
            ),
            ConceptualDomain(
                id="mundane_rest",
                name="Rest",
                description="Sleep and restoration",
                dimensions=[27],
            ),
        ]
        self.domains = domains


@dataclass
class InstitutionalTaxonomy:
    """
    Layer 2: The Institutional Taxonomy

    Dimensions capturing formal social structures, norms, and power relations.
    Based on new institutional economics and sociological institutionalism.

    Sublayers (8 sublayers, ~27 dimensions):
    2.1 Governance - Political and administrative structures
    2.2 Economy - Market structures and economic systems
    2.3 Law - Legal systems and enforcement
    2.4 Religion - Sacred institutions and practices
    2.5 Education - Knowledge transmission systems
    2.6 Family - Kinship structures
    2.7 Healthcare - Medical institutions
    2.8 Media - Information and communication systems
    """

    dimensions: List[TaxonomyDimension] = field(default_factory=list)
    domains: List[ConceptualDomain] = field(default_factory=list)

    def __post_init__(self):
        self._build_institutional_dimensions()
        self._build_institutional_domains()

    def _build_institutional_dimensions(self):
        """Build the 27 institutional dimensions."""
        institutional_dims = [
            # 2.1 Governance (dims 28-31)
            (28, "Authority_Structure", "Governance", "Decentralized", "Centralized",
             "Distribution of political authority"),
            (29, "Legitimacy_Basis", "Governance", "Traditional", "Rational-Legal",
             "Source of institutional legitimacy (Weber)"),
            (30, "Participation_Level", "Governance", "Excluded", "Participant",
             "Degree of civic participation"),
            (31, "Transparency", "Governance", "Opaque", "Transparent",
             "Visibility of decision-making processes"),

            # 2.2 Economy (dims 32-35)
            (32, "Market_Integration", "Economy", "Subsistence", "Market-embedded",
             "Degree of market participation"),
            (33, "Property_Rights", "Economy", "Communal", "Private",
             "Nature of property ownership"),
            (34, "Economic_Hierarchy", "Economy", "Egalitarian", "Stratified",
             "Degree of economic inequality"),
            (35, "Exchange_Mode", "Economy", "Reciprocal", "Transactional",
             "Nature of economic exchange"),

            # 2.3 Law (dims 36-39)
            (36, "Legal_Formality", "Law", "Customary", "Codified",
             "Degree of legal formalization"),
            (37, "Enforcement_Intensity", "Law", "Lax", "Strict",
             "Rigor of rule enforcement"),
            (38, "Justice_Orientation", "Law", "Retributive", "Restorative",
             "Approach to justice"),
            (39, "Rights_Recognition", "Law", "Limited", "Expansive",
             "Scope of recognized rights"),

            # 2.4 Religion (dims 40-43)
            (40, "Sacred_Presence", "Religion", "Secular", "Sacral",
             "Presence of religious framing"),
            (41, "Ritual_Density", "Religion", "Minimal", "Elaborate",
             "Richness of ritual practice"),
            (42, "Orthodoxy", "Religion", "Heterodox", "Orthodox",
             "Strictness of doctrinal adherence"),
            (43, "Transcendence_Orientation", "Religion", "Immanent", "Transcendent",
             "Focus on worldly vs. otherworldly"),

            # 2.5 Education (dims 44-47)
            (44, "Knowledge_Access", "Education", "Restricted", "Universal",
             "Accessibility of education"),
            (45, "Pedagogy_Mode", "Education", "Rote", "Critical",
             "Approach to teaching"),
            (46, "Credential_Importance", "Education", "Informal", "Credentialed",
             "Role of formal qualifications"),
            (47, "Canon_Rigidity", "Education", "Fluid", "Fixed",
             "Flexibility of curriculum"),

            # 2.6 Family (dims 48-51)
            (48, "Kinship_Structure", "Family", "Nuclear", "Extended",
             "Scope of family unit"),
            (49, "Authority_Pattern", "Family", "Egalitarian", "Patriarchal",
             "Distribution of family authority"),
            (50, "Marital_Norms", "Family", "Flexible", "Rigid",
             "Strictness of marital expectations"),
            (51, "Intergenerational_Obligation", "Family", "Weak", "Strong",
             "Strength of generational bonds"),

            # 2.7 Healthcare (dims 52-54)
            (52, "Care_Access", "Healthcare", "Exclusive", "Universal",
             "Accessibility of healthcare"),
            (53, "Medical_Authority", "Healthcare", "Pluralistic", "Monopolistic",
             "Control over medical practice"),
            (54, "Body_Autonomy", "Healthcare", "Paternalistic", "Autonomous",
             "Patient control over treatment"),
        ]

        for dim_tuple in institutional_dims:
            dim = TaxonomyDimension(
                id=dim_tuple[0],
                name=dim_tuple[1],
                layer="Institutional",
                sublayer=dim_tuple[2],
                low_anchor=dim_tuple[3],
                high_anchor=dim_tuple[4],
                theoretical_basis=dim_tuple[5],
            )
            self.dimensions.append(dim)

    def _build_institutional_domains(self):
        """Build conceptual domains for institutions."""
        domains = [
            ConceptualDomain(
                id="inst_governance",
                name="Governance",
                description="Political and administrative structures",
                dimensions=[28, 29, 30, 31],
            ),
            ConceptualDomain(
                id="inst_economy",
                name="Economy",
                description="Market structures and economic systems",
                dimensions=[32, 33, 34, 35],
            ),
            ConceptualDomain(
                id="inst_law",
                name="Law",
                description="Legal systems and enforcement",
                dimensions=[36, 37, 38, 39],
            ),
            ConceptualDomain(
                id="inst_religion",
                name="Religion",
                description="Sacred institutions and practices",
                dimensions=[40, 41, 42, 43],
            ),
            ConceptualDomain(
                id="inst_education",
                name="Education",
                description="Knowledge transmission systems",
                dimensions=[44, 45, 46, 47],
            ),
            ConceptualDomain(
                id="inst_family",
                name="Family",
                description="Kinship structures and norms",
                dimensions=[48, 49, 50, 51],
            ),
            ConceptualDomain(
                id="inst_healthcare",
                name="Healthcare",
                description="Medical institutions and care",
                dimensions=[52, 53, 54],
            ),
        ]
        self.domains = domains


@dataclass
class AestheticTaxonomy:
    """
    Layer 3: The Aesthetic Taxonomy

    Dimensions capturing aesthetic, symbolic, and expressive qualities.
    Based on aesthetic philosophy and cognitive aesthetics.

    Sublayers (8 sublayers, ~26 dimensions):
    3.1 Visual Aesthetics - Color, form, composition
    3.2 Temporal Aesthetics - Rhythm, pace, duration
    3.3 Symbolic Density - Meaning saturation
    3.4 Emotional Register - Affective qualities
    3.5 Cultural Reference - Tradition and innovation
    3.6 Authenticity - Genuine vs. artificial
    3.7 Status Signaling - Social distinction
    3.8 Narrative Quality - Story and coherence
    """

    dimensions: List[TaxonomyDimension] = field(default_factory=list)
    domains: List[ConceptualDomain] = field(default_factory=list)

    def __post_init__(self):
        self._build_aesthetic_dimensions()
        self._build_aesthetic_domains()

    def _build_aesthetic_dimensions(self):
        """Build the 26 aesthetic dimensions."""
        aesthetic_dims = [
            # 3.1 Visual Aesthetics (dims 55-58)
            (55, "Chromatic_Intensity", "Visual", "Muted", "Vibrant",
             "Saturation and intensity of colors"),
            (56, "Formal_Complexity", "Visual", "Simple", "Ornate",
             "Complexity of visual forms"),
            (57, "Scale", "Visual", "Intimate", "Monumental",
             "Size and grandeur"),
            (58, "Light_Quality", "Visual", "Dim", "Radiant",
             "Luminosity and light treatment"),

            # 3.2 Temporal Aesthetics (dims 59-61)
            (59, "Temporal_Pace", "Temporal", "Languid", "Frenetic",
             "Speed of events and changes"),
            (60, "Rhythmic_Pattern", "Temporal", "Arrhythmic", "Patterned",
             "Regularity of temporal structure"),
            (61, "Duration_Sense", "Temporal", "Fleeting", "Enduring",
             "Perceived duration"),

            # 3.3 Symbolic Density (dims 62-65)
            (62, "Meaning_Saturation", "Symbolic", "Literal", "Symbolic",
             "Density of symbolic meaning"),
            (63, "Reference_Density", "Symbolic", "Self-contained", "Allusive",
             "Degree of external reference"),
            (64, "Ambiguity", "Symbolic", "Determinate", "Polysemous",
             "Openness to interpretation"),
            (65, "Irony_Level", "Symbolic", "Sincere", "Ironic",
             "Presence of ironic distance"),

            # 3.4 Emotional Register (dims 66-69)
            (66, "Affective_Intensity", "Emotional", "Subdued", "Intense",
             "Strength of emotional evocation"),
            (67, "Valence_Tone", "Emotional", "Dark", "Light",
             "Overall emotional coloring"),
            (68, "Tension_Level", "Emotional", "Relaxed", "Tense",
             "Degree of psychological tension"),
            (69, "Wonder_Evocation", "Emotional", "Mundane", "Wondrous",
             "Capacity to evoke awe"),

            # 3.5 Cultural Reference (dims 70-72)
            (70, "Temporal_Orientation", "Cultural", "Traditional", "Contemporary",
             "Reference to past vs. present"),
            (71, "Cultural_Specificity", "Cultural", "Universal", "Particular",
             "Degree of cultural specificity"),
            (72, "Innovation_Degree", "Cultural", "Conventional", "Avant-garde",
             "Novelty and experimentation"),

            # 3.6 Authenticity (dims 73-75)
            (73, "Authenticity", "Authenticity", "Artificial", "Genuine",
             "Perceived genuineness"),
            (74, "Craft_Evidence", "Authenticity", "Mass-produced", "Handcrafted",
             "Visible evidence of making"),
            (75, "Patina", "Authenticity", "Pristine", "Weathered",
             "Signs of age and use"),

            # 3.7 Status Signaling (dims 76-78)
            (76, "Status_Signal", "Status", "Humble", "Prestigious",
             "Social status indication"),
            (77, "Exclusivity", "Status", "Common", "Rare",
             "Scarcity and exclusivity"),
            (78, "Taste_Marker", "Status", "Lowbrow", "Highbrow",
             "Cultural capital indication"),

            # 3.8 Narrative Quality (dims 79-80)
            (79, "Narrative_Coherence", "Narrative", "Fragmented", "Unified",
             "Story coherence and completeness"),
            (80, "Genre_Adherence", "Narrative", "Transgressive", "Conventional",
             "Conformity to genre expectations"),
        ]

        for dim_tuple in aesthetic_dims:
            dim = TaxonomyDimension(
                id=dim_tuple[0],
                name=dim_tuple[1],
                layer="Aesthetic",
                sublayer=dim_tuple[2],
                low_anchor=dim_tuple[3],
                high_anchor=dim_tuple[4],
                theoretical_basis=dim_tuple[5],
            )
            self.dimensions.append(dim)

    def _build_aesthetic_domains(self):
        """Build conceptual domains for aesthetics."""
        domains = [
            ConceptualDomain(
                id="aesthetic_visual",
                name="Visual Aesthetics",
                description="Color, form, and composition",
                dimensions=[55, 56, 57, 58],
            ),
            ConceptualDomain(
                id="aesthetic_temporal",
                name="Temporal Aesthetics",
                description="Rhythm, pace, and duration",
                dimensions=[59, 60, 61],
            ),
            ConceptualDomain(
                id="aesthetic_symbolic",
                name="Symbolic Density",
                description="Meaning saturation and reference",
                dimensions=[62, 63, 64, 65],
            ),
            ConceptualDomain(
                id="aesthetic_emotional",
                name="Emotional Register",
                description="Affective qualities and evocation",
                dimensions=[66, 67, 68, 69],
            ),
            ConceptualDomain(
                id="aesthetic_cultural",
                name="Cultural Reference",
                description="Tradition and innovation",
                dimensions=[70, 71, 72],
            ),
            ConceptualDomain(
                id="aesthetic_authenticity",
                name="Authenticity",
                description="Genuine vs. artificial",
                dimensions=[73, 74, 75],
            ),
            ConceptualDomain(
                id="aesthetic_status",
                name="Status Signaling",
                description="Social distinction markers",
                dimensions=[76, 77, 78],
            ),
            ConceptualDomain(
                id="aesthetic_narrative",
                name="Narrative Quality",
                description="Story and coherence",
                dimensions=[79, 80],
            ),
        ]
        self.domains = domains


@dataclass
class FullTaxonomy:
    """
    The Complete 80-Dimension Taxonomy.

    Integrates all three layers (Mundane, Institutional, Aesthetic)
    into a unified coordinate system for semantic positioning.
    """

    mundane: MundaneTaxonomy = field(default_factory=MundaneTaxonomy)
    institutional: InstitutionalTaxonomy = field(default_factory=InstitutionalTaxonomy)
    aesthetic: AestheticTaxonomy = field(default_factory=AestheticTaxonomy)

    # Cross-domain mappings for conceptual metaphor
    metaphor_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        self._build_metaphor_mappings()

    def _build_metaphor_mappings(self):
        """Build conceptual metaphor mappings between domains."""
        # Based on Lakoff & Johnson's conceptual metaphor theory
        self.metaphor_mappings = {
            # SOCIAL IS SPATIAL mappings
            "mundane_mobility": {
                "mundane_social": "Social distance = Physical distance",
                "inst_governance": "Political position = Spatial position",
            },
            # ARGUMENT IS WAR
            "mundane_work": {
                "inst_law": "Legal contest = Labor contest",
            },
            # STATUS IS VERTICAL
            "aesthetic_status": {
                "inst_economy": "Economic hierarchy = Aesthetic hierarchy",
                "mundane_social": "Social position = Status display",
            },
            # TIME IS MONEY
            "aesthetic_temporal": {
                "inst_economy": "Time investment = Economic investment",
            },
        }

    def get_all_dimensions(self) -> List[TaxonomyDimension]:
        """Get all 80 dimensions across all layers."""
        return (
            self.mundane.dimensions +
            self.institutional.dimensions +
            self.aesthetic.dimensions
        )

    def get_all_domains(self) -> List[ConceptualDomain]:
        """Get all conceptual domains across all layers."""
        return (
            self.mundane.domains +
            self.institutional.domains +
            self.aesthetic.domains
        )

    def get_dimension_by_id(self, dim_id: int) -> Optional[TaxonomyDimension]:
        """Look up a dimension by its ID."""
        for dim in self.get_all_dimensions():
            if dim.id == dim_id:
                return dim
        return None

    def get_dimension_by_name(self, name: str) -> Optional[TaxonomyDimension]:
        """Look up a dimension by its name."""
        for dim in self.get_all_dimensions():
            if dim.name == name:
                return dim
        return None

    def get_dimensions_for_layer(self, layer: TaxonomyLayer) -> List[TaxonomyDimension]:
        """Get all dimensions for a specific layer."""
        if layer == TaxonomyLayer.MUNDANE:
            return self.mundane.dimensions
        elif layer == TaxonomyLayer.INSTITUTIONAL:
            return self.institutional.dimensions
        elif layer == TaxonomyLayer.AESTHETIC:
            return self.aesthetic.dimensions
        return []

    def get_domain_by_id(self, domain_id: str) -> Optional[ConceptualDomain]:
        """Look up a domain by its ID."""
        for domain in self.get_all_domains():
            if domain.id == domain_id:
                return domain
        return None

    def create_taxonomy_vector(self, positions: Dict[int, float]) -> np.ndarray:
        """
        Create an 80-dimensional taxonomy position vector.

        Args:
            positions: Dict mapping dimension IDs to position values (0.0-1.0)

        Returns:
            80-dimensional numpy array
        """
        vector = np.zeros(80)
        for dim_id, value in positions.items():
            if 1 <= dim_id <= 80:
                vector[dim_id - 1] = np.clip(value, 0.0, 1.0)
        return vector

    def describe_position(self, vector: np.ndarray) -> Dict[str, str]:
        """
        Generate human-readable description of a taxonomy position.

        Args:
            vector: 80-dimensional taxonomy position

        Returns:
            Dict with dimension names and their semantic interpretation
        """
        description = {}
        for dim in self.get_all_dimensions():
            value = vector[dim.id - 1]
            if value < 0.3:
                desc = f"Low ({dim.low_anchor})"
            elif value > 0.7:
                desc = f"High ({dim.high_anchor})"
            else:
                desc = f"Moderate"
            description[dim.name] = desc
        return description

    def compute_semantic_distance(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute semantic distance between two taxonomy positions.

        Args:
            vec1: First 80-dimensional position
            vec2: Second 80-dimensional position
            weights: Optional dimension weights

        Returns:
            Semantic distance (0.0 = identical, higher = more different)
        """
        if weights is None:
            weights = np.ones(80)

        diff = (vec1 - vec2) * weights
        return float(np.sqrt(np.sum(diff ** 2)))

    def get_layer_for_dimension(self, dim_id: int) -> Optional[TaxonomyLayer]:
        """Determine which layer a dimension belongs to."""
        if 1 <= dim_id <= 27:
            return TaxonomyLayer.MUNDANE
        elif 28 <= dim_id <= 54:
            return TaxonomyLayer.INSTITUTIONAL
        elif 55 <= dim_id <= 80:
            return TaxonomyLayer.AESTHETIC
        return None


# Pre-built archetypes positioned in the 80-dimensional taxonomy space
ARCHETYPE_TAXONOMY_POSITIONS = {
    "Hero": {
        # Mundane: Active, skilled, socially central
        13: 0.9,  # Productive_Engagement: High
        14: 0.8,  # Skill_Application: Expert
        22: 0.8,  # Social_Status: Central
        # Institutional: High status, protagonist of justice
        30: 0.7,  # Participation_Level: Participant
        38: 0.6,  # Justice_Orientation: Mixed
        # Aesthetic: Monumental, intense, prestigious
        57: 0.8,  # Scale: Monumental
        66: 0.8,  # Affective_Intensity: Intense
        76: 0.8,  # Status_Signal: Prestigious
    },
    "Rebel": {
        # Mundane: Marginal, autonomous, transgressive
        15: 0.9,  # Work_Autonomy: Self-directed
        22: 0.3,  # Social_Status: Marginal
        # Institutional: Challenges authority, heterodox
        28: 0.2,  # Authority_Structure: Decentralized preference
        42: 0.2,  # Orthodoxy: Heterodox
        # Aesthetic: Avant-garde, ironic, transgressive
        65: 0.7,  # Irony_Level: Ironic
        72: 0.9,  # Innovation_Degree: Avant-garde
        80: 0.2,  # Genre_Adherence: Transgressive
    },
    "Sage": {
        # Mundane: Learned, reflective, communicative
        26: 0.9,  # Learning_Mode: Discovering
        21: 0.7,  # Relationship_Depth: Deep
        # Institutional: Values education, universal knowledge
        44: 0.9,  # Knowledge_Access: Universal
        45: 0.9,  # Pedagogy_Mode: Critical
        # Aesthetic: Complex, meaningful, sincere
        56: 0.7,  # Formal_Complexity: Ornate
        62: 0.9,  # Meaning_Saturation: Symbolic
        65: 0.2,  # Irony_Level: Sincere
    },
    "Trickster": {
        # Mundane: Mobile, socially fluid, exchange-focused
        8: 0.9,   # Spatial_Range: Wide
        24: 0.8,  # Exchange_Activity: Active trader
        # Institutional: Liminal, challenges norms
        36: 0.3,  # Legal_Formality: Informal preference
        37: 0.3,  # Enforcement_Intensity: Evades enforcement
        # Aesthetic: Ironic, ambiguous, innovative
        64: 0.9,  # Ambiguity: Polysemous
        65: 0.9,  # Irony_Level: Ironic
        72: 0.7,  # Innovation_Degree: Novel
    },
    "Caregiver": {
        # Mundane: Nurturing, domestically engaged, social
        3: 0.9,   # Commensality: Communal
        12: 0.7,  # Domestic_Burden: Carries burden
        21: 0.9,  # Relationship_Depth: Intimate
        # Institutional: Family-focused, strong obligations
        48: 0.7,  # Kinship_Structure: Extended family
        51: 0.9,  # Intergenerational_Obligation: Strong
        # Aesthetic: Warm, authentic, humble
        67: 0.8,  # Valence_Tone: Warm
        73: 0.9,  # Authenticity: Genuine
        76: 0.3,  # Status_Signal: Humble
    },
}


# Pre-built institutional settings positioned in taxonomy space
INSTITUTION_TAXONOMY_POSITIONS = {
    "Courtroom": {
        # Mundane dimensions
        15: 0.1,  # Work_Autonomy: Highly directed (by judge/procedure)
        23: 0.9,  # Communication_Mode: Highly explicit (legal language)
        # Institutional dimensions
        28: 0.9,  # Authority_Structure: Highly centralized (judge)
        29: 0.9,  # Legitimacy_Basis: Rational-legal
        36: 0.95, # Legal_Formality: Highly codified
        37: 0.9,  # Enforcement_Intensity: Strict
        # Aesthetic dimensions
        56: 0.7,  # Formal_Complexity: Ornate (ceremony)
        57: 0.7,  # Scale: Monumental (grandeur)
        68: 0.8,  # Tension_Level: High (stakes)
    },
    "Marketplace": {
        # Mundane dimensions
        24: 0.9,  # Exchange_Activity: High exchange
        25: 0.6,  # Transaction_Complexity: Variable
        # Institutional dimensions
        32: 0.9,  # Market_Integration: High
        33: 0.8,  # Property_Rights: Private
        35: 0.9,  # Exchange_Mode: Transactional
        # Aesthetic dimensions
        55: 0.7,  # Chromatic_Intensity: Vibrant
        59: 0.8,  # Temporal_Pace: Fast
    },
    "Temple": {
        # Mundane dimensions
        19: 0.8,  # Social_Dimension: Collective
        17: 0.7,  # Leisure_Engagement: Absorbed (contemplation)
        # Institutional dimensions
        40: 0.95, # Sacred_Presence: Highly sacral
        41: 0.9,  # Ritual_Density: Elaborate
        42: 0.8,  # Orthodoxy: Orthodox
        43: 0.9,  # Transcendence_Orientation: Transcendent
        # Aesthetic dimensions
        57: 0.8,  # Scale: Monumental
        62: 0.9,  # Meaning_Saturation: Highly symbolic
        69: 0.9,  # Wonder_Evocation: Wondrous
    },
    "Hospital": {
        # Mundane dimensions
        5: 0.9,   # Health_Maintenance: Central focus
        # Institutional dimensions
        52: 0.7,  # Care_Access: Aspires to universal
        53: 0.8,  # Medical_Authority: Professional monopoly
        54: 0.4,  # Body_Autonomy: Some paternalism
        # Aesthetic dimensions
        55: 0.2,  # Chromatic_Intensity: Muted (sterile)
        56: 0.3,  # Formal_Complexity: Simple/functional
        68: 0.6,  # Tension_Level: Moderate anxiety
    },
    "School": {
        # Mundane dimensions
        26: 0.8,  # Learning_Mode: Mixed receiving/discovering
        20: 0.7,  # Social_Frequency: Connected
        # Institutional dimensions
        44: 0.8,  # Knowledge_Access: Broadly accessible
        45: 0.5,  # Pedagogy_Mode: Mixed
        46: 0.7,  # Credential_Importance: Formal credentials
        47: 0.6,  # Canon_Rigidity: Structured curriculum
        # Aesthetic dimensions
        56: 0.4,  # Formal_Complexity: Moderate
        60: 0.8,  # Rhythmic_Pattern: Regular (schedule)
    },
}
