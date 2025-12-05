"""
Soul Map Ontology: Complete 181-dimensional psychological grounding for ToM-NAS

Scientific Foundation:
- 9 layers mapping human psychological constitution
- Each dimension grounded in psychological literature
- Provides semantic space for all beliefs and mental states
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field


@dataclass
class OntologyDimension:
    """Single dimension in the ontology space"""
    name: str
    layer: int
    index: int
    min_val: float = 0.0
    max_val: float = 1.0
    default: float = 0.5
    description: str = ""
    source: str = ""  # Literature reference


class SoulMapOntology:
    """
    Complete 181-dimensional ontology mapping human psychological constitution.

    LAYERS:
    0. Biological (15 dims) - Embodied grounding
    1. Affective (24 dims) - Emotions and feelings
    2. Motivational (30 dims) - Goals and drives
    3. Cognitive (24 dims) - Reasoning and processing
    4. Self (21 dims) - Identity and self-concept
    5. Social Cognition (25 dims) - ToM capabilities
    6. Values/Beliefs (18 dims) - Moral foundations
    7. Contextual (12 dims) - Cultural and situational
    8. Existential (12 dims) - Meaning and mortality

    TOTAL: 181 dimensions
    """

    def __init__(self):
        self.dimensions: List[OntologyDimension] = []
        self.name_to_idx: Dict[str, int] = {}
        self.layer_ranges: Dict[int, Tuple[int, int]] = {}
        self.total_dims = 181

        # Build all 9 layers
        self._build_all_layers()

        # Validate
        assert len(self.dimensions) == self.total_dims, \
            f"Expected {self.total_dims} dims, got {len(self.dimensions)}"

    def _build_all_layers(self):
        """Build complete ontology structure"""
        self._build_layer_0_biological()
        self._build_layer_1_affective()
        self._build_layer_2_motivational()
        self._build_layer_3_cognitive()
        self._build_layer_4_self()
        self._build_layer_5_social_cognition()
        self._build_layer_6_values_beliefs()
        self._build_layer_7_contextual()
        self._build_layer_8_existential()

    def _add_dimension(self, name: str, layer: int, description: str = "",
                       source: str = "", default: float = 0.5):
        """Add a single dimension to the ontology"""
        idx = len(self.dimensions)
        dim = OntologyDimension(
            name=name,
            layer=layer,
            index=idx,
            default=default,
            description=description,
            source=source
        )
        self.dimensions.append(dim)
        self.name_to_idx[name] = idx

    def _build_layer_0_biological(self):
        """
        Layer 0: Biological (15 dimensions)
        Grounds cognition in embodied experience.
        """
        start_idx = len(self.dimensions)

        # Exteroceptive senses (5)
        self._add_dimension("bio.extero_vision", 0, "Visual perception acuity")
        self._add_dimension("bio.extero_audition", 0, "Auditory perception")
        self._add_dimension("bio.extero_touch", 0, "Tactile sensation")
        self._add_dimension("bio.extero_olfaction", 0, "Smell perception")
        self._add_dimension("bio.extero_gustation", 0, "Taste perception")

        # Interoceptive states (5)
        self._add_dimension("bio.intero_hunger", 0, "Hunger level", default=0.3)
        self._add_dimension("bio.intero_thirst", 0, "Thirst level", default=0.3)
        self._add_dimension("bio.intero_pain", 0, "Pain sensation", default=0.1)
        self._add_dimension("bio.intero_temperature", 0, "Temperature comfort", default=0.5)
        self._add_dimension("bio.intero_arousal", 0, "Physiological arousal", default=0.4)

        # Body state (5)
        self._add_dimension("bio.proprioception", 0, "Body awareness")
        self._add_dimension("bio.energy_level", 0, "Available energy", default=0.7)
        self._add_dimension("bio.fatigue", 0, "Fatigue level", default=0.2)
        self._add_dimension("bio.homeostatic_balance", 0, "Overall balance", default=0.6)
        self._add_dimension("bio.circadian_phase", 0, "Sleep-wake cycle")

        self.layer_ranges[0] = (start_idx, len(self.dimensions) - 1)

    def _build_layer_1_affective(self):
        """
        Layer 1: Affective (24 dimensions)
        Emotions critical for social reasoning.
        """
        start_idx = len(self.dimensions)

        # Primary emotions (Ekman)
        self._add_dimension("affect.joy", 1, "Happiness, pleasure", default=0.4)
        self._add_dimension("affect.sadness", 1, "Grief, sorrow", default=0.2)
        self._add_dimension("affect.fear", 1, "Fear response", default=0.2)
        self._add_dimension("affect.anger", 1, "Anger, frustration", default=0.2)
        self._add_dimension("affect.disgust", 1, "Disgust response", default=0.1)
        self._add_dimension("affect.surprise", 1, "Surprise reaction", default=0.3)

        # Social emotions
        self._add_dimension("affect.shame", 1, "Shame feeling", default=0.1)
        self._add_dimension("affect.guilt", 1, "Guilt feeling", default=0.1)
        self._add_dimension("affect.pride", 1, "Pride in accomplishment", default=0.3)
        self._add_dimension("affect.envy", 1, "Envy of others", default=0.1)
        self._add_dimension("affect.gratitude", 1, "Feeling grateful", default=0.4)
        self._add_dimension("affect.schadenfreude", 1, "Pleasure at others' misfortune", default=0.05)

        # Anticipatory emotions
        self._add_dimension("affect.hope", 1, "Hopeful anticipation", default=0.5)
        self._add_dimension("affect.anxiety", 1, "Anxious anticipation", default=0.2)
        self._add_dimension("affect.dread", 1, "Dread of future", default=0.1)
        self._add_dimension("affect.anticipation", 1, "General anticipation", default=0.4)

        # Dimensional model (Russell's circumplex)
        self._add_dimension("affect.valence", 1, "Positive-negative continuum", default=0.5)
        self._add_dimension("affect.arousal_level", 1, "High-low activation", default=0.4)
        self._add_dimension("affect.dominance", 1, "Control-submission", default=0.5)

        # Meta-emotional
        self._add_dimension("affect.emotion_regulation", 1, "Ability to regulate emotions", default=0.5)
        self._add_dimension("affect.alexithymia", 1, "Difficulty identifying emotions", default=0.2)
        self._add_dimension("affect.empathic_distress", 1, "Distress from others' suffering", default=0.3)
        self._add_dimension("affect.compassion", 1, "Compassionate feeling", default=0.5)
        self._add_dimension("affect.trust", 1, "General trust tendency", default=0.5)

        self.layer_ranges[1] = (start_idx, len(self.dimensions) - 1)

    def _build_layer_2_motivational(self):
        """
        Layer 2: Motivational (30 dimensions)
        Goals drive decisions. ToM requires modeling others' goals.
        """
        start_idx = len(self.dimensions)

        # Maslow's hierarchy
        self._add_dimension("motiv.physiological_need", 2, "Basic survival needs", default=0.3)
        self._add_dimension("motiv.safety_need", 2, "Security and stability", default=0.4)
        self._add_dimension("motiv.belonging_need", 2, "Social connection", default=0.5)
        self._add_dimension("motiv.esteem_need", 2, "Recognition and respect", default=0.4)
        self._add_dimension("motiv.self_actualization", 2, "Reaching potential", default=0.3)

        # Self-determination theory (Deci & Ryan)
        self._add_dimension("motiv.autonomy_need", 2, "Need for independence", default=0.5)
        self._add_dimension("motiv.competence_need", 2, "Need for mastery", default=0.5)
        self._add_dimension("motiv.relatedness_need", 2, "Need for connection", default=0.5)

        # Drives
        self._add_dimension("motiv.curiosity_drive", 2, "Information seeking", default=0.5)
        self._add_dimension("motiv.status_seeking", 2, "Desire for status", default=0.4)
        self._add_dimension("motiv.power_seeking", 2, "Desire for power", default=0.3)
        self._add_dimension("motiv.affiliation_seeking", 2, "Desire for friendship", default=0.5)
        self._add_dimension("motiv.achievement_drive", 2, "Achievement motivation", default=0.5)
        self._add_dimension("motiv.security_drive", 2, "Risk avoidance", default=0.4)
        self._add_dimension("motiv.novelty_seeking", 2, "Desire for new experiences", default=0.4)

        # Temporal orientation
        self._add_dimension("motiv.present_focus", 2, "Present-moment orientation", default=0.5)
        self._add_dimension("motiv.future_focus", 2, "Future-oriented planning", default=0.5)
        self._add_dimension("motiv.delayed_gratification", 2, "Ability to delay reward", default=0.5)

        # Incentive sensitivity
        self._add_dimension("motiv.reward_sensitivity", 2, "Response to rewards", default=0.5)
        self._add_dimension("motiv.punishment_sensitivity", 2, "Response to punishment", default=0.5)
        self._add_dimension("motiv.loss_aversion", 2, "Avoiding losses", default=0.5)

        # Goal-related
        self._add_dimension("motiv.goal_clarity", 2, "Clarity of goals", default=0.5)
        self._add_dimension("motiv.goal_commitment", 2, "Commitment to goals", default=0.5)
        self._add_dimension("motiv.intrinsic_motivation", 2, "Internal motivation", default=0.5)
        self._add_dimension("motiv.extrinsic_motivation", 2, "External motivation", default=0.4)

        # Social motivation
        self._add_dimension("motiv.prosocial_motivation", 2, "Helping others", default=0.5)
        self._add_dimension("motiv.competitive_motivation", 2, "Desire to compete", default=0.4)
        self._add_dimension("motiv.cooperative_motivation", 2, "Desire to cooperate", default=0.5)
        self._add_dimension("motiv.revenge_motivation", 2, "Desire for revenge", default=0.1)
        self._add_dimension("motiv.fairness_motivation", 2, "Desire for fairness", default=0.6)

        self.layer_ranges[2] = (start_idx, len(self.dimensions) - 1)

    def _build_layer_3_cognitive(self):
        """
        Layer 3: Cognitive/Epistemic (24 dimensions)
        Model computational constraints. ToM respects these in self and others.
        """
        start_idx = len(self.dimensions)

        # Cognitive capacity
        self._add_dimension("cog.working_memory_capacity", 3, "Working memory span", default=0.5)
        self._add_dimension("cog.attention_span", 3, "Sustained attention ability", default=0.5)
        self._add_dimension("cog.processing_speed", 3, "Mental processing speed", default=0.5)
        self._add_dimension("cog.cognitive_flexibility", 3, "Task switching ability", default=0.5)
        self._add_dimension("cog.inhibitory_control", 3, "Response inhibition", default=0.5)

        # Memory systems
        self._add_dimension("cog.episodic_memory", 3, "Event memory strength", default=0.5)
        self._add_dimension("cog.semantic_memory", 3, "Factual memory strength", default=0.5)
        self._add_dimension("cog.procedural_memory", 3, "Skill memory strength", default=0.5)
        self._add_dimension("cog.prospective_memory", 3, "Future intention memory", default=0.5)

        # Reasoning abilities
        self._add_dimension("cog.causal_reasoning", 3, "Understanding causation", default=0.5)
        self._add_dimension("cog.counterfactual_reasoning", 3, "What-if thinking", default=0.5)
        self._add_dimension("cog.analogical_reasoning", 3, "Analogy making", default=0.5)
        self._add_dimension("cog.abstract_reasoning", 3, "Abstract thought", default=0.5)
        self._add_dimension("cog.logical_reasoning", 3, "Logical inference", default=0.5)
        self._add_dimension("cog.probabilistic_reasoning", 3, "Probability estimation", default=0.5)

        # Metacognition
        self._add_dimension("cog.metacognitive_accuracy", 3, "Knowing what you know", default=0.5)
        self._add_dimension("cog.confidence_calibration", 3, "Appropriate confidence", default=0.5)
        self._add_dimension("cog.theory_of_mind_depth", 3, "ToM recursion capacity", default=0.5)

        # Epistemic states
        self._add_dimension("cog.uncertainty_tolerance", 3, "Tolerance for uncertainty", default=0.5)
        self._add_dimension("cog.need_for_closure", 3, "Need for definite answers", default=0.4)
        self._add_dimension("cog.cognitive_complexity", 3, "Tolerance for complexity", default=0.5)
        self._add_dimension("cog.naive_realism", 3, "Belief that perception=reality", default=0.4)
        self._add_dimension("cog.belief_updating_speed", 3, "How fast beliefs change", default=0.5)
        self._add_dimension("cog.confirmation_bias", 3, "Seeking confirming evidence", default=0.4)

        self.layer_ranges[3] = (start_idx, len(self.dimensions) - 1)

    def _build_layer_4_self(self):
        """
        Layer 4: Self (21 dimensions)
        Self/other distinction is fundamental to ToM.
        """
        start_idx = len(self.dimensions)

        # Self-concept
        self._add_dimension("self.physical_self_concept", 4, "Body image", default=0.5)
        self._add_dimension("self.social_self_concept", 4, "Social identity", default=0.5)
        self._add_dimension("self.personal_self_concept", 4, "Personal identity", default=0.5)
        self._add_dimension("self.aspirational_self", 4, "Ideal self image", default=0.6)
        self._add_dimension("self.actual_self", 4, "Current self perception", default=0.5)

        # Self-evaluation
        self._add_dimension("self.self_esteem", 4, "Self-worth", default=0.5)
        self._add_dimension("self.self_efficacy", 4, "Belief in own abilities", default=0.5)
        self._add_dimension("self.self_consistency", 4, "Consistent self-view", default=0.6)
        self._add_dimension("self.authenticity", 4, "Being true to self", default=0.5)

        # Identity
        self._add_dimension("self.identity_clarity", 4, "Clear sense of who you are", default=0.5)
        self._add_dimension("self.identity_stability", 4, "Stable identity over time", default=0.6)
        self._add_dimension("self.identity_commitment", 4, "Commitment to identity", default=0.5)
        self._add_dimension("self.identity_exploration", 4, "Exploring identity options", default=0.4)

        # Agency
        self._add_dimension("self.perceived_agency", 4, "Sense of control over actions", default=0.6)
        self._add_dimension("self.locus_of_control_internal", 4, "Internal locus of control", default=0.5)
        self._add_dimension("self.learned_helplessness", 4, "Feeling unable to affect outcomes", default=0.2)
        self._add_dimension("self.self_determination", 4, "Autonomous decision making", default=0.5)

        # Self-awareness
        self._add_dimension("self.private_self_awareness", 4, "Awareness of internal states", default=0.5)
        self._add_dimension("self.public_self_awareness", 4, "Awareness of how others see you", default=0.5)
        self._add_dimension("self.self_other_boundary", 4, "Clear self-other distinction", default=0.7)
        self._add_dimension("self.narrative_identity", 4, "Coherent life story", default=0.5)

        self.layer_ranges[4] = (start_idx, len(self.dimensions) - 1)

    def _build_layer_5_social_cognition(self):
        """
        Layer 5: Social Cognition (25 dimensions)
        Direct ToM capabilities - the phenomenon we're trying to evolve.
        """
        start_idx = len(self.dimensions)

        # ToM components
        self._add_dimension("social.perspective_taking", 5, "Seeing from others' viewpoint", default=0.5)
        self._add_dimension("social.recursive_depth", 5, "Levels of nested belief", default=0.3)
        self._add_dimension("social.belief_attribution", 5, "Attributing beliefs to others", default=0.5)
        self._add_dimension("social.desire_attribution", 5, "Attributing desires to others", default=0.5)
        self._add_dimension("social.intention_attribution", 5, "Attributing intentions to others", default=0.5)
        self._add_dimension("social.knowledge_attribution", 5, "Tracking what others know", default=0.5)

        # Empathy
        self._add_dimension("social.cognitive_empathy", 5, "Understanding others' thoughts", default=0.5)
        self._add_dimension("social.affective_empathy", 5, "Feeling others' emotions", default=0.5)
        self._add_dimension("social.empathic_accuracy", 5, "Accuracy in reading others", default=0.4)
        self._add_dimension("social.empathic_concern", 5, "Concern for others' welfare", default=0.5)

        # Social perception
        self._add_dimension("social.emotion_recognition", 5, "Reading others' emotions", default=0.5)
        self._add_dimension("social.deception_detection", 5, "Detecting lies", default=0.4)
        self._add_dimension("social.social_cue_reading", 5, "Reading social signals", default=0.5)
        self._add_dimension("social.trustworthiness_judgment", 5, "Judging who to trust", default=0.5)

        # Relationships
        self._add_dimension("social.attachment_security", 5, "Secure attachment style", default=0.5)
        self._add_dimension("social.attachment_avoidance", 5, "Avoidant attachment", default=0.3)
        self._add_dimension("social.attachment_anxiety", 5, "Anxious attachment", default=0.3)
        self._add_dimension("social.social_dominance", 5, "Tendency to dominate", default=0.4)
        self._add_dimension("social.social_submission", 5, "Tendency to submit", default=0.3)

        # Reciprocity
        self._add_dimension("social.reciprocity_norm", 5, "Belief in reciprocity", default=0.6)
        self._add_dimension("social.fairness_sensitivity", 5, "Sensitivity to fairness", default=0.6)
        self._add_dimension("social.equity_sensitivity", 5, "Sensitivity to equal treatment", default=0.5)
        self._add_dimension("social.coalition_detection", 5, "Detecting group membership", default=0.5)
        self._add_dimension("social.mentalizing_tendency", 5, "Tendency to think about minds", default=0.5)
        self._add_dimension("social.anthropomorphism", 5, "Attributing minds to non-agents", default=0.3)

        self.layer_ranges[5] = (start_idx, len(self.dimensions) - 1)

    def _build_layer_6_values_beliefs(self):
        """
        Layer 6: Values/Beliefs (18 dimensions)
        Values drive decisions. ToM requires modeling others' values.
        """
        start_idx = len(self.dimensions)

        # Moral foundations (Haidt)
        self._add_dimension("values.care_harm", 6, "Caring for others, avoiding harm", default=0.6)
        self._add_dimension("values.fairness_cheating", 6, "Justice and proportionality", default=0.6)
        self._add_dimension("values.loyalty_betrayal", 6, "Group loyalty", default=0.5)
        self._add_dimension("values.authority_subversion", 6, "Respect for authority", default=0.4)
        self._add_dimension("values.sanctity_degradation", 6, "Purity concerns", default=0.4)
        self._add_dimension("values.liberty_oppression", 6, "Freedom from tyranny", default=0.5)

        # Epistemic values
        self._add_dimension("values.truth_seeking", 6, "Valuing truth", default=0.6)
        self._add_dimension("values.intellectual_humility", 6, "Openness to being wrong", default=0.5)
        self._add_dimension("values.dogmatism", 6, "Rigid beliefs", default=0.3)
        self._add_dimension("values.conspiracy_thinking", 6, "Tendency for conspiracies", default=0.2)

        # Ideology
        self._add_dimension("values.tradition_vs_change", 6, "Traditional vs progressive", default=0.5)
        self._add_dimension("values.hierarchy_vs_equality", 6, "Hierarchical vs egalitarian", default=0.5)
        self._add_dimension("values.individualism_collectivism", 6, "Individual vs group focus", default=0.5)

        # Commitment
        self._add_dimension("values.value_consistency", 6, "Acting on values", default=0.5)
        self._add_dimension("values.moral_identity", 6, "Morality central to identity", default=0.5)
        self._add_dimension("values.sacred_values", 6, "Values not tradeable", default=0.4)
        self._add_dimension("values.utilitarian_tendency", 6, "Maximizing outcomes", default=0.4)
        self._add_dimension("values.deontological_tendency", 6, "Following rules", default=0.5)

        self.layer_ranges[6] = (start_idx, len(self.dimensions) - 1)

    def _build_layer_7_contextual(self):
        """
        Layer 7: Contextual (12 dimensions)
        Context matters. Same action means different things in different contexts.
        """
        start_idx = len(self.dimensions)

        # Demographics
        self._add_dimension("context.life_stage", 7, "Developmental stage", default=0.5)
        self._add_dimension("context.socioeconomic_status", 7, "Resource level", default=0.5)
        self._add_dimension("context.education_level", 7, "Education attainment", default=0.5)

        # Culture
        self._add_dimension("context.cultural_tightness", 7, "Strict vs loose norms", default=0.5)
        self._add_dimension("context.power_distance", 7, "Acceptance of hierarchy", default=0.4)
        self._add_dimension("context.uncertainty_avoidance", 7, "Preference for structure", default=0.5)

        # Environment
        self._add_dimension("context.resource_scarcity", 7, "Environmental scarcity", default=0.3)
        self._add_dimension("context.social_density", 7, "Social crowding", default=0.5)
        self._add_dimension("context.environmental_threat", 7, "External threats", default=0.2)
        self._add_dimension("context.social_support", 7, "Available support", default=0.5)

        # History
        self._add_dimension("context.trauma_exposure", 7, "Past trauma", default=0.2)
        self._add_dimension("context.cumulative_adversity", 7, "Life hardships", default=0.3)

        self.layer_ranges[7] = (start_idx, len(self.dimensions) - 1)

    def _build_layer_8_existential(self):
        """
        Layer 8: Existential (12 dimensions)
        Highest-level concerns. Relevant for mortality and cultural evolution.
        """
        start_idx = len(self.dimensions)

        # Mortality
        self._add_dimension("exist.mortality_salience", 8, "Awareness of death", default=0.3)
        self._add_dimension("exist.death_anxiety", 8, "Fear of death", default=0.3)
        self._add_dimension("exist.afterlife_belief", 8, "Belief in afterlife", default=0.4)

        # Meaning
        self._add_dimension("exist.meaning_in_life", 8, "Perceived life meaning", default=0.5)
        self._add_dimension("exist.purpose_clarity", 8, "Clear life purpose", default=0.5)
        self._add_dimension("exist.coherence_sense", 8, "Life makes sense", default=0.5)
        self._add_dimension("exist.significance_feeling", 8, "Feeling significant", default=0.5)

        # Freedom
        self._add_dimension("exist.existential_anxiety", 8, "Anxiety about existence", default=0.3)
        self._add_dimension("exist.freedom_responsibility", 8, "Acceptance of freedom", default=0.5)
        self._add_dimension("exist.authenticity_seeking", 8, "Seeking authentic life", default=0.5)

        # Legacy
        self._add_dimension("exist.legacy_motivation", 8, "Desire to leave legacy", default=0.4)
        self._add_dimension("exist.generativity", 8, "Concern for future generations", default=0.4)

        self.layer_ranges[8] = (start_idx, len(self.dimensions) - 1)

    # === INTERFACE METHODS ===

    def get_index(self, name: str) -> int:
        """Get index for dimension by name"""
        return self.name_to_idx.get(name, -1)

    def get_dimension(self, idx: int) -> Optional[OntologyDimension]:
        """Get dimension object by index"""
        if 0 <= idx < len(self.dimensions):
            return self.dimensions[idx]
        return None

    def get_dimension_name(self, idx: int) -> str:
        """Get dimension name by index"""
        dim = self.get_dimension(idx)
        return dim.name if dim else f"unknown_{idx}"

    def get_layer_for_dimension(self, idx: int) -> int:
        """Get layer number for a dimension index"""
        dim = self.get_dimension(idx)
        return dim.layer if dim else -1

    def get_layer_dimensions(self, layer: int) -> List[OntologyDimension]:
        """Get all dimensions in a layer"""
        return [d for d in self.dimensions if d.layer == layer]

    def encode(self, state_dict: Dict[str, float]) -> torch.Tensor:
        """
        Encode a dictionary of named values to ontology vector

        Args:
            state_dict: Dictionary mapping dimension names to values

        Returns:
            Tensor of shape (total_dims,) with encoded values
        """
        vector = self.get_default_state()
        for name, value in state_dict.items():
            if name in self.name_to_idx:
                idx = self.name_to_idx[name]
                vector[idx] = torch.tensor(value).float().clamp(0, 1)
        return vector

    def decode(self, vector: torch.Tensor) -> Dict[str, float]:
        """
        Decode ontology vector to named dictionary

        Args:
            vector: Tensor of shape (total_dims,)

        Returns:
            Dictionary mapping dimension names to values
        """
        result = {}
        for dim in self.dimensions:
            result[dim.name] = vector[dim.index].item()
        return result

    def decode_to_interpretable(self, vector: torch.Tensor, top_k: int = 10) -> Dict:
        """
        Decode vector to human-readable interpretation

        Args:
            vector: Ontology vector
            top_k: Number of top activated dimensions to return

        Returns:
            Dictionary with interpretable breakdown
        """
        vec_np = vector.detach().cpu().numpy()

        # Get top activated dimensions
        top_indices = np.argsort(np.abs(vec_np - 0.5))[::-1][:top_k]

        interpretation = {
            "top_active_dimensions": [],
            "layer_summary": {},
            "overall_valence": 0.0
        }

        for idx in top_indices:
            dim = self.dimensions[idx]
            interpretation["top_active_dimensions"].append({
                "name": dim.name,
                "value": float(vec_np[idx]),
                "deviation": float(abs(vec_np[idx] - dim.default)),
                "layer": dim.layer
            })

        # Layer summary
        for layer in range(9):
            if layer in self.layer_ranges:
                start, end = self.layer_ranges[layer]
                layer_vals = vec_np[start:end+1]
                interpretation["layer_summary"][layer] = {
                    "mean": float(np.mean(layer_vals)),
                    "std": float(np.std(layer_vals)),
                    "max": float(np.max(layer_vals)),
                    "min": float(np.min(layer_vals))
                }

        # Overall valence (from affective layer)
        if "affect.valence" in self.name_to_idx:
            interpretation["overall_valence"] = vec_np[self.name_to_idx["affect.valence"]]

        return interpretation

    def get_default_state(self) -> torch.Tensor:
        """Get default ontology state vector"""
        defaults = torch.zeros(self.total_dims)
        for dim in self.dimensions:
            defaults[dim.index] = dim.default
        return defaults

    def similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute semantic similarity between two ontology vectors

        Uses cosine similarity in ontology space.
        """
        vec1_flat = vec1.flatten().float()
        vec2_flat = vec2.flatten().float()

        dot_product = torch.dot(vec1_flat, vec2_flat)
        norm1 = torch.norm(vec1_flat)
        norm2 = torch.norm(vec2_flat)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return (dot_product / (norm1 * norm2)).item()

    def prior_distribution(self, context: Optional[Dict] = None) -> torch.Tensor:
        """
        Get prior distribution over ontology space given context

        Args:
            context: Optional context dictionary (not used in basic implementation)

        Returns:
            Mean of prior distribution (default state)
        """
        return self.get_default_state()

    def perturbation(self, vec: torch.Tensor, mutation_rate: float = 0.1) -> torch.Tensor:
        """
        Apply ontology-respecting perturbation for evolution

        Mutations stay within plausible psychological space.
        """
        perturbed = vec.clone()

        # Add Gaussian noise
        noise = torch.randn_like(vec) * mutation_rate
        perturbed = perturbed + noise

        # Clamp to valid range
        perturbed = torch.clamp(perturbed, 0.0, 1.0)

        return perturbed

    def interpolate(self, vec1: torch.Tensor, vec2: torch.Tensor,
                    alpha: float = 0.5) -> torch.Tensor:
        """Interpolate between two ontology states"""
        return alpha * vec1 + (1 - alpha) * vec2


class OntologyEncoder(nn.Module):
    """Neural network encoder for mapping observations to ontology space"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.ontology = SoulMapOntology()
        self.output_dim = self.ontology.total_dims

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid()  # Ensure [0, 1] output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to ontology space"""
        return self.encoder(x)

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to ontology vector"""
        return self.forward(obs)


class OntologyDecoder(nn.Module):
    """Neural network decoder for mapping ontology states to actions"""

    def __init__(self, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.ontology = SoulMapOntology()
        self.input_dim = self.ontology.total_dims

        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, ontology_state: torch.Tensor) -> torch.Tensor:
        """Decode ontology state to output"""
        return self.decoder(ontology_state)
