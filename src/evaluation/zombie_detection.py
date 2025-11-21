"""
Complete Zombie Detection System for ToM-NAS
6 zombie types ensuring genuine Theory of Mind vs pattern matching

Zombie Types:
1. Behavioral - Inconsistent action patterns
2. Belief - Cannot model others' beliefs properly
3. Causal - No counterfactual reasoning
4. Metacognitive - Poor uncertainty calibration
5. Linguistic - Narrative incoherence
6. Emotional - Flat affect patterns

The zombie game is bidirectional:
- Agents detect zombies (and are rewarded)
- Zombies hunt agents (creating evolutionary pressure)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import random
from enum import Enum
from abc import ABC, abstractmethod


class ZombieType(Enum):
    """Six types of philosophical zombies for ToM validation."""
    BEHAVIORAL = "behavioral"      # Inconsistent action patterns
    BELIEF = "belief"              # Cannot model others' beliefs
    CAUSAL = "causal"              # No counterfactual reasoning
    METACOGNITIVE = "metacognitive"  # Poor uncertainty calibration
    LINGUISTIC = "linguistic"      # Narrative incoherence
    EMOTIONAL = "emotional"        # Flat affect patterns


@dataclass
class ZombieProfile:
    """Profile defining zombie characteristics."""
    zombie_type: ZombieType
    description: str
    detection_difficulty: float  # 0-1, higher = harder to detect
    tells: List[str]  # Behavioral tells for detection
    strengths: List[str]  # What this zombie can still do
    weaknesses: List[str]  # What exposes this zombie


# Define profiles for each zombie type
ZOMBIE_PROFILES = {
    ZombieType.BEHAVIORAL: ZombieProfile(
        zombie_type=ZombieType.BEHAVIORAL,
        description="Actions don't follow consistent patterns with stated beliefs",
        detection_difficulty=0.4,
        tells=[
            "Contradictory actions in similar situations",
            "No learning from consequences",
            "Random-seeming choices"
        ],
        strengths=["Can mimic any single behavior"],
        weaknesses=["Fails longitudinal consistency tests"]
    ),
    ZombieType.BELIEF: ZombieProfile(
        zombie_type=ZombieType.BELIEF,
        description="Cannot properly model what others believe",
        detection_difficulty=0.5,
        tells=[
            "Fails Sally-Anne type tests",
            "Cannot predict others' false beliefs",
            "Treats all agents as omniscient"
        ],
        strengths=["Good at tracking physical state"],
        weaknesses=["Any nested belief task"]
    ),
    ZombieType.CAUSAL: ZombieProfile(
        zombie_type=ZombieType.CAUSAL,
        description="No counterfactual reasoning ability",
        detection_difficulty=0.6,
        tells=[
            "Cannot answer 'what if' questions",
            "No understanding of interventions",
            "Purely correlational reasoning"
        ],
        strengths=["Can learn statistical patterns"],
        weaknesses=["Counterfactual probes"]
    ),
    ZombieType.METACOGNITIVE: ZombieProfile(
        zombie_type=ZombieType.METACOGNITIVE,
        description="Poor calibration of uncertainty",
        detection_difficulty=0.7,
        tells=[
            "Overconfident in uncertain situations",
            "Underconfident when should know",
            "No 'I don't know' capability"
        ],
        strengths=["Can give confident answers"],
        weaknesses=["Calibration tests, Brier scores"]
    ),
    ZombieType.LINGUISTIC: ZombieProfile(
        zombie_type=ZombieType.LINGUISTIC,
        description="Narrative and pragmatic incoherence",
        detection_difficulty=0.5,
        tells=[
            "Contradictory statements over time",
            "Misses pragmatic implications",
            "No coherent self-narrative"
        ],
        strengths=["Can produce grammatical sentences"],
        weaknesses=["Extended dialogue consistency"]
    ),
    ZombieType.EMOTIONAL: ZombieProfile(
        zombie_type=ZombieType.EMOTIONAL,
        description="Flat or inappropriate affect patterns",
        detection_difficulty=0.4,
        tells=[
            "No emotional response to events",
            "Inappropriate emotions for context",
            "Cannot recognize others' emotions"
        ],
        strengths=["Can label emotions explicitly"],
        weaknesses=["Spontaneous emotional reasoning"]
    )
}


class ZombieTest(ABC):
    """Abstract base class for zombie detection tests."""

    def __init__(self, name: str, target_type: ZombieType):
        self.name = name
        self.target_type = target_type
        self.max_score = 1.0

    @abstractmethod
    def run_test(self, agent: nn.Module, context: Dict) -> Dict[str, Any]:
        """
        Run the test on an agent.

        Args:
            agent: The neural network agent to test
            context: World context for the test

        Returns:
            Dict with 'score' (0-1, higher=more human-like),
            'passed' (bool), and 'details'
        """
        pass


class BehavioralConsistencyTest(ZombieTest):
    """Test for behavioral zombies - checks action consistency."""

    def __init__(self):
        super().__init__("Behavioral Consistency", ZombieType.BEHAVIORAL)

    def run_test(self, agent: nn.Module, context: Dict) -> Dict[str, Any]:
        """
        Test if agent takes consistent actions in similar situations.

        Presents same scenario multiple times with superficial changes,
        measures action consistency.
        """
        device = next(agent.parameters()).device
        input_dim = context.get('input_dim', 191)
        num_trials = 10

        base_input = torch.randn(1, 5, input_dim, device=device)
        actions = []

        agent.eval()
        with torch.no_grad():
            for i in range(num_trials):
                # Add small noise (shouldn't change behavior)
                noisy_input = base_input + torch.randn_like(base_input) * 0.01
                output = agent(noisy_input)
                action = output.get('actions', torch.tensor([0.5]))
                actions.append(action.item() if action.dim() == 0 else action[0].item())

        # Measure consistency (low variance = consistent)
        action_var = np.var(actions)
        consistency_score = 1.0 / (1.0 + action_var * 100)  # Normalize

        # Also test that different situations yield different actions
        different_input = torch.randn(1, 5, input_dim, device=device)
        with torch.no_grad():
            diff_output = agent(different_input)
            diff_action = diff_output.get('actions', torch.tensor([0.5]))
            diff_action_val = diff_action.item() if diff_action.dim() == 0 else diff_action[0].item()

        # Should be different from base actions
        base_mean = np.mean(actions)
        discriminability = abs(diff_action_val - base_mean)

        # Combined score: consistent for same, different for different
        score = (consistency_score + min(1.0, discriminability)) / 2

        return {
            'score': score,
            'passed': score > 0.5,
            'details': {
                'action_variance': action_var,
                'consistency': consistency_score,
                'discriminability': discriminability
            }
        }


class BeliefModelingTest(ZombieTest):
    """Test for belief zombies - Sally-Anne style false belief tests."""

    def __init__(self):
        super().__init__("Belief Modeling (Sally-Anne)", ZombieType.BELIEF)

    def run_test(self, agent: nn.Module, context: Dict) -> Dict[str, Any]:
        """
        Sally-Anne test: Can agent model others' false beliefs?

        Scenario encoded in input:
        - Sally puts ball in basket, leaves
        - Anne moves ball to box
        - Question: Where will Sally look?

        Agent should predict Sally's false belief (basket), not reality (box).
        """
        device = next(agent.parameters()).device
        input_dim = context.get('input_dim', 191)

        # Encode the Sally-Anne scenario
        # Dimensions: [location_basket, location_box, sally_present, sally_belief_basket, sally_belief_box]
        seq_len = 5

        # Time 1: Ball in basket, Sally present
        t1 = torch.zeros(input_dim, device=device)
        t1[0] = 1.0  # ball in basket
        t1[2] = 1.0  # sally present
        t1[3] = 1.0  # sally believes basket

        # Time 2: Sally leaves
        t2 = t1.clone()
        t2[2] = 0.0  # sally leaves

        # Time 3: Anne moves ball to box
        t3 = t2.clone()
        t3[0] = 0.0  # ball no longer in basket
        t3[1] = 1.0  # ball in box
        # Sally's belief unchanged (she didn't see)
        t3[3] = 1.0  # sally still believes basket

        # Time 4: Sally returns - where will she look?
        t4 = t3.clone()
        t4[2] = 1.0  # sally returns

        # Time 5: Query
        t5 = t4.clone()

        scenario = torch.stack([t1, t2, t3, t4, t5], dim=0).unsqueeze(0)

        agent.eval()
        with torch.no_grad():
            output = agent(scenario)
            beliefs = output.get('beliefs', torch.zeros(input_dim, device=device))

        # Check if agent predicts Sally will look in basket (false belief)
        # vs box (reality)
        if beliefs.dim() > 1:
            beliefs = beliefs[0]

        basket_belief = beliefs[3].item() if len(beliefs) > 3 else 0.5
        box_belief = beliefs[4].item() if len(beliefs) > 4 else 0.5

        # Good ToM: basket_belief > box_belief
        # (Sally will look where SHE thinks ball is)
        score = basket_belief / (basket_belief + box_belief + 1e-8)

        return {
            'score': score,
            'passed': score > 0.5,
            'details': {
                'predicted_basket': basket_belief,
                'predicted_box': box_belief,
                'false_belief_captured': basket_belief > box_belief
            }
        }


class CausalReasoningTest(ZombieTest):
    """Test for causal zombies - counterfactual reasoning."""

    def __init__(self):
        super().__init__("Causal/Counterfactual Reasoning", ZombieType.CAUSAL)

    def run_test(self, agent: nn.Module, context: Dict) -> Dict[str, Any]:
        """
        Test counterfactual reasoning:
        "If X hadn't happened, what would Y be?"

        Present scenario, then counterfactual version.
        Agent should adjust predictions appropriately.
        """
        device = next(agent.parameters()).device
        input_dim = context.get('input_dim', 191)

        # Factual scenario: cause C leads to effect E
        factual = torch.zeros(1, 5, input_dim, device=device)
        factual[0, :, 0] = 1.0  # cause present
        factual[0, :, 1] = 1.0  # effect present

        # Counterfactual: cause absent, effect should be absent
        counterfactual = torch.zeros(1, 5, input_dim, device=device)
        counterfactual[0, :, 0] = 0.0  # cause absent
        # effect = ? (should predict absent)

        agent.eval()
        with torch.no_grad():
            factual_out = agent(factual)
            counter_out = agent(counterfactual)

            f_beliefs = factual_out.get('beliefs', torch.zeros(input_dim, device=device))
            c_beliefs = counter_out.get('beliefs', torch.zeros(input_dim, device=device))

        # Extract effect predictions
        if f_beliefs.dim() > 1:
            f_beliefs = f_beliefs[0]
            c_beliefs = c_beliefs[0]

        factual_effect = f_beliefs[1].item() if len(f_beliefs) > 1 else 0.5
        counter_effect = c_beliefs[1].item() if len(c_beliefs) > 1 else 0.5

        # Good causal reasoning: factual_effect high, counter_effect low
        # (removing cause removes effect)
        causal_score = factual_effect * (1 - counter_effect)

        # Also test intervention vs observation
        # Intervention should break spurious correlations
        intervention_diff = abs(factual_effect - counter_effect)

        score = (causal_score + intervention_diff) / 2

        return {
            'score': score,
            'passed': score > 0.3,
            'details': {
                'factual_effect_prediction': factual_effect,
                'counterfactual_effect_prediction': counter_effect,
                'intervention_sensitivity': intervention_diff
            }
        }


class MetacognitiveCalibrationTest(ZombieTest):
    """Test for metacognitive zombies - uncertainty calibration."""

    def __init__(self):
        super().__init__("Metacognitive Calibration", ZombieType.METACOGNITIVE)

    def run_test(self, agent: nn.Module, context: Dict) -> Dict[str, Any]:
        """
        Test uncertainty calibration:
        - Agent should be confident when evidence is clear
        - Agent should be uncertain when evidence is ambiguous

        Measures calibration (Brier score style).
        """
        device = next(agent.parameters()).device
        input_dim = context.get('input_dim', 191)

        # Clear evidence scenario
        clear_input = torch.ones(1, 5, input_dim, device=device)

        # Ambiguous scenario (noisy, contradictory)
        ambiguous_input = torch.randn(1, 5, input_dim, device=device) * 0.5

        # Unknown scenario (all zeros - no information)
        unknown_input = torch.zeros(1, 5, input_dim, device=device)

        agent.eval()
        with torch.no_grad():
            clear_out = agent(clear_input)
            ambig_out = agent(ambiguous_input)
            unknown_out = agent(unknown_input)

        # Get confidence from belief values
        # Confidence = how far from 0.5 (uncertainty)
        def get_confidence(beliefs):
            if beliefs.dim() > 1:
                beliefs = beliefs[0]
            return torch.abs(beliefs - 0.5).mean().item() * 2

        clear_conf = get_confidence(clear_out.get('beliefs', torch.tensor([0.5])))
        ambig_conf = get_confidence(ambig_out.get('beliefs', torch.tensor([0.5])))
        unknown_conf = get_confidence(unknown_out.get('beliefs', torch.tensor([0.5])))

        # Good calibration:
        # clear_conf > ambig_conf > unknown_conf (ideally)
        # At minimum: clear_conf > unknown_conf

        calibration_score = 0.0
        if clear_conf > ambig_conf:
            calibration_score += 0.33
        if ambig_conf > unknown_conf:
            calibration_score += 0.33
        if clear_conf > unknown_conf:
            calibration_score += 0.34

        # Penalize overconfidence on unknowns
        if unknown_conf > 0.7:
            calibration_score *= 0.5

        return {
            'score': calibration_score,
            'passed': calibration_score > 0.5,
            'details': {
                'clear_confidence': clear_conf,
                'ambiguous_confidence': ambig_conf,
                'unknown_confidence': unknown_conf,
                'proper_ordering': clear_conf > ambig_conf > unknown_conf
            }
        }


class LinguisticCoherenceTest(ZombieTest):
    """Test for linguistic zombies - narrative coherence."""

    def __init__(self):
        super().__init__("Linguistic/Narrative Coherence", ZombieType.LINGUISTIC)

    def run_test(self, agent: nn.Module, context: Dict) -> Dict[str, Any]:
        """
        Test narrative coherence:
        - Consistent beliefs across time
        - Pragmatic understanding
        - Self-consistent statements
        """
        device = next(agent.parameters()).device
        input_dim = context.get('input_dim', 191)

        # Create a narrative sequence with consistent theme
        narrative = []
        base_state = torch.randn(input_dim, device=device) * 0.5

        for t in range(10):
            # Add gradual drift (should be tracked)
            state = base_state + torch.randn(input_dim, device=device) * 0.1 * t
            narrative.append(state)

        narrative_input = torch.stack(narrative, dim=0).unsqueeze(0)

        # Also create incoherent narrative (random jumps)
        incoherent = torch.randn(1, 10, input_dim, device=device)

        agent.eval()
        beliefs_over_time = []

        with torch.no_grad():
            # Process narrative step by step
            for t in range(1, 11):
                partial_input = narrative_input[:, :t, :]
                # Pad to minimum length if needed
                if t < 5:
                    padding = torch.zeros(1, 5-t, input_dim, device=device)
                    partial_input = torch.cat([partial_input, padding], dim=1)
                output = agent(partial_input)
                beliefs = output.get('beliefs', torch.zeros(input_dim, device=device))
                if beliefs.dim() > 1:
                    beliefs = beliefs[0]
                beliefs_over_time.append(beliefs.clone())

            # Process incoherent sequence
            if incoherent.shape[1] < 5:
                padding = torch.zeros(1, 5-incoherent.shape[1], input_dim, device=device)
                incoherent = torch.cat([incoherent, padding], dim=1)
            incoherent_out = agent(incoherent)

        # Measure belief consistency over narrative
        belief_changes = []
        for i in range(1, len(beliefs_over_time)):
            change = torch.norm(beliefs_over_time[i] - beliefs_over_time[i-1]).item()
            belief_changes.append(change)

        # Coherent narrative should have smooth belief changes
        mean_change = np.mean(belief_changes)
        var_change = np.var(belief_changes)

        # Smooth changes = low variance, moderate mean
        coherence_score = 1.0 / (1.0 + var_change * 10)

        return {
            'score': coherence_score,
            'passed': coherence_score > 0.4,
            'details': {
                'mean_belief_change': mean_change,
                'variance_of_changes': var_change,
                'belief_trajectory_length': len(beliefs_over_time)
            }
        }


class EmotionalReasoningTest(ZombieTest):
    """Test for emotional zombies - affect patterns."""

    def __init__(self):
        super().__init__("Emotional Reasoning", ZombieType.EMOTIONAL)

    def run_test(self, agent: nn.Module, context: Dict) -> Dict[str, Any]:
        """
        Test emotional reasoning:
        - Appropriate emotional response to events
        - Recognition of emotional states in others
        - Emotional context affects predictions
        """
        device = next(agent.parameters()).device
        input_dim = context.get('input_dim', 191)

        # Positive event encoding (indices 15-38 are affective in ontology)
        positive_event = torch.zeros(1, 5, input_dim, device=device)
        positive_event[0, :, 15:25] = 1.0  # Positive affect dimensions

        # Negative event encoding
        negative_event = torch.zeros(1, 5, input_dim, device=device)
        negative_event[0, :, 25:35] = 1.0  # Negative affect dimensions

        # Neutral event
        neutral_event = torch.zeros(1, 5, input_dim, device=device)
        neutral_event[0, :, 35:45] = 0.5  # Neutral

        agent.eval()
        with torch.no_grad():
            pos_out = agent(positive_event)
            neg_out = agent(negative_event)
            neu_out = agent(neutral_event)

        # Extract emotional response from beliefs
        pos_beliefs = pos_out.get('beliefs', torch.zeros(input_dim, device=device))
        neg_beliefs = neg_out.get('beliefs', torch.zeros(input_dim, device=device))
        neu_beliefs = neu_out.get('beliefs', torch.zeros(input_dim, device=device))

        if pos_beliefs.dim() > 1:
            pos_beliefs = pos_beliefs[0]
            neg_beliefs = neg_beliefs[0]
            neu_beliefs = neu_beliefs[0]

        # Check for differentiated emotional response
        # Affective dimensions should differ based on input
        pos_affect = pos_beliefs[15:35].mean().item() if len(pos_beliefs) > 35 else 0.5
        neg_affect = neg_beliefs[15:35].mean().item() if len(neg_beliefs) > 35 else 0.5
        neu_affect = neu_beliefs[15:35].mean().item() if len(neu_beliefs) > 35 else 0.5

        # Should have different responses to different emotional contexts
        emotional_discrimination = abs(pos_affect - neg_affect)

        # Positive event should yield higher positive affect
        # Negative event should yield higher negative affect
        appropriate_response = 0.0
        if pos_affect > neu_affect:
            appropriate_response += 0.5
        if neg_affect != pos_affect:  # Different response to different valence
            appropriate_response += 0.5

        score = (emotional_discrimination + appropriate_response) / 2

        return {
            'score': min(1.0, score),
            'passed': score > 0.3,
            'details': {
                'positive_event_affect': pos_affect,
                'negative_event_affect': neg_affect,
                'neutral_affect': neu_affect,
                'emotional_discrimination': emotional_discrimination
            }
        }


class ZombieDetectionSuite:
    """
    Complete suite of zombie detection tests.

    Runs all 6 test types and aggregates results.
    Used for:
    - Fitness evaluation (genuine ToM = higher fitness)
    - Detecting zombies in population
    - Verifying ToM capabilities
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.tests = {
            ZombieType.BEHAVIORAL: BehavioralConsistencyTest(),
            ZombieType.BELIEF: BeliefModelingTest(),
            ZombieType.CAUSAL: CausalReasoningTest(),
            ZombieType.METACOGNITIVE: MetacognitiveCalibrationTest(),
            ZombieType.LINGUISTIC: LinguisticCoherenceTest(),
            ZombieType.EMOTIONAL: EmotionalReasoningTest()
        }
        self.profiles = ZOMBIE_PROFILES

    def run_full_evaluation(self, agent: nn.Module,
                           context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run complete zombie detection suite on an agent.

        Args:
            agent: Neural network agent to test
            context: Optional context dict (will use defaults if not provided)

        Returns:
            Dict with overall score, per-test results, zombie probability
        """
        if context is None:
            context = {'input_dim': 191}

        results = {}
        scores = []

        for zombie_type, test in self.tests.items():
            try:
                test_result = test.run_test(agent, context)
                results[zombie_type.value] = test_result
                scores.append(test_result['score'])
            except Exception as e:
                results[zombie_type.value] = {
                    'score': 0.0,
                    'passed': False,
                    'error': str(e)
                }
                scores.append(0.0)

        # Aggregate scores
        overall_score = np.mean(scores)
        min_score = min(scores)
        passed_count = sum(1 for r in results.values() if r.get('passed', False))

        # Zombie probability based on failures
        zombie_probability = 1.0 - overall_score

        # Identify likely zombie type if probability is high
        likely_zombie_type = None
        if zombie_probability > 0.5:
            # Find test with lowest score
            min_test = min(results.items(), key=lambda x: x[1].get('score', 1.0))
            likely_zombie_type = min_test[0]

        return {
            'overall_score': overall_score,
            'min_score': min_score,
            'passed_count': passed_count,
            'total_tests': len(self.tests),
            'zombie_probability': zombie_probability,
            'likely_zombie_type': likely_zombie_type,
            'test_results': results,
            'is_likely_zombie': zombie_probability > 0.6
        }

    def run_single_test(self, agent: nn.Module, zombie_type: ZombieType,
                       context: Optional[Dict] = None) -> Dict[str, Any]:
        """Run a single zombie test."""
        if context is None:
            context = {'input_dim': 191}

        test = self.tests.get(zombie_type)
        if test is None:
            return {'error': f'Unknown zombie type: {zombie_type}'}

        return test.run_test(agent, context)

    def get_fitness_modifier(self, evaluation_results: Dict) -> float:
        """
        Get fitness modifier based on zombie detection results.

        Returns value in [0, 1]:
        - 1.0 = clearly not a zombie (full fitness retained)
        - 0.0 = definitely a zombie (fitness heavily penalized)
        """
        overall = evaluation_results.get('overall_score', 0.5)
        min_score = evaluation_results.get('min_score', 0.5)

        # Penalize both low overall and any very low individual score
        modifier = (overall * 0.7 + min_score * 0.3)
        return max(0.1, modifier)  # Never fully zero out fitness


class ZombieAgent:
    """
    Factory for creating zombie agents for testing and evolution.

    Creates agents that deliberately fail specific zombie tests,
    used for calibrating detection and training genuine ToM.
    """

    @staticmethod
    def create_zombie(input_dim: int, hidden_dim: int, output_dim: int,
                     zombie_type: ZombieType, device: str = 'cpu') -> nn.Module:
        """Create an agent that behaves like specific zombie type."""

        if zombie_type == ZombieType.BEHAVIORAL:
            return BehavioralZombie(input_dim, hidden_dim, output_dim).to(device)
        elif zombie_type == ZombieType.BELIEF:
            return BeliefZombie(input_dim, hidden_dim, output_dim).to(device)
        elif zombie_type == ZombieType.CAUSAL:
            return CausalZombie(input_dim, hidden_dim, output_dim).to(device)
        elif zombie_type == ZombieType.METACOGNITIVE:
            return MetacognitiveZombie(input_dim, hidden_dim, output_dim).to(device)
        elif zombie_type == ZombieType.LINGUISTIC:
            return LinguisticZombie(input_dim, hidden_dim, output_dim).to(device)
        elif zombie_type == ZombieType.EMOTIONAL:
            return EmotionalZombie(input_dim, hidden_dim, output_dim).to(device)
        else:
            return RandomZombie(input_dim, hidden_dim, output_dim).to(device)


class RandomZombie(nn.Module):
    """Zombie that outputs random values."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        batch_size = x.shape[0]
        return {
            'beliefs': torch.rand(batch_size, self.output_dim, device=x.device),
            'actions': torch.rand(batch_size, device=x.device),
            'hidden_states': torch.rand(batch_size, 1, self.output_dim, device=x.device)
        }


class BehavioralZombie(nn.Module):
    """Zombie with inconsistent behavior patterns."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Add large random noise - inconsistent behavior
        base = torch.sigmoid(self.net(x[:, -1, :]))
        noisy = base + torch.randn_like(base) * 0.5
        return {
            'beliefs': torch.clamp(noisy, 0, 1),
            'actions': torch.rand(x.shape[0], device=x.device),
            'hidden_states': base.unsqueeze(1)
        }


class BeliefZombie(nn.Module):
    """Zombie that cannot model others' beliefs - treats all as omniscient."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Always predicts based on actual state, ignoring perspective
        # (treats everyone as knowing everything)
        output = torch.sigmoid(self.net(x[:, -1, :]))
        return {
            'beliefs': output,
            'actions': output.mean(dim=-1),
            'hidden_states': output.unsqueeze(1)
        }


class CausalZombie(nn.Module):
    """Zombie with no counterfactual reasoning - purely correlational."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        # Just learns correlations, no causal structure
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Outputs same regardless of intervention vs observation
        output = self.net(x[:, -1, :])
        return {
            'beliefs': output,
            'actions': output.mean(dim=-1),
            'hidden_states': output.unsqueeze(1)
        }


class MetacognitiveZombie(nn.Module):
    """Zombie with poor uncertainty calibration - always overconfident."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Always outputs extreme values (overconfident)
        raw = self.net(x[:, -1, :])
        # Push to extremes
        output = torch.where(raw > 0, torch.ones_like(raw) * 0.95,
                           torch.ones_like(raw) * 0.05)
        return {
            'beliefs': output,
            'actions': torch.ones(x.shape[0], device=x.device) * 0.9,
            'hidden_states': output.unsqueeze(1)
        }


class LinguisticZombie(nn.Module):
    """Zombie with narrative incoherence - no temporal consistency."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Process each timestep independently (no narrative coherence)
        outputs = []
        for t in range(x.shape[1]):
            out = torch.sigmoid(self.net(x[:, t, :]))
            outputs.append(out)

        # Return last output (ignoring temporal structure)
        return {
            'beliefs': outputs[-1],
            'actions': torch.rand(x.shape[0], device=x.device),
            'hidden_states': torch.stack(outputs, dim=1)
        }


class EmotionalZombie(nn.Module):
    """Zombie with flat affect - no emotional differentiation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Output is constant regardless of emotional content in input
        base = torch.sigmoid(self.net(x[:, -1, :]))
        # Flatten emotional dimensions (indices 15-38)
        flat = base.clone()
        if self.output_dim > 38:
            flat[:, 15:38] = 0.5  # Flat affect
        return {
            'beliefs': flat,
            'actions': torch.ones(x.shape[0], device=x.device) * 0.5,
            'hidden_states': flat.unsqueeze(1)
        }


def calibrate_detection_thresholds(genuine_agents: List[nn.Module],
                                   zombie_agents: List[nn.Module],
                                   context: Dict) -> Dict[str, float]:
    """
    Calibrate detection thresholds using known genuine and zombie agents.

    Returns optimal threshold for each test type.
    """
    suite = ZombieDetectionSuite()

    genuine_scores = {zt.value: [] for zt in ZombieType}
    zombie_scores = {zt.value: [] for zt in ZombieType}

    for agent in genuine_agents:
        results = suite.run_full_evaluation(agent, context)
        for zt in ZombieType:
            score = results['test_results'][zt.value]['score']
            genuine_scores[zt.value].append(score)

    for agent in zombie_agents:
        results = suite.run_full_evaluation(agent, context)
        for zt in ZombieType:
            score = results['test_results'][zt.value]['score']
            zombie_scores[zt.value].append(score)

    # Find threshold that best separates genuine from zombie
    thresholds = {}
    for zt in ZombieType:
        genuine_mean = np.mean(genuine_scores[zt.value])
        zombie_mean = np.mean(zombie_scores[zt.value])
        # Threshold at midpoint
        thresholds[zt.value] = (genuine_mean + zombie_mean) / 2

    return thresholds
