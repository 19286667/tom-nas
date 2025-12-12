"""
Researcher Agent: Agents That Write and Execute Real Code

This module implements the core insight: agents that can write executable code
and create simulations containing other agents. This recursive capability
creates genuine selective pressure for Theory of Mind and abductive reasoning.

When Agent A writes code that simulates Agent B, and Agent B's behavior depends
on its model of Agent A, we get true recursive modeling - not just behavioral
mimicry, but genuine higher-order reasoning about reasoning.

Key Capabilities:
1. Code Generation - Writing Python/DSL code for experiments
2. Hypothesis Formation - Abductive reasoning about observations
3. Experimental Design - Creating tests for hypotheses
4. Publication - Communicating findings to other agents
5. Peer Review - Evaluating others' work
6. Recursive Simulation - Creating worlds containing other agents
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import uuid
from datetime import datetime

from src.config import get_logger
from src.config.constants import SOUL_MAP_DIMS, INPUT_DIMS, OUTPUT_DIMS
from src.core.beliefs import RecursiveBeliefState, BeliefNetwork
from src.agents.architectures import TransparentRNN

logger = get_logger(__name__)


class ResearchDomain(Enum):
    """Domains of research that agents can specialize in."""
    COGNITIVE_SCIENCE = "cognitive_science"
    MACHINE_LEARNING = "machine_learning"
    SOCIAL_DYNAMICS = "social_dynamics"
    GAME_THEORY = "game_theory"
    COMPLEXITY_THEORY = "complexity_theory"
    PHILOSOPHY_OF_MIND = "philosophy_of_mind"
    COMPUTATIONAL_SOCIOLOGY = "computational_sociology"


@dataclass
class ResearchAgenda:
    """
    A researcher's current research program.

    Contains hypotheses, methods, and goals that guide research activity.
    """
    domain: ResearchDomain = ResearchDomain.COGNITIVE_SCIENCE
    hypotheses: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    current_experiment: Optional[str] = None
    publications: List[str] = field(default_factory=list)

    # Meta-research: studying how research itself works
    meta_level: int = 0  # 0 = object-level research, 1+ = meta-research


@dataclass
class CodeArtifact:
    """
    A piece of code written by a researcher agent.

    Now uses lambda calculus expressions for intrinsic safety:
    - No sandboxing needed (pure functions only)
    - Composable from verified primitives
    - Supports library compression (Stitch)
    - Auto-documented for capability discovery (AutoDoc)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    author_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Lambda calculus expression (safe by construction)
    lambda_expr: Optional[Any] = None  # LambdaExpr from synthesis module
    source_repr: str = ""  # String representation
    language: str = "lambda_calculus"  # Now lambda calculus, not Python

    # Metadata
    description: str = ""
    purpose: str = ""  # experiment, simulation, analysis, tool
    dependencies: List[str] = field(default_factory=list)

    # Execution results
    executed: bool = False
    execution_result: Optional[Dict[str, Any]] = None
    execution_error: Optional[str] = None

    # Scientific metadata
    hypothesis_tested: Optional[str] = None
    results_support_hypothesis: Optional[bool] = None


@dataclass
class Publication:
    """
    A research publication created by an agent.

    Publications are how agents communicate findings and influence
    the beliefs and research directions of other agents.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    authors: List[str] = field(default_factory=list)
    institution_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Content
    abstract: str = ""
    hypotheses: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    conclusions: List[str] = field(default_factory=list)

    # Code artifacts (reproducibility)
    code_artifacts: List[str] = field(default_factory=list)

    # Impact metrics
    citations: int = 0
    peer_review_score: float = 0.0
    reproducibility_score: float = 0.0

    # The key insight: does this contain a simulation?
    contains_simulation: bool = False
    simulation_depth: int = 0  # How many levels of nested simulation


class CodeGenerator(nn.Module):
    """
    Neural network that generates executable code from research goals.

    This is where the magic happens: agents learn to write code that
    creates simulations, tests hypotheses, and advances understanding.
    """

    def __init__(
        self,
        context_dim: int = INPUT_DIMS,
        hidden_dim: int = 256,
        vocab_size: int = 10000,
        max_code_length: int = 512,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_code_length = max_code_length

        # Encode research context
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Generate code tokens autoregressively
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.code_generator = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nheads=8, batch_first=True),
            num_layers=4,
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        context: torch.Tensor,
        partial_code: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate code given research context.

        Args:
            context: Research context tensor (beliefs, goals, observations)
            partial_code: Optional partial code to continue

        Returns:
            Token probabilities for next code token
        """
        # Encode context
        context_encoded = self.context_encoder(context).unsqueeze(1)

        if partial_code is None:
            # Start token
            partial_code = torch.zeros(context.shape[0], 1, dtype=torch.long)

        # Embed partial code
        code_embedded = self.token_embedding(partial_code)

        # Generate next token
        decoded = self.code_generator(code_embedded, context_encoded)
        logits = self.output_projection(decoded[:, -1, :])

        return logits


class ResearcherAgent:
    """
    An agent that conducts research by synthesizing lambda calculus programs.

    This is the core entity in the recursive simulation framework.
    A ResearcherAgent can:
    1. Form hypotheses about the world (abductive reasoning)
    2. Synthesize λ-expressions to test hypotheses (Lilo-style)
    3. Execute programs safely (intrinsic guardrails via pure functions)
    4. Compress successful patterns into libraries (Stitch)
    5. Document capabilities for sharing (AutoDoc → A2A protocol)
    6. Create simulations containing OTHER agents (recursive ToM)

    Safety Model:
        Unlike sandboxed Python execution, this agent uses neurosymbolic
        program synthesis. Programs are composed from verified primitives
        in lambda calculus, providing intrinsic safety guarantees:
        - No side effects (pure functions only)
        - No I/O, file access, or network operations
        - Compositional verification (valid parts → valid whole)

    The recursive capability (creating simulations with agents) creates
    genuine selective pressure for Theory of Mind.
    """

    def __init__(
        self,
        agent_id: str = None,
        name: str = "Researcher",
        institution_id: str = None,
        specialization: ResearchDomain = ResearchDomain.COGNITIVE_SCIENCE,
    ):
        self.id = agent_id or str(uuid.uuid4())[:8]
        self.name = name
        self.institution_id = institution_id
        self.specialization = specialization

        # Cognitive architecture
        self.belief_state = RecursiveBeliefState(
            agent_id=hash(self.id) % 1000,
            ontology_dim=SOUL_MAP_DIMS,
            max_order=5,
        )

        # Research state
        self.agenda = ResearchAgenda(domain=specialization)
        self.code_artifacts: List[CodeArtifact] = []
        self.publications: List[Publication] = []

        # Neurosymbolic synthesis (replaces sandboxed code generation)
        try:
            from src.synthesis import NeurosymbolicSynthesizer
            self.synthesizer = NeurosymbolicSynthesizer()
        except ImportError:
            self.synthesizer = None
            logger.warning("Synthesis module not available, using fallback")

        # Neural components
        self.reasoning_model = TransparentRNN(
            input_dim=INPUT_DIMS,
            hidden_dim=256,
            output_dim=OUTPUT_DIMS,
        )
        self.code_generator = CodeGenerator()

        # Simulation capability
        self.simulations_created: List[str] = []
        self.simulation_depth: int = 0  # How deep in recursive simulation

        # Track scientific progress
        self.hypotheses_tested: int = 0
        self.hypotheses_confirmed: int = 0
        self.hypotheses_refuted: int = 0

        logger.info(f"Created ResearcherAgent {self.name} ({self.id}) specializing in {specialization.value}")

    def form_hypothesis(self, observations: torch.Tensor) -> str:
        """
        Use abductive reasoning to form a hypothesis from observations.

        This is where Theory of Mind becomes crucial: good hypotheses
        about agent behavior require modeling agent reasoning.
        """
        # Process observations through reasoning model
        with torch.no_grad():
            output = self.reasoning_model(observations.unsqueeze(0).unsqueeze(0))
            beliefs = output['beliefs']

        # Convert belief state to hypothesis (simplified)
        # In full implementation, this would use language generation
        confidence = float(beliefs.mean())
        hypothesis = f"H_{self.hypotheses_tested + 1}: Observable pattern suggests " \
                     f"underlying mechanism with confidence {confidence:.2f}"

        self.agenda.hypotheses.append(hypothesis)
        logger.debug(f"{self.name} formed hypothesis: {hypothesis}")

        return hypothesis

    def design_experiment(self, hypothesis: str) -> CodeArtifact:
        """
        Design an experiment to test a hypothesis.

        Uses neurosymbolic synthesis to create a λ-expression that:
        1. Is safe by construction (no side effects)
        2. Can be compressed into reusable abstractions (Stitch)
        3. Is auto-documented for capability sharing (AutoDoc)
        """
        # Synthesize lambda expression for experiment
        lambda_expr = self._synthesize_experiment(hypothesis)

        artifact = CodeArtifact(
            author_id=self.id,
            lambda_expr=lambda_expr,
            source_repr=lambda_expr.to_string() if lambda_expr else "",
            description=f"Experiment to test: {hypothesis}",
            purpose="experiment",
            hypothesis_tested=hypothesis,
        )

        self.code_artifacts.append(artifact)
        self.agenda.current_experiment = artifact.id

        # Record for library compression
        if self.synthesizer and lambda_expr:
            self.synthesizer.compressor.add_successful_program(
                f"experiment_{artifact.id}",
                lambda_expr
            )

        return artifact

    def _synthesize_experiment(self, hypothesis: str) -> Optional[Any]:
        """
        Synthesize a λ-expression for an experiment.

        Uses Lilo-style neurosymbolic synthesis:
        - LLM proposes candidate expressions
        - Symbolic verification ensures validity
        - Stitch compresses successful patterns

        Returns a lambda calculus expression (intrinsically safe).
        """
        if not self.synthesizer:
            return self._fallback_synthesis(hypothesis)

        # Synthesize from hypothesis specification
        program = self.synthesizer.synthesize(
            specification=f"experiment to test: {hypothesis}",
            examples=None  # Would include I/O examples in full implementation
        )

        if program:
            logger.debug(f"{self.name} synthesized: {program.to_string()}")
            return program

        # Fallback to manual construction
        return self._fallback_synthesis(hypothesis)

    def _fallback_synthesis(self, hypothesis: str) -> Any:
        """
        Fallback: manually construct λ-expression for experiment.

        This constructs from safe primitives only.
        """
        try:
            from src.synthesis import Lam, Var, Prim, Lit, App

            # Construct: λconfig.(mean (map (λx.(+ x (observe agent))) (range 0 100)))
            experiment = Lam('config',
                Prim('mean', [
                    Prim('map', [
                        Lam('x', Prim('+', [Var('x'), Lit(0.0)])),
                        Prim('range', [Lit(0), Lit(100)])
                    ])
                ])
            )
            return experiment
        except ImportError:
            logger.warning("Synthesis module not available")
            return None

    def run_experiment(self, artifact: CodeArtifact, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute an experiment artifact safely.

        Since artifacts are λ-expressions composed from verified primitives,
        execution is intrinsically safe - no sandbox needed.
        """
        if artifact.lambda_expr is None:
            return {"error": "No lambda expression in artifact"}

        try:
            if self.synthesizer:
                result = self.synthesizer.evaluate(artifact.lambda_expr, config or {})
            else:
                result = artifact.lambda_expr.evaluate(config or {})

            return {
                "success": True,
                "result": result,
                "hypothesis": artifact.hypothesis_tested,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "hypothesis": artifact.hypothesis_tested,
            }

    def evolve_capabilities(self) -> List[Dict[str, Any]]:
        """
        Compress successful experiments into reusable library abstractions.

        This is the Stitch algorithm: identify repeated patterns in
        successful programs and extract them as named abstractions.
        These become the agent's "procedural memory".
        """
        if not self.synthesizer:
            return []

        # Compress and document new abstractions
        new_capabilities = self.synthesizer.evolve_library()

        # Convert to A2A-compatible format
        capability_cards = []
        for cap in new_capabilities:
            capability_cards.append(cap.to_agent_card())
            logger.info(f"{self.name} evolved capability: {cap.name}")

        return capability_cards

    def create_simulation(self, config: Dict[str, Any] = None) -> 'RecursiveSimulation':
        """
        Create a simulation containing other agents.

        THIS IS THE KEY RECURSIVE CAPABILITY.

        When an agent creates a simulation containing other agents,
        those inner agents can themselves create simulations.
        This nested structure creates genuine selective pressure for
        Theory of Mind: to predict what simulated agents will do,
        you must model their modeling of other agents.
        """
        from src.institutions.recursive_world import RecursiveSimulation

        config = config or {}
        config['creator_id'] = self.id
        config['depth'] = self.simulation_depth + 1

        simulation = RecursiveSimulation(
            parent_agent=self,
            depth=self.simulation_depth + 1,
            config=config,
        )

        self.simulations_created.append(simulation.id)

        logger.info(f"{self.name} created simulation at depth {simulation.depth}")

        return simulation

    def write_publication(
        self,
        title: str,
        results: Dict[str, Any],
        code_artifacts: List[CodeArtifact] = None,
    ) -> Publication:
        """
        Write a research publication.

        Publications spread findings through the agent population,
        allowing for cumulative scientific progress.
        """
        publication = Publication(
            title=title,
            authors=[self.id],
            institution_id=self.institution_id,
            abstract=f"Research by {self.name} in {self.specialization.value}",
            hypotheses=self.agenda.hypotheses[-3:],  # Recent hypotheses
            results=results,
            code_artifacts=[a.id for a in (code_artifacts or [])],
            contains_simulation=len(self.simulations_created) > 0,
            simulation_depth=self.simulation_depth,
        )

        self.publications.append(publication)
        self.agenda.publications.append(publication.id)

        logger.info(f"{self.name} published: {title}")

        return publication

    def review_publication(self, publication: Publication) -> Dict[str, Any]:
        """
        Peer review another agent's publication.

        This requires Theory of Mind: evaluating whether the research
        methodology and conclusions are sound requires understanding
        the reasoning process of the author.
        """
        review = {
            "reviewer_id": self.id,
            "publication_id": publication.id,
            "scores": {
                "methodology": 0.0,
                "novelty": 0.0,
                "reproducibility": 0.0,
                "significance": 0.0,
            },
            "comments": [],
            "recommendation": "accept",  # accept, revise, reject
        }

        # Evaluate methodology
        if publication.code_artifacts:
            review["scores"]["reproducibility"] = 0.8
            review["comments"].append("Code provided for reproducibility")

        # Evaluate novelty (simplified)
        if publication.contains_simulation:
            review["scores"]["novelty"] = 0.9
            review["comments"].append("Novel use of recursive simulation")

        # Overall score
        review["scores"]["methodology"] = 0.7
        review["scores"]["significance"] = 0.7

        avg_score = sum(review["scores"].values()) / len(review["scores"])
        if avg_score > 0.7:
            review["recommendation"] = "accept"
        elif avg_score > 0.5:
            review["recommendation"] = "revise"
        else:
            review["recommendation"] = "reject"

        publication.peer_review_score = avg_score

        return review

    def update_beliefs(self, evidence: Dict[str, Any]):
        """
        Update beliefs based on new evidence.

        This is Bayesian belief updating informed by
        experimental results and peer publications.
        """
        # Convert evidence to tensor
        evidence_tensor = torch.zeros(SOUL_MAP_DIMS)
        if "confidence" in evidence:
            evidence_tensor[0] = evidence["confidence"]

        # Update belief state
        self.belief_state.update_belief(
            order=1,
            target=0,
            content=evidence_tensor,
            confidence=evidence.get("confidence", 0.5),
            source="evidence",
        )

        # Update hypothesis tracking
        if evidence.get("supports_hypothesis"):
            self.hypotheses_confirmed += 1
        else:
            self.hypotheses_refuted += 1

        self.hypotheses_tested += 1

    def get_scientific_fitness(self) -> float:
        """
        Calculate the agent's scientific fitness.

        This combines publication impact, reproducibility,
        hypothesis accuracy, and simulation sophistication.
        """
        # Publication impact
        pub_score = sum(p.peer_review_score for p in self.publications)

        # Hypothesis accuracy
        if self.hypotheses_tested > 0:
            accuracy = self.hypotheses_confirmed / self.hypotheses_tested
        else:
            accuracy = 0.0

        # Simulation sophistication (recursive depth)
        sim_score = min(1.0, len(self.simulations_created) * 0.2)

        # Code artifacts (reproducibility)
        code_score = min(1.0, len(self.code_artifacts) * 0.1)

        fitness = (
            0.3 * min(1.0, pub_score / 10) +
            0.3 * accuracy +
            0.2 * sim_score +
            0.2 * code_score
        )

        return fitness
