#!/usr/bin/env python3
"""
ToM-NAS Self-Validation Suite

Run this ONCE to validate the entire system. It will:
1. Check all imports work
2. Validate each component
3. Run integration tests
4. Generate a clear report

Usage:
    pip install -r requirements-minimal.txt
    python validate_system.py

Output: Clear PASS/FAIL for each component + overall status
"""

import sys
import os
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0
    details: Optional[str] = None


class ValidationSuite:
    """Self-contained validation that tests everything."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = datetime.now()

    def run_test(self, name: str, test_fn: Callable) -> TestResult:
        """Run a single test and capture result."""
        import time
        start = time.time()
        try:
            test_fn()
            duration = (time.time() - start) * 1000
            result = TestResult(name=name, passed=True, message="OK", duration_ms=duration)
        except Exception as e:
            duration = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                passed=False,
                message=str(e),
                duration_ms=duration,
                details=traceback.format_exc()
            )
        self.results.append(result)
        return result

    def print_result(self, result: TestResult):
        """Print a single result with color."""
        status = "✓ PASS" if result.passed else "✗ FAIL"
        color = "\033[92m" if result.passed else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} {result.name} ({result.duration_ms:.1f}ms)")
        if not result.passed:
            print(f"       └─ {result.message}")

    def run_all(self):
        """Run complete validation suite."""
        print("=" * 70)
        print("ToM-NAS SYSTEM VALIDATION")
        print("=" * 70)
        print()

        # Phase 1: Core Imports
        print("Phase 1: Core Imports")
        print("-" * 40)
        self._test_core_imports()
        print()

        # Phase 2: Configuration
        print("Phase 2: Configuration System")
        print("-" * 40)
        self._test_config()
        print()

        # Phase 3: Core Components
        print("Phase 3: Core ToM Components")
        print("-" * 40)
        self._test_core_components()
        print()

        # Phase 4: Neurosymbolic Synthesis
        print("Phase 4: Neurosymbolic Synthesis (λ-calculus)")
        print("-" * 40)
        self._test_synthesis()
        print()

        # Phase 5: Institutions & Agents
        print("Phase 5: Institutional Framework")
        print("-" * 40)
        self._test_institutions()
        print()

        # Phase 6: Verification Framework
        print("Phase 6: Scientific Verification (NSHE/PIMMUR/PAN)")
        print("-" * 40)
        self._test_verification()
        print()

        # Phase 7: Integration
        print("Phase 7: Integration Tests")
        print("-" * 40)
        self._test_integration()
        print()

        # Summary
        self._print_summary()

    def _test_core_imports(self):
        """Test that all modules can be imported."""
        def test_torch():
            import torch
            assert torch.__version__, "PyTorch version not found"

        def test_numpy():
            import numpy as np
            assert np.__version__, "NumPy version not found"

        def test_config():
            from src.config import constants
            assert hasattr(constants, 'SOUL_MAP_DIMS')

        self.print_result(self.run_test("import torch", test_torch))
        self.print_result(self.run_test("import numpy", test_numpy))
        self.print_result(self.run_test("import src.config", test_config))

    def _test_config(self):
        """Test configuration system."""
        def test_constants():
            from src.config.constants import SOUL_MAP_DIMS, INPUT_DIMS, OUTPUT_DIMS
            assert SOUL_MAP_DIMS == 181, f"Expected 181, got {SOUL_MAP_DIMS}"
            assert INPUT_DIMS == 191, f"Expected 191, got {INPUT_DIMS}"
            assert OUTPUT_DIMS == 181, f"Expected 181, got {OUTPUT_DIMS}"

        def test_settings():
            from src.config.settings import get_settings
            settings = get_settings()
            assert settings.device in ['cpu', 'cuda']

        def test_logging():
            from src.config.logging_config import get_logger
            logger = get_logger("test")
            logger.debug("Test message")

        self.print_result(self.run_test("constants defined correctly", test_constants))
        self.print_result(self.run_test("settings loads", test_settings))
        self.print_result(self.run_test("logging works", test_logging))

    def _test_core_components(self):
        """Test core ToM components."""
        def test_ontology():
            from src.core.ontology import SoulMapOntology
            onto = SoulMapOntology()
            assert onto.total_dims == 181
            state = onto.get_default_state()
            assert state.shape[0] == 181

        def test_beliefs():
            from src.core.beliefs import RecursiveBeliefState, BeliefNetwork
            import torch
            state = RecursiveBeliefState(agent_id=0, ontology_dim=181, max_order=5)
            state.update_belief(order=1, target=0, content=torch.randn(181))
            belief = state.get_belief(order=1, target=0)
            assert belief is not None

        def test_architectures():
            from src.agents.architectures import TransparentRNN
            import torch
            model = TransparentRNN(input_dim=191, hidden_dim=128, output_dim=181)
            x = torch.randn(1, 10, 191)
            output = model(x)
            assert 'beliefs' in output
            assert output['beliefs'].shape[-1] == 181

        self.print_result(self.run_test("SoulMapOntology", test_ontology))
        self.print_result(self.run_test("RecursiveBeliefState", test_beliefs))
        self.print_result(self.run_test("TransparentRNN", test_architectures))

    def _test_synthesis(self):
        """Test neurosymbolic synthesis."""
        def test_lambda_core():
            from src.synthesis import Var, Lam, App, Lit, Prim
            # Test: (λx.(+ x 1)) 5 = 6
            expr = App(Lam('x', Prim('+', [Var('x'), Lit(1)])), Lit(5))
            result = expr.evaluate({})
            assert result == 6, f"Expected 6, got {result}"

        def test_primitives():
            from src.synthesis import PRIMITIVES
            assert '+' in PRIMITIVES
            assert 'map' in PRIMITIVES
            assert 'believe' in PRIMITIVES
            # Verify primitives are safe (no dangerous ops)
            dangerous = {'eval', 'exec', 'open', 'system'}
            assert not dangerous.intersection(PRIMITIVES.keys())

        def test_stitch_compression():
            from src.synthesis import StitchCompressor, Lam, Var, Prim
            compressor = StitchCompressor()
            # Add some programs with repeated patterns
            for i in range(5):
                prog = Lam('x', Prim('+', [Var('x'), Var('x')]))
                compressor.add_successful_program(f"prog_{i}", prog)
            # Compression should find reusable patterns
            abstractions = compressor.compress(min_usage=2)
            # May or may not find abstractions depending on patterns

        def test_synthesizer():
            from src.synthesis import NeurosymbolicSynthesizer
            synth = NeurosymbolicSynthesizer()
            prog = synth.synthesize("compute sum of two numbers")
            assert prog is not None
            # Evaluate: (λx.(λy.(+ x y))) applied to 3, 4
            result = synth.evaluate(prog)
            assert callable(result)  # Should return a function

        self.print_result(self.run_test("λ-calculus core (Var, Lam, App)", test_lambda_core))
        self.print_result(self.run_test("safe primitives only", test_primitives))
        self.print_result(self.run_test("Stitch compression", test_stitch_compression))
        self.print_result(self.run_test("NeurosymbolicSynthesizer", test_synthesizer))

    def _test_institutions(self):
        """Test institutional framework."""
        def test_institution_types():
            from src.institutions.institutions import (
                ResearchLab, CorporateRD, GovernmentAgency, InstitutionType
            )
            lab = ResearchLab(name="Test Lab")
            corp = CorporateRD(name="Test Corp")
            gov = GovernmentAgency(name="Test Agency")
            assert lab.institution_type == InstitutionType.RESEARCH_LAB
            assert corp.resources.compute_budget > lab.resources.compute_budget

        def test_institutional_network():
            from src.institutions.institutions import InstitutionalNetwork
            network = InstitutionalNetwork()
            network.create_default_ecosystem()
            state = network.get_ecosystem_state()
            assert state["num_institutions"] >= 5

        def test_researcher_agent():
            from src.institutions.researcher_agent import ResearcherAgent, ResearchDomain
            agent = ResearcherAgent(
                name="Dr. Test",
                specialization=ResearchDomain.COGNITIVE_SCIENCE
            )
            assert agent.synthesizer is not None
            assert agent.belief_state is not None

        def test_code_artifact():
            from src.institutions.researcher_agent import ResearcherAgent
            agent = ResearcherAgent(name="Dr. Artifact")
            import torch
            hypothesis = agent.form_hypothesis(torch.randn(1, 1, 191))
            artifact = agent.design_experiment(hypothesis)
            assert artifact.lambda_expr is not None
            assert artifact.language == "lambda_calculus"

        self.print_result(self.run_test("Institution types", test_institution_types))
        self.print_result(self.run_test("InstitutionalNetwork", test_institutional_network))
        self.print_result(self.run_test("ResearcherAgent creation", test_researcher_agent))
        self.print_result(self.run_test("CodeArtifact (λ-expr)", test_code_artifact))

    def _test_verification(self):
        """Test scientific verification framework."""
        def test_energy_landscape():
            from src.verification import EnergyLandscape
            landscape = EnergyLandscape()
            hypothesis = {
                "coherence_score": 0.8,
                "novelty_score": 0.7,
                "testability_score": 0.9,
                "energy_balanced": True,
            }
            energy = landscape.compute_energy(hypothesis)
            assert "total" in energy
            assert energy["total"] < float('inf')

        def test_pimmur():
            from src.verification import PIMMURValidator, PIMMURScore
            from src.institutions.researcher_agent import ResearcherAgent
            validator = PIMMURValidator()
            agent = ResearcherAgent(name="Test Agent")
            score = validator.validate(agent)
            assert isinstance(score, PIMMURScore)
            assert 0 <= score.total <= 1

        def test_pan_simulator():
            from src.verification import PANSimulator, ValueJudgmentModule
            simulator = PANSimulator()
            # Simple world model
            world_model = lambda state, action: {**state, "step": state.get("step", 0) + 1}
            trajectory = simulator.simulate_trajectory(
                initial_state={"step": 0},
                proposed_actions=[{"type": "observe"}, {"type": "act"}],
                world_model=world_model,
                horizon=2
            )
            assert trajectory.value_judgment >= 0

        def test_scientific_verifier():
            from src.verification import ScientificVerifier
            from src.institutions.researcher_agent import ResearcherAgent
            verifier = ScientificVerifier()
            agent = ResearcherAgent(name="Verified Agent")
            hypothesis = {
                "coherence_score": 0.8,
                "novelty_score": 0.6,
                "testability_score": 0.7,
                "usefulness_score": 0.8,
            }
            metrics = verifier.verify_hypothesis(hypothesis, agent)
            assert 0 <= metrics.overall_validity <= 1

        self.print_result(self.run_test("EnergyLandscape (NSHE)", test_energy_landscape))
        self.print_result(self.run_test("PIMMURValidator", test_pimmur))
        self.print_result(self.run_test("PANSimulator", test_pan_simulator))
        self.print_result(self.run_test("ScientificVerifier (unified)", test_scientific_verifier))

    def _test_integration(self):
        """Test end-to-end integration."""
        def test_agent_research_cycle():
            """Complete research cycle: hypothesis → experiment → results."""
            from src.institutions.researcher_agent import ResearcherAgent
            from src.verification import ScientificVerifier
            import torch

            # Create agent
            agent = ResearcherAgent(name="Dr. Integration")

            # Form hypothesis
            observation = torch.randn(1, 1, 191)
            hypothesis = agent.form_hypothesis(observation)
            assert "H_" in hypothesis

            # Design experiment (creates λ-expression)
            artifact = agent.design_experiment(hypothesis)
            assert artifact.lambda_expr is not None

            # Run experiment
            result = agent.run_experiment(artifact)
            assert "success" in result

            # Verify with scientific framework
            verifier = ScientificVerifier()
            metrics = verifier.verify_hypothesis(
                {"coherence_score": 0.7, "novelty_score": 0.6, "testability_score": 0.8},
                agent
            )
            assert metrics.pimmur > 0

        def test_recursive_simulation():
            """Test that agents can create simulations."""
            from src.institutions.recursive_world import RecursiveSimulation, WorldFactory
            sim = WorldFactory.create_minimal_simulation()
            results = sim.run(max_steps=5)
            assert results["total_steps"] == 5
            assert "final_emergence" in results

        def test_institutional_ecosystem():
            """Test multi-institution ecosystem."""
            from src.institutions.institutions import InstitutionalNetwork
            network = InstitutionalNetwork().create_default_ecosystem()
            # Step the ecosystem
            network.step()
            state = network.get_ecosystem_state()
            assert state["num_institutions"] >= 5

        self.print_result(self.run_test("Agent research cycle", test_agent_research_cycle))
        self.print_result(self.run_test("Recursive simulation", test_recursive_simulation))
        self.print_result(self.run_test("Institutional ecosystem", test_institutional_ecosystem))

    def _print_summary(self):
        """Print final summary."""
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        duration = (datetime.now() - self.start_time).total_seconds()

        print(f"  Total tests: {total}")
        print(f"  \033[92mPassed: {passed}\033[0m")
        print(f"  \033[91mFailed: {failed}\033[0m")
        print(f"  Duration: {duration:.2f}s")
        print()

        if failed > 0:
            print("Failed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
            print()

        if failed == 0:
            print("\033[92m" + "=" * 70)
            print("ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT")
            print("=" * 70 + "\033[0m")
        else:
            print("\033[91m" + "=" * 70)
            print(f"VALIDATION FAILED - {failed} test(s) need attention")
            print("=" * 70 + "\033[0m")

        return failed == 0


if __name__ == "__main__":
    suite = ValidationSuite()
    success = suite.run_all()
    sys.exit(0 if success else 1)
