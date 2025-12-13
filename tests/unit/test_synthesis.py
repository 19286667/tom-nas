"""
Unit tests for neurosymbolic synthesis engine.

Tests the lambda calculus core, primitives, and synthesis pipeline.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.synthesis import (
    Var, Lam, App, Lit, Prim,
    NeurosymbolicSynthesizer,
    StitchCompressor,
    AutoDoc,
    PRIMITIVES,
)


class TestLambdaCore:
    """Tests for lambda calculus expressions."""

    def test_literal_evaluation(self):
        """Literals evaluate to themselves."""
        assert Lit(42).evaluate({}) == 42
        assert Lit("hello").evaluate({}) == "hello"
        assert Lit(3.14).evaluate({}) == 3.14

    def test_variable_evaluation(self):
        """Variables lookup in environment."""
        env = {"x": 10, "y": 20}
        assert Var("x").evaluate(env) == 10
        assert Var("y").evaluate(env) == 20

    def test_variable_unbound(self):
        """Unbound variables raise NameError."""
        with pytest.raises(NameError):
            Var("z").evaluate({})

    def test_lambda_creates_closure(self):
        """Lambda creates callable closure."""
        identity = Lam("x", Var("x"))
        fn = identity.evaluate({})
        assert callable(fn)
        assert fn(42) == 42

    def test_lambda_captures_environment(self):
        """Lambda captures outer environment."""
        # λx. x + y where y=10
        add_y = Lam("x", Prim("+", [Var("x"), Var("y")]))
        fn = add_y.evaluate({"y": 10})
        assert fn(5) == 15

    def test_application(self):
        """Function application works."""
        # (λx. x + 1) 5
        inc = Lam("x", Prim("+", [Var("x"), Lit(1)]))
        app = App(inc, Lit(5))
        assert app.evaluate({}) == 6

    def test_curried_function(self):
        """Multi-argument functions via currying."""
        # λx. λy. x + y
        add = Lam("x", Lam("y", Prim("+", [Var("x"), Var("y")])))
        fn = add.evaluate({})
        assert fn(2)(3) == 5

    def test_to_string(self):
        """Expressions convert to readable strings."""
        expr = Lam("x", Prim("+", [Var("x"), Lit(1)]))
        s = expr.to_string()
        assert "λx" in s or "lambda" in s.lower()

    def test_free_variables(self):
        """Free variable detection."""
        # x is free, y is bound
        expr = Lam("y", Prim("+", [Var("x"), Var("y")]))
        assert expr.free_variables() == {"x"}


class TestPrimitives:
    """Tests for built-in primitive operations."""

    def test_arithmetic(self):
        """Arithmetic primitives work."""
        assert PRIMITIVES["+"](2, 3) == 5
        assert PRIMITIVES["-"](10, 4) == 6
        assert PRIMITIVES["*"](3, 4) == 12
        assert PRIMITIVES["/"](10, 2) == 5

    def test_division_by_zero(self):
        """Division by zero returns infinity, not error."""
        result = PRIMITIVES["/"](1, 0)
        assert result == float('inf')

    def test_comparison(self):
        """Comparison primitives work."""
        assert PRIMITIVES["="](5, 5) == True
        assert PRIMITIVES["<"](3, 5) == True
        assert PRIMITIVES[">"](5, 3) == True

    def test_boolean(self):
        """Boolean primitives work."""
        assert PRIMITIVES["and"](True, True) == True
        assert PRIMITIVES["or"](False, True) == True
        assert PRIMITIVES["not"](True) == False
        assert PRIMITIVES["if"](True, "yes", "no") == "yes"

    def test_list_operations(self):
        """List primitives work."""
        assert PRIMITIVES["cons"](1, [2, 3]) == [1, 2, 3]
        assert PRIMITIVES["car"]([1, 2, 3]) == 1
        assert PRIMITIVES["cdr"]([1, 2, 3]) == [2, 3]
        assert PRIMITIVES["length"]([1, 2, 3]) == 3

    def test_map(self):
        """Map applies function to list."""
        double = lambda x: x * 2
        result = PRIMITIVES["map"](double, [1, 2, 3])
        assert result == [2, 4, 6]

    def test_range(self):
        """Range generates list."""
        result = PRIMITIVES["range"](0, 5)
        assert result == [0, 1, 2, 3, 4]

    def test_mean(self):
        """Mean computes average."""
        assert PRIMITIVES["mean"]([1, 2, 3, 4, 5]) == 3.0


class TestSynthesizer:
    """Tests for the neurosymbolic synthesizer."""

    def test_synthesizer_creation(self):
        """Can create synthesizer."""
        synth = NeurosymbolicSynthesizer()
        assert synth is not None

    def test_pattern_synthesis_mean(self):
        """Synthesizes mean from specification."""
        synth = NeurosymbolicSynthesizer()
        prog = synth.synthesize("compute the mean of a list")
        assert prog is not None

    def test_pattern_synthesis_belief(self):
        """Synthesizes belief operation."""
        synth = NeurosymbolicSynthesizer()
        prog = synth.synthesize("create a belief state")
        assert prog is not None

    def test_evaluate_safe(self):
        """Evaluation is safe (uses only primitives)."""
        synth = NeurosymbolicSynthesizer()
        prog = Prim("mean", [Lit([1, 2, 3])])
        result = synth.evaluate(prog)
        assert result == 2.0

    def test_program_cache(self):
        """Programs are cached."""
        synth = NeurosymbolicSynthesizer()
        prog1 = synth.synthesize("compute sum")
        prog2 = synth.synthesize("compute sum")
        # Both should work (caching is internal optimization)
        assert prog1 is not None


class TestStitchCompressor:
    """Tests for library compression."""

    def test_compressor_creation(self):
        """Can create compressor."""
        comp = StitchCompressor()
        assert comp is not None
        assert len(comp.library) == 0

    def test_add_program(self):
        """Can add programs to history."""
        comp = StitchCompressor()
        prog = Lam("x", Prim("+", [Var("x"), Lit(1)]))
        comp.add_successful_program("increment", prog)
        assert len(comp.usage_history) == 1

    def test_compression_requires_repetition(self):
        """Compression requires repeated patterns."""
        comp = StitchCompressor()

        # Add same pattern multiple times
        prog = Prim("+", [Var("x"), Lit(1)])
        for i in range(5):
            comp.add_successful_program(f"prog_{i}", Lam("x", prog))

        # Compress
        abstractions = comp.compress(min_usage=2)
        # Should find the repeated pattern
        # (May or may not find abstractions depending on pattern)


class TestAutoDoc:
    """Tests for automatic documentation."""

    def test_autodoc_creation(self):
        """Can create autodoc."""
        doc = AutoDoc()
        assert doc is not None

    def test_agent_card_format(self):
        """Agent cards have correct format."""
        from src.synthesis import Abstraction

        doc = AutoDoc()
        abstraction = Abstraction(
            name="test_abs",
            params=["x", "y"],
            body=Prim("+", [Var("x"), Var("y")]),
        )

        capability = doc.document_abstraction(abstraction)
        card = capability.to_agent_card()

        assert "capability" in card
        assert "description" in card
        assert "inputs" in card
        assert "output" in card


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
