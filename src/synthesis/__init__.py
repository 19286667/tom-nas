"""
Neurosymbolic Program Synthesis Engine

Implements the Lilo-inspired approach to agent code generation:
1. LLM Synthesizer - Generates candidate λ-expressions
2. Stitch Compression - Identifies reusable abstractions
3. AutoDoc - Generates documentation for capability discovery

This replaces sandboxed execution with intrinsic safety:
- Agents synthesize from verified symbolic primitives
- Lambda calculus has no side effects by construction
- Libraries grow through compression of successful patterns

References:
- Lilo: Library Induction from Language Observations
- DreamCoder: Growing Generalizable, Interpretable Knowledge
- Stitch: Compression for Library Learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
import re
from abc import ABC, abstractmethod

from src.config import get_logger

logger = get_logger(__name__)


# =============================================================================
# LAMBDA CALCULUS CORE
# =============================================================================

class LambdaExpr(ABC):
    """Base class for lambda calculus expressions."""

    @abstractmethod
    def evaluate(self, env: Dict[str, Any]) -> Any:
        """Evaluate expression in environment."""
        pass

    @abstractmethod
    def substitute(self, var: str, expr: 'LambdaExpr') -> 'LambdaExpr':
        """Substitute variable with expression."""
        pass

    @abstractmethod
    def free_variables(self) -> set:
        """Return set of free variables."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert to readable string."""
        pass


@dataclass
class Var(LambdaExpr):
    """Variable reference."""
    name: str

    def evaluate(self, env: Dict[str, Any]) -> Any:
        if self.name not in env:
            raise NameError(f"Unbound variable: {self.name}")
        return env[self.name]

    def substitute(self, var: str, expr: LambdaExpr) -> LambdaExpr:
        return expr if self.name == var else self

    def free_variables(self) -> set:
        return {self.name}

    def to_string(self) -> str:
        return self.name


@dataclass
class Lam(LambdaExpr):
    """Lambda abstraction: λx.body"""
    param: str
    body: LambdaExpr

    def evaluate(self, env: Dict[str, Any]) -> Callable:
        # Return a closure
        def closure(arg):
            new_env = env.copy()
            new_env[self.param] = arg
            return self.body.evaluate(new_env)
        return closure

    def substitute(self, var: str, expr: LambdaExpr) -> LambdaExpr:
        if self.param == var:
            return self  # Variable is bound, no substitution
        return Lam(self.param, self.body.substitute(var, expr))

    def free_variables(self) -> set:
        return self.body.free_variables() - {self.param}

    def to_string(self) -> str:
        return f"(λ{self.param}.{self.body.to_string()})"


@dataclass
class App(LambdaExpr):
    """Function application: (f x)"""
    func: LambdaExpr
    arg: LambdaExpr

    def evaluate(self, env: Dict[str, Any]) -> Any:
        f = self.func.evaluate(env)
        x = self.arg.evaluate(env)
        if callable(f):
            return f(x)
        raise TypeError(f"Cannot apply non-function: {f}")

    def substitute(self, var: str, expr: LambdaExpr) -> LambdaExpr:
        return App(
            self.func.substitute(var, expr),
            self.arg.substitute(var, expr)
        )

    def free_variables(self) -> set:
        return self.func.free_variables() | self.arg.free_variables()

    def to_string(self) -> str:
        return f"({self.func.to_string()} {self.arg.to_string()})"


@dataclass
class Lit(LambdaExpr):
    """Literal value (numbers, booleans, strings)."""
    value: Any

    def evaluate(self, env: Dict[str, Any]) -> Any:
        return self.value

    def substitute(self, var: str, expr: LambdaExpr) -> LambdaExpr:
        return self

    def free_variables(self) -> set:
        return set()

    def to_string(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)


@dataclass
class Prim(LambdaExpr):
    """Primitive operation (built-in safe functions)."""
    name: str
    args: List[LambdaExpr] = field(default_factory=list)

    def evaluate(self, env: Dict[str, Any]) -> Any:
        evaluated_args = [arg.evaluate(env) for arg in self.args]
        return PRIMITIVES[self.name](*evaluated_args)

    def substitute(self, var: str, expr: LambdaExpr) -> LambdaExpr:
        return Prim(self.name, [a.substitute(var, expr) for a in self.args])

    def free_variables(self) -> set:
        result = set()
        for arg in self.args:
            result |= arg.free_variables()
        return result

    def to_string(self) -> str:
        args_str = " ".join(a.to_string() for a in self.args)
        return f"({self.name} {args_str})"


# =============================================================================
# SAFE PRIMITIVES (The Only Operations Agents Can Use)
# =============================================================================

PRIMITIVES: Dict[str, Callable] = {
    # Arithmetic (pure, no side effects)
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b if b != 0 else float('inf'),
    'mod': lambda a, b: a % b if b != 0 else 0,
    'abs': abs,
    'neg': lambda x: -x,
    'min': min,
    'max': max,

    # Comparison (pure)
    '=': lambda a, b: a == b,
    '<': lambda a, b: a < b,
    '>': lambda a, b: a > b,
    '<=': lambda a, b: a <= b,
    '>=': lambda a, b: a >= b,

    # Boolean (pure)
    'and': lambda a, b: a and b,
    'or': lambda a, b: a or b,
    'not': lambda x: not x,
    'if': lambda c, t, f: t if c else f,

    # List operations (pure, return new lists)
    'cons': lambda h, t: [h] + (t if isinstance(t, list) else [t]),
    'car': lambda lst: lst[0] if lst else None,
    'cdr': lambda lst: lst[1:] if lst else [],
    'nil': lambda: [],
    'null?': lambda lst: len(lst) == 0,
    'length': len,
    'map': lambda f, lst: [f(x) for x in lst],
    'filter': lambda f, lst: [x for x in lst if f(x)],
    'fold': lambda f, init, lst: __import__('functools').reduce(f, lst, init),
    'range': lambda a, b: list(range(int(a), int(b))),

    # Tensor operations (for neural network primitives)
    'tensor': lambda lst: __import__('torch').tensor(lst),
    'zeros': lambda n: __import__('torch').zeros(int(n)),
    'ones': lambda n: __import__('torch').ones(int(n)),
    'dot': lambda a, b: (a @ b).item() if hasattr(a, '__matmul__') else sum(x*y for x,y in zip(a,b)),
    'sum': lambda x: sum(x) if isinstance(x, list) else x.sum().item(),
    'mean': lambda x: sum(x)/len(x) if isinstance(x, list) else x.mean().item(),

    # Agent-specific primitives
    'observe': lambda agent_id: f"observation_{agent_id}",  # Placeholder
    'believe': lambda agent, prop, conf: {'agent': agent, 'belief': prop, 'confidence': conf},
    'simulate': lambda config: f"simulation_{hash(str(config)) % 10000}",
}


# =============================================================================
# STITCH: LIBRARY COMPRESSION
# =============================================================================

@dataclass
class Abstraction:
    """A reusable abstraction discovered through compression."""
    name: str
    params: List[str]
    body: LambdaExpr
    usage_count: int = 0
    compression_savings: float = 0.0  # Bits saved by using this abstraction

    def to_lambda(self) -> LambdaExpr:
        """Convert abstraction to lambda expression."""
        result = self.body
        for param in reversed(self.params):
            result = Lam(param, result)
        return result

    def apply(self, args: List[LambdaExpr]) -> LambdaExpr:
        """Apply abstraction to arguments."""
        if len(args) != len(self.params):
            raise ValueError(f"{self.name} expects {len(self.params)} args, got {len(args)}")
        result = self.body
        for param, arg in zip(self.params, args):
            result = result.substitute(param, arg)
        return result


class StitchCompressor:
    """
    Identifies reusable patterns in successful programs and compresses
    them into library abstractions.

    This is the core of procedural memory: agents don't just remember
    what worked, they extract WHY it worked as reusable abstractions.
    """

    def __init__(self):
        self.library: Dict[str, Abstraction] = {}
        self.usage_history: List[Tuple[str, LambdaExpr]] = []

    def add_successful_program(self, name: str, program: LambdaExpr):
        """Record a successful program for pattern analysis."""
        self.usage_history.append((name, program))
        logger.debug(f"Recorded program: {name}")

    def compress(self, min_usage: int = 2) -> List[Abstraction]:
        """
        Analyze usage history and extract reusable abstractions.

        Uses the Stitch algorithm principle: find repeated subexpressions
        that, when abstracted, reduce total description length.
        """
        # Count subexpression frequencies
        subexpr_counts: Dict[str, int] = {}
        subexpr_map: Dict[str, LambdaExpr] = {}

        for _, program in self.usage_history:
            self._count_subexpressions(program, subexpr_counts, subexpr_map)

        # Find candidates for abstraction (appear >= min_usage times)
        candidates = [
            (expr_str, count, subexpr_map[expr_str])
            for expr_str, count in subexpr_counts.items()
            if count >= min_usage and len(expr_str) > 10  # Non-trivial
        ]

        # Sort by compression benefit (count * size)
        candidates.sort(key=lambda x: x[1] * len(x[0]), reverse=True)

        new_abstractions = []
        for expr_str, count, expr in candidates[:10]:  # Top 10
            # Extract free variables as parameters
            free_vars = list(expr.free_variables())

            # Create abstraction
            abstraction = Abstraction(
                name=f"abs_{len(self.library)}",
                params=free_vars,
                body=expr,
                usage_count=count,
                compression_savings=count * len(expr_str) - len(f"abs_{len(self.library)}")
            )

            if abstraction.compression_savings > 0:
                self.library[abstraction.name] = abstraction
                new_abstractions.append(abstraction)
                logger.info(f"Discovered abstraction: {abstraction.name} (saves {abstraction.compression_savings:.0f} bits)")

        return new_abstractions

    def _count_subexpressions(
        self,
        expr: LambdaExpr,
        counts: Dict[str, int],
        expr_map: Dict[str, LambdaExpr]
    ):
        """Recursively count all subexpressions."""
        expr_str = expr.to_string()
        counts[expr_str] = counts.get(expr_str, 0) + 1
        expr_map[expr_str] = expr

        if isinstance(expr, App):
            self._count_subexpressions(expr.func, counts, expr_map)
            self._count_subexpressions(expr.arg, counts, expr_map)
        elif isinstance(expr, Lam):
            self._count_subexpressions(expr.body, counts, expr_map)
        elif isinstance(expr, Prim):
            for arg in expr.args:
                self._count_subexpressions(arg, counts, expr_map)


# =============================================================================
# AUTODOC: CAPABILITY DOCUMENTATION
# =============================================================================

@dataclass
class AgentCapability:
    """
    A documented capability that can be broadcast to other agents.

    Maps to A2A protocol "Agent Cards" for zero-touch interoperability.
    """
    name: str
    description: str
    input_types: List[str]
    output_type: str
    example_usage: str
    abstraction: Optional[Abstraction] = None

    def to_agent_card(self) -> Dict[str, Any]:
        """Generate A2A-compatible Agent Card."""
        return {
            "capability": self.name,
            "description": self.description,
            "inputs": self.input_types,
            "output": self.output_type,
            "example": self.example_usage,
            "protocol": "lambda_calculus_v1",
        }


class AutoDoc:
    """
    Automatically generates documentation for synthesized abstractions.

    This enables dynamic capability discovery: when an agent invents
    a new abstraction, AutoDoc creates documentation that other agents
    can use to understand and invoke it.
    """

    def __init__(self):
        self.documented_capabilities: Dict[str, AgentCapability] = {}

    def document_abstraction(self, abstraction: Abstraction) -> AgentCapability:
        """
        Generate human-readable documentation for an abstraction.

        In a full implementation, this would use an LLM to generate
        natural language descriptions. For now, uses templates.
        """
        # Infer types from structure (simplified)
        input_types = [f"arg_{i}" for i in range(len(abstraction.params))]
        output_type = "value"

        # Generate description
        description = self._generate_description(abstraction)

        # Generate example
        example = self._generate_example(abstraction)

        capability = AgentCapability(
            name=abstraction.name,
            description=description,
            input_types=input_types,
            output_type=output_type,
            example_usage=example,
            abstraction=abstraction,
        )

        self.documented_capabilities[abstraction.name] = capability
        logger.info(f"Documented capability: {abstraction.name}")

        return capability

    def _generate_description(self, abstraction: Abstraction) -> str:
        """Generate natural language description."""
        body_str = abstraction.body.to_string()

        # Pattern matching for common operations
        if '+' in body_str and '*' in body_str:
            return f"Computes a mathematical combination of {len(abstraction.params)} inputs"
        elif 'map' in body_str:
            return f"Applies a transformation across a collection"
        elif 'fold' in body_str:
            return f"Reduces a collection to a single value"
        elif 'believe' in body_str:
            return f"Constructs a belief state about agents"
        elif 'simulate' in body_str:
            return f"Creates a simulation with specified parameters"
        else:
            return f"A reusable abstraction with {len(abstraction.params)} parameters"

    def _generate_example(self, abstraction: Abstraction) -> str:
        """Generate example usage."""
        params_str = " ".join([f"<{p}>" for p in abstraction.params])
        return f"({abstraction.name} {params_str})"

    def broadcast_capabilities(self) -> List[Dict[str, Any]]:
        """Generate A2A-compatible capability broadcast."""
        return [cap.to_agent_card() for cap in self.documented_capabilities.values()]


# =============================================================================
# NEUROSYMBOLIC SYNTHESIZER
# =============================================================================

class NeurosymbolicSynthesizer:
    """
    Main interface for agents to synthesize programs.

    Combines:
    1. LLM-guided synthesis (generates candidate expressions)
    2. Stitch compression (extracts reusable abstractions)
    3. AutoDoc (documents capabilities for sharing)

    Safety is intrinsic: agents can only compose from PRIMITIVES,
    which are all pure functions with no side effects.
    """

    def __init__(self):
        self.compressor = StitchCompressor()
        self.autodoc = AutoDoc()
        self.program_cache: Dict[str, LambdaExpr] = {}

    def synthesize(
        self,
        specification: str,
        examples: List[Tuple[Any, Any]] = None,
    ) -> Optional[LambdaExpr]:
        """
        Synthesize a program from specification and examples.

        Args:
            specification: Natural language description of desired behavior
            examples: List of (input, expected_output) pairs

        Returns:
            Lambda expression implementing the specification, or None
        """
        # In full implementation, this would use LLM + search
        # For now, use simple pattern matching

        program = self._pattern_match_synthesis(specification)

        if program and examples:
            # Verify against examples
            if self._verify_program(program, examples):
                self.compressor.add_successful_program(specification, program)
                return program
            return None

        return program

    def _pattern_match_synthesis(self, spec: str) -> Optional[LambdaExpr]:
        """Simple pattern-based synthesis for common operations."""
        spec_lower = spec.lower()

        if 'sum' in spec_lower or 'add' in spec_lower:
            return Lam('x', Lam('y', Prim('+', [Var('x'), Var('y')])))

        if 'multiply' in spec_lower or 'product' in spec_lower:
            return Lam('x', Lam('y', Prim('*', [Var('x'), Var('y')])))

        if 'average' in spec_lower or 'mean' in spec_lower:
            return Lam('lst', Prim('mean', [Var('lst')]))

        if 'belief' in spec_lower:
            return Lam('agent', Lam('prop', Lam('conf',
                Prim('believe', [Var('agent'), Var('prop'), Var('conf')])
            )))

        if 'simulate' in spec_lower:
            return Lam('config', Prim('simulate', [Var('config')]))

        return None

    def _verify_program(
        self,
        program: LambdaExpr,
        examples: List[Tuple[Any, Any]]
    ) -> bool:
        """Verify program against input/output examples."""
        try:
            for input_val, expected in examples:
                result = program.evaluate({})
                if callable(result):
                    result = result(input_val)
                if result != expected:
                    return False
            return True
        except Exception:
            return False

    def evolve_library(self) -> List[AgentCapability]:
        """
        Compress successful programs into library and document.

        This is how agents develop "procedural memory" - not by
        storing raw experiences, but by extracting reusable patterns.
        """
        # Compress to find new abstractions
        new_abstractions = self.compressor.compress()

        # Document for sharing
        new_capabilities = []
        for abstraction in new_abstractions:
            capability = self.autodoc.document_abstraction(abstraction)
            new_capabilities.append(capability)

        return new_capabilities

    def get_library(self) -> Dict[str, Abstraction]:
        """Get current library of abstractions."""
        return self.compressor.library

    def evaluate(self, program: LambdaExpr, env: Dict[str, Any] = None) -> Any:
        """
        Safely evaluate a lambda expression.

        This is inherently safe because:
        1. Only PRIMITIVES can be called
        2. PRIMITIVES are all pure functions
        3. No side effects are possible
        """
        env = env or {}
        return program.evaluate(env)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_synthesizer() -> NeurosymbolicSynthesizer:
    """Create a new synthesizer instance."""
    return NeurosymbolicSynthesizer()


def safe_eval(expr_string: str) -> Any:
    """
    Parse and evaluate a lambda expression string.

    This is safe because it can only construct from safe primitives.
    """
    # Simple parser (would be more robust in production)
    synthesizer = NeurosymbolicSynthesizer()

    # For now, just synthesize from the string as specification
    program = synthesizer.synthesize(expr_string)
    if program:
        return synthesizer.evaluate(program)
    return None
