"""
LLM Integration Layer

Provides unified interface to LLM backends for neurosymbolic synthesis.
Supports Google Vertex AI (primary), local models, and fallback logic.

This is the bridge between neural (LLM proposals) and symbolic (λ-calculus verification).
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class SynthesisRequest:
    """Request for program synthesis."""
    specification: str
    examples: Optional[List[Dict[str, Any]]] = None
    constraints: List[str] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.7
    num_candidates: int = 3


@dataclass
class SynthesisCandidate:
    """A candidate program from synthesis."""
    source: str
    confidence: float
    reasoning: str = ""
    valid: bool = False
    error: Optional[str] = None


class LLMBackend(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from prompt."""
        pass

    @abstractmethod
    async def synthesize_program(self, request: SynthesisRequest) -> List[SynthesisCandidate]:
        """Synthesize candidate programs from specification."""
        pass


class VertexAIBackend(LLMBackend):
    """
    Google Vertex AI backend.

    Uses Gemini models for program synthesis.
    Requires GOOGLE_CLOUD_PROJECT and appropriate credentials.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model: str = "gemini-1.5-pro",
    ):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Vertex AI client."""
        if self._client is None:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel

                vertexai.init(project=self.project_id, location=self.location)
                self._client = GenerativeModel(self.model)
                logger.info(f"Initialized Vertex AI with model {self.model}")
            except ImportError:
                logger.warning("vertexai package not installed, using fallback")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI: {e}")
                return None
        return self._client

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion using Gemini."""
        client = self._get_client()
        if client is None:
            return self._fallback_generate(prompt)

        try:
            response = await asyncio.to_thread(
                client.generate_content,
                prompt,
                generation_config={
                    "max_output_tokens": kwargs.get("max_tokens", 1024),
                    "temperature": kwargs.get("temperature", 0.7),
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Vertex AI generation failed: {e}")
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        """Fallback when Vertex AI unavailable."""
        # Return a simple lambda expression based on keywords
        if "experiment" in prompt.lower():
            return "(λconfig. (mean (map (λx. (+ x 1)) (range 0 100))))"
        elif "hypothesis" in prompt.lower():
            return "(λobservations. (> (mean observations) 0.5))"
        elif "simulate" in prompt.lower():
            return "(λagent. (believe agent 'hypothesis' 0.7))"
        else:
            return "(λx. x)"  # Identity function

    async def synthesize_program(self, request: SynthesisRequest) -> List[SynthesisCandidate]:
        """Synthesize λ-calculus programs from specification."""

        prompt = self._build_synthesis_prompt(request)
        candidates = []

        for i in range(request.num_candidates):
            try:
                response = await self.generate(
                    prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature + (i * 0.1),  # Increase diversity
                )

                candidate = self._parse_synthesis_response(response)
                candidates.append(candidate)

            except Exception as e:
                logger.error(f"Synthesis attempt {i} failed: {e}")
                candidates.append(SynthesisCandidate(
                    source="(λx. x)",
                    confidence=0.0,
                    error=str(e)
                ))

        return candidates

    def _build_synthesis_prompt(self, request: SynthesisRequest) -> str:
        """Build prompt for program synthesis."""

        prompt = f"""You are a program synthesizer that generates lambda calculus expressions.

TASK: {request.specification}

AVAILABLE PRIMITIVES:
- Arithmetic: +, -, *, /, mod, abs, min, max
- Comparison: =, <, >, <=, >=, !=
- Logic: and, or, not, if
- Lists: cons, car, cdr, null?, map, filter, fold, range, length, nth
- Math: mean, sum, variance, sqrt, exp, log, sin, cos
- Theory of Mind: believe, doubt, infer, predict, simulate

CONSTRAINTS:
- Output must be a valid lambda calculus expression
- No side effects (pure functions only)
- No I/O operations
{chr(10).join(f'- {c}' for c in request.constraints)}

"""

        if request.examples:
            prompt += "\nEXAMPLES:\n"
            for ex in request.examples[:5]:
                prompt += f"  Input: {ex.get('input')} → Output: {ex.get('output')}\n"

        prompt += """
OUTPUT FORMAT:
```lambda
<your lambda expression here>
```

REASONING:
<brief explanation of your approach>

Generate a lambda calculus expression that satisfies the specification:"""

        return prompt

    def _parse_synthesis_response(self, response: str) -> SynthesisCandidate:
        """Parse LLM response into synthesis candidate."""

        # Extract lambda expression
        source = "(λx. x)"  # Default
        reasoning = ""

        # Try to find code block
        if "```lambda" in response:
            start = response.find("```lambda") + 9
            end = response.find("```", start)
            if end > start:
                source = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                source = response[start:end].strip()
        elif "(λ" in response or "(lambda" in response:
            # Try to extract inline expression
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("(λ") or line.startswith("(lambda"):
                    source = line
                    break

        # Extract reasoning
        if "REASONING:" in response:
            reasoning = response.split("REASONING:")[-1].strip()

        # Normalize lambda syntax
        source = source.replace("lambda", "λ")

        return SynthesisCandidate(
            source=source,
            confidence=0.7,  # Will be updated by verification
            reasoning=reasoning,
        )


class LocalModelBackend(LLMBackend):
    """
    Local model backend for offline operation.

    Uses smaller models that can run on CPU.
    Good for development and testing.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load local model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = self.model_path or "microsoft/phi-2"
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
            )
            logger.info(f"Loaded local model: {model_id}")
        except ImportError:
            logger.warning("transformers not installed")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using local model."""
        self._load_model()

        if self._model is None:
            return self._template_generate(prompt)

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=True,
            )
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            return self._template_generate(prompt)

    def _template_generate(self, prompt: str) -> str:
        """Template-based generation when model unavailable."""
        # Pattern matching on common requests
        templates = {
            "experiment": "(λconfig. (let ((data (range 0 100))) (mean (map (λx. (* x (get config 'scale' 1))) data))))",
            "hypothesis": "(λobs. (> (mean obs) (* (variance obs) 0.5)))",
            "test": "(λh. (λdata. (if (h data) 'supported' 'refuted')))",
            "simulate": "(λagent. (λsteps. (fold (λacc. (λ_. (update acc agent))) (range 0 steps) (init-state))))",
            "belief": "(λagent. (λprop. (λconf. (believe agent prop conf))))",
            "predict": "(λmodel. (λinput. (apply model input)))",
        }

        prompt_lower = prompt.lower()
        for key, template in templates.items():
            if key in prompt_lower:
                return template

        return "(λx. x)"

    async def synthesize_program(self, request: SynthesisRequest) -> List[SynthesisCandidate]:
        """Synthesize using local model or templates."""

        candidates = []

        # Generate using template matching
        source = self._template_generate(request.specification)
        candidates.append(SynthesisCandidate(
            source=source,
            confidence=0.5,
            reasoning="Template-based synthesis",
        ))

        # Try variations
        for i in range(min(request.num_candidates - 1, 2)):
            variant = self._generate_variant(source, i)
            candidates.append(SynthesisCandidate(
                source=variant,
                confidence=0.3,
                reasoning=f"Variant {i+1}",
            ))

        return candidates

    def _generate_variant(self, base: str, variant_idx: int) -> str:
        """Generate variant of base expression."""
        # Simple modifications
        variants = [
            lambda s: s.replace("mean", "sum"),
            lambda s: s.replace("100", "50"),
            lambda s: s.replace("0.5", "0.7"),
        ]

        if variant_idx < len(variants):
            return variants[variant_idx](base)
        return base


class LLMIntegration:
    """
    Main integration class for LLM-powered synthesis.

    Manages backends, caching, and verification pipeline.
    """

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        cache_size: int = 100,
    ):
        self.backend = backend or self._auto_select_backend()
        self._cache: Dict[str, List[SynthesisCandidate]] = {}
        self._cache_size = cache_size

        # Verification callback
        self._verifier: Optional[Callable[[str], bool]] = None

    def _auto_select_backend(self) -> LLMBackend:
        """Auto-select best available backend."""

        # Try Vertex AI first (if on GCP)
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            logger.info("Using Vertex AI backend")
            return VertexAIBackend()

        # Fall back to local
        logger.info("Using local model backend")
        return LocalModelBackend()

    def set_verifier(self, verifier: Callable[[str], bool]):
        """Set verification callback for validating programs."""
        self._verifier = verifier

    async def synthesize(
        self,
        specification: str,
        examples: Optional[List[Dict[str, Any]]] = None,
        constraints: Optional[List[str]] = None,
    ) -> Optional[SynthesisCandidate]:
        """
        Synthesize a verified program from specification.

        Returns the best valid candidate, or None if synthesis fails.
        """

        # Check cache
        cache_key = f"{specification}:{json.dumps(examples or [])}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            valid = [c for c in cached if c.valid]
            if valid:
                return max(valid, key=lambda c: c.confidence)

        # Create request
        request = SynthesisRequest(
            specification=specification,
            examples=examples,
            constraints=constraints or [],
        )

        # Generate candidates
        candidates = await self.backend.synthesize_program(request)

        # Verify candidates
        for candidate in candidates:
            if self._verifier:
                try:
                    candidate.valid = self._verifier(candidate.source)
                    if candidate.valid:
                        candidate.confidence = min(1.0, candidate.confidence + 0.2)
                except Exception as e:
                    candidate.valid = False
                    candidate.error = str(e)
            else:
                # No verifier, assume valid if parseable
                candidate.valid = self._basic_validation(candidate.source)

        # Cache results
        self._cache[cache_key] = candidates
        if len(self._cache) > self._cache_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        # Return best valid candidate
        valid_candidates = [c for c in candidates if c.valid]
        if valid_candidates:
            return max(valid_candidates, key=lambda c: c.confidence)

        return None

    def _basic_validation(self, source: str) -> bool:
        """Basic syntactic validation."""
        # Check balanced parentheses
        depth = 0
        for char in source:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            if depth < 0:
                return False

        if depth != 0:
            return False

        # Check for lambda
        return "λ" in source or "lambda" in source or source.startswith("(")

    async def batch_synthesize(
        self,
        specifications: List[str],
    ) -> List[Optional[SynthesisCandidate]]:
        """Synthesize multiple programs in parallel."""
        tasks = [self.synthesize(spec) for spec in specifications]
        return await asyncio.gather(*tasks)


# Singleton instance
_integration: Optional[LLMIntegration] = None


def get_llm_integration() -> LLMIntegration:
    """Get or create the global LLM integration instance."""
    global _integration
    if _integration is None:
        _integration = LLMIntegration()
    return _integration


# Convenience functions
async def synthesize_program(
    specification: str,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Synthesize a program from specification."""
    integration = get_llm_integration()
    candidate = await integration.synthesize(specification, examples)
    return candidate.source if candidate else None
