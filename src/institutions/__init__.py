"""
Computational Sociology: Institutional Agent Framework

This module implements agents as researchers within institutional contexts,
capable of writing and executing real code, creating recursive simulations,
and generating emergent dimensionality through nested self-reference.

The key insight: agents that simulate agents that simulate agents create
selective pressure for discovering genuine Theory of Mind and abductive
reasoning - because only agents with true ToM can effectively model
the reasoning of other modeling agents.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    COMPUTATIONAL SOCIOLOGY                       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  INSTITUTIONS          AGENTS              ARTIFACTS             │
    │  ┌──────────┐         ┌──────────┐        ┌──────────┐          │
    │  │ Research │◄───────►│Researcher│───────►│   Code   │          │
    │  │   Lab    │         │  Agent   │        │ Artifact │          │
    │  └──────────┘         └────┬─────┘        └────┬─────┘          │
    │  ┌──────────┐              │                   │                │
    │  │Corporate │              │ writes            │ executes       │
    │  │   R&D    │              ▼                   ▼                │
    │  └──────────┘         ┌──────────┐        ┌──────────┐          │
    │  ┌──────────┐         │Simulation│◄──────►│ Sandbox  │          │
    │  │Government│         │  Model   │        │ Runtime  │          │
    │  │  Agency  │         └──────────┘        └──────────┘          │
    │  └──────────┘              │                                    │
    │                            │ contains                           │
    │                            ▼                                    │
    │                       ┌──────────┐                              │
    │                       │  Nested  │ ← Recursive!                 │
    │                       │  Agents  │                              │
    │                       └──────────┘                              │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Key Principle: Emergent Dimensionality
    When agents can create simulations containing agents that themselves
    create simulations, the representational capacity of the inner worlds
    can exceed that of the outer world through compression and abstraction.
    This mirrors how human minds create models richer than raw sensory input.
"""

from .institutions import (
    Institution,
    ResearchLab,
    CorporateRD,
    GovernmentAgency,
    InstitutionalNetwork,
)
from .researcher_agent import (
    ResearcherAgent,
    ResearchAgenda,
    Publication,
    CodeArtifact,
)
from .code_executor import (
    SandboxedExecutor,
    ExecutionResult,
    CodeValidator,
)
from .recursive_world import (
    RecursiveSimulation,
    WorldFactory,
    EmergentDimensionality,
)

__all__ = [
    # Institutions
    'Institution',
    'ResearchLab',
    'CorporateRD',
    'GovernmentAgency',
    'InstitutionalNetwork',
    # Agents
    'ResearcherAgent',
    'ResearchAgenda',
    'Publication',
    'CodeArtifact',
    # Execution
    'SandboxedExecutor',
    'ExecutionResult',
    'CodeValidator',
    # Recursion
    'RecursiveSimulation',
    'WorldFactory',
    'EmergentDimensionality',
]
