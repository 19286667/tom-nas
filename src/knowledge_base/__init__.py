"""
Semiotic Knowledge Graph - Indra's Net Implementation

This module implements the Omnipresent Semantic Web where every entity exists
in a superposition of latent meanings collapsed by context. The physical is
cognitive - every object, action, and utterance is a node in a hyperlinked
semantic web based on the 80-Dimension Taxonomy.

Key Components:
- SemanticNode: Base node in the semiotic web
- IndrasNet: The main graph database implementing the semantic substrate
- TaxonomyLoader: Ingests taxonomies (Mundane Life, Institutions, Aesthetics)
- SemanticQueryEngine: Traverses the graph to activate associated concepts

Theoretical Foundation:
- Symbol Grounding Problem (Harnad): Symbols anchored to physical referents
- Semantic Prototype Theory: Entities exist as prototype-stereotype pairs
- Conceptual Metaphor Theory (Lakoff): Physical-cognitive isomorphism

Author: ToM-NAS Project
"""

from .indras_net import IndrasNet
from .query_engine import SemanticQueryEngine
from .schemas import (
    ActivationContext,
    ConceptualDomain,
    SemanticActivation,
    SemanticEdge,
    SemanticNode,
    TaxonomyDimension,
)
from .taxonomy import (
    AestheticTaxonomy,
    FullTaxonomy,
    InstitutionalTaxonomy,
    MundaneTaxonomy,
    TaxonomyLayer,
)

__all__ = [
    # Core schemas
    "SemanticNode",
    "SemanticEdge",
    "TaxonomyDimension",
    "ConceptualDomain",
    "ActivationContext",
    "SemanticActivation",
    # Graph database
    "IndrasNet",
    # Taxonomies
    "TaxonomyLayer",
    "MundaneTaxonomy",
    "InstitutionalTaxonomy",
    "AestheticTaxonomy",
    "FullTaxonomy",
    # Query engine
    "SemanticQueryEngine",
]
