"""
Agent architectures module for ToM-NAS.

Contains three main neural architectures:
- TransparentRNN (TRN): Interpretable recurrent network
- RecursiveSelfAttention (RSAN): Emergent recursive reasoning through attention
- TransformerToMAgent: Communication and pragmatic reasoning
- HybridArchitecture: Evolutionary combination of architectures
"""

from .architectures import (
    TransparentRNN,
    RecursiveSelfAttention,
    TransformerToMAgent,
    HybridArchitecture,
)

__all__ = [
    "TransparentRNN",
    "RecursiveSelfAttention",
    "TransformerToMAgent",
    "HybridArchitecture",
]
