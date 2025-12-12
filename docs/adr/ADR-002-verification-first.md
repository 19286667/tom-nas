# ADR-002: Verification-First Hypothesis Generation

## Status
Accepted

## Context
Traditional AI systems generate outputs then filter invalid ones. This is wasteful:
- Resources spent on invalid hypotheses
- Post-hoc filtering misses subtle constraint violations
- No guarantee filtering catches all problems

Scientific hypothesis generation requires:
- Adherence to physical laws (conservation, causality)
- Internal consistency of beliefs
- Testability and reproducibility

## Decision
Implement real-time constraint propagation that modulates neural activations during generation, not after.

Three verification layers:
1. **NSHE (Energy-Based)**: Constraint satisfaction during forward pass
2. **PIMMUR**: Agent validity verification
3. **PAN**: Simulative reasoning before execution

Hypotheses that would violate constraints are suppressed before they fully form.

## Consequences

### Positive
- **Efficiency**: No wasted computation on invalid hypotheses
- **Correctness**: Guarantees at generation time, not validation time
- **Interpretability**: Energy landscape shows why hypotheses were/weren't generated
- **Scientific rigor**: Aligns with how actual scientific constraints work

### Negative
- **Complexity**: More sophisticated generation pipeline
- **Constraint engineering**: Must correctly specify domain constraints
- **Potential over-constraint**: May suppress novel valid hypotheses

### Mitigations
- Soft constraints allow gradation (not binary filtering)
- Constraint weights are tunable
- Energy landscape is inspectable for debugging

## References
- Temporally-Grounded Constraint Propagation
- Energy-Based Models for structured prediction
- PIMMUR protocol for agent validity
