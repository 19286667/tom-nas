# ADR-001: Lambda Calculus for Agent Program Synthesis

## Status
Accepted

## Context
Agents in ToM-NAS need to write and execute code to conduct experiments. The initial approach used sandboxed Python execution with AST validation and restricted builtins. This created several problems:

1. **Security by exclusion**: Blocking dangerous operations is error-prone; new attack vectors emerge
2. **Complexity**: Subprocess isolation, signal handlers, memory limits add operational burden
3. **Auditability**: Difficult to formally verify Python programs are safe
4. **Composability**: No guarantee that combining safe programs yields safe programs

## Decision
Replace sandboxed Python with lambda calculus-based program synthesis.

Programs are composed exclusively from a whitelist of pure primitives:
- Arithmetic: `+`, `-`, `*`, `/`
- Logic: `and`, `or`, `not`, `if`
- Collections: `map`, `filter`, `fold`, `cons`, `car`, `cdr`
- Domain-specific: `believe`, `simulate`, `observe`

## Consequences

### Positive
- **Security by construction**: Cannot express I/O, file access, or network operations
- **Formal verification**: Lambda calculus has well-understood semantics
- **Composability**: Pure functions compose safely
- **Library learning**: Stitch compression extracts reusable abstractions

### Negative
- **Expressiveness**: Some computations harder to express than in Python
- **Performance**: Interpretation overhead vs compiled code
- **Learning curve**: Developers must understand functional paradigm

### Mitigations
- Provide rich primitive library for common operations
- Optimize hot paths with memoization
- Document patterns and provide examples

## References
- Lilo: Library Induction from Language Observations
- DreamCoder: Growing Generalizable, Interpretable Knowledge
- Church, A. (1936). An unsolvable problem of elementary number theory
