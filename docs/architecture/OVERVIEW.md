# ToM-NAS Architecture Overview

## System Context

ToM-NAS (Theory of Mind - Neural Architecture Search) is a research platform for evolving neural architectures capable of recursive belief modeling. The system enables agents to conduct research by synthesizing verifiable programs, creating nested simulations, and validating hypotheses through rigorous scientific protocols.

## Architecture Principles

This system follows the Google Cloud Well-Architected Framework:

| Pillar | Application |
|--------|-------------|
| Operational Excellence | Structured logging, health endpoints, GitOps deployment |
| Security | Lambda calculus intrinsic safety, no arbitrary code execution |
| Reliability | Stateless compute, graceful degradation, circuit breakers |
| Cost Optimization | Tiered requirements, scale-to-zero, spot instances for training |
| Performance | Zero-cost proxies for architecture filtering, lazy evaluation |

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   REST API      │  │   Streamlit     │  │   CLI Tools     │             │
│  │   (FastAPI)     │  │   Dashboard     │  │   (tom-train)   │             │
│  │   Port 8080     │  │   Port 8501     │  │                 │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
└───────────┼─────────────────────┼─────────────────────┼─────────────────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                          SERVICE LAYER                                       │
├─────────────────────────────────┼───────────────────────────────────────────┤
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     VERIFICATION GATEWAY                             │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │    NSHE     │  │   PIMMUR    │  │     PAN     │                  │    │
│  │  │  (Energy)   │  │ (Validity)  │  │(Simulation) │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                            │
│  ┌──────────────────────────────┼──────────────────────────────────────┐    │
│  │                    SYNTHESIS ENGINE                                  │    │
│  │                              ▼                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │  λ-Calculus │  │   Stitch    │  │   AutoDoc   │                  │    │
│  │  │    Core     │──│ Compression │──│  (A2A)      │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                 │                                            │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                          DOMAIN LAYER                                        │
├─────────────────────────────────┼───────────────────────────────────────────┤
│                                 ▼                                            │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐        │
│  │    INSTITUTIONS   │  │  RESEARCHER       │  │   RECURSIVE       │        │
│  │                   │  │  AGENTS           │  │   SIMULATIONS     │        │
│  │  • ResearchLab    │  │                   │  │                   │        │
│  │  • CorporateRD    │  │  • Beliefs        │  │  • WorldFactory   │        │
│  │  • Government     │  │  • Hypotheses     │  │  • Nested Agents  │        │
│  │  • Network        │  │  • Publications   │  │  • Emergence      │        │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘        │
│                                 │                                            │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                          FOUNDATION LAYER                                    │
├─────────────────────────────────┼───────────────────────────────────────────┤
│                                 ▼                                            │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐        │
│  │   CORE ToM        │  │   NEURAL          │  │   EVOLUTION       │        │
│  │                   │  │   ARCHITECTURES   │  │   ENGINE          │        │
│  │  • SoulMap (181d) │  │                   │  │                   │        │
│  │  • Beliefs (5th)  │  │  • TRN            │  │  • NAS            │        │
│  │  • Events         │  │  • RSAN           │  │  • Fitness        │        │
│  │                   │  │  • Transformer    │  │  • Zero-cost      │        │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘        │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                    CONFIGURATION & OBSERVABILITY                   │      │
│  │  • Constants (single source of truth)                             │      │
│  │  • Settings (environment-aware)                                   │      │
│  │  • Logging (structured, Cloud Logging compatible)                 │      │
│  │  • Metrics (Prometheus)                                           │      │
│  └───────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Hypothesis Generation Flow

```
Observation → Agent.form_hypothesis() → λ-expression synthesis
                                              ↓
                              ConstraintPropagator (energy check)
                                              ↓
                              StitchCompressor (pattern extraction)
                                              ↓
                              AutoDoc (capability card)
                                              ↓
                              Publication (to network)
```

### Verification Flow

```
Hypothesis → EnergyLandscape.compute_energy()
                    ↓
          ┌────────┴────────┐
          ↓                 ↓
    Hard Constraints    Soft Constraints
    (must pass)         (weighted score)
          ↓                 ↓
          └────────┬────────┘
                   ↓
          PIMMURValidator.validate()
                   ↓
          PANSimulator.simulate_trajectory()
                   ↓
          VerificationMetrics (CSI, TQS, NUTC)
```

## Stateless Design

All compute components are stateless:

| Component | State Location | Rationale |
|-----------|----------------|-----------|
| API Server | None (request-scoped) | Horizontal scaling |
| Agents | Serializable belief state | Can be checkpointed/restored |
| Simulations | In-memory during run | Results persisted to storage |
| Training | Checkpoints to GCS | Resume from any point |

## Deployment Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     Google Cloud Platform                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │   Cloud Run     │     │   Cloud Run     │                    │
│  │   (API)         │     │   (Dashboard)   │                    │
│  │   min: 0        │     │   min: 0        │                    │
│  │   max: 10       │     │   max: 5        │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       │                                          │
│                       ▼                                          │
│           ┌─────────────────────┐                               │
│           │   Cloud Storage     │                               │
│           │   (Checkpoints)     │                               │
│           └─────────────────────┘                               │
│                       │                                          │
│                       ▼                                          │
│           ┌─────────────────────┐                               │
│           │   Cloud Logging     │                               │
│           │   Cloud Monitoring  │                               │
│           └─────────────────────┘                               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Training (On-Demand)                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │ Compute     │  │ Compute     │  │ Compute     │      │    │
│  │  │ Engine      │  │ Engine      │  │ Engine      │      │    │
│  │  │ (Spot)      │  │ (Spot)      │  │ (Spot)      │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Security Model

### Defense in Depth

```
Layer 1: Network
├── Cloud Run IAM (authentication)
├── VPC Service Controls (optional)
└── Cloud Armor (DDoS protection)

Layer 2: Application
├── Lambda calculus (no arbitrary execution)
├── Primitive whitelist (only safe operations)
└── Input validation (Pydantic models)

Layer 3: Data
├── Encryption at rest (GCS default)
├── Encryption in transit (TLS 1.3)
└── Secret Manager (credentials)
```

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Arbitrary code execution | Lambda calculus can only compose safe primitives |
| Data exfiltration | No network primitives in λ-calculus |
| Resource exhaustion | Evaluation limits, Cloud Run CPU/memory caps |
| Model poisoning | Verification framework (NSHE/PIMMUR/PAN) |

## Key Design Decisions

See `/docs/adr/` for detailed Architecture Decision Records:

- ADR-001: Lambda calculus over sandboxed Python
- ADR-002: Verification-first hypothesis generation
- ADR-003: Stateless compute with persistent storage
- ADR-004: PIMMUR protocol for agent validity
- ADR-005: Energy-based constraint propagation
