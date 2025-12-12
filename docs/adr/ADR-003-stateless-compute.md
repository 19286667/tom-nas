# ADR-003: Stateless Compute with Persistent Storage

## Status
Accepted

## Context
The system has multiple compute components:
- API servers handling inference requests
- Training workers running evolution
- Simulation engines running recursive worlds

Stateful architectures create problems:
- Scaling requires sticky sessions
- Failures lose in-progress work
- Cold starts require state reconstruction

## Decision
All compute is stateless. State is externalized:

| Component | State Strategy |
|-----------|---------------|
| API Server | Request-scoped only; no session state |
| Agent Beliefs | Serializable tensors; checkpoint to GCS |
| Training Progress | Periodic checkpoints; resume from any |
| Simulation State | In-memory during run; results to storage |

Cloud Run enforces this naturally with container recycling.

## Consequences

### Positive
- **Horizontal scaling**: Add instances without coordination
- **Fault tolerance**: Restart from last checkpoint
- **Cost efficiency**: Scale to zero when idle
- **Simplicity**: No distributed state coordination

### Negative
- **Checkpoint overhead**: Periodic serialization cost
- **Cold start latency**: Must load model weights
- **Storage costs**: Checkpoints consume GCS storage

### Mitigations
- Checkpoint frequency tuned to cost/risk tradeoff
- Model warmup on Cloud Run min-instances
- Checkpoint retention policy (delete old)

## Implementation

```python
# Agent state is fully serializable
@dataclass
class AgentState:
    beliefs: torch.Tensor
    publications: List[str]
    # ... all state explicit

def checkpoint(agent: ResearcherAgent, path: str):
    state = AgentState(
        beliefs=agent.belief_state.beliefs,
        publications=[p.id for p in agent.publications],
    )
    torch.save(state, path)

def restore(path: str) -> ResearcherAgent:
    state = torch.load(path)
    agent = ResearcherAgent()
    agent.belief_state.beliefs = state.beliefs
    # ... restore all state
    return agent
```

## References
- Twelve-Factor App: Processes
- Google Cloud Well-Architected: Reliability
