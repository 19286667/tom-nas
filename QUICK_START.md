# ToM-NAS Quick Start Guide

## 🚀 Running Your First Demo (3 Steps)

### Step 1: Install Dependencies

```bash
pip install torch numpy networkx matplotlib scikit-learn tqdm pandas
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Test Suite

```bash
python test_system.py
```

**Expected output:**
```
============================================================
ToM-NAS Complete System Test
============================================================
  ✓ Ontology: 181 dimensions
  ✓ Beliefs: 5th-order recursion
  ✓ Architectures: TRN, RSAN, Transformer
  ✓ Social World: 6 agents
============================================================
✓ ALL TESTS PASSED!
============================================================
```

### Step 3: Run the Full Demonstration

```bash
python demo_full_run.py
```

This will show you:
- Complete system initialization
- All architecture details
- Forward pass demonstrations
- Social world simulation
- Belief network operations
- System capabilities summary

---

## 📋 Available Scripts

| Script | Purpose | Output Detail | Runtime |
|--------|---------|---------------|---------|
| `test_system.py` | Quick smoke test | Minimal | ~1 sec |
| `integrated_tom_system.py` | Basic integration test | Medium | ~2 sec |
| `demo_full_run.py` | **Complete demonstration** | **Maximum** | ~5 sec |

---

## 🎯 What Each Script Shows

### `test_system.py` - Quick Health Check
- ✅ Verifies all imports work
- ✅ Tests component initialization
- ✅ Confirms basic functionality
- Use this for: Quick validation after changes

### `integrated_tom_system.py` - Integration Test
- ✅ Initializes all components
- ✅ Tests forward passes
- ✅ Verifies output shapes
- Use this for: Basic system verification

### `demo_full_run.py` - Full Demonstration ⭐
- ✅ Shows ontology details (dimensions, layers, encoding)
- ✅ Demonstrates belief system (all 5 orders)
- ✅ Details all three architectures (parameters, features)
- ✅ Shows forward propagation with metrics
- ✅ Simulates social world interactions
- ✅ Demonstrates complete integration pipeline
- ✅ Provides system summary and capabilities
- Use this for: Understanding exactly what the system does

---

## 🔧 System Configuration

### Default Parameters

```python
# Agent Configuration
num_agents = 6-10           # Number of agents in world
ontology_dim = 181          # Psychological state dimensions
input_dim = 191             # 181 + 10 context features
hidden_dim = 128            # Hidden layer size
max_belief_order = 5        # Up to 5th-order ToM

# Architecture Sizes
TRN:         ~500K parameters
RSAN:        ~600K parameters
Transformer: ~800K parameters

# World Configuration
num_zombies = 1-2           # Zombie agents for detection
zombie_types = 6            # Different zombie varieties
```

### Customization

To change parameters, edit the scripts:

```python
# In demo_full_run.py, modify:
num_agents = 10              # Your desired number
hidden_dim = 256             # Larger hidden dimension
```

---

## 💻 Example Usage in Python

### Quick Test

```python
from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork
from src.agents.architectures import TransparentRNN

# Initialize components
ontology = SoulMapOntology()
beliefs = BeliefNetwork(num_agents=6, ontology_dim=181, max_order=5)
trn = TransparentRNN(input_dim=191, hidden_dim=128, output_dim=181)

# Test forward pass
import torch
test_input = torch.randn(1, 10, 191)
output = trn(test_input)

print(f"Beliefs shape: {output['beliefs'].shape}")
print(f"Actions shape: {output['actions'].shape}")
```

### Create and Simulate Social World

```python
from src.world.social_world import SocialWorld4

# Create world with zombies
world = SocialWorld4(num_agents=10, ontology_dim=181, num_zombies=2)

# Simulate timesteps
for t in range(10):
    actions = [{'action': 'cooperate'} for _ in range(10)]
    result = world.step(actions)
    print(f"Timestep {t}: {result['timestep']}")
```

### Work with Beliefs

```python
from src.core.beliefs import RecursiveBeliefState
import torch

# Create belief state for an agent
agent_beliefs = RecursiveBeliefState(agent_id=0, ontology_dim=181, max_order=5)

# Add 1st-order belief (I believe agent 1's state)
belief_content = torch.randn(181)
agent_beliefs.update_belief(order=1, target=1, content=belief_content,
                           confidence=0.9, source="observation")

# Add 2nd-order belief (I believe agent 1 believes about agent 2)
agent_beliefs.update_belief(order=2, target=2, content=belief_content,
                           confidence=0.7, source="inference")

# Query beliefs
belief = agent_beliefs.get_belief(order=1, target=1)
print(f"Confidence: {belief.confidence}")
```

---

## 📊 Understanding the Output

### When you see this in ontology section:
```
Total Dimensions: 181
Layers Defined: 2
```
- The ontology has 181 psychological dimensions
- Currently 2 layers fully defined (biological, affective)
- More layers can be added as needed

### When you see this in architecture section:
```
Total parameters: 548,609
```
- This is the number of trainable parameters
- ~500K-800K is appropriate for this task
- Comparable to small transformer models

### When you see this in forward pass:
```
Belief range: [0.234, 0.876]
Mean belief value: 0.512
```
- Beliefs are normalized to [0, 1]
- Mean around 0.5 indicates balanced uncertainty
- Range shows belief diversity

### When you see this in social world:
```
Agent 3: Resources=100.0, Energy=100.0 [ZOMBIE] (behavioral)
```
- This agent is a zombie (not genuine ToM)
- Type "behavioral" means inconsistent action patterns
- Genuine agents must detect them

---

## 🐛 Troubleshooting

### Error: "No module named 'torch'"
**Solution:**
```bash
pip install torch numpy
```

### Error: "No module named 'src'"
**Solution:** Make sure you're in the tom-nas directory:
```bash
cd /path/to/tom-nas
python demo_full_run.py
```

### Error: Import errors
**Solution:** Install all dependencies:
```bash
pip install -r requirements.txt
```

### Script runs but output is unclear
**Solution:** Use `demo_full_run.py` for the most detailed output

---

## 📖 Understanding the System

### What is ToM-NAS?

**ToM-NAS** = Theory of Mind + Neural Architecture Search

- **Theory of Mind**: Ability to reason about others' mental states
- **Neural Architecture Search**: Evolving optimal neural network designs
- **This Project**: Combines both to create AI with genuine ToM capabilities

### Key Innovations

1. **181-Dimensional Ontology**: Comprehensive psychological state space
2. **5th-Order Beliefs**: Deep recursive reasoning (I believe you believe I believe...)
3. **Three Architectures**: TRN (transparent), RSAN (recursive attention), Transformer
4. **Zombie Detection**: Tests for genuine vs. fake ToM
5. **Coevolution**: Architectures, tasks, and tests evolve together

### Architecture Roles

- **TRN (Transparent RNN)**: Interpretable, step-by-step reasoning
- **RSAN (Recursive Self-Attention)**: Hierarchical belief modeling
- **Transformer**: Communication and pragmatic reasoning
- **Hybrid (future)**: Best of all three through evolution

---

## 🎓 For Research & Development

### Adding New Components

```python
# Create new architecture in src/agents/architectures.py
class MyNewArchitecture(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Your architecture here

    def forward(self, x):
        # Return dict with 'beliefs' and 'actions'
        return {'beliefs': ..., 'actions': ...}
```

### Testing Your Changes

1. Run test suite: `python test_system.py`
2. Run integration: `python integrated_tom_system.py`
3. Run full demo: `python demo_full_run.py`

### Benchmark Evaluation (Coming Soon)

```python
# Will be in src/evaluation/benchmarks.py
from src.evaluation.benchmarks import SallyAnneTest

test = SallyAnneTest()
results = test.evaluate(agent)
print(f"Accuracy: {results['accuracy']}")
```

---

## 📚 Documentation Structure

- `QUICK_START.md` (this file) - Get started quickly
- `REPOSITORY_REVIEW.md` - Comprehensive analysis
- `PROJECT_TRACKER.md` - Project management and progress
- `SESSION_GUIDE.md` - Session continuity protocols
- `README.md` - Project overview

---

## ✨ Quick Reference Commands

```bash
# First time setup
pip install -r requirements.txt

# Quick test (1 second)
python test_system.py

# Basic demo (2 seconds)
python integrated_tom_system.py

# Full demonstration with detailed output (5 seconds) ⭐
python demo_full_run.py

# Check code quality
flake8 src/

# Future: Run training
python train.py  # (to be implemented)

# Future: Run benchmarks
python benchmark.py  # (to be implemented)
```

---

## 🎯 Next Steps

After running the demos:

1. **Understand the system**: Read through `REPOSITORY_REVIEW.md`
2. **Plan development**: Check `PROJECT_TRACKER.md`
3. **Implement evolution**: Priority for research contribution
4. **Create benchmarks**: Priority for validation
5. **Run experiments**: Collect results for dissertation

---

## 💡 Tips

- Start with `demo_full_run.py` to understand the full system
- Use `test_system.py` for quick validation during development
- Check `REPOSITORY_REVIEW.md` for detailed component analysis
- Refer to `PROJECT_TRACKER.md` for development priorities

---

**Remember:** The system is fully functional for component testing and demonstration. The main work needed is implementing the evolution engine and evaluation benchmarks to complete the research pipeline.

**For help or questions, refer to the comprehensive documentation files in this repository.**

---

Last Updated: November 20, 2025
