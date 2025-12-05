# ToM-NAS: Theory of Mind Neural Architecture Search

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for evolving neural architectures capable of Theory of Mind (ToM) reasoning through coevolutionary neural architecture search.

## Overview

ToM-NAS implements a complete system for discovering neural architectures that can reason about beliefs, intentions, and mental states of other agents. The framework combines:

- **Soul Map Ontology**: 181-dimensional psychological grounding across 9 layers
- **Recursive Belief System**: Supports up to 5th-order nested belief reasoning
- **Multiple Neural Architectures**: TRN, RSAN, and Transformer models
- **Social World Simulation**: Complex multi-agent environment with zombie detection
- **Evolutionary Framework**: Multi-level coevolution for architecture discovery
- **Liminal Game Environment**: Advanced psychological game world for ToM testing

## Installation

### Requirements

- Python 3.9 or higher
- PyTorch 2.0+
- NumPy, NetworkX, Matplotlib, scikit-learn, pandas, tqdm

### Quick Install

```bash
# Clone the repository
git clone https://github.com/19286667/tom-nas.git
cd tom-nas

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python test_system.py
```

### Development Install

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Verify Installation

```bash
python test_system.py
```

### 2. Run a Complete Demo

```bash
python run_complete_demo.py
```

### 3. Train with Evolution

```bash
python train.py --architecture TRN --epochs 50
```

### 4. Run Experiments

```bash
python experiment_runner.py --experiment evolution --num-generations 50
```

### 5. Explore the Liminal Environment

```bash
python run_liminal_demo.py
```

## Project Structure

```
tom-nas/
├── src/                          # Main source code
│   ├── core/                     # Core ToM components
│   │   ├── ontology.py           # 181-dimensional Soul Map Ontology
│   │   └── beliefs.py            # 5th-order Recursive Beliefs
│   ├── agents/                   # Neural architectures
│   │   └── architectures.py      # TRN, RSAN, Transformer models
│   ├── world/                    # Environment simulation
│   │   └── social_world.py       # Social World 4 simulator
│   ├── evolution/                # Neural Architecture Search
│   │   ├── nas_engine.py         # Main evolution engine
│   │   ├── fitness.py            # Fitness evaluation
│   │   └── operators.py          # Genetic operators
│   ├── evaluation/               # Benchmarking suite
│   │   ├── benchmarks.py         # ToM tests
│   │   └── metrics.py            # Performance tracking
│   └── liminal/                  # Game environment
│       ├── game_environment.py   # Main environment
│       ├── soul_map.py           # 60-dimensional psychology
│       ├── realms.py             # Five game realms
│       ├── npcs/                 # NPC system
│       └── mechanics/            # Game mechanics
├── test_*.py                     # Test files
├── train.py                      # Training pipeline
├── train_coevolution.py          # Multi-architecture training
├── experiment_runner.py          # Experiment framework
├── requirements.txt              # Dependencies
└── setup.py                      # Package configuration
```

## Key Features

### Neural Architectures

1. **Transparent Recurrent Network (TRN)**: Interpretable RNN with gated mechanisms
2. **Recursive Self-Attention Network (RSAN)**: Multi-head attention for recursive reasoning
3. **Transformer**: Standard transformer encoder for pragmatic communication

### Theory of Mind Capabilities

- **False Belief Tasks**: Sally-Anne test and variations
- **Higher-Order ToM**: 1st through 5th order belief reasoning
- **Zombie Detection**: 6 types of philosophical zombie detection
- **Cooperation Games**: Repeated Prisoner's Dilemma with reputation

### Evolutionary Features

- Tournament selection with elitism
- Adaptive mutation rates
- Speciation for diversity preservation
- Multi-architecture coevolution

## Running Tests

```bash
# Quick system test
python test_system.py

# Comprehensive test suite
python test_comprehensive.py

# Liminal environment tests
python test_liminal.py

# Run with pytest
pytest -v
```

## Configuration

The framework uses `EvolutionConfig` for customization:

```python
from src.evolution.nas_engine import EvolutionConfig, NASEngine

config = EvolutionConfig(
    population_size=20,
    num_generations=100,
    elite_size=2,
    mutation_rate=0.1,
    crossover_rate=0.7,
)
```

## Documentation

- `QUICK_START.md` - Getting started guide
- `WHERE_IS_EVERYTHING.md` - Navigation guide
- `COMPLETION_SUMMARY.md` - Feature summary
- `REPOSITORY_REVIEW.md` - Code analysis

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work, please cite:

```bibtex
@software{tom_nas_2024,
  title = {ToM-NAS: Theory of Mind Neural Architecture Search},
  author = {ToM-NAS Research Team},
  year = {2024},
  url = {https://github.com/19286667/tom-nas}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For questions or issues, please open a GitHub issue.
