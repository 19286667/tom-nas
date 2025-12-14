# GitHub Copilot Instructions for ToM-NAS

## Project Overview

ToM-NAS (Theory of Mind Neural Architecture Search) is a research framework for evolving neural architectures capable of Theory of Mind (ToM) reasoning through coevolutionary neural architecture search. This is a PhD research project focused on developing AI systems that can understand and reason about mental states of others.

### Key Objectives
- Implement genuine Theory of Mind capabilities through neural architecture search
- Support 5th-order recursive belief reasoning ("Alice believes that Bob thinks that Charlie knows...")
- Provide transparent and interpretable AI systems
- Enable social interaction simulation and evaluation

## Core Components

### 1. Soul Map Ontology (`src/core/ontology.py`)
- 181-dimensional psychological state representation
- 9 layers covering biological, affective, and cognitive dimensions
- Provides grounding for mental state representation

### 2. Belief Systems (`src/core/beliefs.py`)
- 5th-order recursive belief structures
- Nested mental state modeling
- Confidence tracking and belief propagation

### 3. Agent Architectures (`src/agents/architectures.py`)
- **TRN (Transparent Recurrent Network)**: Interpretable recurrent architecture
- **RSAN (Relational Self-Attention Network)**: Attention-based relational reasoning
- **Transformer**: Standard transformer architecture adapted for ToM
- All architectures use the 181-dimensional ontology as state representation

### 4. Social World Simulation (`src/world/social_world.py`)
- Multi-agent social environment
- Game types: Cooperation, Communication, Resource Sharing, Zombie Detection
- Reputation dynamics and coalition formation
- Observable social interactions for training

### 5. Evolution/NAS Engine (`src/evolution/`)
- **nas_engine.py**: Population-based architecture evolution
- **fitness.py**: ToM-specific fitness evaluation (Sally-Anne tests, higher-order ToM)
- **operators.py**: Genetic operators for architecture modification

### 6. Evaluation & Benchmarks (`src/evaluation/`)
- Sally-Anne test variants
- Higher-order ToM tests (1st through 5th order)
- Zombie detection tests
- Cooperation and social reasoning benchmarks

## Technology Stack

- **Python 3.9+**: Primary language
- **PyTorch 2.0+**: Neural network framework
- **NumPy**: Numerical computations
- **NetworkX**: Graph structures for belief networks
- **Matplotlib/Plotly**: Visualization
- **Streamlit**: Interactive dashboards

## Development Workflow

### Installation
```bash
pip install -r requirements.txt
```

### Quick Health Check
```bash
python test_system.py
```

### Run Full Test Suite
```bash
pytest tests/
# or
python test_comprehensive.py
```

### Run Demonstrations
```bash
python demo_full_run.py        # Full system demonstration
python integrated_tom_system.py # Basic integration test
```

### Training and Evolution
```bash
python train.py                # Basic training
python train_coevolution.py    # Coevolutionary training
python experiment_runner.py    # Run experiments
```

### Visualization
```bash
python visualize.py            # Generate plots and visualizations
```

## Coding Standards

### Python Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Docstrings for all public functions and classes
- Maximum line length: 100 characters (as seen in existing code)

### Code Organization
- One class per major concept
- Clear separation of concerns
- Modular architecture components
- Comprehensive documentation in docstrings

### Naming Conventions
- Classes: `PascalCase` (e.g., `SoulMapOntology`, `TransparentRNN`)
- Functions/methods: `snake_case` (e.g., `evaluate_fitness`, `run_evolution`)
- Constants: `UPPER_CASE` (e.g., `MAX_GENERATIONS`, `POPULATION_SIZE`)
- Private methods: prefix with `_` (e.g., `_build_layers`)

### Documentation Style
```python
def function_name(param1: type, param2: type) -> return_type:
    """Brief description.
    
    Detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> result = function_name(arg1, arg2)
    """
```

## Testing Guidelines

### Test Location
- Unit tests: `tests/` directory
- Integration tests: Root level files like `test_system.py`, `test_comprehensive.py`
- All test files should start with `test_`

### Test Execution
```bash
# Quick test
python test_system.py

# Comprehensive tests
python test_comprehensive.py
pytest tests/test_comprehensive.py

# Specific test files
pytest tests/test_tom_nas_integration.py
pytest tests/test_liminal.py
```

### Test Requirements
- All new features should include tests
- Tests should verify both functionality and ToM capabilities
- Use existing test patterns as templates
- Tests should be self-contained and not depend on external data

### Expected Test Patterns
- Unit tests for individual components
- Integration tests for component interactions
- Benchmark tests for ToM performance
- Validation tests for architecture outputs

## File Organization

```
/home/runner/work/tom-nas/tom-nas/
├── src/                          # Main source code
│   ├── core/                     # Core ontology and beliefs
│   ├── agents/                   # Agent architectures
│   ├── world/                    # Social world simulation
│   ├── evolution/                # NAS and evolution
│   ├── evaluation/               # Benchmarks and metrics
│   ├── visualization/            # Plotting and dashboards
│   ├── cognition/                # Cognitive modules
│   ├── knowledge/                # Knowledge representation
│   ├── simulation/               # Simulation utilities
│   └── benchmarks/               # Benchmark loaders
├── tests/                        # Unit and integration tests
├── scripts/                      # Utility scripts
├── *.py                          # Main entry points and demos
└── *.md                          # Documentation
```

## Key Design Principles

1. **Transparency**: All architectures should be interpretable
2. **Modularity**: Components should be loosely coupled and reusable
3. **Scalability**: Support for large populations and long evolution runs
4. **Research-Oriented**: Code should facilitate experimentation and analysis
5. **ToM-Centric**: All features should support Theory of Mind research goals

## Common Patterns

### Creating New Architectures
- Inherit from `nn.Module`
- Accept `input_dim=181` (ontology size) in constructor
- Implement forward pass with belief state inputs
- Include methods for belief extraction and introspection
- Register in architecture registry

### Adding New Tests
- Use existing Sally-Anne test structure as template
- Include confidence calibration
- Test multiple belief orders (1st through 5th)
- Validate against random baseline

### Evolution Integration
- Define fitness function in `src/evolution/fitness.py`
- Add genetic operators in `src/evolution/operators.py`
- Configure evolution parameters in `nas_engine.py`
- Log checkpoints for reproducibility

## Important Notes for AI Assistants

1. **Preserve Existing Functionality**: This is research code with validated results. Changes should be additive unless fixing bugs.

2. **Maintain Compatibility**: The 181-dimensional ontology is fixed. Don't change this dimension.

3. **Test Changes**: Always run `python test_system.py` after modifications to ensure basic functionality.

4. **Document Research Context**: When adding features, explain how they relate to ToM research goals.

5. **Handle PyTorch Carefully**: Ensure proper tensor shapes, device handling, and gradient flow.

6. **Respect Belief Order**: When working with beliefs, maintain the recursive structure (5th-order maximum).

7. **Follow Existing Patterns**: Use existing code as templates for new features.

8. **Consider Performance**: Evolution runs can be long; optimize where possible without sacrificing clarity.

## Quick Reference

### Run Before Committing
```bash
python test_system.py          # Basic validation
pytest tests/                  # Full test suite
```

### Common Issues
- **Import errors**: Ensure you're in the project root and `src/` is in PYTHONPATH
- **CUDA errors**: Code works on CPU; GPU is optional
- **Dimension mismatches**: Check that all architectures use 181-dimensional inputs
- **Belief order errors**: Validate recursive belief structures don't exceed 5th order

### Getting Help
- Check documentation files: `QUICK_START.md`, `WHERE_IS_EVERYTHING.md`, `COMPLETION_SUMMARY.md`
- Review existing tests for usage examples
- Look at demonstration scripts for component integration

## Special Considerations

### Research Requirements
- This is PhD-level research code
- Maintain scientific rigor in implementations
- Document assumptions and limitations
- Preserve reproducibility through checkpointing

### Performance Considerations
- Evolution can run for 100+ generations
- Social worlds can have 6+ agents
- Population sizes of 20+ individuals
- Optimize critical paths without sacrificing readability

### Extensibility
- New architectures should integrate with existing evolution pipeline
- New benchmarks should follow standard evaluation interface
- New social games should work with existing world simulation
- New belief structures should maintain 5th-order compatibility
