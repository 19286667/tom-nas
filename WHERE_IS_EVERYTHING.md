# 📍 WHERE IS EVERYTHING - Quick Location Guide

## 🏠 Your Project Location

**Absolute Path:** `/home/user/tom-nas`

To navigate there from anywhere:
```bash
cd /home/user/tom-nas
```

---

## 📂 Directory Structure

```
/home/user/tom-nas/
│
├── 🎯 MAIN SCRIPTS (What to run)
│   ├── run_complete_demo.py          # ⭐ START HERE - Complete demonstration
│   ├── test_system.py                # Quick health check
│   ├── test_comprehensive.py         # Full test suite
│   ├── train.py                      # Training pipeline
│   ├── experiment_runner.py          # Run experiments
│   ├── visualize.py                  # Create plots
│   ├── demo_full_run.py              # Detailed demo
│   └── integrated_tom_system.py      # Basic integration
│
├── 📦 SOURCE CODE (src/)
│   ├── core/
│   │   ├── ontology.py               # 181-dim psychological ontology
│   │   └── beliefs.py                # 5th-order recursive beliefs
│   │
│   ├── agents/
│   │   └── architectures.py          # TRN, RSAN, Transformer models
│   │
│   ├── world/
│   │   └── social_world.py           # Social World 4 simulation
│   │
│   ├── evolution/
│   │   ├── nas_engine.py             # Evolution/NAS engine
│   │   ├── fitness.py                # Fitness evaluation
│   │   └── operators.py              # Genetic operators
│   │
│   └── evaluation/
│       ├── benchmarks.py             # Sally-Anne, ToM tests
│       └── metrics.py                # Performance tracking
│
└── 📚 DOCUMENTATION
    ├── COMPLETION_SUMMARY.md         # ⭐ Complete usage guide
    ├── QUICK_START.md                # Quick reference
    ├── REPOSITORY_REVIEW.md          # Code analysis
    ├── PROJECT_TRACKER.md            # Project history
    ├── SESSION_GUIDE.md              # Development guide
    ├── README.md                     # Project overview
    └── WHERE_IS_EVERYTHING.md        # This file!
```

---

## 🚀 How to Run Everything

### 1. First Time Setup
```bash
cd /home/user/tom-nas
pip install -r requirements.txt
```

### 2. Quick Test (30 seconds)
```bash
cd /home/user/tom-nas
python test_system.py
```

### 3. Complete Demo (2-3 minutes)
```bash
cd /home/user/tom-nas
python run_complete_demo.py
```

### 4. Full Test Suite (5 minutes)
```bash
cd /home/user/tom-nas
python test_comprehensive.py
```

### 5. Train an Agent (30 min - 2 hours)
```bash
cd /home/user/tom-nas
python train.py --architecture TRN --epochs 100
```

### 6. Run Evolution (2-8 hours)
```bash
cd /home/user/tom-nas
python experiment_runner.py --experiment evolution --num-generations 50
```

### 7. Generate Visualizations
```bash
cd /home/user/tom-nas
python visualize.py --all
```

---

## 📖 Which Documentation to Read?

### Start Here:
1. **COMPLETION_SUMMARY.md** - Everything you need to know
2. **QUICK_START.md** - Fast reference for commands

### For Development:
3. **REPOSITORY_REVIEW.md** - Detailed code analysis
4. **PROJECT_TRACKER.md** - Development history

### For Understanding:
5. **README.md** - Project overview
6. **SESSION_GUIDE.md** - How the project was built

---

## 🗂️ Where Results Go

### Training Results
```
/home/user/tom-nas/checkpoints/
├── best_model.pt
├── checkpoint_epoch_20.pt
└── metrics.json
```

### Experiment Results
```
/home/user/tom-nas/results/
├── baseline_results.json
├── evolution/
│   └── evolution_summary.json
├── complete_results.json
└── figures/
    ├── training_curves.png
    ├── architecture_comparison.png
    └── summary_dashboard.png
```

---

## 🔍 How to Find Specific Things

### Find a specific function or class:
```bash
cd /home/user/tom-nas
grep -r "class ClassName" src/
grep -r "def function_name" src/
```

### Find which file contains something:
```bash
cd /home/user/tom-nas
grep -r "SallyAnne" src/
grep -r "Evolution" src/
```

### List all Python files:
```bash
cd /home/user/tom-nas
find . -name "*.py" -type f
```

---

## 📊 Key Files by Purpose

### Want to understand the ontology?
➜ `src/core/ontology.py`

### Want to see belief reasoning?
➜ `src/core/beliefs.py`

### Want to understand the architectures?
➜ `src/agents/architectures.py`

### Want to see the social simulation?
➜ `src/world/social_world.py`

### Want to understand evolution?
➜ `src/evolution/nas_engine.py`

### Want to see benchmarks?
➜ `src/evaluation/benchmarks.py`

### Want to train a model?
➜ `train.py`

### Want to run experiments?
➜ `experiment_runner.py`

---

## 💡 Quick Commands Cheatsheet

```bash
# Navigate to project
cd /home/user/tom-nas

# See what's here
ls -la

# Run the demo
python run_complete_demo.py

# Run tests
python test_comprehensive.py

# Train TRN
python train.py --architecture TRN --epochs 50

# Train RSAN
python train.py --architecture RSAN --epochs 50

# Train Transformer
python train.py --architecture Transformer --epochs 50

# Run evolution
python experiment_runner.py --experiment evolution

# Run baseline comparison
python experiment_runner.py --experiment baseline

# Make plots
python visualize.py --all

# Check git status
git status

# See what branch you're on
git branch
```

---

## 🎯 If You Get Lost

**Just remember:** Everything is in `/home/user/tom-nas`

From anywhere on your system:
```bash
cd /home/user/tom-nas
ls
```

Then start with:
```bash
python run_complete_demo.py
```

---

## 📞 Need Help?

1. Check **COMPLETION_SUMMARY.md** first
2. Check **QUICK_START.md** for commands
3. Check **REPOSITORY_REVIEW.md** for code details

---

**Last Updated:** November 20, 2025
**Project Status:** ✅ 100% Complete
