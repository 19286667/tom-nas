#!/bin/bash
# Run Full ToM-NAS Experiment Suite
#
# This script runs all experiments in sequence:
# 1. Baseline benchmark study
# 2. Evolutionary NAS on all tasks
# 3. Ablation studies
# 4. Analysis and report generation
#
# Usage:
#   ./scripts/run_full_suite.sh
#   ./scripts/run_full_suite.sh --quick  # Quick test mode
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/results/$(date +%Y%m%d_%H%M%S)"

# Parse arguments
QUICK_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "ToM-NAS Full Experiment Suite"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "Quick mode: $QUICK_MODE"
echo "=================================================="

cd "$PROJECT_ROOT"

# Step 1: Run baseline benchmark study
echo ""
echo "Step 1: Running baseline benchmark study..."
echo "--------------------------------------------------"
python scripts/run_experiments.py \
    --mode baseline \
    --output "$OUTPUT_DIR"

# Step 2: Run evolutionary NAS
echo ""
echo "Step 2: Running evolutionary NAS experiments..."
echo "--------------------------------------------------"

if [ "$QUICK_MODE" = true ]; then
    # Quick mode: fewer tasks, smaller population
    python scripts/run_experiments.py \
        --mode quick \
        --tasks simple_sequence tomi hitom_2 \
        --population 32 \
        --generations 10 \
        --seeds 42 \
        --output "$OUTPUT_DIR"
else
    # Full mode: all tasks, full evolution
    python scripts/run_experiments.py \
        --mode hypothesis \
        --methods CMA_ES OpenES \
        --population 128 \
        --generations 50 \
        --seeds 42 123 456 \
        --output "$OUTPUT_DIR"
fi

# Step 3: Run ablation studies
echo ""
echo "Step 3: Running ablation studies..."
echo "--------------------------------------------------"

if [ "$QUICK_MODE" = true ]; then
    python scripts/run_experiments.py \
        --mode ablation \
        --tasks simple_sequence tomi \
        --population 16 \
        --generations 5 \
        --seeds 42 \
        --output "$OUTPUT_DIR"
else
    python scripts/run_experiments.py \
        --mode ablation \
        --tasks simple_sequence babi_1 tomi hitom_2 hitom_4 \
        --population 64 \
        --generations 25 \
        --seeds 42 123 456 \
        --output "$OUTPUT_DIR"
fi

# Step 4: Generate final report
echo ""
echo "Step 4: Generating analysis and report..."
echo "--------------------------------------------------"

python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')

from pathlib import Path
import json

output_dir = Path('$OUTPUT_DIR')

# Find all result files
result_files = list(output_dir.rglob('*_results.json')) + list(output_dir.rglob('summary_stats.json'))

print(f'Found {len(result_files)} result files')

# Generate combined summary
summary_lines = []
summary_lines.append('# ToM-NAS Experiment Results')
summary_lines.append('')
summary_lines.append(f'Output directory: $OUTPUT_DIR')
summary_lines.append('')

for f in result_files:
    summary_lines.append(f'## {f.name}')
    summary_lines.append(f'Path: {f}')
    summary_lines.append('')

# Write summary
summary_path = output_dir / 'EXPERIMENT_SUMMARY.md'
with open(summary_path, 'w') as f:
    f.write('\n'.join(summary_lines))

print(f'Summary written to {summary_path}')
"

echo ""
echo "=================================================="
echo "EXPERIMENT SUITE COMPLETE"
echo "=================================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key output files:"
echo "  - */summary_stats.json: Statistical analysis"
echo "  - */statistical_report.md: Hypothesis test results"
echo "  - */all_results.json: Complete experiment data"
echo "=================================================="
