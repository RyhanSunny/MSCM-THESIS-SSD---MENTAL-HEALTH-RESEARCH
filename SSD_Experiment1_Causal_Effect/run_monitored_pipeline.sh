#!/bin/bash

# Monitored Pipeline Runner
# Author: Research Assistant
# Date: 2025-06-29

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure we have conda activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Activating conda base environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi

# Create logs directory
mkdir -p logs/monitoring

# Install psutil if not present
python3 -c "import psutil" 2>/dev/null || pip install psutil

echo "=== SSD Pipeline with Enhanced Monitoring ==="
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo ""

# Function to run with monitoring
run_with_monitoring() {
    local mode="$1"
    local step="$2"
    
    echo "Starting monitored execution: $mode $step"
    
    # Run the monitoring script
    if [ "$mode" = "step" ]; then
        bash scripts/monitor_pipeline.sh step "$step"
    else
        bash scripts/monitor_pipeline.sh full
    fi
}

# Check command line arguments
case "${1:-full}" in
    "full")
        echo "Running full pipeline with monitoring..."
        run_with_monitoring "full"
        ;;
    "step")
        if [ -z "$2" ]; then
            echo "Usage: $0 step <step_name>"
            echo "Available steps: cohort, exposure, mediator, outcomes, confounders, lab, referral, pre-imputation-master, missing-master, master, sequential, ps, causal-mi, pool-mi"
            exit 1
        fi
        echo "Running step '$2' with monitoring..."
        run_with_monitoring "step" "$2"
        ;;
    "test")
        echo "Running memory optimization test..."
        python3 scripts/optimize_memory.py
        ;;
    *)
        echo "Usage: $0 [full|step <step_name>|test]"
        echo "  full: Run complete pipeline with monitoring"
        echo "  step <name>: Run individual step with monitoring"
        echo "  test: Test memory optimization"
        exit 1
        ;;
esac

echo ""
echo "=== Execution completed at $(date) ==="
echo "Check logs in: logs/monitoring/" 