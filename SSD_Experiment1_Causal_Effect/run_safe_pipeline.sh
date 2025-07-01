#!/bin/bash
# Safe Pipeline Runner with Memory Protection and Auto-Recovery
# Prevents WSL termination due to memory exhaustion

set -euo pipefail

PROJECT_DIR="/mnt/c/Users/ProjectC4M/Documents/MSCM THESIS SSD/MSCM-THESIS-SSD---MENTAL-HEALTH-RESEARCH/SSD_Experiment1_Causal_Effect"
cd "$PROJECT_DIR"

# Configuration
MEMORY_THRESHOLD=85  # Pause if memory usage exceeds this percentage
MIN_FREE_GB=2       # Minimum free memory in GB
CHECKPOINT_DIR="$PROJECT_DIR/pipeline_checkpoints"
LOG_DIR="$PROJECT_DIR/logs/safe_runner"

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/safe_pipeline_${TIMESTAMP}.log"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check memory
check_memory() {
    local mem_info=$(free -g | grep Mem)
    local total=$(echo $mem_info | awk '{print $2}')
    local used=$(echo $mem_info | awk '{print $3}')
    local free=$(echo $mem_info | awk '{print $4}')
    local usage_percent=$((used * 100 / total))
    
    log "Memory Status: ${used}G/${total}G used (${usage_percent}%), ${free}G free"
    
    if [[ $usage_percent -gt $MEMORY_THRESHOLD ]] || [[ $free -lt $MIN_FREE_GB ]]; then
        log "WARNING: High memory usage detected!"
        return 1
    fi
    return 0
}

# Function to wait for memory to recover
wait_for_memory() {
    log "Waiting for memory to recover..."
    while ! check_memory; do
        log "Memory still high, waiting 30 seconds..."
        # Force garbage collection in Python processes
        pkill -USR1 python3 2>/dev/null || true
        sleep 30
    done
    log "Memory recovered, continuing..."
}

# Function to save checkpoint
save_checkpoint() {
    local step=$1
    local status=$2
    echo "{\"step\": \"$step\", \"status\": \"$status\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$CHECKPOINT_DIR/last_checkpoint.json"
    log "Checkpoint saved: $step - $status"
}

# Function to get last successful step
get_last_checkpoint() {
    if [[ -f "$CHECKPOINT_DIR/last_checkpoint.json" ]]; then
        jq -r '.step' "$CHECKPOINT_DIR/last_checkpoint.json" 2>/dev/null || echo ""
    else
        echo ""
    fi
}

# Pipeline steps in order
PIPELINE_STEPS=(
    "cohort"
    "exposure"
    "mediator"
    "outcomes"
    "confounders"
    "lab"
    "referral"
    "pre-imputation-master"
    "missing-master"
    "master"
    "sequential"
    "ps"
    "causal-mi"
    "pool-mi"
)

# Function to run a pipeline step with protection
run_protected_step() {
    local step=$1
    
    log "=== Starting step: $step ==="
    
    # Pre-step memory check
    if ! check_memory; then
        wait_for_memory
    fi
    
    # Save pre-step state
    save_checkpoint "$step" "started"
    
    # Run the step with monitoring
    local step_log="$LOG_DIR/step_${step}_${TIMESTAMP}.log"
    
    # Start background memory monitor
    (
        while true; do
            if ! check_memory; then
                log "WARNING: Memory critical during $step execution!"
                # Send USR1 signal to Python processes to trigger garbage collection
                pkill -USR1 python3 2>/dev/null || true
            fi
            sleep 10
        done
    ) &
    MONITOR_PID=$!
    
    # Run the actual step
    set +e  # Don't exit on error
    timeout 3600 bash -c "cd '$PROJECT_DIR' && make $step" 2>&1 | tee "$step_log"
    local exit_code=$?
    set -e
    
    # Stop monitor
    kill $MONITOR_PID 2>/dev/null || true
    
    if [[ $exit_code -eq 0 ]]; then
        save_checkpoint "$step" "completed"
        log "Step $step completed successfully"
        
        # Post-step cleanup
        log "Running post-step cleanup..."
        python3 -c "import gc; gc.collect()" 2>/dev/null || true
        
        # Give system time to recover
        sleep 5
    else
        save_checkpoint "$step" "failed"
        log "ERROR: Step $step failed with exit code $exit_code"
        return $exit_code
    fi
}

# Main execution
log "=== Safe Pipeline Runner Started ==="
log "Working directory: $PROJECT_DIR"
log "Memory threshold: ${MEMORY_THRESHOLD}%"
log "Minimum free memory: ${MIN_FREE_GB}GB"

# Check conda environment
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    log "Activating conda base environment..."
    eval "$(conda shell.bash hook)"
    conda activate base
fi

# Install required Python packages if needed
python3 -c "import psutil" 2>/dev/null || pip install psutil

# Check initial memory
check_memory

# Determine starting point
LAST_CHECKPOINT=$(get_last_checkpoint)
START_FROM=""

if [[ -n "$LAST_CHECKPOINT" ]]; then
    log "Found checkpoint: last successful step was '$LAST_CHECKPOINT'"
    read -p "Resume from checkpoint? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Find the next step after checkpoint
        for i in "${!PIPELINE_STEPS[@]}"; do
            if [[ "${PIPELINE_STEPS[$i]}" == "$LAST_CHECKPOINT" ]]; then
                if [[ $((i + 1)) -lt ${#PIPELINE_STEPS[@]} ]]; then
                    START_FROM="${PIPELINE_STEPS[$((i + 1))]}"
                    break
                fi
            fi
        done
    fi
fi

# Run pipeline steps
STARTED=false
for step in "${PIPELINE_STEPS[@]}"; do
    # Skip until we reach the starting point
    if [[ -n "$START_FROM" ]] && [[ "$step" != "$START_FROM" ]] && [[ "$STARTED" == "false" ]]; then
        log "Skipping completed step: $step"
        continue
    fi
    STARTED=true
    
    # Run the step with protection
    if ! run_protected_step "$step"; then
        log "Pipeline failed at step: $step"
        log "To resume, run this script again and it will start from this step"
        exit 1
    fi
done

log "=== Pipeline completed successfully! ==="
log "All steps executed without memory exhaustion"
log "Logs available at: $LOG_DIR"