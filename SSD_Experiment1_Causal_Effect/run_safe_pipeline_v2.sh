#!/bin/bash
# Safe Pipeline Runner v2 with Better Error Handling and Resume
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
LOG_FILE="$LOG_DIR/safe_pipeline_v2_${TIMESTAMP}.log"

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

# Function to get last checkpoint
get_last_checkpoint() {
    if [[ -f "$CHECKPOINT_DIR/last_checkpoint.json" ]]; then
        local last_step=$(jq -r '.step' "$CHECKPOINT_DIR/last_checkpoint.json" 2>/dev/null || echo "")
        local last_status=$(jq -r '.status' "$CHECKPOINT_DIR/last_checkpoint.json" 2>/dev/null || echo "")
        
        # Only return checkpoint if the last status was completed
        if [[ "$last_status" == "completed" ]]; then
            echo "$last_step"
        else
            # If last step failed, return the one before it
            for i in "${!PIPELINE_STEPS[@]}"; do
                if [[ "${PIPELINE_STEPS[$i]}" == "$last_step" ]]; then
                    if [[ $i -gt 0 ]]; then
                        echo "${PIPELINE_STEPS[$((i - 1))]}"
                    fi
                    break
                fi
            done
        fi
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

# Function to verify step output
verify_step_output() {
    local step=$1
    
    case $step in
        "cohort")
            if [[ ! -f "$PROJECT_DIR/data_derived/cohort.parquet" ]]; then
                log "ERROR: cohort.parquet not found after cohort step"
                return 1
            fi
            ;;
        "exposure")
            # Check for either exposure.parquet or exposure_or.parquet
            if [[ ! -f "$PROJECT_DIR/data_derived/exposure.parquet" ]] && [[ ! -f "$PROJECT_DIR/data_derived/exposure_or.parquet" ]]; then
                log "ERROR: exposure output not found after exposure step"
                return 1
            fi
            # Create symlink if needed for backward compatibility
            if [[ -f "$PROJECT_DIR/data_derived/exposure_or.parquet" ]] && [[ ! -f "$PROJECT_DIR/data_derived/exposure.parquet" ]]; then
                ln -sf exposure_or.parquet "$PROJECT_DIR/data_derived/exposure.parquet"
                log "Created symlink: exposure.parquet -> exposure_or.parquet"
            fi
            ;;
    esac
    return 0
}

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
    
    # Run the actual step using the correct Python
    set +e  # Don't exit on error
    timeout 3600 bash -c "cd '$PROJECT_DIR' && make $step" 2>&1 | tee "$step_log"
    local exit_code=$?
    set -e
    
    # Stop monitor
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true
    
    # Verify output was created
    if [[ $exit_code -eq 0 ]]; then
        if verify_step_output "$step"; then
            save_checkpoint "$step" "completed"
            log "Step $step completed successfully"
            
            # Post-step cleanup
            log "Running post-step cleanup..."
            /home/roject4/miniconda3/bin/python3 -c "import gc; gc.collect()" 2>/dev/null || true
            
            # Give system time to recover
            sleep 5
        else
            save_checkpoint "$step" "failed"
            log "ERROR: Step $step completed but output verification failed"
            return 1
        fi
    else
        save_checkpoint "$step" "failed"
        log "ERROR: Step $step failed with exit code $exit_code"
        return $exit_code
    fi
}

# Main execution
log "=== Safe Pipeline Runner v2 Started ==="
log "Working directory: $PROJECT_DIR"
log "Memory threshold: ${MEMORY_THRESHOLD}%"
log "Minimum free memory: ${MIN_FREE_GB}GB"
log "Using Python: /home/roject4/miniconda3/bin/python3"

# Verify Python installation
/home/roject4/miniconda3/bin/python3 -c "import pandas, tensorflow; print('Dependencies OK')" || {
    log "ERROR: Python dependencies not installed"
    exit 1
}

# Check initial memory
check_memory

# Determine starting point
LAST_CHECKPOINT=$(get_last_checkpoint)
START_FROM=""
SKIP_RESUME_PROMPT=false

# Check command line argument for auto-resume
if [[ "${1:-}" == "--resume" ]]; then
    SKIP_RESUME_PROMPT=true
fi

if [[ -n "$LAST_CHECKPOINT" ]]; then
    log "Found checkpoint: last completed step was '$LAST_CHECKPOINT'"
    
    if [[ "$SKIP_RESUME_PROMPT" == "true" ]]; then
        REPLY="y"
    else
        read -p "Resume from after '$LAST_CHECKPOINT'? (y/n): " -n 1 -r
        echo
    fi
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Find the next step after checkpoint
        for i in "${!PIPELINE_STEPS[@]}"; do
            if [[ "${PIPELINE_STEPS[$i]}" == "$LAST_CHECKPOINT" ]]; then
                if [[ $((i + 1)) -lt ${#PIPELINE_STEPS[@]} ]]; then
                    START_FROM="${PIPELINE_STEPS[$((i + 1))]}"
                    log "Resuming from step: $START_FROM"
                    break
                else
                    log "Last checkpoint was the final step. Pipeline complete!"
                    exit 0
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
        log "To resume, run: $0 --resume"
        exit 1
    fi
done

log "=== Pipeline completed successfully! ==="
log "All steps executed without memory exhaustion"
log "Logs available at: $LOG_DIR"