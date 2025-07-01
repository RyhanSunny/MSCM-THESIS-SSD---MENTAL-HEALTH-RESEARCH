#!/bin/bash

# Enhanced Pipeline Monitor with Resource Tracking
# Author: Research Assistant
# Date: 2025-06-29

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/monitoring"
RESOURCE_LOG="$LOG_DIR/resource_usage_$TIMESTAMP.log"
PIPELINE_LOG="$LOG_DIR/pipeline_execution_$TIMESTAMP.log"

# Create monitoring directory
mkdir -p "$LOG_DIR"

echo "=== SSD Pipeline Monitor Started at $(date) ===" | tee "$RESOURCE_LOG"
echo "Project Directory: $PROJECT_DIR" | tee -a "$RESOURCE_LOG"
echo "Resource Log: $RESOURCE_LOG" | tee -a "$RESOURCE_LOG"
echo "Pipeline Log: $PIPELINE_LOG" | tee -a "$RESOURCE_LOG"
echo "" | tee -a "$RESOURCE_LOG"

# Function to log system resources
log_resources() {
    local step_name="$1"
    echo "=== RESOURCE CHECK: $step_name at $(date) ===" >> "$RESOURCE_LOG"
    
    # Memory usage
    echo "Memory Usage:" >> "$RESOURCE_LOG"
    free -h >> "$RESOURCE_LOG"
    echo "" >> "$RESOURCE_LOG"
    
    # Disk usage
    echo "Disk Usage:" >> "$RESOURCE_LOG"
    df -h "$PROJECT_DIR" >> "$RESOURCE_LOG"
    echo "" >> "$RESOURCE_LOG"
    
    # CPU load
    echo "CPU Load:" >> "$RESOURCE_LOG"
    uptime >> "$RESOURCE_LOG"
    echo "" >> "$RESOURCE_LOG"
    
    # Process info
    echo "Python Processes:" >> "$RESOURCE_LOG"
    ps aux | grep python | grep -v grep >> "$RESOURCE_LOG" || echo "No Python processes found" >> "$RESOURCE_LOG"
    echo "" >> "$RESOURCE_LOG"
    
    # Memory-hungry processes
    echo "Top Memory Consumers:" >> "$RESOURCE_LOG"
    ps aux --sort=-%mem | head -10 >> "$RESOURCE_LOG"
    echo "" >> "$RESOURCE_LOG"
    
    echo "----------------------------------------" >> "$RESOURCE_LOG"
}

# Function to monitor a specific step
monitor_step() {
    local step_name="$1"
    local make_target="$2"
    
    echo "Starting step: $step_name" | tee -a "$PIPELINE_LOG"
    log_resources "BEFORE_$step_name"
    
    # Start resource monitoring in background
    (
        while true; do
            sleep 30  # Log every 30 seconds
            log_resources "DURING_${step_name}_$(date +%H%M%S)"
        done
    ) &
    local monitor_pid=$!
    
    # Run the actual step
    local start_time=$(date +%s)
    echo "[$(date)] Starting: $step_name" | tee -a "$PIPELINE_LOG"
    
    if timeout 1800 make "$make_target" 2>&1 | tee -a "$PIPELINE_LOG"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "[$(date)] Completed: $step_name (${duration}s)" | tee -a "$PIPELINE_LOG"
        local success=true
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "[$(date)] FAILED/TIMEOUT: $step_name (${duration}s)" | tee -a "$PIPELINE_LOG"
        local success=false
    fi
    
    # Stop resource monitoring
    kill $monitor_pid 2>/dev/null || true
    wait $monitor_pid 2>/dev/null || true
    
    log_resources "AFTER_$step_name"
    
    if [ "$success" = false ]; then
        echo "Step $step_name failed or timed out. Check logs for details." | tee -a "$PIPELINE_LOG"
        return 1
    fi
    
    return 0
}

# Function to run full pipeline with monitoring
run_monitored_pipeline() {
    echo "Starting monitored pipeline execution..." | tee -a "$PIPELINE_LOG"
    
    # Initial system check
    log_resources "INITIAL_SYSTEM_STATE"
    
    # Check available memory
    local available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    echo "Available memory: ${available_mem}MB" | tee -a "$RESOURCE_LOG"
    
    if [ "$available_mem" -lt 4000 ]; then
        echo "WARNING: Low available memory (${available_mem}MB). Consider closing other applications." | tee -a "$RESOURCE_LOG"
    fi
    
    # Run each step with monitoring
    local steps=(
        "cohort:cohort"
        "exposure:exposure"
        "mediator:mediator"
        "outcomes:outcomes"
        "confounders:confounders"
        "lab:lab"
        "referral:referral"
        "pre-imputation-master:pre-imputation-master"
        "missing-master:missing-master"
        "master:master"
        "sequential:sequential"
        "ps:ps"
        "causal-mi:causal-mi"
        "pool-mi:pool-mi"
    )
    
    for step_info in "${steps[@]}"; do
        IFS=':' read -r step_name make_target <<< "$step_info"
        
        echo "" | tee -a "$PIPELINE_LOG"
        echo "======================================" | tee -a "$PIPELINE_LOG"
        echo "PIPELINE STEP: $step_name" | tee -a "$PIPELINE_LOG"
        echo "======================================" | tee -a "$PIPELINE_LOG"
        
        if ! monitor_step "$step_name" "$make_target"; then
            echo "Pipeline failed at step: $step_name" | tee -a "$PIPELINE_LOG"
            echo "Check resource logs: $RESOURCE_LOG" | tee -a "$PIPELINE_LOG"
            exit 1
        fi
        
        # Brief pause between steps
        sleep 5
    done
    
    echo "" | tee -a "$PIPELINE_LOG"
    echo "=== PIPELINE COMPLETED SUCCESSFULLY ===" | tee -a "$PIPELINE_LOG"
    echo "Resource log: $RESOURCE_LOG" | tee -a "$PIPELINE_LOG"
    echo "Pipeline log: $PIPELINE_LOG" | tee -a "$PIPELINE_LOG"
}

# Function to run individual step with monitoring
run_monitored_step() {
    local step_name="$1"
    
    if [ -z "$step_name" ]; then
        echo "Usage: $0 step <step_name>"
        echo "Available steps: cohort, exposure, mediator, outcomes, etc."
        exit 1
    fi
    
    echo "Running monitored step: $step_name" | tee -a "$PIPELINE_LOG"
    log_resources "INITIAL_SYSTEM_STATE"
    
    if ! monitor_step "$step_name" "$step_name"; then
        echo "Step failed: $step_name" | tee -a "$PIPELINE_LOG"
        echo "Check resource logs: $RESOURCE_LOG" | tee -a "$PIPELINE_LOG"
        exit 1
    fi
    
    echo "Step completed successfully: $step_name" | tee -a "$PIPELINE_LOG"
}

# Main execution
case "${1:-full}" in
    "full")
        run_monitored_pipeline
        ;;
    "step")
        run_monitored_step "$2"
        ;;
    *)
        echo "Usage: $0 [full|step <step_name>]"
        echo "  full: Run complete pipeline with monitoring"
        echo "  step <name>: Run individual step with monitoring"
        exit 1
        ;;
esac 