#!/bin/bash
# make_wrapper.sh - Bash wrapper for make operations with comprehensive logging
# Following CLAUDE.md requirements for timestamped execution logs

# Usage: ./scripts/make_wrapper.sh causal
#        ./scripts/make_wrapper.sh all

set -euo pipefail

# Check if python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please ensure Python is installed and in PATH."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Check if make_logger.py exists
if [[ ! -f "scripts/make_logger.py" ]]; then
    echo "❌ make_logger.py not found in scripts directory"
    exit 1
fi

# Execute make with logging
echo "🔧 Executing make with comprehensive logging..."
python scripts/make_logger.py make "$@"