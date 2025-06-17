"""
Compatibility module for importing reconcile_estimates functionality.
This module re-exports all functions from 16_reconcile_estimates.py.
"""

# Import all functions from the actual implementation
# Note: Python doesn't allow imports from modules starting with numbers
# This is a placeholder file to maintain compatibility

import importlib
import sys

# Import the module with numeric prefix using importlib
reconcile_module = importlib.import_module('src.16_reconcile_estimates')

# Re-export all functions
compare_ate_estimates = reconcile_module.compare_ate_estimates
reconcile_causal_estimates = reconcile_module.reconcile_causal_estimates
load_estimate_yaml = reconcile_module.load_estimate_yaml
create_reconciliation_report = reconcile_module.create_reconciliation_report

# Ensure the module can be imported directly
__all__ = [
    'compare_ate_estimates',
    'reconcile_causal_estimates',
    'load_estimate_yaml',
    'create_reconciliation_report'
] 