#!/usr/bin/env python3
"""
Fixed wrapper using MC-SIMEX adjusted treatment variable
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

def run_causal_estimation_on_imputation(df, imp_num):
    """Run all causal methods on a single imputed dataset"""
    from src.config_loader import load_config
    
    # Get config
    config = load_config()
    
    # Define variables
    outcome_col = 'total_encounters'
    treatment_col = 'ssd_flag_adj'  # FIXED: Using MC-SIMEX adjusted flag
    
    # Check if adjusted flag exists, fallback to H1 if not
    if treatment_col not in df.columns:
        print(f"    WARNING: {treatment_col} not found, trying H1_normal_labs")
        treatment_col = 'H1_normal_labs'
        if treatment_col not in df.columns:
            print(f"    ERROR: No suitable treatment column found")
            return {'error': 'No treatment column found'}
    
    # Handle age column compatibility
    age_col = 'age' if 'age' in df.columns else 'Age_at_2015'
    base_covariates = ['sex_M', 'charlson_score', 'baseline_encounters', 'baseline_high_utilizer']
    if age_col in df.columns:
        base_covariates.insert(0, age_col)
    
    # Add confounder columns
    covariate_cols = [col for col in df.columns if col.endswith('_conf') or col in base_covariates]
    covariate_cols = [col for col in covariate_cols if col in df.columns]
    
    results = {
        'imputation': imp_num,
        'treatment_variable': treatment_col,
        'n_obs': len(df),
        'n_treated': int(df[treatment_col].sum()),
        'estimates': []
    }
    
    print(f"    Using treatment: {treatment_col} ({results['n_treated']} treated)")
    
    # Import estimation functions
    from src.causal_estimators_simplified import (
        run_tmle_simple, run_dml_simple, run_causal_forest_simple
    )
    
    # Run each method
    methods = [
        ('TMLE', run_tmle_simple),
        ('Double ML', run_dml_simple),
        ('Causal Forest', run_causal_forest_simple)
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"    Running {method_name}...")
            method_results = method_func(df, outcome_col, treatment_col, covariate_cols)
            results['estimates'].append(method_results)
        except Exception as e:
            print(f"    {method_name} failed: {str(e)}")
            results['estimates'].append({
                'method': method_name,
                'estimate': None,
                'ci_lower': None,
                'ci_upper': None,
                'error': str(e)
            })
    
    return results
