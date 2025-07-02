#!/usr/bin/env python3
"""
Wrapper for running causal estimation on a single imputation
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
    treatment_col = 'exposure_flag'  # Using exposure_flag directly
    
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
        'n_obs': len(df),
        'n_treated': int(df[treatment_col].sum()),
        'estimates': []
    }
    
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
