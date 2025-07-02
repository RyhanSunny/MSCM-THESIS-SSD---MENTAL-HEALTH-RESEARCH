#!/usr/bin/env python3
"""
Multi-hypothesis wrapper for running causal estimation on each hypothesis separately
Following the methodology blueprint that defines H1, H2, H3 as separate hypotheses

Author: Assistant to Ryhan Suny
Date: 2025-07-02
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

def run_causal_estimation_on_imputation(df: pd.DataFrame, 
                                       imp_num: int, 
                                       treatment_col: str = 'ssd_flag_adj') -> Dict[str, Any]:
    """
    Run all causal methods on a single imputed dataset with specified treatment
    
    Parameters:
    -----------
    df : pd.DataFrame
        Imputed dataset
    imp_num : int
        Imputation number
    treatment_col : str
        Treatment column to use (default: ssd_flag_adj)
        
    Returns:
    --------
    Dict containing causal estimates for all methods
    """
    from src.config_loader import load_config
    
    # Get config
    config = load_config()
    
    # Define variables
    outcome_col = 'total_encounters'
    
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
        'n_treated': int(df[treatment_col].sum()) if treatment_col in df.columns else 0,
        'estimates': []
    }
    
    # Check if treatment column exists
    if treatment_col not in df.columns:
        results['error'] = f"Treatment column '{treatment_col}' not found in data"
        return results
    
    # Check if we have enough treated units
    n_treated = df[treatment_col].sum()
    if n_treated < 10:
        results['warning'] = f"Only {n_treated} treated units - results may be unstable"
    
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
            print(f"    Running {method_name} with {treatment_col}...")
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


def run_multi_hypothesis_estimation(df: pd.DataFrame, imp_num: int) -> Dict[str, Dict[str, Any]]:
    """
    Run causal estimation for each hypothesis separately
    
    Following the methodology blueprint:
    - H1: Normal labs (≥3) → Healthcare utilization
    - H2: Referral loops (≥2) → Healthcare utilization  
    - H3: Drug persistence (≥180 days) → Healthcare utilization
    - Combined: OR logic of H1/H2/H3
    - Adjusted: MC-SIMEX adjusted exposure
    
    Parameters:
    -----------
    df : pd.DataFrame
        Imputed dataset
    imp_num : int
        Imputation number
        
    Returns:
    --------
    Dict with results for each hypothesis
    """
    hypothesis_configs = [
        ('H1_NormalLabs', 'H1_normal_labs', 'Diagnostic cascade hypothesis'),
        ('H2_ReferralLoop', 'H2_referral_loop', 'Specialist referral loop hypothesis'),
        ('H3_DrugPersistence', 'H3_drug_persistence', 'Medication persistence hypothesis'),
        ('Combined_AND', 'ssd_flag', 'Combined exposure (AND logic) - original'),
        ('Combined_AND_strict', 'ssd_flag_strict', 'Combined exposure (AND logic) - strict'),
        ('MC_SIMEX_Adjusted', 'ssd_flag_adj', 'MC-SIMEX misclassification adjusted'),
        ('Exposure_OR', 'exposure_flag', 'Combined exposure (OR logic) if available')
    ]
    
    results = {
        'imputation': imp_num,
        'hypotheses': {}
    }
    
    for hyp_name, treatment_col, description in hypothesis_configs:
        if treatment_col in df.columns:
            print(f"\n  Testing {hyp_name}: {description}")
            n_treated = df[treatment_col].sum()
            print(f"    Exposed: {n_treated} ({n_treated/len(df)*100:.1f}%)")
            
            # Run causal estimation
            hyp_results = run_causal_estimation_on_imputation(df, imp_num, treatment_col)
            hyp_results['description'] = description
            results['hypotheses'][hyp_name] = hyp_results
        else:
            print(f"\n  Skipping {hyp_name}: column '{treatment_col}' not found")
            results['hypotheses'][hyp_name] = {
                'status': 'skipped',
                'reason': f"Column '{treatment_col}' not found in data"
            }
    
    # Add summary statistics
    results['summary'] = {
        'n_hypotheses_tested': sum(1 for h in results['hypotheses'].values() 
                                  if h.get('status') != 'skipped'),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    return results


def compare_hypothesis_results(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create comparison table of results across hypotheses
    
    Parameters:
    -----------
    results : Dict
        Results from run_multi_hypothesis_estimation
        
    Returns:
    --------
    pd.DataFrame with comparison of estimates across hypotheses
    """
    comparison_data = []
    
    for hyp_name, hyp_results in results['hypotheses'].items():
        if hyp_results.get('status') == 'skipped':
            continue
            
        row = {
            'hypothesis': hyp_name,
            'n_treated': hyp_results.get('n_treated', 0),
            'pct_treated': hyp_results.get('n_treated', 0) / hyp_results.get('n_obs', 1) * 100
        }
        
        # Extract estimates for each method
        for estimate in hyp_results.get('estimates', []):
            method = estimate.get('method', 'Unknown')
            row[f'{method}_estimate'] = estimate.get('estimate')
            row[f'{method}_ci_lower'] = estimate.get('ci_lower')
            row[f'{method}_ci_upper'] = estimate.get('ci_upper')
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    # Example usage
    print("Multi-hypothesis causal wrapper ready")
    print("Use run_multi_hypothesis_estimation(df, imp_num) to test all hypotheses")
    print("Use run_causal_estimation_on_imputation(df, imp_num, treatment_col) for single hypothesis")