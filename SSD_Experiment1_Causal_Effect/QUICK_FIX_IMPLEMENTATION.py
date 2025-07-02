#!/usr/bin/env python3
"""
Quick implementation script to fix the pipeline issues
Run this after reading the INTEGRATION_PLAN_FOR_FIXES.md

Author: Assistant to Ryhan Suny
Date: 2025-07-02
"""
import os
import sys
import shutil
from pathlib import Path

def main():
    print("="*80)
    print("SSD Pipeline Quick Fix Implementation")
    print("="*80)
    
    # Check current directory
    if not Path("src").exists() or not Path("data_derived").exists():
        print("ERROR: Must run from project root directory")
        print("Current directory:", os.getcwd())
        return 1
    
    print("\n1. Backing up original files...")
    
    # Backup original files
    backups = [
        ("src/imputed_causal_wrapper.py", "src/imputed_causal_wrapper.ORIGINAL.py"),
        ("src/imputed_causal_pipeline_progress.py", "src/imputed_causal_pipeline_progress.ORIGINAL.py")
    ]
    
    for src, dst in backups:
        if Path(src).exists() and not Path(dst).exists():
            shutil.copy2(src, dst)
            print(f"   ✓ Backed up {src}")
    
    print("\n2. Applying fixes...")
    
    # Fix 1: Replace pipeline progress script
    if Path("src/imputed_causal_pipeline_progress_FIXED.py").exists():
        shutil.copy2("src/imputed_causal_pipeline_progress_FIXED.py", 
                     "src/imputed_causal_pipeline_progress.py")
        print("   ✓ Fixed format string error in pipeline progress")
    
    # Fix 2: Create updated wrapper
    wrapper_fix = '''#!/usr/bin/env python3
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
'''
    
    with open("src/imputed_causal_wrapper_FIXED.py", "w") as f:
        f.write(wrapper_fix)
    print("   ✓ Created fixed wrapper using ssd_flag_adj")
    
    # Create symlink or copy
    if not Path("src/imputed_causal_wrapper_ORIGINAL.py").exists():
        shutil.copy2("src/imputed_causal_wrapper.py", "src/imputed_causal_wrapper_ORIGINAL.py")
    shutil.copy2("src/imputed_causal_wrapper_FIXED.py", "src/imputed_causal_wrapper.py")
    print("   ✓ Replaced wrapper with fixed version")
    
    print("\n3. Verification steps...")
    
    # Check if key files exist
    print("\n   Checking data files:")
    data_files = [
        "data_derived/patient_master.parquet",
        "data_derived/ps_weighted.parquet",
        "data_derived/imputed_master/master_imputed_1.parquet"
    ]
    
    for f in data_files:
        if Path(f).exists():
            print(f"   ✓ {f} exists")
        else:
            print(f"   ✗ {f} NOT FOUND")
    
    # Check treatment variables
    print("\n   Checking treatment variables in patient_master:")
    try:
        import pandas as pd
        master = pd.read_parquet("data_derived/patient_master.parquet")
        treatment_vars = ['ssd_flag', 'ssd_flag_adj', 'H1_normal_labs', 'H2_referral_loop', 'H3_drug_persistence']
        
        for var in treatment_vars:
            if var in master.columns:
                n_treated = master[var].sum()
                pct = n_treated / len(master) * 100
                print(f"   ✓ {var}: {n_treated} treated ({pct:.1f}%)")
            else:
                print(f"   ✗ {var}: NOT FOUND")
    except Exception as e:
        print(f"   ERROR checking variables: {e}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Re-run Step 2 in notebook with --logic both:")
    print('   result = run_pipeline_script("02_exposure_flag.py", args="--logic both")')
    print("\n2. Re-run Steps 8-15 to propagate the fixes")
    print("\n3. Use multi-hypothesis wrapper for comprehensive testing:")
    print("   from src.imputed_causal_wrapper_multi_hypothesis import run_multi_hypothesis_estimation")
    print("\n4. Monitor that:")
    print("   - Exposed count > 40,000 for adjusted flag")
    print("   - PS matching achieves SMD < 0.1")
    print("   - TMLE estimates are non-zero")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())