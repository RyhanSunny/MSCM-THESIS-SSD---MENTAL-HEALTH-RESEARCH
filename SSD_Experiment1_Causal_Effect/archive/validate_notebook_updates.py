#!/usr/bin/env python3
"""
Validation script to check if notebook updates were successful
Run after updating the notebook to verify all changes
"""

import json
import sys
from pathlib import Path

def validate_notebook(notebook_path):
    """Check if all required updates are in place"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    issues = []
    fixes = []
    
    # Check all cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Check 1: Pre-imputation master filename
            if 'pre_imputation_master.parquet' in source:
                issues.append(f"Cell {i}: Still references old filename 'pre_imputation_master.parquet'")
            elif 'master_with_missing.parquet' in source:
                fixes.append(f"Cell {i}: Correctly uses 'master_with_missing.parquet'")
            
            # Check 2: Progress script imports
            if 'progress_script_content = ' in source and 'import time' in source:
                if 'from datetime import datetime, timedelta' not in source:
                    issues.append(f"Cell {i}: Progress script missing timedelta import")
                else:
                    fixes.append(f"Cell {i}: Progress script has correct imports")
            
            # Check 3: Rubin's pooling pattern
            if 'rubins_pooling_engine.py' in source:
                if '--pattern causal_results_imp*.json' in source:
                    fixes.append(f"Cell {i}: Rubin's pooling has correct pattern argument")
                else:
                    issues.append(f"Cell {i}: Rubin's pooling may have filename mismatch")
            
            # Check 4: Competing risk script
            if 'competing_risk_analysis.py' in source:
                issues.append(f"Cell {i}: References non-existent 'competing_risk_analysis.py'")
            
            # Check 5: Hierarchical index validation
            if 'IndexDate_unified' in source and 'index_date_source' in source:
                fixes.append(f"Cell {i}: Has hierarchical index date validation")
            
            # Check 6: H2 tier validation
            if 'h2_tier2_enhanced' in source or 'h2_tier3_full' in source:
                fixes.append(f"Cell {i}: Has H2 tier validation")
            
            # Check 7: Datetime exclusion confirmation
            if 'Datetime exclusion confirmed' in source or 'datetime columns excluded' in source:
                fixes.append(f"Cell {i}: Has datetime exclusion confirmation")
    
    # Print results
    print("="*60)
    print("NOTEBOOK VALIDATION RESULTS")
    print("="*60)
    
    if fixes:
        print(f"\n✅ FIXES FOUND ({len(fixes)}):")
        for fix in fixes:
            print(f"  {fix}")
    
    if issues:
        print(f"\n❌ ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ NO ISSUES FOUND - All updates appear to be in place!")
    
    print("\n" + "="*60)
    return len(issues) == 0

if __name__ == "__main__":
    notebook_path = Path("SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)
    
    success = validate_notebook(notebook_path)
    sys.exit(0 if success else 1)