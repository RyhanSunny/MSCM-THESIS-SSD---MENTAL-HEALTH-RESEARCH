#!/usr/bin/env python3
"""
validate_notebook_completeness.py - Check notebook for potential issues

This script validates the notebook against known implementation changes
to identify any additional updates needed.

Author: Validation script
Date: July 2, 2025
Version: 1.0.0
"""

import json
import re
from pathlib import Path


def load_notebook(path):
    """Load notebook JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_for_patterns(notebook):
    """Check for various patterns that might need updating."""
    issues = []
    
    # Patterns to check
    checks = [
        {
            'name': 'IndexDate_unified references',
            'pattern': r'IndexDate_unified',
            'expected': True,
            'message': 'Should reference IndexDate_unified for hierarchical dates'
        },
        {
            'name': 'index_date_source references',
            'pattern': r'index_date_source',
            'expected': True,
            'message': 'Should check index_date_source for date hierarchy'
        },
        {
            'name': 'lab_utilization_phenotype references',
            'pattern': r'lab_utilization_phenotype',
            'expected': True,
            'message': 'Should validate phenotype assignment'
        },
        {
            'name': 'h2_tier references',
            'pattern': r'h2_tier[123]|h2_any_tier',
            'expected': True,
            'message': 'Should validate H2 tier implementation'
        },
        {
            'name': 'pre_imputation_master.parquet',
            'pattern': r'pre_imputation_master\.parquet',
            'expected': False,
            'message': 'Should use master_with_missing.parquet instead'
        },
        {
            'name': 'competing_risk_analysis.py',
            'pattern': r'competing_risk_analysis\.py',
            'expected': False,
            'message': 'Script does not exist, use death_rates_analysis.py'
        },
        {
            'name': 'datetime exclusion mention',
            'pattern': r'datetime.*exclud|exclud.*datetime',
            'expected': True,
            'message': 'Should mention datetime exclusion from imputation'
        },
        {
            'name': 'causal_results_imp vs causal_estimates_imp',
            'pattern': r'causal_results_imp\d+\.json',
            'expected': True,
            'message': 'Check if rubins pooling expects different filename'
        }
    ]
    
    # Check each pattern
    for check in checks:
        found = False
        found_cells = []
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if re.search(check['pattern'], source, re.IGNORECASE):
                    found = True
                    found_cells.append(i)
        
        if found != check['expected']:
            if check['expected']:
                issues.append({
                    'type': 'missing',
                    'check': check['name'],
                    'message': check['message']
                })
            else:
                issues.append({
                    'type': 'unwanted',
                    'check': check['name'],
                    'message': check['message'],
                    'cells': found_cells
                })
        elif found and check['expected']:
            print(f"✓ Found {check['name']} in cells: {found_cells[:5]}{'...' if len(found_cells) > 5 else ''}")
    
    return issues


def check_script_references(notebook):
    """Check that all referenced scripts exist."""
    script_pattern = r'run_pipeline_script\("([^"]+\.py)"'
    referenced_scripts = set()
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            matches = re.findall(script_pattern, source)
            referenced_scripts.update(matches)
    
    src_dir = Path('src')
    missing_scripts = []
    
    for script in referenced_scripts:
        script_path = src_dir / script
        if not script_path.exists():
            missing_scripts.append(script)
    
    return referenced_scripts, missing_scripts


def check_data_flow(notebook):
    """Check data flow consistency."""
    flow_issues = []
    
    # Check for file paths
    file_patterns = {
        'master_with_missing.parquet': r'master_with_missing\.parquet',
        'pre_imputation_master.parquet': r'pre_imputation_master\.parquet',
        'patient_master.parquet': r'patient_master\.parquet',
        'cohort.parquet': r'cohort\.parquet',
        'exposure.parquet': r'exposure\.parquet'
    }
    
    file_usage = {}
    for filename, pattern in file_patterns.items():
        cells = []
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if re.search(pattern, source):
                    cells.append(i)
        if cells:
            file_usage[filename] = cells
    
    # Check for conflicts
    if 'master_with_missing.parquet' in file_usage and 'pre_imputation_master.parquet' in file_usage:
        flow_issues.append("Both master_with_missing.parquet and pre_imputation_master.parquet are referenced")
    
    return file_usage, flow_issues


def main():
    """Main validation function."""
    notebook_path = Path("SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return
    
    print("SSD Notebook Completeness Validation")
    print("=" * 50)
    
    # Load notebook
    notebook = load_notebook(notebook_path)
    print(f"Loaded {len(notebook['cells'])} cells")
    
    # Check patterns
    print("\nChecking for implementation patterns...")
    issues = check_for_patterns(notebook)
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} potential issues:")
        for issue in issues:
            if issue['type'] == 'missing':
                print(f"  - MISSING: {issue['check']} - {issue['message']}")
            else:
                print(f"  - UNWANTED: {issue['check']} in cells {issue['cells']} - {issue['message']}")
    
    # Check scripts
    print("\nChecking script references...")
    referenced, missing = check_script_references(notebook)
    print(f"Total scripts referenced: {len(referenced)}")
    
    if missing:
        print(f"\n❌ Missing scripts:")
        for script in missing:
            print(f"  - {script}")
    else:
        print("✓ All referenced scripts exist")
    
    # Check data flow
    print("\nChecking data flow...")
    file_usage, flow_issues = check_data_flow(notebook)
    
    print("\nFile usage:")
    for filename, cells in file_usage.items():
        print(f"  - {filename}: used in {len(cells)} cells")
    
    if flow_issues:
        print(f"\n⚠️  Data flow issues:")
        for issue in flow_issues:
            print(f"  - {issue}")
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    total_issues = len(issues) + len(missing) + len(flow_issues)
    if total_issues == 0:
        print("✓ No issues found!")
    else:
        print(f"⚠️  Total issues found: {total_issues}")
        print("\nRecommended actions:")
        print("1. Run update_notebook_surgical.py to fix known issues")
        print("2. Manually review and fix any remaining issues")
        print("3. Re-run this validation to confirm")


if __name__ == "__main__":
    main()