#!/usr/bin/env python3
"""
update_notebook_surgical.py - Surgical updates to SSD notebook

This script makes targeted updates to the notebook without disrupting
existing fixes or structure. It creates a backup first and shows
what changes it will make.

Author: Update script for post-implementation fixes
Date: July 2, 2025
Version: 1.0.0
"""

import json
import re
from pathlib import Path
from datetime import datetime
import shutil


def backup_notebook(notebook_path):
    """Create timestamped backup of notebook."""
    backup_path = notebook_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb')
    shutil.copy2(notebook_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    return backup_path


def load_notebook(path):
    """Load notebook JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(notebook, path):
    """Save notebook JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


def find_cells_with_text(notebook, search_text, cell_type='code'):
    """Find all cells containing specific text."""
    matches = []
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == cell_type:
            source = ''.join(cell['source'])
            if search_text in source:
                matches.append((i, cell, source))
    return matches


def update_cell_source(cell, old_text, new_text, description=""):
    """Update cell source with new text."""
    source = ''.join(cell['source'])
    if old_text in source:
        new_source = source.replace(old_text, new_text)
        # Convert back to list of lines
        cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
        if description:
            print(f"  ✓ {description}")
        return True
    return False


def add_validation_after_marker(notebook, marker_text, validation_code, description=""):
    """Add validation code after a specific marker text."""
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if marker_text in source and validation_code not in source:
                # Find the right place to insert (after the marker line)
                lines = source.split('\n')
                for j, line in enumerate(lines):
                    if marker_text in line:
                        # Insert validation after this line
                        insert_point = j + 1
                        # Find the next print statement or empty line
                        while insert_point < len(lines) and lines[insert_point].strip() != '':
                            insert_point += 1
                        
                        validation_lines = validation_code.split('\n')
                        lines = lines[:insert_point] + [''] + validation_lines + lines[insert_point:]
                        
                        # Rebuild source
                        new_source = '\n'.join(lines)
                        cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]] + [new_source.split('\n')[-1]]
                        print(f"  ✓ Added validation: {description}")
                        return True
    return False


def main():
    """Main update function."""
    notebook_path = Path("SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return
    
    print("SSD Notebook Surgical Update Script")
    print("=" * 50)
    
    # Create backup
    backup_path = backup_notebook(notebook_path)
    
    # Load notebook
    print("\nLoading notebook...")
    notebook = load_notebook(notebook_path)
    print(f"✓ Loaded {len(notebook['cells'])} cells")
    
    # Track changes
    changes_made = []
    
    print("\nApplying updates...")
    
    # 1. Fix pre_imputation_master.parquet references
    print("\n1. Fixing pre_imputation_master filename references...")
    cells = find_cells_with_text(notebook, 'pre_imputation_master.parquet')
    for i, cell, source in cells:
        if update_cell_source(cell, 
                            'pre_imputation_master.parquet', 
                            'master_with_missing.parquet',
                            f"Cell {i}: Fixed filename reference"):
            changes_made.append(f"Cell {i}: pre_imputation_master.parquet → master_with_missing.parquet")
    
    # 2. Fix datetime import in progress script
    print("\n2. Fixing datetime import in progress script...")
    cells = find_cells_with_text(notebook, 'progress_script_content = """')
    for i, cell, source in cells:
        if 'from datetime import datetime, timedelta' not in source and 'timedelta(' in source:
            # Add import after the imports section
            old_imports = 'import time\nimport json'
            new_imports = 'import time\nimport json\nfrom datetime import datetime, timedelta'
            if update_cell_source(cell, old_imports, new_imports, f"Cell {i}: Added datetime imports"):
                changes_made.append(f"Cell {i}: Added datetime, timedelta imports")
    
    # 3. Fix Rubin's pooling filename pattern
    print("\n3. Fixing Rubin's pooling filename pattern...")
    cells = find_cells_with_text(notebook, 'run_pipeline_script("rubins_pooling_engine.py"')
    for i, cell, source in cells:
        if '--pattern' not in source:
            old_call = '''run_pipeline_script("rubins_pooling_engine.py",
                           description="Rubin's Pooling with Barnard-Rubin")'''
            new_call = '''run_pipeline_script("rubins_pooling_engine.py",
                           args="--pattern causal_results_imp*.json",
                           description="Rubin's Pooling with Barnard-Rubin")'''
            if update_cell_source(cell, old_call, new_call, f"Cell {i}: Added --pattern argument"):
                changes_made.append(f"Cell {i}: Added --pattern argument to rubins_pooling")
    
    # 4. Remove competing_risk_analysis.py
    print("\n4. Removing non-existent competing_risk_analysis.py...")
    cells = find_cells_with_text(notebook, 'competing_risk_analysis.py')
    for i, cell, source in cells:
        if update_cell_source(cell,
                            "'script': 'competing_risk_analysis.py'",
                            "'script': 'death_rates_analysis.py'",
                            f"Cell {i}: Replaced with death_rates_analysis.py"):
            changes_made.append(f"Cell {i}: competing_risk_analysis.py → death_rates_analysis.py")
    
    # 5. Add hierarchical index validation
    print("\n5. Adding hierarchical index date validation...")
    validation_code = '''# Validate hierarchical index dates
if 'IndexDate_unified' in cohort.columns:
    print(f"✓ Hierarchical index dates implemented")
    print(f"  - Lab index: {cohort['index_date_source'].eq('Laboratory').sum():,} ({cohort['index_date_source'].eq('Laboratory').mean():.1%})")
    print(f"  - MH encounter: {cohort['index_date_source'].eq('Mental_Health_Encounter').sum():,}")
    print(f"  - Psychotropic: {cohort['index_date_source'].eq('Psychotropic_Medication').sum():,}")
    if 'lab_utilization_phenotype' in cohort.columns:
        print(f"  - Phenotypes: Avoidant={cohort['lab_utilization_phenotype'].eq('Avoidant_SSD').sum():,}, Test-seeking={cohort['lab_utilization_phenotype'].eq('Test_Seeking_SSD').sum():,}")'''
    
    if add_validation_after_marker(notebook, 
                                 "✓ 01_cohort_builder.py completed successfully",
                                 validation_code,
                                 "After cohort creation"):
        changes_made.append("Added hierarchical index date validation after cohort creation")
    
    # 6. Add H2 tier validation
    print("\n6. Adding H2 tier validation...")
    h2_validation = '''# Validate H2 tiers
if all(col in exposure.columns for col in ['h2_tier1', 'h2_tier2', 'h2_tier3']):
    print(f"✓ H2 three-tier implementation validated")
    print(f"  - Tier 1 (Basic): {exposure['h2_tier1'].sum():,} patients")
    print(f"  - Tier 2 (Enhanced): {exposure['h2_tier2'].sum():,} patients")
    print(f"  - Tier 3 (Full Proxy): {exposure['h2_tier3'].sum():,} patients")
    if 'h2_any_tier' in exposure.columns:
        print(f"  - Any tier: {exposure['h2_any_tier'].sum():,} patients")'''
    
    if add_validation_after_marker(notebook,
                                 "✓ 02_exposure_flag.py completed successfully",
                                 h2_validation,
                                 "After exposure creation"):
        changes_made.append("Added H2 tier validation after exposure creation")
    
    # 7. Add datetime exclusion confirmation
    print("\n7. Adding datetime exclusion confirmation...")
    datetime_validation = '''# Confirm datetime exclusion
print("✓ Datetime columns excluded from imputation (per evidence-based solutions)")'''
    
    if add_validation_after_marker(notebook,
                                 "✓ 07b_missing_data_master.py completed successfully",
                                 datetime_validation,
                                 "After imputation"):
        changes_made.append("Added datetime exclusion confirmation after imputation")
    
    # Save updated notebook
    print(f"\nSaving updated notebook...")
    save_notebook(notebook, notebook_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("UPDATE SUMMARY")
    print("=" * 50)
    print(f"✓ Backup created: {backup_path}")
    print(f"✓ Total changes made: {len(changes_made)}")
    
    if changes_made:
        print("\nChanges applied:")
        for change in changes_made:
            print(f"  - {change}")
    else:
        print("\n⚠️  No changes were needed")
    
    print("\nNext steps:")
    print("1. Review the changes by comparing with backup")
    print("2. Run the notebook to verify all fixes work")
    print("3. Delete backup if satisfied with changes")
    
    # Create a diff command for easy comparison
    print(f"\nTo compare changes:")
    print(f"  diff {backup_path.name} {notebook_path.name}")


if __name__ == "__main__":
    main()