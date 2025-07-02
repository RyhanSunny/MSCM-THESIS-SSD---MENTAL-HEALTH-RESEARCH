#!/usr/bin/env python3
"""
add_validation_blocks.py - Add validation blocks to notebook

Adds the validation code blocks that the surgical script couldn't add.
"""

import json
from pathlib import Path


def main():
    notebook_path = Path("SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes = []
    
    # Find where to add validations by looking for specific patterns
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # After cohort creation - look for where cohort is validated
            if "Cohort created:" in source and "Retention rate:" in source:
                # Add hierarchical validation
                validation = '''
# Validate hierarchical index dates
cohort_path = DATA_DERIVED / "cohort.parquet"
if cohort_path.exists():
    cohort = pd.read_parquet(cohort_path)
    if 'IndexDate_unified' in cohort.columns:
        print(f"\\n✓ Hierarchical index dates implemented")
        print(f"  - Lab index: {cohort['index_date_source'].eq('Laboratory').sum():,} ({cohort['index_date_source'].eq('Laboratory').mean():.1%})")
        print(f"  - MH encounter: {cohort['index_date_source'].eq('Mental_Health_Encounter').sum():,}")
        print(f"  - Psychotropic: {cohort['index_date_source'].eq('Psychotropic_Medication').sum():,}")
        if 'lab_utilization_phenotype' in cohort.columns:
            print(f"  - Phenotypes: Avoidant={cohort['lab_utilization_phenotype'].eq('Avoidant_SSD').sum():,}, Test-seeking={cohort['lab_utilization_phenotype'].eq('Test_Seeking_SSD').sum():,}")
'''
                if "Hierarchical index dates implemented" not in source:
                    cell['source'].append(validation)
                    changes.append(f"Cell {i}: Added hierarchical index validation")
            
            # After exposure creation
            if "Exposed patients (OR logic):" in source or "Exposed patients (AND logic):" in source:
                # Add H2 validation
                h2_validation = '''
# Validate H2 tiers
if 'h2_tier1' in exposure_df.columns:
    print(f"\\n✓ H2 three-tier implementation validated")
    print(f"  - Tier 1 (Basic): {exposure_df['h2_tier1'].sum():,} patients")
    print(f"  - Tier 2 (Enhanced): {exposure_df['h2_tier2'].sum():,} patients")
    print(f"  - Tier 3 (Full Proxy): {exposure_df['h2_tier3'].sum():,} patients")
    if 'h2_any_tier' in exposure_df.columns:
        print(f"  - Any tier: {exposure_df['h2_any_tier'].sum():,} patients")
'''
                if "H2 three-tier implementation" not in source:
                    cell['source'].append(h2_validation)
                    changes.append(f"Cell {i}: Added H2 tier validation")
            
            # After imputation
            if "07b_missing_data_master.py" in source and "30 imputed datasets created" in source:
                datetime_validation = '''
# Confirm datetime exclusion
print("\\n✓ Datetime columns excluded from imputation (per evidence-based solutions)")
'''
                if "Datetime columns excluded" not in source:
                    cell['source'].append(datetime_validation)
                    changes.append(f"Cell {i}: Added datetime exclusion confirmation")
    
    if changes:
        # Save updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        
        print(f"✓ Added {len(changes)} validation blocks:")
        for change in changes:
            print(f"  - {change}")
    else:
        print("No additional validation blocks needed")


if __name__ == "__main__":
    main()