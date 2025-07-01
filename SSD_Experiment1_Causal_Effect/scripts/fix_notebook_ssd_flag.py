#!/usr/bin/env python3
"""
Fix ssd_flag KeyError in notebook by adding compatibility layer.

This script updates the notebook to handle both 'exposure_flag' (from step 2)
and 'ssd_flag' (from step 8) column names gracefully.
"""

import json
import re
from pathlib import Path
from datetime import datetime
import shutil

def fix_ssd_flag_references(notebook_path):
    """Fix ssd_flag references in notebook to handle both column names."""
    
    # Create backup
    backup_path = notebook_path.with_suffix('.ipynb.bak')
    shutil.copy2(notebook_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Pattern to find exposure_df['ssd_flag'] references
    pattern = r"(exposure_df = pd\.read_parquet\(exposure_path\))\n(\s*)(n_exposed = exposure_df\['ssd_flag'\])"
    
    # Replacement adds compatibility check
    replacement = r"\1\n\2# Add compatibility for different pipeline stages\n\2if 'exposure_flag' in exposure_df.columns and 'ssd_flag' not in exposure_df.columns:\n\2    exposure_df['ssd_flag'] = exposure_df['exposure_flag']\n\2\3"
    
    cells_updated = 0
    
    # Process each code cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Join source lines
            source = ''.join(cell['source'])
            
            # Check if this cell needs updating
            if "exposure_df['ssd_flag']" in source and "pd.read_parquet(exposure_path)" in source:
                # Apply fix
                new_source = re.sub(pattern, replacement, source)
                
                if new_source != source:
                    # Split back into lines and update cell
                    cell['source'] = [line + '\n' for line in new_source.rstrip('\n').split('\n')]
                    cells_updated += 1
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Updated {cells_updated} cells in {notebook_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return cells_updated

def main():
    """Fix ssd_flag references in the main analysis notebook."""
    
    notebook_path = Path("SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return 1
    
    print("Fixing ssd_flag KeyError in notebook...")
    print("=" * 60)
    
    cells_updated = fix_ssd_flag_references(notebook_path)
    
    if cells_updated > 0:
        print("\n✓ Fix applied successfully!")
        print("\nThe notebook now handles both column names:")
        print("  - 'exposure_flag' (from 02_exposure_flag.py)")
        print("  - 'ssd_flag' (from 08_patient_master_table.py)")
        print("\nThis ensures compatibility across all pipeline stages.")
    else:
        print("\n⚠️ No updates needed - notebook may already be fixed.")
    
    return 0

if __name__ == "__main__":
    exit(main())