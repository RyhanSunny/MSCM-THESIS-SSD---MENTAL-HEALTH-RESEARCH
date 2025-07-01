#!/usr/bin/env python3
"""
Comprehensive fix for ssd_flag vs exposure_flag naming issue.

This script adds a compatibility layer to handle the naming difference between:
- Early pipeline stages (exposure_flag)
- Analysis stages (ssd_flag)
"""

import json
import re
from pathlib import Path
from datetime import datetime
import shutil
import sys

def create_compatibility_snippet():
    """Create the compatibility code snippet."""
    return """    # Add compatibility for ssd_flag vs exposure_flag naming
    if 'exposure_flag' in exposure_df.columns and 'ssd_flag' not in exposure_df.columns:
        exposure_df['ssd_flag'] = exposure_df['exposure_flag']
    if 'exposure_flag_strict' in exposure_df.columns and 'ssd_flag_strict' not in exposure_df.columns:
        exposure_df['ssd_flag_strict'] = exposure_df['exposure_flag_strict']"""

def fix_notebook_comprehensively(notebook_path):
    """Apply comprehensive fix to notebook."""
    
    # Create timestamped backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = notebook_path.with_suffix(f'.ipynb.backup_{timestamp}')
    shutil.copy2(notebook_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    compatibility_snippet = create_compatibility_snippet()
    cells_updated = 0
    
    # Process each code cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            source = ''.join(source_lines)
            
            # Check if this cell loads exposure data and uses ssd_flag
            if ('pd.read_parquet' in source and 
                'exposure' in source and 
                ("exposure_df['ssd_flag']" in source or 'exposure_df["ssd_flag"]' in source)):
                
                # Check if compatibility fix already exists
                if 'Add compatibility for ssd_flag' in source:
                    print(f"Cell {i} already has compatibility fix")
                    continue
                
                new_lines = []
                j = 0
                while j < len(source_lines):
                    line = source_lines[j]
                    new_lines.append(line)
                    
                    # After reading parquet file, add compatibility layer
                    if 'pd.read_parquet(exposure_path)' in line:
                        # Find the indentation of the next line
                        if j + 1 < len(source_lines):
                            next_line = source_lines[j + 1]
                            indent = len(next_line) - len(next_line.lstrip())
                            
                            # Add compatibility snippet with proper indentation
                            compat_lines = compatibility_snippet.split('\n')
                            for compat_line in compat_lines:
                                if compat_line.strip():  # Skip empty lines
                                    new_lines.append(compat_line[4:] + '\n')  # Remove 4 spaces and add newline
                    j += 1
                
                # Update cell source
                cell['source'] = new_lines
                cells_updated += 1
                print(f"Updated cell {i}")
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    return cells_updated

def verify_fix(notebook_path):
    """Verify the fix was applied correctly."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count occurrences
    ssd_flag_count = content.count("exposure_df['ssd_flag']")
    compat_count = content.count("Add compatibility for ssd_flag")
    
    print(f"\nVerification:")
    print(f"  - Total ssd_flag references: {ssd_flag_count}")
    print(f"  - Compatibility fixes added: {compat_count}")
    
    return compat_count > 0

def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        notebook_path = Path(sys.argv[1])
    else:
        notebook_path = Path("SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return 1
    
    print(f"Fixing ssd_flag naming issue in: {notebook_path}")
    print("=" * 70)
    
    # Apply fix
    cells_updated = fix_notebook_comprehensively(notebook_path)
    
    print(f"\n✓ Updated {cells_updated} cells")
    
    # Verify
    if verify_fix(notebook_path):
        print("\n✅ Fix applied successfully!")
        print("\nThe notebook now handles both naming conventions:")
        print("  - 'exposure_flag' → 'ssd_flag'")
        print("  - 'exposure_flag_strict' → 'ssd_flag_strict'")
        print("\nThis ensures the notebook works at any pipeline stage.")
    else:
        print("\n⚠️ Warning: Fix may not have been applied correctly.")
    
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0

if __name__ == "__main__":
    exit(main())