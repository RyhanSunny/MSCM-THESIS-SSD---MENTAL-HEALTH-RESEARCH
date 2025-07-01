#!/usr/bin/env python3
"""
Fix the indentation issue in the notebook cells after adding compatibility code.
"""

import json
from pathlib import Path
from datetime import datetime
import shutil

def fix_indentation_in_notebook(notebook_path):
    """Fix indentation issues in the notebook."""
    
    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = notebook_path.with_suffix(f'.ipynb.backup_indent_{timestamp}')
    shutil.copy2(notebook_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells_fixed = 0
    
    # Process each code cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            
            # Check if this cell has the compatibility code
            has_compat = any('Add compatibility for ssd_flag' in line for line in source_lines)
            if not has_compat:
                continue
                
            # Fix the specific pattern we're looking for
            new_lines = []
            for j, line in enumerate(source_lines):
                new_lines.append(line)
                
                # After the last compatibility line, ensure proper indentation
                if 'exposure_df[\'ssd_flag_strict\'] = exposure_df[\'exposure_flag_strict\']' in line:
                    # The next lines should maintain the same indentation as the if statement
                    base_indent = 4  # spaces for the if block level
                    cells_fixed += 1
            
            # Check if we need to fix this cell
            needs_fix = False
            for j, line in enumerate(source_lines):
                if j > 0 and 'n_exposed = exposure_df' in line and not line.startswith('    '):
                    needs_fix = True
                    break
            
            if needs_fix:
                # Rebuild the cell with correct indentation
                new_lines = []
                inside_if_block = False
                
                for line in source_lines:
                    if 'if exposure_path.exists():' in line:
                        inside_if_block = True
                        new_lines.append(line)
                    elif 'else:' in line and inside_if_block:
                        inside_if_block = False
                        new_lines.append(line)
                    elif inside_if_block and line.strip() and not line.startswith('    '):
                        # This line should be indented but isn't
                        new_lines.append('    ' + line.lstrip())
                    else:
                        new_lines.append(line)
                
                cell['source'] = new_lines
                cells_fixed += 1
                print(f"Fixed indentation in cell {i}")
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    return cells_fixed

def main():
    """Fix indentation issues."""
    notebook_path = Path("SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return 1
    
    print("Checking and fixing indentation issues...")
    print("=" * 60)
    
    cells_fixed = fix_indentation_in_notebook(notebook_path)
    
    if cells_fixed > 0:
        print(f"\n✓ Fixed indentation in {cells_fixed} cells")
    else:
        print("\n✓ No indentation issues found")
    
    return 0

if __name__ == "__main__":
    exit(main())