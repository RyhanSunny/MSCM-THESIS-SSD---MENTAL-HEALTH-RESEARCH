#!/usr/bin/env python3
"""
Fix the indentation issue in notebook cells where compatibility code was added incorrectly.
"""

import json
from pathlib import Path
from datetime import datetime
import shutil

def fix_cell_indentation(source_lines):
    """Fix indentation in a cell that has the compatibility code."""
    new_lines = []
    
    for i, line in enumerate(source_lines):
        if i == 15 and line.strip() == "# Add compatibility for ssd_flag vs exposure_flag naming":
            # This line and the next 4 lines need to be indented by 4 more spaces
            new_lines.append('    ' + line)
        elif i in [16, 17, 18, 19] and not line.startswith('        '):
            # These lines need proper indentation
            new_lines.append('    ' + line)
        else:
            new_lines.append(line)
    
    return new_lines

def fix_notebook_indentation(notebook_path):
    """Fix the indentation in the notebook."""
    
    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = notebook_path.with_suffix(f'.ipynb.backup_final_{timestamp}')
    shutil.copy2(notebook_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells_fixed = 0
    
    # Check cell 11 specifically (and any others with the same pattern)
    for idx, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Check if this cell has the misindented compatibility code
            if ('# Add compatibility for ssd_flag vs exposure_flag naming' in source and
                'if exposure_path.exists():' in source and
                source.find('# Add compatibility') < source.find("n_exposed = exposure_df['ssd_flag']")):
                
                # This cell needs fixing
                print(f"Fixing cell {idx}")
                
                # Reconstruct the cell with proper indentation
                new_source_lines = []
                inside_if_block = False
                
                for line in cell['source']:
                    if line.strip() == 'if exposure_path.exists():':
                        inside_if_block = True
                        new_source_lines.append(line)
                    elif line.strip() == 'else:' and inside_if_block:
                        inside_if_block = False
                        new_source_lines.append(line)
                    elif inside_if_block:
                        # Everything inside the if block should be indented
                        if line.strip().startswith('# Add compatibility'):
                            new_source_lines.append('    ' + line.lstrip())
                        elif (line.strip().startswith("if 'exposure_flag'") or 
                              line.strip().startswith("if 'exposure_flag_strict'") or
                              line.strip().startswith("exposure_df['ssd_flag']") or
                              line.strip().startswith("exposure_df['ssd_flag_strict']")):
                            # These need to be indented by 4 spaces from the if block
                            new_source_lines.append('    ' + line.lstrip())
                        elif line.strip() and not line.startswith('    '):
                            # Other lines that should be indented but aren't
                            new_source_lines.append('    ' + line.lstrip())
                        else:
                            new_source_lines.append(line)
                    else:
                        new_source_lines.append(line)
                
                cell['source'] = new_source_lines
                cells_fixed += 1
    
    # Save the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    return cells_fixed

def main():
    """Main function to fix the notebook."""
    notebook_path = Path("SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return 1
    
    print("Fixing indentation in notebook cells...")
    print("=" * 60)
    
    cells_fixed = fix_notebook_indentation(notebook_path)
    
    print(f"\n✓ Fixed {cells_fixed} cells")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0

if __name__ == "__main__":
    exit(main())