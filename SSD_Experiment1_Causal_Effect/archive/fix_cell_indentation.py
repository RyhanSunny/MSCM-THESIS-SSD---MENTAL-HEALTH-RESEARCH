#!/usr/bin/env python3
"""Fix indentation issue in STEP 2 cell of the notebook"""

import json

# Read the notebook
with open('SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find and fix cell 11 (STEP 2)
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and any('STEP 2: Exposure Flags' in line for line in cell['source']):
        print(f"Found STEP 2 cell at index {i}")
        
        # Fix the indentation in the source
        fixed_source = []
        for line in cell['source']:
            if "exposure_df['ssd_flag'] = exposure_df['exposure_flag']" in line:
                # This line needs more indentation
                fixed_source.append("        exposure_df['ssd_flag'] = exposure_df['exposure_flag']\n")
            elif "exposure_df['ssd_flag_strict'] = exposure_df['exposure_flag_strict']" in line:
                # This line needs more indentation
                fixed_source.append("        exposure_df['ssd_flag_strict'] = exposure_df['exposure_flag_strict']\n")
            else:
                fixed_source.append(line)
        
        # Update the cell source
        cell['source'] = fixed_source
        print("Fixed indentation in STEP 2 cell")
        break

# Write the fixed notebook
with open('SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook has been fixed!") 