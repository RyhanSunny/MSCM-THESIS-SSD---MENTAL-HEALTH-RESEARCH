#!/usr/bin/env python3
"""
Quick check of exposure data structure
"""
import pandas as pd

# Load exposure data
exposure = pd.read_parquet('data_derived/exposure.parquet')

# Check columns
print("Columns in exposure data:")
print(exposure.columns.tolist())
print()

# Check for both flags
if 'exposure_flag' in exposure.columns:
    print(f"exposure_flag (OR logic) - Exposed: {exposure['exposure_flag'].sum():,} ({exposure['exposure_flag'].mean()*100:.1f}%)")

if 'exposure_flag_strict' in exposure.columns:
    print(f"exposure_flag_strict (AND logic) - Exposed: {exposure['exposure_flag_strict'].sum():,} ({exposure['exposure_flag_strict'].mean()*100:.1f}%)")

# Check individual criteria
for col in exposure.columns:
    if col.startswith('crit'):
        print(f"{col}: {exposure[col].sum():,} ({exposure[col].mean()*100:.1f}%)")