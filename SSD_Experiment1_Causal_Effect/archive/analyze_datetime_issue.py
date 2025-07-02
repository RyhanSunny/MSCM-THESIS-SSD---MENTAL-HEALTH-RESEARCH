#!/usr/bin/env python3
"""Check datetime handling in imputation process."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the master file with missing data
master_path = Path("/mnt/c/Users/ProjectC4M/Documents/MSCM THESIS SSD/MSCM-THESIS-SSD---MENTAL-HEALTH-RESEARCH/SSD_Experiment1_Causal_Effect/data_derived/master_with_missing.parquet")
df = pd.read_parquet(master_path)

print("Column types analysis:")
print("="*60)

# Get all columns by dtype
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols[:5]}...")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Datetime columns ({len(datetime_cols)}): {datetime_cols}")

# Check what columns would be excluded from imputation
excluded_cols = ['Patient_ID'] + datetime_cols
print(f"\nColumns excluded from imputation: {excluded_cols}")

# Check missing values in datetime columns
print("\nMissing values in datetime columns:")
for col in datetime_cols:
    missing = df[col].isnull().sum()
    pct = (missing / len(df)) * 100
    print(f"  {col}: {missing:,} ({pct:.2f}%)")

# Check if any other columns are being excluded
all_cols = set(df.columns)
included_in_imputation = set(numeric_cols + categorical_cols)
excluded_from_imputation = all_cols - included_in_imputation - {'Patient_ID'}

print(f"\nColumns that will be excluded from imputation (besides Patient_ID):")
for col in excluded_from_imputation:
    dtype = df[col].dtype
    missing = df[col].isnull().sum()
    print(f"  {col} (dtype: {dtype}): {missing:,} missing values")