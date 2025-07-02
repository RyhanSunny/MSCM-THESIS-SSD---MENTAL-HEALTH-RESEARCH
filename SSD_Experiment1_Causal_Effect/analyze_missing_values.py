#!/usr/bin/env python3
"""Analyze missing values in imputed and pre-imputation master files."""

import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
base_path = Path("/mnt/c/Users/ProjectC4M/Documents/MSCM THESIS SSD/MSCM-THESIS-SSD---MENTAL-HEALTH-RESEARCH/SSD_Experiment1_Causal_Effect")
imputed_path = base_path / "data_derived" / "imputed_master" / "master_imputed_1.parquet"
pre_impute_path = base_path / "data_derived" / "master_with_missing.parquet"

print("="*80)
print("MISSING VALUES ANALYSIS")
print("="*80)

# Load the first imputed file
print("\n1. Loading master_imputed_1.parquet...")
df_imputed = pd.read_parquet(imputed_path)
print(f"   Shape: {df_imputed.shape}")

# Check total missing values
total_missing = df_imputed.isnull().sum().sum()
print(f"\n2. Total missing values in imputed file: {total_missing:,}")

# Find columns with missing values
cols_with_missing = df_imputed.columns[df_imputed.isnull().any()].tolist()
print(f"\n3. Number of columns with missing values: {len(cols_with_missing)}")

if cols_with_missing:
    print("\n4. Columns with missing values and their counts:")
    missing_counts = df_imputed[cols_with_missing].isnull().sum().sort_values(ascending=False)
    for col, count in missing_counts.items():
        pct = (count / len(df_imputed)) * 100
        print(f"   - {col}: {count:,} ({pct:.2f}%)")
    
    print("\n5. Data types of columns with missing values:")
    for col in cols_with_missing:
        dtype = df_imputed[col].dtype
        print(f"   - {col}: {dtype}")

# Load pre-imputation file for comparison
print("\n" + "="*80)
print("COMPARISON WITH PRE-IMPUTATION FILE")
print("="*80)

if pre_impute_path.exists():
    print("\n6. Loading master_with_missing.parquet...")
    df_pre = pd.read_parquet(pre_impute_path)
    print(f"   Shape: {df_pre.shape}")
    
    # Check total missing values before imputation
    total_missing_pre = df_pre.isnull().sum().sum()
    print(f"\n7. Total missing values before imputation: {total_missing_pre:,}")
    
    # Compare columns
    print(f"\n8. Column comparison:")
    print(f"   - Columns in pre-imputation: {len(df_pre.columns)}")
    print(f"   - Columns in imputed: {len(df_imputed.columns)}")
    
    # Check if same columns
    cols_only_pre = set(df_pre.columns) - set(df_imputed.columns)
    cols_only_imputed = set(df_imputed.columns) - set(df_pre.columns)
    
    if cols_only_pre:
        print(f"\n   Columns only in pre-imputation file: {cols_only_pre}")
    if cols_only_imputed:
        print(f"\n   Columns only in imputed file: {cols_only_imputed}")
    
    # Compare missing values for columns that still have missing values
    if cols_with_missing:
        print("\n9. Missing value comparison for problematic columns:")
        for col in cols_with_missing:
            if col in df_pre.columns:
                missing_pre = df_pre[col].isnull().sum()
                missing_post = df_imputed[col].isnull().sum()
                print(f"   - {col}: {missing_pre:,} -> {missing_post:,}")
            else:
                print(f"   - {col}: Not in pre-imputation file")
else:
    print(f"\nPre-imputation file not found at: {pre_impute_path}")

# Check a sample of rows with missing values
if cols_with_missing:
    print("\n" + "="*80)
    print("SAMPLE ROWS WITH MISSING VALUES")
    print("="*80)
    
    # Get rows with any missing values
    rows_with_missing = df_imputed[df_imputed[cols_with_missing].isnull().any(axis=1)]
    print(f"\n10. Number of rows with at least one missing value: {len(rows_with_missing):,}")
    
    if len(rows_with_missing) > 0:
        print("\n11. First 5 rows with missing values:")
        print(rows_with_missing[cols_with_missing].head())

# Check if all imputed files have the same pattern
print("\n" + "="*80)
print("CHECKING CONSISTENCY ACROSS IMPUTED FILES")
print("="*80)

print("\n12. Checking first 5 imputed files for consistency...")
for i in range(1, min(6, 31)):  # Check first 5 files
    file_path = base_path / "data_derived" / "imputed_master" / f"master_imputed_{i}.parquet"
    df_temp = pd.read_parquet(file_path)
    missing_temp = df_temp.isnull().sum().sum()
    print(f"   - master_imputed_{i}.parquet: {missing_temp:,} missing values")