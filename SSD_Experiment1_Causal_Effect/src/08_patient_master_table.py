#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_patient_master_table.py - Patient Master Table Creation

Merges all derived datasets into a single master table for analysis.
Critical for ensuring all analyses use the same patient population.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add src and utils to path
SRC = (Path(__file__).resolve().parents[1] / "src").as_posix()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

UTILS = (Path(__file__).resolve().parents[1] / "utils").as_posix()
if UTILS not in sys.path:
    sys.path.insert(0, UTILS)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("08_patient_master_table.log", mode="w")
    ])
log = logging.getLogger("master_table")

# Import utilities
try:
    from global_seeds import set_global_seeds
    set_global_seeds()
    log.info("Global seeds set for reproducibility")
except ImportError:
    log.warning("Could not import global_seeds utility")

try:
    from config_loader import load_config, get_config
    config = load_config()
    log.info("Configuration loaded successfully")
except Exception as e:
    log.error(f"Could not load configuration: {e}")
    raise

# Paths
ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / 'data_derived'
OUT_PATH = DERIVED / 'patient_master.parquet'

# Define expected files and their key columns
EXPECTED_FILES = {
    'cohort.parquet': ['Patient_ID', 'Sex', 'BirthYear', 'Age_at_2015', 'SpanMonths', 
                      'IndexDate_lab', 'Charlson', 'LongCOVID_flag', 'NYD_count'],
    'exposure.parquet': ['Patient_ID', 'exposure_flag', 'normal_lab_count', 
                        'symptom_referral_n', 'drug_days_in_window'],
    'mediator_autoencoder.parquet': ['Patient_ID', 'SSD_severity_index'],
    'outcomes.parquet': ['Patient_ID', 'total_encounters', 'ed_visits', 
                        'specialist_referrals', 'medical_costs', 'inappropriate_meds', 
                        'polypharmacy', 'high_utilization'],
    'confounders.parquet': ['Patient_ID'],  # Corrected file name
    'lab_sensitivity.parquet': ['Patient_ID'],  # Corrected file name
    'referral_sequences.parquet': ['Patient_ID', 'referral_loop', 'loop_count', 
                                  'sequence_length', 'has_circular_pattern', 
                                  'mean_referral_interval_days']
}

# Optional files (may not exist yet)
OPTIONAL_FILES = ['referral_sequences.parquet']

# Load all available datasets
log.info("Loading derived datasets...")
datasets = {}
missing_files = []

for filename, expected_cols in EXPECTED_FILES.items():
    filepath = DERIVED / filename
    if filepath.exists():
        log.info(f"  Loading {filename}")
        df = pd.read_parquet(filepath)
        
        # Check for expected columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols and filename not in OPTIONAL_FILES:
            log.warning(f"    Missing expected columns: {missing_cols}")
        
        datasets[filename] = df
        log.info(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")
    else:
        if filename not in OPTIONAL_FILES:
            log.warning(f"  Missing required file: {filename}")
            missing_files.append(filename)
        else:
            log.info(f"  Optional file not found: {filename}")

# Check if we have minimum required files
required_files = ['cohort.parquet', 'exposure.parquet']
missing_required = [f for f in missing_files if f in required_files]
if missing_required:
    raise FileNotFoundError(f"Missing required files: {missing_required}")

# Start with cohort as base
log.info("\nMerging datasets...")
master = datasets['cohort.parquet'].copy()
initial_rows = len(master)
log.info(f"Starting with cohort: {initial_rows:,} patients")

# Track merge statistics
merge_stats = []

# Merge each dataset
for filename, df in datasets.items():
    if filename == 'cohort.parquet':
        continue  # Skip base dataset
    
    # Remove duplicates if any
    if df['Patient_ID'].duplicated().any():
        log.warning(f"  Found {df['Patient_ID'].duplicated().sum()} duplicate Patient_IDs in {filename}")
        df = df.drop_duplicates(subset='Patient_ID', keep='first')
        log.info(f"  After deduplication: {len(df)} rows")
    
    # Determine columns to merge (exclude Patient_ID and overlapping columns)
    merge_cols = [col for col in df.columns if col != 'Patient_ID']
    
    # Check for overlapping columns and exclude them from merge
    overlap = set(master.columns) & set(merge_cols)
    if overlap:
        log.warning(f"  Overlapping columns with {filename}: {overlap}")
        log.info(f"  Excluding overlapping columns from merge")
        merge_cols = [col for col in merge_cols if col not in overlap]
    
    if not merge_cols:
        log.warning(f"  No columns to merge from {filename} after removing overlaps")
        continue
        
    # Perform merge
    log.info(f"  Merging {filename} ({len(merge_cols)} columns)")
    master = master.merge(df[['Patient_ID'] + merge_cols], 
                         on='Patient_ID', 
                         how='left', 
                         validate='one_to_one')
    
    # Track statistics
    rows_after = len(master)
    if rows_after != initial_rows:
        log.error(f"    Row count changed! Before: {initial_rows}, After: {rows_after}")
        raise ValueError("Merge resulted in row count change")
    
    # Check for successful merge
    null_count = master[merge_cols].isnull().sum().sum()
    merge_stats.append({
        'file': filename,
        'columns_added': len(merge_cols),
        'null_values': null_count,
        'null_pct': null_count / (len(master) * len(merge_cols)) * 100
    })
    
    log.info(f"    Added {len(merge_cols)} columns, {null_count:,} null values ({merge_stats[-1]['null_pct']:.1f}%)")

# Verify final shape
log.info(f"\nFinal master table shape: {master.shape}")
log.info(f"  Rows: {len(master):,} (expected: {initial_rows:,})")
log.info(f"  Columns: {len(master.columns)}")

# Check data types
log.info("\nData type summary:")
dtype_counts = master.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    log.info(f"  {dtype}: {count} columns")

# Check for any remaining nulls in key columns
key_cols = ['Patient_ID', 'exposure_flag', 'Age_at_2015', 'Sex']
null_check = master[key_cols].isnull().sum()
if null_check.any():
    log.warning("\nNull values in key columns:")
    for col, nulls in null_check[null_check > 0].items():
        log.warning(f"  {col}: {nulls} nulls")

# Summary statistics
log.info("\nKey variable summary:")
if 'exposure_flag' in master.columns:
    exposed = master['exposure_flag'].sum()
    log.info(f"  Exposed patients: {exposed:,} ({exposed/len(master):.1%})")

if 'SSD_severity_index' in master.columns:
    log.info(f"  Mean severity index: {master['SSD_severity_index'].mean():.2f}")

if 'total_encounters' in master.columns:
    log.info(f"  Mean encounters: {master['total_encounters'].mean():.2f}")

if 'medical_costs' in master.columns:
    log.info(f"  Mean medical costs: ${master['medical_costs'].mean():.2f}")

# Save master table
log.info(f"\nSaving master table to {OUT_PATH}")
master.to_parquet(OUT_PATH, index=False)
log.info(f"Saved {len(master):,} rows × {len(master.columns)} columns")

# Create a summary report
summary_path = DERIVED / 'master_table_summary.txt'
with open(summary_path, 'w') as f:
    f.write("Patient Master Table Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total patients: {len(master):,}\n")
    f.write(f"Total columns: {len(master.columns)}\n\n")
    
    f.write("Merge Statistics:\n")
    for stat in merge_stats:
        f.write(f"  {stat['file']}: {stat['columns_added']} columns, "
                f"{stat['null_pct']:.1f}% null\n")
    
    f.write("\nColumn List:\n")
    for col in sorted(master.columns):
        dtype = str(master[col].dtype)
        nulls = master[col].isnull().sum()
        f.write(f"  {col}: {dtype}, {nulls} nulls\n")

log.info(f"Summary saved to {summary_path}")

# Update study documentation
import subprocess
try:
    result = subprocess.run([
        sys.executable, 
        str(ROOT / "scripts" / "update_study_doc.py"),
        "--step", "Patient master table created",
        "--kv", f"artefact=patient_master.parquet",
        "--kv", f"patient_master_rows={len(master)}",
        "--kv", f"patient_master_cols={len(master.columns)}",
        "--kv", f"files_merged={len(datasets)}",
        "--kv", "hypotheses=All",
        "--kv", f"script=08_patient_master_table.py"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")