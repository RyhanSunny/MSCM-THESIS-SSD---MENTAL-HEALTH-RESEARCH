#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_lab_flag.py – Lab-based Sensitivity Flagging

- Loads the validated cohort and lab data
- Calculates normal lab counts in multiple time windows
- Creates sensitivity thresholds for analysis
- Saves lab_sensitivity.parquet to data_derived/

HYPOTHESIS MAPPING:
This script supports:
- RQ: Provides lab utilization data for prevalence analysis
- H1: Lab ordering patterns as part of healthcare utilization
- H5: Lab results correlating with health anxiety behaviors
- General QA: Validates lab data quality and completeness
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import timedelta

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
        logging.FileHandler("06_lab_flag.log", mode="w")
    ])
log = logging.getLogger("lab_flag")

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

try:
    from helpers.lab_utils import is_normal_lab, add_normal_flags
    log.info("Lab utilities imported successfully")
    USE_LAB_UTILS = True
except ImportError as e:
    log.warning(f"Could not import lab utilities: {e}")
    log.warning("Will attempt to use existing is_normal column if available")
    USE_LAB_UTILS = False

# Paths
ROOT = Path(__file__).resolve().parents[1]
COHORT_PATH = ROOT / 'data_derived' / 'cohort.parquet'
CHECKPOINT_ROOT = ROOT / get_config("paths.checkpoint_root", "Notebooks/data/interim")
OUT_PATH = ROOT / 'data_derived' / 'lab_sensitivity.parquet'

# Find latest checkpoint
def latest_checkpoint(base: Path) -> Path:
    cps = sorted(base.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cps:
        raise FileNotFoundError(f"No checkpoint_* folder found in {base}")
    return cps[0]

CKPT = latest_checkpoint(CHECKPOINT_ROOT)
log.info(f"Using checkpoint: {CKPT}")

# Load cohort
log.info(f"Loading cohort from {COHORT_PATH}")
cohort = pd.read_parquet(COHORT_PATH)
log.info(f"Loaded {len(cohort):,} patients")

# Load lab data
log.info("Loading lab data...")
lab_path = CKPT / "lab.parquet"
if lab_path.exists():
    lab = pd.read_parquet(lab_path)
else:
    csv_path = CKPT / "lab.csv"
    if csv_path.exists():
        # Process in chunks for large CSV
        log.info("Processing lab data in chunks...")
        chunk_size = 100000
        
        # First pass: get patient IDs and date range
        patient_ids = set(cohort.Patient_ID)
        lab_chunks = []
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
            # Filter to cohort patients
            chunk = chunk[chunk.Patient_ID.isin(patient_ids)]
            if len(chunk) > 0:
                # Parse dates
                chunk["PerformedDate"] = pd.to_datetime(chunk.PerformedDate, errors="coerce")
                chunk = chunk.dropna(subset=["PerformedDate"])
                lab_chunks.append(chunk)
        
        if lab_chunks:
            lab = pd.concat(lab_chunks, ignore_index=True)
        else:
            raise ValueError("No lab data found for cohort patients")
    else:
        raise FileNotFoundError("No lab data found in checkpoint")

log.info(f"Loaded {len(lab):,} lab records for {lab.Patient_ID.nunique():,} patients")

# Ensure PerformedDate is datetime
lab["PerformedDate"] = pd.to_datetime(lab.PerformedDate, errors="coerce")
lab = lab.dropna(subset=["PerformedDate"])

# Add normal flag if not present
if "is_normal" not in lab.columns:
    if USE_LAB_UTILS:
        log.info("Adding normal flags to lab data using lab_utils...")
        lab = add_normal_flags(lab)
        log.info(f"Normal flags added - {lab['is_normal'].sum():,} normal results out of {len(lab):,}")
    else:
        log.error("Lab data missing 'is_normal' column and lab_utils not available")
        log.error("Cannot proceed without ability to determine normal lab results")
        raise ValueError("Missing is_normal column and lab_utils unavailable")
else:
    log.info("Using existing is_normal column from lab data")

# Define time windows for sensitivity analysis
windows = {
    "6m": 180,
    "12m": 365,
    "18m": 548,
    "24m": 730
}

# Initialize output dataframe
lab_flags = cohort[["Patient_ID", "IndexDate_lab"]].copy()

# For each time window, calculate lab counts
for window_name, days in windows.items():
    log.info(f"\nCalculating lab counts for {window_name} window...")
    
    # Merge lab with cohort dates
    lab_window = lab.merge(cohort[["Patient_ID", "IndexDate_lab"]], 
                          on="Patient_ID", how="inner")
    
    # Define window boundaries
    lab_window["window_start"] = lab_window.IndexDate_lab
    lab_window["window_end"] = lab_window.IndexDate_lab + pd.Timedelta(days=days)
    
    # Filter to window
    lab_in_window = lab_window[
        (lab_window.PerformedDate >= lab_window.window_start) &
        (lab_window.PerformedDate <= lab_window.window_end)
    ]
    
    # Count total labs
    total_labs = lab_in_window.groupby("Patient_ID").size()
    total_labs.name = f"lab_count_{window_name}"
    
    # Count normal labs
    normal_labs = lab_in_window.groupby("Patient_ID")["is_normal"].sum()
    normal_labs.name = f"normal_count_{window_name}"
    
    # Calculate ratio
    lab_stats = pd.DataFrame({
        f"lab_count_{window_name}": total_labs,
        f"normal_count_{window_name}": normal_labs
    })
    lab_stats[f"normal_ratio_{window_name}"] = (
        lab_stats[f"normal_count_{window_name}"] / 
        lab_stats[f"lab_count_{window_name}"]
    ).fillna(0)
    
    # Merge to main dataframe
    lab_flags = lab_flags.merge(lab_stats, left_on="Patient_ID", 
                               right_index=True, how="left")
    
    # Fill missing values
    lab_flags[f"lab_count_{window_name}"] = lab_flags[f"lab_count_{window_name}"].fillna(0)
    lab_flags[f"normal_count_{window_name}"] = lab_flags[f"normal_count_{window_name}"].fillna(0)
    lab_flags[f"normal_ratio_{window_name}"] = lab_flags[f"normal_ratio_{window_name}"].fillna(0)
    
    # Log statistics
    log.info(f"  Mean lab count: {lab_flags[f'lab_count_{window_name}'].mean():.2f}")
    log.info(f"  Mean normal count: {lab_flags[f'normal_count_{window_name}'].mean():.2f}")
    log.info(f"  Mean normal ratio: {lab_flags[f'normal_ratio_{window_name}'].mean():.3f}")

# Create sensitivity threshold flags
log.info("\nCreating sensitivity threshold flags...")

# For 12-month window (primary analysis)
for threshold in [2, 3, 4, 5]:
    flag_name = f"normal_labs_gte_{threshold}_12m"
    lab_flags[flag_name] = (lab_flags["normal_count_12m"] >= threshold).astype(int)
    pct = lab_flags[flag_name].mean() * 100
    log.info(f"  {flag_name}: {pct:.1f}% of patients")

# High lab utilization flag (top decile)
for window_name in ["6m", "12m", "24m"]:
    col_name = f"lab_count_{window_name}"
    threshold = lab_flags[col_name].quantile(0.9)
    flag_name = f"high_lab_use_{window_name}"
    lab_flags[flag_name] = (lab_flags[col_name] >= threshold).astype(int)
    pct = lab_flags[flag_name].mean() * 100
    log.info(f"  {flag_name}: {pct:.1f}% of patients (>={threshold:.0f} labs)")

# Calculate mean normal lab count in 12m window (for reporting)
normal_lab_n12_mean = lab_flags["normal_count_12m"].mean()
log.info(f"\nMean normal lab count in 12m window: {normal_lab_n12_mean:.2f}")

# Lab type analysis (if TestName available)
if "TestName_calc" in lab.columns:
    log.info("\nAnalyzing most common lab tests...")
    top_tests = lab.TestName_calc.value_counts().head(20)
    log.info("Top 20 lab tests:")
    for test, count in top_tests.items():
        log.info(f"  {test}: {count:,} ({count/len(lab):.1%})")

# Summary statistics
log.info("\nLab utilization summary:")
log.info(f"  Patients with any labs: {(lab_flags['lab_count_12m'] > 0).sum():,} "
         f"({(lab_flags['lab_count_12m'] > 0).mean():.1%})")
log.info(f"  Median labs per patient (12m): {lab_flags['lab_count_12m'].median():.0f}")
log.info(f"  75th percentile (12m): {lab_flags['lab_count_12m'].quantile(0.75):.0f}")
log.info(f"  90th percentile (12m): {lab_flags['lab_count_12m'].quantile(0.9):.0f}")

# Drop IndexDate_lab before saving (redundant with cohort)
lab_flags = lab_flags.drop("IndexDate_lab", axis=1)

# Save output
log.info(f"\nSaving lab sensitivity data to {OUT_PATH}")
lab_flags.to_parquet(OUT_PATH, index=False)
log.info(f"Saved {len(lab_flags):,} rows with {len(lab_flags.columns)} columns")

# Update study documentation
import subprocess
try:
    result = subprocess.run([
        sys.executable, 
        str(ROOT / "scripts" / "update_study_doc.py"),
        "--step", "Lab sensitivity flags generated",
        "--kv", f"artefact=lab_sensitivity.parquet",
        "--kv", f"n_patients={len(lab_flags)}",
        "--kv", f"normal_lab_n12_mean={normal_lab_n12_mean:.2f}",
        "--kv", f"n_features={len(lab_flags.columns)-1}",
        "--kv", "hypotheses=RQ,H1,H5",
        "--kv", f"script=06_lab_flag.py",
        "--kv", "status=implemented"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")