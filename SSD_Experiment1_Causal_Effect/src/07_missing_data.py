#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_missing_data.py - Missing Data Imputation

Handles missing data using miceforest (MICE with random forests).
Supports all hypotheses by ensuring complete data for analysis.
"""

from miceforest import ImputationKernel
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
        logging.FileHandler("07_missing_data.log", mode="w")
    ])
log = logging.getLogger("missing_data")

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
COHORT_PATH = ROOT / 'data_derived' / 'cohort.parquet'
OUT_PATH = ROOT / 'data_derived' / 'cohort_imputed.parquet'

# Load cohort
log.info(f"Loading cohort from {COHORT_PATH}")
cohort = pd.read_parquet(COHORT_PATH)
log.info(f"Loaded {len(cohort):,} patients")

# Check missingness
missing_pct = cohort.isnull().sum() / len(cohort) * 100
log.info("\nMissing data summary:")
missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
if len(missing_cols) > 0:
    for col, pct in missing_cols.items():
        log.info(f"  {col}: {pct:.2f}%")
else:
    log.info("  No missing data found!")

# Impute if needed
max_missing = get_config("qc.max_missing_percent", 5.0)
if (missing_pct > 0).any():
    # Check if any column exceeds threshold
    if (missing_pct > max_missing).any():
        high_missing = missing_pct[missing_pct > max_missing]
        log.warning(f"Columns with >{max_missing}% missing: {high_missing.to_dict()}")
    
    # Separate numeric and categorical columns
    numeric_cols = cohort.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = cohort.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove ID columns from imputation
    if 'Patient_ID' in numeric_cols:
        numeric_cols.remove('Patient_ID')
    
    log.info(f"Numeric columns for imputation: {len(numeric_cols)}")
    log.info(f"Categorical columns for imputation: {len(categorical_cols)}")
    
    # Create imputation kernel
    log.info("Initializing MICE imputation kernel...")
    kernel = ImputationKernel(
        cohort,
        save_all_iterations=False,
        random_state=get_config("random_state.global_seed", 42)
    )
    
    # Run MICE
    iterations = 20
    log.info(f"Running MICE for {iterations} iterations...")
    kernel.mice(iterations=iterations, verbose=True)
    
    # Get completed data
    cohort_imputed = kernel.complete_data()
    
    # Verify no missing data remains
    remaining_missing = cohort_imputed.isnull().sum().sum()
    log.info(f"Remaining missing values: {remaining_missing}")
    
else:
    log.info("No missing data to impute")
    cohort_imputed = cohort.copy()

# Save imputed data
log.info(f"Saving imputed cohort to {OUT_PATH}")
cohort_imputed.to_parquet(OUT_PATH, index=False)
log.info(f"Saved {len(cohort_imputed):,} rows")

# Update study documentation
import subprocess
try:
    result = subprocess.run([
        sys.executable, 
        str(ROOT / "scripts" / "update_study_doc.py"),
        "--step", "Missing data imputation completed",
        "--kv", f"artefact=cohort_imputed.parquet",
        "--kv", f"n_patients={len(cohort_imputed)}",
        "--kv", f"missing_cols_imputed={len(missing_cols)}",
        "--kv", "missing_data_method=miceforest",
        "--kv", "hypotheses=All",
        "--kv", f"script=07_missing_data.py"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")