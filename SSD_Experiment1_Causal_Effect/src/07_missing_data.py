#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_missing_data.py - Missing Data Imputation

Handles missing data using miceforest (MICE with random forests).
Supports all hypotheses by ensuring complete data for analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import warnings
from typing import Dict, List, Any, Optional, Tuple

# Try multiple imputation libraries with fallbacks
try:
    from miceforest import ImputationKernel
    MICEFOREST_AVAILABLE = True
except ImportError:
    MICEFOREST_AVAILABLE = False
    warnings.warn("miceforest not available, using sklearn fallback")

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    SKLEARN_IMPUTER_AVAILABLE = True
except ImportError:
    SKLEARN_IMPUTER_AVAILABLE = False
    warnings.warn("sklearn IterativeImputer not available")

try:
    from sklearn.impute import SimpleImputer
    SIMPLE_IMPUTER_AVAILABLE = True
except ImportError:
    SIMPLE_IMPUTER_AVAILABLE = False

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

def perform_multiple_imputation(df: pd.DataFrame, m: int = 5, method: str = 'auto') -> Dict[str, Any]:
    """
    Perform multiple imputation following Rubin (1987) methodology.
    
    Args:
        df: DataFrame with missing data
        m: Number of imputations (default 5 per Rubin's recommendation)
        method: Imputation method ('miceforest', 'sklearn', 'simple', 'auto')
        
    Returns:
        Dictionary with imputed datasets and metadata
    """
    log.info(f"Starting multiple imputation with m={m} imputations")
    
    # Auto-select method based on available libraries
    if method == 'auto':
        if MICEFOREST_AVAILABLE:
            method = 'miceforest'
        elif SKLEARN_IMPUTER_AVAILABLE:
            method = 'sklearn'
        else:
            method = 'simple'
    
    log.info(f"Using imputation method: {method}")
    
    # Prepare data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove ID columns
    if 'Patient_ID' in numeric_cols:
        numeric_cols.remove('Patient_ID')
    
    imputed_datasets = []
    
    if method == 'miceforest' and MICEFOREST_AVAILABLE:
        # MICEforest implementation
        for i in range(m):
            log.info(f"Creating imputation {i+1}/{m}")
            kernel = ImputationKernel(
                df,
                save_all_iterations=False,
                random_state=42 + i  # Different seed for each imputation
            )
            kernel.mice(iterations=10, verbose=False)
            imputed_df = kernel.complete_data()
            imputed_datasets.append(imputed_df)
    
    elif method == 'sklearn' and SKLEARN_IMPUTER_AVAILABLE:
        # Sklearn IterativeImputer implementation
        for i in range(m):
            log.info(f"Creating imputation {i+1}/{m}")
            
            # Numeric imputation
            numeric_imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=10, random_state=42+i),
                random_state=42+i,
                max_iter=10
            )
            
            df_imputed = df.copy()
            if len(numeric_cols) > 0:
                df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            
            # Simple mode imputation for categorical
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            
            imputed_datasets.append(df_imputed)
    
    else:
        # Fallback: Simple imputation
        log.warning("Using simple imputation fallback - variance estimates will be underestimated")
        for i in range(m):
            df_imputed = df.copy()
            
            # Numeric: mean imputation
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy='mean')
                df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            
            # Categorical: mode imputation
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            
            imputed_datasets.append(df_imputed)
    
    return {
        'imputed_datasets': imputed_datasets,
        'n_imputations': m,
        'method': method,
        'numeric_cols_imputed': numeric_cols,
        'categorical_cols_imputed': categorical_cols,
        'original_missing_pct': df.isnull().sum() / len(df) * 100
    }

def save_multiple_imputations(imputation_results: Dict[str, Any], output_dir: Path) -> List[Path]:
    """
    Save multiple imputed datasets to separate files.
    
    Args:
        imputation_results: Results from perform_multiple_imputation
        output_dir: Directory to save imputed datasets
        
    Returns:
        List of paths to saved imputation files
    """
    output_dir.mkdir(exist_ok=True)
    saved_paths = []
    
    for i, imputed_df in enumerate(imputation_results['imputed_datasets']):
        filename = f"cohort_imputed_{i+1}.parquet"
        filepath = output_dir / filename
        imputed_df.to_parquet(filepath, index=False)
        saved_paths.append(filepath)
        log.info(f"Saved imputation {i+1} to {filepath}")
    
    # Save metadata
    metadata = {
        'n_imputations': imputation_results['n_imputations'],
        'method': imputation_results['method'],
        'numeric_cols_imputed': imputation_results['numeric_cols_imputed'],
        'categorical_cols_imputed': imputation_results['categorical_cols_imputed'],
        'original_missing_pct': imputation_results['original_missing_pct'].to_dict(),
        'imputation_files': [str(p) for p in saved_paths]
    }
    
    import json
    metadata_path = output_dir / "imputation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    log.info(f"Saved imputation metadata to {metadata_path}")
    return saved_paths

# Impute if needed
max_missing = get_config("qc.max_missing_percent", 5.0)
if (missing_pct > 0).any():
    # Check if any column exceeds threshold
    if (missing_pct > max_missing).any():
        high_missing = missing_pct[missing_pct > max_missing]
        log.warning(f"Columns with >{max_missing}% missing: {high_missing.to_dict()}")
    
    # Perform multiple imputation (Rubin's method with m=5)
    m_imputations = get_config("imputation.n_imputations", 5)
    imputation_results = perform_multiple_imputation(cohort, m=m_imputations)
    
    # Save all imputed datasets
    imputed_dir = ROOT / 'data_derived' / 'imputed'
    saved_paths = save_multiple_imputations(imputation_results, imputed_dir)
    
    # For backwards compatibility, also save the first imputation as main file
    cohort_imputed = imputation_results['imputed_datasets'][0]
    
    # Verify no missing data remains
    remaining_missing = cohort_imputed.isnull().sum().sum()
    log.info(f"Remaining missing values: {remaining_missing}")
    
    # Log imputation summary
    log.info(f"Multiple imputation completed:")
    log.info(f"  Method: {imputation_results['method']}")
    log.info(f"  Number of imputations: {imputation_results['n_imputations']}")
    log.info(f"  Numeric columns imputed: {len(imputation_results['numeric_cols_imputed'])}")
    log.info(f"  Categorical columns imputed: {len(imputation_results['categorical_cols_imputed'])}")
    
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
        "--kv", f"missing_data_method={imputation_results.get('method', 'unknown')}",
        "--kv", f"n_imputations={imputation_results.get('n_imputations', 1)}",
        "--kv", "hypotheses=All",
        "--kv", f"script=07_missing_data.py"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")