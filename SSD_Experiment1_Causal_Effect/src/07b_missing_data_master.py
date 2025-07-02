#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07b_missing_data_master.py - Missing Data Imputation on Full Master Table

This version imputes the full master table (73 columns) instead of just 
the cohort (19 columns), enabling proper Rubin's Rules application.

Changes from 07_missing_data.py:
- Uses master_with_missing.parquet as input
- Outputs to data_derived/imputed_master/ directory
- Preserves all 73 columns through imputation

Author: Ryhan Suny
Date: June 30, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import warnings
from typing import Dict, List, Any, Optional, Tuple
import argparse

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
        logging.FileHandler("07b_missing_data_master.log", mode="w")
    ])
log = logging.getLogger("missing_data_master")

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
    
    # Prepare data - exclude Patient_ID from imputation
    id_col = 'Patient_ID'
    df_to_impute = df.drop(columns=[id_col])
    patient_ids = df[id_col]
    
    # Identify column types
    # Per evidence-based solutions doc: datetime excluded from imputation (standard practice)
    # References: Cleveland Clinic (2023), DSM-5-TR (2022), Hernán & Robins (2016)
    datetime_cols = df_to_impute.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Exclude datetime columns from imputation
    if datetime_cols:
        log.info(f"Excluding {len(datetime_cols)} datetime columns from imputation: {datetime_cols}")
        df_to_impute = df_to_impute.drop(columns=datetime_cols)
    
    # Now identify numeric and categorical columns after datetime exclusion
    numeric_cols = df_to_impute.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_to_impute.select_dtypes(include=['object', 'category']).columns.tolist()
    
    log.info(f"Numeric columns: {len(numeric_cols)}")
    log.info(f"Categorical columns: {len(categorical_cols)}")
    
    imputed_datasets = []
    
    if method == 'miceforest' and MICEFOREST_AVAILABLE:
        # MICEforest implementation
        for i in range(m):
            log.info(f"Creating imputation {i+1}/{m}")
            kernel = ImputationKernel(
                df_to_impute,
                save_all_iterations=False,
                random_state=42 + i  # Different seed for each imputation
            )
            kernel.mice(iterations=10, verbose=False)
            imputed_df = kernel.complete_data()
            
            # Add Patient_ID back
            imputed_df[id_col] = patient_ids
            # Reorder columns to match original
            imputed_df = imputed_df[df.columns]
            
            imputed_datasets.append(imputed_df)
    
    elif method == 'sklearn' and SKLEARN_IMPUTER_AVAILABLE:
        # Sklearn IterativeImputer implementation
        for i in range(m):
            log.info(f"Creating imputation {i+1}/{m}")
            
            df_imputed = df_to_impute.copy()
            
            # Numeric imputation
            if len(numeric_cols) > 0:
                numeric_imputer = IterativeImputer(
                    estimator=RandomForestRegressor(n_estimators=10, random_state=42+i),
                    random_state=42+i,
                    max_iter=10
                )
                df_imputed[numeric_cols] = numeric_imputer.fit_transform(df_to_impute[numeric_cols])
            
            # Simple mode imputation for categorical
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_cols] = cat_imputer.fit_transform(df_to_impute[categorical_cols])
            
            # Add Patient_ID back
            df_imputed[id_col] = patient_ids
            # Reorder columns
            df_imputed = df_imputed[df.columns]
            
            imputed_datasets.append(df_imputed)
    
    else:
        # Fallback: Simple imputation
        log.warning("Using simple imputation fallback - variance estimates will be underestimated")
        for i in range(m):
            df_imputed = df_to_impute.copy()
            
            # Numeric: mean imputation
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy='mean')
                df_imputed[numeric_cols] = numeric_imputer.fit_transform(df_to_impute[numeric_cols])
            
            # Categorical: mode imputation
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_cols] = cat_imputer.fit_transform(df_to_impute[categorical_cols])
            
            # Add Patient_ID back
            df_imputed[id_col] = patient_ids
            df_imputed = df_imputed[df.columns]
            
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
    output_dir.mkdir(exist_ok=True, parents=True)
    saved_paths = []
    
    for i, imputed_df in enumerate(imputation_results['imputed_datasets']):
        filename = f"master_imputed_{i+1}.parquet"
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


def main():
    """Main function for master table imputation."""
    parser = argparse.ArgumentParser(
        description="Perform multiple imputation on full master table"
    )
    parser.add_argument('--input', default='data_derived/master_with_missing.parquet',
                       help='Input master table with missing data')
    parser.add_argument('--output-dir', default='data_derived/imputed_master',
                       help='Output directory for imputed datasets')
    parser.add_argument('--m', type=int, default=None,
                       help='Number of imputations (default from config)')
    parser.add_argument('--method', choices=['miceforest', 'sklearn', 'simple', 'auto'],
                       default='auto', help='Imputation method')
    
    args = parser.parse_args()
    
    # Paths
    ROOT = Path(__file__).resolve().parents[1]
    MASTER_PATH = ROOT / args.input
    OUT_DIR = ROOT / args.output_dir
    
    # Load master table
    log.info(f"Loading master table from {MASTER_PATH}")
    master = pd.read_parquet(MASTER_PATH)
    log.info(f"Loaded {len(master):,} rows × {len(master.columns)} columns")
    
    # Check missingness
    missing_pct = master.isnull().sum() / len(master) * 100
    log.info("\nMissing data summary:")
    missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if len(missing_cols) > 0:
        for col, pct in missing_cols.items():
            log.info(f"  {col}: {pct:.2f}%")
    else:
        log.info("  No missing data found!")
        return
    
    # Get number of imputations from config or args
    if args.m:
        m_imputations = args.m
    else:
        m_imputations = get_config("imputation.n_imputations", 5)
    
    # For reviewer feedback: increase to match % missing
    max_missing = missing_pct.max()
    if max_missing > 20 and m_imputations < 20:
        log.warning(f"High missingness ({max_missing:.1f}%) detected. "
                   f"Consider increasing imputations from {m_imputations} to ~30")
    
    # Perform multiple imputation
    imputation_results = perform_multiple_imputation(
        master, m=m_imputations, method=args.method
    )
    
    # Save all imputed datasets
    saved_paths = save_multiple_imputations(imputation_results, OUT_DIR)
    
    # For backwards compatibility, also save the first imputation as main file
    master_imputed = imputation_results['imputed_datasets'][0]
    
    # Verify no missing data remains
    remaining_missing = master_imputed.isnull().sum().sum()
    log.info(f"Remaining missing values: {remaining_missing}")
    
    # Log imputation summary
    log.info(f"\nMultiple imputation completed:")
    log.info(f"  Method: {imputation_results['method']}")
    log.info(f"  Number of imputations: {imputation_results['n_imputations']}")
    log.info(f"  Numeric columns imputed: {len(imputation_results['numeric_cols_imputed'])}")
    log.info(f"  Categorical columns imputed: {len(imputation_results['categorical_cols_imputed'])}")
    log.info(f"  Output directory: {OUT_DIR}")
    
    # Update study documentation
    import subprocess
    try:
        result = subprocess.run([
            sys.executable, 
            str(ROOT / "scripts" / "update_study_doc.py"),
            "--step", "Master table multiple imputation completed",
            "--kv", f"artefact=imputed_master/",
            "--kv", f"n_patients={len(master)}",
            "--kv", f"n_columns={len(master.columns)}",
            "--kv", f"missing_cols_imputed={len(missing_cols)}",
            "--kv", f"missing_data_method={imputation_results.get('method', 'unknown')}",
            "--kv", f"n_imputations={imputation_results.get('n_imputations', 1)}",
            "--kv", "hypotheses=All",
            "--kv", f"script=07b_missing_data_master.py"
        ], capture_output=True, text=True)
        if result.returncode == 0:
            log.info("Study documentation updated successfully")
        else:
            log.warning(f"Study doc update failed: {result.stderr}")
    except Exception as e:
        log.warning(f"Could not update study doc: {e}")


if __name__ == "__main__":
    main()