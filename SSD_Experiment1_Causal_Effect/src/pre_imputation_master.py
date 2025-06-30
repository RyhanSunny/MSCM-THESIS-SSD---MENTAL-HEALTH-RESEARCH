#!/usr/bin/env python3
"""
pre_imputation_master.py - Create master table with all features BEFORE imputation

CRITICAL: This fixes the pipeline order issue where imputation was happening
too early (on 19-column cohort) instead of on the full 102-column dataset.

This module combines all derived features (exposure, mediator, outcomes, 
confounders) with the base cohort BEFORE imputation, preserving missingness
patterns for proper multiple imputation analysis.

Author: Ryhan Suny
Date: June 30, 2025
Version: 1.0.0

Following CLAUDE.md requirements:
- Functions ≤50 lines
- Meaningful variable names
- Comprehensive error handling
- Version numbering and timestamps
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Union, List
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_all_features(data_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load all feature datasets required for master table.
    
    Parameters:
    -----------
    data_dir : Union[str, Path]
        Directory containing derived data files
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping dataset names to DataFrames
        
    Raises:
    -------
    FileNotFoundError
        If any required file is missing
    """
    data_dir = Path(data_dir)
    
    required_files = {
        'cohort': 'cohort.parquet',
        'exposure': 'exposure.parquet',
        'mediator': 'mediator_autoencoder.parquet',
        'outcomes': 'outcomes.parquet',
        'confounders': 'confounders.parquet'
    }
    
    datasets = {}
    
    for name, filename in required_files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required file missing: {filepath}")
        
        logger.info(f"Loading {name} from {filepath}")
        datasets[name] = pd.read_parquet(filepath)
        logger.info(f"  Shape: {datasets[name].shape}")
    
    return datasets


def validate_merge_keys(datasets: Dict[str, pd.DataFrame]) -> Tuple[bool, Dict]:
    """
    Validate that all datasets have consistent Patient_IDs for merging.
    
    Parameters:
    -----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of loaded datasets
        
    Returns:
    --------
    Tuple[bool, Dict]
        (is_valid, validation_report)
    """
    all_patient_ids = set()
    patient_ids_by_dataset = {}
    
    # Collect all Patient_IDs
    for name, df in datasets.items():
        if 'Patient_ID' not in df.columns:
            return False, {"error": f"Missing Patient_ID in {name}"}
        
        ids = set(df['Patient_ID'].unique())
        patient_ids_by_dataset[name] = ids
        all_patient_ids.update(ids)
    
    # Check consistency
    report = {
        'n_unique_patients': len(all_patient_ids),
        'all_patients_present': True,
        'missing_by_dataset': {}
    }
    
    for name, ids in patient_ids_by_dataset.items():
        missing = all_patient_ids - ids
        if missing:
            report['all_patients_present'] = False
            report['missing_by_dataset'][name] = list(missing)
    
    is_valid = report['all_patients_present']
    
    if is_valid:
        logger.info(f"Validation passed: {report['n_unique_patients']} patients")
    else:
        logger.warning(f"Validation failed: {report}")
    
    return is_valid, report


def combine_features_with_missingness(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all features preserving missingness patterns.
    
    Uses outer joins to ensure no patient is dropped and missingness
    is preserved for proper imputation.
    
    Parameters:
    -----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of loaded datasets
        
    Returns:
    --------
    pd.DataFrame
        Combined master table with all features and preserved missingness
    """
    # Start with cohort as base
    master = datasets['cohort'].copy()
    logger.info(f"Starting with cohort: {master.shape}")
    
    # Merge each dataset
    merge_order = ['exposure', 'mediator', 'outcomes', 'confounders']
    
    for dataset_name in merge_order:
        df = datasets[dataset_name]
        
        # Identify overlapping columns (except Patient_ID)
        overlap = set(master.columns) & set(df.columns) - {'Patient_ID'}
        if overlap:
            logger.warning(f"Overlapping columns in {dataset_name}: {overlap}")
        
        # Merge
        n_before = len(master)
        master = master.merge(df, on='Patient_ID', how='outer', 
                            suffixes=('', f'_{dataset_name}'))
        n_after = len(master)
        
        if n_after != n_before:
            logger.warning(f"Row count changed after {dataset_name} merge: "
                         f"{n_before} -> {n_after}")
        
        logger.info(f"After merging {dataset_name}: {master.shape}")
    
    # Sort by Patient_ID for consistency
    master = master.sort_values('Patient_ID').reset_index(drop=True)
    
    return master


def get_missingness_report(df: pd.DataFrame) -> Dict:
    """
    Generate detailed missingness report for the combined dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Combined master table
        
    Returns:
    --------
    Dict
        Missingness statistics
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    
    missing_by_col = df.isna().sum()
    missing_pct_by_col = (missing_by_col / len(df) * 100).round(2)
    
    report = {
        'total_missing_pct': round(total_missing / total_cells * 100, 2),
        'columns_with_missing': list(missing_by_col[missing_by_col > 0].index),
        'missing_by_column': missing_pct_by_col[missing_pct_by_col > 0].to_dict(),
        'max_missing_pct': missing_pct_by_col.max(),
        'n_complete_rows': df.dropna().shape[0],
        'n_complete_columns': (missing_by_col == 0).sum()
    }
    
    return report


def create_pre_imputation_master(data_dir: Union[str, Path],
                                output_path: Union[str, Path]) -> Dict:
    """
    Main function to create pre-imputation master table.
    
    Parameters:
    -----------
    data_dir : Union[str, Path]
        Directory containing derived data files
    output_path : Union[str, Path]
        Path to save the master table
        
    Returns:
    --------
    Dict
        Execution summary with statistics
    """
    start_time = datetime.now()
    
    try:
        # Load all datasets
        datasets = load_all_features(data_dir)
        
        # Validate merge keys
        is_valid, validation_report = validate_merge_keys(datasets)
        if not is_valid:
            raise ValueError(f"Merge validation failed: {validation_report}")
        
        # Combine features
        master = combine_features_with_missingness(datasets)
        
        # Generate missingness report
        missing_report = get_missingness_report(master)
        
        # Save master table
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        master.to_parquet(output_path, index=False)
        
        # Save metadata
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'shape': list(master.shape),
            'columns': list(master.columns),
            'missingness_report': convert_to_serializable(missing_report),
            'validation_report': convert_to_serializable(validation_report),
            'source_files': list(datasets.keys())
        }
        
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ Pre-imputation master created successfully")
        logger.info(f"  Shape: {master.shape}")
        logger.info(f"  Missing: {missing_report['total_missing_pct']}%")
        logger.info(f"  Saved to: {output_path}")
        logger.info(f"  Duration: {duration:.2f}s")
        
        return {
            'success': True,
            'shape': master.shape,
            'missing_pct': missing_report['total_missing_pct'],
            'duration_seconds': duration,
            'output_path': str(output_path),
            'metadata_path': str(metadata_path)
        }
        
    except Exception as e:
        logger.error(f"Failed to create pre-imputation master: {e}")
        return {
            'success': False,
            'error': str(e),
            'duration_seconds': (datetime.now() - start_time).total_seconds()
        }


def main():
    """Command-line interface for pre-imputation master creation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create master table with all features before imputation"
    )
    parser.add_argument('--data-dir', default='data_derived',
                       help='Directory containing derived data files')
    parser.add_argument('--output', default='data_derived/master_with_missing.parquet',
                       help='Output path for master table')
    
    args = parser.parse_args()
    
    result = create_pre_imputation_master(args.data_dir, args.output)
    
    if not result['success']:
        logger.error(f"Execution failed: {result['error']}")
        exit(1)
    
    logger.info("Pre-imputation master creation complete")


if __name__ == "__main__":
    main()