"""
MC-SIMEX Flag Merger: Integrates bias-corrected flags into patient master table.

This module merges the ssd_flag_adj column from cohort_bias_corrected.parquet 
into the main patient_master.parquet file, enabling downstream scripts to use
the bias-corrected treatment assignment.

Author: Ryhan Suny
Date: 2025-06-21
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Union, Optional
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_bias_corrected_flag(
    master_path: Union[str, Path],
    corrected_path: Union[str, Path],
    backup: bool = True
) -> None:
    """
    Merge ssd_flag_adj from bias-corrected cohort into patient master table.
    
    Parameters:
    -----------
    master_path : Union[str, Path]
        Path to patient_master.parquet file
    corrected_path : Union[str, Path]
        Path to cohort_bias_corrected.parquet file
    backup : bool, default=True
        Whether to create backup of original master file
        
    Raises:
    -------
    FileNotFoundError
        If master_path or corrected_path doesn't exist
    ValueError
        If patient IDs don't match between files
    KeyError
        If ssd_flag_adj column missing from corrected file
    """
    master_path = Path(master_path)
    corrected_path = Path(corrected_path)
    
    # Validate files exist
    if not master_path.exists():
        raise FileNotFoundError(f"Patient master file not found: {master_path}")
    if not corrected_path.exists():
        raise FileNotFoundError(f"Bias-corrected file not found: {corrected_path}")
    
    logger.info(f"Loading patient master from {master_path}")
    master_df = pd.read_parquet(master_path)
    initial_rows = len(master_df)
    
    logger.info(f"Loading bias-corrected cohort from {corrected_path}")
    corrected_df = pd.read_parquet(corrected_path)
    
    # Validate required columns
    if 'ssd_flag_adj' not in corrected_df.columns:
        raise KeyError("ssd_flag_adj column not found in bias-corrected file")
    
    if 'Patient_ID' not in master_df.columns or 'Patient_ID' not in corrected_df.columns:
        raise KeyError("Patient_ID column required in both files for alignment")
    
    # Validate patient ID alignment
    master_ids = set(master_df['patient_id'])
    corrected_ids = set(corrected_df['patient_id'])
    
    if master_ids != corrected_ids:
        missing_in_corrected = master_ids - corrected_ids
        missing_in_master = corrected_ids - master_ids
        error_msg = f"Patient IDs do not match between files. "
        if missing_in_corrected:
            error_msg += f"Missing in corrected: {len(missing_in_corrected)} IDs. "
        if missing_in_master:
            error_msg += f"Missing in master: {len(missing_in_master)} IDs."
        raise ValueError(error_msg)
    
    # Create backup if requested
    if backup:
        backup_path = master_path.with_suffix('.parquet.backup')
        logger.info(f"Creating backup at {backup_path}")
        master_df.to_parquet(backup_path, index=False)
    
    # Merge ssd_flag_adj column
    logger.info("Merging ssd_flag_adj column...")
    
    # Sort both dataframes by Patient_ID for reliable merge
    master_df = master_df.sort_values('patient_id').reset_index(drop=True)
    corrected_df = corrected_df.sort_values('patient_id').reset_index(drop=True)
    
    # Extract just the ssd_flag_adj column with Patient_ID for merge
    adj_flags = corrected_df[['patient_id', 'ssd_flag_adj']].copy()
    
    # Merge on patient_id
    merged_df = master_df.merge(adj_flags, on='Patient_ID', how='left', validate='one_to_one')
    
    # Validate merge results
    assert len(merged_df) == initial_rows, f"Row count changed: {initial_rows} -> {len(merged_df)}"
    assert merged_df['ssd_flag_adj'].isna().sum() == 0, "Some ssd_flag_adj values are missing after merge"
    
    # Log summary statistics
    original_flag_sum = master_df['ssd_flag'].sum() if 'ssd_flag' in master_df.columns else 0
    adjusted_flag_sum = merged_df['ssd_flag_adj'].sum()
    logger.info(f"Flag comparison - Original: {original_flag_sum}, Adjusted: {adjusted_flag_sum}")
    logger.info(f"Net change: {adjusted_flag_sum - original_flag_sum} patients")
    
    # Save updated master file
    logger.info(f"Saving updated master file to {master_path}")
    merged_df.to_parquet(master_path, index=False)
    
    logger.info("✓ MC-SIMEX flag integration completed successfully")


def main():
    """Command-line interface for MC-SIMEX flag merger."""
    parser = argparse.ArgumentParser(
        description="Merge MC-SIMEX bias-corrected flags into patient master table"
    )
    parser.add_argument(
        '--master-path', 
        default='data_derived/patient_master.parquet',
        help='Path to patient master parquet file'
    )
    parser.add_argument(
        '--corrected-path',
        default='data_derived/cohort_bias_corrected.parquet', 
        help='Path to bias-corrected cohort parquet file'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup of original master file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate files and show merge preview without making changes'
    )
    
    args = parser.parse_args()
    
    try:
        if args.dry_run:
            logger.info("DRY RUN MODE - No files will be modified")
            
            # Load and validate files
            master_path = Path(args.master_path)
            corrected_path = Path(args.corrected_path)
            
            if not master_path.exists():
                logger.error(f"Master file not found: {master_path}")
                return 1
            if not corrected_path.exists():
                logger.error(f"Corrected file not found: {corrected_path}")
                return 1
                
            master_df = pd.read_parquet(master_path)
            corrected_df = pd.read_parquet(corrected_path)
            
            logger.info(f"Master file: {len(master_df)} rows, {len(master_df.columns)} columns")
            logger.info(f"Corrected file: {len(corrected_df)} rows, {len(corrected_df.columns)} columns")
            
            if 'ssd_flag_adj' in corrected_df.columns:
                adj_sum = corrected_df['ssd_flag_adj'].sum()
                logger.info(f"ssd_flag_adj: {adj_sum}/{len(corrected_df)} flagged ({adj_sum/len(corrected_df):.3f} rate)")
            else:
                logger.error("ssd_flag_adj column not found in corrected file")
                return 1
                
            logger.info("✓ Dry run completed successfully")
            return 0
        else:
            merge_bias_corrected_flag(
                master_path=args.master_path,
                corrected_path=args.corrected_path,
                backup=not args.no_backup
            )
            return 0
            
    except Exception as e:
        logger.error(f"Error during MC-SIMEX flag merge: {e}")
        return 1


if __name__ == "__main__":
    exit(main())