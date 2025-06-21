"""
SES Data Cleaner: Removes synthetic socioeconomic status data from pipeline.

This module identifies and removes synthetic SES data from the pipeline,
including ICES marginals CSV and code references to unavailable SES variables.

Author: Ryhan Suny
Date: 2025-06-21
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
import argparse
from typing import List, Union, Optional
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def remove_synthetic_ices_marginals(
    ices_path: Union[str, Path],
    backup: bool = True
) -> None:
    """
    Remove synthetic socioeconomic quintile data from ICES marginals CSV.
    
    Parameters:
    -----------
    ices_path : Union[str, Path]
        Path to ices_marginals.csv file
    backup : bool, default=True
        Whether to create backup before modification
        
    Raises:
    -------
    FileNotFoundError
        If ices_path doesn't exist
    """
    ices_path = Path(ices_path)
    
    if not ices_path.exists():
        raise FileNotFoundError(f"ICES marginals file not found: {ices_path}")
    
    logger.info(f"Removing synthetic SES data from {ices_path}")
    
    # Create backup if requested
    if backup:
        backup_path = ices_path.with_suffix('.csv.backup')
        logger.info(f"Creating backup at {backup_path}")
        shutil.copy2(ices_path, backup_path)
    
    # Load CSV
    df = pd.read_csv(ices_path)
    initial_rows = len(df)
    
    # Remove socioeconomic_quintile rows
    ses_rows_before = df[df['variable'] == 'socioeconomic_quintile']
    logger.info(f"Found {len(ses_rows_before)} socioeconomic_quintile rows to remove")
    
    df_cleaned = df[df['variable'] != 'socioeconomic_quintile'].copy()
    final_rows = len(df_cleaned)
    
    # Save cleaned CSV
    df_cleaned.to_csv(ices_path, index=False)
    
    logger.info(f"✓ Removed {initial_rows - final_rows} SES rows from ICES marginals")
    logger.info(f"Final file has {final_rows} rows (was {initial_rows})")


def remove_ses_references_from_code(
    source_path: Union[str, Path],
    backup: bool = True
) -> None:
    """
    Remove or comment out SES references from Python source code.
    
    Parameters:
    -----------
    source_path : Union[str, Path]
        Path to Python source file
    backup : bool, default=True
        Whether to create backup before modification
        
    Raises:
    -------
    FileNotFoundError
        If source_path doesn't exist
    """
    source_path = Path(source_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    logger.info(f"Removing SES references from {source_path}")
    
    # Create backup if requested
    if backup:
        backup_path = source_path.with_suffix('.py.backup')
        logger.info(f"Creating backup at {backup_path}")
        shutil.copy2(source_path, backup_path)
    
    # Read source file
    with open(source_path, 'r') as f:
        content = f.read()
    
    # Define SES patterns to remove
    ses_patterns = [
        r".*deprivation_quintile.*",
        r".*socioeconomic_quintile.*",
        r".*high_deprivation.*",
        r".*low_deprivation.*",
        r".*income_quintile.*"
    ]
    
    lines = content.split('\n')
    modified_lines = []
    changes_made = 0
    
    for line in lines:
        line_modified = False
        for pattern in ses_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Comment out the line with explanation
                indentation = len(line) - len(line.lstrip())
                commented_line = ' ' * indentation + f"# REMOVED SES: {line.strip()}"
                modified_lines.append(commented_line)
                changes_made += 1
                line_modified = True
                break
        
        if not line_modified:
            modified_lines.append(line)
    
    # Write modified content
    modified_content = '\n'.join(modified_lines)
    with open(source_path, 'w') as f:
        f.write(modified_content)
    
    logger.info(f"✓ Commented out {changes_made} SES references in {source_path}")


def validate_no_synthetic_ses(
    ices_path: Union[str, Path],
    source_files: List[Union[str, Path]]
) -> bool:
    """
    Validate that no synthetic SES data remains in pipeline.
    
    Parameters:
    -----------
    ices_path : Union[str, Path]
        Path to ices_marginals.csv file
    source_files : List[Union[str, Path]]
        List of source files to check
        
    Returns:
    --------
    bool
        True if no synthetic SES data found, False otherwise
    """
    logger.info("Validating removal of synthetic SES data...")
    
    validation_passed = True
    
    # Check ICES marginals CSV
    ices_path = Path(ices_path)
    if ices_path.exists():
        df = pd.read_csv(ices_path)
        ses_rows = df[df['variable'] == 'socioeconomic_quintile']
        if len(ses_rows) > 0:
            logger.error(f"Found {len(ses_rows)} socioeconomic_quintile rows in {ices_path}")
            validation_passed = False
        else:
            logger.info(f"✓ No SES data found in {ices_path}")
    
    # Check source files for uncommented SES references
    ses_patterns = ['deprivation_quintile', 'socioeconomic_quintile', 'income_quintile']
    
    for source_file in source_files:
        source_path = Path(source_file)
        
        # Skip the SES cleaner module itself (it legitimately contains SES references)
        if source_path.name == 'ses_data_cleaner.py':
            logger.info(f"✓ Skipping SES cleaner module: {source_path}")
            continue
            
        if source_path.exists():
            with open(source_path, 'r') as f:
                content = f.read()
            
            for pattern in ses_patterns:
                # Look for uncommented references
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if pattern in line and not line.strip().startswith('#'):
                        logger.error(f"Found uncommented SES reference in {source_path}:{i}: {line.strip()}")
                        validation_passed = False
            
            if validation_passed:
                logger.info(f"✓ No uncommented SES references in {source_path}")
    
    if validation_passed:
        logger.info("✓ All synthetic SES data successfully removed")
    else:
        logger.error("❌ Synthetic SES data still present - manual review required")
    
    return validation_passed


def find_ses_references_in_codebase(
    src_dir: Union[str, Path] = "src"
) -> List[Path]:
    """
    Find all Python files with SES references in the codebase.
    
    Parameters:
    -----------
    src_dir : Union[str, Path], default="src"
        Directory to search for Python files
        
    Returns:
    --------
    List[Path]
        List of Python files containing SES references
    """
    src_dir = Path(src_dir)
    ses_patterns = ['deprivation_quintile', 'socioeconomic_quintile', 'income_quintile']
    files_with_ses = []
    
    for py_file in src_dir.glob("**/*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            for pattern in ses_patterns:
                if pattern in content:
                    files_with_ses.append(py_file)
                    break
        except Exception as e:
            logger.warning(f"Could not read {py_file}: {e}")
    
    return files_with_ses


def main():
    """Command-line interface for SES data cleaner."""
    parser = argparse.ArgumentParser(
        description="Remove synthetic socioeconomic status data from pipeline"
    )
    parser.add_argument(
        '--ices-path',
        default='data/external/ices_marginals.csv',
        help='Path to ICES marginals CSV file'
    )
    parser.add_argument(
        '--src-dir',
        default='src',
        help='Source directory to scan for SES references'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without making changes'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate that SES data has been removed'
    )
    
    args = parser.parse_args()
    
    try:
        if args.validate_only:
            # Find source files with SES references
            source_files = find_ses_references_in_codebase(args.src_dir)
            
            # Validate
            is_clean = validate_no_synthetic_ses(
                ices_path=args.ices_path,
                source_files=source_files
            )
            return 0 if is_clean else 1
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No files will be modified")
            
            # Check ICES file
            ices_path = Path(args.ices_path)
            if ices_path.exists():
                df = pd.read_csv(ices_path)
                ses_rows = df[df['variable'] == 'socioeconomic_quintile']
                logger.info(f"Would remove {len(ses_rows)} SES rows from {ices_path}")
            else:
                logger.info(f"ICES file not found: {ices_path}")
            
            # Check source files
            source_files = find_ses_references_in_codebase(args.src_dir)
            logger.info(f"Found {len(source_files)} source files with SES references:")
            for file in source_files:
                logger.info(f"  - {file}")
            
            return 0
        
        # Remove synthetic SES data
        logger.info("Removing synthetic SES data from pipeline...")
        
        # Remove from ICES marginals
        ices_path = Path(args.ices_path)
        if ices_path.exists():
            remove_synthetic_ices_marginals(ices_path, backup=not args.no_backup)
        else:
            logger.warning(f"ICES marginals file not found: {ices_path}")
        
        # Remove from source files
        source_files = find_ses_references_in_codebase(args.src_dir)
        logger.info(f"Processing {len(source_files)} source files with SES references...")
        
        for source_file in source_files:
            remove_ses_references_from_code(source_file, backup=not args.no_backup)
        
        # Validate removal
        is_clean = validate_no_synthetic_ses(
            ices_path=ices_path,
            source_files=source_files
        )
        
        if is_clean:
            logger.info("✓ Successfully removed all synthetic SES data")
            return 0
        else:
            logger.error("❌ Some SES data may remain - manual review recommended")
            return 1
            
    except Exception as e:
        logger.error(f"Error during SES data removal: {e}")
        return 1


if __name__ == "__main__":
    exit(main())