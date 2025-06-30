#!/usr/bin/env python3
"""
imputed_causal_pipeline.py - Run causal estimation on each imputed dataset separately

RATIONALE:
For proper multiple imputation analysis, we must:
1. Run complete causal analysis on each imputed dataset separately
2. Store results from each dataset with clear naming convention
3. Pool results using Rubin's Rules via rubins_pooling_engine.py

INTEGRATION POINTS:
- Uses existing 06_causal_estimators.py on each imputed dataset
- Reads imputation metadata from data_derived/imputed/imputation_metadata.json
- Outputs per-imputation results to results/causal_estimates_imp*.json
- Compatible with Rubin's pooling engine for final inference

Following CLAUDE.md requirements:
- TDD approach with comprehensive error handling
- Functions ≤50 lines
- Evidence-based implementation (no assumptions)
- Version numbering and timestamps

Author: Ryhan Suny (Toronto Metropolitan University)
Supervisor: Dr. Aziz Guergachi  
Research Team: Car4Mind, University of Toronto
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_imputation_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load multiple imputation metadata to identify imputed datasets.
    
    Parameters:
    -----------
    metadata_path : Path
        Path to imputation_metadata.json
        
    Returns:
    --------
    Dict[str, Any]
        Imputation metadata including file paths
        
    Raises:
    -------
    FileNotFoundError
        If metadata file doesn't exist
    ValueError
        If metadata is invalid
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Imputation metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Validate required fields
    required_fields = ['n_imputations', 'imputation_files']
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Missing required field in metadata: {field}")
    
    # Convert Windows paths to Unix paths for WSL compatibility
    if 'imputation_files' in metadata:
        unix_paths = []
        for path in metadata['imputation_files']:
            # Convert Windows path to Unix path for WSL
            unix_path = path.replace('C:\\', '/mnt/c/').replace('\\', '/')
            unix_paths.append(unix_path)
        metadata['imputation_files'] = unix_paths
    
    logger.info(f"Loaded metadata for {metadata['n_imputations']} imputations")
    return metadata


def run_causal_estimation_single_imputation(imputation_file: str,
                                          imputation_number: int,
                                          treatment_col: str,
                                          cluster_col: Optional[str] = None,
                                          results_dir: Path = Path("results")) -> Dict[str, Any]:
    """
    Run causal estimation on a single imputed dataset.
    
    DESIGN RATIONALE:
    This function calls the existing 06_causal_estimators.py script 
    with modified parameters to process a single imputed dataset.
    
    Parameters:
    -----------
    imputation_file : str
        Path to imputed dataset parquet file
    imputation_number : int
        Imputation dataset number (1-indexed)
    treatment_col : str
        Treatment column name
    cluster_col : Optional[str]
        Cluster column for robust standard errors
    results_dir : Path
        Output directory for results
        
    Returns:
    --------
    Dict[str, Any]
        Results from causal estimation
        
    Raises:
    -------
    subprocess.CalledProcessError
        If causal estimation script fails
    """
    results_dir.mkdir(exist_ok=True)
    
    # Build command for causal estimation
    cmd = [
        sys.executable,
        "src/06_causal_estimators.py",
        "--treatment-col", treatment_col,
        "--input-file", imputation_file,
        "--output-file", str(results_dir / f"causal_estimates_imp{imputation_number}.json")
    ]
    
    if cluster_col:
        cmd.extend(["--cluster-col", cluster_col])
    
    logger.info(f"Running causal estimation for imputation {imputation_number}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run causal estimation with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            check=True
        )
        
        logger.info(f"Causal estimation completed for imputation {imputation_number}")
        logger.info(f"STDOUT: {result.stdout}")
        
        if result.stderr:
            logger.warning(f"STDERR: {result.stderr}")
        
        # Load and return results
        output_file = results_dir / f"causal_estimates_imp{imputation_number}.json"
        if output_file.exists():
            with open(output_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Expected output file not created: {output_file}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"Causal estimation timed out for imputation {imputation_number}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Causal estimation failed for imputation {imputation_number}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise


def validate_imputed_file(file_path: str, treatment_col: str) -> bool:
    """
    Validate that imputed dataset has required columns and structure.
    
    Parameters:
    -----------
    file_path : str
        Path to imputed dataset
    treatment_col : str
        Required treatment column
        
    Returns:
    --------
    bool
        True if file is valid for causal analysis
    """
    try:
        # Check file exists
        if not Path(file_path).exists():
            logger.error(f"Imputed file not found: {file_path}")
            return False
        
        # Load and check basic structure
        df = pd.read_parquet(file_path)
        
        # Check required columns
        required_cols = [treatment_col, 'Patient_ID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns in {file_path}: {missing_cols}")
            return False
        
        # Check data size
        if len(df) == 0:
            logger.error(f"Empty dataset: {file_path}")
            return False
        
        logger.info(f"Validated imputed dataset: {file_path} ({len(df)} patients)")
        return True
        
    except Exception as e:
        logger.error(f"Error validating {file_path}: {e}")
        return False


def main():
    """
    Main pipeline for running causal estimation on multiple imputed datasets.
    
    Following CLAUDE.md requirement for functions ≤50 lines.
    """
    parser = argparse.ArgumentParser(
        description="Run causal estimation on multiple imputed datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--treatment-col', default='ssd_flag',
                       help='Treatment column name')
    parser.add_argument('--cluster-col', default='site_id',
                       help='Cluster column for robust standard errors')
    parser.add_argument('--metadata-file', 
                       default='data_derived/imputed/imputation_metadata.json',
                       help='Path to imputation metadata')
    parser.add_argument('--results-dir', default='results',
                       help='Results output directory')
    parser.add_argument('--max-imputations', type=int, default=None,
                       help='Maximum number of imputations to process (for testing)')
    
    args = parser.parse_args()
    
    try:
        # Load imputation metadata
        metadata_path = Path(args.metadata_file)
        metadata = load_imputation_metadata(metadata_path)
        
        imputation_files = metadata['imputation_files']
        n_imputations = len(imputation_files)
        
        if args.max_imputations:
            n_imputations = min(n_imputations, args.max_imputations)
            imputation_files = imputation_files[:n_imputations]
        
        logger.info(f"Processing {n_imputations} imputed datasets")
        
        # Results tracking
        results_summary = {
            'n_imputations_processed': 0,
            'successful_estimations': 0,
            'failed_estimations': 0,
            'results_files': [],
            'timestamp': datetime.now().isoformat(),
            'treatment_col': args.treatment_col,
            'cluster_col': args.cluster_col
        }
        
        # Process each imputed dataset
        for i, imputation_file in enumerate(imputation_files, 1):
            try:
                # Validate imputed dataset
                if not validate_imputed_file(imputation_file, args.treatment_col):
                    logger.error(f"Skipping invalid imputed dataset {i}")
                    results_summary['failed_estimations'] += 1
                    continue
                
                # Run causal estimation
                results = run_causal_estimation_single_imputation(
                    imputation_file=imputation_file,
                    imputation_number=i,
                    treatment_col=args.treatment_col,
                    cluster_col=args.cluster_col,
                    results_dir=Path(args.results_dir)
                )
                
                output_file = f"causal_estimates_imp{i}.json"
                results_summary['results_files'].append(output_file)
                results_summary['successful_estimations'] += 1
                
                logger.info(f"✓ Completed imputation {i}/{n_imputations}")
                
            except Exception as e:
                logger.error(f"Failed processing imputation {i}: {e}")
                results_summary['failed_estimations'] += 1
                continue
            
            results_summary['n_imputations_processed'] += 1
        
        # Save processing summary
        summary_file = Path(args.results_dir) / "imputed_causal_pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Final status
        logger.info(f"PIPELINE COMPLETE:")
        logger.info(f"  Processed: {results_summary['n_imputations_processed']}")
        logger.info(f"  Successful: {results_summary['successful_estimations']}")
        logger.info(f"  Failed: {results_summary['failed_estimations']}")
        logger.info(f"  Results files: {results_summary['results_files']}")
        
        if results_summary['failed_estimations'] > 0:
            logger.warning("Some imputations failed - check logs for details")
            sys.exit(1)
        
        logger.info("All imputed causal estimations completed successfully")
        logger.info("Next step: Run Rubin's Rules pooling with rubins_pooling_engine.py")
        
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}")
        raise


if __name__ == "__main__":
    main()