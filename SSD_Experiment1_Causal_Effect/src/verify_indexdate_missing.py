#!/usr/bin/env python
"""
Verify IndexDate_lab missing data patterns
Purpose: Check data types and whether patients truly have NO labs
Author: Ryhan Suny
Date: 2025-01-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def main():
    """Verify IndexDate_lab missing patterns"""
    
    # Define paths - use forward slashes for Path
    base_path = Path("/mnt/c/Users/ProjectC4M/Documents/MSCM THESIS SSD/MSCM-THESIS-SSD---MENTAL-HEALTH-RESEARCH/SSD_Experiment1_Causal_Effect")
    
    # 1. Check master table data types
    log.info("=" * 60)
    log.info("1. CHECKING MASTER TABLE DATA TYPES")
    log.info("=" * 60)
    
    master_path = base_path / "data_derived" / "master_with_missing.parquet"
    if master_path.exists():
        master = pd.read_parquet(master_path)
        log.info(f"Master table shape: {master.shape}")
        log.info(f"\nIndexDate_lab data type: {master['IndexDate_lab'].dtype}")
        log.info(f"Is datetime64? {pd.api.types.is_datetime64_any_dtype(master['IndexDate_lab'])}")
        
        # Check missing counts
        missing_count = master['IndexDate_lab'].isna().sum()
        missing_pct = (missing_count / len(master)) * 100
        log.info(f"\nMissing IndexDate_lab: {missing_count:,} ({missing_pct:.1f}%)")
    else:
        log.error(f"Master table not found at {master_path}")
        return
    
    # 2. Load original cohort and lab data
    log.info("\n" + "=" * 60)
    log.info("2. CHECKING IF PATIENTS HAVE LAB RECORDS")
    log.info("=" * 60)
    
    cohort_path = base_path / "data_derived" / "cohort.parquet"
    lab_path = base_path / "Notebooks" / "data" / "interim" / "checkpoint_1_20250318_024427" / "lab.parquet"
    
    if cohort_path.exists() and lab_path.exists():
        cohort = pd.read_parquet(cohort_path)
        lab = pd.read_parquet(lab_path)
        
        log.info(f"Cohort shape: {cohort.shape}")
        log.info(f"Lab records shape: {lab.shape}")
        
        # Find patients with missing IndexDate_lab
        missing_pts = cohort[cohort['IndexDate_lab'].isna()]['Patient_ID'].unique()
        log.info(f"\nPatients with missing IndexDate_lab: {len(missing_pts):,}")
        
        # Check if these patients have ANY lab records
        pts_with_labs = lab[lab['Patient_ID'].isin(missing_pts)]['Patient_ID'].unique()
        pts_no_labs = set(missing_pts) - set(pts_with_labs)
        
        log.info(f"\nOf these {len(missing_pts):,} patients:")
        log.info(f"  - Have lab records but missing date: {len(pts_with_labs):,} ({len(pts_with_labs)/len(missing_pts)*100:.1f}%)")
        log.info(f"  - Have NO lab records at all: {len(pts_no_labs):,} ({len(pts_no_labs)/len(missing_pts)*100:.1f}%)")
        
        # Sample check - look at a few patients with labs but missing date
        if len(pts_with_labs) > 0:
            log.info("\n" + "-" * 40)
            log.info("SAMPLE: Patients with labs but missing IndexDate_lab")
            sample_pts = list(pts_with_labs)[:5]
            for pt in sample_pts:
                pt_labs = lab[lab['Patient_ID'] == pt]
                log.info(f"\nPatient {pt}:")
                log.info(f"  - Number of lab records: {len(pt_labs)}")
                if len(pt_labs) > 0:
                    log.info(f"  - Date range: {pt_labs['PerformedDate'].min()} to {pt_labs['PerformedDate'].max()}")
    else:
        log.error(f"Required files not found")
        log.error(f"Cohort exists: {cohort_path.exists()}")
        log.error(f"Lab exists: {lab_path.exists()}")
    
    # 3. Check datetime columns in master
    log.info("\n" + "=" * 60)
    log.info("3. ALL DATETIME COLUMNS IN MASTER TABLE")
    log.info("=" * 60)
    
    datetime_cols = master.select_dtypes(include=['datetime64']).columns.tolist()
    log.info(f"Number of datetime columns: {len(datetime_cols)}")
    for col in datetime_cols:
        missing = master[col].isna().sum()
        log.info(f"  - {col}: {missing:,} missing ({missing/len(master)*100:.1f}%)")
    
    # 4. Summary and recommendations
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"1. IndexDate_lab IS a datetime column: {pd.api.types.is_datetime64_any_dtype(master['IndexDate_lab'])}")
    log.info(f"2. Total missing IndexDate_lab: {missing_count:,}")
    if 'pts_no_labs' in locals():
        log.info(f"3. Patients with NO lab records: {len(pts_no_labs):,}")
        log.info(f"4. Patients with labs but missing date: {len(pts_with_labs):,}")
    log.info(f"5. Total datetime columns to exclude: {len(datetime_cols)}")

if __name__ == "__main__":
    main()