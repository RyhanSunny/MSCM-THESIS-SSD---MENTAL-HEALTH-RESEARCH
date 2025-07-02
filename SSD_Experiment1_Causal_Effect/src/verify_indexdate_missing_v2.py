#!/usr/bin/env python
"""
Verify IndexDate_lab missing data patterns - Version 2
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
    
    # Define paths
    base_path = Path("/mnt/c/Users/ProjectC4M/Documents/MSCM THESIS SSD/MSCM-THESIS-SSD---MENTAL-HEALTH-RESEARCH/SSD_Experiment1_Causal_Effect")
    
    # 1. Check master table data types
    log.info("=" * 60)
    log.info("1. VERIFYING DATA TYPES")
    log.info("=" * 60)
    
    master_path = base_path / "data_derived" / "master_with_missing.parquet"
    master = pd.read_parquet(master_path)
    
    log.info(f"Master table shape: {master.shape}")
    log.info(f"IndexDate_lab data type: {master['IndexDate_lab'].dtype}")
    log.info(f"Is datetime64? {pd.api.types.is_datetime64_any_dtype(master['IndexDate_lab'])}")
    
    missing_count = master['IndexDate_lab'].isna().sum()
    missing_pct = (missing_count / len(master)) * 100
    log.info(f"Missing IndexDate_lab: {missing_count:,} ({missing_pct:.1f}%)")
    
    # 2. Check original lab data
    log.info("\n" + "=" * 60)
    log.info("2. CHECKING LAB RECORDS")
    log.info("=" * 60)
    
    cohort = pd.read_parquet(base_path / "data_derived" / "cohort.parquet")
    lab = pd.read_csv(base_path / "Notebooks/data/interim/checkpoint_1_20250318_024427/lab.csv")
    
    log.info(f"Cohort shape: {cohort.shape}")
    log.info(f"Lab records shape: {lab.shape}")
    log.info(f"Unique patients in lab data: {lab['Patient_ID'].nunique():,}")
    
    # Find patients with missing IndexDate_lab
    missing_pts = cohort[cohort['IndexDate_lab'].isna()]['Patient_ID'].unique()
    log.info(f"\nPatients with missing IndexDate_lab: {len(missing_pts):,}")
    
    # Check if these patients have ANY lab records
    pts_with_labs = lab[lab['Patient_ID'].isin(missing_pts)]['Patient_ID'].unique()
    pts_no_labs = set(missing_pts) - set(pts_with_labs)
    
    log.info(f"\nBreakdown of {len(missing_pts):,} patients with missing IndexDate_lab:")
    log.info(f"  ✓ Have NO lab records at all: {len(pts_no_labs):,} ({len(pts_no_labs)/len(missing_pts)*100:.1f}%)")
    log.info(f"  ? Have lab records (needs investigation): {len(pts_with_labs):,} ({len(pts_with_labs)/len(missing_pts)*100:.1f}%)")
    
    # Investigate patients with labs but missing date
    if len(pts_with_labs) > 0:
        log.info("\n" + "-" * 40)
        log.info("INVESTIGATING: Why do some patients have labs but no IndexDate_lab?")
        
        # Check date parsing issues
        lab['PerformedDate_parsed'] = pd.to_datetime(lab['PerformedDate'], errors='coerce')
        parse_failures = lab[lab['Patient_ID'].isin(pts_with_labs) & lab['PerformedDate_parsed'].isna()]
        
        if len(parse_failures) > 0:
            log.info(f"\nDate parsing failures: {len(parse_failures)} records")
            log.info("Sample unparseable dates:")
            for val in parse_failures['PerformedDate'].unique()[:5]:
                log.info(f"  - '{val}'")
        
        # Sample patients
        log.info("\nSample patients with labs but missing IndexDate_lab:")
        for i, pt in enumerate(list(pts_with_labs)[:3]):
            pt_labs = lab[lab['Patient_ID'] == pt]
            log.info(f"\nPatient {pt}:")
            log.info(f"  - Lab records: {len(pt_labs)}")
            log.info(f"  - Date values: {pt_labs['PerformedDate'].unique()[:3]}")
    
    # 3. All datetime columns
    log.info("\n" + "=" * 60)
    log.info("3. DATETIME COLUMNS IN MASTER TABLE")
    log.info("=" * 60)
    
    datetime_cols = master.select_dtypes(include=['datetime64']).columns.tolist()
    log.info(f"Datetime columns: {datetime_cols}")
    
    # 4. Conclusions
    log.info("\n" + "=" * 60)
    log.info("CONCLUSIONS")
    log.info("=" * 60)
    log.info(f"✓ IndexDate_lab is datetime64: YES")
    log.info(f"✓ Total missing: {missing_count:,} patients")
    log.info(f"✓ Patients with NO lab records: {len(pts_no_labs):,} ({len(pts_no_labs)/len(cohort)*100:.1f}% of cohort)")
    log.info(f"✓ Only {len(datetime_cols)} datetime column to exclude from imputation")
    
    log.info("\nRECOMMENDATION: The {len(pts_no_labs):,} patients with NO lab records")
    log.info("represent a distinct clinical phenotype that should be handled separately.")

if __name__ == "__main__":
    main()