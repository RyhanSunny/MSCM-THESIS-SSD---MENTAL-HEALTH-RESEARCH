#!/usr/bin/env python3
"""
Enhanced SSD Cohort Builder with REAL Patient Data - NYD Enhancements

This version works with real patient data from the checkpoint directory.

Author: Ryhan Suny
Date: January 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
DERIVED = ROOT / "data_derived"
CONFIG_PATH = ROOT / "config" / "config.yaml"

# Find latest checkpoint
CHECKPOINTS = ROOT / "Notebooks" / "data" / "interim"
CKPT = max(CHECKPOINTS.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime)
logger.info(f"Using checkpoint: {CKPT.name}")


def load_real_nyd_data():
    """Load real NYD diagnosis records from encounter_diagnosis"""
    logger.info("Loading real NYD diagnosis records...")
    
    # Load encounter diagnosis data
    enc_diag = pd.read_parquet(CKPT / "encounter_diagnosis.parquet")
    
    # NYD codes (780-799 range)
    nyd_pattern = r'^(78[0-9]|799)'
    nyd_diagnoses = enc_diag[
        enc_diag['DiagnosisCode_calc'].str.match(nyd_pattern, na=False)
    ].copy()
    
    # Get patient IDs and ICD codes
    nyd_data = nyd_diagnoses[['Patient_ID', 'DiagnosisCode_calc']].copy()
    nyd_data = nyd_data.rename(columns={'DiagnosisCode_calc': 'ICD_code'})
    nyd_data = nyd_data.drop_duplicates()
    
    logger.info(f"Real NYD data loaded: {len(nyd_data):,} records for {nyd_data['Patient_ID'].nunique():,} patients")
    
    return nyd_data


def create_nyd_body_part_mapping():
    """
    Create comprehensive NYD ICD code to body part mapping
    """
    logger.info("Creating NYD body part mapping...")
    
    nyd_mapping = {
        # General/Unspecified conditions
        '799.9': 'General',
        '780.9': 'General', 
        '799.8': 'General',
        '780.79': 'General',
        
        # Mental/Behavioral conditions  
        'V71.0': 'Mental/Behavioral',
        'V71.09': 'Mental/Behavioral',
        '300.9': 'Mental/Behavioral',
        '799.29': 'Mental/Behavioral',
        
        # Neurological conditions
        'V71.1': 'Neurological',
        '780.4': 'Neurological',
        '344.9': 'Neurological',
        '781.3': 'Neurological',
        
        # Cardiovascular conditions
        '785.9': 'Cardiovascular',
        '796.3': 'Cardiovascular',
        '785.1': 'Cardiovascular',
        
        # Respiratory conditions
        '786.9': 'Respiratory',
        '786.50': 'Respiratory',
        '786.59': 'Respiratory',
        
        # Gastrointestinal conditions
        '787.9': 'Gastrointestinal',
        '789.9': 'Gastrointestinal',
        '787.91': 'Gastrointestinal',
        
        # Musculoskeletal conditions
        '719.9': 'Musculoskeletal',
        '729.9': 'Musculoskeletal',
        '724.9': 'Musculoskeletal',
        
        # Dermatological conditions
        '709.9': 'Dermatological',
        '782.9': 'Dermatological',
        
        # Genitourinary conditions
        '599.9': 'Genitourinary',
        '788.9': 'Genitourinary'
    }
    
    logger.info(f"Created NYD mapping with {len(nyd_mapping)} codes across {len(set(nyd_mapping.values()))} body systems")
    
    return nyd_mapping


def add_nyd_binary_flags(cohort_df, nyd_data):
    """
    Add NYD binary flags to cohort based on REAL NYD diagnoses
    """
    logger.info("Adding NYD binary flags to real patient cohort...")
    
    enhanced_cohort = cohort_df.copy()
    nyd_mapping = create_nyd_body_part_mapping()
    
    # Handle empty NYD data
    if len(nyd_data) == 0:
        logger.info("No NYD data found - setting all flags to 0")
        body_part_flags = ['NYD_yn', 'NYD_general_yn', 'NYD_mental_yn', 'NYD_neuro_yn', 
                          'NYD_cardio_yn', 'NYD_resp_yn', 'NYD_gi_yn', 'NYD_musculo_yn', 
                          'NYD_derm_yn', 'NYD_gu_yn']
        for flag in body_part_flags:
            enhanced_cohort[flag] = 0
        return enhanced_cohort
    
    # Map ICD codes to body parts
    nyd_data = nyd_data.copy()
    nyd_data['body_part'] = nyd_data['ICD_code'].map(nyd_mapping).fillna('Unknown')
    
    # Calculate overall NYD binary flag
    patients_with_nyd = set(nyd_data['Patient_ID'].unique())
    enhanced_cohort['NYD_yn'] = enhanced_cohort['Patient_ID'].isin(patients_with_nyd).astype(int)
    
    # Calculate body part-specific flags
    body_part_mapping = {
        'General': 'NYD_general_yn',
        'Mental/Behavioral': 'NYD_mental_yn', 
        'Neurological': 'NYD_neuro_yn',
        'Cardiovascular': 'NYD_cardio_yn',
        'Respiratory': 'NYD_resp_yn',
        'Gastrointestinal': 'NYD_gi_yn',
        'Musculoskeletal': 'NYD_musculo_yn',
        'Dermatological': 'NYD_derm_yn',
        'Genitourinary': 'NYD_gu_yn'
    }
    
    for body_part, column_name in body_part_mapping.items():
        patients_with_body_part = set(
            nyd_data[nyd_data['body_part'] == body_part]['Patient_ID'].unique()
        )
        enhanced_cohort[column_name] = enhanced_cohort['Patient_ID'].isin(patients_with_body_part).astype(int)
    
    # Log summary statistics for REAL data
    nyd_count = enhanced_cohort['NYD_yn'].sum()
    total_patients = len(enhanced_cohort)
    logger.info(f"NYD binary flags added to REAL cohort:")
    logger.info(f"  Total cohort size: {total_patients:,} patients")
    logger.info(f"  Patients with NYD diagnoses: {nyd_count:,} ({nyd_count/total_patients*100:.1f}%)")
    
    for body_part, column_name in body_part_mapping.items():
        count = enhanced_cohort[column_name].sum()
        if count > 0:
            logger.info(f"  {body_part}: {count:,} patients ({count/total_patients*100:.1f}%)")
    
    return enhanced_cohort


def run_real_enhanced_cohort():
    """Run enhanced cohort builder with REAL patient data"""
    logger.info("=== RUNNING ENHANCED COHORT BUILDER WITH REAL PATIENT DATA ===")
    
    # Load real base cohort
    logger.info("Loading real base cohort...")
    base_cohort = pd.read_parquet(DERIVED / "cohort.parquet")
    logger.info(f"Base cohort loaded: {len(base_cohort):,} patients")
    
    # Load real NYD data
    real_nyd_data = load_real_nyd_data()
    
    # Filter NYD data to cohort patients only
    cohort_patients = set(base_cohort['Patient_ID'])
    real_nyd_data = real_nyd_data[real_nyd_data['Patient_ID'].isin(cohort_patients)]
    logger.info(f"NYD data filtered to cohort: {len(real_nyd_data):,} records for {real_nyd_data['Patient_ID'].nunique():,} patients")
    
    # Build enhanced cohort
    enhanced_cohort = add_nyd_binary_flags(base_cohort, real_nyd_data)
    
    # Save enhanced cohort
    output_path = DERIVED / "cohort_enhanced_real.parquet"
    enhanced_cohort.to_parquet(output_path, index=False)
    logger.info(f"Enhanced cohort saved: {output_path}")
    
    # Generate summary report
    total_patients = len(enhanced_cohort)
    nyd_patients = enhanced_cohort['NYD_yn'].sum()
    
    summary = f"""
# Enhanced Cohort Summary - REAL Patient Data
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Real Patient Statistics
- **Total Cohort Size**: {total_patients:,} patients
- **Patients with NYD Diagnoses**: {nyd_patients:,} ({nyd_patients/total_patients*100:.1f}%)

## Body Part Distribution (Real Data)
"""
    
    body_part_columns = [col for col in enhanced_cohort.columns if col.startswith('NYD_') and col.endswith('_yn') and col != 'NYD_yn']
    for col in body_part_columns:
        part_name = col.replace('NYD_', '').replace('_yn', '')
        count = enhanced_cohort[col].sum()
        pct = count/total_patients*100
        summary += f"- **{part_name}**: {count:,} patients ({pct:.1f}%)\n"
    
    # Save summary
    summary_path = DERIVED / "cohort_enhanced_real_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Summary report saved: {summary_path}")
    logger.info("=== REAL ENHANCED COHORT BUILDING COMPLETE ===")
    
    return enhanced_cohort


if __name__ == "__main__":
    # Run with REAL patient data
    enhanced_cohort = run_real_enhanced_cohort()
    print(f"\nEnhanced cohort created with {len(enhanced_cohort):,} REAL patients!") 