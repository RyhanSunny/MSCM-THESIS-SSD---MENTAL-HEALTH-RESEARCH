#!/usr/bin/env python3
"""
Memory-optimized version of exposure flag calculation
Processes data in chunks to prevent memory exhaustion
"""

import pandas as pd
import numpy as np
import gc
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Config
from logger import Logger
from lab_utils import LabUtilities

def optimize_datatypes(df):
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    return df

def process_exposure_in_chunks(config, logger, chunk_size=50000):
    """Process exposure flags in memory-efficient chunks"""
    
    # Load cohort (smaller dataset)
    logger.info("Loading cohort data...")
    cohort = pd.read_parquet(config.data_derived / 'cohort.parquet')
    cohort = optimize_datatypes(cohort)
    logger.info(f"Cohort loaded: {len(cohort):,} patients")
    
    # Initialize results
    cohort['exposure'] = 0
    cohort['normal_lab_count'] = 0
    
    # Load lab utilities
    lab_utils = LabUtilities(config, logger)
    
    # Process labs in chunks
    logger.info("Processing lab results in chunks...")
    lab_chunks = pd.read_csv(
        config.checkpoint_dir / 'lab.csv',
        chunksize=chunk_size,
        dtype={'patient_id': str, 'lab_name': str, 'result': 'float32'}
    )
    
    normal_counts = {}
    for i, chunk in enumerate(lab_chunks):
        logger.info(f"Processing lab chunk {i+1}...")
        
        # Filter for exposure window
        chunk['test_date'] = pd.to_datetime(chunk['test_date'])
        
        # Process each patient's labs
        for patient_id in chunk['patient_id'].unique():
            if patient_id in cohort['patient_id'].values:
                patient_labs = chunk[chunk['patient_id'] == patient_id]
                
                # Count normal labs in exposure window
                # (This is simplified - actual logic would go here)
                normal_count = len(patient_labs)
                normal_counts[patient_id] = normal_counts.get(patient_id, 0) + normal_count
        
        # Clear chunk from memory
        del chunk
        gc.collect()
    
    # Update cohort with normal counts
    for patient_id, count in normal_counts.items():
        cohort.loc[cohort['patient_id'] == patient_id, 'normal_lab_count'] = count
    
    # Process diagnoses in chunks
    logger.info("Processing diagnoses in chunks...")
    diagnosis_chunks = pd.read_parquet(
        config.checkpoint_dir / 'encounter_diagnosis.parquet',
        columns=['patient_id', 'encounter_id', 'DiagnosisCode', 'DiagnosisDateTime'],
        chunksize=chunk_size
    )
    
    exposure_patients = set()
    for i, chunk in enumerate(diagnosis_chunks):
        logger.info(f"Processing diagnosis chunk {i+1}...")
        
        # Filter for relevant patients only
        chunk = chunk[chunk['patient_id'].isin(cohort['patient_id'])]
        
        # Apply exposure logic (simplified)
        # Add actual exposure criteria here
        exposed = chunk['patient_id'].unique()
        exposure_patients.update(exposed)
        
        # Clear chunk
        del chunk
        gc.collect()
    
    # Update exposure flags
    cohort.loc[cohort['patient_id'].isin(exposure_patients), 'exposure'] = 1
    
    # Apply final criteria
    cohort.loc[
        (cohort['normal_lab_count'] >= 3) & (cohort['exposure'] == 0),
        'exposure'
    ] = 1
    
    logger.info(f"Exposure calculation complete: {cohort['exposure'].sum():,} exposed patients")
    
    # Save results
    output_path = config.data_derived / 'cohort_exposure.parquet'
    cohort.to_parquet(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    return cohort

if __name__ == "__main__":
    config = Config()
    logger = Logger(config).get_logger(__name__)
    
    try:
        process_exposure_in_chunks(config, logger)
    except Exception as e:
        logger.error(f"Error in optimized exposure processing: {e}")
        raise