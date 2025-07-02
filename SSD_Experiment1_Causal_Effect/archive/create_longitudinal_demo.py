#!/usr/bin/env python3
"""
create_longitudinal_demo.py - Create toy longitudinal data for MSM testing

Creates a small longitudinal dataset for testing MSM functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_longitudinal_demo():
    """Create toy longitudinal data"""
    np.random.seed(42)
    
    # Parameters
    n_patients = 1000
    n_timepoints = 6
    
    data = []
    
    for patient_id in range(n_patients):
        # Patient baseline characteristics
        age = np.random.normal(45, 15)
        sex_M = np.random.binomial(1, 0.5)
        baseline_severity = np.random.uniform(0.1, 0.8)
        
        for t in range(n_timepoints):
            # Time-varying variables
            time_period = t
            stress_level = baseline_severity + np.random.normal(0, 0.1) + t * 0.01
            severity = baseline_severity + np.random.normal(0, 0.1) + t * 0.02
            
            # Time-varying treatment
            ssd_prob = np.clip(severity * 0.6 + stress_level * 0.3 + age * 0.001, 0.05, 0.95)
            ssd_flag = np.random.binomial(1, ssd_prob)
            
            # Outcome influenced by treatment and confounders
            encounter_lambda = np.clip(
                2 + ssd_flag * 1.5 + severity * 2 + stress_level * 1 + age * 0.02,
                0.5, 15
            )
            encounters = np.random.poisson(encounter_lambda)
            
            # Other variables
            index_date = pd.Timestamp('2015-01-01') + pd.DateOffset(months=t)
            
            data.append({
                'patient_id': patient_id,
                'time_period': t,
                'index_date': index_date,
                'age': age,
                'sex_M': sex_M,
                'stress_level': stress_level,
                'severity': severity,
                'ssd_flag': ssd_flag,
                'encounters': encounters,
                'baseline_severity': baseline_severity,
                'total_encounters': encounters  # For compatibility
            })
    
    df = pd.DataFrame(data)
    
    # Save to parquet
    output_path = Path("tests/data/longitudinal_demo.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Created longitudinal demo data: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Time points per patient: {df.groupby('patient_id').size().iloc[0]}")
    print(f"SSD prevalence: {df['ssd_flag'].mean():.2f}")
    print(f"Mean encounters: {df['encounters'].mean():.1f}")
    
    return output_path

if __name__ == "__main__":
    create_longitudinal_demo()