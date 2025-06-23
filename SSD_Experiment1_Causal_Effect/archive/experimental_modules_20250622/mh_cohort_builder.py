#!/usr/bin/env python3
"""
mh_cohort_builder.py - Mental Health-specific cohort enhancements

Implements mental health ICD filtering, enhanced drug persistence calculation,
and psychiatric referral logic as specified in SSD THESIS blueprint.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_mental_health_diagnosis(diagnosis_code: str) -> bool:
    """
    Check if diagnosis code indicates mental health condition
    
    Parameters:
    -----------
    diagnosis_code : str
        ICD-9 or ICD-10 diagnosis code
        
    Returns:
    --------
    bool
        True if code indicates mental health condition
        
    Example:
    --------
    >>> is_mental_health_diagnosis('F32.1')
    True
    >>> is_mental_health_diagnosis('I10') 
    False
    """
    if pd.isna(diagnosis_code):
        return False
        
    code = str(diagnosis_code).strip().upper()
    
    # ICD-10 Mental Health codes (F32-F48)
    if re.match(r'^F(3[2-9]|4[0-8])', code):
        return True
    
    # ICD-9 Mental Health codes (296.*, 300.*)
    if re.match(r'^296\.', code):
        return True
    if re.match(r'^300\.', code):
        return True
        
    return False


def categorize_mental_health_diagnosis(diagnosis_code: str) -> str:
    """
    Categorize mental health diagnosis into clinical groups
    
    Parameters:
    -----------
    diagnosis_code : str
        ICD diagnosis code
        
    Returns:
    --------
    str
        Mental health category or 'not_mental_health'
    """
    if not is_mental_health_diagnosis(diagnosis_code):
        return 'not_mental_health'
    
    code = str(diagnosis_code).strip().upper()
    
    # Depression (F32-F34, 296.2-296.3)
    if re.match(r'^F3[2-4]', code) or re.match(r'^296\.[23]', code):
        return 'depression'
    
    # Anxiety (F40-F42, 300.0*)
    if re.match(r'^F4[0-2]', code) or re.match(r'^300\.0', code):
        return 'anxiety'
    
    # PTSD/Stress (F43, 308.*, 309.*)
    if re.match(r'^F43', code) or re.match(r'^30[89]\.', code):
        return 'trauma_stress'
    
    # Bipolar (F31, 296.0-296.1, 296.4-296.8)
    if re.match(r'^F31', code) or re.match(r'^296\.[01]', code) or re.match(r'^296\.[4-8]', code):
        return 'bipolar'
    
    # Other mental health
    return 'other_mental_health'


def filter_mental_health_patients(df: pd.DataFrame, 
                                 diagnosis_col: str = 'diagnosis_codes') -> pd.DataFrame:
    """
    Filter dataframe to mental health patients only
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with diagnosis codes
    diagnosis_col : str
        Column containing diagnosis codes
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe with mental health patients only
    """
    logger.info("Filtering for mental health patients...")
    
    # Apply mental health filter
    mh_mask = df[diagnosis_col].apply(is_mental_health_diagnosis)
    mh_patients = df[mh_mask].copy()
    
    # Add mental health category
    mh_patients['mh_diagnosis_category'] = mh_patients[diagnosis_col].apply(
        categorize_mental_health_diagnosis
    )
    mh_patients['is_mental_health'] = True
    
    logger.info(f"Mental health patients: {len(mh_patients):,} of {len(df):,} "
               f"({len(mh_patients)/len(df)*100:.1f}%)")
    
    return mh_patients


def classify_psychotropic_drugs(drug_codes: List[str]) -> Dict[str, str]:
    """
    Classify drug codes into psychotropic categories
    
    Parameters:
    -----------
    drug_codes : List[str]
        List of ATC drug codes
        
    Returns:
    --------
    Dict[str, str]
        Mapping of drug codes to classifications
    """
    classifications = {}
    
    for code in drug_codes:
        if pd.isna(code):
            classifications[code] = 'unknown'
            continue
            
        code_str = str(code).strip().upper()
        
        # N06A - Antidepressants
        if code_str.startswith('N06A'):
            classifications[code] = 'antidepressant'
        # N03A - Anticonvulsants (many used for anxiety/mood)
        elif code_str.startswith('N03A'):
            classifications[code] = 'anticonvulsant_anxiolytic'
        # N05A - Antipsychotics
        elif code_str.startswith('N05A'):
            classifications[code] = 'antipsychotic'
        # N05B - Anxiolytics
        elif code_str.startswith('N05B'):
            classifications[code] = 'anxiolytic'
        # N05C - Hypnotics and sedatives
        elif code_str.startswith('N05C'):
            classifications[code] = 'hypnotic'
        else:
            classifications[code] = 'not_psychotropic'
    
    return classifications


def calculate_drug_persistence_180(prescriptions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 180-day drug persistence for enhanced analysis
    
    Parameters:
    -----------
    prescriptions_df : pd.DataFrame
        Prescription data with patient_id, drug_code, date_prescribed, days_supply
        
    Returns:
    --------
    pd.DataFrame
        Patient-level persistence flags
    """
    logger.info("Calculating 180-day drug persistence...")
    
    # Classify psychotropic drugs
    unique_drugs = prescriptions_df['drug_code'].unique()
    drug_classifications = classify_psychotropic_drugs(unique_drugs)
    
    # Add classification to prescriptions
    prescriptions_df['drug_class'] = prescriptions_df['drug_code'].map(drug_classifications)
    
    # Filter to psychotropic drugs only
    psychotropic_rx = prescriptions_df[
        prescriptions_df['drug_class'] != 'not_psychotropic'
    ].copy()
    
    # Sort by patient and date
    psychotropic_rx = psychotropic_rx.sort_values(['patient_id', 'date_prescribed'])
    
    # Calculate persistence by patient
    persistence_results = []
    
    for patient_id, patient_rx in psychotropic_rx.groupby('patient_id'):
        patient_rx = patient_rx.reset_index(drop=True)
        
        if len(patient_rx) < 2:
            # Need at least 2 prescriptions for persistence
            persistence_results.append({
                'patient_id': patient_id,
                'persistent_180': False,
                'total_rx_days': patient_rx['days_supply'].sum(),
                'prescription_count': len(patient_rx)
            })
            continue
        
        # Calculate gaps between prescriptions
        patient_rx['next_rx_date'] = patient_rx['date_prescribed'].shift(-1)
        patient_rx['rx_end_date'] = (
            patient_rx['date_prescribed'] + 
            pd.to_timedelta(patient_rx['days_supply'], unit='days')
        )
        
        # Check for gaps > 30 days
        patient_rx['gap_days'] = (
            patient_rx['next_rx_date'] - patient_rx['rx_end_date']
        ).dt.days
        
        # Calculate total covered days
        first_rx = patient_rx['date_prescribed'].min()
        last_rx_end = (
            patient_rx['date_prescribed'].max() + 
            pd.to_timedelta(patient_rx['days_supply'].iloc[-1], unit='days')
        )
        
        total_period = (last_rx_end - first_rx).days
        max_gap = patient_rx['gap_days'].max() if not patient_rx['gap_days'].isna().all() else 0
        
        # 180-day persistence: coverage >= 180 days AND max gap <= 30 days
        persistent_180 = (total_period >= 180) and (max_gap <= 30)
        
        persistence_results.append({
            'patient_id': patient_id,
            'persistent_180': persistent_180,
            'total_period_days': total_period,
            'max_gap_days': max_gap,
            'prescription_count': len(patient_rx),
            'total_rx_days': patient_rx['days_supply'].sum()
        })
    
    persistence_df = pd.DataFrame(persistence_results)
    
    logger.info(f"180-day persistence: {persistence_df['persistent_180'].sum():,} of "
               f"{len(persistence_df):,} patients "
               f"({persistence_df['persistent_180'].mean()*100:.1f}%)")
    
    return persistence_df


def identify_psychiatric_referrals(referrals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify psychiatric/mental health referrals
    
    Parameters:
    -----------
    referrals_df : pd.DataFrame
        Referral data with specialty information
        
    Returns:
    --------
    pd.DataFrame
        Psychiatric referrals only
    """
    psychiatric_specialties = [
        'psychiatry', 'psychology', 'mental health', 
        'psychiatric', 'behavioral health', 'addiction',
        'substance abuse', 'psychologist', 'psychiatrist'
    ]
    
    # Create case-insensitive pattern
    pattern = '|'.join(psychiatric_specialties)
    
    # Filter psychiatric referrals
    psych_mask = referrals_df['referral_specialty'].str.contains(
        pattern, case=False, na=False
    )
    
    psychiatric_referrals = referrals_df[psych_mask].copy()
    psychiatric_referrals['is_psychiatric_referral'] = True
    
    logger.info(f"Psychiatric referrals: {len(psychiatric_referrals):,} of "
               f"{len(referrals_df):,} ({len(psychiatric_referrals)/len(referrals_df)*100:.1f}%)")
    
    return psychiatric_referrals


def validate_mh_cohort_size(cohort_df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate mental health cohort meets expected size thresholds
    
    Parameters:
    -----------
    cohort_df : pd.DataFrame
        Cohort dataframe with mental health flags
        
    Returns:
    --------
    Dict
        Validation results and statistics
    """
    total_patients = len(cohort_df)
    mh_patients = cohort_df['is_mental_health'].sum() if 'is_mental_health' in cohort_df.columns else 0
    mh_percentage = (mh_patients / total_patients * 100) if total_patients > 0 else 0
    
    # Expected thresholds from blueprint
    MIN_TOTAL_PATIENTS = 200000  # Expecting ~256k
    MIN_MH_PERCENTAGE = 60       # Expecting high MH prevalence
    
    validation = {
        'total_patients': total_patients,
        'mh_patients': mh_patients,
        'mh_percentage': mh_percentage,
        'meets_minimum': (
            total_patients >= MIN_TOTAL_PATIENTS and 
            mh_percentage >= MIN_MH_PERCENTAGE
        ),
        'expected_total': 256746,  # From blueprint
        'min_thresholds': {
            'total': MIN_TOTAL_PATIENTS,
            'mh_percentage': MIN_MH_PERCENTAGE
        }
    }
    
    logger.info(f"Cohort validation: {total_patients:,} total, {mh_patients:,} MH "
               f"({mh_percentage:.1f}%), meets minimum: {validation['meets_minimum']}")
    
    return validation


def enhance_existing_cohort(existing_cohort: pd.DataFrame,
                           diagnosis_data: pd.DataFrame,
                           output_dir: Path) -> pd.DataFrame:
    """
    Enhance existing cohort with mental health-specific flags
    
    Parameters:
    -----------
    existing_cohort : pd.DataFrame
        Current cohort data
    diagnosis_data : pd.DataFrame
        Diagnosis data for filtering
    output_dir : Path
        Output directory for enhanced cohort
        
    Returns:
    --------
    pd.DataFrame
        Enhanced cohort with mental health flags
    """
    logger.info("Enhancing existing cohort with mental health specificity...")
    
    # Merge with diagnosis data
    enhanced = existing_cohort.merge(
        diagnosis_data[['patient_id', 'diagnosis_codes']], 
        on='patient_id', 
        how='left'
    )
    
    # Add mental health flags
    enhanced['is_mental_health'] = enhanced['diagnosis_codes'].apply(
        is_mental_health_diagnosis
    )
    enhanced['mh_diagnosis_category'] = enhanced['diagnosis_codes'].apply(
        categorize_mental_health_diagnosis
    )
    
    # Filter to mental health patients only (as per blueprint requirement)
    mh_cohort = enhanced[enhanced['is_mental_health']].copy()
    
    # Validate cohort size
    validation = validate_mh_cohort_size(mh_cohort)
    
    # Save enhanced cohort
    output_path = output_dir / 'cohort_mh_enhanced.parquet'
    mh_cohort.to_parquet(output_path, index=False)
    
    logger.info(f"Enhanced mental health cohort saved: {output_path}")
    logger.info(f"Mental health cohort size: {len(mh_cohort):,}")
    
    return mh_cohort


def main():
    """Main execution for mental health cohort enhancement"""
    logger.info("Starting mental health cohort enhancement...")
    
    # This would integrate with existing pipeline
    print("Mental health cohort builder ready for integration")
    print("Key functions available:")
    print("  - filter_mental_health_patients()")
    print("  - calculate_drug_persistence_180()")
    print("  - identify_psychiatric_referrals()")
    print("  - enhance_existing_cohort()")


if __name__ == "__main__":
    main()