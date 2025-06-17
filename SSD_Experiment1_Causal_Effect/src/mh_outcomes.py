#!/usr/bin/env python3
"""
mh_outcomes.py - Mental Health-specific outcome identification

Implements MH service encounter counting and psychiatric ED visit identification
as required by H1-H3 hypotheses in the SSD THESIS blueprint.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_provider_specialty(specialty: str) -> bool:
    """
    Classify if provider specialty is mental health-related
    
    Parameters:
    -----------
    specialty : str
        Provider specialty description
        
    Returns:
    --------
    bool
        True if mental health specialty
    """
    if pd.isna(specialty):
        return False
    
    specialty_lower = str(specialty).lower()
    
    mh_specialties = [
        'psychiatry', 'psychology', 'mental health', 'behavioral health',
        'psychiatric', 'psychologist', 'psychiatrist', 'addiction',
        'substance abuse', 'psychotherapy', 'counseling'
    ]
    
    return any(mh_spec in specialty_lower for mh_spec in mh_specialties)


def has_mh_diagnosis(diagnosis_codes: str) -> bool:
    """
    Check if encounter has mental health diagnosis codes
    
    Parameters:
    -----------
    diagnosis_codes : str
        Comma or semicolon-separated diagnosis codes
        
    Returns:
    --------
    bool
        True if any diagnosis is mental health-related
    """
    if pd.isna(diagnosis_codes) or not diagnosis_codes:
        return False
    
    # Split on common separators
    codes = re.split('[,;|\\s]+', str(diagnosis_codes))
    
    for code in codes:
        code = code.strip().upper()
        if not code:
            continue
            
        # ICD-10 Mental Health codes (F32-F48)
        if re.match(r'^F(3[2-9]|4[0-8])', code):
            return True
            
        # ICD-9 Mental Health codes (296.*, 300.*)
        if re.match(r'^296\.', code) or re.match(r'^300\.', code):
            return True
    
    return False


def has_psychiatric_discharge(disposition: str) -> bool:
    """
    Check if discharge disposition indicates psychiatric care
    
    Parameters:
    -----------
    disposition : str
        Discharge disposition description
        
    Returns:
    --------
    bool
        True if psychiatric discharge
    """
    if pd.isna(disposition):
        return False
    
    disposition_lower = str(disposition).lower()
    
    psychiatric_dispositions = [
        'psychiatric', 'mental health', 'behavioral health',
        'psych unit', 'psych hospital', 'psychiatric hospital',
        'mental health facility', 'crisis center', 'involuntary hold'
    ]
    
    return any(psych_disp in disposition_lower for psych_disp in psychiatric_dispositions)


def identify_mh_encounters(encounters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify mental health service encounters
    
    Parameters:
    -----------
    encounters_df : pd.DataFrame
        Encounter data with provider specialty and diagnosis codes
        
    Returns:
    --------
    pd.DataFrame
        Mental health encounters only
    """
    logger.info("Identifying mental health service encounters...")
    
    encounters = encounters_df.copy()
    
    # Create MH identification flags
    encounters['mh_by_specialty'] = encounters['provider_specialty'].apply(
        classify_provider_specialty
    )
    encounters['mh_by_diagnosis'] = encounters['diagnosis_codes'].apply(
        has_mh_diagnosis
    )
    
    # Combined MH encounter flag (either specialty OR diagnosis)
    encounters['is_mh_encounter'] = (
        encounters['mh_by_specialty'] | encounters['mh_by_diagnosis']
    )
    
    # Filter to MH encounters
    mh_encounters = encounters[encounters['is_mh_encounter']].copy()
    
    logger.info(f"Mental health encounters: {len(mh_encounters):,} of "
               f"{len(encounters):,} total ({len(mh_encounters)/len(encounters)*100:.1f}%)")
    
    # Add MH encounter categorization
    def categorize_mh_encounter(row):
        if row['mh_by_specialty'] and row['mh_by_diagnosis']:
            return 'specialty_and_diagnosis'
        elif row['mh_by_specialty']:
            return 'specialty_based'
        elif row['mh_by_diagnosis']:
            return 'diagnosis_based'
        else:
            return 'not_mh'
    
    mh_encounters['mh_encounter_type'] = mh_encounters.apply(
        categorize_mh_encounter, axis=1
    )
    
    return mh_encounters


def meets_psychiatric_ed_criteria(encounter_type: str,
                                 diagnosis_codes: str,
                                 discharge_disposition: str = None,
                                 provider_specialty: str = None) -> bool:
    """
    Check if encounter meets psychiatric ED visit criteria
    
    Parameters:
    -----------
    encounter_type : str
        Type of encounter (e.g., 'Emergency', 'Inpatient')
    diagnosis_codes : str
        Diagnosis codes for encounter
    discharge_disposition : str, optional
        Discharge disposition
    provider_specialty : str, optional
        Provider specialty
        
    Returns:
    --------
    bool
        True if meets psychiatric ED criteria
    """
    # Must be emergency encounter
    if pd.isna(encounter_type) or 'emergency' not in str(encounter_type).lower():
        return False
    
    # Check for psychiatric indicators
    psychiatric_indicators = []
    
    # 1. Mental health diagnosis
    if has_mh_diagnosis(diagnosis_codes):
        psychiatric_indicators.append('mh_diagnosis')
    
    # 2. Psychiatric discharge disposition
    if discharge_disposition and has_psychiatric_discharge(discharge_disposition):
        psychiatric_indicators.append('psychiatric_discharge')
    
    # 3. Mental health provider in ED
    if provider_specialty and classify_provider_specialty(provider_specialty):
        psychiatric_indicators.append('mh_provider')
    
    # Requires at least one psychiatric indicator
    return len(psychiatric_indicators) > 0


def identify_psychiatric_ed_visits(encounters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify psychiatric emergency department visits
    
    Parameters:
    -----------
    encounters_df : pd.DataFrame
        Emergency department encounter data
        
    Returns:
    --------
    pd.DataFrame
        Psychiatric ED visits only
    """
    logger.info("Identifying psychiatric emergency department visits...")
    
    encounters = encounters_df.copy()
    
    # Apply psychiatric ED criteria
    encounters['is_psychiatric_ed'] = encounters.apply(
        lambda row: meets_psychiatric_ed_criteria(
            row.get('encounter_type', ''),
            row.get('diagnosis_codes', ''),
            row.get('discharge_disposition', ''),
            row.get('provider_specialty', '')
        ), axis=1
    )
    
    # Filter to psychiatric ED visits
    psychiatric_ed = encounters[encounters['is_psychiatric_ed']].copy()
    
    logger.info(f"Psychiatric ED visits: {len(psychiatric_ed):,} of "
               f"{len(encounters):,} total ED visits "
               f"({len(psychiatric_ed)/len(encounters)*100:.1f}%)")
    
    return psychiatric_ed


def count_mh_encounters_by_patient(encounters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count mental health encounters per patient
    
    Parameters:
    -----------
    encounters_df : pd.DataFrame
        Encounter data with MH flags
        
    Returns:
    --------
    pd.DataFrame
        Patient-level MH encounter counts
    """
    logger.info("Counting mental health encounters by patient...")
    
    # Ensure MH flag exists
    if 'is_mh_encounter' not in encounters_df.columns:
        encounters_df = identify_mh_encounters(encounters_df)
    
    # Count by patient
    patient_counts = encounters_df.groupby('patient_id').agg({
        'is_mh_encounter': ['sum', 'count'],
        'encounter_id': 'nunique'
    }).reset_index()
    
    # Flatten column names
    patient_counts.columns = [
        'patient_id', 'mh_encounters_count', 'total_encounters_count', 'unique_encounters'
    ]
    
    # Calculate proportions
    patient_counts['mh_encounter_proportion'] = (
        patient_counts['mh_encounters_count'] / 
        patient_counts['total_encounters_count']
    ).fillna(0)
    
    # Add MH utilization flags
    patient_counts['has_mh_encounters'] = patient_counts['mh_encounters_count'] > 0
    patient_counts['high_mh_utilization'] = patient_counts['mh_encounters_count'] >= 5
    
    logger.info(f"Patients with MH encounters: {patient_counts['has_mh_encounters'].sum():,} of "
               f"{len(patient_counts):,} ({patient_counts['has_mh_encounters'].mean()*100:.1f}%)")
    
    return patient_counts


def flag_psychiatric_ed_by_patient(encounters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create patient-level psychiatric ED visit flags
    
    Parameters:
    -----------
    encounters_df : pd.DataFrame
        Encounter data
        
    Returns:
    --------
    pd.DataFrame
        Patient-level psychiatric ED flags
    """
    logger.info("Creating patient-level psychiatric ED flags...")
    
    # Identify psychiatric ED visits
    psychiatric_ed = identify_psychiatric_ed_visits(encounters_df)
    
    # Aggregate by patient
    patient_psych_ed = psychiatric_ed.groupby('patient_id').agg({
        'is_psychiatric_ed': ['sum', 'any'],
        'encounter_date': ['min', 'max', 'count']
    }).reset_index()
    
    # Flatten columns
    patient_psych_ed.columns = [
        'patient_id', 'psychiatric_ed_count', 'psychiatric_ed_visit',
        'first_psychiatric_ed', 'last_psychiatric_ed', 'psychiatric_ed_visits_total'
    ]
    
    # Add all patients (including those without psychiatric ED)
    all_patients = pd.DataFrame({'patient_id': encounters_df['patient_id'].unique()})
    patient_flags = all_patients.merge(patient_psych_ed, on='patient_id', how='left')
    
    # Fill missing values
    patient_flags['psychiatric_ed_visit'] = patient_flags['psychiatric_ed_visit'].fillna(False)
    patient_flags['psychiatric_ed_count'] = patient_flags['psychiatric_ed_count'].fillna(0)
    
    # Add severity flags
    patient_flags['multiple_psychiatric_ed'] = patient_flags['psychiatric_ed_count'] >= 2
    patient_flags['frequent_psychiatric_ed'] = patient_flags['psychiatric_ed_count'] >= 3
    
    logger.info(f"Patients with psychiatric ED: {patient_flags['psychiatric_ed_visit'].sum():,} of "
               f"{len(patient_flags):,} ({patient_flags['psychiatric_ed_visit'].mean()*100:.1f}%)")
    
    return patient_flags


def enhance_outcome_flags(cohort_df: pd.DataFrame,
                         encounters_df: pd.DataFrame,
                         output_dir: Path) -> pd.DataFrame:
    """
    Enhance cohort with mental health-specific outcome flags
    
    Parameters:
    -----------
    cohort_df : pd.DataFrame
        Base cohort data
    encounters_df : pd.DataFrame
        Encounter data
    output_dir : Path
        Output directory for enhanced outcomes
        
    Returns:
    --------
    pd.DataFrame
        Cohort with MH outcome flags
    """
    logger.info("Enhancing cohort with mental health outcome flags...")
    
    enhanced_cohort = cohort_df.copy()
    
    # Generate MH encounter counts
    mh_counts = count_mh_encounters_by_patient(encounters_df)
    enhanced_cohort = enhanced_cohort.merge(
        mh_counts[['patient_id', 'mh_encounters_count', 'has_mh_encounters']],
        on='patient_id',
        how='left'
    )
    
    # Generate psychiatric ED flags
    psych_ed_flags = flag_psychiatric_ed_by_patient(encounters_df)
    enhanced_cohort = enhanced_cohort.merge(
        psych_ed_flags[['patient_id', 'psychiatric_ed_visit', 'psychiatric_ed_count']],
        on='patient_id',
        how='left'
    )
    
    # Fill missing values
    enhanced_cohort['mh_encounters_count'] = enhanced_cohort['mh_encounters_count'].fillna(0)
    enhanced_cohort['has_mh_encounters'] = enhanced_cohort['has_mh_encounters'].fillna(False)
    enhanced_cohort['psychiatric_ed_visit'] = enhanced_cohort['psychiatric_ed_visit'].fillna(False)
    enhanced_cohort['psychiatric_ed_count'] = enhanced_cohort['psychiatric_ed_count'].fillna(0)
    
    # Save enhanced outcomes
    output_path = output_dir / 'cohort_mh_outcomes.parquet'
    enhanced_cohort.to_parquet(output_path, index=False)
    
    logger.info(f"Enhanced MH outcomes saved: {output_path}")
    logger.info(f"Summary - MH encounters: {enhanced_cohort['mh_encounters_count'].sum():,} total, "
               f"Psychiatric ED: {enhanced_cohort['psychiatric_ed_visit'].sum():,} patients")
    
    return enhanced_cohort


def main():
    """Main execution for mental health outcomes"""
    logger.info("Mental health outcomes module ready")
    
    print("Mental health outcome functions available:")
    print("  - identify_mh_encounters() - MH service identification")
    print("  - identify_psychiatric_ed_visits() - Psychiatric ED visits")
    print("  - count_mh_encounters_by_patient() - Patient-level MH counts")
    print("  - enhance_outcome_flags() - Cohort enhancement")


if __name__ == "__main__":
    main()