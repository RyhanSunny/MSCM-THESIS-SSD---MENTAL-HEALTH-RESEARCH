#!/usr/bin/env python3
"""
mh_exposure_enhanced.py - Enhanced exposure definitions for mental health population

Implements 180-day drug persistence, enhanced drug classes (N06A, N03A, N05A),
and psychiatric referral logic for Week 4 domain alignment.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_enhanced_drug_mapping() -> Dict[str, str]:
    """
    Create comprehensive ATC drug code mapping including enhanced classes
    
    Returns:
    --------
    Dict[str, str]
        Mapping of ATC codes to drug categories
    """
    return {
        # N06A - Antidepressants (Dr. Felipe priority)
        'N06A': 'antidepressants',
        'N06AB': 'ssri_antidepressants',
        'N06AC': 'tricyclic_antidepressants', 
        'N06AF': 'maoi_antidepressants',
        'N06AX': 'other_antidepressants',
        
        # N03A - Anticonvulsants (used for mood/anxiety)
        'N03A': 'anticonvulsants',
        'N03AE': 'benzodiazepine_anticonvulsants',
        'N03AF': 'carboxamide_anticonvulsants',
        'N03AG': 'fatty_acid_anticonvulsants',
        
        # N05A - Antipsychotics (Dr. Felipe priority)
        'N05A': 'antipsychotics',
        'N05AA': 'phenothiazine_antipsychotics',
        'N05AB': 'phenothiazine_piperazine',
        'N05AC': 'phenothiazine_piperidine',
        'N05AD': 'butyrophenone_antipsychotics',
        'N05AE': 'indole_antipsychotics',
        'N05AF': 'thioxanthene_antipsychotics',
        'N05AG': 'diphenylbutylpiperidine',
        'N05AH': 'diazepines_oxazepines',
        'N05AL': 'benzisoxazole_antipsychotics',
        'N05AN': 'lithium_compounds',
        'N05AX': 'other_antipsychotics',
        
        # Existing codes (maintain compatibility)
        'N05B': 'anxiolytics',
        'N05C': 'hypnotics',
        'N02B': 'analgesics'
    }


def calculate_enhanced_drug_persistence(prescriptions_df: pd.DataFrame,
                                       duration_threshold: int = 180) -> pd.DataFrame:
    """
    Calculate drug persistence with enhanced 180-day threshold
    
    Parameters:
    -----------
    prescriptions_df : pd.DataFrame
        Prescription data with patient_id, drug_code, date_prescribed, days_supply
    duration_threshold : int
        Duration threshold in days (default 180)
        
    Returns:
    --------
    pd.DataFrame
        Patient-level persistence results
    """
    logger.info(f"Calculating {duration_threshold}-day drug persistence...")
    
    # Get enhanced drug mapping
    drug_mapping = create_enhanced_drug_mapping()
    
    # Classify prescriptions
    def classify_drug(code):
        if pd.isna(code):
            return 'unknown'
        
        code_str = str(code).upper()
        for atc_prefix, category in drug_mapping.items():
            if code_str.startswith(atc_prefix):
                return category
        return 'not_psychotropic'
    
    prescriptions_df['drug_category'] = prescriptions_df['drug_code'].apply(classify_drug)
    
    # Filter to psychotropic medications only
    psychotropic_rx = prescriptions_df[
        prescriptions_df['drug_category'] != 'not_psychotropic'
    ].copy()
    
    logger.info(f"Psychotropic prescriptions: {len(psychotropic_rx):,} of "
               f"{len(prescriptions_df):,} total")
    
    # Calculate persistence by patient
    persistence_results = []
    
    for patient_id, patient_rx in psychotropic_rx.groupby('patient_id'):
        patient_rx = patient_rx.sort_values('date_prescribed').reset_index(drop=True)
        
        if len(patient_rx) == 0:
            continue
            
        # Calculate coverage periods
        patient_rx['rx_end_date'] = (
            patient_rx['date_prescribed'] + 
            pd.to_timedelta(patient_rx['days_supply'], unit='days')
        )
        
        # Find continuous coverage periods
        first_date = patient_rx['date_prescribed'].min()
        last_end_date = patient_rx['rx_end_date'].max()
        total_span = (last_end_date - first_date).days
        
        # Check for gaps > 30 days
        if len(patient_rx) > 1:
            patient_rx['gap_to_next'] = (
                patient_rx['date_prescribed'].shift(-1) - patient_rx['rx_end_date']
            ).dt.days
            max_gap = patient_rx['gap_to_next'].max()
        else:
            max_gap = 0
        
        # Persistence criteria:
        # 1. Total span >= threshold days
        # 2. No gaps > 30 days
        # 3. At least 2 prescriptions for true persistence
        persistent = (
            total_span >= duration_threshold and 
            (pd.isna(max_gap) or max_gap <= 30) and
            len(patient_rx) >= 2
        )
        
        # Enhanced drug class analysis
        drug_classes = patient_rx['drug_category'].unique()
        primary_class = patient_rx['drug_category'].mode().iloc[0] if len(patient_rx) > 0 else 'unknown'
        
        persistence_results.append({
            'patient_id': patient_id,
            f'persistent_{duration_threshold}': persistent,
            'total_span_days': total_span,
            'max_gap_days': max_gap if not pd.isna(max_gap) else 0,
            'prescription_count': len(patient_rx),
            'drug_classes_count': len(drug_classes),
            'primary_drug_class': primary_class,
            'total_supply_days': patient_rx['days_supply'].sum()
        })
    
    results_df = pd.DataFrame(persistence_results)
    
    persistent_count = results_df[f'persistent_{duration_threshold}'].sum()
    logger.info(f"{duration_threshold}-day persistence: {persistent_count:,} of "
               f"{len(results_df):,} patients "
               f"({persistent_count/len(results_df)*100:.1f}%)")
    
    return results_df


def identify_enhanced_referral_patterns(referrals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify psychiatric referral patterns with enhanced logic
    
    Parameters:
    -----------
    referrals_df : pd.DataFrame
        Referral data with specialty and status information
        
    Returns:
    --------
    pd.DataFrame
        Enhanced referral pattern analysis
    """
    logger.info("Identifying enhanced psychiatric referral patterns...")
    
    # Enhanced psychiatric specialties
    psychiatric_specialties = {
        'psychiatry': 'psychiatry',
        'psychology': 'psychology', 
        'mental health': 'mental_health',
        'behavioral health': 'behavioral_health',
        'addiction': 'addiction_psychiatry',
        'substance abuse': 'substance_abuse',
        'psychotherapy': 'psychotherapy',
        'psychiatric': 'psychiatry',
        'psychologist': 'psychology',
        'psychiatrist': 'psychiatry'
    }
    
    # Classify referrals
    def classify_referral(specialty):
        if pd.isna(specialty):
            return 'not_psychiatric'
        
        specialty_lower = str(specialty).lower()
        for keyword, category in psychiatric_specialties.items():
            if keyword in specialty_lower:
                return category
        return 'not_psychiatric'
    
    referrals_df['psychiatric_specialty'] = referrals_df['referral_specialty'].apply(
        classify_referral
    )
    
    # Filter psychiatric referrals
    psychiatric_referrals = referrals_df[
        referrals_df['psychiatric_specialty'] != 'not_psychiatric'
    ].copy()
    
    # Analyze referral patterns by patient
    patient_patterns = []
    
    for patient_id, patient_refs in psychiatric_referrals.groupby('patient_id'):
        # Count referral types
        specialty_counts = patient_refs['psychiatric_specialty'].value_counts()
        
        # Identify unresolved referrals (NYD equivalent)
        unresolved_refs = patient_refs[
            patient_refs['referral_status'].isin(['pending', 'incomplete', 'cancelled'])
        ]
        
        patient_patterns.append({
            'patient_id': patient_id,
            'total_psychiatric_referrals': len(patient_refs),
            'unresolved_psychiatric_referrals': len(unresolved_refs),
            'has_psychiatry_referral': 'psychiatry' in specialty_counts.index,
            'has_psychology_referral': 'psychology' in specialty_counts.index,
            'multiple_psychiatric_specialties': len(specialty_counts) > 1,
            'referral_loop_pattern': len(unresolved_refs) >= 2,  # H2 criterion
            'primary_psychiatric_specialty': specialty_counts.index[0] if len(specialty_counts) > 0 else None
        })
    
    patterns_df = pd.DataFrame(patient_patterns)
    
    loop_patients = patterns_df['referral_loop_pattern'].sum()
    logger.info(f"Psychiatric referral loop patterns: {loop_patients:,} patients")
    
    return patterns_df


def create_enhanced_exposure_flags(cohort_df: pd.DataFrame,
                                  prescriptions_df: pd.DataFrame,
                                  referrals_df: pd.DataFrame,
                                  lab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced exposure flags for H1-H3 with 180-day thresholds
    
    Parameters:
    -----------
    cohort_df : pd.DataFrame
        Mental health cohort
    prescriptions_df : pd.DataFrame
        Prescription data
    referrals_df : pd.DataFrame
        Referral data
    lab_df : pd.DataFrame
        Laboratory data
        
    Returns:
    --------
    pd.DataFrame
        Cohort with enhanced exposure flags
    """
    logger.info("Creating enhanced exposure flags...")
    
    enhanced_cohort = cohort_df.copy()
    
    # H1: Normal lab cascade (unchanged logic)
    # This would integrate with existing lab analysis
    # For now, placeholder
    enhanced_cohort['h1_normal_lab_cascade'] = False
    
    # H2: Enhanced psychiatric referral patterns
    if not referrals_df.empty:
        referral_patterns = identify_enhanced_referral_patterns(referrals_df)
        enhanced_cohort = enhanced_cohort.merge(
            referral_patterns[['patient_id', 'referral_loop_pattern']],
            on='patient_id',
            how='left'
        )
        enhanced_cohort['h2_psychiatric_referral_loop'] = enhanced_cohort['referral_loop_pattern'].fillna(False)
    else:
        enhanced_cohort['h2_psychiatric_referral_loop'] = False
    
    # H3: Enhanced 180-day drug persistence
    if not prescriptions_df.empty:
        drug_persistence = calculate_enhanced_drug_persistence(prescriptions_df, 180)
        enhanced_cohort = enhanced_cohort.merge(
            drug_persistence[['patient_id', 'persistent_180', 'primary_drug_class']],
            on='patient_id',
            how='left'
        )
        enhanced_cohort['h3_drug_persistence_180'] = enhanced_cohort['persistent_180'].fillna(False)
        enhanced_cohort['h3_primary_drug_class'] = enhanced_cohort['primary_drug_class'].fillna('none')
    else:
        enhanced_cohort['h3_drug_persistence_180'] = False
        enhanced_cohort['h3_primary_drug_class'] = 'none'
    
    # Summary exposure flag (OR logic as per blueprint decision)
    enhanced_cohort['ssd_exposure_enhanced'] = (
        enhanced_cohort['h1_normal_lab_cascade'] |
        enhanced_cohort['h2_psychiatric_referral_loop'] |
        enhanced_cohort['h3_drug_persistence_180']
    )
    
    # Log exposure statistics
    h1_count = enhanced_cohort['h1_normal_lab_cascade'].sum()
    h2_count = enhanced_cohort['h2_psychiatric_referral_loop'].sum()
    h3_count = enhanced_cohort['h3_drug_persistence_180'].sum()
    total_exposed = enhanced_cohort['ssd_exposure_enhanced'].sum()
    
    logger.info(f"Enhanced exposure flags:")
    logger.info(f"  H1 (normal labs): {h1_count:,} ({h1_count/len(enhanced_cohort)*100:.1f}%)")
    logger.info(f"  H2 (psych referrals): {h2_count:,} ({h2_count/len(enhanced_cohort)*100:.1f}%)")
    logger.info(f"  H3 (drug persistence 180d): {h3_count:,} ({h3_count/len(enhanced_cohort)*100:.1f}%)")
    logger.info(f"  Total SSD exposure: {total_exposed:,} ({total_exposed/len(enhanced_cohort)*100:.1f}%)")
    
    return enhanced_cohort


def main():
    """Main execution for enhanced exposure definitions"""
    logger.info("Enhanced MH exposure definitions ready")
    
    print("Enhanced exposure functions available:")
    print("  - calculate_enhanced_drug_persistence() - 180-day threshold")
    print("  - identify_enhanced_referral_patterns() - psychiatric specialties")
    print("  - create_enhanced_exposure_flags() - integrated H1-H3")
    print("  - Enhanced drug classes: N06A, N03A, N05A")


if __name__ == "__main__":
    main()