#!/usr/bin/env python3
"""
Clinical Data Validation & Sanity Check - Simplified Version
Comprehensive review of all processed data for clinical plausibility
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("=== CLINICAL DATA VALIDATION & SANITY CHECK ===\n")

# 1. Load processed datasets
print("1. Loading processed datasets...")
try:
    # Load processed data with relative paths
    cohort = pd.read_parquet("data_derived/cohort.parquet")
    exposure = pd.read_parquet("data_derived/exposure.parquet")
    severity = pd.read_parquet("data_derived/mediator_autoencoder.parquet")
    outcomes = pd.read_parquet("data_derived/outcomes.parquet")
    print(f"✓ Loaded processed: cohort ({len(cohort):,}), exposure ({len(exposure):,}), severity ({len(severity):,}), outcomes ({len(outcomes):,})")
    
except Exception as e:
    print(f"✗ Error loading processed data: {e}")
    print("Attempting to load raw data for validation...")
    try:
        # Load raw data instead
        patient = pd.read_parquet("Notebooks/data/interim/checkpoint_1_20250318_024427/patient.parquet")
        patient_demo = pd.read_parquet("Notebooks/data/interim/checkpoint_1_20250318_024427/patient_demographic.parquet")
        encounter = pd.read_parquet("Notebooks/data/interim/checkpoint_1_20250318_024427/encounter.parquet")
        encounter_dx = pd.read_parquet("Notebooks/data/interim/checkpoint_1_20250318_024427/encounter_diagnosis.parquet")
        referral = pd.read_parquet("Notebooks/data/interim/checkpoint_1_20250318_024427/referral.parquet")
        lab = pd.read_csv("Notebooks/data/interim/checkpoint_1_20250318_024427/lab.csv")
        medication = pd.read_parquet("Notebooks/data/interim/checkpoint_1_20250318_024427/medication.parquet")
        
        print(f"✓ Loaded raw data: patient ({len(patient):,}), encounters ({len(encounter):,}), diagnoses ({len(encounter_dx):,})")
        print(f"✓ Additional: referrals ({len(referral):,}), labs ({len(lab):,}), medications ({len(medication):,})")
        
        # Perform raw data validation
        print("\n=== RAW DATA VALIDATION ===")
        
        # Patient demographics validation
        print("\n2. PATIENT DEMOGRAPHICS")
        print("-" * 40)
        
        # Merge patient and demographics
        patients_merged = patient.merge(patient_demo, on="Patient_ID", how="inner")
        print(f"Patients with demographics: {len(patients_merged):,}")
        
        # Age validation (assuming BirthYear column)
        if 'BirthYear' in patients_merged.columns:
            current_year = 2022  # Study end year
            patients_merged['age'] = current_year - patients_merged['BirthYear']
            age_stats = patients_merged['age'].describe()
            print(f"Age distribution:")
            print(f"  Mean: {age_stats['mean']:.1f} years")
            print(f"  Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")
            
            # Flag unusual ages
            unusual_ages = patients_merged[(patients_merged['age'] < 0) | (patients_merged['age'] > 120)]
            if len(unusual_ages) > 0:
                print(f"⚠️  {len(unusual_ages)} patients with impossible ages")
        
        # Sex distribution
        if 'Sex' in patients_merged.columns:
            sex_dist = patients_merged['Sex'].value_counts(normalize=True) * 100
            print(f"\nSex distribution:")
            for sex, pct in sex_dist.items():
                print(f"  {sex}: {pct:.1f}%")
        
        # 3. Encounter validation
        print("\n3. ENCOUNTER VALIDATION")
        print("-" * 40)
        
        print(f"Total encounters: {len(encounter):,}")
        
        # Date validation
        if 'EncounterDateTime' in encounter.columns:
            encounter['EncounterDate'] = pd.to_datetime(encounter['EncounterDateTime'])
            date_range = encounter['EncounterDate'].dropna()
            if len(date_range) > 0:
                print(f"Encounter date range: {date_range.min().date()} to {date_range.max().date()}")
                
                # Check for future dates
                today = pd.Timestamp.now()
                future_encounters = (encounter['EncounterDate'] > today).sum()
                if future_encounters > 0:
                    print(f"⚠️  {future_encounters} encounters with future dates")
        
        # Encounters per patient
        enc_per_patient = encounter.groupby('Patient_ID').size()
        print(f"Encounters per patient:")
        print(f"  Mean: {enc_per_patient.mean():.1f}")
        print(f"  Median: {enc_per_patient.median():.0f}")
        print(f"  Max: {enc_per_patient.max():.0f}")
        
        # 4. Diagnosis validation
        print("\n4. DIAGNOSIS VALIDATION")
        print("-" * 40)
        
        print(f"Total diagnosis records: {len(encounter_dx):,}")
        
        # Diagnoses per encounter
        dx_per_encounter = encounter_dx.groupby('Encounter_ID').size()
        print(f"Diagnoses per encounter:")
        print(f"  Mean: {dx_per_encounter.mean():.1f}")
        print(f"  Max: {dx_per_encounter.max():.0f}")
        
        # Check for ICD code formats
        if 'DiagnosisText_calc' in encounter_dx.columns:
            icd_codes = encounter_dx['DiagnosisText_calc'].dropna()
            print(f"ICD codes available: {len(icd_codes):,}")
            
            # Sample of most common codes
            top_codes = icd_codes.value_counts().head(10)
            print("\nTop 10 diagnosis codes:")
            for code, count in top_codes.items():
                print(f"  {code}: {count:,}")
        
        # 5. Referral validation
        print("\n5. REFERRAL VALIDATION")
        print("-" * 40)
        
        print(f"Total referrals: {len(referral):,}")
        
        # Referrals per patient
        ref_per_patient = referral.groupby('Patient_ID').size()
        print(f"Referrals per patient:")
        print(f"  Mean: {ref_per_patient.mean():.2f}")
        print(f"  Max: {ref_per_patient.max():.0f}")
        
        # Check for suspicious high referral counts (data quality issue)
        high_referrals = ref_per_patient[ref_per_patient > 1000]
        if len(high_referrals) > 0:
            print(f"⚠️  {len(high_referrals)} patients with >1000 referrals (data quality issue)")
            print(f"    Max referrals: {ref_per_patient.max():,}")
        
        # Referral specialties
        if 'ReferralTo_calc' in referral.columns:
            specialties = referral['ReferralTo_calc'].value_counts().head(10)
            print("\nTop referral specialties:")
            for spec, count in specialties.items():
                print(f"  {spec}: {count:,}")
        
        # 6. Lab validation
        print("\n6. LAB VALIDATION")
        print("-" * 40)
        
        print(f"Total lab records: {len(lab):,}")
        
        # Labs per patient
        if 'Patient_ID' in lab.columns:
            lab_per_patient = lab.groupby('Patient_ID').size()
            print(f"Labs per patient:")
            print(f"  Mean: {lab_per_patient.mean():.1f}")
            print(f"  Max: {lab_per_patient.max():.0f}")
        
        # Lab test types
        if 'TestName_calc' in lab.columns:
            test_types = lab['TestName_calc'].value_counts().head(10)
            print("\nTop lab tests:")
            for test, count in test_types.items():
                print(f"  {test}: {count:,}")
        
        # Normal range availability
        if 'NormalRange_calc' in lab.columns:
            normal_range_available = lab['NormalRange_calc'].notna().sum()
            print(f"\nLab normal ranges available: {normal_range_available:,} ({normal_range_available/len(lab)*100:.1f}%)")
        
        # 7. Medication validation
        print("\n7. MEDICATION VALIDATION")
        print("-" * 40)
        
        print(f"Total medication records: {len(medication):,}")
        
        # Medications per patient
        med_per_patient = medication.groupby('Patient_ID').size()
        print(f"Medications per patient:")
        print(f"  Mean: {med_per_patient.mean():.1f}")
        print(f"  Max: {med_per_patient.max():.0f}")
        
        # Drug categories
        if 'DrugName_calc' in medication.columns:
            drugs = medication['DrugName_calc'].value_counts().head(10)
            print("\nTop medications:")
            for drug, count in drugs.items():
                print(f"  {drug}: {count:,}")
        
        print("\n=== RAW DATA VALIDATION COMPLETE ===")
        print("Processed data files need to be generated for full validation")
        exit(0)
        
    except Exception as e2:
        print(f"✗ Error loading raw data: {e2}")
        exit(1)

# If we reach here, processed data was loaded successfully
print("\n=== PROCESSED DATA VALIDATION ===")

# 2. Basic demographic validation
print("\n2. DEMOGRAPHIC VALIDATION")
print("-" * 40)

# Age distribution
if 'age_baseline' in cohort.columns:
    age_stats = cohort['age_baseline'].describe()
    print(f"Age distribution:")
    print(f"  Mean: {age_stats['mean']:.1f} years")
    print(f"  Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")
    print(f"  IQR: {age_stats['25%']:.0f} - {age_stats['75%']:.0f} years")
    
    # Flag unusual ages
    unusual_ages = cohort[(cohort['age_baseline'] < 18) | (cohort['age_baseline'] > 100)]
    if len(unusual_ages) > 0:
        print(f"⚠️  {len(unusual_ages)} patients with unusual ages (<18 or >100)")

# Sex distribution
if 'sex' in cohort.columns:
    sex_dist = cohort['sex'].value_counts(normalize=True) * 100
    print(f"\nSex distribution:")
    for sex, pct in sex_dist.items():
        print(f"  {sex}: {pct:.1f}%")

# 3. Exposure validation
print("\n3. EXPOSURE VALIDATION")
print("-" * 40)

# Exposure prevalence by hypothesis
for h in ['H1_normal_labs', 'H2_referral_loop', 'H3_drug_persistence']:
    if h in exposure.columns:
        prev = exposure[h].mean() * 100
        print(f"{h}: {prev:.1f}% exposed")

# Combined exposure
if 'exposure_flag' in exposure.columns:
    total_exposed = exposure['exposure_flag'].sum()
    exposure_rate = total_exposed / len(exposure) * 100
    print(f"\nTotal exposed (OR logic): {total_exposed:,} ({exposure_rate:.1f}%)")
    
    # Clinical plausibility check
    if exposure_rate < 10:
        print("⚠️  Low exposure rate may indicate restrictive criteria")
    elif exposure_rate > 80:
        print("⚠️  High exposure rate may indicate permissive criteria")

# 4. Severity index validation
print("\n4. SEVERITY INDEX VALIDATION")
print("-" * 40)

if 'severity_index' in severity.columns:
    severity_stats = severity['severity_index'].describe()
    print(f"Severity Index (0-100 scale):")
    print(f"  Mean: {severity_stats['mean']:.2f}")
    print(f"  Range: {severity_stats['min']:.2f} - {severity_stats['max']:.2f}")
    print(f"  Std: {severity_stats['std']:.2f}")
    
    # Check for proper scaling
    if severity_stats['min'] < 0 or severity_stats['max'] > 100:
        print("⚠️  Severity index outside expected 0-100 range")

# 5. Outcomes validation
print("\n5. OUTCOMES VALIDATION")
print("-" * 40)

# Cost outcomes
cost_cols = [col for col in outcomes.columns if 'cost' in col.lower()]
for col in cost_cols:
    if col in outcomes.columns:
        cost_stats = outcomes[col].describe()
        print(f"\n{col}:")
        print(f"  Mean: ${cost_stats['mean']:,.2f}")
        print(f"  Median: ${cost_stats['50%']:,.2f}")
        print(f"  Max: ${cost_stats['max']:,.2f}")
        
        # Flag unrealistic costs
        if cost_stats['max'] > 100000:  # >$100k
            high_cost = (outcomes[col] > 100000).sum()
            print(f"⚠️  {high_cost} patients with costs >$100k")

print("\n" + "=" * 50)
print("CLINICAL VALIDATION COMPLETE")
print("=" * 50)