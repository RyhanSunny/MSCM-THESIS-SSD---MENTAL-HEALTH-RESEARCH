#!/usr/bin/env python3
"""
Clinical Data Validation & Sanity Check
Comprehensive review of all processed data for clinical plausibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths - using string paths for Windows compatibility
base_path = "/mnt/c/Users/ProjectC4M/Documents/MSCM THESIS SSD/MSCM-THESIS-SSD---MENTAL-HEALTH-RESEARCH/SSD_Experiment1_Causal_Effect"
data_path = f"{base_path}/Notebooks/data/interim/checkpoint_1_20250318_024427"
derived_path = f"{base_path}/data_derived"

print("=== CLINICAL DATA VALIDATION & SANITY CHECK ===\n")

# 1. Load all processed datasets
print("1. Loading processed datasets...")
try:
    # Load processed data
    cohort = pd.read_parquet(f"{derived_path}/cohort.parquet")
    exposure = pd.read_parquet(f"{derived_path}/exposure.parquet")
    severity = pd.read_parquet(f"{derived_path}/mediator_autoencoder.parquet")
    outcomes = pd.read_parquet(f"{derived_path}/outcomes.parquet")
    print(f"✓ Loaded processed: cohort ({len(cohort):,}), exposure ({len(exposure):,}), severity ({len(severity):,}), outcomes ({len(outcomes):,})")
    
    # Also load raw data for comprehensive validation
    print("1b. Loading raw datasets for comparison...")
    patient = pd.read_parquet(f"{data_path}/patient.parquet")
    patient_demo = pd.read_parquet(f"{data_path}/patient_demographic.parquet")
    encounter = pd.read_parquet(f"{data_path}/encounter.parquet")
    encounter_dx = pd.read_parquet(f"{data_path}/encounter_diagnosis.parquet")
    referral = pd.read_parquet(f"{data_path}/referral.parquet")
    lab = pd.read_csv(f"{data_path}/lab.csv")
    medication = pd.read_parquet(f"{data_path}/medication.parquet")
    print(f"✓ Loaded raw: patient ({len(patient):,}), encounters ({len(encounter):,}), diagnoses ({len(encounter_dx):,})")
    
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

# 2. Basic demographic validation
print("\n2. DEMOGRAPHIC VALIDATION")
print("-" * 40)

# Age distribution
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
sex_dist = cohort['sex'].value_counts(normalize=True) * 100
print(f"\nSex distribution:")
for sex, pct in sex_dist.items():
    print(f"  {sex}: {pct:.1f}%")

# Check for missing demographics
missing_demo = cohort[['age_baseline', 'sex']].isnull().sum()
if missing_demo.sum() > 0:
    print(f"⚠️  Missing demographics: {missing_demo.to_dict()}")

# 3. Temporal validation
print("\n3. TEMPORAL VALIDATION")
print("-" * 40)

# Check date ranges
date_cols = [col for col in cohort.columns if 'date' in col.lower()]
for col in date_cols:
    if col in cohort.columns:
        date_range = cohort[col].dropna()
        if len(date_range) > 0:
            print(f"{col}: {date_range.min()} to {date_range.max()}")

# Study period validation
study_start = pd.to_datetime("2015-01-01")
study_end = pd.to_datetime("2022-12-31")
if 'index_date' in cohort.columns:
    index_dates = pd.to_datetime(cohort['index_date'])
    outside_study = ((index_dates < study_start) | (index_dates > study_end)).sum()
    if outside_study > 0:
        print(f"⚠️  {outside_study} patients with index dates outside study period")

# 4. Clinical variables validation
print("\n4. CLINICAL VARIABLES VALIDATION")
print("-" * 40)

# Charlson Comorbidity Index
if 'charlson_index' in cohort.columns:
    charlson_stats = cohort['charlson_index'].describe()
    print(f"Charlson Index:")
    print(f"  Mean: {charlson_stats['mean']:.2f}")
    print(f"  Range: {charlson_stats['min']:.0f} - {charlson_stats['max']:.0f}")
    
    # Clinical plausibility - most patients should have low scores
    high_charlson = (cohort['charlson_index'] > 10).sum()
    if high_charlson > len(cohort) * 0.05:  # >5% with very high scores
        print(f"⚠️  {high_charlson} patients ({high_charlson/len(cohort)*100:.1f}%) with Charlson >10")

# NYD (Not Yet Diagnosed) count validation
if 'NYD_count' in cohort.columns:
    nyd_stats = cohort['NYD_count'].describe()
    print(f"\nNYD count:")
    print(f"  Mean: {nyd_stats['mean']:.2f}")
    print(f"  Range: {nyd_stats['min']:.0f} - {nyd_stats['max']:.0f}")
    
    # Flag patients with excessive NYD codes
    excessive_nyd = (cohort['NYD_count'] > 20).sum()
    if excessive_nyd > 0:
        print(f"⚠️  {excessive_nyd} patients with >20 NYD codes")

# 5. Exposure validation
print("\n5. EXPOSURE VALIDATION")
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

# Check for exposure criteria values
criteria_cols = [col for col in exposure.columns if col.startswith('crit')]
for col in criteria_cols:
    if col in exposure.columns:
        stats = exposure[col].describe()
        print(f"\n{col}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Max: {stats['max']:.0f}")
        
        # Flag suspicious high values (data quality issue)
        if stats['max'] > 1000:
            suspicious = (exposure[col] > 1000).sum()
            print(f"⚠️  {suspicious} patients with suspicious high values (>{1000})")

# 6. Severity index validation
print("\n6. SEVERITY INDEX VALIDATION")
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
    
    # Distribution check
    low_severity = (severity['severity_index'] < 25).mean() * 100
    high_severity = (severity['severity_index'] > 75).mean() * 100
    print(f"  Low severity (<25): {low_severity:.1f}%")
    print(f"  High severity (>75): {high_severity:.1f}%")

# 7. Outcomes validation
print("\n7. OUTCOMES VALIDATION")
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

# Utilization outcomes
util_cols = [col for col in outcomes.columns if any(x in col.lower() for x in ['visit', 'encounter', 'referral'])]
for col in util_cols:
    if col in outcomes.columns:
        util_stats = outcomes[col].describe()
        print(f"\n{col}:")
        print(f"  Mean: {util_stats['mean']:.2f}")
        print(f"  Max: {util_stats['max']:.0f}")
        
        # Flag excessive utilization
        if util_stats['max'] > 100:
            excessive = (outcomes[col] > 100).sum()
            print(f"⚠️  {excessive} patients with >{100} {col}")

# 8. Data completeness check
print("\n8. DATA COMPLETENESS")
print("-" * 40)

datasets = {
    'cohort': cohort,
    'exposure': exposure, 
    'severity': severity,
    'outcomes': outcomes
}

for name, df in datasets.items():
    missing_pct = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_pct[missing_pct > 20]  # >20% missing
    
    if len(high_missing) > 0:
        print(f"\n{name} - High missing data:")
        for col, pct in high_missing.items():
            print(f"  {col}: {pct:.1f}% missing")
    else:
        print(f"✓ {name}: Good data completeness (<20% missing)")

# 9. Cross-dataset consistency
print("\n9. CROSS-DATASET CONSISTENCY")
print("-" * 40)

# Check patient ID consistency
base_patients = set(cohort['Patient_ID'])
for name, df in [('exposure', exposure), ('severity', severity), ('outcomes', outcomes)]:
    if 'Patient_ID' in df.columns:
        dataset_patients = set(df['Patient_ID'])
        missing_in_dataset = len(base_patients - dataset_patients)
        extra_in_dataset = len(dataset_patients - base_patients)
        
        if missing_in_dataset > 0:
            print(f"⚠️  {name}: {missing_in_dataset} cohort patients missing")
        if extra_in_dataset > 0:
            print(f"⚠️  {name}: {extra_in_dataset} extra patients not in cohort")
        
        if missing_in_dataset == 0 and extra_in_dataset == 0:
            print(f"✓ {name}: Perfect patient ID alignment")

# 10. Clinical plausibility summary
print("\n10. CLINICAL PLAUSIBILITY SUMMARY")
print("=" * 50)

issues_found = []

# Age issues
if len(unusual_ages) > 0:
    issues_found.append(f"Unusual ages: {len(unusual_ages)} patients")

# Exposure issues  
if 'exposure_flag' in exposure.columns:
    exposure_rate = exposure['exposure_flag'].mean() * 100
    if exposure_rate < 10 or exposure_rate > 80:
        issues_found.append(f"Unusual exposure rate: {exposure_rate:.1f}%")

# Cost issues
for col in cost_cols:
    if col in outcomes.columns:
        high_cost = (outcomes[col] > 100000).sum()
        if high_cost > 0:
            issues_found.append(f"High costs in {col}: {high_cost} patients")

if len(issues_found) == 0:
    print("✓ NO MAJOR CLINICAL PLAUSIBILITY ISSUES DETECTED")
    print("Data appears clinically reasonable and ready for analysis")
else:
    print("⚠️  ISSUES DETECTED:")
    for issue in issues_found:
        print(f"  - {issue}")
    print("\nRecommend investigating these issues before proceeding")

print("\n" + "=" * 50)
print("CLINICAL VALIDATION COMPLETE")
print("=" * 50)