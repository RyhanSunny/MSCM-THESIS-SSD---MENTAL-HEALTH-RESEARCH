#!/usr/bin/env python3
"""
Detailed Clinical Audit - Deep dive into data quality and clinical plausibility
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== DETAILED CLINICAL AUDIT ===\n")

# Load all processed data
print("Loading processed datasets...")
cohort = pd.read_parquet("data_derived/cohort.parquet")
exposure = pd.read_parquet("data_derived/exposure.parquet")
severity = pd.read_parquet("data_derived/mediator_autoencoder.parquet")
outcomes = pd.read_parquet("data_derived/outcomes.parquet")

print(f"✓ Loaded: cohort ({len(cohort):,}), exposure ({len(exposure):,}), severity ({len(severity):,}), outcomes ({len(outcomes):,})")

# 1. Examine column structure
print("\n1. DATA STRUCTURE EXAMINATION")
print("=" * 50)

print("\nCOHORT COLUMNS:")
print(f"Total columns: {len(cohort.columns)}")
for col in sorted(cohort.columns):
    print(f"  - {col}")

print("\nEXPOSURE COLUMNS:")
print(f"Total columns: {len(exposure.columns)}")
for col in sorted(exposure.columns):
    print(f"  - {col}")

print("\nSEVERITY COLUMNS:")
print(f"Total columns: {len(severity.columns)}")
for col in sorted(severity.columns):
    print(f"  - {col}")

print("\nOUTCOMES COLUMNS:")
print(f"Total columns: {len(outcomes.columns)}")
for col in sorted(outcomes.columns):
    print(f"  - {col}")

# 2. Cohort validation
print("\n\n2. COHORT VALIDATION")
print("=" * 50)

# Patient ID consistency
print(f"Unique patients in cohort: {cohort['Patient_ID'].nunique():,}")
print(f"Total rows in cohort: {len(cohort):,}")

if cohort['Patient_ID'].nunique() != len(cohort):
    print("⚠️  Duplicate patients in cohort!")

# Age validation
if 'age_baseline' in cohort.columns:
    age_stats = cohort['age_baseline'].describe()
    print(f"\nAge at baseline:")
    print(f"  Mean: {age_stats['mean']:.1f} ± {cohort['age_baseline'].std():.1f}")
    print(f"  Range: {age_stats['min']:.0f} - {age_stats['max']:.0f}")
    print(f"  Median: {age_stats['50%']:.0f}")
    
    # Age distribution
    age_bins = pd.cut(cohort['age_baseline'], bins=[0, 18, 30, 50, 65, 80, 120], labels=['<18', '18-29', '30-49', '50-64', '65-79', '80+'])
    age_dist = age_bins.value_counts(normalize=True) * 100
    print(f"\nAge distribution:")
    for age_group, pct in age_dist.items():
        print(f"  {age_group}: {pct:.1f}%")

# Sex distribution
if 'sex' in cohort.columns:
    sex_dist = cohort['sex'].value_counts(normalize=True) * 100
    print(f"\nSex distribution:")
    for sex, pct in sex_dist.items():
        print(f"  {sex}: {pct:.1f}%")

# Study dates
if 'index_date' in cohort.columns:
    cohort['index_date'] = pd.to_datetime(cohort['index_date'])
    print(f"\nIndex date range: {cohort['index_date'].min().date()} to {cohort['index_date'].max().date()}")
    
    # Monthly distribution
    cohort['index_year'] = cohort['index_date'].dt.year
    year_dist = cohort['index_year'].value_counts().sort_index()
    print(f"\nPatients by year:")
    for year, count in year_dist.items():
        print(f"  {year}: {count:,}")

# Charlson index
if 'charlson_index' in cohort.columns:
    charlson_stats = cohort['charlson_index'].describe()
    print(f"\nCharlson Comorbidity Index:")
    print(f"  Mean: {charlson_stats['mean']:.2f}")
    print(f"  Range: {charlson_stats['min']:.0f} - {charlson_stats['max']:.0f}")
    
    # Charlson distribution
    charlson_bins = pd.cut(cohort['charlson_index'], bins=[-1, 0, 2, 4, 6, 20], labels=['0', '1-2', '3-4', '5-6', '7+'])
    charlson_dist = charlson_bins.value_counts(normalize=True) * 100
    print(f"\nCharlson distribution:")
    for score, pct in charlson_dist.items():
        print(f"  {score}: {pct:.1f}%")

# NYD count
if 'NYD_count' in cohort.columns:
    nyd_stats = cohort['NYD_count'].describe()
    print(f"\nNot Yet Diagnosed (NYD) count:")
    print(f"  Mean: {nyd_stats['mean']:.2f}")
    print(f"  Range: {nyd_stats['min']:.0f} - {nyd_stats['max']:.0f}")
    print(f"  Patients with ≥1 NYD: {(cohort['NYD_count'] > 0).sum():,} ({(cohort['NYD_count'] > 0).mean()*100:.1f}%)")

# 3. Exposure analysis
print("\n\n3. EXPOSURE ANALYSIS")
print("=" * 50)

print("Exposure prevalence by hypothesis:")
h1_exposed = exposure['H1_normal_labs'].sum()
h2_exposed = exposure['H2_referral_loop'].sum()
h3_exposed = exposure['H3_drug_persistence'].sum()
total_exposed = exposure['exposure_flag'].sum()

print(f"  H1 (Normal labs): {h1_exposed:,} ({h1_exposed/len(exposure)*100:.1f}%)")
print(f"  H2 (Referral loop): {h2_exposed:,} ({h2_exposed/len(exposure)*100:.1f}%)")
print(f"  H3 (Drug persistence): {h3_exposed:,} ({h3_exposed/len(exposure)*100:.1f}%)")
print(f"  Combined (OR logic): {total_exposed:,} ({total_exposed/len(exposure)*100:.1f}%)")

# Overlap analysis
print("\nExposure overlap:")
h1_h2 = ((exposure['H1_normal_labs']) & (exposure['H2_referral_loop'])).sum()
h1_h3 = ((exposure['H1_normal_labs']) & (exposure['H3_drug_persistence'])).sum()
h2_h3 = ((exposure['H2_referral_loop']) & (exposure['H3_drug_persistence'])).sum()
all_three = ((exposure['H1_normal_labs']) & (exposure['H2_referral_loop']) & (exposure['H3_drug_persistence'])).sum()

print(f"  H1 & H2: {h1_h2:,}")
print(f"  H1 & H3: {h1_h3:,}")
print(f"  H2 & H3: {h2_h3:,}")
print(f"  All three: {all_three:,}")

# Criteria values examination
criteria_cols = [col for col in exposure.columns if col.startswith('crit')]
print(f"\nCriteria values examination:")
for col in criteria_cols:
    if col in exposure.columns:
        stats = exposure[col].describe()
        print(f"\n{col}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['50%']:.2f}")
        print(f"  Range: {stats['min']:.0f} - {stats['max']:.0f}")
        
        # Check for extreme values
        if stats['max'] > 1000:
            extreme = (exposure[col] > 1000).sum()
            print(f"  ⚠️  {extreme} patients with values >1000 (possible data quality issue)")
            
            # Show distribution of extreme values
            extreme_values = exposure[exposure[col] > 1000][col]
            print(f"    Extreme values: {extreme_values.min():.0f} - {extreme_values.max():.0f}")

# 4. Severity index analysis
print("\n\n4. SEVERITY INDEX ANALYSIS")
print("=" * 50)

if 'severity_index' in severity.columns:
    sev_stats = severity['severity_index'].describe()
    print(f"Severity index (0-100 scale):")
    print(f"  Mean: {sev_stats['mean']:.2f} ± {severity['severity_index'].std():.2f}")
    print(f"  Range: {sev_stats['min']:.2f} - {sev_stats['max']:.2f}")
    print(f"  Median: {sev_stats['50%']:.2f}")
    
    # Severity distribution
    sev_bins = pd.cut(severity['severity_index'], bins=[0, 25, 50, 75, 100], labels=['Low (0-25)', 'Moderate (25-50)', 'High (50-75)', 'Very High (75-100)'])
    sev_dist = sev_bins.value_counts(normalize=True) * 100
    print(f"\nSeverity distribution:")
    for sev_level, pct in sev_dist.items():
        print(f"  {sev_level}: {pct:.1f}%")
    
    # Check if severity correlates with exposure
    merged_sev = severity.merge(exposure[['Patient_ID', 'exposure_flag']], on='Patient_ID')
    exposed_severity = merged_sev[merged_sev['exposure_flag']]['severity_index'].mean()
    unexposed_severity = merged_sev[~merged_sev['exposure_flag']]['severity_index'].mean()
    print(f"\nSeverity by exposure status:")
    print(f"  Exposed patients: {exposed_severity:.2f}")
    print(f"  Unexposed patients: {unexposed_severity:.2f}")
    print(f"  Difference: {exposed_severity - unexposed_severity:.2f}")

# 5. Outcome analysis
print("\n\n5. OUTCOME ANALYSIS")
print("=" * 50)

# Examine all outcome variables
outcome_cols = [col for col in outcomes.columns if col != 'Patient_ID']
print(f"Outcome variables ({len(outcome_cols)}):")

for col in outcome_cols:
    if outcomes[col].dtype in ['int64', 'float64']:
        stats = outcomes[col].describe()
        print(f"\n{col}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['50%']:.2f}")
        print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
        
        # Check for extreme values
        if 'cost' in col.lower() and stats['max'] > 50000:
            extreme = (outcomes[col] > 50000).sum()
            print(f"  ⚠️  {extreme} patients with {col} >$50k")
        elif 'visit' in col.lower() and stats['max'] > 50:
            extreme = (outcomes[col] > 50).sum()
            print(f"  ⚠️  {extreme} patients with {col} >50")

# 6. Data completeness audit
print("\n\n6. DATA COMPLETENESS AUDIT")
print("=" * 50)

datasets = {'cohort': cohort, 'exposure': exposure, 'severity': severity, 'outcomes': outcomes}

for name, df in datasets.items():
    print(f"\n{name.upper()}:")
    missing_analysis = df.isnull().sum()
    missing_pct = (missing_analysis / len(df) * 100).round(2)
    
    if missing_analysis.sum() == 0:
        print("  ✓ Complete data (no missing values)")
    else:
        print("  Missing data:")
        for col, count in missing_analysis[missing_analysis > 0].items():
            print(f"    {col}: {count:,} ({missing_pct[col]:.1f}%)")

# 7. Cross-dataset consistency
print("\n\n7. CROSS-DATASET CONSISTENCY")
print("=" * 50)

base_patients = set(cohort['Patient_ID'])
print(f"Base cohort patients: {len(base_patients):,}")

for name, df in [('exposure', exposure), ('severity', severity), ('outcomes', outcomes)]:
    dataset_patients = set(df['Patient_ID'])
    missing = len(base_patients - dataset_patients)
    extra = len(dataset_patients - base_patients)
    
    if missing == 0 and extra == 0:
        print(f"✓ {name}: Perfect alignment")
    else:
        if missing > 0:
            print(f"⚠️  {name}: {missing} cohort patients missing")
        if extra > 0:
            print(f"⚠️  {name}: {extra} extra patients")

# 8. Clinical plausibility summary
print("\n\n8. CLINICAL PLAUSIBILITY SUMMARY")
print("=" * 50)

issues = []
warnings = []

# Check age distribution
if 'age_baseline' in cohort.columns:
    unusual_ages = ((cohort['age_baseline'] < 18) | (cohort['age_baseline'] > 100)).sum()
    if unusual_ages > 0:
        issues.append(f"Unusual ages: {unusual_ages} patients")

# Check exposure rates
exposure_rate = (exposure['exposure_flag'].sum() / len(exposure)) * 100
if exposure_rate < 10:
    warnings.append(f"Low exposure rate: {exposure_rate:.1f}%")
elif exposure_rate > 80:
    warnings.append(f"High exposure rate: {exposure_rate:.1f}%")

# Check for extreme values in criteria
for col in criteria_cols:
    if col in exposure.columns:
        extreme = (exposure[col] > 1000).sum()
        if extreme > 0:
            issues.append(f"Extreme values in {col}: {extreme} patients >1000")

# Check costs
if 'medical_costs' in outcomes.columns:
    high_cost = (outcomes['medical_costs'] > 100000).sum()
    if high_cost > 0:
        warnings.append(f"High medical costs: {high_cost} patients >$100k")

print("ISSUES REQUIRING ATTENTION:")
if len(issues) == 0:
    print("✓ No critical issues detected")
else:
    for issue in issues:
        print(f"  ❌ {issue}")

print("\nWARNINGS FOR REVIEW:")
if len(warnings) == 0:
    print("✓ No warnings")
else:
    for warning in warnings:
        print(f"  ⚠️  {warning}")

print("\n" + "=" * 70)
print("DETAILED CLINICAL AUDIT COMPLETE")
print("=" * 70)

# 9. Summary statistics for clinical review
print("\n\n9. SUMMARY FOR CLINICAL REVIEW")
print("=" * 50)

print(f"Study Population: {len(cohort):,} patients")
if 'age_baseline' in cohort.columns:
    print(f"Age: {cohort['age_baseline'].mean():.1f} ± {cohort['age_baseline'].std():.1f} years")
if 'sex' in cohort.columns:
    female_pct = (cohort['sex'] == 'F').mean() * 100
    print(f"Female: {female_pct:.1f}%")

print(f"\nExposure Patterns:")
print(f"  Normal lab pattern (H1): {h1_exposed/len(exposure)*100:.1f}%")
print(f"  Referral loop pattern (H2): {h2_exposed/len(exposure)*100:.1f}%")
print(f"  Drug persistence pattern (H3): {h3_exposed/len(exposure)*100:.1f}%")
print(f"  Any pattern (combined): {total_exposed/len(exposure)*100:.1f}%")

if 'severity_index' in severity.columns:
    print(f"\nSeverity: {severity['severity_index'].mean():.1f} ± {severity['severity_index'].std():.1f} (0-100 scale)")

if 'medical_costs' in outcomes.columns:
    print(f"Medical costs: ${outcomes['medical_costs'].mean():.2f} ± ${outcomes['medical_costs'].std():.2f}")

print(f"\nData Quality: {'✓ Good' if len(issues) == 0 else '⚠️ Needs attention'}")