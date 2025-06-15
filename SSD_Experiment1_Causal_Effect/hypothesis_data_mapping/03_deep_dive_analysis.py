#!/usr/bin/env python3
"""
03_deep_dive_analysis.py - Deep dive into specific data requirements
Author: Ryhan Suny  
Date: 2025-05-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
CHECKPOINT = Path("Notebooks/data/interim/checkpoint_1_20250318_024427")
OUTPUT_DIR = Path("hypothesis_data_mapping")

print("=" * 80)
print("DEEP DIVE ANALYSIS - DATA AVAILABILITY FOR HYPOTHESES")
print("=" * 80)

# 1. Check for mental health codes (ICD-9 since no ICD-10 F-codes found)
print("\n1. MENTAL HEALTH DIAGNOSES CHECK (ICD-9 290-319):")
print("-" * 60)

health_cond = pd.read_parquet(CHECKPOINT / "health_condition.parquet")
enc_diag = pd.read_parquet(CHECKPOINT / "encounter_diagnosis.parquet")

# ICD-9 mental health codes
mh_icd9_prefixes = [str(i) for i in range(290, 320)]  # 290-319
mh_health = health_cond[health_cond['DiagnosisCode_calc'].str[:3].isin(mh_icd9_prefixes)]
mh_enc = enc_diag[enc_diag['DiagnosisCode_calc'].str[:3].isin(mh_icd9_prefixes)]

print(f"Health conditions with MH codes (290-319): {len(mh_health):,}")
print(f"Encounter diagnoses with MH codes: {len(mh_enc):,}")
print(f"Unique patients with MH conditions: {mh_health['Patient_ID'].nunique():,}")

# Top mental health diagnoses
if len(mh_health) > 0:
    print("\nTop 10 mental health diagnoses:")
    top_mh = mh_health['DiagnosisCode_calc'].value_counts().head(10)
    for code, count in top_mh.items():
        print(f"  {code}: {count:,}")

# 2. Check for anxiety/depression specifically
print("\n2. ANXIETY & DEPRESSION CODES:")
print("-" * 60)

anxiety_codes = ['300', '300.0', '300.00', '300.01', '300.02']  # Anxiety disorders
depression_codes = ['296', '296.2', '296.3', '311']  # Depression

anxiety_cond = health_cond[health_cond['DiagnosisCode_calc'].str.startswith(tuple(anxiety_codes), na=False)]
depression_cond = health_cond[health_cond['DiagnosisCode_calc'].str.startswith(tuple(depression_codes), na=False)]

print(f"Patients with anxiety diagnoses: {anxiety_cond['Patient_ID'].nunique():,}")
print(f"Patients with depression diagnoses: {depression_cond['Patient_ID'].nunique():,}")

# 3. Check ED visits in encounters
print("\n3. EMERGENCY DEPARTMENT VISITS:")
print("-" * 60)

encounter = pd.read_parquet(CHECKPOINT / "encounter.parquet")
print("\nEncounter types distribution:")
enc_types = encounter['EncounterType'].value_counts()
for enc_type, count in enc_types.items():
    print(f"  {enc_type}: {count:,}")
    
# Check reason field for ED indicators
if 'Reason_orig' in encounter.columns:
    ed_keywords = ['emergency', 'urgent', 'ED', 'ER']
    ed_reasons = encounter['Reason_orig'].astype(str).str.lower()
    ed_matches = ed_reasons.str.contains('|'.join(ed_keywords), na=False)
    print(f"\nEncounters with ED keywords in reason: {ed_matches.sum():,}")

# 4. Check SES completeness
print("\n4. SOCIOECONOMIC INDICATORS COMPLETENESS:")
print("-" * 60)

demo = pd.read_parquet(CHECKPOINT / "patient_demographic.parquet")
ses_fields = ['Occupation', 'HighestEducation', 'HousingStatus', 'ResidencePostalCode']

for field in ses_fields:
    if field in demo.columns:
        non_null = demo[field].notna().sum()
        pct = non_null / len(demo) * 100
        print(f"{field}: {non_null:,} / {len(demo):,} ({pct:.1f}%)")
        
        # Sample values if any exist
        if non_null > 0:
            print(f"  Sample values: {demo[field].dropna().head(5).tolist()}")

# 5. Provider change analysis feasibility
print("\n5. PROVIDER CHANGE ANALYSIS:")
print("-" * 60)

# Check provider ID distribution
provider_counts = encounter.groupby('Patient_ID')['Provider_ID'].nunique()
multi_provider = (provider_counts > 1).sum()
print(f"Patients seeing multiple providers: {multi_provider:,} / {len(provider_counts):,} ({multi_provider/len(provider_counts)*100:.1f}%)")
print(f"Average providers per patient: {provider_counts.mean():.2f}")
print(f"Max providers for one patient: {provider_counts.max()}")

# 6. NYD code analysis
print("\n6. NOT YET DIAGNOSED (NYD) CODES:")
print("-" * 60)

nyd_codes = ["799.9", "V71.0", "V71.1", "V71.2", "V71.3", "V71.4", "V71.5", "V71.6", "V71.7", "V71.8", "V71.9"]
nyd_health = health_cond[health_cond['DiagnosisCode_calc'].isin(nyd_codes)]
nyd_enc = enc_diag[enc_diag['DiagnosisCode_calc'].isin(nyd_codes)]

print(f"Health conditions with NYD codes: {len(nyd_health):,}")
print(f"Encounter diagnoses with NYD codes: {len(nyd_enc):,}")
print(f"Unique patients with NYD codes: {nyd_health['Patient_ID'].nunique():,}")

# 7. Temporal data availability
print("\n7. TEMPORAL DATA COVERAGE:")
print("-" * 60)

enc_dates = pd.to_datetime(encounter['EncounterDate'])
print(f"Earliest encounter: {enc_dates.min()}")
print(f"Latest encounter: {enc_dates.max()}")
print(f"Total time span: {(enc_dates.max() - enc_dates.min()).days / 365.25:.1f} years")

# Encounters by year
encounter['year'] = enc_dates.dt.year
year_counts = encounter['year'].value_counts().sort_index()
print("\nEncounters by year:")
for year, count in year_counts.items():
    if pd.notna(year):
        print(f"  {int(year)}: {count:,}")

# Create visualization of data gaps
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Mental health diagnosis availability
ax1 = axes[0, 0]
mh_data = pd.DataFrame({
    'Category': ['Has MH Diagnosis', 'No MH Diagnosis'],
    'Count': [mh_health['Patient_ID'].nunique(), 
              len(pd.read_parquet(CHECKPOINT / "patient.parquet")) - mh_health['Patient_ID'].nunique()]
})
ax1.pie(mh_data['Count'], labels=mh_data['Category'], autopct='%1.1f%%', startangle=90)
ax1.set_title('Mental Health Diagnosis Coverage')

# 2. SES data completeness
ax2 = axes[0, 1]
ses_completeness = []
for field in ses_fields:
    if field in demo.columns:
        pct = demo[field].notna().mean() * 100
        ses_completeness.append({'Field': field, 'Completeness': pct})
ses_df = pd.DataFrame(ses_completeness)
bars = ax2.bar(ses_df['Field'], ses_df['Completeness'])
ax2.set_ylabel('Completeness %')
ax2.set_title('Socioeconomic Data Completeness')
ax2.set_xticklabels(ses_df['Field'], rotation=45, ha='right')

# Color bars based on completeness
for i, bar in enumerate(bars):
    if ses_df.iloc[i]['Completeness'] == 0:
        bar.set_color('#e74c3c')
    elif ses_df.iloc[i]['Completeness'] < 50:
        bar.set_color('#f39c12')
    else:
        bar.set_color('#2ecc71')

# 3. Provider distribution
ax3 = axes[1, 0]
provider_hist = provider_counts.value_counts().sort_index().head(10)
ax3.bar(provider_hist.index, provider_hist.values)
ax3.set_xlabel('Number of Providers')
ax3.set_ylabel('Number of Patients')
ax3.set_title('Distribution of Provider Count per Patient')

# 4. Temporal coverage
ax4 = axes[1, 1]
year_data = year_counts[year_counts.index.notna()].sort_index()
ax4.plot(year_data.index, year_data.values, marker='o')
ax4.set_xlabel('Year')
ax4.set_ylabel('Number of Encounters')
ax4.set_title('Temporal Coverage of Encounter Data')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'deep_dive_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Save summary report
summary = f"""
DEEP DIVE ANALYSIS SUMMARY
=========================

1. MENTAL HEALTH DATA:
   - ICD-9 MH codes (290-319): {mh_health['Patient_ID'].nunique():,} patients
   - Anxiety disorders: {anxiety_cond['Patient_ID'].nunique():,} patients  
   - Depression: {depression_cond['Patient_ID'].nunique():,} patients
   - Status: {'AVAILABLE' if mh_health['Patient_ID'].nunique() > 0 else 'NOT AVAILABLE'}

2. EMERGENCY VISITS:
   - Direct ED type: Not found in EncounterType
   - ED keywords in reasons: {ed_matches.sum() if 'ed_matches' in locals() else 'N/A'}
   - Status: REQUIRES DERIVATION

3. SOCIOECONOMIC DATA:
   - Postal codes: 0% complete (CRITICAL GAP)
   - Education: {demo['HighestEducation'].notna().mean()*100:.1f}% complete
   - Occupation: {demo['Occupation'].notna().mean()*100:.1f}% complete  
   - Housing: {demo['HousingStatus'].notna().mean()*100:.1f}% complete
   - Status: PARTIAL - Can use education/occupation/housing

4. PROVIDER CHANGES:
   - Multi-provider patients: {multi_provider:,} ({multi_provider/len(provider_counts)*100:.1f}%)
   - Average providers/patient: {provider_counts.mean():.2f}
   - Status: AVAILABLE

5. NYD CODES:
   - Patients with NYD: {nyd_health['Patient_ID'].nunique():,}
   - Status: {'AVAILABLE' if nyd_health['Patient_ID'].nunique() > 0 else 'NOT AVAILABLE'}

6. TEMPORAL DATA:
   - Coverage: {enc_dates.min().strftime('%Y-%m-%d')} to {enc_dates.max().strftime('%Y-%m-%d')}
   - Span: {(enc_dates.max() - enc_dates.min()).days / 365.25:.1f} years
   - Status: EXCELLENT
"""

with open(OUTPUT_DIR / 'deep_dive_summary.txt', 'w') as f:
    f.write(summary)

print(summary)