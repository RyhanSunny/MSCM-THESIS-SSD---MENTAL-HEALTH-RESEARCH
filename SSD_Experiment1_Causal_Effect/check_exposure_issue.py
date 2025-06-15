import pandas as pd
import numpy as np

# Load the exposure data created by 02_exposure_flag.py
exposure = pd.read_parquet('data_derived/exposure.parquet')

print("Exposure data shape:", exposure.shape)
print("\nColumns:", exposure.columns.tolist())

# Check the symptom_referral_n column
print("\n=== Symptom Referral Count Analysis ===")
print(f"Min: {exposure['symptom_referral_n'].min()}")
print(f"Max: {exposure['symptom_referral_n'].max()}")
print(f"Mean: {exposure['symptom_referral_n'].mean():.2f}")
print(f"Median: {exposure['symptom_referral_n'].median():.2f}")
print(f"Unique values: {exposure['symptom_referral_n'].nunique()}")

# Check distribution
print("\nValue counts for symptom_referral_n (top 20):")
print(exposure['symptom_referral_n'].value_counts().head(20))

# Check data type
print(f"\nData type of symptom_referral_n: {exposure['symptom_referral_n'].dtype}")

# Check for patients with extreme values
extreme_patients = exposure[exposure['symptom_referral_n'] > 1000]
print(f"\nPatients with >1000 referrals: {len(extreme_patients)}")
if len(extreme_patients) > 0:
    print("Sample of extreme values:")
    print(extreme_patients[['Patient_ID', 'symptom_referral_n']].head(10))

# Check correlation with other criteria
print("\n=== Cross-check with other criteria ===")
print("Patients meeting each criterion:")
print(f"H1 (normal labs): {exposure['H1_normal_labs'].sum()}")
print(f"H2 (referral loops): {exposure['H2_referral_loop'].sum()}")
print(f"H3 (drug persistence): {exposure['H3_drug_persistence'].sum()}")

# Check if there's a pattern in the extreme values
if len(extreme_patients) > 0:
    print("\nExtreme value pattern check:")
    extreme_vals = extreme_patients['symptom_referral_n'].unique()
    print(f"Unique extreme values: {sorted(extreme_vals)[:10]}")  # First 10 unique extreme values