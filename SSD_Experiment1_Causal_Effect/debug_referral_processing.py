import pandas as pd
import numpy as np
import re

# Load data
print("Loading data...")
cohort = pd.read_parquet('data_derived/cohort.parquet')
referral = pd.read_parquet('Notebooks/data/interim/checkpoint_1_20250318_024427/referral.parquet')
enc_diag = pd.read_parquet('Notebooks/data/interim/checkpoint_1_20250318_024427/encounter_diagnosis.parquet')

# Set up cohort dates
cohort["exp_start"] = cohort.IndexDate_lab
cohort["exp_end"] = cohort.IndexDate_lab + pd.Timedelta(days=365)

# Filter encounter diagnoses for symptom codes
SYMPTOM_RE = re.compile(r"^(78[0-9]|799)")
enc_diag_filtered = enc_diag[enc_diag.DiagnosisCode_calc.str.match(SYMPTOM_RE, na=False)][["Encounter_ID", "DiagnosisCode_calc"]]
print(f"\nFiltered encounter diagnoses: {len(enc_diag_filtered):,} rows")
print(f"Unique encounters with symptom codes: {enc_diag_filtered['Encounter_ID'].nunique():,}")

# Check for duplicates
print(f"\nDuplicate encounter IDs in filtered data: {enc_diag_filtered.duplicated(subset=['Encounter_ID']).sum()}")

# Process referrals
referral = referral.dropna(subset=["CompletedDate", "DateCreated"])
referral["ReferralDate"] = pd.to_datetime(
    referral.CompletedDate.fillna(referral.DateCreated), errors="coerce")
referral = referral.dropna(subset=["ReferralDate"])
referral_subset = referral[["Patient_ID", "Encounter_ID", "Name_calc", "ReferralDate"]].copy()

# Check referral encounter IDs
print(f"\nReferrals with non-null Encounter_ID: {referral_subset['Encounter_ID'].notna().sum()}")
print(f"Unique encounter IDs in referrals: {referral_subset['Encounter_ID'].nunique()}")

# Do the merge
print("\nPerforming merge...")
merged = referral_subset.merge(enc_diag_filtered, on="Encounter_ID", how="inner")
print(f"Rows after merge: {len(merged):,}")

# Check a specific patient with extreme values
test_patient = 1002000000022138  # Patient with 22271 referrals
print(f"\n=== Debugging patient {test_patient} ===")

# Original referrals for this patient
patient_refs = referral[referral.Patient_ID == test_patient]
print(f"Original referrals for patient: {len(patient_refs)}")

# After merge
patient_merged = merged[merged.Patient_ID == test_patient]
print(f"Rows after merge for patient: {len(patient_merged)}")

# Check encounter IDs
if len(patient_refs) > 0:
    print(f"Unique encounter IDs in patient's referrals: {patient_refs['Encounter_ID'].nunique()}")
    print(f"Sample encounter IDs: {patient_refs['Encounter_ID'].dropna().head().tolist()}")
    
if len(patient_merged) > 0:
    print(f"Unique encounter IDs after merge: {patient_merged['Encounter_ID'].nunique()}")
    
    # Check for duplicate encounter IDs in the filtered diagnosis data
    enc_ids = patient_merged['Encounter_ID'].unique()
    for enc_id in enc_ids[:5]:  # Check first 5
        count = len(enc_diag_filtered[enc_diag_filtered.Encounter_ID == enc_id])
        print(f"  Encounter {enc_id}: {count} diagnosis rows")