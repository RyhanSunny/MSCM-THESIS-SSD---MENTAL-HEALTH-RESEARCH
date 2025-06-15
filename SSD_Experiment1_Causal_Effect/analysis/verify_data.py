import pandas as pd

# Load data
exposure = pd.read_parquet('data_derived/exposure.parquet')
outcomes = pd.read_parquet('data_derived/outcomes.parquet')
cohort = pd.read_parquet('data_derived/cohort.parquet')

# Merge
data = exposure.merge(outcomes[['Patient_ID', 'total_encounters', 'medical_costs']], on='Patient_ID')

# Calculate statistics
or_exp = data[data['exposure_flag']]
or_unexp = data[~data['exposure_flag']]
and_exp = data[data['exposure_flag_strict']]
and_unexp = data[~data['exposure_flag_strict']]

print("VERIFICATION OF ACTUAL DATA:")
print("="*50)
print(f"Total patients: {len(data):,}")
print(f"\nOR Logic:")
print(f"  Exposed: {len(or_exp):,} ({len(or_exp)/len(data)*100:.1f}%)")
print(f"  Encounters: {or_exp.total_encounters.mean():.1f}")
print(f"  Cost: ${or_exp.medical_costs.mean():.0f}")

print(f"\nAND Logic:")
print(f"  Exposed: {len(and_exp):,} ({len(and_exp)/len(data)*100:.1f}%)")
print(f"  Encounters: {and_exp.total_encounters.mean():.1f}")
print(f"  Cost: ${and_exp.medical_costs.mean():.0f}")

print(f"\nDiscrepancy factor: {len(or_exp)/len(and_exp):.0f}x")

# Check individual criteria
print(f"\nIndividual criteria (from exposure data):")
print(f"  H1 (Normal Labs): {exposure.H1_normal_labs.sum():,}")
print(f"  H2 (Referral Loops): {exposure.H2_referral_loop.sum():,}")
print(f"  H3 (Drug Persistence): {exposure.H3_drug_persistence.sum():,}")