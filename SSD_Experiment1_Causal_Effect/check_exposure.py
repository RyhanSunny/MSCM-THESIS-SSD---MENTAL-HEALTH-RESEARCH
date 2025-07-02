import pandas as pd

# Load exposure data
df = pd.read_parquet('data_derived/exposure.parquet')

print("Individual criteria counts:")
print(f"crit1_normal_labs: {df['crit1_normal_labs'].sum()}")
print(f"crit2_sympt_ref: {df['crit2_sympt_ref'].sum()}")
print(f"crit3_drug_90d: {df['crit3_drug_90d'].sum()}")

# Manual OR calculation
or_result = df['crit1_normal_labs'] | df['crit2_sympt_ref'] | df['crit3_drug_90d']
print(f"\nManual OR calculation: {or_result.sum()}")
print(f"Stored exposure_flag: {df['exposure_flag'].sum()}")

# Check overlap
print("\nOverlap analysis:")
c1_only = df['crit1_normal_labs'] & ~df['crit2_sympt_ref'] & ~df['crit3_drug_90d']
c2_only = ~df['crit1_normal_labs'] & df['crit2_sympt_ref'] & ~df['crit3_drug_90d']
c3_only = ~df['crit1_normal_labs'] & ~df['crit2_sympt_ref'] & df['crit3_drug_90d']
all_three = df['crit1_normal_labs'] & df['crit2_sympt_ref'] & df['crit3_drug_90d']

print(f"Only crit1: {c1_only.sum()}")
print(f"Only crit2: {c2_only.sum()}")
print(f"Only crit3: {c3_only.sum()}")
print(f"All three: {all_three.sum()}")