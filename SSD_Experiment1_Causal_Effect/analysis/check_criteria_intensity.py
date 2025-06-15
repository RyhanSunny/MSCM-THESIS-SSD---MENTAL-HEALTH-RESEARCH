import pandas as pd

# Load exposure data
exposure = pd.read_parquet('data_derived/exposure.parquet')

# Check the actual values for each criterion intensity
print("CRITERIA INTENSITY VALUES:")
print("="*50)

# H1: Normal lab count
h1_data = exposure[exposure['H1_normal_labs']]
print(f"\nH1 - Normal Lab Count (n={len(h1_data):,}):")
print(f"  Min: {h1_data['normal_lab_count'].min()}")
print(f"  Max: {h1_data['normal_lab_count'].max()}")
print(f"  Mean: {h1_data['normal_lab_count'].mean():.2f}")
print(f"  Median: {h1_data['normal_lab_count'].median()}")
print(f"  Unique values: {sorted(h1_data['normal_lab_count'].unique())[:10]}")

# H2: Referral count
h2_data = exposure[exposure['H2_referral_loop']]
print(f"\nH2 - Referral Count (n={len(h2_data):,}):")
print(f"  Min: {h2_data['symptom_referral_n'].min()}")
print(f"  Max: {h2_data['symptom_referral_n'].max()}")
print(f"  Mean: {h2_data['symptom_referral_n'].mean():.2f}")
print(f"  Median: {h2_data['symptom_referral_n'].median()}")
print(f"  Unique values: {sorted(h2_data['symptom_referral_n'].unique())[:10]}")

# H3: Drug persistence days
h3_data = exposure[exposure['H3_drug_persistence']]
print(f"\nH3 - Drug Days (n={len(h3_data):,}):")
print(f"  Min: {h3_data['drug_days_in_window'].min()}")
print(f"  Max: {h3_data['drug_days_in_window'].max()}")
print(f"  Mean: {h3_data['drug_days_in_window'].mean():.2f}")
print(f"  Median: {h3_data['drug_days_in_window'].median()}")
print(f"  Unique values: {sorted(h3_data['drug_days_in_window'].unique())[:10]}")

# Check for outliers
print("\nOUTLIER CHECK:")
print(f"H2 values > 1000: {(h2_data['symptom_referral_n'] > 1000).sum()}")
print(f"H2 values > 10000: {(h2_data['symptom_referral_n'] > 10000).sum()}")
print(f"H2 values > 20000: {(h2_data['symptom_referral_n'] > 20000).sum()}")