import pandas as pd

# Load first imputed dataset
df = pd.read_parquet('data_derived/imputed_master/master_imputed_1.parquet')

print('Checking individual hypotheses:')
print(f'H1_normal_labs: {df["H1_normal_labs"].sum()} ({df["H1_normal_labs"].mean()*100:.1f}%)')
print(f'H2_referral_loop: {df["H2_referral_loop"].sum()} ({df["H2_referral_loop"].mean()*100:.1f}%)')
print(f'H3_drug_persistence: {df["H3_drug_persistence"].sum()} ({df["H3_drug_persistence"].mean()*100:.1f}%)')

print('\nChecking OR combination:')
or_combo = df['H1_normal_labs'] | df['H2_referral_loop'] | df['H3_drug_persistence']
print(f'OR combination (calculated): {or_combo.sum()} ({or_combo.mean()*100:.1f}%)')
print(f'exposure_flag (stored): {df["exposure_flag"].sum()} ({df["exposure_flag"].mean()*100:.1f}%)')

print('\nDiscrepancy analysis:')
print(f'Difference: {or_combo.sum() - df["exposure_flag"].sum()} patients')

# Check if exposure_flag is the AND combination instead
and_combo = df['H1_normal_labs'] & df['H2_referral_loop'] & df['H3_drug_persistence']
print(f'\nAND combination: {and_combo.sum()} ({and_combo.mean()*100:.1f}%)')

# Check overlaps
print('\nOverlaps between hypotheses:')
print(f'H1 & H2: {(df["H1_normal_labs"] & df["H2_referral_loop"]).sum()}')
print(f'H1 & H3: {(df["H1_normal_labs"] & df["H3_drug_persistence"]).sum()}')
print(f'H2 & H3: {(df["H2_referral_loop"] & df["H3_drug_persistence"]).sum()}')
print(f'All three: {and_combo.sum()}')