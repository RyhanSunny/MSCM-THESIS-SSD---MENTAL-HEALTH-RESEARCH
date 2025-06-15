#!/usr/bin/env python3
"""Check exposure criteria breakdown"""

import pandas as pd

# Load exposure data
exposure = pd.read_parquet('data_derived/exposure.parquet')
print('Individual criteria analysis:')
print(f'Total patients: {len(exposure):,}')
print()

# Calculate individual criteria
crit1 = exposure['normal_lab_count'] >= 3
crit2 = exposure['symptom_referral_n'] >= 2  
crit3 = exposure['drug_days_in_window'] >= 90

print(f'Criterion 1 (≥3 normal labs): {crit1.sum():,} patients ({crit1.mean():.1%})')
print(f'Criterion 2 (≥2 symptom referrals): {crit2.sum():,} patients ({crit2.mean():.1%})')
print(f'Criterion 3 (≥90 drug days): {crit3.sum():,} patients ({crit3.mean():.1%})')
print()

print('Combinations:')
or_combo = crit1 | crit2 | crit3
c1_c2 = crit1 & crit2
c1_c3 = crit1 & crit3
c2_c3 = crit2 & crit3
all_three = crit1 & crit2 & crit3

print(f'C1 OR C2 OR C3: {or_combo.sum():,} patients ({or_combo.mean():.1%})')
print(f'C1 AND C2: {c1_c2.sum():,} patients ({c1_c2.mean():.1%})')
print(f'C1 AND C3: {c1_c3.sum():,} patients ({c1_c3.mean():.1%})')
print(f'C2 AND C3: {c2_c3.sum():,} patients ({c2_c3.mean():.1%})')
print(f'C1 AND C2 AND C3 (current): {all_three.sum():,} patients ({all_three.mean():.1%})')
print()

print('Data distributions:')
print(f'Normal lab count - Mean: {exposure["normal_lab_count"].mean():.1f}, Max: {exposure["normal_lab_count"].max():.0f}')
print(f'Symptom referrals - Mean: {exposure["symptom_referral_n"].mean():.1f}, Max: {exposure["symptom_referral_n"].max():.0f}')  
print(f'Drug days - Mean: {exposure["drug_days_in_window"].mean():.1f}, Max: {exposure["drug_days_in_window"].max():.0f}')

print('\nDetailed breakdown:')
print('Normal lab count distribution:')
print(exposure['normal_lab_count'].value_counts().sort_index().head(10))

print('\nSymptom referral distribution:') 
print(exposure['symptom_referral_n'].value_counts().sort_index().head(10))

print('\nDrug days distribution (first 10 values):')
print(exposure['drug_days_in_window'].value_counts().sort_index().head(10))