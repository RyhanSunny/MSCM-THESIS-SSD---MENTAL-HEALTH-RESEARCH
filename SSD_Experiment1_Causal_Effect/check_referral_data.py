import pandas as pd
import numpy as np

# Load referral data
referral = pd.read_parquet('Notebooks/data/interim/checkpoint_1_20250318_024427/referral.parquet')

print('Referral Data Shape:', referral.shape)
print('\nColumn Names:')
print(referral.columns.tolist())
print('\nFirst few columns of first 5 rows:')
print(referral[['Referral_ID', 'Patient_ID', 'Encounter_ID', 'CompletedDate']].head())
print('\nData types:')
print(referral.dtypes)
print('\nUnique Patient_IDs:', referral['Patient_ID'].nunique())
print('\nReferrals per patient stats:')
ref_counts = referral['Patient_ID'].value_counts()
print(f'Min: {ref_counts.min()}, Max: {ref_counts.max()}, Mean: {ref_counts.mean():.2f}, Median: {ref_counts.median():.2f}')
print(f'Total unique patients with referrals: {referral["Patient_ID"].nunique()}')
print('\nTop 10 patients by referral count:')
print(ref_counts.head(10))
print('\nSample of patients with high referral counts:')
high_ref_patients = ref_counts[ref_counts > 1000].head(10)
print(high_ref_patients)