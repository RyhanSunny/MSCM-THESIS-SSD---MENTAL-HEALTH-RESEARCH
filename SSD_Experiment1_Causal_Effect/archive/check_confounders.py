import pandas as pd

# Load confounders
conf = pd.read_parquet('data_derived/confounders.parquet')
print('CONFOUNDER MATRIX SUMMARY')
print('=' * 50)
print(f'Total patients: {len(conf):,}')
print(f'Total covariates: {len(conf.columns) - 1}')  # Exclude Patient_ID

print(f'\nCovariate list:')
covariate_cols = [col for col in conf.columns if col != 'Patient_ID']
for i, col in enumerate(sorted(covariate_cols), 1):
    print(f'  {i:2d}. {col}')

print(f'\nSample statistics:')
numeric_cols = conf.select_dtypes(include=['int16', 'int32', 'int64', 'float64']).columns
for col in ['Age_at_2015', 'Charlson', 'baseline_encounters', 'baseline_med_count']:
    if col in numeric_cols:
        mean_val = conf[col].mean()
        std_val = conf[col].std()
        print(f'  {col}: mean={mean_val:.2f}, std={std_val:.2f}')

print(f'\nBinary variable prevalences:')
binary_cols = []
for col in conf.columns:
    if col != 'Patient_ID' and conf[col].dtype in ['int32', 'int64']:
        unique_vals = conf[col].unique()
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            binary_cols.append(col)

for col in sorted(binary_cols)[:10]:  # Show first 10
    prev = conf[col].mean() * 100
    print(f'  {col}: {prev:.1f}%')

# Check for significant imbalances
print(f'\nTotal binary variables: {len(binary_cols)}')
print(f'Total numeric variables: {len([c for c in covariate_cols if c not in binary_cols])}')

# Check data completeness
missing_counts = conf.isnull().sum()
if missing_counts.sum() > 0:
    print(f'\nMissing data:')
    for col, count in missing_counts[missing_counts > 0].items():
        print(f'  {col}: {count:,} ({count/len(conf)*100:.1f}%)')
else:
    print(f'\n✓ No missing data in confounder matrix')