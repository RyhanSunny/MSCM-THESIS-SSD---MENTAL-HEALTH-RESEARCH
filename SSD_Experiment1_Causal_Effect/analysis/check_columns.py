import pandas as pd

# Check columns in each file
files = {
    'exposure': 'data_derived/exposure.parquet',
    'cohort': 'data_derived/cohort.parquet',
    'outcomes': 'data_derived/outcomes.parquet',
    'confounders': 'data_derived/confounders.parquet',
    'autoencoder': 'data_derived/mediator_autoencoder.parquet'
}

for name, path in files.items():
    df = pd.read_parquet(path)
    print(f"\n{name.upper()} columns:")
    print(df.columns.tolist())