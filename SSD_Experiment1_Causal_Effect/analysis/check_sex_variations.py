import pandas as pd

# Load cohort data
cohort = pd.read_parquet('data_derived/cohort.parquet')

# Check unique sex values
print("Unique sex values in cohort:")
print(cohort['Sex'].value_counts())
print(f"\nUnique values: {cohort['Sex'].unique()}")

# Check for different case variations
print("\nChecking case variations:")
sex_upper = cohort['Sex'].str.upper().value_counts()
print(sex_upper)