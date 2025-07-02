#!/usr/bin/env python
"""Test script to verify environment and basic imports"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("Current directory:", Path.cwd())

# Test if we can read the cohort file
cohort_path = Path("data_derived/cohort.parquet")
if cohort_path.exists():
    print(f"Cohort file exists: {cohort_path}")
    df = pd.read_parquet(cohort_path)
    print(f"Cohort shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("First few rows:")
    print(df.head())
else:
    print("Cohort file not found")