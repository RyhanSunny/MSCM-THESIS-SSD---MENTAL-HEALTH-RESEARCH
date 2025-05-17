#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_lab_flag.py – Lab-based Flagging

- Loads the validated cohort
- Flags patients based on lab results (to be implemented)
- Saves lab_flag.parquet to data_derived/
"""

import pandas as pd
from pathlib import Path

COHORT_PATH = Path(__file__).resolve().parents[1] / 'data_derived' / 'cohort.parquet'
OUT_PATH = Path(__file__).resolve().parents[1] / 'data_derived' / 'lab_flag.parquet'

# Load cohort
df = pd.read_parquet(COHORT_PATH)

# TODO: Implement lab-based flag logic here
df['Lab_flag'] = 0  # Placeholder

# Save output
df[['Patient_ID', 'Lab_flag']].to_parquet(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}") 