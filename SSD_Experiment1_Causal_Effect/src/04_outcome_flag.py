#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_outcome_flag.py – Outcome Flagging

- Loads the validated cohort
- Flags outcomes of interest (to be implemented)
- Saves outcome_flag.parquet to data_derived/
"""

import pandas as pd
from pathlib import Path

COHORT_PATH = Path(__file__).resolve().parents[1] / 'data_derived' / 'cohort.parquet'
OUT_PATH = Path(__file__).resolve().parents[1] / 'data_derived' / 'outcome_flag.parquet'

# Load cohort
df = pd.read_parquet(COHORT_PATH)

# TODO: Implement outcome flag logic here
df['Outcome_flag'] = 0  # Placeholder

# Save output
df[['Patient_ID', 'Outcome_flag']].to_parquet(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}") 