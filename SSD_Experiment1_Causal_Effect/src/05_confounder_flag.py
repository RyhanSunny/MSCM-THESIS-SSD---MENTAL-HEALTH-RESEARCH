#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_confounder_flag.py – Confounder/Covariate Flagging

- Loads the validated cohort
- Flags confounders/covariates (to be implemented)
- Saves confounder_flag.parquet to data_derived/
"""

import pandas as pd
from pathlib import Path

COHORT_PATH = Path(__file__).resolve().parents[1] / 'data_derived' / 'cohort.parquet'
OUT_PATH = Path(__file__).resolve().parents[1] / 'data_derived' / 'confounder_flag.parquet'

# Load cohort
df = pd.read_parquet(COHORT_PATH)

# TODO: Implement confounder/covariate flag logic here
df['Confounder_flag'] = 0  # Placeholder

# Save output
df[['Patient_ID', 'Confounder_flag']].to_parquet(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}") 