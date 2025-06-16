#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compare_exposure_logic.py

Quick utility to print side-by-side statistics for OR-logic vs AND-logic exposure cohorts.
Usage:
    python scripts/compare_exposure_logic.py data_derived/exposure_or.parquet data_derived/exposure_and.parquet
"""

import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: compare_exposure_logic.py <or_file> <and_file>")
    sys.exit(1)

or_file = Path(sys.argv[1])
and_file = Path(sys.argv[2])

if not or_file.exists() or not and_file.exists():
    print("Both OR and AND exposure parquet files must exist.")
    sys.exit(1)

or_df = pd.read_parquet(or_file)
and_df = pd.read_parquet(and_file)

n_total = len(or_df)

a_or = or_df["exposure_flag"].sum()
a_and = and_df["exposure_flag"].sum()

print("================= SSD Exposure Logic Comparison =================")
print(f"Cohort size                       : {n_total:,}")
print()
print("Logic      Exposed   Prevalence")
print("----------- ---------- ----------")
print(f"OR (DSM-5) {a_or:10,} {a_or/n_total:10.1%}")
print(f"AND (DSM-IV){a_and:10,} {a_and/n_total:10.1%}")
print("===============================================================")

# Overlap
or_ids = set(or_df.loc[or_df.exposure_flag, "Patient_ID"])
and_ids = set(and_df.loc[and_df.exposure_flag, "Patient_ID"])
print(f"AND subset within OR? {len(and_ids - or_ids) == 0}")
print(f"Jaccard similarity    : {len(and_ids & or_ids) / len(or_ids):.3f}") 