#!/usr/bin/env python3
"""Generate Dr. Felipe's patient characteristics table."""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data_derived"


def generate_felipe_table(cohort: pd.DataFrame, referral: pd.DataFrame) -> pd.DataFrame:
    """Return a table with key patient characteristics required by Dr. Felipe."""
    table = pd.DataFrame()
    table["PID"] = cohort["Patient_ID"]

    age_col = next((c for c in cohort.columns if c.startswith("Age_at_")), None)
    if age_col:
        table["age"] = cohort[age_col]
    else:
        table["age"] = pd.NA

    table["NYD_yn"] = (cohort.get("NYD_count", 0) > 0).astype(int)

    psych_pattern = r"psychiatr|mental|psych|behav|counsel|therapy"
    psy_flag = (referral["Name_calc"].str.contains(psych_pattern, case=False, na=False) |
                referral.get("Type", "").str.contains("psychiatric", case=False, na=False))
    psy_flag = psy_flag.groupby(referral["Patient_ID"]).any().astype(int)

    table = table.merge(psy_flag.rename("referred_to_psy_yn"), left_on="PID", right_index=True, how="left")
    table["referred_to_psy_yn"] = table["referred_to_psy_yn"].fillna(0).astype(int)
    return table


def main() -> None:
    cohort_path = DERIVED / "cohort.parquet"
    referral_path = DERIVED / "referral.parquet"
    if not cohort_path.exists() or not referral_path.exists():
        log.error("Required cohort or referral data not found")
        return

    cohort = pd.read_parquet(cohort_path)
    referral = pd.read_parquet(referral_path)
    table = generate_felipe_table(cohort, referral)

    out_file = DERIVED / "felipe_patient_table.parquet"
    table.to_parquet(out_file, index=False)
    log.info(f"Felipe patient table saved to {out_file}")


if __name__ == "__main__":
    main()
