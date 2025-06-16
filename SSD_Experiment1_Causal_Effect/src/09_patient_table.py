#!/usr/bin/env python3
"""Generate unified patient characteristics.

This module consolidates demographics, comorbidities and SSD severity
indicators into a single table.  The layout follows suggestions from
Dr. Felipe (see meeting notes) but is written for general use.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data_derived"
OUTPUTS = ROOT / "outputs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_dataset(path: Path) -> pd.DataFrame:
    """Load a parquet dataset if present, otherwise return empty DataFrame."""
    try:
        logger.info(f"Loading {path.name}")
        return pd.read_parquet(path)
    except FileNotFoundError:
        logger.warning(f"File not found: {path}")
        return pd.DataFrame()


def create_patient_table() -> pd.DataFrame:
    """Create the patient characteristics table."""
    cohort = _load_dataset(DERIVED / "cohort.parquet")
    if cohort.empty:
        raise FileNotFoundError("cohort.parquet is required to build patient table")

    exposure = _load_dataset(DERIVED / "exposure.parquet")
    severity = _load_dataset(DERIVED / "mediator_autoencoder.parquet")
    referrals = _load_dataset(DERIVED / "referral_enhanced.parquet")
    confounders = _load_dataset(DERIVED / "confounders.parquet")

    table = pd.DataFrame()
    table["PID"] = cohort["Patient_ID"]
    # Prefer the harmonized 2015 baseline age; fall back to older 2018 label if necessary for
    # backward-compatibility with any pre-migration artefacts.
    if "Age_at_2015" in cohort.columns:
        table["age"] = cohort["Age_at_2015"]
    else:
        table["age"] = cohort.get("Age_at_2018")
    table["sex"] = cohort.get("Sex_clean", cohort.get("Sex"))
    table["NYD_yn"] = cohort.get("NYD_yn")
    table["body_part"] = cohort.get("NYD_body_part_summary")

    if not referrals.empty:
        table["referred_to_psy_yn"] = referrals.get("has_psychiatric_referral", False)
        table["referred_to_other_yn"] = referrals.get("has_medical_specialist", False)
        table["num_specialist_referrals"] = referrals.get("total_specialist_referrals", 0)
    else:
        table["referred_to_psy_yn"] = False
        table["referred_to_other_yn"] = False
        table["num_specialist_referrals"] = 0

    if not exposure.empty:
        table["SSD_flag"] = exposure.get("exposure_flag")
        for col in ["H1_normal_labs", "H2_referral_loop", "H3_drug_persistence"]:
            if col in exposure.columns:
                table[col] = exposure[col]
    else:
        table["SSD_flag"] = 0

    if not severity.empty and "SSD_severity_index" in severity.columns:
        table["SSD_severity_index"] = severity["SSD_severity_index"]

    if not confounders.empty:
        for col in [
            "Charlson",
            "LongCOVID_flag",
            "baseline_encounters",
            "baseline_med_count",
        ]:
            if col in confounders.columns:
                table[col] = confounders[col]

    return table


def summarize_patient_table(table: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics suitable for publication."""
    summary = {
        "N": len(table),
        "Age_mean": table["age"].mean(),
        "Age_std": table["age"].std(),
        "Female_pct": (table["sex"].str.lower().eq("female").mean() * 100),
        "NYD_pct": (table.get("NYD_yn", 0).mean() * 100),
        "Psych_referral_pct": (table["referred_to_psy_yn"].mean() * 100),
        "SSD_pct": (table["SSD_flag"].mean() * 100),
    }
    for col in ["H1_normal_labs", "H2_referral_loop", "H3_drug_persistence"]:
        if col in table.columns:
            summary[f"{col}_pct"] = table[col].mean() * 100
    for col in ["baseline_encounters", "baseline_med_count"]:
        if col in table.columns:
            summary[f"{col}_mean"] = table[col].mean()
    if "SSD_severity_index" in table.columns:
        summary["Severity_mean"] = table["SSD_severity_index"].mean()
    return pd.DataFrame([summary])


def main() -> None:
    table = create_patient_table()
    DERIVED.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    table_path = DERIVED / "patient_characteristics.parquet"
    table.to_parquet(table_path, index=False)
    logger.info(f"Patient table saved to {table_path}")

    summary = summarize_patient_table(table)
    summary_path = OUTPUTS / "patient_characteristics_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
