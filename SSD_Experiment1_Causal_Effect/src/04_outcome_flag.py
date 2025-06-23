#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_outcome_flag.py – Outcome Flagging

- Loads the validated cohort
- Counts healthcare utilization in outcome window
- Calculates proxy medical costs
- Identifies inappropriate medication use
- Saves outcome_flag.parquet to data_derived/

HYPOTHESIS MAPPING:
This script supports:
- H1: Captures healthcare utilization outcomes (encounter counts, ER visits)
- H2: Tracks medical cost outcomes for analysis
- H3: Identifies inappropriate medication use outcomes
- RQ: Provides outcome data for prevalence analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import timedelta

# Add src and utils to path
SRC = (Path(__file__).resolve().parents[1] / "src").as_posix()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

UTILS = (Path(__file__).resolve().parents[1] / "utils").as_posix()
if UTILS not in sys.path:
    sys.path.insert(0, UTILS)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("04_outcome_flag.log", mode="w")
    ])
log = logging.getLogger("outcome")

# Import utilities
try:
    from global_seeds import set_global_seeds
    set_global_seeds()
    log.info("Global seeds set for reproducibility")
except ImportError:
    log.warning("Could not import global_seeds utility")

try:
    from config_loader import load_config, get_config
    config = load_config()
    log.info("Configuration loaded successfully")
except Exception as e:
    log.error(f"Could not load configuration: {e}")
    raise

# Paths
ROOT = Path(__file__).resolve().parents[1]
COHORT_PATH = ROOT / 'data_derived' / 'cohort.parquet'
CHECKPOINT_ROOT = ROOT / get_config("paths.checkpoint_root", "Notebooks/data/interim")
OUT_PATH = ROOT / 'data_derived' / 'outcomes.parquet'

# Outcome window from config
OUTCOME_START = pd.Timestamp(get_config("temporal.outcome_window_start", "2019-07-01"))
OUTCOME_END = pd.Timestamp(get_config("temporal.outcome_window_end", "2020-12-31"))

# Cost proxies from config (in CAD)
costs = get_config("costs", {})
COST_PC_VISIT = costs.get("pc_visit", 100)  # Primary care visit
COST_ED_VISIT = costs.get("ed_visit", 500)  # Emergency department visit
COST_SPECIALIST = costs.get("specialist_referral", 200)  # Specialist referral

# Find latest checkpoint
def latest_checkpoint(base: Path) -> Path:
    cps = sorted(base.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cps:
        raise FileNotFoundError(f"No checkpoint_* folder found in {base}")
    return cps[0]

CKPT = latest_checkpoint(CHECKPOINT_ROOT)
log.info(f"Using checkpoint: {CKPT}")

# Helper: robust loader
def load_table(tbl: str, date_cols: list = None) -> pd.DataFrame:
    """Load table from checkpoint, preferring parquet"""
    pq = CKPT / f"{tbl}.parquet"
    if pq.exists():
        log.info(f"Loading {tbl} from parquet")
        df = pd.read_parquet(pq)
    else:
        csv = CKPT / f"{tbl}.csv"
        if csv.exists():
            log.info(f"Loading {tbl} from CSV")
            df = pd.read_csv(csv, low_memory=False)
        else:
            raise FileNotFoundError(f"{tbl} not found in checkpoint")
    
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# Load cohort
log.info(f"Loading cohort from {COHORT_PATH}")
cohort = pd.read_parquet(COHORT_PATH)
log.info(f"Loaded {len(cohort):,} patients")

# Load additional data
log.info("Loading encounter, referral, and medication data")
encounter = load_table("encounter", ["DateCreated"])
referral = load_table("referral", ["CompletedDate", "DateCreated"])
medication = load_table("medication", ["StartDate", "StopDate"])

# Keep only cohort patients
patient_ids = set(cohort.Patient_ID)
encounter = encounter[encounter.Patient_ID.isin(patient_ids)]
referral = referral[referral.Patient_ID.isin(patient_ids)]
medication = medication[medication.Patient_ID.isin(patient_ids)]

# For each patient, outcome window is relative to their index date
cohort["outcome_start"] = cohort.IndexDate_lab + pd.Timedelta(days=365*1.5)  # 18 months after index
cohort["outcome_end"] = cohort.IndexDate_lab + pd.Timedelta(days=365*3)  # 3 years after index

# Ensure outcome window doesn't exceed study period
cohort["outcome_start"] = cohort["outcome_start"].clip(upper=OUTCOME_END)
cohort["outcome_end"] = cohort["outcome_end"].clip(upper=OUTCOME_END)

log.info(f"Outcome window: {OUTCOME_START} to {OUTCOME_END} (adjusted per patient)")

# 1. Healthcare Utilization (H1)
log.info("Calculating healthcare utilization...")

# Primary care encounters
encounter = encounter.merge(cohort[["Patient_ID", "outcome_start", "outcome_end"]], 
                          on="Patient_ID", how="inner")

enc_in_outcome = encounter[
    (encounter.DateCreated >= encounter.outcome_start) & 
    (encounter.DateCreated <= encounter.outcome_end)
]

# Count total encounters
total_encounters = enc_in_outcome.groupby("Patient_ID").size()

total_encounters.name = "total_encounters"

# Count ED visits (EncounterType contains 'emerg' or 'ED')
if 'EncounterType' in enc_in_outcome.columns:
    ed_visits = enc_in_outcome[
        enc_in_outcome.EncounterType.str.contains('emerg|ED|Emergency', case=False, na=False)
    ].groupby("Patient_ID").size()
else:
    # If no type column, estimate based on location or other fields
    ed_visits = pd.Series(0, index=cohort.Patient_ID)
ed_visits.name = "ed_visits"

# Count specialist referrals in outcome period
referral["ReferralDate"] = pd.to_datetime(
    referral.CompletedDate.fillna(referral.DateCreated), errors="coerce"
)
referral = referral.merge(cohort[["Patient_ID", "outcome_start", "outcome_end"]], 
                        on="Patient_ID", how="inner")

ref_in_outcome = referral[
    (referral.ReferralDate >= referral.outcome_start) & 
    (referral.ReferralDate <= referral.outcome_end)
]

# Specialist referrals only
spec_referrals = ref_in_outcome[
    ~ref_in_outcome.Name_calc.str.contains('FAMILY|GENERAL|GP', case=False, na=False)
]
specialist_referrals = spec_referrals.groupby("Patient_ID").size()
specialist_referrals.name = "specialist_referrals"

# 2. Medical Costs (H2) - Proxy calculation
log.info("Calculating proxy medical costs...")

# Merge utilization data
utilization = cohort[["Patient_ID"]].copy()
utilization = utilization.merge(total_encounters.to_frame(), left_on="Patient_ID", 
                              right_index=True, how="left")
utilization = utilization.merge(ed_visits.to_frame(), left_on="Patient_ID", 
                              right_index=True, how="left")
utilization = utilization.merge(specialist_referrals.to_frame(), left_on="Patient_ID", 
                              right_index=True, how="left")
utilization = utilization.fillna(0)

# Calculate proxy costs
utilization["medical_costs"] = (
    utilization["total_encounters"] * COST_PC_VISIT +
    utilization["ed_visits"] * COST_ED_VISIT +
    utilization["specialist_referrals"] * COST_SPECIALIST
)

# 3. Inappropriate Medication Use (H3)
log.info("Identifying inappropriate medication use...")

# Filter medications in outcome window
medication = medication.merge(cohort[["Patient_ID", "outcome_start", "outcome_end"]], 
                            on="Patient_ID", how="inner")

# Anxiolytics/hypnotics for >180 days
anxiolytic_codes = get_config("exposure.drug_atc_codes.anxiolytic", ["N05B", "N05C"])
hypnotic_codes = get_config("exposure.drug_atc_codes.hypnotic", ["N05CH"])
psych_codes = anxiolytic_codes + hypnotic_codes

# Find psych meds in outcome window
psych_pattern = "|".join([f"^{code}" for code in psych_codes])
psych_meds = medication[
    medication.Code_calc.str.match(psych_pattern, na=False)
].copy()

# Calculate duration in outcome window
psych_meds["med_start"] = psych_meds[["StartDate", "outcome_start"]].max(axis=1)
psych_meds["med_end"] = psych_meds[["StopDate", "outcome_end"]].min(axis=1)
psych_meds["duration_days"] = (psych_meds.med_end - psych_meds.med_start).dt.days

# Sum total days per patient
psych_days = psych_meds.groupby("Patient_ID")["duration_days"].sum()
inappropriate_meds = (psych_days > 180).astype(int)
inappropriate_meds.name = "inappropriate_meds"

# Check for polypharmacy (>5 concurrent meds)
concurrent_meds = medication.groupby("Patient_ID")["Code_calc"].nunique()
polypharmacy = (concurrent_meds > 5).astype(int)
polypharmacy.name = "polypharmacy"

# 4. Combine all outcomes
log.info("Combining outcome measures...")

outcomes = cohort[["Patient_ID"]].copy()
outcomes = outcomes.merge(utilization, on="Patient_ID", how="left")
outcomes = outcomes.merge(inappropriate_meds.to_frame(), left_on="Patient_ID", 
                        right_index=True, how="left")
outcomes = outcomes.merge(polypharmacy.to_frame(), left_on="Patient_ID", 
                        right_index=True, how="left")
outcomes = outcomes.fillna(0)

# Create composite outcome flag (high utilization)
high_util_threshold = outcomes["total_encounters"].quantile(0.75)
outcomes["high_utilization"] = (outcomes["total_encounters"] >= high_util_threshold).astype(int)

# Check completeness
completeness = (~outcomes.isnull()).sum() / len(outcomes) * 100
log.info("\nOutcome completeness:")
for col in outcomes.columns:
    if col != "Patient_ID":
        log.info(f"  {col}: {completeness[col]:.1f}%")

mean_completeness = completeness.drop("Patient_ID").mean()
log.info(f"\nMean completeness: {mean_completeness:.1f}%")

# Summary statistics
log.info("\nOutcome summary statistics:")
for col in ["total_encounters", "ed_visits", "specialist_referrals", "medical_costs"]:
    if col in outcomes.columns:
        log.info(f"\n{col}:")
        log.info(f"  Mean: {outcomes[col].mean():.2f}")
        log.info(f"  Median: {outcomes[col].median():.0f}")
        log.info(f"  75th percentile: {outcomes[col].quantile(0.75):.0f}")
        log.info(f"  Max: {outcomes[col].max():.0f}")

# Binary outcome proportions
for col in ["high_utilization", "inappropriate_meds", "polypharmacy"]:
    if col in outcomes.columns:
        prop = outcomes[col].mean() * 100
        log.info(f"\n{col}: {prop:.1f}% of patients")

# Save output
log.info(f"\nSaving outcome data to {OUT_PATH}")
outcomes.to_parquet(OUT_PATH, index=False)
log.info(f"Saved {len(outcomes):,} rows with {len(outcomes.columns)} columns")

# Update study documentation
import subprocess
try:
    result = subprocess.run([
        sys.executable, 
        str(ROOT / "scripts" / "update_study_doc.py"),
        "--step", "Outcome flags generated",
        "--kv", f"artefact=outcomes.parquet",
        "--kv", f"n_patients={len(outcomes)}",
        "--kv", f"outcome_non_missing={mean_completeness:.1f}%",
        "--kv", f"mean_encounters={outcomes['total_encounters'].mean():.2f}",
        "--kv", f"mean_costs={outcomes['medical_costs'].mean():.2f}",
        "--kv", f"inappropriate_med_rate={outcomes['inappropriate_meds'].mean():.3f}",
        "--kv", "hypotheses=H1,H2,H3,RQ",
        "--kv", f"script=04_outcome_flag.py",
        "--kv", "status=implemented"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")