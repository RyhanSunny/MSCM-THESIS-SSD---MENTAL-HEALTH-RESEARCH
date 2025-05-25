#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_cohort_builder.py  – Eligibility cohort for the SSD thesis

• Looks for the most-recent  checkpoint_* folder under
    Notebooks/data/interim/
• Loads patient, encounter, health_condition, lab, encounter_diagnosis
  (Parquet preferred; falls back to CSV automatically).
• Applies the inclusion / exclusion rules in the methods blueprint.
• Writes   data_derived/cohort.parquet

Only eligibility logic lives here – no NYD, no lab-normal counts, no SSD flag.

HYPOTHESIS MAPPING:
This script supports ALL hypotheses (H1-H6) and the Research Question by:
- Building the base cohort that all subsequent analyses depend on
- Ensuring data quality and eligibility criteria are met
- Providing the denominator for prevalence calculations (RQ)
"""

# --------------------------------------------------------------------------- #
# Imports & configuration
# --------------------------------------------------------------------------- #
from __future__ import annotations
import sys, logging, re, glob
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy  as np

# --- make sure project/src is importable --------------------------
SRC = (Path(__file__).resolve().parents[1] / "src").as_posix()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Add utils to path as well
UTILS = (Path(__file__).resolve().parents[1] / "utils").as_posix()
if UTILS not in sys.path:
    sys.path.insert(0, UTILS)
# ------------------------------------------------------------------

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("01_cohort_builder.log", mode="w")])
log = logging.getLogger("cohort")

# Import and set global seeds for reproducibility
try:
    from global_seeds import set_global_seeds
    set_global_seeds()
    log.info("Global seeds set for reproducibility")
except ImportError:
    log.warning("Could not import global_seeds utility - proceeding without seed setting")

# Import configuration
try:
    from config_loader import load_config, get_config
    config = load_config()
    log.info("Configuration loaded successfully")
except Exception as e:
    log.error(f"Could not load configuration: {e}")
    raise

# ---------- project roots ----------
ROOT       = Path(__file__).resolve().parents[1]               # project root
CHECK_ROOT = ROOT / "Notebooks" / "data" / "interim"           # where checkpoints live
DERIVED    = ROOT / "data_derived"
DERIVED.mkdir(exist_ok=True, parents=True)

OUT_FILE   = DERIVED / "cohort.parquet"

# ---------- constants from config ----------
REF_DATE        = pd.Timestamp(get_config("temporal.reference_date"))
CENSOR_DATE     = pd.Timestamp(get_config("temporal.censor_date"))
SPAN_REQ_MONTHS = get_config("cohort.min_observation_months", 30)
CHARLSON_MAX    = get_config("cohort.max_charlson_score", 5)
PALLIATIVE_CODES = get_config("cohort.palliative_codes", ["V66.7", "Z51.5"])
PALLIATIVE_RE   = re.compile("|".join(f"^{code}$" for code in PALLIATIVE_CODES))
OPT_OUT_COL     = get_config("cohort.opt_out_column", "OptedOut")

# --------------------------------------------------------------------------- #
# Helper: find newest checkpoint folder
# --------------------------------------------------------------------------- #
def latest_checkpoint(base: Path) -> Path:
    cps = sorted(base.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cps:
        raise FileNotFoundError(f"No checkpoint_* folder found in {base}")
    newest = cps[0]
    log.info(f"Using newest checkpoint folder:  {newest}")
    return newest

CKPT = latest_checkpoint(CHECK_ROOT)

# --------------------------------------------------------------------------- #
# Helper: robust loader (prefers Parquet, falls back to CSV)
# --------------------------------------------------------------------------- #
def load(tbl: str,
         date_cols: list[str] | None = None,
         usecols: list[str] | None = None) -> pd.DataFrame:
    """
    Load <tbl>.parquet or <tbl>.csv from CKPT, apply dtype fixes.
    """
    pq = CKPT / f"{tbl}.parquet"
    if pq.exists():
        log.info(f"Loading {pq.name:25s} (Parquet)")
        df = pd.read_parquet(pq, columns=usecols)
    else:
        csv = CKPT / f"{tbl}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{tbl} not found as .parquet or .csv in {CKPT}")
        log.info(f"Loading {csv.name:25s} (CSV)")
        df = pd.read_csv(csv, low_memory=False, usecols=usecols)

    # parse dates
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# 1  Load tables
# --------------------------------------------------------------------------- #
patient          = load("patient",               date_cols=None)
encounter        = load("encounter",             date_cols=["DateCreated"])
health_condition = load("health_condition",      date_cols=None)
lab              = load("lab",                   date_cols=["PerformedDate"])
enc_diag         = load("encounter_diagnosis",   date_cols=None)

log.info("Tables loaded.\n"
         f"  patient             {len(patient):,}\n"
         f"  encounter           {len(encounter):,}\n"
         f"  health_condition    {len(health_condition):,}\n"
         f"  lab                 {len(lab):,}\n"
         f"  encounter_diagnosis {len(enc_diag):,}")

# --------------------------------------------------------------------------- #
# 2  Age ≥18 on reference date
# --------------------------------------------------------------------------- #
MIN_AGE = get_config("cohort.min_age", 18)
patient[f"Age_at_{REF_DATE.year}"] = REF_DATE.year - patient["BirthYear"]
elig = patient.loc[patient[f"Age_at_{REF_DATE.year}"] >= MIN_AGE].copy()
log.info(f">={MIN_AGE} y filter: {len(elig):,} remain")

# Remove CPCSSN opt-outs
if OPT_OUT_COL in elig.columns:
    pre = len(elig)
    elig = elig[elig[OPT_OUT_COL] == 0]
    log.info(f"Opt-out exclusion: {pre-len(elig):,}")

# --------------------------------------------------------------------------- #
# 3  Determine observation span (first & last ANY record date)
# --------------------------------------------------------------------------- #
enc_dates = encounter[["Patient_ID", "DateCreated"]].dropna()
enc_dates = enc_dates.rename(columns={"DateCreated": "RecordDate"})

lab_dates = lab[["Patient_ID", "PerformedDate"]].rename(
    columns={"PerformedDate": "RecordDate"}).dropna()

all_dates = pd.concat([enc_dates, lab_dates])

# Ensure RecordDate is datetime
all_dates["RecordDate"] = pd.to_datetime(all_dates["RecordDate"], errors="coerce")

grp        = all_dates.groupby("Patient_ID")["RecordDate"]
span_month = ((grp.max() - grp.min()).dt.days / 30.44).rename("SpanMonths")
elig = elig.merge(span_month, left_on="Patient_ID", right_index=True, how="left")

pre = len(elig)
elig = elig[elig["SpanMonths"].fillna(0) >= SPAN_REQ_MONTHS]
log.info(f">={SPAN_REQ_MONTHS} mo data: excluded {pre-len(elig):,}")

# --------------------------------------------------------------------------- #
# 4  Index date = first lab within span
# --------------------------------------------------------------------------- #
idx_lab = lab.groupby("Patient_ID")["PerformedDate"].min().rename("IndexDate_lab")
elig = elig.merge(idx_lab, left_on="Patient_ID", right_index=True, how="left")

# --------------------------------------------------------------------------- #
# 5  Charlson Comorbidity Index
#      Re-use helper saved in Notebook 1
# --------------------------------------------------------------------------- #
try:
    from icd_utils import charlson_index
except ImportError:
    log.warning("icd_utils.charlson_index not found, Charlson set to 0")
    elig["Charlson"] = 0
else:
    # Calculate Charlson scores and ensure they're integers
    charlson_scores = charlson_index(health_condition)
    charlson_scores = charlson_scores.fillna(0).astype('int16')
    
    # Merge scores into eligibility dataframe
    elig = elig.merge(
        charlson_scores.rename('Charlson'),
        left_on='Patient_ID',
        right_index=True,
        how='left'
    )
    elig['Charlson'] = elig['Charlson'].fillna(0).astype('int16')
    
    # Diagnostic checks
    log.info("\nCharlson score distribution:")
    log.info(elig["Charlson"].describe(percentiles=[.5, .75, .9, .95]))
    
    # Check unique codes per patient
    tmp = health_condition.groupby("Patient_ID")["DiagnosisCode_calc"].nunique()
    log.info("\nUnique diagnosis codes per patient:")
    log.info(tmp.describe())
    
    # Verify Charlson column is integer
    log.info(f"\nCharlson column dtype: {elig['Charlson'].dtype}")
    log.info(f"Sample values: {elig['Charlson'].head().tolist()}")
    log.info(f"Value counts:\n{elig['Charlson'].value_counts().sort_index().head(10)}")

# --------------------------------------------------------------------------- #
# 6  Palliative-care exclusion & Charlson > 5
# --------------------------------------------------------------------------- #
pall_ids = health_condition.loc[
    health_condition["DiagnosisCode_calc"].str.match(PALLIATIVE_RE, na=False),
    "Patient_ID"].unique()

pre = len(elig)
elig = elig[~elig["Patient_ID"].isin(pall_ids)]
log.info(f"Palliative-care exclusion: {pre-len(elig):,}")

pre = len(elig)
elig = elig[elig["Charlson"] <= CHARLSON_MAX]
log.info(f"Charlson > {CHARLSON_MAX} exclusion: {pre-len(elig):,}")

# --------------------------------------------------------------------------- #
# 6.5  Add Long-COVID and NYD flags
# --------------------------------------------------------------------------- #
# Long-COVID flag from config
covid_codes = get_config("long_covid.icd_codes", ["U07.1", "U09.9"])
covid_patterns = get_config("long_covid.text_patterns", ["post-acute COVID", "long COVID"])
covid_code_re = re.compile("|".join(f"^{code}" for code in covid_codes))
covid_text_re = re.compile("|".join(covid_patterns), re.IGNORECASE)

covid_ids = health_condition.loc[
    health_condition["DiagnosisCode_calc"].str.match(covid_code_re, na=False) |
    health_condition["DiagnosisText_calc"].str.contains(covid_text_re, na=False),
    "Patient_ID"].unique()

elig["LongCOVID_flag"] = elig["Patient_ID"].isin(covid_ids).astype(int)
log.info(f"Long-COVID patients identified: {elig['LongCOVID_flag'].sum():,}")

# NYD flag counter from config
nyd_codes = get_config("nyd.codes", ["799.9", "V71.0", "V71.1", "V71.2", "V71.3", "V71.4", "V71.5", "V71.6", "V71.7", "V71.8", "V71.9"])
nyd_re = re.compile("|".join(f"^{code}" for code in nyd_codes))
nyd_counts = health_condition.loc[
    health_condition["DiagnosisCode_calc"].str.match(nyd_re, na=False)
].groupby("Patient_ID").size()

elig = elig.merge(nyd_counts.rename("NYD_count"), left_on="Patient_ID", right_index=True, how="left")
elig["NYD_count"] = elig["NYD_count"].fillna(0).astype(int)
log.info(f"Patients with NYD codes: {(elig['NYD_count'] > 0).sum():,}")

# --------------------------------------------------------------------------- #
# 7  Save cohort
# --------------------------------------------------------------------------- #
age_col = f"Age_at_{REF_DATE.year}"
cols_out = ["Patient_ID", "Sex", "BirthYear", age_col,
            "SpanMonths", "IndexDate_lab", "Charlson", 
            "LongCOVID_flag", "NYD_count"]
elig = elig[cols_out]

log.info(f"Writing cohort -> {OUT_FILE}  ({len(elig):,} rows)")
elig.to_parquet(OUT_FILE, index=False, compression="snappy")
log.info("Done.")

# --------------------------------------------------------------------------- #
# 8  Update study documentation
# --------------------------------------------------------------------------- #
import subprocess
try:
    result = subprocess.run([
        sys.executable, 
        str(ROOT / "scripts" / "update_study_doc.py"),
        "--step", "Cohort eligibility completed",
        "--kv", f"artefact=cohort.parquet",
        "--kv", f"n_patients={len(elig)}",
        "--kv", "hypotheses=H1,H2,H3,H4,H5,H6,RQ",
        "--kv", f"script=01_cohort_builder.py"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")
