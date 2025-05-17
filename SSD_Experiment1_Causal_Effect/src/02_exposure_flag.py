#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_exposure_flag.py  – "SSD-pattern" treatment indicator

The flag is **1** if ALL of the following are true in the 12-month *exposure*
window (index-date → index-date + 365 d):

1. ≥ 3 laboratory results within normal limits  
2. ≥ 2 specialist referrals whose *final* diagnosis remains in ICD-9 780–789  
3. ≥ 90 consecutive days of prescription coverage for
   • anxiolytic (ATC N05B / N05C) OR
   • non-opioid analgesic (N02B) OR
   • non-benzo hypnotic (N05CH, e.g. z-drugs)

Outputs  
`data_derived/exposure.parquet`
"""

from __future__ import annotations
import sys, logging, re
from pathlib import Path
from datetime import timedelta
import locale

import pandas as pd
import numpy  as np

# Set locale to UTF-8
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        pass

# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #
# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("02_exposure_flag.log", mode="w", encoding='utf-8')
    ])
log = logging.getLogger("exposure")

# ------------------------------------------------------------------ #
#  Project paths
# ------------------------------------------------------------------ #
ROOT        = Path(__file__).resolve().parents[1]
DERIVED     = ROOT / "data_derived"
CHECKPOINTS = ROOT / "Notebooks" / "data" / "interim"
OUT_PATH    = DERIVED / "exposure.parquet"

try:
    DERIVED.mkdir(exist_ok=True, parents=True)
except Exception as e:
    log.error(f"Failed to create directory {DERIVED}: {e}")
    sys.exit(1)

# Latest checkpoint folder
try:
    CKPT = max(CHECKPOINTS.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime)
    log.info(f"Using checkpoint → {CKPT.name}")
except Exception as e:
    log.error(f"Failed to find checkpoint directory: {e}")
    sys.exit(1)

# ------------------------------------------------------------------ #
#  Helper – robust loader
# ------------------------------------------------------------------ #
def load(tbl: str, date_cols: list[str] | None = None) -> pd.DataFrame:
    try:
        pq = CKPT / f"{tbl}.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
        else:
            csv_path = CKPT / f"{tbl}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Neither {pq} nor {csv_path} exists")
            df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8')
        if date_cols:
            for c in date_cols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
    except Exception as e:
        log.error(f"Failed to load {tbl}: {e}")
        raise

# ------------------------------------------------------------------ #
#  1  Load cohort + source tables
# ------------------------------------------------------------------ #
cohort   = pd.read_parquet(DERIVED / "cohort.parquet")
lab      = load("lab",      ["PerformedDate"])
referral = load("referral", ["CompletedDate", "DateCreated"])
med      = load("medication", ["StartDate", "StopDate"])
enc_diag = load("encounter_diagnosis")

#  Keep only cohort patients in every table (speed)
keep = set(cohort.Patient_ID)
lab      = lab[lab.Patient_ID.isin(keep)]
referral = referral[referral.Patient_ID.isin(keep)]
med      = med[med.Patient_ID.isin(keep)]
enc_diag = enc_diag[enc_diag.Patient_ID.isin(keep)]

# ------------------------------------------------------------------ #
#  2  Window boundaries
# ------------------------------------------------------------------ #
# cohort has per-patient IndexDate_lab (already calculated)
cohort["exp_start"] = cohort.IndexDate_lab
cohort["exp_end"]   = cohort.IndexDate_lab + pd.Timedelta(days=365)

# ------------------------------------------------------------------ #
#  3  Criterion 1 – ≥3 normal labs
# ------------------------------------------------------------------ #
# Normal-lab detection from Notebook 1
if "is_normal" not in lab.columns:
    raise RuntimeError("Column is_normal missing – run Notebook 1 first!")

# Process lab data in chunks if it's a CSV
if isinstance(lab, str) and lab.endswith('.csv'):
    log.info("Processing lab data in chunks...")
    chunk_size = 100000  # Adjust based on available memory
    normal_counts = pd.Series(0, index=cohort.Patient_ID)
    
    for chunk in pd.read_csv(CKPT / "lab.csv", 
                           chunksize=chunk_size,
                           low_memory=False,
                           encoding='utf-8'):
        # Process each chunk
        chunk = chunk.dropna(subset=["PerformedDate"])
        chunk["PerformedDate"] = pd.to_datetime(chunk.PerformedDate, errors="coerce")
        chunk = chunk.merge(cohort[["Patient_ID", "exp_start", "exp_end"]],
                          on="Patient_ID", how="inner")
        
        chunk_in_window = chunk[(chunk.PerformedDate >= chunk.exp_start) &
                              (chunk.PerformedDate <= chunk.exp_end)]
        
        if "is_normal" in chunk_in_window.columns:
            chunk_counts = (chunk_in_window.groupby("Patient_ID")["is_normal"]
                          .apply(lambda x: (x==True).sum()))
            normal_counts = normal_counts.add(chunk_counts, fill_value=0)
    
    norm_count = normal_counts.rename("normal_lab_count")
else:
    # Original processing for parquet files
    lab = lab.dropna(subset=["PerformedDate"])
    lab = lab.merge(cohort[["Patient_ID", "exp_start", "exp_end"]],
                   on="Patient_ID", how="inner")
    lab_in_window = lab[(lab.PerformedDate >= lab.exp_start) &
                       (lab.PerformedDate <= lab.exp_end)]
    
    norm_count = (lab_in_window.groupby("Patient_ID")["is_normal"]
                 .apply(lambda x: (x==True).sum())
                 .rename("normal_lab_count"))

crit1 = norm_count >= 3
log.info(f"Patients meeting ≥3 normal-lab rule: {crit1.sum():,}")

# Clean up memory
del lab
import gc
gc.collect()

# ------------------------------------------------------------------ #
#  4  Criterion 2 – ≥2 symptom-referrals
# ------------------------------------------------------------------ #
SYMPTOM_RE = re.compile(r"^(78[0-9]|799)")

# Step a: Filter encounter_diagnosis first to reduce memory usage
enc_diag_filtered = enc_diag[enc_diag.DiagnosisCode_calc.str.match(SYMPTOM_RE, na=False)][["Encounter_ID", "DiagnosisCode_calc"]]
enc_diag_filtered["DiagnosisCode_calc"] = enc_diag_filtered["DiagnosisCode_calc"].astype("category")
log.info(f"Filtered encounter diagnoses to {len(enc_diag_filtered):,} rows")

# Step b: Filter referral data before merge
referral = referral.dropna(subset=["CompletedDate", "DateCreated"])
referral["ReferralDate"] = pd.to_datetime(
    referral.CompletedDate.fillna(referral.DateCreated), errors="coerce")
referral = referral.dropna(subset=["ReferralDate"])
referral = referral[["Patient_ID", "Encounter_ID", "Name_calc", "ReferralDate"]]
referral["Name_calc"] = referral["Name_calc"].astype("category")

# Map specialty (quick upper-case)
spec = referral.Name_calc.str.upper().fillna("")
referral["is_specialist"] = ~spec.str.contains("FAMILY|GENERAL PRACTICE|GP")

# Process referral in chunks to avoid memory error
def process_referral_chunks(referral_df, enc_diag_filtered, cohort):
    chunk_size = 100000
    results = []
    for start in range(0, len(referral_df), chunk_size):
        chunk = referral_df.iloc[start:start+chunk_size].copy()
        chunk = chunk.merge(enc_diag_filtered, on="Encounter_ID", how="inner")
        chunk["symptom_dx"] = chunk.DiagnosisCode_calc.str.match(SYMPTOM_RE, na=False)
        chunk["qualifies"] = chunk.is_specialist & chunk.symptom_dx
        chunk = chunk.merge(cohort[["Patient_ID", "exp_start", "exp_end"]], on="Patient_ID", how="inner")
        in_win = chunk[(chunk.ReferralDate >= chunk.exp_start) &
                       (chunk.ReferralDate <= chunk.exp_end) &
                       (chunk.qualifies)]
        results.append(in_win[["Patient_ID"]])
        del chunk, in_win
        import gc
        gc.collect()
    if results:
        all_in_win = pd.concat(results, ignore_index=True)
        ref_count = all_in_win.groupby("Patient_ID").size().rename("symptom_referral_n")
    else:
        ref_count = pd.Series(dtype=int, name="symptom_referral_n")
    return ref_count

ref_count = process_referral_chunks(referral, enc_diag_filtered, cohort)
crit2 = ref_count >= 2
log.info(f"Patients meeting ≥2 symptom-referral rule: {crit2.sum():,}")

del referral, enc_diag_filtered
import gc
gc.collect()

# ------------------------------------------------------------------ #
#  5  Criterion 3 – ≥90 d prescription coverage
# ------------------------------------------------------------------ #
ATC_KEEP = (
    med.Code_calc.str.startswith(("N05B", "N05C", "N02B", "N05CH"), na=False) |
    med.Name_calc.str.contains(
        r"ZOPICLONE|ZOLPIDEM|BUSPIRONE|BENZODIAZEPINE|GABAPENTIN",
        case=False, na=False)
)

med = med[ATC_KEEP].copy()
med["StartDate"] = pd.to_datetime(med.StartDate, errors="coerce")
med["StopDate"]  = pd.to_datetime(med.StopDate,  errors="coerce")

# Replace missing StopDate with StartDate+30 d (one refill)
med["StopDate"] = med.StopDate.fillna(med.StartDate + pd.Timedelta(days=30))

med = med.merge(cohort[["Patient_ID", "exp_start", "exp_end"]],
                on="Patient_ID")

# Clip to exposure window
med["clip_start"] = med[["StartDate", "exp_start"]].max(axis=1)
med["clip_stop"]  = med[["StopDate",  "exp_end"]].min(axis=1)
med["days"] = (med.clip_stop - med.clip_start).dt.days.clip(lower=0)

drug_days = (med.groupby("Patient_ID")["days"].sum()
             .rename("drug_days_in_window"))
crit3 = drug_days >= 90
log.info(f"Patients with ≥90 d qualifying Rx: {crit3.sum():,}")

# ------------------------------------------------------------------ #
#  6  Put it together
# ------------------------------------------------------------------ #
exposure = cohort[["Patient_ID"]].copy()
exposure = (exposure
            .merge(norm_count,              how="left")
            .merge(ref_count,               how="left")
            .merge(drug_days,               how="left")
            .fillna(0))

exposure["crit1_normal_labs"]   = exposure.normal_lab_count   >= 3
exposure["crit2_sympt_ref"]     = exposure.symptom_referral_n >= 2
exposure["crit3_drug_90d"]      = exposure.drug_days_in_window >= 90

exposure["exposure_flag"] = (
    exposure.crit1_normal_labs &
    exposure.crit2_sympt_ref   &
    exposure.crit3_drug_90d
)

log.info(f"Final exposure-positive patients: {exposure.exposure_flag.sum():,}"
         f" / {len(exposure):,}  "
         f"({exposure.exposure_flag.mean():.2%})")

# ------------------------------------------------------------------ #
#  7  Save
# ------------------------------------------------------------------ #
cols_keep = ["Patient_ID",
             "exposure_flag",
             "normal_lab_count",
             "symptom_referral_n",
             "drug_days_in_window"]

exposure[cols_keep].to_parquet(OUT_PATH, index=False, compression="snappy")
log.info(f"Wrote {OUT_PATH}")
