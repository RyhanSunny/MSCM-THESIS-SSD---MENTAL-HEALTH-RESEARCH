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

HYPOTHESIS MAPPING:
This script directly supports:
- H1, H2, H3: The exposure flag identifies patients with SSD patterns which are
  the treatment group for testing effects on healthcare utilization and costs
- H4, H5, H6: Provides the exposure variable for mediation analyses through
  psychological factors, health anxiety, and physician factors
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

# Add src and utils to path
SRC = (Path(__file__).resolve().parents[1] / "src").as_posix()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

UTILS = (Path(__file__).resolve().parents[1] / "utils").as_posix()
if UTILS not in sys.path:
    sys.path.insert(0, UTILS)

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

# Import and set global seeds
try:
    from global_seeds import set_global_seeds
    set_global_seeds()
    log.info("Global seeds set for reproducibility")
except ImportError:
    log.warning("Could not import global_seeds utility")

# Import configuration
try:
    from config_loader import load_config, get_config
    config = load_config()
    log.info("Configuration loaded successfully")
except Exception as e:
    log.error(f"Could not load configuration: {e}")
    raise

# Import lab utilities
try:
    from helpers.lab_utils import is_normal_lab, add_normal_flags
    log.info("Lab utilities imported successfully")
except ImportError as e:
    log.warning(f"Could not import lab utilities: {e} - will use existing is_normal column")

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
# Use exposure window from config (2018-01-01 to 2019-01-01)
exp_window_start = pd.Timestamp(get_config("temporal.exposure_window_start"))
exp_window_end = pd.Timestamp(get_config("temporal.exposure_window_end"))

# For each patient, exposure window is relative to their index date
cohort["exp_start"] = cohort.IndexDate_lab
cohort["exp_end"]   = cohort.IndexDate_lab + pd.Timedelta(days=365)

log.info(f"Exposure window: {exp_window_start} to {exp_window_end} (relative to index date)")

# ------------------------------------------------------------------ #
#  3  Criterion 1 – ≥{config} normal labs
# ------------------------------------------------------------------ #
MIN_NORMAL_LABS = get_config("exposure.min_normal_labs", 3)

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

crit1 = norm_count >= MIN_NORMAL_LABS
log.info(f"Patients meeting ≥{MIN_NORMAL_LABS} normal-lab rule: {crit1.sum():,}")

# Clean up memory
del lab
import gc
gc.collect()

# ------------------------------------------------------------------ #
#  4  Criterion 2 – ≥{config} symptom-referrals
# ------------------------------------------------------------------ #
MIN_SYMPTOM_REFERRALS = get_config("exposure.min_symptom_referrals", 2)
SYMPTOM_RE = re.compile(get_config("exposure.symptom_code_regex", r"^(78[0-9]|799)"))

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
crit2 = ref_count >= MIN_SYMPTOM_REFERRALS
log.info(f"Patients meeting ≥{MIN_SYMPTOM_REFERRALS} symptom-referral rule: {crit2.sum():,}")

del referral, enc_diag_filtered
import gc
gc.collect()

# ------------------------------------------------------------------ #
#  5  Criterion 3 – ≥{config} d prescription coverage
# ------------------------------------------------------------------ #
MIN_DRUG_DAYS = get_config("exposure.min_drug_days", 90)

# Load drug ATC codes from config
drug_atc_config = get_config("exposure.drug_atc_codes", {})
all_atc_codes = []
for drug_class, codes in drug_atc_config.items():
    all_atc_codes.extend(codes)

# Also load from drug_atc.csv if available
drug_atc_path = ROOT / 'code_lists' / 'drug_atc.csv'
if drug_atc_path.exists():
    log.info(f"Loading drug ATC codes from {drug_atc_path}")
    drug_atc_df = pd.read_csv(drug_atc_path)
    # Add any codes from CSV that aren't in config
    csv_codes = drug_atc_df['atc_code'].unique().tolist()
    all_atc_codes.extend([code for code in csv_codes if code not in all_atc_codes])
    log.info(f"Total ATC codes: {len(all_atc_codes)}")

# Drug name patterns from config
drug_name_patterns = get_config("exposure.drug_name_patterns", [])
drug_name_regex = "|".join(drug_name_patterns) if drug_name_patterns else None

# Filter medications
atc_conditions = [med.Code_calc.str.startswith(code, na=False) for code in all_atc_codes]
ATC_KEEP = pd.concat(atc_conditions, axis=1).any(axis=1)

if drug_name_regex:
    ATC_KEEP = ATC_KEEP | med.Name_calc.str.contains(drug_name_regex, case=False, na=False)

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
crit3 = drug_days >= MIN_DRUG_DAYS
log.info(f"Patients with ≥{MIN_DRUG_DAYS} d qualifying Rx: {crit3.sum():,}")

# ------------------------------------------------------------------ #
#  6  Put it together
# ------------------------------------------------------------------ #
exposure = cohort[["Patient_ID"]].copy()
exposure = (exposure
            .merge(norm_count,              left_on="Patient_ID", right_index=True, how="left")
            .merge(ref_count,               left_on="Patient_ID", right_index=True, how="left")
            .merge(drug_days,               left_on="Patient_ID", right_index=True, how="left")
            .fillna(0))

exposure["crit1_normal_labs"]   = exposure.normal_lab_count   >= MIN_NORMAL_LABS
exposure["crit2_sympt_ref"]     = exposure.symptom_referral_n >= MIN_SYMPTOM_REFERRALS
exposure["crit3_drug_90d"]      = exposure.drug_days_in_window >= MIN_DRUG_DAYS

# Create separate exposure flags for each hypothesis
exposure["H1_normal_labs"] = exposure.crit1_normal_labs
exposure["H2_referral_loop"] = exposure.crit2_sympt_ref  
exposure["H3_drug_persistence"] = exposure.crit3_drug_90d

# Combined exposure flag - OR logic (any pattern qualifies)
exposure["exposure_flag"] = (
    exposure.crit1_normal_labs |
    exposure.crit2_sympt_ref   |
    exposure.crit3_drug_90d
)

# Also create AND version for comparison
exposure["exposure_flag_strict"] = (
    exposure.crit1_normal_labs &
    exposure.crit2_sympt_ref   &
    exposure.crit3_drug_90d
)

log.info(f"Individual criteria summary:")
log.info(f"  H1 (≥{MIN_NORMAL_LABS} normal labs): {exposure.H1_normal_labs.sum():,} patients ({exposure.H1_normal_labs.mean():.1%})")
log.info(f"  H2 (≥{MIN_SYMPTOM_REFERRALS} symptom referrals): {exposure.H2_referral_loop.sum():,} patients ({exposure.H2_referral_loop.mean():.1%})")
log.info(f"  H3 (≥{MIN_DRUG_DAYS} drug days): {exposure.H3_drug_persistence.sum():,} patients ({exposure.H3_drug_persistence.mean():.1%})")

log.info(f"Combined exposure (OR logic): {exposure.exposure_flag.sum():,}"
         f" / {len(exposure):,} ({exposure.exposure_flag.mean():.1%})")
log.info(f"Strict exposure (AND logic): {exposure.exposure_flag_strict.sum():,}"
         f" / {len(exposure):,} ({exposure.exposure_flag_strict.mean():.1%})")

# ------------------------------------------------------------------ #
#  7  Save
# ------------------------------------------------------------------ #
cols_keep = ["Patient_ID",
             "exposure_flag",
             "exposure_flag_strict", 
             "H1_normal_labs",
             "H2_referral_loop",
             "H3_drug_persistence",
             "normal_lab_count",
             "symptom_referral_n",
             "drug_days_in_window"]

exposure[cols_keep].to_parquet(OUT_PATH, index=False, compression="snappy")
log.info(f"Wrote {OUT_PATH}")

# ------------------------------------------------------------------ #
#  8  Update study documentation
# ------------------------------------------------------------------ #
import subprocess
try:
    result = subprocess.run([
        sys.executable, 
        str(ROOT / "scripts" / "update_study_doc.py"),
        "--step", "Exposure flag generated",
        "--kv", f"artefact=exposure.parquet",
        "--kv", f"n_exposed={exposure.exposure_flag.sum()}",
        "--kv", f"pct_exposed={exposure.exposure_flag.mean():.2%}",
        "--kv", "hypotheses=H1,H2,H3,H4,H5,H6",
        "--kv", f"script=02_exposure_flag.py"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")
