#!/usr/bin/env python3
"""
02_exposure_flag.py – Identify exposure (somatic patterns of healthcare-seeking)

This version fixes the referral counting issue by deduplicating before counting.
"""

import pandas as pd
import numpy as np
import re
import sys
from datetime import datetime
from pathlib import Path
import logging

# Add src to path
SRC = (Path(__file__).resolve().parents[1] / "src").as_posix()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Setup paths
ROOT = Path(__file__).parent.parent
from config_loader import get_config

# Key config: exposure window
# Load checkpoint 1
CKPT = ROOT / "Notebooks" / "data" / "interim" / "checkpoint_1_20250318_024427"
if not CKPT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

# Load all checkpoint tables
log.info(f"Loading data from checkpoint: {CKPT}")
cohort = pd.read_parquet(ROOT / "data_derived" / "cohort.parquet")
lab = pd.read_csv(CKPT / "lab.csv", low_memory=False, encoding='utf-8') if (CKPT / "lab.csv").exists() else pd.read_parquet(CKPT / "lab.parquet")
referral = pd.read_parquet(CKPT / "referral.parquet")
enc_diag = pd.read_parquet(CKPT / "encounter_diagnosis.parquet")
med = pd.read_parquet(CKPT / "medication.parquet")

log.info(f"Loaded cohort: {len(cohort):,} patients")

# ------------------------------------------------------------------ #
#  2  Window boundaries
# ------------------------------------------------------------------ #
# Use exposure window from config (2015-01-01 to 2016-01-01)
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
#  4  Criterion 2 – ≥{config} symptom-referrals (FIXED)
# ------------------------------------------------------------------ #
MIN_SYMPTOM_REFERRALS = get_config("exposure.min_symptom_referrals", 2)
SYMPTOM_RE = re.compile(get_config("exposure.symptom_code_regex", r"^(78[0-9]|799)"))

# Step a: Get encounters with symptom diagnoses (deduplicated)
enc_with_symptoms = enc_diag[enc_diag.DiagnosisCode_calc.str.match(SYMPTOM_RE, na=False)]['Encounter_ID'].unique()
log.info(f"Encounters with symptom diagnoses: {len(enc_with_symptoms):,}")

# Step b: Filter referrals
referral = referral.dropna(subset=["CompletedDate", "DateCreated"])
referral["ReferralDate"] = pd.to_datetime(
    referral.CompletedDate.fillna(referral.DateCreated), errors="coerce")
referral = referral.dropna(subset=["ReferralDate"])

# Map specialty (quick upper-case)
spec = referral.Name_calc.str.upper().fillna("")
referral["is_specialist"] = ~spec.str.contains("FAMILY|GENERAL PRACTICE|GP")

# Filter to specialist referrals only
specialist_refs = referral[referral.is_specialist].copy()

# Check which referrals are associated with symptom encounters
specialist_refs["has_symptom_encounter"] = specialist_refs.Encounter_ID.isin(enc_with_symptoms)

# Merge with cohort for window filtering
specialist_refs = specialist_refs.merge(cohort[["Patient_ID", "exp_start", "exp_end"]], 
                                      on="Patient_ID", how="inner")

# Filter to exposure window and symptom encounters
qualifying_refs = specialist_refs[
    (specialist_refs.ReferralDate >= specialist_refs.exp_start) &
    (specialist_refs.ReferralDate <= specialist_refs.exp_end) &
    (specialist_refs.has_symptom_encounter)
]

# Count qualifying referrals per patient
ref_count = (qualifying_refs.groupby("Patient_ID")
            .size()
            .rename("symptom_referral_n"))

crit2 = ref_count >= MIN_SYMPTOM_REFERRALS
log.info(f"Patients meeting ≥{MIN_SYMPTOM_REFERRALS} symptom-referral rule: {crit2.sum():,}")

del referral, specialist_refs, qualifying_refs
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

# Also create strict AND logic flag for comparison
exposure["exposure_flag_strict"] = (
    exposure.crit1_normal_labs &
    exposure.crit2_sympt_ref   &
    exposure.crit3_drug_90d
)

log.info("=" * 60)
log.info("EXPOSURE FLAG SUMMARY")
log.info("=" * 60)
log.info(f"Total cohort: {len(exposure):,} patients")
log.info(f"H1 - Normal labs (≥{MIN_NORMAL_LABS}): {exposure.H1_normal_labs.sum():,} ({exposure.H1_normal_labs.mean()*100:.1f}%)")
log.info(f"H2 - Referral loops (≥{MIN_SYMPTOM_REFERRALS}): {exposure.H2_referral_loop.sum():,} ({exposure.H2_referral_loop.mean()*100:.1f}%)")
log.info(f"H3 - Drug persistence (≥{MIN_DRUG_DAYS}d): {exposure.H3_drug_persistence.sum():,} ({exposure.H3_drug_persistence.mean()*100:.1f}%)")
log.info("-" * 60)
log.info(f"Exposed (OR logic - any pattern): {exposure.exposure_flag.sum():,} ({exposure.exposure_flag.mean()*100:.1f}%)")
log.info(f"Exposed (AND logic - all patterns): {exposure.exposure_flag_strict.sum():,} ({exposure.exposure_flag_strict.mean()*100:.1f}%)")
log.info("=" * 60)

# Save
log.info(f"Saving exposure flags to data_derived/exposure.parquet...")
exposure.to_parquet(ROOT / "data_derived" / "exposure.parquet")
log.info("Done!")

# Also save timestamp
from artefact_tracker import track_artefact
track_artefact("exposure.parquet", "Exposure flags - fixed referral counting", 
               {"n_exposed_or": int(exposure.exposure_flag.sum()),
                "n_exposed_and": int(exposure.exposure_flag_strict.sum()),
                "n_h1": int(exposure.H1_normal_labs.sum()),
                "n_h2": int(exposure.H2_referral_loop.sum()),
                "n_h3": int(exposure.H3_drug_persistence.sum())})

# Generate summary report
summary = f"""
Exposure Flag Generation Report
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Cohort Size: {len(exposure):,} patients

Individual Hypotheses:
- H1 (Normal Labs ≥{MIN_NORMAL_LABS}): {exposure.H1_normal_labs.sum():,} patients ({exposure.H1_normal_labs.mean()*100:.1f}%)
- H2 (Symptom Referrals ≥{MIN_SYMPTOM_REFERRALS}): {exposure.H2_referral_loop.sum():,} patients ({exposure.H2_referral_loop.mean()*100:.1f}%)  
- H3 (Drug Days ≥{MIN_DRUG_DAYS}): {exposure.H3_drug_persistence.sum():,} patients ({exposure.H3_drug_persistence.mean()*100:.1f}%)

Combined Exposure:
- OR Logic (any pattern): {exposure.exposure_flag.sum():,} patients ({exposure.exposure_flag.mean()*100:.1f}%)
- AND Logic (all patterns): {exposure.exposure_flag_strict.sum():,} patients ({exposure.exposure_flag_strict.mean()*100:.1f}%)

Distribution of Exposure Patterns:
"""

# Count combinations
pattern_counts = exposure.groupby(['H1_normal_labs', 'H2_referral_loop', 'H3_drug_persistence']).size()
for (h1, h2, h3), count in pattern_counts.items():
    if count > 0:
        patterns = []
        if h1: patterns.append("H1")
        if h2: patterns.append("H2") 
        if h3: patterns.append("H3")
        pattern_str = "+".join(patterns) if patterns else "None"
        summary += f"- {pattern_str}: {count:,} patients ({count/len(exposure)*100:.1f}%)\n"

# Save report
with open(ROOT / "data_derived" / "exposure_report.txt", "w") as f:
    f.write(summary)

print(summary)