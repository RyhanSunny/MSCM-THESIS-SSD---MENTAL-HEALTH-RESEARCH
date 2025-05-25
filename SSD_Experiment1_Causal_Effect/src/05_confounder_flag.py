#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_confounder_flag.py – Confounder/Covariate Matrix Building

- Loads the validated cohort and health data
- Extracts baseline confounders for causal adjustment
- Saves confounder_flag.parquet to data_derived/

HYPOTHESIS MAPPING:
This script supports ALL hypotheses (H1-H6) and RQ by:
- Extracting and preparing confounding variables for causal adjustment
- Providing covariates for propensity score matching
- Ensuring valid causal inference through proper confounder control
- Supporting stratified analyses and sensitivity checks
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
        logging.FileHandler("05_confounder_flag.log", mode="w")
    ])
log = logging.getLogger("confounder")

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
OUT_PATH = ROOT / 'data_derived' / 'confounders.parquet'

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

# Load additional data for baseline confounders
log.info("Loading additional data for confounder extraction")
health_condition = load_table("health_condition")
encounter = load_table("encounter", ["DateCreated"])
medication = load_table("medication", ["StartDate", "StopDate"])

# Keep only cohort patients
patient_ids = set(cohort.Patient_ID)
health_condition = health_condition[health_condition.Patient_ID.isin(patient_ids)]
encounter = encounter[encounter.Patient_ID.isin(patient_ids)]
medication = medication[medication.Patient_ID.isin(patient_ids)]

# Define baseline period (-12 to -6 months before index date)
cohort["baseline_start"] = cohort.IndexDate_lab - pd.Timedelta(days=365)
cohort["baseline_end"] = cohort.IndexDate_lab - pd.Timedelta(days=180)

# Initialize confounder dataframe with existing cohort variables
log.info("Building confounder matrix...")
confounders = cohort[['Patient_ID', 'Sex', 'BirthYear', 'Age_at_2018', 
                     'Charlson', 'LongCOVID_flag', 'NYD_count']].copy()

# Convert Sex to binary (assuming M/F)
confounders['male'] = (confounders['Sex'] == 'M').astype(int)
confounders.drop('Sex', axis=1, inplace=True)

# Age categories
confounders['age_18_34'] = ((confounders['Age_at_2018'] >= 18) & 
                            (confounders['Age_at_2018'] < 35)).astype(int)
confounders['age_35_49'] = ((confounders['Age_at_2018'] >= 35) & 
                            (confounders['Age_at_2018'] < 50)).astype(int)
confounders['age_50_64'] = ((confounders['Age_at_2018'] >= 50) & 
                            (confounders['Age_at_2018'] < 65)).astype(int)
confounders['age_65_plus'] = (confounders['Age_at_2018'] >= 65).astype(int)

# 1. Baseline Healthcare Utilization
log.info("Calculating baseline healthcare utilization...")
encounter = encounter.merge(cohort[["Patient_ID", "baseline_start", "baseline_end"]], 
                          on="Patient_ID", how="inner")

baseline_encounters = encounter[
    (encounter.DateCreated >= encounter.baseline_start) & 
    (encounter.DateCreated <= encounter.baseline_end)
]

# Count baseline encounters
baseline_enc_counts = baseline_encounters.groupby("Patient_ID").size()
baseline_enc_counts.name = "baseline_encounters"
confounders = confounders.merge(baseline_enc_counts.to_frame(), 
                              left_on="Patient_ID", right_index=True, how="left")
confounders["baseline_encounters"] = confounders["baseline_encounters"].fillna(0)

# High utilizer flag (top quartile)
high_util_threshold = confounders["baseline_encounters"].quantile(0.75)
confounders["baseline_high_utilizer"] = (
    confounders["baseline_encounters"] >= high_util_threshold
).astype(int)

# 2. Pre-existing Mental Health Conditions
log.info("Identifying pre-existing mental health conditions...")
mental_health_codes = ['296', '300', '311', 'F32', 'F33', 'F40', 'F41', 'F43']
mh_pattern = '|'.join([f'^{code}' for code in mental_health_codes])
mh_patients = health_condition[
    health_condition.DiagnosisCode_calc.str.match(mh_pattern, na=False)
]['Patient_ID'].unique()
confounders['mental_health_dx'] = confounders.Patient_ID.isin(mh_patients).astype(int)

# Depression specifically
depression_codes = ['296.2', '296.3', '311', 'F32', 'F33']
dep_pattern = '|'.join([f'^{code}' for code in depression_codes])
dep_patients = health_condition[
    health_condition.DiagnosisCode_calc.str.match(dep_pattern, na=False)
]['Patient_ID'].unique()
confounders['depression_dx'] = confounders.Patient_ID.isin(dep_patients).astype(int)

# Anxiety specifically
anxiety_codes = ['300', 'F40', 'F41']
anx_pattern = '|'.join([f'^{code}' for code in anxiety_codes])
anx_patients = health_condition[
    health_condition.DiagnosisCode_calc.str.match(anx_pattern, na=False)
]['Patient_ID'].unique()
confounders['anxiety_dx'] = confounders.Patient_ID.isin(anx_patients).astype(int)

# 3. Chronic Disease Flags
log.info("Identifying chronic diseases...")

# Diabetes
diabetes_codes = ['250', 'E10', 'E11']
diab_pattern = '|'.join([f'^{code}' for code in diabetes_codes])
diab_patients = health_condition[
    health_condition.DiagnosisCode_calc.str.match(diab_pattern, na=False)
]['Patient_ID'].unique()
confounders['diabetes'] = confounders.Patient_ID.isin(diab_patients).astype(int)

# Hypertension
htn_codes = ['401', 'I10']
htn_pattern = '|'.join([f'^{code}' for code in htn_codes])
htn_patients = health_condition[
    health_condition.DiagnosisCode_calc.str.match(htn_pattern, na=False)
]['Patient_ID'].unique()
confounders['hypertension'] = confounders.Patient_ID.isin(htn_patients).astype(int)

# COPD/Asthma
resp_codes = ['490', '491', '492', '493', '496', 'J40', 'J41', 'J42', 'J43', 'J44', 'J45']
resp_pattern = '|'.join([f'^{code}' for code in resp_codes])
resp_patients = health_condition[
    health_condition.DiagnosisCode_calc.str.match(resp_pattern, na=False)
]['Patient_ID'].unique()
confounders['respiratory_disease'] = confounders.Patient_ID.isin(resp_patients).astype(int)

# 4. Medication Use Indicators
log.info("Calculating baseline medication use...")

# Baseline medications
medication = medication.merge(cohort[["Patient_ID", "baseline_start", "baseline_end"]], 
                            on="Patient_ID", how="inner")

baseline_meds = medication[
    (medication.StartDate <= medication.baseline_end) & 
    (medication.StopDate >= medication.baseline_start)
]

# Count unique medications
med_counts = baseline_meds.groupby("Patient_ID")["Code_calc"].nunique()
med_counts.name = "baseline_med_count"
confounders = confounders.merge(med_counts.to_frame(), 
                              left_on="Patient_ID", right_index=True, how="left")
confounders["baseline_med_count"] = confounders["baseline_med_count"].fillna(0)

# Polypharmacy flag (>5 meds)
confounders["baseline_polypharmacy"] = (confounders["baseline_med_count"] > 5).astype(int)

# Specific medication classes
# Antidepressants
antidep_codes = ['N06A']
antidep_pattern = '|'.join([f'^{code}' for code in antidep_codes])
antidep_users = baseline_meds[
    baseline_meds.Code_calc.str.match(antidep_pattern, na=False)
]['Patient_ID'].unique()
confounders['baseline_antidepressant'] = confounders.Patient_ID.isin(antidep_users).astype(int)

# Opioids
opioid_codes = ['N02A']
opioid_pattern = '|'.join([f'^{code}' for code in opioid_codes])
opioid_users = baseline_meds[
    baseline_meds.Code_calc.str.match(opioid_pattern, na=False)
]['Patient_ID'].unique()
confounders['baseline_opioid'] = confounders.Patient_ID.isin(opioid_users).astype(int)

# 5. Healthcare access proxy (rural/urban could be derived from postal code if available)
# For now, use encounter frequency as proxy
confounders['low_access'] = (confounders["baseline_encounters"] < 2).astype(int)

# 6. Create interaction terms for key variables
confounders['age_male_interaction'] = confounders['Age_at_2018'] * confounders['male']
confounders['charlson_age_interaction'] = confounders['Charlson'] * confounders['Age_at_2018']

# Summary of confounders
log.info("\nConfounder summary:")
confounder_vars = [col for col in confounders.columns if col not in ['Patient_ID', 'BirthYear']]
log.info(f"Total confounder variables: {len(confounder_vars)}")

# Check balance between exposed/unexposed (if exposure data available)
try:
    exposure = pd.read_parquet(ROOT / 'data_derived' / 'exposure.parquet')
    confounders_exp = confounders.merge(exposure[['Patient_ID', 'exposure_flag']], 
                                      on='Patient_ID', how='left')
    
    log.info("\nPre-matching standardized mean differences:")
    smds = []
    for var in confounder_vars:
        if var in ['Age_at_2018', 'Charlson', 'baseline_encounters', 'baseline_med_count']:
            # Continuous variables
            exposed_mean = confounders_exp[confounders_exp.exposure_flag == 1][var].mean()
            unexposed_mean = confounders_exp[confounders_exp.exposure_flag == 0][var].mean()
            pooled_std = np.sqrt(
                (confounders_exp[confounders_exp.exposure_flag == 1][var].var() +
                 confounders_exp[confounders_exp.exposure_flag == 0][var].var()) / 2
            )
            smd = abs(exposed_mean - unexposed_mean) / pooled_std if pooled_std > 0 else 0
        else:
            # Binary variables
            exposed_prop = confounders_exp[confounders_exp.exposure_flag == 1][var].mean()
            unexposed_prop = confounders_exp[confounders_exp.exposure_flag == 0][var].mean()
            smd = abs(exposed_prop - unexposed_prop) / np.sqrt(
                (exposed_prop * (1 - exposed_prop) + unexposed_prop * (1 - unexposed_prop)) / 2
            )
        
        smds.append(smd)
        if smd > 0.1:  # Flag imbalanced variables
            log.info(f"  {var}: SMD = {smd:.3f} *")
    
    # Calculate maximum SMD from actual data
    max_smd = max(smds) if smds else 0.0
    log.info(f"\nMaximum pre-weight SMD: {max_smd:.3f}")
    
except FileNotFoundError:
    log.warning("Exposure data not found - skipping balance checks")
    max_smd = None

# Save confounders
log.info(f"\nSaving confounder data to {OUT_PATH}")
confounders.to_parquet(OUT_PATH, index=False)
log.info(f"Saved {len(confounders):,} rows with {len(confounders.columns)} columns")

# Update study documentation
import subprocess
try:
    result = subprocess.run([
        sys.executable, 
        str(ROOT / "scripts" / "update_study_doc.py"),
        "--step", "Confounder matrix generated",
        "--kv", f"artefact=confounders.parquet",
        "--kv", f"n_patients={len(confounders)}",
        "--kv", f"covariates={len(confounder_vars)}",
        "--kv", f"max_pre_weight_smd={max_smd:.3f}" if max_smd else "max_pre_weight_smd=NA",
        "--kv", "hypotheses=H1,H2,H3,H4,H5,H6,RQ",
        "--kv", f"script=05_confounder_flag.py",
        "--kv", "status=implemented"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")