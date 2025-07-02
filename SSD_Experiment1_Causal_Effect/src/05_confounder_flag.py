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

# Load underutilized tables for comprehensive confounder adjustment
try:
    patient_demographic = load_table("patient_demographic")
    log.info(f"Loaded patient_demographic: {len(patient_demographic):,} rows")
except FileNotFoundError:
    log.warning("patient_demographic table not found - skipping social determinants")
    patient_demographic = None

try:
    family_history = load_table("family_history") 
    log.info(f"Loaded family_history: {len(family_history):,} rows")
except FileNotFoundError:
    log.warning("family_history table not found - skipping genetic risk factors")
    family_history = None

try:
    risk_factor = load_table("risk_factor", ["StartDate", "EndDate"])
    log.info(f"Loaded risk_factor: {len(risk_factor):,} rows") 
except FileNotFoundError:
    log.warning("risk_factor table not found - skipping risk factor adjustment")
    risk_factor = None

try:
    medical_procedure = load_table("medical_procedure", ["PerformedDate"])
    log.info(f"Loaded medical_procedure: {len(medical_procedure):,} rows")
except FileNotFoundError:
    log.warning("medical_procedure table not found - skipping procedure history")
    medical_procedure = None

# Keep only cohort patients
patient_ids = set(cohort.Patient_ID)
health_condition = health_condition[health_condition.Patient_ID.isin(patient_ids)]
encounter = encounter[encounter.Patient_ID.isin(patient_ids)]
medication = medication[medication.Patient_ID.isin(patient_ids)]

# Filter additional tables to cohort patients
if patient_demographic is not None:
    patient_demographic = patient_demographic[patient_demographic.Patient_ID.isin(patient_ids)]
if family_history is not None:
    family_history = family_history[family_history.Patient_ID.isin(patient_ids)]
if risk_factor is not None:
    risk_factor = risk_factor[risk_factor.Patient_ID.isin(patient_ids)]
if medical_procedure is not None:
    medical_procedure = medical_procedure[medical_procedure.Patient_ID.isin(patient_ids)]

# Define baseline period (-12 to -6 months before index date)
# Use unified index date from hierarchical implementation (addresses 28.3% missing lab dates)
if "IndexDate_unified" in cohort.columns:
    cohort["baseline_start"] = cohort.IndexDate_unified - pd.Timedelta(days=365)
    cohort["baseline_end"] = cohort.IndexDate_unified - pd.Timedelta(days=180)
    log.info("Using IndexDate_unified for baseline period (hierarchical implementation)")
else:
    # Fallback to lab index for backward compatibility
    cohort["baseline_start"] = cohort.IndexDate_lab - pd.Timedelta(days=365)
    cohort["baseline_end"] = cohort.IndexDate_lab - pd.Timedelta(days=180)
    log.warning("IndexDate_unified not found, falling back to IndexDate_lab")

# Initialize confounder dataframe with existing cohort variables
log.info("Building confounder matrix...")
confounders = cohort[['Patient_ID', 'Sex', 'BirthYear', 'Age_at_2015', 
                     'Charlson', 'LongCOVID_flag', 'NYD_count']].copy()

# Convert Sex to binary (handling various formats)
confounders['male'] = (confounders['Sex'].isin(['M', 'Male', 'MALE'])).astype(int)
confounders.drop('Sex', axis=1, inplace=True)

# Age categories
confounders['age_18_34'] = ((confounders['Age_at_2015'] >= 18) & 
                            (confounders['Age_at_2015'] < 35)).astype(int)
confounders['age_35_49'] = ((confounders['Age_at_2015'] >= 35) & 
                            (confounders['Age_at_2015'] < 50)).astype(int)
confounders['age_50_64'] = ((confounders['Age_at_2015'] >= 50) & 
                            (confounders['Age_at_2015'] < 65)).astype(int)
confounders['age_65_plus'] = (confounders['Age_at_2015'] >= 65).astype(int)

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
log.info("All patients are mental health patients (data pre-filtered)")
# Since cohort is pre-filtered for MH patients, all patients have MH diagnosis
confounders['mental_health_dx'] = 1

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

# 6. Social Determinants of Health (from patient_demographic if available)
if patient_demographic is not None:
    log.info("Adding social determinants of health...")
    
    # Merge demographic data
    demo_subset = patient_demographic[['Patient_ID', 'Occupation', 'HighestEducation', 
                                     'HousingStatus', 'Language', 'Ethnicity']].copy()
    confounders = confounders.merge(demo_subset, on='Patient_ID', how='left')
    
    # Education level
    confounders['high_education'] = (confounders['HighestEducation'].str.contains(
        'university|college|bachelor|master|phd', case=False, na=False)).astype(int)
    
    # Housing stability
    confounders['housing_unstable'] = (confounders['HousingStatus'].str.contains(
        'homeless|shelter|temp', case=False, na=False)).astype(int)
    
    # Language barrier
    confounders['language_barrier'] = (~confounders['Language'].str.contains(
        'english|français|french', case=False, na=False)).astype(int)
    
    # Drop raw categorical columns
    confounders.drop(['Occupation', 'HighestEducation', 'HousingStatus', 'Language', 'Ethnicity'], 
                    axis=1, inplace=True)

# 7. Family History Risk Factors (if available)
if family_history is not None:
    log.info("Adding family history risk factors...")
    
    # Mental health family history - use text field since code is empty
    mh_fhx_keywords = ['depression', 'anxiety', 'mental', 'psychiatric', 'bipolar', 'schizophrenia']
    mh_fhx_pattern = '|'.join(mh_fhx_keywords)
    if 'DiagnosisText_calc' in family_history.columns and family_history['DiagnosisText_calc'].dtype == 'object':
        mh_fhx_patients = family_history[
            family_history.DiagnosisText_calc.str.contains(mh_fhx_pattern, case=False, na=False)
        ]['Patient_ID'].unique()
    else:
        mh_fhx_patients = []
    confounders['family_mental_health'] = confounders.Patient_ID.isin(mh_fhx_patients).astype(int)
    
    # Cancer family history - use text field
    cancer_fhx_keywords = ['cancer', 'carcinoma', 'tumor', 'malignant', 'lymphoma', 'leukemia']
    cancer_fhx_pattern = '|'.join(cancer_fhx_keywords)
    if 'DiagnosisText_calc' in family_history.columns and family_history['DiagnosisText_calc'].dtype == 'object':
        cancer_fhx_patients = family_history[
            family_history.DiagnosisText_calc.str.contains(cancer_fhx_pattern, case=False, na=False)
        ]['Patient_ID'].unique()
    else:
        cancer_fhx_patients = []
    confounders['family_cancer'] = confounders.Patient_ID.isin(cancer_fhx_patients).astype(int)

# 8. Risk Factors (smoking, obesity, etc. if available)
if risk_factor is not None:
    log.info("Adding risk factor indicators...")
    
    # Merge baseline timeline for risk factors
    risk_factor = risk_factor.merge(cohort[["Patient_ID", "baseline_start", "baseline_end"]], 
                                  on="Patient_ID", how="inner")
    
    # Filter to baseline period
    baseline_risks = risk_factor[
        ((risk_factor.StartDate <= risk_factor.baseline_end) & 
         (risk_factor.EndDate.isna() | (risk_factor.EndDate >= risk_factor.baseline_start))) |
        (risk_factor.StartDate.isna() & risk_factor.EndDate.isna())
    ]
    
    # Smoking
    smoking_patients = baseline_risks[
        baseline_risks.Name_calc.str.contains('smok', case=False, na=False)
    ]['Patient_ID'].unique()
    confounders['smoking'] = confounders.Patient_ID.isin(smoking_patients).astype(int)
    
    # Alcohol use
    alcohol_patients = baseline_risks[
        baseline_risks.Name_calc.str.contains('alcohol', case=False, na=False)
    ]['Patient_ID'].unique()
    confounders['alcohol_use'] = confounders.Patient_ID.isin(alcohol_patients).astype(int)
    
    # Obesity/BMI
    obesity_patients = baseline_risks[
        baseline_risks.Name_calc.str.contains('obese|obesity|bmi', case=False, na=False)
    ]['Patient_ID'].unique()
    confounders['obesity'] = confounders.Patient_ID.isin(obesity_patients).astype(int)

# 9. Procedure History (if available)
if medical_procedure is not None:
    log.info("Adding procedure history indicators...")
    
    # Merge baseline timeline
    medical_procedure = medical_procedure.merge(cohort[["Patient_ID", "baseline_start", "baseline_end"]], 
                                              on="Patient_ID", how="inner")
    
    # Baseline procedures
    baseline_procedures = medical_procedure[
        (medical_procedure.PerformedDate >= medical_procedure.baseline_start) & 
        (medical_procedure.PerformedDate <= medical_procedure.baseline_end)
    ]
    
    # Count procedures
    proc_counts = baseline_procedures.groupby("Patient_ID").size()
    proc_counts.name = "baseline_procedures"
    confounders = confounders.merge(proc_counts.to_frame(), 
                                  left_on="Patient_ID", right_index=True, how="left")
    confounders["baseline_procedures"] = confounders["baseline_procedures"].fillna(0)
    
    # High procedure use (top quartile)
    high_proc_threshold = confounders["baseline_procedures"].quantile(0.75)
    confounders["high_procedure_use"] = (
        confounders["baseline_procedures"] >= high_proc_threshold
    ).astype(int)

# 10. Create interaction terms for key variables
confounders['age_male_interaction'] = confounders['Age_at_2015'] * confounders['male']
confounders['charlson_age_interaction'] = confounders['Charlson'] * confounders['Age_at_2015']

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
        if var in ['Age_at_2015', 'Charlson', 'baseline_encounters', 'baseline_med_count']:
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