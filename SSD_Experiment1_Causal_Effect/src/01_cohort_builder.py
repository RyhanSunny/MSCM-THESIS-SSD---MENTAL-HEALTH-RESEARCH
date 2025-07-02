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
from datetime import datetime, timedelta
import locale

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
medication       = load("medication",            date_cols=["StartDate", "StopDate", "PrescriptionDate"])

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
# 4  Hierarchical Index Dates - Addressing 28.3% missing lab dates
# --------------------------------------------------------------------------- #
# References:
# - Cleveland Clinic (2023): "May avoid doctor... or seek repeated reassurance"
# - DSM-5-TR (2022): Persistence >6 months requires temporal anchor
# - Hernán & Robins (2016): Target trial emulation requires clear index event
# - van der Feltz-Cornelis et al. (2022): Different phenotypes in SSD

log.info("Creating hierarchical index dates for all patients")

# 1. Lab index (primary for test-seeking phenotype)
idx_lab = lab.groupby("Patient_ID")["PerformedDate"].min().rename("IndexDate_lab")
elig = elig.merge(idx_lab, left_on="Patient_ID", right_index=True, how="left")

# 2. First mental health encounter as alternative index
mh_encounters = encounter[
    encounter['Encounter_ID'].isin(
        enc_diag[enc_diag.DiagnosisCode_calc.str.match(r'^(29[0-9]|3[0-3][0-9])', na=False)]['Encounter_ID']
    )
]
idx_mh = mh_encounters.groupby('Patient_ID')['DateCreated'].min().rename('IndexDate_mh') 
elig = elig.merge(idx_mh, left_on="Patient_ID", right_index=True, how="left")

# 3. First psychotropic prescription ≥180 days (DSM-5 B-criteria proxy)
psychotropic_atc = ['N05', 'N06']  # Anxiolytics, antidepressants
psych_meds = medication[medication.Code_calc.str.startswith(tuple(psychotropic_atc), na=False)]

# Calculate duration for each prescription
psych_meds['duration_days'] = (
    pd.to_datetime(psych_meds['StopDate'], errors='coerce') - 
    pd.to_datetime(psych_meds['StartDate'], errors='coerce')
).dt.days.fillna(30)  # Default 30 days if missing

# Find patients with ≥180 days total psychotropic use
psych_duration = psych_meds.groupby('Patient_ID')['duration_days'].sum()
long_psych = psych_duration[psych_duration >= 180].index

# Get first prescription date for qualifying patients
first_psych = psych_meds[psych_meds.Patient_ID.isin(long_psych)].groupby('Patient_ID')['StartDate'].min()
idx_psych = first_psych.rename('IndexDate_psych')
elig = elig.merge(idx_psych, left_on="Patient_ID", right_index=True, how="left")

# 4. Create unified index date using hierarchical logic
elig['IndexDate_unified'] = elig['IndexDate_lab'].fillna(
    elig['IndexDate_mh'].fillna(
        elig['IndexDate_psych']
    )
)

# 5. Track index date source
elig['index_date_source'] = np.select(
    [
        elig['IndexDate_lab'].notna(),
        elig['IndexDate_mh'].notna(), 
        elig['IndexDate_psych'].notna()
    ],
    ['Laboratory', 'Mental_Health_Encounter', 'Psychotropic_Medication'],
    default='No_Index'
)

# 6. Create phenotype indicators (van der Feltz-Cornelis et al., 2022)
elig['lab_utilization_phenotype'] = np.where(
    elig['IndexDate_lab'].isna(),
    'Avoidant_SSD',      # Expected ~28.3%
    'Test_Seeking_SSD'   # Expected ~71.7%
)

# Log distribution
source_counts = elig['index_date_source'].value_counts()
phenotype_counts = elig['lab_utilization_phenotype'].value_counts()

log.info(f"Index date sources:")
for source, count in source_counts.items():
    log.info(f"  {source}: {count:,} ({count/len(elig)*100:.1f}%)")

log.info(f"\nPhenotype distribution:")
for phenotype, count in phenotype_counts.items():
    log.info(f"  {phenotype}: {count:,} ({count/len(elig)*100:.1f}%)")

# Ensure all patients have an index date
no_index = elig[elig['index_date_source'] == 'No_Index']
if len(no_index) > 0:
    log.warning(f"WARNING: {len(no_index)} patients have no index date")
    # For these patients, use first encounter as fallback
    first_enc = encounter.groupby('Patient_ID')['DateCreated'].min()
    elig.loc[elig['index_date_source'] == 'No_Index', 'IndexDate_unified'] = \
        elig.loc[elig['index_date_source'] == 'No_Index', 'Patient_ID'].map(first_enc)
    elig.loc[elig['index_date_source'] == 'No_Index', 'index_date_source'] = 'First_Encounter_Fallback'

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
            "SpanMonths", "IndexDate_lab", "IndexDate_unified", 
            "index_date_source", "lab_utilization_phenotype",
            "Charlson", "LongCOVID_flag", "NYD_count"]
elig = elig[cols_out]

# Original cohort building complete - now applying Felipe's enhancements

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

# ------------------------------------------------------------------ #
#  NYD Enhancement Functions (Dr. Felipe's Suggestions)
# ------------------------------------------------------------------ #
def create_nyd_body_part_mapping():
    """
    Create validated NYD ICD code to body part mapping
    Based on ICD-9 780-789 range (Symptoms, Signs, and Ill-defined Conditions)
    """
    log.info("Creating validated NYD body part mapping...")
    
    # Clinically validated ICD-9 codes 780-789
    nyd_mapping = {
        # General/Systemic symptoms (780-789)
        '780.0': 'General',  # Alteration of consciousness
        '780.1': 'General',  # Hallucinations
        '780.2': 'General',  # Syncope and collapse
        '780.3': 'General',  # Convulsions
        '780.4': 'Neurological',  # Dizziness and giddiness
        '780.5': 'General',  # Sleep disturbances
        '780.6': 'General',  # Fever and other physiologic disturbances
        '780.7': 'General',  # Malaise and fatigue
        '780.8': 'General',  # Generalized hyperhidrosis
        '780.9': 'General',  # Other general symptoms
        '780.93': 'General', # Memory loss
        '780.94': 'General', # Early satiety
        '780.95': 'General', # Excessive crying
        '780.96': 'General', # Generalized pain
        '780.97': 'General', # Altered mental status
        
        # Symptoms involving nervous system (781)
        '781.0': 'Neurological',  # Abnormal involuntary movements
        '781.1': 'Neurological',  # Disturbances of sensation of smell/taste
        '781.2': 'Neurological',  # Abnormality of gait
        '781.3': 'Neurological',  # Lack of coordination
        '781.4': 'Neurological',  # Transient paralysis of limb
        '781.5': 'Neurological',  # Clubbing of fingers
        '781.6': 'Neurological',  # Meningismus
        '781.7': 'Neurological',  # Tetany
        '781.8': 'Neurological',  # Neurologic neglect syndrome
        '781.9': 'Neurological',  # Other symptoms nervous system
        
        # Symptoms involving skin (782)
        '782.0': 'Dermatological',  # Disturbance of skin sensation
        '782.1': 'Dermatological',  # Rash and other nonspecific skin eruption
        '782.2': 'Dermatological',  # Localized superficial swelling
        '782.3': 'Dermatological',  # Edema
        '782.4': 'Dermatological',  # Jaundice, unspecified
        '782.5': 'Dermatological',  # Cyanosis
        '782.6': 'Dermatological',  # Pallor and flushing
        '782.7': 'Dermatological',  # Spontaneous ecchymoses
        '782.8': 'Dermatological',  # Changes in skin texture
        '782.9': 'Dermatological',  # Other symptoms skin/integumentary
        
        # Symptoms concerning nutrition/metabolism (783)
        '783.0': 'General',  # Anorexia
        '783.1': 'General',  # Abnormal weight gain
        '783.2': 'General',  # Abnormal loss of weight
        '783.3': 'General',  # Feeding difficulties and mismanagement
        '783.4': 'General',  # Lack of expected normal physiological development
        '783.5': 'General',  # Polydipsia
        '783.6': 'General',  # Polyphagia
        '783.7': 'General',  # Adult failure to thrive
        '783.9': 'General',  # Other symptoms nutrition/metabolism
        
        # Symptoms involving head and neck (784)
        '784.0': 'Neurological',  # Headache
        '784.1': 'Respiratory',   # Throat pain
        '784.2': 'Respiratory',   # Swelling, mass, or lump in head and neck
        '784.3': 'Neurological',  # Aphasia
        '784.4': 'Neurological',  # Voice disturbance
        '784.5': 'Neurological',  # Other speech disturbance
        '784.6': 'Neurological',  # Other symbolic dysfunction
        '784.7': 'Neurological',  # Epistaxis
        '784.8': 'Neurological',  # Hemorrhage from throat
        '784.9': 'Neurological',  # Other symptoms head and neck
        
        # Symptoms involving cardiovascular system (785)
        '785.0': 'Cardiovascular',  # Tachycardia
        '785.1': 'Cardiovascular',  # Palpitations
        '785.2': 'Cardiovascular',  # Undiagnosed cardiac murmurs
        '785.3': 'Cardiovascular',  # Other abnormal heart sounds
        '785.4': 'Cardiovascular',  # Gangrene
        '785.5': 'Cardiovascular',  # Shock
        '785.6': 'Cardiovascular',  # Enlargement of lymph nodes
        '785.9': 'Cardiovascular',  # Other symptoms circulatory system
        
        # Symptoms involving respiratory system (786)
        '786.0': 'Respiratory',  # Dyspnea and respiratory abnormalities
        '786.1': 'Respiratory',  # Stridor
        '786.2': 'Respiratory',  # Cough
        '786.3': 'Respiratory',  # Hemoptysis
        '786.4': 'Respiratory',  # Abnormal sputum
        '786.5': 'Respiratory',  # Chest pain
        '786.50': 'Respiratory', # Chest pain, unspecified
        '786.51': 'Respiratory', # Precordial pain
        '786.52': 'Respiratory', # Painful respiration
        '786.59': 'Respiratory', # Other chest pain
        '786.6': 'Respiratory',  # Swelling, mass, or lump in chest
        '786.7': 'Respiratory',  # Abnormal chest sounds
        '786.8': 'Respiratory',  # Hiccough
        '786.9': 'Respiratory',  # Other symptoms respiratory system
        
        # Symptoms involving digestive system (787)
        '787.0': 'Gastrointestinal',  # Nausea and vomiting
        '787.1': 'Gastrointestinal',  # Heartburn
        '787.2': 'Gastrointestinal',  # Dysphagia
        '787.3': 'Gastrointestinal',  # Flatulence, eructation, and gas pain
        '787.4': 'Gastrointestinal',  # Visible peristalsis
        '787.5': 'Gastrointestinal',  # Abnormal bowel sounds
        '787.6': 'Gastrointestinal',  # Incontinence of feces
        '787.7': 'Gastrointestinal',  # Abnormal feces
        '787.9': 'Gastrointestinal',  # Other symptoms digestive system
        '787.91': 'Gastrointestinal', # Diarrhea
        
        # Symptoms involving urinary system (788)
        '788.0': 'Genitourinary',  # Renal colic
        '788.1': 'Genitourinary',  # Dysuria
        '788.2': 'Genitourinary',  # Retention of urine
        '788.3': 'Genitourinary',  # Urinary incontinence
        '788.4': 'Genitourinary',  # Frequency of urination and polyuria
        '788.5': 'Genitourinary',  # Oliguria and anuria
        '788.6': 'Genitourinary',  # Other abnormality of urination
        '788.7': 'Genitourinary',  # Urethral discharge
        '788.8': 'Genitourinary',  # Extravasation of urine
        '788.9': 'Genitourinary',  # Other symptoms urinary system
        
        # Other symptoms and signs (789)
        '789.0': 'Gastrointestinal',  # Abdominal pain
        '789.1': 'General',           # Hepatomegaly
        '789.2': 'General',           # Splenomegaly
        '789.3': 'Gastrointestinal',  # Abdominal or pelvic swelling, mass, or lump
        '789.4': 'Gastrointestinal',  # Abdominal rigidity
        '789.5': 'Gastrointestinal',  # Ascites
        '789.6': 'Gastrointestinal',  # Abdominal tenderness
        '789.7': 'Gastrointestinal',  # Colic
        '789.9': 'Gastrointestinal',  # Other symptoms abdomen/pelvis
        
        # Mental/Behavioral V codes related to symptoms
        'V71.0': 'Mental/Behavioral',  # Observation for suspected mental condition
        'V71.09': 'Mental/Behavioral', # Observation for other suspected mental condition
    }
    
    log.info(f"Validated NYD mapping created: {len(nyd_mapping)} codes across {len(set(nyd_mapping.values()))} body systems")
    
    # Save mapping for clinical reference
    mapping_df = pd.DataFrame([
        {'icd_code': code, 'body_part': body_part, 'clinical_category': 'NYD_symptoms'}
        for code, body_part in nyd_mapping.items()
    ])
    
    mapping_path = ROOT / "code_lists" / "nyd_body_part_mapping_validated.csv"
    mapping_path.parent.mkdir(exist_ok=True)
    mapping_df.to_csv(mapping_path, index=False)
    log.info(f"Clinical mapping saved: {mapping_path}")
    
    return nyd_mapping

def load_real_nyd_data():
    """Load real NYD diagnosis records from encounter_diagnosis"""
    log.info("Loading real NYD diagnosis records from encounter data...")
    
    # Load encounter diagnosis data from checkpoint
    enc_diag = pd.read_parquet(CKPT / "encounter_diagnosis.parquet")
    
    # NYD codes (780-789 range) - clinically validated
    nyd_pattern = r'^(78[0-9]|799|V71\.0)'
    nyd_diagnoses = enc_diag[
        enc_diag['DiagnosisCode_calc'].str.match(nyd_pattern, na=False)
    ].copy()
    
    # Get patient IDs and ICD codes with encounter context
    nyd_data = nyd_diagnoses[['Patient_ID', 'DiagnosisCode_calc', 'Encounter_ID']].copy()
    nyd_data = nyd_data.rename(columns={'DiagnosisCode_calc': 'ICD_code'})
    nyd_data = nyd_data.drop_duplicates()
    
    log.info(f"Real NYD data loaded: {len(nyd_data):,} records for {nyd_data['Patient_ID'].nunique():,} patients")
    
    return nyd_data

def add_nyd_enhancements(cohort_df):
    """
    Add NYD enhancements to cohort based on REAL patient data
    """
    log.info("Adding NYD enhancements to main cohort...")
    
    # Deep copy to preserve datetime columns
    enhanced_cohort = cohort_df.copy(deep=True)
    
    # Load real NYD data
    nyd_data = load_real_nyd_data()
    
    # Filter to cohort patients only
    cohort_patients = set(cohort_df['Patient_ID'])
    nyd_data = nyd_data[nyd_data['Patient_ID'].isin(cohort_patients)]
    
    # Create mapping
    nyd_mapping = create_nyd_body_part_mapping()
    
    if len(nyd_data) == 0:
        log.info("No NYD data found for cohort - setting all flags to 0")
        body_part_flags = ['NYD_yn', 'NYD_general_yn', 'NYD_mental_yn', 'NYD_neuro_yn', 
                          'NYD_cardio_yn', 'NYD_resp_yn', 'NYD_gi_yn', 'NYD_musculo_yn', 
                          'NYD_derm_yn', 'NYD_gu_yn']
        for flag in body_part_flags:
            enhanced_cohort[flag] = 0
        return enhanced_cohort
    
    # Map ICD codes to body parts
    nyd_data = nyd_data.copy()
    nyd_data['body_part'] = nyd_data['ICD_code'].map(nyd_mapping).fillna('Unknown')
    
    # Calculate overall NYD binary flag
    patients_with_nyd = set(nyd_data['Patient_ID'].unique())
    enhanced_cohort['NYD_yn'] = enhanced_cohort['Patient_ID'].isin(patients_with_nyd).astype(int)
    
    # Calculate body part-specific flags
    body_part_mapping = {
        'General': 'NYD_general_yn',
        'Mental/Behavioral': 'NYD_mental_yn', 
        'Neurological': 'NYD_neuro_yn',
        'Cardiovascular': 'NYD_cardio_yn',
        'Respiratory': 'NYD_resp_yn',
        'Gastrointestinal': 'NYD_gi_yn',
        'Musculoskeletal': 'NYD_musculo_yn',
        'Dermatological': 'NYD_derm_yn',
        'Genitourinary': 'NYD_gu_yn'
    }
    
    for body_part, column_name in body_part_mapping.items():
        patients_with_body_part = set(
            nyd_data[nyd_data['body_part'] == body_part]['Patient_ID'].unique()
        )
        enhanced_cohort[column_name] = enhanced_cohort['Patient_ID'].isin(patients_with_body_part).astype(int)
    
    # Calculate NYD counts for existing compatibility (preserve original NYD_count)
    nyd_counts = nyd_data.groupby('Patient_ID').size().rename('NYD_count_enhanced')
    enhanced_cohort = enhanced_cohort.merge(nyd_counts, left_on='Patient_ID', right_index=True, how='left')
    enhanced_cohort['NYD_count_enhanced'] = enhanced_cohort['NYD_count_enhanced'].fillna(0)
    
    # Update the original NYD_count with enhanced count if it's higher
    if 'NYD_count' in enhanced_cohort.columns:
        enhanced_cohort['NYD_count'] = enhanced_cohort[['NYD_count', 'NYD_count_enhanced']].max(axis=1)
    else:
        enhanced_cohort['NYD_count'] = enhanced_cohort['NYD_count_enhanced']
    
    # Drop the temporary column
    enhanced_cohort = enhanced_cohort.drop(columns=['NYD_count_enhanced'], errors='ignore')
    
    # Log summary statistics for REAL data
    nyd_count = enhanced_cohort['NYD_yn'].sum()
    total_patients = len(enhanced_cohort)
    log.info(f"NYD enhancements added to main cohort:")
    log.info(f"  Total cohort size: {total_patients:,} patients")
    log.info(f"  Patients with NYD diagnoses: {nyd_count:,} ({nyd_count/total_patients*100:.1f}%)")
    
    for body_part, column_name in body_part_mapping.items():
        count = enhanced_cohort[column_name].sum()
        if count > 0:
            log.info(f"  {body_part}: {count:,} patients ({count/total_patients*100:.1f}%)")
    
    return enhanced_cohort

# ------------------------------------------------------------------ #
#  Enhanced execution with Felipe's NYD suggestions
# ------------------------------------------------------------------ #

# Add Felipe's NYD enhancements to the existing cohort
log.info("=== APPLYING FELIPE'S NYD ENHANCEMENTS ===")
log.info(f"Columns in elig before enhancement: {list(elig.columns)}")
log.info(f"IndexDate_unified dtype: {elig['IndexDate_unified'].dtype if 'IndexDate_unified' in elig.columns else 'NOT FOUND'}")
log.info(f"Sample IndexDate_unified values: {elig['IndexDate_unified'].head() if 'IndexDate_unified' in elig.columns else 'NOT FOUND'}")
enhanced_cohort = add_nyd_enhancements(elig)
log.info(f"Columns in enhanced_cohort after enhancement: {list(enhanced_cohort.columns)}")

# Update the final cohort with enhanced columns
age_col = f"Age_at_{REF_DATE.year}"
enhanced_cols_out = ["Patient_ID", "Sex", "BirthYear", age_col,
                    "SpanMonths", "IndexDate_lab", "IndexDate_unified", 
                    "index_date_source", "lab_utilization_phenotype",
                    "Charlson", "LongCOVID_flag", "NYD_count",
                    "NYD_yn", "NYD_general_yn", "NYD_mental_yn", "NYD_neuro_yn", 
                    "NYD_cardio_yn", "NYD_resp_yn", "NYD_gi_yn", "NYD_musculo_yn", 
                    "NYD_derm_yn", "NYD_gu_yn"]

# Ensure all enhanced columns exist (except datetime columns)
datetime_cols = ['IndexDate_lab', 'IndexDate_unified']
for col in enhanced_cols_out:
    if col not in enhanced_cohort.columns:
        if col in datetime_cols:
            log.error(f"CRITICAL: DateTime column {col} not found in enhanced cohort!")
            raise ValueError(f"Missing required datetime column: {col}")
        else:
            log.warning(f"Column {col} not found, setting to 0")
            enhanced_cohort[col] = 0

final_cohort = enhanced_cohort[enhanced_cols_out]

log.info(f"Writing ENHANCED cohort -> {OUT_FILE}  ({len(final_cohort):,} rows)")
final_cohort.to_parquet(OUT_FILE, index=False, compression="snappy")

# Log enhancement statistics
nyd_count = final_cohort['NYD_yn'].sum()
total_patients = len(final_cohort)
log.info("Felipe's NYD enhancements applied:")
log.info(f"  Total cohort: {total_patients:,} patients")
log.info(f"  NYD patients: {nyd_count:,} ({nyd_count/total_patients*100:.1f}%)")
log.info("Enhanced cohort saved with body system flags")
log.info("Done.")
