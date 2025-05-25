#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_mediator_autoencoder.py – SSD Severity Index

- Loads the validated cohort and health data
- Builds a sparse autoencoder for SSD severity
- Saves mediator_autoencoder.parquet to data_derived/

HYPOTHESIS MAPPING:
This script supports:
- H4: Generates the psychological distress mediator variable for testing
  mediation of SSD effects through psychological factors
- H5: Provides severity index that correlates with health anxiety patterns
- H6: Creates mediator data for analyzing physician-patient interaction effects
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import warnings
warnings.filterwarnings('ignore')

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
        logging.FileHandler("03_mediator_autoencoder.log", mode="w")
    ])
log = logging.getLogger("mediator")

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
EXPOSURE_PATH = ROOT / 'data_derived' / 'exposure.parquet'
CHECKPOINT_ROOT = ROOT / get_config("paths.checkpoint_root", "Notebooks/data/interim")
OUT_PATH = ROOT / 'data_derived' / 'mediator_autoencoder.parquet'
FEATURE_LIST_PATH = ROOT / 'code_lists' / 'ae56_features.csv'

# Autoencoder settings from config
INPUT_DIM = get_config("autoencoder.input_features", 56)
ENCODING_DIM = get_config("autoencoder.encoding_dim", 16)
HIDDEN_DIM = get_config("autoencoder.hidden_dim", 32)
REGULARIZATION = get_config("autoencoder.regularization", 1e-5)
EPOCHS = get_config("autoencoder.epochs", 100)
BATCH_SIZE = get_config("autoencoder.batch_size", 256)
VALIDATION_SPLIT = get_config("autoencoder.validation_split", 0.2)
EARLY_STOPPING_PATIENCE = get_config("autoencoder.early_stopping_patience", 10)

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

# Load data
log.info("Loading cohort and exposure data")
cohort = pd.read_parquet(COHORT_PATH)
exposure = pd.read_parquet(EXPOSURE_PATH)

# Merge cohort with exposure
df = cohort.merge(exposure[['Patient_ID', 'exposure_flag', 'normal_lab_count', 
                           'symptom_referral_n', 'drug_days_in_window']], 
                  on='Patient_ID', how='left')

log.info(f"Loaded {len(df):,} patients")

# Load additional data for features
log.info("Loading additional data for feature engineering")
health_condition = load_table("health_condition")
encounter = load_table("encounter", ["DateCreated"])
referral = load_table("referral", ["CompletedDate", "DateCreated"])
medication = load_table("medication", ["StartDate", "StopDate"])

# Keep only cohort patients
patient_ids = set(df.Patient_ID)
health_condition = health_condition[health_condition.Patient_ID.isin(patient_ids)]
encounter = encounter[encounter.Patient_ID.isin(patient_ids)]
referral = referral[referral.Patient_ID.isin(patient_ids)]
medication = medication[medication.Patient_ID.isin(patient_ids)]

# Feature engineering for autoencoder
log.info("Engineering features for autoencoder")

# 1. Symptom code features (780-789)
symptom_codes = []
for code in range(780, 790):
    code_pattern = f"^{code}"
    code_counts = health_condition[
        health_condition.DiagnosisCode_calc.str.match(code_pattern, na=False)
    ].groupby('Patient_ID').size()
    code_counts.name = f'symptom_{code}_count'
    symptom_codes.append(code_counts)

symptom_features = pd.concat(symptom_codes, axis=1).fillna(0)

# 2. Visit pattern features
# Calculate visit counts in 6-month windows
encounter['DateCreated'] = pd.to_datetime(encounter['DateCreated'])
six_months_ago = encounter['DateCreated'].max() - pd.Timedelta(days=180)
recent_visits = encounter[encounter['DateCreated'] >= six_months_ago].groupby('Patient_ID').size()
recent_visits.name = 'visit_count_6m'

# 3. Referral patterns
referral_counts = referral.groupby('Patient_ID').size()
referral_counts.name = 'referral_count'

# Specialist referrals
spec_referrals = referral[~referral.Name_calc.str.contains('FAMILY|GENERAL|GP', case=False, na=False)]
spec_referral_counts = spec_referrals.groupby('Patient_ID').size()
spec_referral_counts.name = 'specialist_referral_count'

# 4. Psychological indicators
psych_codes = ['300', '296', 'F32', 'F33', 'F40', 'F41', 'F43']
psych_pattern = '|'.join([f'^{code}' for code in psych_codes])
psych_dx = health_condition[
    health_condition.DiagnosisCode_calc.str.match(psych_pattern, na=False)
].groupby('Patient_ID')['DiagnosisCode_calc'].nunique()
psych_dx.name = 'psych_dx_count'

# Anxiety specific
anxiety_codes = ['300', 'F40', 'F41']
anxiety_pattern = '|'.join([f'^{code}' for code in anxiety_codes])
anxiety_flag = health_condition[
    health_condition.DiagnosisCode_calc.str.match(anxiety_pattern, na=False)
]['Patient_ID'].unique()

# 5. Medication patterns
# Anxiolytic use
anxiolytic_codes = get_config("exposure.drug_atc_codes.anxiolytic", ["N05B", "N05C"])
anx_pattern = '|'.join([f'^{code}' for code in anxiolytic_codes])
anxiolytic_users = medication[
    medication.Code_calc.str.match(anx_pattern, na=False)
]['Patient_ID'].unique()

# 6. Pain-related features
pain_codes = ['338', '780.96', 'M79', 'R52']
pain_pattern = '|'.join([f'^{code}' for code in pain_codes])
pain_dx = health_condition[
    health_condition.DiagnosisCode_calc.str.match(pain_pattern, na=False)
].groupby('Patient_ID').size()
pain_dx.name = 'pain_dx_count'

# 7. GI symptoms
gi_codes = ['787', 'K58', 'K59', 'R10']
gi_pattern = '|'.join([f'^{code}' for code in gi_codes])
gi_dx = health_condition[
    health_condition.DiagnosisCode_calc.str.match(gi_pattern, na=False)
].groupby('Patient_ID').size()
gi_dx.name = 'gi_symptom_count'

# 8. Fatigue/malaise
fatigue_codes = ['780.7', 'R53']
fatigue_pattern = '|'.join([f'^{code}' for code in fatigue_codes])
fatigue_dx = health_condition[
    health_condition.DiagnosisCode_calc.str.match(fatigue_pattern, na=False)
].groupby('Patient_ID').size()
fatigue_dx.name = 'fatigue_count'

# Combine all features
log.info("Combining features into feature matrix")
feature_dfs = [
    symptom_features,
    recent_visits.to_frame(),
    referral_counts.to_frame(),
    spec_referral_counts.to_frame(),
    psych_dx.to_frame(),
    pain_dx.to_frame(),
    gi_dx.to_frame(),
    fatigue_dx.to_frame()
]

# Add binary flags
df['anxiety_flag'] = df.Patient_ID.isin(anxiety_flag).astype(int)
df['anxiolytic_flag'] = df.Patient_ID.isin(anxiolytic_users).astype(int)

# Merge all features
for feat_df in feature_dfs:
    df = df.merge(feat_df, left_on='Patient_ID', right_index=True, how='left')

# Fill missing values
df = df.fillna(0)

# Select final features for autoencoder
feature_cols = [col for col in df.columns if any([
    col.startswith('symptom_'),
    col.endswith('_count'),
    col.endswith('_flag'),
    col in ['normal_lab_count', 'symptom_referral_n', 'drug_days_in_window']
])]

# Ensure we have appropriate number of features
if len(feature_cols) > INPUT_DIM:
    # Select top features by variance
    variances = df[feature_cols].var()
    feature_cols = variances.nlargest(INPUT_DIM).index.tolist()
elif len(feature_cols) < INPUT_DIM:
    # Add additional features or pad
    log.warning(f"Only {len(feature_cols)} features available, less than target {INPUT_DIM}")
    INPUT_DIM = len(feature_cols)

log.info(f"Selected {len(feature_cols)} features for autoencoder")

# Save feature list
feature_df = pd.DataFrame({'feature_name': feature_cols})
feature_df.to_csv(FEATURE_LIST_PATH, index=False)
log.info(f"Saved feature list to {FEATURE_LIST_PATH}")

# Prepare data for autoencoder
X = df[feature_cols].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for training
X_train, X_val = train_test_split(X_scaled, test_size=VALIDATION_SPLIT, 
                                  random_state=get_config("random_state.global_seed", 42))

log.info(f"Training data shape: {X_train.shape}")
log.info(f"Validation data shape: {X_val.shape}")

# Build sparse autoencoder
log.info("Building sparse autoencoder model")

# Input layer
input_layer = keras.Input(shape=(INPUT_DIM,))

# Encoder
encoded = layers.Dense(HIDDEN_DIM, activation='relu',
                      activity_regularizer=regularizers.l1(REGULARIZATION))(input_layer)
encoded = layers.BatchNormalization()(encoded)
encoded = layers.Dropout(0.2)(encoded)
encoded = layers.Dense(ENCODING_DIM, activation='relu',
                      activity_regularizer=regularizers.l1(REGULARIZATION))(encoded)

# Decoder
decoded = layers.Dense(HIDDEN_DIM, activation='relu')(encoded)
decoded = layers.BatchNormalization()(decoded)
decoded = layers.Dropout(0.2)(decoded)
decoded = layers.Dense(INPUT_DIM, activation='sigmoid')(decoded)

# Create models
autoencoder = keras.Model(input_layer, decoded)
encoder = keras.Model(input_layer, encoded)

# Compile
autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

log.info("Model architecture:")
autoencoder.summary(print_fn=log.info)

# Train with early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

log.info("Training autoencoder...")
history = autoencoder.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Generate encoded representations
log.info("Generating severity index from encoded representations")
encoded_features = encoder.predict(X_scaled, batch_size=BATCH_SIZE)

# Calculate severity index as mean of encoded features (scaled to 0-100)
severity_raw = np.mean(encoded_features, axis=1)
severity_min, severity_max = severity_raw.min(), severity_raw.max()
severity_index = 100 * (severity_raw - severity_min) / (severity_max - severity_min)

df['SSD_severity_index'] = severity_index

# Calculate AUROC vs high utilization
# Define high utilization as top quartile of visit counts
if 'visit_count_6m' in df.columns:
    high_utilization = df['visit_count_6m'] >= df['visit_count_6m'].quantile(0.75)
    auroc = roc_auc_score(high_utilization, severity_index)
    log.info(f"AUROC for severity index vs high utilization: {auroc:.3f}")
else:
    auroc = None
    log.warning("Could not calculate AUROC - visit count not available")

# Summary statistics
log.info("\nSeverity Index Statistics:")
log.info(f"Mean: {severity_index.mean():.2f}")
log.info(f"Std: {severity_index.std():.2f}")
log.info(f"Min: {severity_index.min():.2f}")
log.info(f"Max: {severity_index.max():.2f}")
log.info(f"Median: {np.median(severity_index):.2f}")

# Check correlation with exposure flag
if 'exposure_flag' in df.columns:
    exposed_severity = df[df['exposure_flag'] == 1]['SSD_severity_index'].mean()
    unexposed_severity = df[df['exposure_flag'] == 0]['SSD_severity_index'].mean()
    log.info(f"\nMean severity - Exposed: {exposed_severity:.2f}, Unexposed: {unexposed_severity:.2f}")

# Save output
log.info(f"Saving mediator data to {OUT_PATH}")
output_df = df[['Patient_ID', 'SSD_severity_index']]
output_df.to_parquet(OUT_PATH, index=False)
log.info(f"Saved: {OUT_PATH}")

# Save model
model_path = ROOT / 'models'
model_path.mkdir(exist_ok=True)
autoencoder.save(model_path / 'ssd_autoencoder.h5')
encoder.save(model_path / 'ssd_encoder.h5')
log.info(f"Saved models to {model_path}")

# Update study documentation
import subprocess
try:
    result = subprocess.run([
        sys.executable, 
        str(ROOT / "scripts" / "update_study_doc.py"),
        "--step", "Mediator autoencoder completed",
        "--kv", f"artefact=mediator_autoencoder.parquet",
        "--kv", f"n_patients={len(df)}",
        "--kv", f"n_features={len(feature_cols)}",
        "--kv", f"mediator_auroc={auroc:.3f}" if auroc else "mediator_auroc=NA",
        "--kv", "hypotheses=H4,H5,H6",
        "--kv", f"script=03_mediator_autoencoder.py",
        "--kv", "ae_feature_manifest=code_lists/ae56_features.csv",
        "--kv", "status=implemented"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")