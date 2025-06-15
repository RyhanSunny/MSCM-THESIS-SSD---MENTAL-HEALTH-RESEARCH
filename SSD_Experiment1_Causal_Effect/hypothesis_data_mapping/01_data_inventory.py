#!/usr/bin/env python3
"""
01_data_inventory.py - Comprehensive inventory of available data
Author: Ryhan Suny
Date: 2025-05-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hypothesis_data_mapping/data_inventory.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Paths
CHECKPOINT = Path("Notebooks/data/interim/checkpoint_1_20250318_024427")
OUTPUT_DIR = Path("hypothesis_data_mapping")

log.info("=" * 80)
log.info("DATA INVENTORY FOR HYPOTHESIS TESTING")
log.info("=" * 80)

# Load metadata
with open(CHECKPOINT / "metadata.json", "r") as f:
    metadata = json.load(f)

log.info(f"\nCheckpoint Date: {metadata.get('timestamp', 'Unknown')}")
log.info(f"Total Tables: {len(metadata.get('tables', {}))}")

# Inventory all tables
inventory = {}

for table_name, table_info in metadata.get('tables', {}).items():
    log.info(f"\n{'='*60}")
    log.info(f"TABLE: {table_name}")
    log.info(f"{'='*60}")
    row_count = table_info.get('row_count', 'Unknown')
    if isinstance(row_count, int):
        log.info(f"Rows: {row_count:,}")
    else:
        log.info(f"Rows: {row_count}")
    log.info(f"Columns: {table_info.get('column_count', 'Unknown')}")
    
    # Load actual data to check columns
    file_path = CHECKPOINT / f"{table_name}.parquet"
    csv_path = CHECKPOINT / f"{table_name}.csv"
    
    if file_path.exists():
        df = pd.read_parquet(file_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, nrows=1000)  # Just for column inspection
    else:
        log.warning(f"Could not find data file for {table_name}")
        continue
    
    # Store column information
    inventory[table_name] = {
        'row_count': len(df) if file_path.exists() else table_info.get('row_count', 0),
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    log.info(f"\nColumns ({len(df.columns)}):")
    for col in sorted(df.columns):
        log.info(f"  - {col} ({df[col].dtype})")
    
    # Check for key fields
    log.info("\nKey Field Checks:")
    
    # Check for postal code
    postal_cols = [col for col in df.columns if 'postal' in col.lower() or 'zip' in col.lower()]
    if postal_cols:
        log.info(f"  ✓ Postal code columns found: {postal_cols}")
        # Sample postal codes
        for col in postal_cols:
            if col in df.columns:
                sample = df[col].dropna().head(5).tolist()
                log.info(f"    Sample {col}: {sample}")
    else:
        log.info("  ✗ No postal code columns found")
    
    # Check for socioeconomic indicators
    ses_cols = [col for col in df.columns if any(term in col.lower() for term in 
                ['income', 'education', 'occupation', 'housing', 'employment', 'deprivation'])]
    if ses_cols:
        log.info(f"  ✓ Socioeconomic columns found: {ses_cols}")
    else:
        log.info("  ✗ No direct socioeconomic columns found")
    
    # Check for dates
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        log.info(f"  ✓ Date columns found: {date_cols}")
    
    # Check for IDs
    id_cols = [col for col in df.columns if '_id' in col.lower() or col.lower().endswith('id')]
    if id_cols:
        log.info(f"  ✓ ID columns found: {id_cols}")

# Save inventory
with open(OUTPUT_DIR / "data_inventory.json", "w") as f:
    json.dump(inventory, f, indent=2)

log.info("\n" + "="*80)
log.info("SUMMARY OF KEY FINDINGS")
log.info("="*80)

# Check specific requirements
log.info("\n1. DEPRIVATION/PAMPALON INDEX REQUIREMENTS:")
log.info("   - Postal code needed: Checking patient_demographic table...")

if 'patient_demographic' in inventory:
    demo_cols = inventory['patient_demographic']['columns']
    postal_found = [col for col in demo_cols if 'postal' in col.lower()]
    if postal_found:
        log.info(f"   ✓ FOUND: {postal_found}")
        # Load sample to check format
        demo_df = pd.read_parquet(CHECKPOINT / "patient_demographic.parquet")
        postal_col = postal_found[0]
        sample = demo_df[postal_col].dropna().head(10)
        log.info(f"   - Sample values: {sample.tolist()}")
        log.info(f"   - Non-null count: {demo_df[postal_col].notna().sum():,} / {len(demo_df):,}")
        log.info(f"   - Completeness: {demo_df[postal_col].notna().mean()*100:.1f}%")
    else:
        log.info("   ✗ NO POSTAL CODE FOUND IN PATIENT_DEMOGRAPHIC")
else:
    log.info("   ✗ PATIENT_DEMOGRAPHIC TABLE NOT FOUND")

log.info("\n2. MENTAL HEALTH DIAGNOSES:")
health_cond = pd.read_parquet(CHECKPOINT / "health_condition.parquet")
enc_diag = pd.read_parquet(CHECKPOINT / "encounter_diagnosis.parquet")

# Check for mental health codes
mh_codes = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']  # ICD-10 mental health
mh_health = health_cond[health_cond['DiagnosisCode_calc'].str.startswith(tuple(mh_codes), na=False)]
mh_enc = enc_diag[enc_diag['DiagnosisCode_calc'].str.startswith(tuple(mh_codes), na=False)]

log.info(f"   - Mental health conditions in health_condition: {len(mh_health):,}")
log.info(f"   - Mental health diagnoses in encounters: {len(mh_enc):,}")
log.info(f"   - Unique patients with MH conditions: {mh_health['Patient_ID'].nunique():,}")

log.info("\n3. HEALTHCARE UTILIZATION METRICS:")
encounter = pd.read_parquet(CHECKPOINT / "encounter.parquet")
log.info(f"   - Total encounters: {len(encounter):,}")
log.info(f"   - Encounter types: {encounter['EncounterType'].value_counts().head()}")

log.info("\n4. MEDICATION DATA:")
med = pd.read_parquet(CHECKPOINT / "medication.parquet")
log.info(f"   - Total prescriptions: {len(med):,}")
log.info(f"   - Has ATC codes: {'Code_calc' in med.columns}")
log.info(f"   - Has duration info: {'DurationCount' in med.columns}")

log.info("\n5. LAB DATA:")
log.info(f"   - Lab file exists: {(CHECKPOINT / 'lab.csv').exists()}")
log.info(f"   - Has normal ranges: Check needed")

log.info("\n6. REFERRAL DATA:")
referral = pd.read_parquet(CHECKPOINT / "referral.parquet")
log.info(f"   - Total referrals: {len(referral):,}")
log.info(f"   - Has specialty info: {'Name_calc' in referral.columns}")

print("\nData inventory complete. Check hypothesis_data_mapping/data_inventory.log for details.")