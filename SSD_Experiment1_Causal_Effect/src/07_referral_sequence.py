#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_referral_sequence.py - Referral Sequence Analysis

Analyzes referral patterns and identifies referral loops.
Supports H2 by identifying inefficient care patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from collections import Counter

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
        logging.FileHandler("07_referral_sequence.log", mode="w")
    ])
log = logging.getLogger("referral_sequence")

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
OUT_PATH = ROOT / 'data_derived' / 'referral_sequences.parquet'

# Find latest checkpoint
def latest_checkpoint(base: Path) -> Path:
    cps = sorted(base.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cps:
        raise FileNotFoundError(f"No checkpoint_* folder found in {base}")
    return cps[0]

CKPT = latest_checkpoint(CHECKPOINT_ROOT)
log.info(f"Using checkpoint: {CKPT}")

# Load data
log.info("Loading cohort and referral data")
cohort = pd.read_parquet(COHORT_PATH)

# Load referral data
pq_path = CKPT / "referral.parquet"
if pq_path.exists():
    referrals = pd.read_parquet(pq_path)
else:
    csv_path = CKPT / "referral.csv"
    if csv_path.exists():
        referrals = pd.read_csv(csv_path, low_memory=False)
    else:
        raise FileNotFoundError("No referral data found in checkpoint")

# Parse dates
date_cols = ["CompletedDate", "DateCreated"]
for col in date_cols:
    if col in referrals.columns:
        referrals[col] = pd.to_datetime(referrals[col], errors="coerce")

# Keep only cohort patients
patient_ids = set(cohort.Patient_ID)
referrals = referrals[referrals.Patient_ID.isin(patient_ids)]

log.info(f"Processing {len(referrals):,} referrals for {referrals.Patient_ID.nunique():,} patients")

# Create referral date (prefer completed, fallback to created)
referrals["ReferralDate"] = referrals.CompletedDate.fillna(referrals.DateCreated)
referrals = referrals.dropna(subset=["ReferralDate"])

# Sort by patient and date
referrals = referrals.sort_values(["Patient_ID", "ReferralDate"])

# Standardize specialty names
referrals["Specialty"] = referrals.Name_calc.str.upper().str.strip()

# Create sequences per patient
log.info("Building referral sequences...")
sequences = referrals.groupby("Patient_ID").apply(
    lambda x: x.sort_values("ReferralDate")["Specialty"].tolist()
).to_dict()

# Analyze patterns
log.info("Analyzing referral patterns...")

# 1. Identify loops (same specialty >=2 times)
referral_loops = {}
loop_counts = {}

for patient_id, seq in sequences.items():
    if len(seq) > 1:
        specialty_counts = Counter(seq)
        loops = {spec: count for spec, count in specialty_counts.items() if count >= 2}
        if loops:
            referral_loops[patient_id] = True
            loop_counts[patient_id] = sum(loops.values()) - len(loops)  # Extra visits
        else:
            referral_loops[patient_id] = False
            loop_counts[patient_id] = 0
    else:
        referral_loops[patient_id] = False
        loop_counts[patient_id] = 0

# 2. Calculate sequence length
sequence_lengths = {pid: len(seq) for pid, seq in sequences.items()}

# 3. Identify circular patterns (A->B->A)
circular_patterns = {}
for patient_id, seq in sequences.items():
    if len(seq) >= 3:
        has_circular = False
        for i in range(len(seq) - 2):
            if seq[i] == seq[i + 2] and seq[i] != seq[i + 1]:
                has_circular = True
                break
        circular_patterns[patient_id] = has_circular
    else:
        circular_patterns[patient_id] = False

# 4. Time between referrals
log.info("Calculating referral intervals...")
referral_intervals = referrals.groupby("Patient_ID").apply(
    lambda x: x["ReferralDate"].diff().dt.days.dropna().mean()
).fillna(0)

# 5. Most common referral paths
log.info("Identifying common referral paths...")
path_counts = Counter()
for seq in sequences.values():
    if len(seq) >= 2:
        for i in range(len(seq) - 1):
            path = f"{seq[i]} -> {seq[i+1]}"
            path_counts[path] += 1

log.info("\nTop 10 referral paths:")
for path, count in path_counts.most_common(10):
    log.info(f"  {path}: {count} times")

# Create output dataframe
log.info("Creating output dataset...")
output = cohort[["Patient_ID"]].copy()
output["referral_loop"] = output.Patient_ID.map(referral_loops).fillna(False)
output["loop_count"] = output.Patient_ID.map(loop_counts).fillna(0)
output["sequence_length"] = output.Patient_ID.map(sequence_lengths).fillna(0)
output["has_circular_pattern"] = output.Patient_ID.map(circular_patterns).fillna(False)
output["mean_referral_interval_days"] = output.Patient_ID.map(referral_intervals).fillna(0)

# Summary statistics
log.info("\nReferral pattern summary:")
log.info(f"  Patients with referral loops: {output['referral_loop'].sum():,} ({output['referral_loop'].mean():.1%})")
log.info(f"  Mean sequence length: {output['sequence_length'].mean():.2f}")
log.info(f"  Patients with circular patterns: {output['has_circular_pattern'].sum():,} ({output['has_circular_pattern'].mean():.1%})")
log.info(f"  Mean interval between referrals: {output['mean_referral_interval_days'].mean():.1f} days")

# Save output
log.info(f"\nSaving referral sequence data to {OUT_PATH}")
output.to_parquet(OUT_PATH, index=False)
log.info(f"Saved {len(output):,} rows")

# Update study documentation
import subprocess
try:
    result = subprocess.run([
        sys.executable, 
        str(ROOT / "scripts" / "update_study_doc.py"),
        "--step", "Referral sequence analysis completed",
        "--kv", f"artefact=referral_sequences.parquet",
        "--kv", f"n_patients={len(output)}",
        "--kv", f"pct_with_loops={output['referral_loop'].mean():.3f}",
        "--kv", f"mean_sequence_length={output['sequence_length'].mean():.2f}",
        "--kv", "referral_sequence=added",
        "--kv", "hypotheses=H2",
        "--kv", f"script=07_referral_sequence.py"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        log.info("Study documentation updated successfully")
    else:
        log.warning(f"Study doc update failed: {result.stderr}")
except Exception as e:
    log.warning(f"Could not update study doc: {e}")