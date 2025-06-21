#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_referral_sequence.py  – Enhanced Referral Sequence Analysis

Enhanced with Felipe's suggestions:
- Psychiatric vs medical specialist separation
- Dual pathway tracking (medical → psychiatric)
- Enhanced referral loop detection

Outputs:
`data_derived/referral_sequences.parquet`

HYPOTHESIS MAPPING:
Supports H2 enhanced criteria with psychiatric pathway analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from collections import Counter
from datetime import datetime

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

# Apply Felipe's psychiatric pathway enhancements
log.info("=== APPLYING FELIPE'S PSYCHIATRIC PATHWAY ENHANCEMENTS ===")

# Perform dual pathway analysis
pathway_df, psychiatric_refs, medical_refs = analyze_dual_pathway_patterns(referrals, cohort)

# Enhance H2 criteria
h2_enhanced = enhance_h2_referral_criteria(pathway_df, cohort)

# Create enhanced output dataframe
log.info("Creating enhanced output dataset...")
output = cohort[["Patient_ID"]].copy()

# Original referral loop analysis
output["referral_loop"] = output.Patient_ID.map(referral_loops).fillna(False)
output["loop_count"] = output.Patient_ID.map(loop_counts).fillna(0)
output["sequence_length"] = output.Patient_ID.map(sequence_lengths).fillna(0)
output["has_circular_pattern"] = output.Patient_ID.map(circular_patterns).fillna(False)
output["mean_referral_interval_days"] = output.Patient_ID.map(referral_intervals).fillna(0)

# Felipe's enhancements - merge psychiatric pathway analysis
if len(h2_enhanced) > 0:
    enhanced_columns = ['Patient_ID', 'H2_referral_loop_enhanced', 'dual_pathway', 
                       'has_psychiatric_referral', 'has_medical_specialist', 'total_specialist_referrals']
    enhanced_data = h2_enhanced[enhanced_columns]
    output = output.merge(enhanced_data, on='Patient_ID', how='left')
    
    # Fill missing values (patients with no specialist referrals)
    referral_columns = ['H2_referral_loop_enhanced', 'dual_pathway', 'has_psychiatric_referral', 
                       'has_medical_specialist']
    for col in referral_columns:
        output[col] = output[col].fillna(False)
    
    output['total_specialist_referrals'] = output['total_specialist_referrals'].fillna(0)
    
    log.info("✓ Felipe's psychiatric pathway enhancements integrated")
else:
    log.warning("No enhanced referral data available - setting Felipe columns to defaults")
    output['H2_referral_loop_enhanced'] = False
    output['dual_pathway'] = False
    output['has_psychiatric_referral'] = False
    output['has_medical_specialist'] = False
    output['total_specialist_referrals'] = 0

# Enhanced summary statistics (original + Felipe's enhancements)
log.info("\nEnhanced referral pattern summary:")
log.info(f"  Patients with referral loops (original): {output['referral_loop'].sum():,} ({output['referral_loop'].mean():.1%})")
log.info(f"  Mean sequence length: {output['sequence_length'].mean():.2f}")
log.info(f"  Patients with circular patterns: {output['has_circular_pattern'].sum():,} ({output['has_circular_pattern'].mean():.1%})")
log.info(f"  Mean interval between referrals: {output['mean_referral_interval_days'].mean():.1f} days")

# Felipe's enhanced metrics
if 'H2_referral_loop_enhanced' in output.columns:
    log.info(f"\n✓ Felipe's psychiatric pathway enhancements:")
    log.info(f"  H2 Enhanced referral loops: {output['H2_referral_loop_enhanced'].sum():,} ({output['H2_referral_loop_enhanced'].mean():.1%})")
    log.info(f"  Dual pathway patients (medical + psychiatric): {output['dual_pathway'].sum():,} ({output['dual_pathway'].mean():.1%})")
    log.info(f"  Patients with psychiatric referrals: {output['has_psychiatric_referral'].sum():,} ({output['has_psychiatric_referral'].mean():.1%})")
    log.info(f"  Patients with medical specialist referrals: {output['has_medical_specialist'].sum():,} ({output['has_medical_specialist'].mean():.1%})")
    log.info(f"  Mean specialist referrals per patient: {output['total_specialist_referrals'].mean():.2f}")

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

def identify_psychiatric_referrals(referrals):
    """Identify psychiatric/mental health referrals with enhanced patterns"""
    log.info("Identifying psychiatric referrals...")
    
    # Enhanced psychiatric keywords based on clinical review
    psychiatric_keywords = [
        'psychiatr', 'mental health', 'psych', 'behavioral health',
        'addiction', 'substance', 'counsell', 'therapy', 'therapist',
        'psychology', 'psychologist', 'cognitive', 'behavioral',
        'mood', 'anxiety', 'depression', 'bipolar', 'schizophren',
        'eating disorder', 'ptsd', 'trauma', 'stress management',
        'crisis', 'suicide', 'self harm', 'mental wellness'
    ]
    
    # Create pattern for case-insensitive matching
    pattern = '|'.join(psychiatric_keywords)
    
    # Handle empty dataframe case
    if len(referrals) == 0:
        log.info("No referrals to analyze")
        empty_result = referrals.copy()
        empty_result['referral_type'] = pd.Series([], dtype='object')
        return empty_result
    
    # Identify psychiatric referrals
    psych_mask = referrals['Name_calc'].str.contains(pattern, case=False, na=False)
    
    # Also check specialty codes if available
    if 'SpecialtyCode' in referrals.columns:
        psych_codes = ['PSYC', 'MENT', 'BEHV', 'ADDI', 'COUN']
        code_pattern = '|'.join(psych_codes)
        psych_mask |= referrals['SpecialtyCode'].str.contains(code_pattern, case=False, na=False)
    
    psychiatric_referrals = referrals[psych_mask].copy()
    psychiatric_referrals['referral_type'] = 'psychiatric'
    
    log.info(f"Psychiatric referrals identified: {len(psychiatric_referrals):,}")
    
    return psychiatric_referrals

def analyze_dual_pathway_patterns(referrals, cohort):
    """Analyze medical → psychiatric referral pathways"""
    log.info("Analyzing dual pathway patterns...")
    
    # Get psychiatric and medical referrals
    psychiatric_refs = identify_psychiatric_referrals(referrals)
    
    # Identify medical specialists (non-psychiatric)
    medical_specialties = [
        'cardio', 'gastro', 'neuro', 'orthop', 'rheuma', 'endocrin',
        'pulmon', 'nephro', 'oncol', 'dermat', 'urol', 'ophthal',
        'otolaryn', 'ent', 'allergy', 'immunol', 'hematol',
        'infectious', 'radiol', 'pathol', 'surgery', 'surgeon'
    ]
    medical_pattern = '|'.join(medical_specialties)
    
    # Exclude general practice and psychiatric
    exclude_patterns = ['family', 'general', 'gp', 'primary', 'walk-in', 'clinic'] + psychiatric_keywords
    exclude_pattern = '|'.join(exclude_patterns)
    
    medical_mask = (referrals['Name_calc'].str.contains(medical_pattern, case=False, na=False) & 
                   ~referrals['Name_calc'].str.contains(exclude_pattern, case=False, na=False))
    
    medical_refs = referrals[medical_mask].copy()
    medical_refs['referral_type'] = 'medical_specialist'
    
    # Combine and sort by patient and date
    all_specialist_refs = pd.concat([psychiatric_refs, medical_refs])
    all_specialist_refs = all_specialist_refs.sort_values(['Patient_ID', 'CompletedDate'])
    
    # Analyze patient-level patterns
    pathway_analysis = {}
    
    for patient_id in all_specialist_refs['Patient_ID'].unique():
        patient_refs = all_specialist_refs[all_specialist_refs['Patient_ID'] == patient_id]
        
        # Get referral sequence
        referral_sequence = patient_refs['referral_type'].tolist()
        referral_dates = patient_refs['CompletedDate'].tolist()
        
        # Analyze patterns
        has_medical = 'medical_specialist' in referral_sequence
        has_psychiatric = 'psychiatric' in referral_sequence
        
        # Check for medical → psychiatric sequence
        medical_to_psych = False
        if has_medical and has_psychiatric:
            for i, ref_type in enumerate(referral_sequence[:-1]):
                if ref_type == 'medical_specialist' and 'psychiatric' in referral_sequence[i+1:]:
                    medical_to_psych = True
                    break
        
        pathway_analysis[patient_id] = {
            'total_specialist_referrals': len(patient_refs),
            'medical_referrals': sum(1 for x in referral_sequence if x == 'medical_specialist'),
            'psychiatric_referrals': sum(1 for x in referral_sequence if x == 'psychiatric'),
            'has_medical_specialist': has_medical,
            'has_psychiatric_referral': has_psychiatric,
            'dual_pathway': has_medical and has_psychiatric,
            'medical_to_psychiatric_sequence': medical_to_psych
        }
    
    # Convert to DataFrame
    pathway_df = pd.DataFrame.from_dict(pathway_analysis, orient='index')
    pathway_df.index.name = 'Patient_ID'
    pathway_df = pathway_df.reset_index()
    
    return pathway_df, psychiatric_refs, medical_refs

def enhance_h2_referral_criteria(pathway_df, cohort):
    """Enhanced H2 referral loop criteria with psychiatric specialization"""
    log.info("Enhancing H2 referral loop criteria...")
    
    # Enhanced H2 criteria:
    # 1. ≥2 medical specialist referrals with no clear resolution, OR
    # 2. ≥1 medical specialist + ≥1 psychiatric referral (dual pathway), OR  
    # 3. ≥3 total specialist referrals of any type
    
    h2_enhanced_criteria = pathway_df.copy()
    
    # Criterion 1: Multiple medical specialists (original H2 concept)
    h2_enhanced_criteria['h2_medical_loop'] = h2_enhanced_criteria['medical_referrals'] >= 2
    
    # Criterion 2: Dual pathway (Felipe enhancement)
    h2_enhanced_criteria['h2_dual_pathway'] = h2_enhanced_criteria['dual_pathway']
    
    # Criterion 3: High specialist utilization
    h2_enhanced_criteria['h2_high_utilization'] = h2_enhanced_criteria['total_specialist_referrals'] >= 3
    
    # Combined H2 (any of the above)
    h2_enhanced_criteria['H2_referral_loop_enhanced'] = (
        h2_enhanced_criteria['h2_medical_loop'] |
        h2_enhanced_criteria['h2_dual_pathway'] |
        h2_enhanced_criteria['h2_high_utilization']
    )
    
    # Statistics
    h2_original_count = h2_enhanced_criteria['h2_medical_loop'].sum()
    h2_dual_count = h2_enhanced_criteria['h2_dual_pathway'].sum()
    h2_high_util_count = h2_enhanced_criteria['h2_high_utilization'].sum()
    h2_enhanced_total = h2_enhanced_criteria['H2_referral_loop_enhanced'].sum()
    
    log.info(f"Enhanced H2 referral loop results:")
    log.info(f"  H2 Medical loops (≥2 medical specialists): {h2_original_count:,}")
    log.info(f"  H2 Dual pathways (medical + psychiatric): {h2_dual_count:,}")
    log.info(f"  H2 High utilization (≥3 total specialists): {h2_high_util_count:,}")
    log.info(f"  H2 Enhanced total: {h2_enhanced_total:,}")
    
    return h2_enhanced_criteria

# Felipe's enhancement functions are now integrated into the main execution above