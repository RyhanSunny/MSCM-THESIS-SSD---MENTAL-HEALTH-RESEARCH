#!/usr/bin/env python3
"""
Examine referral data to understand what psychiatrist/mental health referrals are available
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Set up paths
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "Notebooks/data/interim"
CKPT = sorted(CHECKPOINT_ROOT.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime, reverse=True)[0]

print(f"Using checkpoint: {CKPT}")

# Load referral data
pq_path = CKPT / "referral.parquet"
if pq_path.exists():
    referrals = pd.read_parquet(pq_path)
    print("Loaded referral.parquet")
else:
    csv_path = CKPT / "referral.csv"
    if csv_path.exists():
        referrals = pd.read_csv(csv_path, low_memory=False)
        print("Loaded referral.csv")
    else:
        print("No referral data found")
        sys.exit(1)

print(f"\nTotal referrals: {len(referrals):,}")
print(f"Unique patients: {referrals.Patient_ID.nunique():,}")

# Examine Name_calc field for specialty types
print("\n=== Referral Specialties (Name_calc) ===")
if 'Name_calc' in referrals.columns:
    name_calc_counts = referrals['Name_calc'].value_counts()
    print(f"Total unique specialties: {len(name_calc_counts)}")
    print("\nTop 20 most common:")
    for specialty, count in name_calc_counts.head(20).items():
        print(f"  {specialty}: {count:,}")
    
    # Look for psychiatrist/mental health terms
    print("\n=== Psychiatrist/Mental Health Related ===")
    psych_terms = ['psychiatr', 'mental', 'psych', 'behavior', 'addiction', 'counseling', 'counselling', 'therapy']
    
    for term in psych_terms:
        matches = name_calc_counts[name_calc_counts.index.str.contains(term, case=False, na=False)]
        if len(matches) > 0:
            print(f"\nContaining '{term}':")
            for specialty, count in matches.items():
                print(f"  {specialty}: {count:,}")

# Also check Name_orig
print("\n=== Referral Specialties (Name_orig) ===")
if 'Name_orig' in referrals.columns:
    name_orig_counts = referrals['Name_orig'].value_counts()
    print(f"Total unique specialties (orig): {len(name_orig_counts)}")
    
    print("\n=== Psychiatrist/Mental Health Related (orig) ===")
    for term in psych_terms:
        matches = name_orig_counts[name_orig_counts.index.str.contains(term, case=False, na=False)]
        if len(matches) > 0:
            print(f"\nContaining '{term}' (orig):")
            for specialty, count in matches.head(10).items():
                print(f"  {specialty}: {count:,}")

# Check ConceptCode and DescriptionCode if available
if 'ConceptCode' in referrals.columns:
    print("\n=== Concept Codes ===")
    concept_counts = referrals['ConceptCode'].value_counts()
    print(f"Total unique concept codes: {len(concept_counts)}")
    print("Top 10:")
    for code, count in concept_counts.head(10).items():
        print(f"  {code}: {count:,}")

if 'DescriptionCode' in referrals.columns:
    print("\n=== Description Codes ===")
    desc_counts = referrals['DescriptionCode'].value_counts()
    print(f"Total unique description codes: {len(desc_counts)}")
    print("Top 10:")
    for code, count in desc_counts.head(10).items():
        print(f"  {code}: {count:,}")

# Show sample records
print("\n=== Sample Referral Records ===")
sample_cols = ['Patient_ID', 'Name_calc', 'Name_orig', 'ConceptCode', 'DescriptionCode', 'CompletedDate', 'DateCreated']
available_cols = [col for col in sample_cols if col in referrals.columns]
print(f"Available columns: {available_cols}")
print(referrals[available_cols].head(10))