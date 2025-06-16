#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze Mental Health Population Characteristics
"""

import pandas as pd
import numpy as np

# Load the master table to understand MH population
master = pd.read_parquet('data_derived/patient_master.parquet')

print('=== MENTAL HEALTH POPULATION CHARACTERISTICS ===')
print(f'Total patients: {len(master):,}')

# Mental health indicators
print(f'\n--- Mental Health Diagnoses ---')
if 'mental_health_dx' in master.columns:
    mh_rate = master['mental_health_dx'].mean() * 100
    print(f'Any mental health diagnosis: {mh_rate:.1f}%')

if 'depression_dx' in master.columns:
    dep_rate = master['depression_dx'].mean() * 100
    print(f'Depression diagnosis: {dep_rate:.1f}%')

if 'anxiety_dx' in master.columns:
    anx_rate = master['anxiety_dx'].mean() * 100
    print(f'Anxiety diagnosis: {anx_rate:.1f}%')

# Check medication patterns
print(f'\n--- Medication Patterns ---')
if 'baseline_antidepressant' in master.columns:
    ad_rate = master['baseline_antidepressant'].mean() * 100
    print(f'Baseline antidepressant use: {ad_rate:.1f}%')

# SSD patterns in MH population
print(f'\n--- SSD Patterns ---')
print(f'H1 (≥3 normal labs): {master["H1_normal_labs"].sum():,} patients ({master["H1_normal_labs"].mean()*100:.1f}%)')
print(f'H2 (referral loops): {master["H2_referral_loop"].sum():,} patients ({master["H2_referral_loop"].mean()*100:.1f}%)')
print(f'H3 (drug persistence): {master["H3_drug_persistence"].sum():,} patients ({master["H3_drug_persistence"].mean()*100:.1f}%)')

# Severity index distribution
print(f'\n--- SSD Severity Distribution ---')
print(f'Mean severity index: {master["SSD_severity_index"].mean():.2f}')
print(f'Std severity index: {master["SSD_severity_index"].std():.2f}')

# Healthcare utilization
print(f'\n--- Healthcare Utilization ---')
print(f'Mean encounters: {master["total_encounters"].mean():.2f}')
print(f'Mean ED visits: {master["ed_visits"].mean():.2f}')
print(f'Mean medical costs: ${master["medical_costs"].mean():.2f}')

# Age and demographics
print(f'\n--- Demographics ---')
print(f'Mean age: {master["Age_at_2018"].mean():.1f} years')
print(f'Female: {(master["Sex"] == "F").mean()*100:.1f}%')

# Mental health specific analysis
print(f'\n--- MH Population Subset Analysis ---')
if 'mental_health_dx' in master.columns:
    mh_subset = master[master['mental_health_dx'] == 1]
    print(f'MH subset size: {len(mh_subset):,} patients')
    print(f'MH subset exposure rate: {mh_subset["exposure_flag"].mean()*100:.1f}%')
    print(f'MH subset mean encounters: {mh_subset["total_encounters"].mean():.2f}')
    print(f'MH subset mean severity: {mh_subset["SSD_severity_index"].mean():.2f}')