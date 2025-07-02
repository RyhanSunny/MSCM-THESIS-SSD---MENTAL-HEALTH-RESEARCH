#!/usr/bin/env python3

import pandas as pd
from datetime import datetime
import os

print('FILE COMPATIBILITY CHECK')
print('=' * 40)

cohort_time = os.path.getmtime('data_derived/cohort.parquet')
cohort_dt = datetime.fromtimestamp(cohort_time)

files_to_check = [
    'exposure.parquet', 
    'mediator_autoencoder.parquet', 
    'outcomes.parquet', 
    'lab_sensitivity.parquet', 
    'referral_sequences.parquet'
]

needs_regeneration = []

for file in files_to_check:
    filepath = f'data_derived/{file}'
    if os.path.exists(filepath):
        file_time = os.path.getmtime(filepath)
        file_dt = datetime.fromtimestamp(file_time)
        is_outdated = file_time < cohort_time
        
        df = pd.read_parquet(filepath)
        patient_count = len(df)
        
        status = '🔴 OUTDATED' if is_outdated else '✅ CURRENT'
        if patient_count != 250025:
            status += f' (Count: {patient_count:,} ≠ 250,025)'
        
        print(f'{file:25} {file_dt:%Y-%m-%d %H:%M} {status}')
        
        if is_outdated or patient_count != 250025:
            needs_regeneration.append(file.replace('.parquet', ''))

print(f'\nCohort rebuilt: {cohort_dt:%Y-%m-%d %H:%M}')
print(f'\n🔧 Files needing regeneration:')
for file in needs_regeneration:
    print(f'  - {file}')

print(f'\n📋 Regeneration plan:')
if 'exposure' in needs_regeneration:
    print('  1. python src/02_exposure_flag.py')
if 'mediator_autoencoder' in needs_regeneration:
    print('  2. python src/03_mediator_autoencoder.py')
if 'outcomes' in needs_regeneration:
    print('  3. python src/04_outcome_flag.py')
if 'lab_sensitivity' in needs_regeneration:
    print('  4. python src/06_lab_flag.py')
if 'referral_sequences' in needs_regeneration:
    print('  5. python src/07_referral_sequence.py')
print('  6. python src/08_patient_master_table.py  # Rebuild master')