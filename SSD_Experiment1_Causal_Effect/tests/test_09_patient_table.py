#!/usr/bin/env python3
"""
Tests for patient characteristics table generation.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import patch
import importlib.util
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

spec = importlib.util.spec_from_file_location(
    'patient_table',
    Path(__file__).parent.parent / 'src' / '09_patient_table.py'
)
patient_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patient_table)

create_patient_table = patient_table.create_patient_table
summarize_patient_table = patient_table.summarize_patient_table


def mock_datasets():
    cohort = pd.DataFrame({
        'Patient_ID': [1, 2],
        'Age_at_2015': [50, 60],
        'Sex_clean': ['Female', 'Male'],
        'NYD_yn': [1, 0],
        'NYD_body_part_summary': ['General', 'None']
    })
    exposure = pd.DataFrame({
        'Patient_ID': [1, 2],
        'exposure_flag': [1, 0],
        'H1_normal_labs': [1, 0],
        'H2_referral_loop': [0, 1],
        'H3_drug_persistence': [1, 1],
    })
    severity = pd.DataFrame({'Patient_ID': [1, 2], 'SSD_severity_index': [0.6, 0.2]})
    referrals = pd.DataFrame({
        'Patient_ID': [1, 2],
        'has_psychiatric_referral': [True, False],
        'has_medical_specialist': [False, True],
        'total_specialist_referrals': [1, 2]
    })
    confounders = pd.DataFrame({
        'Patient_ID': [1, 2],
        'Charlson': [2, 3],
        'LongCOVID_flag': [0, 1],
        'baseline_encounters': [10, 5],
        'baseline_med_count': [2, 1],
    })
    return cohort, exposure, severity, referrals, confounders


def test_create_patient_table_columns():
    cohort, exposure, severity, referrals, confounders = mock_datasets()

    def reader(path):
        name = Path(path).name
        if name == 'cohort.parquet':
            return cohort
        if name == 'exposure.parquet':
            return exposure
        if name == 'mediator_autoencoder.parquet':
            return severity
        if name == 'referral_enhanced.parquet':
            return referrals
        if name == 'confounders.parquet':
            return confounders
        return pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with patch.object(patient_table, 'DERIVED', tmp_path), \
             patch.object(patient_table, 'OUTPUTS', tmp_path / 'out'), \
             patch.object(patient_table.pd, 'read_parquet', side_effect=reader):
            table = create_patient_table()
            expected = {
                'PID', 'age', 'sex', 'NYD_yn', 'body_part',
                'referred_to_psy_yn', 'referred_to_other_yn',
                'num_specialist_referrals', 'SSD_flag', 'SSD_severity_index',
                'Charlson', 'LongCOVID_flag',
                'baseline_encounters', 'baseline_med_count',
                'H1_normal_labs', 'H2_referral_loop', 'H3_drug_persistence'
            }
            assert expected.issubset(table.columns)
            assert len(table) == 2


def test_summarize_patient_table_output():
    cohort, exposure, severity, referrals, confounders = mock_datasets()

    with patch.object(patient_table.pd, 'read_parquet') as mock_read:
        mock_read.side_effect = [cohort, exposure, severity, referrals, confounders]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(patient_table, 'DERIVED', Path(tmp)), \
                 patch.object(patient_table, 'OUTPUTS', Path(tmp) / 'out'):
                table = create_patient_table()
                summary = summarize_patient_table(table)
                assert 'N' in summary.columns
                assert summary['N'].iloc[0] == 2
                assert summary['SSD_pct'].iloc[0] == 50.0
                assert summary['H1_normal_labs_pct'].iloc[0] == 50.0
                assert summary['H2_referral_loop_pct'].iloc[0] == 50.0
                assert summary['H3_drug_persistence_pct'].iloc[0] == 100.0
                assert summary['baseline_encounters_mean'].iloc[0] == 7.5
                assert summary['baseline_med_count_mean'].iloc[0] == 1.5


