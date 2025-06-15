#!/usr/bin/env python3
"""
pytest configuration and fixtures for SSD causal analysis test suite
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
UTILS_PATH = PROJECT_ROOT / "utils"

sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(UTILS_PATH))


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "study": {
            "name": "SSD Causal Effect Analysis",
            "version": "1.1.0"
        },
        "temporal": {
            "reference_date": "2018-01-01",
            "censor_date": "2018-06-30",
            "exposure_window_start": "2018-01-01",
            "exposure_window_end": "2019-01-01",
            "outcome_window_start": "2019-07-01",
            "outcome_window_end": "2020-12-31"
        },
        "cohort": {
            "min_age": 18,
            "min_observation_months": 30,
            "max_charlson_score": 5
        },
        "exposure": {
            "min_normal_labs": 3,
            "min_symptom_referrals": 2,
            "min_drug_days": 90
        }
    }


@pytest.fixture
def mock_patient_data():
    """Mock patient data for testing."""
    np.random.seed(42)
    n_patients = 1000
    
    return pd.DataFrame({
        'Patient_ID': range(1, n_patients + 1),
        'Sex': np.random.choice(['M', 'F'], n_patients),
        'BirthYear': np.random.randint(1940, 2000, n_patients),
        'Age_at_2018': np.random.randint(18, 78, n_patients),
        'Charlson': np.random.poisson(1, n_patients),
        'SpanMonths': np.random.uniform(30, 60, n_patients),
        'IndexDate_lab': pd.date_range('2018-01-01', periods=n_patients, freq='H')[:n_patients],
        'LongCOVID_flag': np.random.binomial(1, 0.05, n_patients),
        'NYD_count': np.random.poisson(0.5, n_patients)
    })


@pytest.fixture
def mock_encounter_data():
    """Mock encounter data for testing."""
    np.random.seed(42)
    n_encounters = 5000
    n_patients = 1000
    
    return pd.DataFrame({
        'Encounter_ID': range(1, n_encounters + 1),
        'Patient_ID': np.random.randint(1, n_patients + 1, n_encounters),
        'EncounterDate': pd.date_range('2018-01-01', '2020-12-31', periods=n_encounters),
        'EncounterType': np.random.choice(['Primary Care', 'Emergency', 'Specialist'], n_encounters),
        'Provider_ID': np.random.randint(1, 100, n_encounters)
    })


@pytest.fixture
def mock_lab_data():
    """Mock laboratory data for testing."""
    np.random.seed(42)
    n_labs = 3000
    n_patients = 1000
    
    lab_data = pd.DataFrame({
        'Lab_ID': range(1, n_labs + 1),
        'Patient_ID': np.random.randint(1, n_patients + 1, n_labs),
        'PerformedDate': pd.date_range('2018-01-01', '2020-12-31', periods=n_labs),
        'TestResult_calc': np.random.normal(5.0, 2.0, n_labs),
        'LowerNormal': np.random.normal(3.0, 0.5, n_labs),
        'UpperNormal': np.random.normal(7.0, 0.5, n_labs),
        'Name_calc': np.random.choice(['Glucose', 'Hemoglobin', 'Creatinine'], n_labs)
    })
    
    # Ensure some labs are normal
    lab_data['is_normal'] = (
        (lab_data['TestResult_calc'] >= lab_data['LowerNormal']) & 
        (lab_data['TestResult_calc'] <= lab_data['UpperNormal'])
    )
    
    return lab_data


@pytest.fixture
def mock_medication_data():
    """Mock medication data for testing."""
    np.random.seed(42)
    n_meds = 2000
    n_patients = 1000
    
    return pd.DataFrame({
        'Medication_ID': range(1, n_meds + 1),
        'Patient_ID': np.random.randint(1, n_patients + 1, n_meds),
        'StartDate': pd.date_range('2018-01-01', '2020-06-30', periods=n_meds),
        'EndDate': pd.date_range('2018-02-01', '2020-12-31', periods=n_meds),
        'DrugName': np.random.choice(['GABAPENTIN', 'ZOPICLONE', 'IBUPROFEN', 'ACETAMINOPHEN'], n_meds),
        'ATC_Code': np.random.choice(['N05B', 'N05C', 'N02B', 'N02A'], n_meds)
    })


@pytest.fixture
def mock_referral_data():
    """Mock referral data for testing."""
    np.random.seed(42)
    n_referrals = 1500
    n_patients = 1000
    
    return pd.DataFrame({
        'Referral_ID': range(1, n_referrals + 1),
        'Patient_ID': np.random.randint(1, n_patients + 1, n_referrals),
        'ReferralDate': pd.date_range('2018-01-01', '2020-06-30', periods=n_referrals),
        'CompletedDate': pd.date_range('2018-02-01', '2020-12-31', periods=n_referrals),
        'Name_calc': np.random.choice(['Cardiology', 'Neurology', 'Gastroenterology'], n_referrals),
        'DiagnosisCode': np.random.choice(['786.50', '780.4', '787.91', 'V71.09'], n_referrals)
    })


@pytest.fixture
def mock_exposure_data():
    """Mock exposure (SSD pattern) data for testing."""
    np.random.seed(42)
    n_patients = 1000
    
    return pd.DataFrame({
        'Patient_ID': range(1, n_patients + 1),
        'ssd_flag': np.random.binomial(1, 0.15, n_patients),  # 15% exposed
        'normal_lab_count_12m': np.random.poisson(2.5, n_patients),
        'unresolved_referral_count': np.random.poisson(1.2, n_patients),
        'medication_days_continuous': np.random.exponential(60, n_patients),
        'symptom_code_count': np.random.poisson(1.8, n_patients)
    })


@pytest.fixture
def mock_outcome_data():
    """Mock outcome data for testing."""
    np.random.seed(42)
    n_patients = 1000
    
    return pd.DataFrame({
        'Patient_ID': range(1, n_patients + 1),
        'total_encounters_12m': np.random.poisson(8, n_patients),
        'primary_care_encounters': np.random.poisson(6, n_patients),
        'ed_visits': np.random.poisson(0.5, n_patients),
        'specialist_referrals': np.random.poisson(1.2, n_patients),
        'total_cost_proxy': np.random.gamma(2, 500, n_patients),
        'inappropriate_med_continuation': np.random.binomial(1, 0.3, n_patients)
    })


@pytest.fixture
def mock_confounder_data():
    """Mock confounder data for testing."""
    np.random.seed(42)
    n_patients = 1000
    
    return pd.DataFrame({
        'Patient_ID': range(1, n_patients + 1),
        'depression_flag': np.random.binomial(1, 0.12, n_patients),
        'anxiety_flag': np.random.binomial(1, 0.15, n_patients),
        'ptsd_flag': np.random.binomial(1, 0.03, n_patients),
        'baseline_encounters_6m': np.random.poisson(4, n_patients),
        'deprivation_quintile': np.random.randint(1, 6, n_patients),
        'site_id': np.random.randint(1, 20, n_patients),
        'calendar_year': np.random.choice([2018, 2019], n_patients)
    })


@pytest.fixture
def mock_ps_results():
    """Mock propensity score results for testing."""
    np.random.seed(42)
    n_patients = 1000
    
    return {
        'propensity_scores': np.random.beta(2, 5, n_patients),  # Skewed toward 0
        'iptw_weights': np.random.gamma(2, 0.5, n_patients),
        'effective_sample_size': 750,
        'max_smd_post': 0.08,
        'overlap_assessment': 'Good',
        'balance': {
            'age': {'smd_pre': 0.25, 'smd_post': 0.04},
            'sex': {'smd_pre': 0.15, 'smd_post': 0.02},
            'charlson': {'smd_pre': 0.32, 'smd_post': 0.07}
        }
    }


@pytest.fixture
def mock_ate_results():
    """Mock average treatment effect results for testing."""
    return {
        'estimates': [
            {'method': 'TMLE', 'estimate': 1.25, 'ci_lower': 1.10, 'ci_upper': 1.40, 'se': 0.08},
            {'method': 'DML', 'estimate': 1.22, 'ci_lower': 1.08, 'ci_upper': 1.36, 'se': 0.07},
            {'method': 'CausalForest', 'estimate': 1.28, 'ci_lower': 1.12, 'ci_upper': 1.44, 'se': 0.08}
        ],
        'primary_method': 'TMLE',
        'convergence': True,
        'n_bootstrap': 1000
    }


@pytest.fixture
def mock_robustness_results():
    """Mock robustness check results for testing."""
    return {
        'placebo_test_passed': True,
        'flu_vaccination_rr': 1.02,
        'placebo_exposure_rr': 0.98,
        'evalue_global': 2.18,
        'sensitivity_analyses': {
            'misclassification_10pct': {'estimate': 1.20, 'ci_lower': 1.05, 'ci_upper': 1.35},
            'misclassification_20pct': {'estimate': 1.15, 'ci_lower': 1.01, 'ci_upper': 1.29}
        }
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory structure for testing."""
    # Create directory structure
    data_dir = tmp_path / "data_derived"
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    
    data_dir.mkdir()
    results_dir.mkdir()
    figures_dir.mkdir()
    
    return {
        'root': tmp_path,
        'data': data_dir,
        'results': results_dir,
        'figures': figures_dir
    }


@pytest.fixture
def mock_checkpoint_data(temp_data_dir):
    """Create mock checkpoint data files."""
    checkpoint_dir = temp_data_dir['root'] / "Notebooks" / "data" / "interim" / "checkpoint_test"
    checkpoint_dir.mkdir(parents=True)
    
    # Create mock data files
    files_to_create = [
        'patient.parquet',
        'encounter.parquet', 
        'health_condition.parquet',
        'lab.csv',
        'medication.parquet',
        'referral.parquet'
    ]
    
    for filename in files_to_create:
        (checkpoint_dir / filename).touch()
    
    return checkpoint_dir


class MockLogger:
    """Mock logger for testing."""
    
    def __init__(self):
        self.messages = []
    
    def info(self, msg):
        self.messages.append(('INFO', msg))
    
    def warning(self, msg):
        self.messages.append(('WARNING', msg))
    
    def error(self, msg):
        self.messages.append(('ERROR', msg))
    
    def debug(self, msg):
        self.messages.append(('DEBUG', msg))


@pytest.fixture
def mock_logger():
    """Mock logger fixture."""
    return MockLogger()


# Test utilities
def assert_dataframe_properties(df, expected_columns=None, min_rows=0, max_rows=None):
    """Assert basic DataFrame properties."""
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= min_rows
    
    if max_rows is not None:
        assert len(df) <= max_rows
    
    if expected_columns is not None:
        for col in expected_columns:
            assert col in df.columns
    
    # Check for duplicated Patient_IDs if column exists
    if 'Patient_ID' in df.columns:
        assert not df['Patient_ID'].duplicated().any()


def assert_temporal_consistency(df, date_columns, start_date=None, end_date=None):
    """Assert temporal consistency in DataFrame."""
    for col in date_columns:
        if col in df.columns:
            # Check dates are valid
            assert pd.api.types.is_datetime64_any_dtype(df[col])
            
            # Check date ranges if specified
            if start_date is not None:
                assert df[col].min() >= pd.Timestamp(start_date)
            
            if end_date is not None:
                assert df[col].max() <= pd.Timestamp(end_date)


def assert_numeric_bounds(series, min_val=None, max_val=None, allow_na=True):
    """Assert numeric series is within expected bounds."""
    if not allow_na:
        assert not series.isna().any()
    
    if min_val is not None:
        assert series.min() >= min_val
    
    if max_val is not None:
        assert series.max() <= max_val


# Custom pytest markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow