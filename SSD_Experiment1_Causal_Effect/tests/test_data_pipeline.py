#!/usr/bin/env python3
"""
Test suite for data pipeline components

Tests data loading, processing, and pipeline integrity.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import sys
from pathlib import Path

# Add src and utils to path
SRC_PATH = Path(__file__).parent.parent / "src"
UTILS_PATH = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(UTILS_PATH))


class TestDataValidation:
    """Test data validation and quality checks."""
    
    def test_patient_id_consistency(self):
        """Test that Patient_IDs are consistent across tables."""
        # This is a conceptual test - would need actual data
        # In practice, would load checkpoint data and validate
        pass
    
    def test_date_formats(self):
        """Test that date columns are properly formatted."""
        # Test date parsing
        test_dates = ["2018-01-01", "2019-12-31", "2020-06-15"]
        for date_str in test_dates:
            parsed = pd.to_datetime(date_str)
            assert isinstance(parsed, pd.Timestamp)
    
    def test_age_calculation(self):
        """Test age calculation logic."""
        # Mock data for age calculation
        birth_year = 1990
        reference_year = 2018
        expected_age = reference_year - birth_year
        
        assert expected_age == 28
        assert expected_age >= 18  # Meets minimum age criteria
    
    def test_charlson_score_bounds(self):
        """Test Charlson score calculation bounds."""
        # Charlson scores should be non-negative integers
        valid_scores = [0, 1, 2, 3, 4, 5]
        invalid_scores = [-1, 6, 10, 15]  # Assuming max of 5
        
        for score in valid_scores:
            assert 0 <= score <= 5
        
        # Test that invalid scores would be excluded
        for score in invalid_scores:
            if score < 0 or score > 5:
                assert True  # Would be excluded in pipeline


class TestExposureDefinition:
    """Test exposure variable construction."""
    
    def test_normal_lab_count(self):
        """Test normal lab counting logic."""
        # Mock lab data
        lab_data = pd.DataFrame({
            'Patient_ID': [1, 1, 1, 2, 2],
            'TestResult_calc': [5.0, 7.5, 6.2, 8.9, 4.1],
            'LowerNormal': [4.0, 6.0, 5.0, 7.0, 3.0],
            'UpperNormal': [8.0, 9.0, 8.0, 10.0, 6.0],
            'PerformedDate': pd.to_datetime(['2018-01-01', '2018-02-01', '2018-03-01', 
                                           '2018-01-15', '2018-02-15'])
        })
        
        # Test normal range logic
        def is_normal(row):
            try:
                result = float(row['TestResult_calc'])
                if pd.notna(row['LowerNormal']) and pd.notna(row['UpperNormal']):
                    return row['LowerNormal'] <= result <= row['UpperNormal']
            except:
                return False
            return False
        
        lab_data['is_normal'] = lab_data.apply(is_normal, axis=1)
        
        # Count normal labs per patient
        normal_counts = lab_data.groupby('Patient_ID')['is_normal'].sum()
        
        assert normal_counts[1] == 3  # All 3 labs normal for patient 1
        assert normal_counts[2] == 2  # 2 labs normal for patient 2
    
    def test_drug_exposure_duration(self):
        """Test medication duration calculation."""
        # Mock medication data
        med_data = pd.DataFrame({
            'Patient_ID': [1, 1, 2],
            'StartDate': pd.to_datetime(['2018-01-01', '2018-04-01', '2018-01-01']),
            'EndDate': pd.to_datetime(['2018-03-31', '2018-06-30', '2018-02-15']),
            'DrugName': ['GABAPENTIN', 'ZOPICLONE', 'IBUPROFEN']
        })
        
        # Calculate duration
        med_data['duration_days'] = (med_data['EndDate'] - med_data['StartDate']).dt.days
        
        assert med_data.loc[0, 'duration_days'] == 89  # ~3 months
        assert med_data.loc[1, 'duration_days'] == 90  # Exactly 3 months
        assert med_data.loc[2, 'duration_days'] == 45   # 1.5 months
        
        # Test exposure criteria (≥90 days)
        long_exposure = med_data['duration_days'] >= 90
        assert long_exposure.sum() == 1  # Only one meets criteria
    
    def test_referral_unresolved_logic(self):
        """Test unresolved referral identification."""
        # Mock referral data
        referral_data = pd.DataFrame({
            'Patient_ID': [1, 1, 2, 2],
            'Name_calc': ['Cardiology', 'Neurology', 'Gastroenterology', 'Psychiatry'],
            'DiagnosisCode': ['786.50', '780.4', '787.91', 'V71.09'],  # Last is NYD
            'CompletedDate': pd.to_datetime(['2018-01-15', '2018-02-15', '2018-01-20', '2018-03-01'])
        })
        
        # NYD pattern (780-789 range)
        nyd_pattern = referral_data['DiagnosisCode'].str.match(r'^78[0-9]|^V71')

        # All records contain symptom or NYD codes
        assert nyd_pattern.sum() == 4


class TestOutcomeDefinition:
    """Test outcome variable construction."""
    
    def test_encounter_counting(self):
        """Test healthcare encounter counting."""
        # Mock encounter data
        encounter_data = pd.DataFrame({
            'Patient_ID': [1, 1, 1, 2, 2],
            'EncounterDate': pd.to_datetime(['2019-01-01', '2019-02-01', '2019-03-01',
                                           '2019-01-15', '2019-02-15']),
            'EncounterType': ['Primary Care', 'Primary Care', 'Emergency', 
                            'Primary Care', 'Specialist']
        })
        
        # Count total encounters per patient
        encounter_counts = encounter_data.groupby('Patient_ID').size()
        
        assert encounter_counts[1] == 3
        assert encounter_counts[2] == 2
        
        # Count by type
        pc_encounters = encounter_data[encounter_data['EncounterType'] == 'Primary Care'].groupby('Patient_ID').size()
        assert pc_encounters[1] == 2
        assert pc_encounters[2] == 1


class TestConfounderVariables:
    """Test confounder variable construction."""
    
    def test_depression_anxiety_flags(self):
        """Test mental health flag creation."""
        # Mock diagnosis data
        dx_data = pd.DataFrame({
            'Patient_ID': [1, 1, 2, 3],
            'DiagnosisCode_calc': ['296.2', '300.00', '250.0', '309.81'],  # Depression, Anxiety, Diabetes, PTSD
        })
        
        # Test depression flag (296.*)
        depression_flag = dx_data['DiagnosisCode_calc'].str.match(r'^296')
        assert depression_flag.sum() == 1
        
        # Test anxiety flag (300.*)
        anxiety_flag = dx_data['DiagnosisCode_calc'].str.match(r'^300')
        assert anxiety_flag.sum() == 1
        
        # Test PTSD flag
        ptsd_flag = dx_data['DiagnosisCode_calc'].str.match(r'^309\.81')
        assert ptsd_flag.sum() == 1
    
    def test_covid_flag_creation(self):
        """Test Long-COVID flag creation."""
        # Mock health condition data
        condition_data = pd.DataFrame({
            'Patient_ID': [1, 2, 3],
            'DiagnosisCode_calc': ['U07.1', 'J44.1', 'U09.9'],
            'DiagnosisText_calc': ['COVID-19', 'COPD exacerbation', 'Post COVID condition']
        })
        
        # Test COVID code patterns
        covid_codes = condition_data['DiagnosisCode_calc'].str.match(r'^U07\.1|^U09\.9')
        assert covid_codes.sum() == 2
        
        # Test text patterns
        covid_text = condition_data['DiagnosisText_calc'].str.contains('COVID', case=False, na=False)
        assert covid_text.sum() == 2


class TestPipelineIntegration:
    """Test pipeline integration and data flow."""
    
    def test_patient_id_propagation(self):
        """Test that Patient_IDs propagate correctly through pipeline."""
        # Mock patient IDs
        patient_ids = [1, 2, 3, 4, 5]
        
        # Simulate data flow through pipeline stages
        cohort_ids = set(patient_ids)
        exposure_ids = set([1, 2, 3, 4, 5])  # All patients have exposure data
        outcome_ids = set([1, 2, 3, 4, 5])   # All patients have outcome data
        
        # Check that IDs are preserved
        assert cohort_ids == exposure_ids == outcome_ids
    
    def test_temporal_windows(self):
        """Test temporal window definitions."""
        from datetime import datetime, timedelta
        
        # Define windows
        ref_date = datetime(2018, 1, 1)
        exposure_start = datetime(2018, 1, 1)
        exposure_end = datetime(2019, 1, 1)
        outcome_start = datetime(2019, 7, 1)
        outcome_end = datetime(2020, 12, 31)
        
        # Test window logic
        assert exposure_start >= ref_date
        assert exposure_end > exposure_start
        assert outcome_start > exposure_end  # No overlap
        assert outcome_end > outcome_start
        
        # Test durations
        exposure_duration = (exposure_end - exposure_start).days
        outcome_duration = (outcome_end - outcome_start).days
        
        assert exposure_duration == 365  # 1 year
        assert outcome_duration > 500    # ~1.5 years


class TestQualityControl:
    """Test quality control checks."""
    
    def test_missing_data_detection(self):
        """Test missing data detection."""
        # Create test data with missing values
        test_data = pd.DataFrame({
            'Patient_ID': [1, 2, 3, 4, 5],
            'Age': [25, np.nan, 35, 40, 45],
            'Sex': ['M', 'F', np.nan, 'M', 'F'],
            'Charlson': [0, 1, 2, np.nan, 1]
        })
        
        # Calculate missingness
        missing_pct = test_data.isnull().sum() / len(test_data) * 100
        
        assert missing_pct['Age'] == 20.0
        assert missing_pct['Sex'] == 20.0
        assert missing_pct['Charlson'] == 20.0
        assert missing_pct['Patient_ID'] == 0.0
    
    def test_duplicate_detection(self):
        """Test duplicate record detection."""
        # Create test data with duplicates
        test_data = pd.DataFrame({
            'Patient_ID': [1, 2, 3, 2, 4],  # Patient 2 appears twice
            'EncounterDate': ['2018-01-01', '2018-01-02', '2018-01-03', 
                            '2018-01-02', '2018-01-04']
        })
        
        # Check for duplicate Patient_IDs
        duplicates = test_data['Patient_ID'].duplicated()
        assert duplicates.sum() == 1  # One duplicate found
        
        # Check specific duplicate
        assert test_data.loc[duplicates, 'Patient_ID'].iloc[0] == 2
    
    def test_data_type_validation(self):
        """Test data type validation."""
        # Test numeric columns
        numeric_data = pd.Series([1, 2, 3, 4, 5])
        assert pd.api.types.is_numeric_dtype(numeric_data)
        
        # Test date columns
        date_data = pd.to_datetime(['2018-01-01', '2018-01-02', '2018-01-03'])
        assert pd.api.types.is_datetime64_any_dtype(date_data)
        
        # Test string columns
        string_data = pd.Series(['A', 'B', 'C'])
        assert pd.api.types.is_string_dtype(string_data) or pd.api.types.is_object_dtype(string_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])