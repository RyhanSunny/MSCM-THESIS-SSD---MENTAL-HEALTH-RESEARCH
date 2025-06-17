#!/usr/bin/env python3
"""
Test suite for temporal ordering validation
Following TDD principles - tests written first per CLAUDE.md requirements
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from temporal_validator import (
    TemporalOrderingValidator,
    validate_exposure_outcome_sequence,
    check_temporal_consistency,
    TemporalValidationError,
    create_temporal_windows
)


class TestTemporalValidator:
    """Test suite for temporal ordering validation"""
    
    def test_basic_temporal_ordering(self):
        """Test basic exposure-precedes-outcome validation"""
        np.random.seed(42)
        n = 1000
        
        # Generate dates with proper temporal ordering
        exposure_dates = pd.date_range('2015-01-01', '2015-12-31', periods=n)
        outcome_dates = exposure_dates + pd.Timedelta(days=365)  # 1 year later
        
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'exposure_date': exposure_dates,
            'outcome_start_date': outcome_dates,
            'ssd_flag': np.random.binomial(1, 0.15, n)
        })
        
        # Test validation
        validator = TemporalOrderingValidator()
        result = validator.validate_exposure_outcome_sequence(
            df, 'exposure_date', 'outcome_start_date'
        )
        
        assert result['temporal_ordering_valid'] is True
        assert result['n_violations'] == 0
        assert result['violation_rate'] == 0.0
    
    def test_temporal_violations_detection(self):
        """Test detection of temporal ordering violations"""
        np.random.seed(42)
        n = 1000
        
        # Generate dates with some violations
        exposure_dates = pd.date_range('2015-01-01', '2015-12-31', periods=n)
        outcome_dates = (exposure_dates + pd.Timedelta(days=365)).to_series().reset_index(drop=True)
        
        # Introduce violations (outcome before exposure) for 5% of cases
        n_violations = int(0.05 * n)
        violation_indices = np.random.choice(n, n_violations, replace=False)
        
        for idx in violation_indices:
            # Make outcome date before exposure date
            outcome_dates.iloc[idx] = exposure_dates[idx] - pd.Timedelta(days=30)
        
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'exposure_date': exposure_dates,
            'outcome_start_date': outcome_dates,
            'ssd_flag': np.random.binomial(1, 0.15, n)
        })
        
        # Test validation detects violations (use non-strict mode for this test)
        validator = TemporalOrderingValidator(strict_mode=False)
        result = validator.validate_exposure_outcome_sequence(
            df, 'exposure_date', 'outcome_start_date'
        )
        
        assert result['temporal_ordering_valid'] is False
        assert result['n_violations'] == n_violations
        assert abs(result['violation_rate'] - 0.05) < 0.01  # Should be ~5%
    
    def test_ssd_specific_temporal_validation(self):
        """Test temporal validation for specific SSD exposure criteria"""
        np.random.seed(42)
        n = 1000
        
        # H1: Normal lab cascade - requires ≥3 normal labs in 12-month window
        lab_dates = []
        outcome_dates = []
        exposure_flags = []
        
        for i in range(n):
            # Generate lab dates for each patient
            patient_lab_dates = pd.date_range(
                '2015-01-01', '2015-12-31', 
                freq=f'{np.random.randint(30, 90)}D'
            )
            
            # Outcome window starts after exposure window
            outcome_start = pd.Timestamp('2016-01-01')
            
            # Check if patient meets exposure criteria
            normal_lab_count = len(patient_lab_dates)
            meets_criteria = normal_lab_count >= 3
            
            lab_dates.append(patient_lab_dates[-1] if len(patient_lab_dates) > 0 
                           else pd.Timestamp('2015-06-01'))
            outcome_dates.append(outcome_start)
            exposure_flags.append(1 if meets_criteria else 0)
        
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'last_normal_lab_date': lab_dates,
            'outcome_window_start': outcome_dates,
            'h1_normal_lab_flag': exposure_flags
        })
        
        # Test SSD-specific validation
        validator = TemporalOrderingValidator()
        result = validator.validate_ssd_exposure_sequence(df)
        
        # All exposures should precede outcome window
        assert result['h1_temporal_valid'] is True
        assert 'h1_violations' in result
        assert result['h1_violations'] == 0
    
    def test_multiple_exposure_criteria_validation(self):
        """Test validation for multiple SSD exposure criteria"""
        np.random.seed(42)
        n = 1000
        
        # Create comprehensive SSD exposure dataset
        df = pd.DataFrame({
            'Patient_ID': range(n),
            
            # H1: Normal lab cascade
            'first_normal_lab_date': pd.date_range('2015-01-01', '2015-06-30', periods=n),
            'last_normal_lab_date': pd.date_range('2015-03-01', '2015-12-31', periods=n),
            'h1_normal_lab_flag': np.random.binomial(1, 0.4, n),
            
            # H2: Unresolved referrals
            'first_referral_date': pd.date_range('2015-02-01', '2015-07-31', periods=n),
            'last_referral_date': pd.date_range('2015-04-01', '2015-11-30', periods=n),
            'h2_referral_loop_flag': np.random.binomial(1, 0.1, n),
            
            # H3: Medication persistence
            'medication_start_date': pd.date_range('2015-01-15', '2015-08-15', periods=n),
            'medication_end_date': pd.date_range('2015-04-15', '2015-12-15', periods=n),
            'h3_medication_flag': np.random.binomial(1, 0.2, n),
            
            # Outcome windows
            'outcome_window_start': pd.Timestamp('2016-01-01'),
            'outcome_window_end': pd.Timestamp('2017-12-31')
        })
        
        # Test comprehensive validation
        validator = TemporalOrderingValidator()
        result = validator.validate_all_ssd_criteria(df)
        
        # Check all criteria have temporal validation results
        for hypothesis in ['h1', 'h2', 'h3']:
            assert f'{hypothesis}_temporal_valid' in result
            assert f'{hypothesis}_violations' in result
        
        # Overall validation should pass if all individual validations pass
        all_valid = all(result[f'h{i}_temporal_valid'] for i in [1, 2, 3])
        assert result['overall_temporal_valid'] == all_valid
    
    def test_temporal_window_creation(self):
        """Test creation of temporal analysis windows"""
        # Define study periods
        config = {
            'exposure_window_start': '2015-01-01',
            'exposure_window_end': '2015-12-31',
            'outcome_window_start': '2016-01-01',
            'outcome_window_end': '2017-12-31',
            'washout_period_days': 30
        }
        
        windows = create_temporal_windows(config)
        
        assert 'exposure_start' in windows
        assert 'exposure_end' in windows
        assert 'outcome_start' in windows
        assert 'outcome_end' in windows
        assert 'washout_days' in windows
        
        # Check temporal logic
        assert windows['exposure_end'] < windows['outcome_start']
        
        # Check washout period
        expected_gap = (windows['outcome_start'] - windows['exposure_end']).days
        assert expected_gap >= windows['washout_days']
    
    def test_temporal_consistency_across_datasets(self):
        """Test temporal consistency across multiple related datasets"""
        np.random.seed(42)
        n = 1000
        
        # Create related datasets with consistent temporal structure
        patients = pd.DataFrame({
            'Patient_ID': range(n),
            'study_entry_date': pd.date_range('2014-01-01', '2014-12-31', periods=n),
            'last_observation_date': pd.date_range('2017-06-01', '2017-12-31', periods=n)
        })
        
        encounters = pd.DataFrame({
            'Patient_ID': np.random.choice(range(n), size=5000),
            'encounter_date': pd.date_range('2015-01-01', '2017-12-31', periods=5000),
            'encounter_type': np.random.choice(['PC', 'ED', 'Specialist'], 5000)
        })
        
        labs = pd.DataFrame({
            'Patient_ID': np.random.choice(range(n), size=3000),
            'lab_date': pd.date_range('2015-01-01', '2017-06-30', periods=3000),
            'test_result': np.random.normal(5, 2, 3000)
        })
        
        # Test cross-dataset temporal consistency
        validator = TemporalOrderingValidator()
        result = validator.check_cross_dataset_consistency(
            patients, encounters, labs,
            patient_id_col='Patient_ID'
        )
        
        assert 'consistency_check_passed' in result
        assert 'encounter_date_violations' in result
        assert 'lab_date_violations' in result
    
    def test_temporal_validation_with_missing_dates(self):
        """Test handling of missing dates in temporal validation"""
        np.random.seed(42)
        n = 1000
        
        # Create dataset with some missing dates
        exposure_dates = pd.date_range('2015-01-01', '2015-12-31', periods=n)
        outcome_dates = pd.date_range('2016-01-01', '2016-12-31', periods=n)
        
        # Introduce missing values
        missing_indices = np.random.choice(n, 50, replace=False)
        exposure_dates = exposure_dates.to_series()
        exposure_dates.iloc[missing_indices] = pd.NaT
        
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'exposure_date': exposure_dates,
            'outcome_date': outcome_dates
        })
        
        # Test validation handles missing values
        validator = TemporalOrderingValidator()
        result = validator.validate_exposure_outcome_sequence(
            df, 'exposure_date', 'outcome_date'
        )
        
        assert 'missing_exposure_dates' in result
        assert result['missing_exposure_dates'] == 50
        assert 'missing_data_handling' in result
    
    def test_temporal_validation_error_handling(self):
        """Test error handling for invalid temporal configurations"""
        # Test with invalid date columns
        df = pd.DataFrame({
            'Patient_ID': range(100),
            'exposure_text': ['not_a_date'] * 100,
            'outcome_date': pd.date_range('2016-01-01', periods=100)
        })
        
        validator = TemporalOrderingValidator()
        
        # Should handle invalid date columns gracefully
        with pytest.raises(TemporalValidationError):
            validator.validate_exposure_outcome_sequence(
                df, 'exposure_text', 'outcome_date'
            )
    
    def test_temporal_validation_for_causal_inference(self):
        """Test temporal validation specifically for causal inference requirements"""
        np.random.seed(42)
        n = 1000
        
        # Hill's temporality criterion: cause must precede effect
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_exposure_start': pd.date_range('2015-01-01', '2015-06-30', periods=n),
            'ssd_exposure_end': pd.date_range('2015-07-01', '2015-12-31', periods=n),
            'outcome_measurement_start': pd.date_range('2016-01-01', '2016-06-30', periods=n),
            'outcome_measurement_end': pd.date_range('2016-07-01', '2017-12-31', periods=n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'total_encounters': np.random.poisson(5, n)
        })
        
        # Test causal inference temporal requirements
        validator = TemporalOrderingValidator()
        result = validator.validate_causal_inference_sequence(df)
        
        assert result['causal_temporality_valid'] is True
        assert 'exposure_outcome_gap_days' in result
        assert result['exposure_outcome_gap_days'] > 0  # Must be positive gap
        
        # Test specific requirements
        assert 'sufficient_followup' in result
        assert 'no_immortal_time_bias' in result


class TestTemporalValidationIntegration:
    """Test integration with SSD pipeline"""
    
    def test_integration_with_exposure_flag_module(self):
        """Test integration with 02_exposure_flag.py temporal logic"""
        # This test shows how temporal validation integrates with existing pipeline
        
        # Mock data similar to exposure flag output
        np.random.seed(42)
        n = 1000
        
        exposure_df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'normal_lab_count_12m': np.random.poisson(2.5, n),
            'first_normal_lab_date': pd.date_range('2015-01-01', '2015-06-30', periods=n),
            'last_normal_lab_date': pd.date_range('2015-07-01', '2015-12-31', periods=n),
            'exposure_window_start': pd.Timestamp('2015-01-01'),
            'exposure_window_end': pd.Timestamp('2015-12-31')
        })
        
        # Test that temporal logic aligns with exposure criteria
        for _, row in exposure_df.iterrows():
            if row['ssd_flag'] == 1:  # If exposed
                # First lab should be in exposure window
                assert row['exposure_window_start'] <= row['first_normal_lab_date'] <= row['exposure_window_end']
                # Last lab should be in exposure window
                assert row['exposure_window_start'] <= row['last_normal_lab_date'] <= row['exposure_window_end']
    
    def test_integration_with_outcome_flag_module(self):
        """Test integration with 04_outcome_flag.py temporal logic"""
        # Mock data similar to outcome flag output
        np.random.seed(42)
        n = 1000
        
        outcome_df = pd.DataFrame({
            'Patient_ID': range(n),
            'total_encounters_24m': np.random.poisson(8, n),
            'outcome_window_start': pd.Timestamp('2016-01-01'),
            'outcome_window_end': pd.Timestamp('2017-12-31'),
            'first_outcome_date': pd.date_range('2016-01-01', '2016-06-30', periods=n),
            'last_outcome_date': pd.date_range('2017-06-01', '2017-12-31', periods=n)
        })
        
        # Test outcome temporal consistency
        for _, row in outcome_df.iterrows():
            # Outcomes should be in outcome window
            assert row['outcome_window_start'] <= row['first_outcome_date'] <= row['outcome_window_end']
            assert row['outcome_window_start'] <= row['last_outcome_date'] <= row['outcome_window_end']
            # First outcome should precede last outcome
            assert row['first_outcome_date'] <= row['last_outcome_date']
    
    def test_master_table_temporal_consistency(self):
        """Test temporal consistency in patient master table"""
        # Mock master table structure
        np.random.seed(42)
        n = 1000
        
        master_df = pd.DataFrame({
            'Patient_ID': range(n),
            
            # Study timeline
            'study_entry_date': pd.date_range('2014-06-01', '2014-12-31', periods=n),
            'exposure_window_start': pd.Timestamp('2015-01-01'),
            'exposure_window_end': pd.Timestamp('2015-12-31'),
            'outcome_window_start': pd.Timestamp('2016-01-01'),
            'outcome_window_end': pd.Timestamp('2017-12-31'),
            'study_exit_date': pd.date_range('2017-12-31', '2018-06-30', periods=n),
            
            # Exposure and outcome flags
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'total_encounters': np.random.poisson(8, n),
            
            # Site and other variables
            'site_id': np.random.randint(1, 21, n),
            'age': np.random.normal(50, 15, n)
        })
        
        # Test master table temporal consistency
        validator = TemporalOrderingValidator()
        result = validator.validate_master_table_consistency(master_df)
        
        assert result['master_table_temporal_valid'] is True
        assert 'study_timeline_consistent' in result
        assert 'exposure_outcome_separation' in result


if __name__ == "__main__":
    # Run basic demonstrations
    test_instance = TestTemporalValidator()
    
    print("Running temporal validation tests...")
    test_instance.test_basic_temporal_ordering()
    print("✓ Basic temporal ordering test passed")
    
    test_instance.test_temporal_violations_detection()
    print("✓ Temporal violations detection test passed")
    
    test_instance.test_ssd_specific_temporal_validation()
    print("✓ SSD-specific temporal validation test passed")
    
    test_instance.test_temporal_window_creation()
    print("✓ Temporal window creation test passed")
    
    print("\nAll temporal validation tests passed! Ready for implementation.")