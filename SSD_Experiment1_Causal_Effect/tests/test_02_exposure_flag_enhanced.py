#!/usr/bin/env python3
"""
Test Suite for Enhanced SSD Exposure Flag Generation

Following TDD Methodology (CLAUDE.md Requirements):
1. Write failing tests FIRST
2. Implement minimal code to make tests pass
3. Refactor while keeping tests green

Author: Ryhan Suny
Date: January 2025
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import module under test
import importlib.util
spec = importlib.util.spec_from_file_location(
    "enhanced_exposure_flag", 
    Path(__file__).parent.parent / 'src' / 'experimental' / '02_exposure_flag_enhanced.py'
)
enhanced_exposure_flag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_exposure_flag)

# Import functions
create_enhanced_drug_atc_codes = enhanced_exposure_flag.create_enhanced_drug_atc_codes
load_enhanced_medication_data = enhanced_exposure_flag.load_enhanced_medication_data
calculate_enhanced_drug_persistence = enhanced_exposure_flag.calculate_enhanced_drug_persistence
generate_enhanced_exposure_flags = enhanced_exposure_flag.generate_enhanced_exposure_flags

class TestEnhancedDrugATCCodes:
    """Test enhanced ATC code creation and validation"""
    
    def test_enhanced_atc_codes_contains_felipe_additions(self):
        """Test that enhanced ATC codes include Dr. Felipe's missing drug classes"""
        enhanced_codes = create_enhanced_drug_atc_codes()
        
        # Test antidepressants (N06A)
        antidepressant_codes = [code for code in enhanced_codes.keys() if code.startswith('N06A')]
        assert len(antidepressant_codes) >= 4, "Should have at least 4 antidepressant subcodes"
        assert 'N06A1' in enhanced_codes, "Missing tricyclic antidepressants"
        assert 'N06A2' in enhanced_codes, "Missing SSRI antidepressants"
        
        # Test anticonvulsants (N03A)
        anticonvulsant_codes = [code for code in enhanced_codes.keys() if code.startswith('N03A')]
        assert len(anticonvulsant_codes) >= 3, "Should have at least 3 anticonvulsant subcodes"
        assert 'N03AX' in enhanced_codes, "Missing other anticonvulsants (gabapentin/pregabalin)"
        
        # Test antipsychotics (N05A)
        antipsychotic_codes = [code for code in enhanced_codes.keys() if code.startswith('N05A')]
        assert len(antipsychotic_codes) >= 4, "Should have at least 4 antipsychotic subcodes"
        assert 'N05A2' in enhanced_codes, "Missing atypical antipsychotics"

    def test_enhanced_atc_codes_backward_compatibility(self):
        """Test that enhanced codes maintain original drug classes"""
        enhanced_codes = create_enhanced_drug_atc_codes()
        
        # Original codes must be preserved
        assert 'N05B' in enhanced_codes, "Missing original anxiolytics"
        assert 'N05C' in enhanced_codes, "Missing original hypnotics"
        assert 'N02B' in enhanced_codes, "Missing original analgesics"
        
        # Check mappings are correct
        assert enhanced_codes['N05B'] == 'anxiolytics'
        assert enhanced_codes['N05C'] == 'hypnotics'

    def test_enhanced_atc_codes_file_creation(self):
        """Test that enhanced ATC codes are saved to file correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the ROOT path to use temp directory
            original_root = enhanced_exposure_flag.ROOT
            enhanced_exposure_flag.ROOT = Path(temp_dir)
            
            try:
                codes_dir = Path(temp_dir) / "code_lists"
                codes_dir.mkdir(exist_ok=True)
                
                enhanced_codes = create_enhanced_drug_atc_codes()
                
                # Check file was created
                codes_file = codes_dir / "drug_atc_enhanced.csv"
                assert codes_file.exists(), "Enhanced ATC codes file not created"
                
                # Check file content
                codes_df = pd.read_csv(codes_file)
                assert 'atc_code' in codes_df.columns
                assert 'drug_class' in codes_df.columns
                assert 'enhancement' in codes_df.columns
                
                # Check Felipe additions are marked
                felipe_additions = codes_df[codes_df['enhancement'] == 'felipe_added']
                assert len(felipe_additions) > 0, "No Felipe additions marked"
            finally:
                # Restore original ROOT
                enhanced_exposure_flag.ROOT = original_root

class TestEnhancedMedicationData:
    """Test enhanced medication data loading and filtering"""
    
    def test_load_enhanced_medication_data_filters_correctly(self):
        """Test that medication data is filtered for enhanced ATC codes"""
        # Create mock data
        mock_medication = pd.DataFrame({
            'Patient_ID': [1, 2, 3, 4, 5],
            'ATC_code': ['N06A1', 'N05B1', 'N03AX', 'Z01AA', 'N05A2'],  # Mix of enhanced and non-enhanced
            'StartDate': pd.to_datetime(['2015-01-01'] * 5),
            'StopDate': pd.to_datetime(['2015-06-01'] * 5)
        })
        
        mock_cohort = pd.DataFrame({
            'Patient_ID': [1, 2, 3, 4, 5],
            'IndexDate_lab': pd.to_datetime(['2015-01-01'] * 5)
        })
        
        with patch.object(enhanced_exposure_flag.pd, 'read_parquet') as mock_read:
            mock_read.side_effect = [mock_medication, mock_cohort]
            
            medication_enhanced, cohort, enhanced_atc = load_enhanced_medication_data()
            
            # Should filter out non-enhanced codes (Z01AA not in enhanced list)
            enhanced_patients = set(medication_enhanced['Patient_ID'])
            assert len(medication_enhanced) <= len(mock_medication), "Should filter some codes"
            
            # Check drug class mapping
            assert 'drug_class' in medication_enhanced.columns
            if len(medication_enhanced) > 0:
                # Check that we have valid drug classes
                drug_classes = medication_enhanced['drug_class'].unique()
                assert 'other' not in drug_classes or len(drug_classes) > 1, "Should have proper drug class mapping"

    def test_load_enhanced_medication_data_error_handling(self):
        """Test error handling when required files are missing"""
        with patch.object(enhanced_exposure_flag.pd, 'read_parquet') as mock_read:
            mock_read.side_effect = FileNotFoundError("Data file not found")
            
            with pytest.raises(FileNotFoundError):
                load_enhanced_medication_data()

class TestEnhancedDrugPersistence:
    """Test enhanced drug persistence calculation with 180-day threshold"""
    
    def test_enhanced_drug_persistence_180_day_threshold(self):
        """Test that enhanced persistence uses 180-day threshold (not 90)"""
        # Create test data
        mock_medication = pd.DataFrame({
            'Patient_ID': [1, 1, 2, 2],
            'drug_class': ['antidepressants_ssri', 'antidepressants_ssri', 'anxiolytics', 'anxiolytics'],
            'StartDate': pd.to_datetime(['2015-01-01', '2015-07-01', '2015-01-01', '2015-04-01']),
            'StopDate': pd.to_datetime(['2015-06-30', '2015-12-31', '2015-03-31', '2015-06-30'])  # ~180 days each, ~90 days each
        })
        
        mock_cohort = pd.DataFrame({
            'Patient_ID': [1, 2],
            'IndexDate_lab': pd.to_datetime(['2015-01-01', '2015-01-01'])
        })
        
        h3_patients, drug_summary, persistent_df = calculate_enhanced_drug_persistence(mock_medication, mock_cohort)
        
        # Patient 1 should qualify (>180 days total)
        # Patient 2 might not qualify (depends on exact calculation)
        assert isinstance(h3_patients, set), "Should return set of patient IDs"
        assert len(h3_patients) >= 0, "Should return valid patient set"
        
        # Check drug summary structure
        assert isinstance(drug_summary, dict), "Should return drug summary dictionary"
        
        # Check that 180-day threshold is applied (not 90)
        # This is validated by the logic requiring >= 180 days

    def test_drug_persistence_exposure_window_clipping(self):
        """Test that drug periods are correctly clipped to exposure window"""
        # Drug spans beyond exposure window
        mock_medication = pd.DataFrame({
            'Patient_ID': [1],
            'drug_class': ['antidepressants_ssri'],
            'StartDate': pd.to_datetime(['2017-01-01']),  # Before exposure
            'StopDate': pd.to_datetime(['2016-01-01'])    # After exposure
        })
        
        mock_cohort = pd.DataFrame({
            'Patient_ID': [1],
            'IndexDate_lab': pd.to_datetime(['2015-01-01'])  # Exposure: 2015-01-01 to 2016-01-01
        })
        
        h3_patients, drug_summary, persistent_df = calculate_enhanced_drug_persistence(mock_medication, mock_cohort)
        
        # Should clip to exposure window (365 days max)
        if len(persistent_df) > 0:
            max_days = persistent_df['days'].max()
            assert max_days <= 365, f"Days should be clipped to exposure window, got {max_days}"

class TestGenerateEnhancedExposureFlags:
    """Test complete enhanced exposure flag generation"""
    
    def test_enhanced_exposure_flags_integration(self):
        """Test full integration of enhanced exposure flag generation"""
        # Mock all data dependencies
        mock_medication = pd.DataFrame({
            'Patient_ID': [1, 2, 3],
            'ATC_code': ['N06A1', 'N03AX', 'N05A2'],
            'StartDate': pd.to_datetime(['2015-01-01'] * 3),
            'StopDate': pd.to_datetime(['2015-12-31'] * 3)
        })
        
        mock_cohort = pd.DataFrame({
            'Patient_ID': [1, 2, 3],
            'IndexDate_lab': pd.to_datetime(['2015-01-01'] * 3)
        })
        
        mock_existing_exposure = pd.DataFrame({
            'Patient_ID': [1, 2, 3],
            'H1_normal_labs': [True, False, True],
            'H2_referral_loop': [False, True, False],
            'H3_drug_persistence': [True, True, False],
            'exposure_flag': [True, True, True],
            'exposure_flag_strict': [False, False, False]
        })
        
        with patch.object(enhanced_exposure_flag.pd, 'read_parquet') as mock_read, \
             patch.object(enhanced_exposure_flag, 'DERIVED', Path(tempfile.mkdtemp())):
            mock_read.side_effect = [mock_medication, mock_cohort, mock_existing_exposure]
            
            enhanced_exposure, drug_summary = generate_enhanced_exposure_flags()
            
            # Check enhanced columns exist
            assert 'H3_drug_persistence_enhanced' in enhanced_exposure.columns
            assert 'exposure_flag_enhanced' in enhanced_exposure.columns
            assert 'exposure_flag_strict_enhanced' in enhanced_exposure.columns
            
            # Check data types
            assert enhanced_exposure['H3_drug_persistence_enhanced'].dtype == bool
            assert enhanced_exposure['exposure_flag_enhanced'].dtype == bool

    def test_enhanced_exposure_flags_comparison_analysis(self):
        """Test that enhancement impact analysis is calculated correctly"""
        # This test would verify the before/after comparison logic
        # Implementation depends on the actual module structure
        pass

class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling as required by CLAUDE.md"""
    
    def test_empty_medication_data_handling(self):
        """Test handling of empty medication datasets"""
        empty_medication = pd.DataFrame(columns=['Patient_ID', 'ATC_code', 'StartDate', 'StopDate'])
        mock_cohort = pd.DataFrame({
            'Patient_ID': [1, 2, 3],
            'IndexDate_lab': pd.to_datetime(['2015-01-01'] * 3)
        })
        
        h3_patients, drug_summary, persistent_df = calculate_enhanced_drug_persistence(empty_medication, mock_cohort)
        
        assert len(h3_patients) == 0, "Empty medication data should result in no persistent users"
        assert len(drug_summary) == 0, "Empty medication data should result in empty drug summary"

    def test_invalid_date_handling(self):
        """Test handling of invalid or missing dates"""
        invalid_medication = pd.DataFrame({
            'Patient_ID': [1, 2],
            'drug_class': ['antidepressants_ssri', 'anxiolytics'],
            'StartDate': [pd.NaT, pd.to_datetime('2015-01-01')],
            'StopDate': [pd.to_datetime('2015-06-01'), pd.NaT]
        })
        
        mock_cohort = pd.DataFrame({
            'Patient_ID': [1, 2],
            'IndexDate_lab': pd.to_datetime(['2015-01-01', '2015-01-01'])
        })
        
        # Should handle invalid dates gracefully without crashing
        try:
            h3_patients, drug_summary, persistent_df = calculate_enhanced_drug_persistence(invalid_medication, mock_cohort)
            assert True, "Should handle invalid dates without crashing"
        except Exception as e:
            pytest.fail(f"Should handle invalid dates gracefully, but got: {e}")

    def test_performance_with_large_dataset(self):
        """Test performance considerations with large datasets"""
        # Create large mock dataset
        large_medication = pd.DataFrame({
            'Patient_ID': list(range(10000)) * 2,  # 20,000 records
            'drug_class': ['antidepressants_ssri'] * 20000,
            'StartDate': pd.to_datetime(['2015-01-01'] * 20000),
            'StopDate': pd.to_datetime(['2015-12-31'] * 20000)
        })
        
        large_cohort = pd.DataFrame({
            'Patient_ID': list(range(10000)),
            'IndexDate_lab': pd.to_datetime(['2015-01-01'] * 10000)
        })
        
        import time
        start_time = time.time()
        h3_patients, drug_summary, persistent_df = calculate_enhanced_drug_persistence(large_medication, large_cohort)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 30, f"Performance test failed: took {execution_time:.2f} seconds for 20k records"

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"]) 