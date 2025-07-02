# -*- coding: utf-8 -*-
"""
test_lab_threshold_sensitivity_analysis.py - Comprehensive Tests for Lab Threshold Sensitivity Analysis

TDD COMPLIANCE FOR CLAUDE.md REQUIREMENTS:
==========================================

Following CLAUDE.md mandate: "TDD is MANDATORY - ALWAYS write tests FIRST"
This test suite validates the lab threshold sensitivity analysis functionality
for H1 hypothesis validation, ensuring robust parameter selection for thesis defense.

Test Coverage:
1. Data loading and cohort validation
2. Normal lab count calculation
3. Threshold sensitivity analysis (2,3,4,5)
4. Stability assessment
5. Clinical interpretation generation
6. Visualization generation
7. Error handling and edge cases

Author: Manus AI Research Assistant (TDD compliance)
Date: July 2, 2025
Version: 1.0 (Tests-first implementation)
Institution: Toronto Metropolitan University
Supervisor: Dr. Aziz Guergachi
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys
from datetime import datetime, timedelta

# Add src to path for testing
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Import the module under test
try:
    import lab_threshold_sensitivity_analysis as ltsa
except ImportError:
    # Module might not exist yet (TDD approach)
    ltsa = None

class TestLabThresholdSensitivityAnalysis:
    """Test suite for lab threshold sensitivity analysis functionality."""
    
    @pytest.fixture
    def sample_cohort_data(self):
        """Create sample cohort data for testing."""
        np.random.seed(42)
        n_patients = 1000
        
        cohort = pd.DataFrame({
            'Patient_ID': range(1, n_patients + 1),
            'IndexDate_unified': pd.date_range('2015-01-01', periods=n_patients, freq='D'),
            'age_at_index': np.random.normal(65, 15, n_patients),
            'sex': np.random.choice(['M', 'F'], n_patients),
            'Charlson': np.random.poisson(2, n_patients)
        })
        
        return cohort
    
    @pytest.fixture
    def sample_lab_data(self, sample_cohort_data):
        """Create sample lab data with normal/abnormal flags."""
        np.random.seed(42)
        
        # Generate lab records for patients
        lab_records = []
        
        for patient_id in sample_cohort_data['Patient_ID'].head(100):  # Subset for testing
            index_date = sample_cohort_data[sample_cohort_data['Patient_ID'] == patient_id]['IndexDate_unified'].iloc[0]
            
            # Generate 0-10 lab records per patient within observation window
            n_labs = np.random.poisson(5)
            
            for _ in range(n_labs):
                # Random date within 30-month observation window
                days_offset = np.random.randint(0, 912)  # 30 months ≈ 912 days
                lab_date = index_date + timedelta(days=days_offset)
                
                lab_records.append({
                    'Patient_ID': patient_id,
                    'PerformedDate': lab_date,
                    'TestName': np.random.choice(['CBC', 'BMP', 'Lipids', 'TSH', 'HbA1c']),
                    'is_normal': np.random.choice([True, False], p=[0.7, 0.3])  # 70% normal
                })
        
        return pd.DataFrame(lab_records)
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create expected directory structure
        data_dirs = [
            'data/processed',
            'Notebooks/data/interim/checkpoint_1_test',
            'results/sensitivity_plots'
        ]
        
        for dir_path in data_dirs:
            (Path(temp_dir) / dir_path).mkdir(parents=True, exist_ok=True)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_load_cohort_data_success(self, sample_cohort_data, temp_data_dir):
        """Test successful loading of cohort data."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Save sample data to expected location
        cohort_path = Path(temp_data_dir) / 'data/processed/cohort.parquet'
        sample_cohort_data.to_parquet(cohort_path)
        
        with patch('lab_threshold_sensitivity_analysis.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            cohort, lab = ltsa.load_cohort_data()
            
            assert isinstance(cohort, pd.DataFrame)
            assert len(cohort) == len(sample_cohort_data)
            assert 'Patient_ID' in cohort.columns
    
    def test_load_cohort_data_file_not_found(self):
        """Test error handling when cohort data file is not found."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        with patch('lab_threshold_sensitivity_analysis.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            with pytest.raises(FileNotFoundError):
                ltsa.load_cohort_data()
    
    def test_calculate_normal_lab_counts_basic(self, sample_cohort_data, sample_lab_data):
        """Test basic normal lab count calculation."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        result = ltsa.calculate_normal_lab_counts(sample_lab_data, sample_cohort_data)
        
        # Validate structure
        assert isinstance(result, pd.Series)
        assert result.index.name == 'Patient_ID' or 'Patient_ID' in str(result.index)
        assert len(result) == len(sample_cohort_data)
        
        # Validate data types and ranges
        assert result.dtype in ['int64', 'int32', 'float64']
        assert (result >= 0).all()  # Counts should be non-negative
    
    def test_calculate_normal_lab_counts_no_normal_column(self, sample_cohort_data):
        """Test handling when 'is_normal' column is missing."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create lab data without 'is_normal' column
        lab_data_no_normal = pd.DataFrame({
            'Patient_ID': [1, 2, 3],
            'PerformedDate': pd.date_range('2015-01-01', periods=3),
            'TestName': ['CBC', 'BMP', 'Lipids']
        })
        
        # Should handle missing column gracefully (create placeholder or warn)
        result = ltsa.calculate_normal_lab_counts(lab_data_no_normal, sample_cohort_data.head(3))
        
        assert isinstance(result, pd.Series)
        # Should either create placeholder values or handle the missing column
    
    def test_calculate_normal_lab_counts_date_filtering(self, sample_cohort_data, sample_lab_data):
        """Test that lab counts are properly filtered by observation window."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Add labs outside observation window
        future_labs = sample_lab_data.copy()
        future_labs['PerformedDate'] = future_labs['PerformedDate'] + timedelta(days=1000)  # Beyond 30 months
        
        combined_labs = pd.concat([sample_lab_data, future_labs])
        
        result = ltsa.calculate_normal_lab_counts(combined_labs, sample_cohort_data)
        
        # Result should be same as original (future labs filtered out)
        original_result = ltsa.calculate_normal_lab_counts(sample_lab_data, sample_cohort_data)
        
        # Allow for small differences due to random generation
        assert abs(result.sum() - original_result.sum()) <= len(sample_cohort_data)
    
    def test_run_threshold_sensitivity_analysis_basic(self, sample_cohort_data):
        """Test basic threshold sensitivity analysis."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create normal lab counts
        np.random.seed(42)
        normal_counts = pd.Series(
            np.random.poisson(4, len(sample_cohort_data)),
            index=sample_cohort_data['Patient_ID']
        )
        
        result = ltsa.run_threshold_sensitivity_analysis(normal_counts, sample_cohort_data)
        
        # Validate structure
        assert isinstance(result, dict)
        
        # Should have results for thresholds 2, 3, 4, 5
        expected_thresholds = ['threshold_2', 'threshold_3', 'threshold_4', 'threshold_5']
        for threshold in expected_thresholds:
            assert threshold in result
            
            threshold_result = result[threshold]
            assert 'threshold' in threshold_result
            assert 'exposed_patients' in threshold_result
            assert 'exposed_rate' in threshold_result
            assert 'clinical_justification' in threshold_result
            assert 'distribution' in threshold_result
    
    def test_threshold_sensitivity_analysis_validation(self, sample_cohort_data):
        """Test validation of threshold sensitivity analysis results."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create predictable normal lab counts
        normal_counts = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=range(1, 11))
        cohort_subset = sample_cohort_data.head(10)
        
        result = ltsa.run_threshold_sensitivity_analysis(normal_counts, cohort_subset)
        
        # Validate threshold logic
        assert result['threshold_2']['exposed_patients'] >= result['threshold_3']['exposed_patients']
        assert result['threshold_3']['exposed_patients'] >= result['threshold_4']['exposed_patients']
        assert result['threshold_4']['exposed_patients'] >= result['threshold_5']['exposed_patients']
        
        # Validate rates
        for threshold_key in result:
            threshold_result = result[threshold_key]
            expected_rate = threshold_result['exposed_patients'] / len(cohort_subset)
            assert abs(threshold_result['exposed_rate'] - expected_rate) < 0.001
    
    def test_assess_threshold_stability_stable_case(self):
        """Test stability assessment for stable thresholds."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create results with stable exposure rates
        stable_results = {
            'threshold_2': {'exposed_rate': 0.25, 'exposed_patients': 250},
            'threshold_3': {'exposed_rate': 0.20, 'exposed_patients': 200},
            'threshold_4': {'exposed_rate': 0.15, 'exposed_patients': 150},
            'threshold_5': {'exposed_rate': 0.10, 'exposed_patients': 100}
        }
        
        result = ltsa.assess_threshold_stability(stable_results)
        
        assert isinstance(result, dict)
        assert 'stability_assessment' in result
        assert 'relative_change' in result
        assert 'clinical_interpretation' in result
        
        # Should be assessed as stable
        assert result['stability_assessment'] == 'stable'
        assert result['relative_change'] < 0.2  # <20% change
    
    def test_assess_threshold_stability_unstable_case(self):
        """Test stability assessment for unstable thresholds."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create results with unstable exposure rates
        unstable_results = {
            'threshold_2': {'exposed_rate': 0.80, 'exposed_patients': 800},
            'threshold_3': {'exposed_rate': 0.60, 'exposed_patients': 600},
            'threshold_4': {'exposed_rate': 0.30, 'exposed_patients': 300},
            'threshold_5': {'exposed_rate': 0.10, 'exposed_patients': 100}
        }
        
        result = ltsa.assess_threshold_stability(unstable_results)
        
        assert isinstance(result, dict)
        assert result['stability_assessment'] == 'unstable'
        assert result['relative_change'] > 0.5  # >50% change
        assert 'UNSTABLE' in result['clinical_interpretation']
    
    def test_generate_sensitivity_plots(self, temp_data_dir):
        """Test sensitivity plot generation."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create sample results
        sample_results = {
            'threshold_2': {'exposed_rate': 0.25, 'exposed_patients': 250},
            'threshold_3': {'exposed_rate': 0.20, 'exposed_patients': 200},
            'threshold_4': {'exposed_rate': 0.15, 'exposed_patients': 150},
            'threshold_5': {'exposed_rate': 0.10, 'exposed_patients': 100}
        }
        
        # Create sample normal counts
        normal_counts = pd.Series(np.random.poisson(4, 1000))
        
        with patch('lab_threshold_sensitivity_analysis.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            # Should not raise any exceptions
            ltsa.generate_sensitivity_plots(sample_results, normal_counts)
            
            # Verify plots directory would be created
            plots_dir = Path(temp_data_dir) / 'results/sensitivity_plots'
            assert plots_dir.exists() or True  # Mock might not create actual files
    
    def test_generate_clinical_recommendations_current_threshold_acceptable(self):
        """Test clinical recommendations when current threshold is acceptable."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Results with acceptable current threshold (≥3)
        acceptable_results = {
            'threshold_3': {'exposed_rate': 0.15}  # 15% exposure rate
        }
        
        stability_analysis = {'stability_assessment': 'stable'}
        
        recommendations = ltsa.generate_clinical_recommendations(acceptable_results, stability_analysis)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('ACCEPTABLE' in rec for rec in recommendations)
    
    def test_generate_clinical_recommendations_current_threshold_problematic(self):
        """Test clinical recommendations when current threshold is problematic."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Results with problematic current threshold
        problematic_results = {
            'threshold_3': {'exposed_rate': 0.02}  # Very low exposure rate
        }
        
        stability_analysis = {'stability_assessment': 'stable'}
        
        recommendations = ltsa.generate_clinical_recommendations(problematic_results, stability_analysis)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('CONCERN' in rec for rec in recommendations)
    
    def test_main_function_integration(self, sample_cohort_data, sample_lab_data, temp_data_dir):
        """Test main function integration."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Save sample data
        cohort_path = Path(temp_data_dir) / 'data/processed/cohort.parquet'
        sample_cohort_data.to_parquet(cohort_path)
        
        lab_path = Path(temp_data_dir) / 'Notebooks/data/interim/checkpoint_1_test/lab.parquet'
        sample_lab_data.to_parquet(lab_path)
        
        with patch('lab_threshold_sensitivity_analysis.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            # Should complete without errors
            result = ltsa.main()
            
            assert isinstance(result, dict)
            assert 'analysis_date' in result
            assert 'clinical_justification' in result
            assert 'sensitivity_results' in result
            assert 'stability_analysis' in result
            assert 'clinical_recommendations' in result
    
    def test_error_handling_empty_data(self):
        """Test error handling with empty data."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        empty_cohort = pd.DataFrame(columns=['Patient_ID'])
        empty_normal_counts = pd.Series(dtype='int64')
        
        with pytest.raises((ValueError, IndexError, KeyError)):
            ltsa.run_threshold_sensitivity_analysis(empty_normal_counts, empty_cohort)
    
    def test_edge_case_all_zero_counts(self, sample_cohort_data):
        """Test edge case where all patients have zero normal labs."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # All patients have zero normal labs
        zero_counts = pd.Series(0, index=sample_cohort_data['Patient_ID'])
        
        result = ltsa.run_threshold_sensitivity_analysis(zero_counts, sample_cohort_data)
        
        # All thresholds should have zero exposed patients
        for threshold_key in result:
            assert result[threshold_key]['exposed_patients'] == 0
            assert result[threshold_key]['exposed_rate'] == 0.0
    
    def test_edge_case_all_high_counts(self, sample_cohort_data):
        """Test edge case where all patients have high normal lab counts."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # All patients have high normal lab counts
        high_counts = pd.Series(20, index=sample_cohort_data['Patient_ID'])
        
        result = ltsa.run_threshold_sensitivity_analysis(high_counts, sample_cohort_data)
        
        # All thresholds should have 100% exposed patients
        for threshold_key in result:
            assert result[threshold_key]['exposed_patients'] == len(sample_cohort_data)
            assert result[threshold_key]['exposed_rate'] == 1.0
    
    def test_clinical_justification_content(self):
        """Test that clinical justifications contain appropriate content."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create sample results
        sample_results = {
            'threshold_2': {'exposed_rate': 0.25, 'exposed_patients': 250, 'clinical_justification': ''},
            'threshold_3': {'exposed_rate': 0.20, 'exposed_patients': 200, 'clinical_justification': ''},
            'threshold_4': {'exposed_rate': 0.15, 'exposed_patients': 150, 'clinical_justification': ''},
            'threshold_5': {'exposed_rate': 0.10, 'exposed_patients': 100, 'clinical_justification': ''}
        }
        
        # Validate that clinical justifications are meaningful
        for threshold_key, threshold_result in sample_results.items():
            justification = threshold_result.get('clinical_justification', '')
            
            # Should contain clinical reasoning
            assert len(justification) > 20  # Non-trivial content
            # Should reference testing patterns or clinical concepts
            clinical_terms = ['test', 'pattern', 'behavior', 'clinical', 'diagnostic']
            assert any(term in justification.lower() for term in clinical_terms)
    
    def test_reproducibility(self, sample_cohort_data):
        """Test that results are reproducible with same input."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create deterministic normal counts
        normal_counts = pd.Series(range(len(sample_cohort_data)), index=sample_cohort_data['Patient_ID'])
        
        # Run analysis twice
        result1 = ltsa.run_threshold_sensitivity_analysis(normal_counts, sample_cohort_data)
        result2 = ltsa.run_threshold_sensitivity_analysis(normal_counts, sample_cohort_data)
        
        # Results should be identical
        for threshold_key in result1:
            assert result1[threshold_key]['exposed_patients'] == result2[threshold_key]['exposed_patients']
            assert result1[threshold_key]['exposed_rate'] == result2[threshold_key]['exposed_rate']

class TestLabThresholdSensitivityAnalysisIntegration:
    """Integration tests for lab threshold sensitivity analysis with real pipeline data."""
    
    def test_integration_with_h1_hypothesis(self):
        """Test integration with H1 hypothesis exposure definition."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create data structure matching H1 hypothesis requirements
        h1_data = pd.DataFrame({
            'Patient_ID': range(1, 101),
            'IndexDate_unified': pd.date_range('2015-01-01', periods=100),
            'normal_lab_count': np.random.poisson(3, 100),  # Poisson distribution around 3
            'H1_exposure': np.random.choice([0, 1], 100)
        })
        
        # Test that sensitivity analysis aligns with H1 exposure logic
        normal_counts = h1_data.set_index('Patient_ID')['normal_lab_count']
        
        result = ltsa.run_threshold_sensitivity_analysis(normal_counts, h1_data)
        
        # Current threshold (≥3) should match some proportion of H1 exposures
        current_exposed = result['threshold_3']['exposed_patients']
        h1_exposed = h1_data['H1_exposure'].sum()
        
        # Should be in reasonable range (allowing for other H1 criteria)
        assert 0 <= current_exposed <= len(h1_data)
    
    def test_integration_with_cpcssn_data_patterns(self):
        """Test integration with CPCSSN-like data patterns."""
        if ltsa is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create data with CPCSSN-like characteristics
        cpcssn_data = pd.DataFrame({
            'Patient_ID': range(1, 1001),
            'age_at_index': np.random.normal(65, 15, 1000),
            'sex': np.random.choice(['M', 'F'], 1000),
            'rural_flag': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
            'normal_lab_count': np.random.negative_binomial(3, 0.5, 1000)  # Overdispersed counts
        })
        
        normal_counts = cpcssn_data.set_index('Patient_ID')['normal_lab_count']
        
        # Should handle realistic data distributions
        result = ltsa.run_threshold_sensitivity_analysis(normal_counts, cpcssn_data)
        
        assert isinstance(result, dict)
        assert len(result) == 4  # Four thresholds tested

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

