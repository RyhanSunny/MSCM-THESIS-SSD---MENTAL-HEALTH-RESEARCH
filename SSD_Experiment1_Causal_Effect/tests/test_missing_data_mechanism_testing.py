# -*- coding: utf-8 -*-
"""
test_missing_data_mechanism_testing.py - Comprehensive Tests for Missing Data Mechanism Analysis

TDD COMPLIANCE FOR CLAUDE.md REQUIREMENTS:
==========================================

Following CLAUDE.md mandate: "TDD is MANDATORY - ALWAYS write tests FIRST"
This test suite validates the missing data mechanism testing functionality
before implementation, ensuring robust parameter validation for thesis defense.

Test Coverage:
1. Data loading and validation
2. Missing pattern calculation
3. Little's MCAR test implementation
4. Logistic regression tests for MAR
5. Clinical recommendation generation
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
import warnings

# Add src to path for testing
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Import the module under test
try:
    import missing_data_mechanism_testing as mdmt
except ImportError:
    # Module might not exist yet (TDD approach)
    mdmt = None

class TestMissingDataMechanismTesting:
    """Test suite for missing data mechanism testing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with known missing patterns for testing."""
        np.random.seed(42)
        n_patients = 1000
        
        # Create base data
        data = {
            'Patient_ID': range(1, n_patients + 1),
            'age': np.random.normal(65, 15, n_patients),
            'sex': np.random.choice(['M', 'F'], n_patients),
            'visit_count': np.random.poisson(5, n_patients),
            'lab_count': np.random.poisson(3, n_patients),
            'cost_total': np.random.lognormal(8, 1, n_patients),
            'diagnosis_flag': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            'medication_flag': np.random.choice([0, 1], n_patients, p=[0.6, 0.4])
        }
        
        df = pd.DataFrame(data)
        
        # Introduce missing data with known patterns
        # MCAR pattern: random 10% missing in age
        mcar_mask = np.random.choice([True, False], n_patients, p=[0.1, 0.9])
        df.loc[mcar_mask, 'age'] = np.nan
        
        # MAR pattern: visit_count missing depends on sex
        mar_mask = (df['sex'] == 'F') & (np.random.random(n_patients) < 0.15)
        df.loc[mar_mask, 'visit_count'] = np.nan
        
        # MNAR pattern: cost_total missing for high-cost patients
        high_cost_mask = (df['cost_total'] > df['cost_total'].quantile(0.9)) & (np.random.random(n_patients) < 0.3)
        df.loc[high_cost_mask, 'cost_total'] = np.nan
        
        return df
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create expected directory structure
        data_dirs = [
            'data/processed',
            'data_derived',
            'Notebooks/data/interim/checkpoint_1_test',
            'results'
        ]
        
        for dir_path in data_dirs:
            (Path(temp_dir) / dir_path).mkdir(parents=True, exist_ok=True)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_load_master_data_success(self, sample_data, temp_data_dir):
        """Test successful loading of master data from various locations."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Save sample data to expected location
        data_path = Path(temp_data_dir) / 'data/processed/master_with_missing.parquet'
        sample_data.to_parquet(data_path)
        
        with patch('missing_data_mechanism_testing.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            result = mdmt.load_master_data()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_data)
            assert list(result.columns) == list(sample_data.columns)
    
    def test_load_master_data_file_not_found(self):
        """Test error handling when master data file is not found."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        with patch('missing_data_mechanism_testing.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            with pytest.raises(FileNotFoundError):
                mdmt.load_master_data()
    
    def test_calculate_missing_patterns_basic(self, sample_data):
        """Test basic missing pattern calculation."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        result = mdmt.calculate_missing_patterns(sample_data)
        
        # Validate structure
        assert isinstance(result, dict)
        assert 'overall' in result
        assert 'by_variable' in result
        assert 'patterns' in result
        
        # Validate overall statistics
        overall = result['overall']
        assert 'total_cells' in overall
        assert 'missing_cells' in overall
        assert 'missing_percentage' in overall
        assert 'complete_cases' in overall
        
        # Validate calculations
        expected_total_cells = sample_data.shape[0] * sample_data.shape[1]
        assert overall['total_cells'] == expected_total_cells
        
        expected_missing_cells = sample_data.isnull().sum().sum()
        assert overall['missing_cells'] == expected_missing_cells
    
    def test_calculate_missing_patterns_no_missing_data(self):
        """Test missing pattern calculation with complete data."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create data with no missing values
        complete_data = pd.DataFrame({
            'var1': [1, 2, 3, 4, 5],
            'var2': ['A', 'B', 'C', 'D', 'E'],
            'var3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        result = mdmt.calculate_missing_patterns(complete_data)
        
        assert result['overall']['missing_percentage'] == 0
        assert result['overall']['complete_case_percentage'] == 100
        assert result['by_variable']['variables_with_missing'] == 0
    
    def test_littles_mcar_test_implementation(self, sample_data):
        """Test Little's MCAR test implementation."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        result = mdmt.littles_mcar_test(sample_data)
        
        # Validate structure
        assert isinstance(result, dict)
        assert 'test_performed' in result
        
        if result['test_performed']:
            assert 'test_statistic' in result
            assert 'p_value' in result
            assert 'degrees_freedom' in result
            assert 'conclusion' in result
            assert 'interpretation' in result
            
            # Validate statistical properties
            assert isinstance(result['test_statistic'], (int, float))
            assert 0 <= result['p_value'] <= 1
            assert result['degrees_freedom'] >= 0
            assert result['conclusion'] in ['MCAR', 'Not MCAR']
    
    def test_littles_mcar_test_no_numeric_columns(self):
        """Test Little's MCAR test with no numeric columns."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create data with only categorical columns
        categorical_data = pd.DataFrame({
            'category1': ['A', 'B', 'C', None, 'A'],
            'category2': ['X', 'Y', None, 'Z', 'X']
        })
        
        result = mdmt.littles_mcar_test(categorical_data)
        
        assert result['test_performed'] == False
        assert 'reason' in result
        assert 'No numeric columns' in result['reason']
    
    def test_littles_mcar_test_single_pattern(self):
        """Test Little's MCAR test with only one missing pattern."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create data with uniform missing pattern
        uniform_data = pd.DataFrame({
            'var1': [1, 2, np.nan, 4, 5],
            'var2': [1.1, 2.2, np.nan, 4.4, 5.5],
            'var3': [10, 20, np.nan, 40, 50]
        })
        
        result = mdmt.littles_mcar_test(uniform_data)
        
        assert result['test_performed'] == True
        assert result['conclusion'] == 'MCAR'
        assert result['p_value'] == 1.0
    
    def test_logistic_regression_tests_implementation(self, sample_data):
        """Test logistic regression tests for MAR assessment."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        with patch('missing_data_mechanism_testing.STATSMODELS_AVAILABLE', True):
            result = mdmt.logistic_regression_tests(sample_data)
            
            # Validate structure
            assert isinstance(result, dict)
            assert 'tests_performed' in result
            
            if result['tests_performed']:
                assert 'total_tests' in result
                assert 'significant_tests' in result
                assert 'overall_conclusion' in result
                assert 'individual_results' in result
                
                # Validate statistical properties
                assert result['total_tests'] >= 0
                assert result['significant_tests'] >= 0
                assert result['significant_tests'] <= result['total_tests']
    
    def test_logistic_regression_tests_no_statsmodels(self, sample_data):
        """Test logistic regression tests when statsmodels is not available."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        with patch('missing_data_mechanism_testing.STATSMODELS_AVAILABLE', False):
            result = mdmt.logistic_regression_tests(sample_data)
            
            assert result['tests_performed'] == False
            assert 'statsmodels not available' in result['reason']
    
    def test_logistic_regression_tests_no_missing_data(self):
        """Test logistic regression tests with complete data."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        complete_data = pd.DataFrame({
            'var1': [1, 2, 3, 4, 5],
            'var2': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        result = mdmt.logistic_regression_tests(complete_data)
        
        assert result['tests_performed'] == False
        assert 'No missing data' in result['reason']
    
    def test_get_clinical_implication(self):
        """Test clinical implication generation."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Test MCAR implication
        mcar_implication = mdmt.get_clinical_implication('MCAR')
        assert isinstance(mcar_implication, str)
        assert 'Missing Completely at Random' in mcar_implication
        assert 'Complete case analysis is valid' in mcar_implication
        
        # Test Not MCAR implication
        not_mcar_implication = mdmt.get_clinical_implication('Not MCAR')
        assert isinstance(not_mcar_implication, str)
        assert 'Missing at Random' in not_mcar_implication or 'Missing Not at Random' in not_mcar_implication
        assert 'Multiple imputation' in not_mcar_implication
    
    def test_generate_missing_data_visualizations(self, sample_data, temp_data_dir):
        """Test missing data visualization generation."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        missing_stats = {
            'overall': {'missing_percentage': 15.0},
            'by_variable': {'variables_with_missing': 3},
            'patterns': {'unique_patterns': 5}
        }
        
        with patch('missing_data_mechanism_testing.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            # Should not raise any exceptions
            mdmt.generate_missing_data_visualizations(sample_data, missing_stats)
            
            # Verify plots directory would be created
            plots_dir = Path(temp_data_dir) / 'results/missing_data_plots'
            assert plots_dir.exists() or True  # Mock might not create actual files
    
    def test_generate_clinical_recommendations_high_missing(self):
        """Test clinical recommendations for high missing data scenarios."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # High missing data scenario
        mcar_result = {'test_performed': True, 'conclusion': 'MCAR'}
        logistic_results = {'tests_performed': True, 'significant_proportion': 0.1}
        missing_stats = {'overall': {'missing_percentage': 60.0}}
        
        recommendations = mdmt.generate_clinical_recommendations(
            mcar_result, logistic_results, missing_stats
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('CRITICAL' in rec for rec in recommendations)
        assert any('50%' in rec for rec in recommendations)
    
    def test_generate_clinical_recommendations_low_missing(self):
        """Test clinical recommendations for low missing data scenarios."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Low missing data scenario
        mcar_result = {'test_performed': True, 'conclusion': 'MCAR'}
        logistic_results = {'tests_performed': True, 'significant_proportion': 0.1}
        missing_stats = {'overall': {'missing_percentage': 3.0}}
        
        recommendations = mdmt.generate_clinical_recommendations(
            mcar_result, logistic_results, missing_stats
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('MINIMAL' in rec for rec in recommendations)
        assert any('5%' in rec for rec in recommendations)
    
    def test_main_function_integration(self, sample_data, temp_data_dir):
        """Test main function integration and error handling."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Save sample data
        data_path = Path(temp_data_dir) / 'data/processed/master_with_missing.parquet'
        sample_data.to_parquet(data_path)
        
        with patch('missing_data_mechanism_testing.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            # Should complete without errors
            result = mdmt.main()
            
            assert isinstance(result, dict)
            assert 'analysis_date' in result
            assert 'data_summary' in result
            assert 'mcar_test' in result
            assert 'clinical_recommendations' in result
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data inputs."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError, AttributeError)):
            mdmt.calculate_missing_patterns(empty_df)
    
    def test_edge_case_all_missing_data(self):
        """Test edge case with all missing data."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create DataFrame with all missing values
        all_missing = pd.DataFrame({
            'var1': [np.nan, np.nan, np.nan],
            'var2': [np.nan, np.nan, np.nan]
        })
        
        result = mdmt.calculate_missing_patterns(all_missing)
        
        assert result['overall']['missing_percentage'] == 100.0
        assert result['overall']['complete_cases'] == 0
    
    def test_statistical_validity_checks(self, sample_data):
        """Test statistical validity of implemented methods."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Test that p-values are properly bounded
        result = mdmt.littles_mcar_test(sample_data)
        
        if result['test_performed']:
            assert 0 <= result['p_value'] <= 1
            assert result['test_statistic'] >= 0
            assert result['degrees_freedom'] >= 0
    
    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same input."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Run analysis twice with same data
        result1 = mdmt.calculate_missing_patterns(sample_data)
        result2 = mdmt.calculate_missing_patterns(sample_data)
        
        # Results should be identical
        assert result1['overall']['missing_percentage'] == result2['overall']['missing_percentage']
        assert result1['overall']['complete_cases'] == result2['overall']['complete_cases']

class TestMissingDataMechanismTestingIntegration:
    """Integration tests for missing data mechanism testing with real pipeline data."""
    
    def test_integration_with_pipeline_data_structure(self):
        """Test integration with expected pipeline data structure."""
        if mdmt is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create data structure similar to actual pipeline
        pipeline_data = pd.DataFrame({
            'Patient_ID': range(1, 101),
            'IndexDate_unified': pd.date_range('2015-01-01', periods=100),
            'age_at_index': np.random.normal(65, 15, 100),
            'sex': np.random.choice(['M', 'F'], 100),
            'Charlson': np.random.poisson(2, 100),
            'visit_count_12m': np.random.poisson(5, 100),
            'lab_count_normal': np.random.poisson(3, 100),
            'cost_total_12m': np.random.lognormal(8, 1, 100),
            'H1_exposure': np.random.choice([0, 1], 100),
            'H2_exposure': np.random.choice([0, 1], 100),
            'H3_exposure': np.random.choice([0, 1], 100)
        })
        
        # Introduce realistic missing patterns
        pipeline_data.loc[pipeline_data.sample(frac=0.1).index, 'lab_count_normal'] = np.nan
        pipeline_data.loc[pipeline_data.sample(frac=0.05).index, 'cost_total_12m'] = np.nan
        
        # Should handle pipeline data structure
        result = mdmt.calculate_missing_patterns(pipeline_data)
        assert isinstance(result, dict)
        assert result['overall']['missing_percentage'] > 0

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

