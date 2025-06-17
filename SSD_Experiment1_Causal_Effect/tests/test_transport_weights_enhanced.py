#!/usr/bin/env python3
"""
test_transport_weights_enhanced.py - Enhanced tests for transport weights

Tests for Week 5 Task C: External-validity weighting finalization
Covers CSV present/missing scenarios and CLI functionality.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
import subprocess

# Import the module we're testing
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from transport_weights import (
    calculate_transport_weights,
    validate_transport_weights,
    create_example_ices_marginals,
    calculate_effective_sample_size,
    main_cli,
    run_transport_analysis
)


class TestTransportWeightsCSVPresent:
    """Test transport weights with CSV file present"""
    
    def test_calculate_weights_with_csv_present(self):
        """Test transport weight calculation with ICES marginals CSV present"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create study data
            study_data = pd.DataFrame({
                'age_group': ['18-34', '35-49', '50-64'] * 100,
                'sex': ['female', 'male', 'female'] * 100,
                'region': ['urban', 'rural', 'urban'] * 100,
                'socioeconomic_quintile': ['q1_lowest', 'q3', 'q5_highest'] * 100
            })
            
            # Create ICES marginals CSV
            marginals_path = tmpdir / 'ices_marginals.csv'
            create_example_ices_marginals(marginals_path)
            
            # Calculate transport weights
            results = calculate_transport_weights(
                study_data, 
                target_marginals_path=marginals_path
            )
            
            assert results['status'] == 'completed'
            assert 'weights' in results
            assert len(results['weights']) == len(study_data)
            assert results['effective_sample_size'] > 0
            assert results['max_weight'] > 0
            assert results['mean_weight'] > 0
            assert 'variables_used' in results
    
    def test_validate_transport_weights_quality(self):
        """Test transport weight quality validation"""
        # Good weights (should pass)
        good_weights = np.array([1.0, 1.1, 0.9, 1.2, 0.8] * 100)
        validation = validate_transport_weights(good_weights)
        
        assert validation['overall_quality'] == True
        assert validation['max_weight_ok'] == True
        assert validation['ess_ratio_ok'] == True
        
        # Bad weights (should fail)
        bad_weights = np.array([1.0, 50.0, 0.1, 1.0, 1.0] * 100)  # Extreme weight
        validation = validate_transport_weights(bad_weights)
        
        assert validation['overall_quality'] == False
        assert validation['max_weight_ok'] == False
    
    def test_effective_sample_size_calculation(self):
        """Test effective sample size calculation using Kish formula"""
        # Equal weights should give ESS = N
        weights = np.ones(100)
        ess = calculate_effective_sample_size(weights)
        assert abs(ess - 100.0) < 1e-10
        
        # Unequal weights should give ESS < N
        weights = np.array([2.0, 0.5, 1.0, 1.5] * 25)
        ess = calculate_effective_sample_size(weights)
        assert ess < 100.0
        assert ess > 0.0


class TestTransportWeightsCSVMissing:
    """Test transport weights with CSV file missing (CI compatibility)"""
    
    def test_calculate_weights_csv_missing(self):
        """Test transport weight calculation with missing ICES marginals"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create study data
            study_data = pd.DataFrame({
                'age_group': ['18-34', '35-49'] * 50,
                'sex': ['female', 'male'] * 50
            })
            
            # Use non-existent CSV path
            missing_csv = tmpdir / 'nonexistent_marginals.csv'
            
            # Calculate transport weights (should return skipped status)
            results = calculate_transport_weights(
                study_data, 
                target_marginals_path=missing_csv
            )
            
            assert results['status'] == 'skipped'
            assert results['reason'] == 'ICES marginals file not available'
            assert len(results['weights']) == len(study_data)
            assert all(w == 1.0 for w in results['weights'])  # Uniform weights
            assert results['effective_sample_size'] == len(study_data)
            assert results['max_weight'] == 1.0
            assert results['mean_weight'] == 1.0
    
    def test_csv_missing_ci_compatibility(self):
        """Test that missing CSV doesn't break CI pipeline"""
        # This should not raise any exceptions
        study_data = pd.DataFrame({'age': [1, 2, 3], 'sex': ['F', 'M', 'F']})
        
        results = calculate_transport_weights(study_data, Path('/nonexistent/path.csv'))
        
        # Should return success status even with missing file
        assert results['status'] == 'skipped'
        assert 'weights' in results
        assert len(results['weights']) == 3


class TestCLIFunctionality:
    """Test command-line interface functionality"""
    
    def test_main_cli_with_csv_present(self):
        """Test CLI execution with CSV present"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create example data
            study_data = pd.DataFrame({
                'age_group': ['18-34', '35-49'] * 10,
                'sex': ['female', 'male'] * 10
            })
            study_path = tmpdir / 'study_data.csv'
            study_data.to_csv(study_path, index=False)
            
            # Create ICES marginals
            marginals_path = tmpdir / 'ices_marginals.csv'
            create_example_ices_marginals(marginals_path)
            
            # Create output directory
            output_dir = tmpdir / 'output'
            output_dir.mkdir()
            
            # Run CLI
            results = main_cli(
                study_data_path=study_path,
                marginals_path=marginals_path,
                output_dir=output_dir
            )
            
            assert results['status'] in ['completed', 'skipped']
            
            # Check output files created
            assert (output_dir / 'transport_weights.csv').exists()
            assert (output_dir / 'transport_diagnostics.json').exists()
    
    def test_main_cli_with_csv_missing(self):
        """Test CLI execution with CSV missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create study data only
            study_data = pd.DataFrame({'age': [1, 2], 'sex': ['F', 'M']})
            study_path = tmpdir / 'study_data.csv'
            study_data.to_csv(study_path, index=False)
            
            # Use non-existent marginals path
            missing_marginals = tmpdir / 'missing.csv'
            output_dir = tmpdir / 'output'
            output_dir.mkdir()
            
            # Should not raise exception
            results = main_cli(
                study_data_path=study_path,
                marginals_path=missing_marginals,
                output_dir=output_dir
            )
            
            assert results['status'] == 'skipped'
            assert (output_dir / 'transport_weights.csv').exists()  # Uniform weights
    
    def test_run_transport_analysis_integration(self):
        """Test integrated transport analysis workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create study data
            study_data = pd.DataFrame({
                'age_group': ['18-34', '35-49', '50-64'] * 50,
                'sex': ['female', 'male', 'female'] * 50,
                'region': ['urban', 'rural', 'urban'] * 50
            })
            study_path = tmpdir / 'study.csv'
            study_data.to_csv(study_path, index=False)
            
            # Run analysis (missing CSV scenario)
            results = run_transport_analysis(
                study_data_path=study_path,
                output_dir=tmpdir
            )
            
            assert 'transport_results' in results
            assert 'validation_results' in results
            assert results['transport_results']['status'] in ['skipped', 'completed']
            
            # Check outputs
            assert (tmpdir / 'transport_weights.csv').exists()
            assert (tmpdir / 'transport_report.md').exists()


class TestExampleDataGeneration:
    """Test example ICES marginals generation"""
    
    def test_create_example_ices_marginals(self):
        """Test creation of example ICES marginals file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_path = tmpdir / 'test_marginals.csv'
            
            create_example_ices_marginals(output_path)
            
            assert output_path.exists()
            
            # Check file content
            marginals = pd.read_csv(output_path)
            assert 'variable' in marginals.columns
            assert 'category' in marginals.columns
            assert 'proportion' in marginals.columns
            
            # Check specific variables exist
            variables = marginals['variable'].unique()
            expected_vars = ['age_group', 'sex', 'region', 'socioeconomic_quintile']
            for var in expected_vars:
                assert var in variables
            
            # Check proportions sum to 1 for each variable
            for var in expected_vars:
                var_data = marginals[marginals['variable'] == var]
                prop_sum = var_data['proportion'].sum()
                assert abs(prop_sum - 1.0) < 1e-10


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_study_data(self):
        """Test handling of empty study data"""
        empty_data = pd.DataFrame()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            marginals_path = tmpdir + '/marginals.csv'
            
            results = calculate_transport_weights(empty_data, Path(marginals_path))
            
            assert results['status'] == 'skipped'
            assert len(results['weights']) == 0
    
    def test_missing_variables(self):
        """Test handling when required variables are missing"""
        # Data with none of the expected variables
        study_data = pd.DataFrame({
            'unexpected_var1': [1, 2, 3],
            'unexpected_var2': ['a', 'b', 'c']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            marginals_path = tmpdir / 'marginals.csv'
            create_example_ices_marginals(marginals_path)
            
            results = calculate_transport_weights(study_data, marginals_path)
            
            assert results['status'] == 'skipped'
            assert results['reason'] == 'no_variables'
    
    def test_corrupted_marginals_file(self):
        """Test handling of corrupted marginals file"""
        study_data = pd.DataFrame({'age': [1, 2], 'sex': ['F', 'M']})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create corrupted CSV
            corrupted_path = tmpdir / 'corrupted.csv'
            with open(corrupted_path, 'w') as f:
                f.write("invalid,csv,content\n1,2")  # Malformed CSV
            
            results = calculate_transport_weights(study_data, corrupted_path)
            
            assert results['status'] == 'skipped'
            assert 'reason' in results


if __name__ == "__main__":
    pytest.main([__file__])