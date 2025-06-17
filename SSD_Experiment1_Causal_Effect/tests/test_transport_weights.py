#!/usr/bin/env python3
"""
test_transport_weights.py - Tests for transportability weights

Tests transportability weight calculation, handling of missing ICES marginals,
and weight validation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestTransportWeights:
    """Test transportability weights functionality"""
    
    def test_transport_weights_missing_file(self, tmp_path):
        """Test transport weights when ICES marginals file is missing"""
        from transport_weights import calculate_transport_weights
        
        # Create test data
        study_data = pd.DataFrame({
            'age_group': ['18-34', '35-49', '50-64'] * 10,
            'sex': ['female', 'male'] * 15,
            'region': ['urban', 'suburban', 'rural'] * 10
        })
        
        # Point to non-existent file
        missing_file = tmp_path / 'missing_ices_marginals.csv'
        
        # Should return skipped status
        result = calculate_transport_weights(
            study_data, target_marginals_path=missing_file
        )
        
        assert result['status'] == 'skipped'
        assert result['reason'] == 'ICES marginals file not available'
        assert len(result['weights']) == len(study_data)
        assert np.all(result['weights'] == 1.0)  # Uniform weights
    
    def test_transport_weights_with_marginals(self, tmp_path):
        """Test transport weights with provided marginals"""
        from transport_weights import calculate_transport_weights, create_example_ices_marginals
        
        # Create example marginals file
        marginals_file = tmp_path / 'ices_marginals.csv'
        create_example_ices_marginals(marginals_file)
        
        # Create test study data with some categories
        study_data = pd.DataFrame({
            'age_group': ['18-34'] * 20 + ['35-49'] * 30 + ['50-64'] * 50,
            'sex': ['female'] * 60 + ['male'] * 40,
            'region': ['urban'] * 80 + ['rural'] * 20
        })
        
        # Calculate transport weights
        result = calculate_transport_weights(
            study_data, target_marginals_path=marginals_file
        )
        
        # Should complete successfully
        assert result['status'] == 'completed'
        assert len(result['weights']) == len(study_data)
        assert result['effective_sample_size'] > 0
        
        # Weights should not all be identical (some reweighting should occur)
        unique_weights = len(np.unique(np.round(result['weights'], 3)))
        assert unique_weights > 1
    
    def test_transport_weights_validation(self):
        """Test transport weight validation"""
        from transport_weights import validate_transport_weights
        
        # Good weights
        good_weights = np.random.gamma(2, 0.5, 1000)
        validation = validate_transport_weights(good_weights)
        
        assert validation['overall_quality'] == True
        assert validation['max_weight_ok'] == True
        assert validation['ess_ratio_ok'] == True
        
        # Bad weights (too extreme)
        bad_weights = np.ones(1000)
        bad_weights[:10] = 50  # Very high weights
        validation = validate_transport_weights(bad_weights, max_weight_threshold=20.0)
        
        assert validation['overall_quality'] == False
        assert validation['max_weight_ok'] == False
    
    def test_create_example_ices_marginals(self, tmp_path):
        """Test creation of example ICES marginals file"""
        from transport_weights import create_example_ices_marginals
        
        output_file = tmp_path / 'test_marginals.csv'
        create_example_ices_marginals(output_file)
        
        # Check file was created
        assert output_file.exists()
        
        # Check file content
        marginals = pd.read_csv(output_file)
        assert 'variable' in marginals.columns
        assert 'category' in marginals.columns
        assert 'proportion' in marginals.columns
        
        # Check some expected categories
        age_data = marginals[marginals['variable'] == 'age_group']
        assert '18-34' in age_data['category'].values
        assert '35-49' in age_data['category'].values
        
        # Proportions should sum to 1 for each variable
        for var in marginals['variable'].unique():
            var_data = marginals[marginals['variable'] == var]
            prop_sum = var_data['proportion'].sum()
            assert abs(prop_sum - 1.0) < 0.01
    
    def test_effective_sample_size_calculation(self):
        """Test effective sample size calculation"""
        from transport_weights import calculate_effective_sample_size
        
        # Uniform weights should give ESS = N
        uniform_weights = np.ones(1000)
        ess = calculate_effective_sample_size(uniform_weights)
        assert abs(ess - 1000) < 0.001
        
        # Highly variable weights should give lower ESS
        variable_weights = np.concatenate([np.ones(900), np.full(100, 10)])
        ess_variable = calculate_effective_sample_size(variable_weights)
        assert ess_variable < 1000
        assert ess_variable > 0
    
    def test_transport_weights_no_variables(self):
        """Test transport weights when no variables available"""
        from transport_weights import calculate_transport_weights
        
        # Data without expected transport variables
        study_data = pd.DataFrame({
            'patient_id': range(100),
            'outcome': np.random.normal(0, 1, 100)
        })
        
        result = calculate_transport_weights(study_data, variables=['age_group', 'sex'])
        
        # Should return skipped status (file missing takes precedence)
        assert result['status'] == 'skipped'
        assert 'ICES marginals' in result['reason'] or result['reason'] == 'no_variables'
        assert len(result['weights']) == len(study_data)
    
    def test_transport_weights_partial_variables(self, tmp_path):
        """Test transport weights with partial variable availability"""
        from transport_weights import calculate_transport_weights, create_example_ices_marginals
        
        # Create marginals file
        marginals_file = tmp_path / 'ices_marginals.csv'
        create_example_ices_marginals(marginals_file)
        
        # Data with only some transport variables
        study_data = pd.DataFrame({
            'age_group': ['18-34', '35-49', '50-64'] * 20,
            'sex': ['female', 'male'] * 30,
            # Missing 'region' and 'socioeconomic_quintile'
        })
        
        result = calculate_transport_weights(
            study_data, target_marginals_path=marginals_file
        )
        
        # Should complete with available variables
        assert result['status'] == 'completed'
        assert 'variables_used' in result
        assert 'age_group' in result['variables_used']
        assert 'sex' in result['variables_used']
        assert len(result['variables_used']) == 2
    
    def test_transport_weights_edge_cases(self):
        """Test transport weights edge cases"""
        from transport_weights import calculate_transport_weights
        
        # Empty dataset
        empty_data = pd.DataFrame()
        result = calculate_transport_weights(empty_data)
        
        assert result['status'] == 'skipped'
        assert len(result['weights']) == 0
        
        # Single observation
        single_data = pd.DataFrame({
            'age_group': ['18-34'],
            'sex': ['female']
        })
        
        result = calculate_transport_weights(single_data)
        # Should handle gracefully (file missing case)
        assert result['status'] == 'skipped'
        assert len(result['weights']) == 1