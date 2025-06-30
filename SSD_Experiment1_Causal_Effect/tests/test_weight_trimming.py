#!/usr/bin/env python3
"""
test_weight_trimming.py - Test suite for weight trimming implementation

Following TDD principles per CLAUDE.md requirements.
Tests the apply_weight_trimming function from 06_causal_estimators.py

Author: Ryhan Suny
Date: June 30, 2025
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the function directly from the module  
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import with the correct module name (includes 06_ prefix)
import importlib.util
spec = importlib.util.spec_from_file_location("causal_estimators", 
    str(Path(__file__).parent.parent / 'src' / '06_causal_estimators.py'))
causal_estimators = importlib.util.module_from_spec(spec)
spec.loader.exec_module(causal_estimators)
apply_weight_trimming = causal_estimators.apply_weight_trimming


class TestWeightTrimming:
    """Test suite for Crump weight trimming implementation"""
    
    def test_crump_trimming_basic(self):
        """Test basic Crump trimming with threshold of 10"""
        # Create weights with some extreme values
        weights = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 100.0])
        
        # Apply trimming with threshold of 10
        trimmed_weights, trim_info = apply_weight_trimming(weights, trim_threshold=10, method='crump')
        
        # Check that weights above 10 are trimmed to 10
        assert np.all(trimmed_weights <= 10)
        assert trimmed_weights[0] == 1.0
        assert trimmed_weights[1] == 5.0
        assert trimmed_weights[2] == 10.0
        assert trimmed_weights[3] == 10.0  # Was 15, now trimmed
        assert trimmed_weights[4] == 10.0  # Was 20, now trimmed
        assert trimmed_weights[5] == 10.0  # Was 100, now trimmed
        
        # Check trim info
        assert trim_info['n_trimmed'] == 3
        assert trim_info['percent_trimmed'] == 50.0
        assert trim_info['max_weight_original'] == 100.0
        assert trim_info['max_weight_trimmed'] == 10.0
    
    def test_crump_trimming_no_extreme_weights(self):
        """Test Crump trimming when no weights exceed threshold"""
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        trimmed_weights, trim_info = apply_weight_trimming(weights, trim_threshold=10, method='crump')
        
        # Weights should be unchanged
        np.testing.assert_array_equal(weights, trimmed_weights)
        
        # Check trim info
        assert trim_info['n_trimmed'] == 0
        assert trim_info['percent_trimmed'] == 0.0
        assert trim_info['max_weight_original'] == 5.0
        assert trim_info['max_weight_trimmed'] == 5.0
    
    def test_percentile_trimming(self):
        """Test percentile-based trimming method"""
        np.random.seed(42)
        weights = np.random.exponential(2, 1000)
        
        # Add some extreme values
        weights[0] = 0.001  # Very small
        weights[1] = 100.0  # Very large
        
        trimmed_weights, trim_info = apply_weight_trimming(weights, method='percentile')
        
        # Check that extreme values are clipped
        lower = np.percentile(weights, 1)
        upper = np.percentile(weights, 99)
        
        assert np.all(trimmed_weights >= lower)
        assert np.all(trimmed_weights <= upper)
        assert trim_info['n_trimmed'] >= 2  # At least the two extreme values
    
    def test_ess_improvement(self):
        """Test that trimming improves ESS"""
        np.random.seed(42)
        n = 1000
        
        # Create weights with some extreme values
        weights = np.random.gamma(2, 0.5, n)
        weights[:10] = 50.0  # 10 extreme weights
        
        trimmed_weights, trim_info = apply_weight_trimming(weights, trim_threshold=10, method='crump')
        
        # ESS should improve after trimming
        assert trim_info['ess_trimmed'] > trim_info['ess_original']
        
        # Check ESS calculation formula: sum(w)^2 / sum(w^2)
        ess_original = (np.sum(weights)**2) / np.sum(weights**2)
        ess_trimmed = (np.sum(trimmed_weights)**2) / np.sum(trimmed_weights**2)
        
        assert abs(trim_info['ess_original'] - ess_original) < 1e-6
        assert abs(trim_info['ess_trimmed'] - ess_trimmed) < 1e-6
    
    def test_invalid_method(self):
        """Test error handling for invalid trimming method"""
        weights = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Unknown trimming method"):
            apply_weight_trimming(weights, method='invalid_method')
    
    def test_pandas_series_input(self):
        """Test that function works with pandas Series input"""
        weights_series = pd.Series([1.0, 5.0, 15.0, 20.0])
        
        trimmed_weights, trim_info = apply_weight_trimming(weights_series, trim_threshold=10)
        
        # Should return numpy array
        assert isinstance(trimmed_weights, np.ndarray)
        assert len(trimmed_weights) == len(weights_series)
        assert trim_info['n_trimmed'] == 2
    
    def test_custom_threshold(self):
        """Test custom trimming thresholds"""
        weights = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
        
        # Test with threshold of 5
        trimmed_weights, trim_info = apply_weight_trimming(weights, trim_threshold=5)
        
        assert np.all(trimmed_weights <= 5)
        assert trim_info['n_trimmed'] == 3
        assert trim_info['max_weight_trimmed'] == 5.0
    
    def test_empty_weights(self):
        """Test handling of edge case with empty weights"""
        weights = np.array([])
        
        # Should handle gracefully
        trimmed_weights, trim_info = apply_weight_trimming(weights, trim_threshold=10)
        
        assert len(trimmed_weights) == 0
        assert trim_info['n_trimmed'] == 0
        assert trim_info['percent_trimmed'] == 0.0
    
    def test_all_weights_extreme(self):
        """Test case where all weights exceed threshold"""
        weights = np.array([20.0, 30.0, 40.0, 50.0])
        
        trimmed_weights, trim_info = apply_weight_trimming(weights, trim_threshold=10)
        
        assert np.all(trimmed_weights == 10.0)
        assert trim_info['n_trimmed'] == 4
        assert trim_info['percent_trimmed'] == 100.0
        
    def test_integration_with_causal_pipeline(self):
        """Test integration with actual causal estimation pipeline"""
        np.random.seed(42)
        n = 1000
        
        # Simulate realistic propensity score weights
        ps = np.random.beta(2, 5, n)
        treatment = np.random.binomial(1, ps, n)
        
        # Calculate IPTW
        iptw = np.where(treatment == 1, 1/ps, 1/(1-ps))
        
        # Apply Crump trimming
        trimmed_iptw, trim_info = apply_weight_trimming(iptw, trim_threshold=10)
        
        # Check that trimming was effective
        assert np.all(trimmed_iptw <= 10)
        assert trim_info['ess_trimmed'] > trim_info['ess_original']
        
        # Verify that trimmed weights maintain basic properties
        assert np.all(trimmed_iptw > 0)
        assert len(trimmed_iptw) == len(iptw)


@pytest.mark.parametrize("threshold,expected_trimmed", [
    (5, 4),   # Strict threshold
    (10, 2),  # Standard Crump threshold
    (20, 1),  # Lenient threshold
    (100, 0)  # Very lenient threshold
])
def test_threshold_sensitivity(threshold, expected_trimmed):
    """Test sensitivity to different trimming thresholds"""
    weights = np.array([1.0, 3.0, 8.0, 12.0, 18.0, 25.0])
    
    trimmed_weights, trim_info = apply_weight_trimming(weights, trim_threshold=threshold)
    
    assert trim_info['n_trimmed'] == expected_trimmed
    assert np.all(trimmed_weights <= threshold)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])