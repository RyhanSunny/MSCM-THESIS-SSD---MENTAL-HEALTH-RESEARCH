#!/usr/bin/env python3
"""
test_ess_calculation.py - Tests for Effective Sample Size (ESS) calculation

Following CLAUDE.md TDD requirements - Tests written FIRST.
These tests verify the implementation of ESS calculation for weighted analyses.

Reference: 
- BMC Med Res Methodol (2024). "Three new methodologies for calculating ESS"
- Austin PC & Stuart EA (2015). "Moving towards best practice..." Stat Med

Formula: ESS = n × sum(w)² / sum(w²)
Where:
- n = original sample size
- w = weights (normalized or unnormalized)

Author: Ryhan Suny
Date: June 30, 2025
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# This will initially fail until we implement the function
from weight_diagnostics_visualizer import calculate_ess


class TestESSCalculation:
    """Test suite for Effective Sample Size calculation."""
    
    def test_ess_function_exists(self):
        """Test that ESS calculation function exists."""
        assert hasattr(calculate_ess, '__call__')
    
    def test_ess_uniform_weights(self):
        """Test ESS with uniform weights (should equal n)."""
        n = 1000
        weights = np.ones(n)
        
        ess = calculate_ess(weights)
        
        # With uniform weights, ESS should equal n
        assert np.isclose(ess, n, rtol=1e-10)
    
    def test_ess_extreme_weights(self):
        """Test ESS with extreme weights (one unit gets all weight)."""
        n = 1000
        weights = np.zeros(n)
        weights[0] = 1.0  # All weight on first unit
        
        ess = calculate_ess(weights)
        
        # With all weight on one unit, ESS should be 1
        assert np.isclose(ess, 1.0, rtol=1e-10)
    
    def test_ess_formula_correctness(self):
        """Test ESS formula with known example."""
        # Example from literature
        weights = np.array([0.5, 0.3, 0.2, 0.8, 1.2])
        n = len(weights)
        
        # Manual calculation
        sum_w = np.sum(weights)
        sum_w2 = np.sum(weights**2)
        expected_ess = n * (sum_w**2) / sum_w2
        
        ess = calculate_ess(weights)
        
        assert np.isclose(ess, expected_ess, rtol=1e-10)
    
    def test_ess_normalized_weights(self):
        """Test ESS with normalized weights (sum to n)."""
        n = 500
        # Create weights that sum to n
        raw_weights = np.random.exponential(1, n)
        weights = raw_weights * (n / np.sum(raw_weights))
        
        ess = calculate_ess(weights)
        
        # ESS should be between 1 and n
        assert 1 <= ess <= n
        # With exponential weights, expect moderate reduction
        assert ess < 0.9 * n  # Some efficiency loss expected
    
    def test_ess_with_dataframe(self):
        """Test ESS calculation accepts DataFrame with weight column."""
        n = 100
        df = pd.DataFrame({
            'id': range(n),
            'weight': np.random.uniform(0.5, 2.0, n)
        })
        
        ess = calculate_ess(df['weight'])
        
        assert isinstance(ess, (int, float))
        assert 1 <= ess <= n
    
    def test_ess_boundary_conditions(self):
        """Test ESS boundary conditions and properties."""
        n = 1000
        
        # Test 1: Equal weights -> ESS = n
        equal_weights = np.ones(n) * 2.5  # Non-unit but equal
        ess_equal = calculate_ess(equal_weights)
        assert np.isclose(ess_equal, n, rtol=1e-10)
        
        # Test 2: Very unequal weights -> ESS << n
        unequal_weights = np.ones(n) * 0.01
        unequal_weights[0] = 100  # One very large weight
        ess_unequal = calculate_ess(unequal_weights)
        assert ess_unequal < n / 10  # Substantial reduction
        
        # Test 3: Slightly unequal weights -> ESS slightly < n
        slight_weights = np.random.normal(1, 0.1, n)
        slight_weights = np.abs(slight_weights)  # Ensure positive
        ess_slight = calculate_ess(slight_weights)
        assert 0.8 * n < ess_slight < n
    
    def test_ess_propensity_score_weights(self):
        """Test ESS with typical propensity score weights."""
        n = 5000
        # Simulate propensity scores
        ps = np.random.beta(2, 5, n)  # Typical PS distribution
        
        # IPTW weights
        treatment = np.random.binomial(1, ps, n)
        weights = np.where(treatment == 1, 1/ps, 1/(1-ps))
        
        # Truncate extreme weights (common practice)
        weights = np.clip(weights, 0.1, 10)
        
        ess = calculate_ess(weights)
        
        # Typical ESS reduction for PS weighting
        assert 0.3 * n < ess < 0.8 * n
        print(f"\nPS weighting ESS: {ess:.1f} ({100*ess/n:.1f}% of n={n})")
    
    def test_ess_error_handling(self):
        """Test ESS calculation error handling."""
        # Empty weights
        with pytest.raises(ValueError, match="empty"):
            calculate_ess(np.array([]))
        
        # Negative weights
        with pytest.raises(ValueError, match="negative"):
            calculate_ess(np.array([1, -1, 2]))
        
        # All zero weights
        with pytest.raises(ValueError, match="zero"):
            calculate_ess(np.zeros(10))
        
        # NaN weights
        with pytest.raises(ValueError, match="NaN"):
            calculate_ess(np.array([1, 2, np.nan, 3]))
    
    def test_ess_reporting_threshold(self):
        """Test ESS reporting includes warning threshold."""
        n = 1000
        
        # Create weights that give ESS < 50% of n
        weights = np.ones(n) * 0.01
        weights[:10] = 10  # 10 units get high weight
        
        ess = calculate_ess(weights)
        ess_ratio = ess / n
        
        # Should flag when ESS < 50% of original n
        assert ess_ratio < 0.5
        
        # Function should return dict with warnings
        result = calculate_ess(weights, return_diagnostics=True)
        assert 'ess' in result
        assert 'ess_ratio' in result
        assert 'warning' in result
        assert result['warning'] is True  # Should warn


@pytest.mark.parametrize("weight_type,expected_range", [
    ("uniform", (0.95, 1.0)),      # Uniform weights: ESS ≈ n
    ("mild_variation", (0.8, 0.95)), # Mild variation: 80-95% of n
    ("moderate_variation", (0.5, 0.8)), # Moderate: 50-80% of n  
    ("extreme_variation", (0.1, 0.5))   # Extreme: 10-50% of n
])
def test_ess_weight_variation_impact(weight_type, expected_range):
    """Test how weight variation impacts ESS."""
    n = 1000
    
    if weight_type == "uniform":
        weights = np.ones(n)
    elif weight_type == "mild_variation":
        weights = np.random.normal(1, 0.2, n)
        weights = np.abs(weights)
    elif weight_type == "moderate_variation":
        weights = np.random.lognormal(0, 0.5, n)
    else:  # extreme_variation
        weights = np.random.lognormal(0, 1.5, n)
    
    ess = calculate_ess(weights)
    ess_ratio = ess / n
    
    assert expected_range[0] <= ess_ratio <= expected_range[1]


def test_ess_integration_with_weight_diagnostics():
    """Test ESS integrates with weight diagnostics module."""
    # Create sample weighted data
    n = 2000
    df = pd.DataFrame({
        'patient_id': range(n),
        'treatment': np.random.binomial(1, 0.3, n),
        'outcome': np.random.binomial(1, 0.2, n),
        'ps_weight': np.random.lognormal(0, 0.5, n)
    })
    
    # Import weight diagnostics (will fail initially)
    from weight_diagnostics_visualizer import generate_weight_diagnostics
    
    # Generate diagnostics
    diagnostics = generate_weight_diagnostics(
        df, 
        weight_col='ps_weight',
        treatment_col='treatment'
    )
    
    # Should include ESS metrics
    assert 'ess_treated' in diagnostics
    assert 'ess_control' in diagnostics
    assert 'ess_overall' in diagnostics
    assert 'ess_ratio' in diagnostics
    
    # ESS should be reasonable
    assert diagnostics['ess_overall'] > 0
    assert diagnostics['ess_overall'] <= n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])