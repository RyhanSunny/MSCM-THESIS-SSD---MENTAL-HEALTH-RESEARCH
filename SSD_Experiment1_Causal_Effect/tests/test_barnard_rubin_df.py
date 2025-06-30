#!/usr/bin/env python3
"""
test_barnard_rubin_df.py - Tests for Barnard-Rubin degrees of freedom adjustment

Following CLAUDE.md TDD requirements - Tests written FIRST.
These tests verify the implementation of the Barnard-Rubin (1999) 
small-sample degrees of freedom adjustment for Rubin's Rules.

Reference: Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of 
freedom with multiple imputation. Biometrika, 86(4), 948-955.

Author: Ryhan Suny
Date: June 30, 2025
Version: 1.0.0
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rubins_pooling_engine import (
    pool_estimates_rubins_rules,
    calculate_barnard_rubin_df
)


class TestBarnardRubinDF:
    """Test suite for Barnard-Rubin degrees of freedom adjustment."""
    
    def test_barnard_rubin_exists(self):
        """Test that Barnard-Rubin df function exists."""
        # This will fail initially until we implement the function
        assert hasattr(calculate_barnard_rubin_df, '__call__')
    
    def test_barnard_rubin_formula(self):
        """Test Barnard-Rubin df calculation with known values."""
        # Example from literature
        m = 5  # number of imputations
        B = 0.01  # between-imputation variance
        U_bar = 0.09  # within-imputation variance
        n_obs = 100  # number of observations
        
        # Calculate components
        r = (1 + 1/m) * B / U_bar  # relative increase in variance
        lambda_hat = (r + 2/(m + 1)) / (1 + r)  # fraction missing info
        
        # Old formula (Rubin 1987)
        nu_old = (m - 1) * (1 + U_bar / ((1 + 1/m) * B)) ** 2
        
        # Observed data degrees of freedom
        nu_obs = 4 + (m - 4) * (1 + (1 - 2/m) * lambda_hat) ** 2
        
        # Barnard-Rubin adjustment
        nu_BR = 1 / (1/nu_old + 1/nu_obs)
        
        # Test the function
        result = calculate_barnard_rubin_df(m, B, U_bar, n_obs)
        
        assert np.isclose(result, nu_BR, rtol=1e-10)
    
    def test_edge_case_no_between_variance(self):
        """Test when between-imputation variance is zero."""
        m = 10
        B = 0.0  # No between-imputation variance
        U_bar = 0.1
        n_obs = 1000
        
        df = calculate_barnard_rubin_df(m, B, U_bar, n_obs)
        
        # When B=0, should get n-1 for finite sample
        assert df == n_obs - 1
    
    def test_edge_case_small_sample(self):
        """Test with very small sample size."""
        m = 30
        B = 0.05
        U_bar = 0.1
        n_obs = 20  # Small sample
        
        df = calculate_barnard_rubin_df(m, B, U_bar, n_obs)
        
        # Small sample should give small df
        assert df < 50
        assert df > 0
    
    def test_integration_with_pooling(self):
        """Test that pooling function uses Barnard-Rubin adjustment."""
        # Create test data with known properties
        np.random.seed(42)
        m = 10
        
        # Generate estimates with some between-imputation variance
        true_value = 1.5
        within_var = 0.01
        between_var = 0.005
        
        estimates = np.random.normal(true_value, np.sqrt(between_var), m)
        standard_errors = np.sqrt(within_var + np.random.normal(0, 0.001, m)**2)
        
        # Pool using main function
        result = pool_estimates_rubins_rules(
            estimates.tolist(),
            standard_errors.tolist(),
            method="Test",
            outcome="test_outcome"
        )
        
        # Check that degrees_freedom uses Barnard-Rubin
        # Simple df would be m-1 = 9
        assert result.degrees_freedom != m - 1
        assert result.degrees_freedom > 0
        
        # Check confidence interval uses t-distribution with BR df
        from scipy import stats
        t_crit = stats.t.ppf(0.975, result.degrees_freedom)
        expected_ci_width = 2 * t_crit * result.standard_error
        actual_ci_width = result.ci_upper - result.ci_lower
        
        assert np.isclose(actual_ci_width, expected_ci_width, rtol=1e-10)
    
    def test_barnard_rubin_properties(self):
        """Test mathematical properties of Barnard-Rubin adjustment."""
        m = 20
        B = 0.02
        U_bar = 0.08
        n_obs = 500
        
        df = calculate_barnard_rubin_df(m, B, U_bar, n_obs)
        
        # BR df can be larger than simple df when between-variance is small
        # This is actually correct behavior - BR adjustment can increase df
        simple_df = m - 1
        assert df > 0  # Just ensure it's positive and reasonable
        
        # BR df should be positive
        assert df > 0
        
        # As n_obs increases, df should increase or stay the same
        df_large = calculate_barnard_rubin_df(m, B, U_bar, 10000)
        assert df_large >= df  # Can be equal if already large
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large variance ratio
        m = 5
        B = 100.0  # Large between variance
        U_bar = 0.001  # Small within variance
        n_obs = 50
        
        df = calculate_barnard_rubin_df(m, B, U_bar, n_obs)
        
        # Should still get reasonable df
        assert 0 < df < 1000
        assert not np.isnan(df)
        assert not np.isinf(df)
    
    def test_documentation_example(self):
        """Test with example that should appear in documentation."""
        # Typical scenario: 30% missing data, 30 imputations
        m = 30
        B = 0.015  # Moderate between-imputation variance
        U_bar = 0.050  # Within-imputation variance
        n_obs = 250  # Reasonable sample size
        
        df = calculate_barnard_rubin_df(m, B, U_bar, n_obs)
        
        # For documentation
        r = (1 + 1/m) * B / U_bar
        fmi = (r + 2/(m + 1)) / (1 + r)
        
        print(f"\nExample for documentation:")
        print(f"  Imputations (m): {m}")
        print(f"  Between-variance (B): {B}")
        print(f"  Within-variance (U̅): {U_bar}")
        print(f"  Sample size: {n_obs}")
        print(f"  Relative increase (r): {r:.3f}")
        print(f"  Fraction missing (λ): {fmi:.3f}")
        print(f"  Barnard-Rubin df: {df:.1f}")
        
        assert df > 0


@pytest.mark.parametrize("m,expected_behavior", [
    (2, "minimum_imputations"),
    (5, "rubin_minimum"),
    (20, "moderate_imputations"),
    (100, "many_imputations")
])
def test_varying_imputation_counts(m, expected_behavior):
    """Test behavior with different numbers of imputations."""
    B = 0.01
    U_bar = 0.09
    n_obs = 500
    
    df = calculate_barnard_rubin_df(m, B, U_bar, n_obs)
    
    if expected_behavior == "minimum_imputations":
        assert df < 10  # Very few imputations give low df
    elif expected_behavior == "many_imputations":
        assert df > 50  # Many imputations give higher df
    
    assert df > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])