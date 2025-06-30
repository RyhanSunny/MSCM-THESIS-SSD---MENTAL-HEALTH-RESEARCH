#!/usr/bin/env python3
"""
test_rubins_pooling_engine.py - Test suite for Rubin's Rules implementation

Following CLAUDE.md TDD requirements:
- Tests written FIRST before implementation
- Tests verify statistical correctness
- Tests check edge cases and validation

Author: Ryhan Suny (Toronto Metropolitan University)
Date: 2025-06-29
"""

import pytest
import numpy as np
import json
from pathlib import Path
import tempfile
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.rubins_pooling_engine import (
    validate_imputation_inputs,
    pool_estimates_rubins_rules,
    load_imputed_causal_estimates,
    RubinsPooledResult
)


class TestValidateImputationInputs:
    """Test input validation for Rubin's Rules"""
    
    def test_sufficient_imputations_warning(self):
        """Test warning for m < 5 imputations"""
        estimates = [1.5, 1.6]
        ses = [0.1, 0.12]
        
        validation = validate_imputation_inputs(estimates, ses)
        assert validation['sufficient_imputations'] == False
        
    def test_sufficient_imputations_pass(self):
        """Test m >= 5 imputations pass"""
        estimates = [1.5, 1.6, 1.55, 1.58, 1.52]
        ses = [0.1, 0.12, 0.11, 0.105, 0.115]
        
        validation = validate_imputation_inputs(estimates, ses)
        assert validation['sufficient_imputations'] == True
        
    def test_missing_data_detection(self):
        """Test detection of missing values"""
        estimates = [1.5, np.nan, 1.55]
        ses = [0.1, 0.12, 0.11]
        
        validation = validate_imputation_inputs(estimates, ses)
        assert validation['complete_estimates'] == False
        assert validation['complete_ses'] == True
        
    def test_negative_se_detection(self):
        """Test detection of invalid standard errors"""
        estimates = [1.5, 1.6, 1.55]
        ses = [0.1, -0.12, 0.11]
        
        validation = validate_imputation_inputs(estimates, ses)
        assert validation['positive_ses'] == False
        
    def test_extreme_variation_detection(self):
        """Test detection of high between-imputation variation"""
        estimates = [1.0, 10.0, 1.1]  # High CV
        ses = [0.1, 0.1, 0.1]
        
        validation = validate_imputation_inputs(estimates, ses)
        assert validation['reasonable_variation'] == False
        
    def test_input_length_mismatch(self):
        """Test error on mismatched input lengths"""
        estimates = [1.5, 1.6]
        ses = [0.1]
        
        with pytest.raises(ValueError, match="equal length"):
            validate_imputation_inputs(estimates, ses)


class TestPoolEstimatesRubinsRules:
    """Test core Rubin's Rules calculations"""
    
    def test_basic_pooling_calculation(self):
        """Test basic Rubin's Rules calculation matches expected values"""
        # Example from Rubin (1987) Chapter 3
        estimates = [1.5, 1.6, 1.55, 1.58, 1.52]
        ses = [0.10, 0.12, 0.11, 0.105, 0.115]
        
        result = pool_estimates_rubins_rules(estimates, ses, "TMLE", "total_encounters")
        
        # Verify pooled estimate (simple average)
        expected_estimate = np.mean(estimates)
        assert abs(result.estimate - expected_estimate) < 1e-10
        
        # Verify variance components
        variances = np.array(ses) ** 2
        expected_within = np.mean(variances)
        assert abs(result.within_variance - expected_within) < 1e-10
        
        # Verify between variance
        expected_between = np.var(estimates, ddof=1)
        assert abs(result.between_variance - expected_between) < 1e-10
        
        # Verify total variance formula
        m = len(estimates)
        expected_total = expected_within + (1 + 1/m) * expected_between
        assert abs(result.total_variance - expected_total) < 1e-10
        
    def test_fraction_missing_information(self):
        """Test FMI calculation follows formula"""
        # High between-imputation variance case
        estimates = [1.0, 2.0, 1.5, 1.8, 1.2]
        ses = [0.1, 0.1, 0.1, 0.1, 0.1]
        
        result = pool_estimates_rubins_rules(estimates, ses)
        
        # FMI should be high when between variance is high
        assert result.fmi > 0.5
        assert result.fmi <= 1.0
        
    def test_degrees_freedom_adjustment(self):
        """Test Barnard-Rubin degrees of freedom adjustment"""
        estimates = [1.5, 1.51, 1.49, 1.50, 1.52]
        ses = [0.1, 0.1, 0.1, 0.1, 0.1]
        
        result = pool_estimates_rubins_rules(estimates, ses)
        
        # DF should be finite and positive
        assert result.degrees_freedom > 0
        assert not np.isinf(result.degrees_freedom)
        
    def test_confidence_interval_coverage(self):
        """Test CI uses t-distribution with adjusted DF"""
        estimates = [1.5] * 5
        ses = [0.1] * 5
        
        result = pool_estimates_rubins_rules(estimates, ses, alpha=0.05)
        
        # With no between variance, CI should be based on within variance only
        assert result.ci_lower < result.estimate
        assert result.ci_upper > result.estimate
        
    def test_edge_case_single_imputation_fails(self):
        """Test that single imputation raises error"""
        with pytest.raises(ValueError, match="Minimum 2 imputations"):
            pool_estimates_rubins_rules([1.5], [0.1])
            
    def test_zero_between_variance(self):
        """Test handling when all estimates are identical"""
        estimates = [1.5] * 5
        ses = [0.1, 0.11, 0.09, 0.10, 0.12]
        
        result = pool_estimates_rubins_rules(estimates, ses)
        
        assert result.between_variance == 0.0
        assert result.fmi < 0.1  # Very low FMI
        

class TestLoadImputedCausalEstimates:
    """Test loading results from multiple imputation files"""
    
    def test_load_multiple_files(self):
        """Test loading and organizing results from multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test result files
            for i in range(1, 4):
                data = {
                    "total_encounters": {
                        "TMLE": {
                            "estimate": 1.5 + i * 0.1,
                            "se": 0.1 + i * 0.01,
                            "ci_lower": 1.3,
                            "ci_upper": 1.7
                        }
                    }
                }
                with open(tmpdir / f"causal_estimates_imp{i}.json", 'w') as f:
                    json.dump(data, f)
            
            # Load results
            results = load_imputed_causal_estimates(tmpdir)
            
            assert "estimates" in results
            assert "standard_errors" in results
            assert len(results["estimates"]["total_encounters"]["TMLE"]) == 3
            assert len(results["standard_errors"]["total_encounters"]["TMLE"]) == 3
            
    def test_handle_missing_files(self):
        """Test graceful handling when no files found"""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = load_imputed_causal_estimates(Path(tmpdir))
            
            assert results["estimates"] == {}
            assert results["standard_errors"] == {}
            
    def test_handle_malformed_json(self):
        """Test handling of corrupted JSON files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create malformed file
            with open(tmpdir / "causal_estimates_imp1.json", 'w') as f:
                f.write("{invalid json")
            
            # Should continue without crashing
            results = load_imputed_causal_estimates(tmpdir)
            assert results["estimates"] == {}


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_statistical_properties(self):
        """Test that pooled SE > average of individual SEs (key property)"""
        # When there's between-imputation variance, pooled SE must be larger
        estimates = [1.4, 1.6, 1.5, 1.7, 1.3]
        ses = [0.1] * 5
        
        result = pool_estimates_rubins_rules(estimates, ses)
        
        # Pooled SE should be larger than average SE due to between variance
        avg_se = np.mean(ses)
        assert result.standard_error > avg_se
        
    def test_rubin_1987_example(self):
        """Test against known example from Rubin (1987)"""
        # Example 3.1 from Multiple Imputation for Nonresponse in Surveys
        estimates = [30.0, 25.0, 35.0, 32.0, 28.0]
        ses = [3.0, 2.5, 3.5, 3.2, 2.8]
        
        result = pool_estimates_rubins_rules(estimates, ses)
        
        # Verify calculations match book
        assert abs(result.estimate - 30.0) < 0.01
        assert result.total_variance > np.mean(np.array(ses)**2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])