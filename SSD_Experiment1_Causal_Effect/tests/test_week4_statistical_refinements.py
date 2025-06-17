#!/usr/bin/env python3
"""
test_week4_statistical_refinements.py - Tests for Week 4 statistical refinements

Tests enhanced mediation analysis, FDR adjustment, E-value computation, 
and extended weight diagnostics following TDD principles.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestWeek4StatisticalRefinements:
    """Test Week 4 statistical refinements"""
    
    def test_enhanced_weight_diagnostics_failure_conditions(self):
        """Test weight diagnostics fail when thresholds violated"""
        from week4_statistical_refinements import calculate_weight_diagnostics
        
        # Create weights that violate ESS threshold
        n = 1000
        bad_weights = np.ones(n)
        bad_weights[:50] = 100  # Very high weights for some patients
        
        diagnostics = calculate_weight_diagnostics(bad_weights)
        
        # ESS should be much less than 0.5 * N
        expected_ess_threshold = 0.5 * n
        assert diagnostics['ess'] < expected_ess_threshold
        
        # Max weight should exceed 10 * median
        median_weight = np.median(bad_weights)
        max_weight = np.max(bad_weights)
        assert max_weight > 10 * median_weight
        
        # Overall quality should be poor
        assert not diagnostics['passes_quality_check']
    
    def test_enhanced_weight_diagnostics_passing_conditions(self):
        """Test weight diagnostics pass with good weights"""
        from week4_statistical_refinements import calculate_weight_diagnostics
        
        # Create reasonable weights
        n = 1000
        np.random.seed(42)
        good_weights = np.random.gamma(2, 0.5, n)  # Reasonable distribution
        
        diagnostics = calculate_weight_diagnostics(good_weights)
        
        # Should meet quality thresholds
        expected_ess_threshold = 0.5 * n
        assert diagnostics['ess'] >= expected_ess_threshold
        
        median_weight = np.median(good_weights)
        max_weight = np.max(good_weights)
        assert max_weight <= 10 * median_weight
        
        # Overall quality should be good
        assert diagnostics['passes_quality_check']
    
    def test_evalue_computation_for_hypotheses(self):
        """Test E-value computation for H1-H3 hypotheses"""
        from week4_statistical_refinements import compute_evalue_for_hypothesis
        
        # Test different effect sizes
        test_cases = [
            {'irr': 1.2, 'expected_evalue': 1.2 + np.sqrt(1.2 * 0.2)},
            {'irr': 1.5, 'expected_evalue': 1.5 + np.sqrt(1.5 * 0.5)},
            {'irr': 0.8, 'expected_evalue': 1 / (1/0.8 + np.sqrt((1/0.8) * (1/0.8 - 1)))}
        ]
        
        for case in test_cases:
            irr = case['irr']
            expected = case['expected_evalue']
            
            evalue_result = compute_evalue_for_hypothesis(irr)
            evalue = evalue_result['evalue']  # Extract the actual E-value
            assert abs(evalue - expected) < 0.001, f"E-value calculation incorrect for IRR={irr}"
    
    def test_fdr_adjustment_implementation(self):
        """Test Benjamini-Hochberg FDR adjustment"""
        from week4_statistical_refinements import apply_benjamini_hochberg_fdr
        
        # Test p-values that should show FDR adjustment
        p_values = [0.001, 0.01, 0.03, 0.04, 0.06, 0.08]
        alpha = 0.05
        
        adjusted_results = apply_benjamini_hochberg_fdr(p_values, alpha)
        
        # Check structure
        assert 'p_values' in adjusted_results
        assert 'p_adjusted' in adjusted_results
        assert 'significant_fdr' in adjusted_results
        assert 'rejected_hypotheses' in adjusted_results
        
        # Should have fewer rejections than naive alpha=0.05
        naive_rejections = sum(p < alpha for p in p_values)
        fdr_rejections = sum(adjusted_results['significant_fdr'])
        
        # FDR should be more conservative
        assert fdr_rejections <= naive_rejections
    
    def test_enhanced_mediation_analysis_dowhy(self):
        """Test enhanced mediation analysis using DoWhy approach"""
        from week4_statistical_refinements import enhanced_mediation_analysis
        
        # Create test data
        np.random.seed(42)
        n = 500
        
        # Generate data with known mediation structure
        X = np.random.normal(0, 1, n)  # Exposure
        M = 0.5 * X + np.random.normal(0, 0.5, n)  # Mediator
        Y = 0.3 * X + 0.4 * M + np.random.normal(0, 0.5, n)  # Outcome
        
        data = pd.DataFrame({
            'exposure': X,
            'mediator': M,
            'outcome': Y,
            'age': np.random.normal(50, 10, n),
            'sex': np.random.choice([0, 1], n)
        })
        
        results = enhanced_mediation_analysis(
            data=data,
            exposure='exposure',
            mediator='mediator', 
            outcome='outcome',
            confounders=['age', 'sex'],
            n_bootstrap=100
        )
        
        # Check required components
        assert 'total_effect' in results
        assert 'direct_effect' in results
        assert 'indirect_effect' in results
        assert 'proportion_mediated' in results
        assert 'bootstrap_ci' in results
        
        # Effects should be reasonable given data generation
        assert 0.1 < results['total_effect'] < 1.0
        assert 0.0 < results['proportion_mediated'] < 1.0
    
    def test_weight_diagnostics_pytest_integration(self):
        """Test weight diagnostics integration with pytest framework"""
        from week4_statistical_refinements import validate_weight_quality
        
        # Test case that should pass
        good_weights = np.random.gamma(2, 0.5, 1000)
        
        # Should not raise any assertions
        try:
            validate_weight_quality(good_weights)
        except AssertionError:
            pytest.fail("Weight quality validation failed for good weights")
        
        # Test case that should fail
        bad_weights = np.ones(1000)
        bad_weights[:50] = 100
        
        # Should raise assertion error
        with pytest.raises(AssertionError):
            validate_weight_quality(bad_weights)
    
    def test_sensitivity_analysis_rho_parameter(self):
        """Test sensitivity analysis with rho parameter"""
        from week4_statistical_refinements import mediation_sensitivity_analysis
        
        # Mock mediation results
        mediation_results = {
            'total_effect': 0.25,
            'direct_effect': 0.15,
            'indirect_effect': 0.10,
            'se_total': 0.05,
            'se_direct': 0.04,
            'se_indirect': 0.03
        }
        
        # Test different sensitivity scenarios
        rho_values = [0.0, 0.1, 0.2, 0.3]
        
        sensitivity_results = mediation_sensitivity_analysis(
            mediation_results, rho_values
        )
        
        # Check structure
        assert 'rho_values' in sensitivity_results
        assert 'adjusted_effects' in sensitivity_results
        assert len(sensitivity_results['adjusted_effects']) == len(rho_values)
        
        # Effects should change with increasing rho
        effects = [res['indirect_effect'] for res in sensitivity_results['adjusted_effects']]
        # Should show some variation due to sensitivity parameter
        assert len(set(np.round(effects, 3))) > 1
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence intervals for mediation"""
        from week4_statistical_refinements import bootstrap_mediation_ci
        
        # Create simple test data
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            'X': np.random.normal(0, 1, n),
            'M': np.random.normal(0, 1, n),
            'Y': np.random.normal(0, 1, n)
        })
        
        # Compute bootstrap CIs
        ci_results = bootstrap_mediation_ci(
            data=data,
            exposure='X',
            mediator='M',
            outcome='Y',
            n_bootstrap=100,
            confidence_level=0.95
        )
        
        # Check structure
        assert 'total_effect_ci' in ci_results
        assert 'direct_effect_ci' in ci_results
        assert 'indirect_effect_ci' in ci_results
        
        # CIs should be two-element tuples/lists
        for ci_key in ['total_effect_ci', 'direct_effect_ci', 'indirect_effect_ci']:
            ci = ci_results[ci_key]
            assert len(ci) == 2
            assert ci[0] <= ci[1]  # Lower bound <= upper bound
    
    def test_evalue_results_integration(self):
        """Test E-value integration with hypothesis results"""
        from week4_statistical_refinements import add_evalues_to_results
        
        # Mock hypothesis results
        hypothesis_results = {
            'h1': {'irr': 1.23, 'ci_lower': 1.10, 'ci_upper': 1.37},
            'h2': {'irr': 1.45, 'ci_lower': 1.25, 'ci_upper': 1.68},
            'h3': {'irr': 1.31, 'ci_lower': 1.15, 'ci_upper': 1.50}
        }
        
        enhanced_results = add_evalues_to_results(hypothesis_results)
        
        # Should add E-values to each hypothesis
        for h in ['h1', 'h2', 'h3']:
            assert 'evalue' in enhanced_results[h]
            assert 'evalue_ci_lower' in enhanced_results[h]
            
            # E-values should be >= 1
            assert enhanced_results[h]['evalue'] >= 1.0
            assert enhanced_results[h]['evalue_ci_lower'] >= 1.0
            
            # E-value should be larger for larger effect sizes
            if h == 'h2':  # Largest IRR
                assert enhanced_results[h]['evalue'] >= enhanced_results['h1']['evalue']
    
    def test_multiple_testing_correction_integration(self):
        """Test multiple testing correction for interaction p-values"""
        from week4_statistical_refinements import correct_interaction_pvalues
        
        # Mock interaction analysis results
        interaction_results = {
            'age_interaction': {'p_value': 0.02},
            'sex_interaction': {'p_value': 0.04}, 
            'comorbidity_interaction': {'p_value': 0.07},
            'baseline_utilization_interaction': {'p_value': 0.01}
        }
        
        corrected_results = correct_interaction_pvalues(
            interaction_results, method='benjamini_hochberg'
        )
        
        # Should add corrected p-values
        for interaction in interaction_results.keys():
            assert 'p_value_fdr' in corrected_results[interaction]
            assert 'significant_fdr' in corrected_results[interaction]
            
            # FDR p-values should be >= original p-values
            original_p = interaction_results[interaction]['p_value']
            fdr_p = corrected_results[interaction]['p_value_fdr']
            assert fdr_p >= original_p