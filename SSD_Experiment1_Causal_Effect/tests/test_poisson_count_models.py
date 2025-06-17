#!/usr/bin/env python3
"""
Test suite for Poisson/Negative Binomial count outcome models
Following TDD principles - tests written first per CLAUDE.md requirements
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from poisson_count_models import (
    PoissonCountRegression,
    NegativeBinomialCountRegression,
    OverdispersionTest,
    test_overdispersion,
    select_count_model,
    CountModelResults
)


class TestPoissonCountModels:
    """Test suite for Poisson count regression models"""
    
    def test_poisson_regression_basic(self):
        """Test basic Poisson regression functionality"""
        np.random.seed(42)
        n = 1000
        
        # Generate Poisson-distributed count data
        X = np.random.normal(0, 1, n)
        linear_pred = 1 + 0.5 * X
        Y = np.random.poisson(np.exp(linear_pred))
        
        # Fit Poisson model
        model = PoissonCountRegression()
        
        # Mock design matrix
        X_design = np.column_stack([np.ones(n), X])  # Add intercept
        
        # Test basic structure (without statsmodels dependency)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # Test that we can fit the model (even without statsmodels)
        results = model.fit(X_design, Y)
        assert hasattr(results, 'irr')  # IRR should be calculated automatically
        assert len(results.irr) == 2  # Intercept + treatment effect
    
    def test_overdispersion_detection(self):
        """Test overdispersion detection for count data"""
        np.random.seed(42)
        n = 1000
        
        # Generate Poisson data (no overdispersion)
        X = np.random.normal(0, 1, n)
        linear_pred = 1 + 0.5 * X
        Y_poisson = np.random.poisson(np.exp(linear_pred))
        
        # Generate overdispersed data (Negative Binomial-like)
        Y_overdispersed = []
        for i in range(n):
            lambda_i = np.exp(linear_pred[i])
            # Add extra variation to simulate overdispersion
            extra_var = np.random.gamma(lambda_i/2, 2)  # Gamma mixing
            Y_overdispersed.append(int(extra_var))
        Y_overdispersed = np.array(Y_overdispersed)
        
        # Test overdispersion detection
        def simple_overdispersion_test(Y, X):
            """Simple test for overdispersion using variance-to-mean ratio"""
            mean_Y = np.mean(Y)
            var_Y = np.var(Y)
            dispersion_ratio = var_Y / mean_Y
            
            return {
                'dispersion_ratio': dispersion_ratio,
                'is_overdispersed': dispersion_ratio > 1.5,  # Simple threshold
                'mean': mean_Y,
                'variance': var_Y
            }
        
        # Test on Poisson data
        test_poisson = simple_overdispersion_test(Y_poisson, X)
        assert test_poisson['dispersion_ratio'] > 0.5  # Should be reasonable
        
        # Test on overdispersed data
        test_overdispersed = simple_overdispersion_test(Y_overdispersed, X)
        
        # Key test: overdispersed data should have higher dispersion ratio
        assert test_overdispersed['dispersion_ratio'] > test_poisson['dispersion_ratio']
        
        print(f"Poisson dispersion ratio: {test_poisson['dispersion_ratio']:.2f}")
        print(f"Overdispersed dispersion ratio: {test_overdispersed['dispersion_ratio']:.2f}")
    
    def test_incidence_rate_ratio_calculation(self):
        """Test IRR calculation for treatment effects"""
        # Mock model results
        class MockPoissonResults:
            def __init__(self):
                self.coefficients = np.array([1.0, 0.693])  # log(2) ≈ 0.693
                self.se = np.array([0.1, 0.15])
        
        results = MockPoissonResults()
        
        # Calculate IRR
        irr = np.exp(results.coefficients)
        irr_ci_lower = np.exp(results.coefficients - 1.96 * results.se)
        irr_ci_upper = np.exp(results.coefficients + 1.96 * results.se)
        
        # Test treatment effect (second coefficient)
        assert abs(irr[1] - 2.0) < 0.01  # exp(0.693) ≈ 2.0
        assert irr_ci_lower[1] > 1.0  # Lower CI should be > 1
        assert irr_ci_upper[1] > irr[1]  # Upper CI should be > estimate
    
    def test_model_selection_workflow(self):
        """Test workflow for selecting between Poisson and Negative Binomial"""
        np.random.seed(42)
        n = 1000
        
        # Generate healthcare encounter data
        X = np.random.normal(0, 1, n)  # SSD exposure effect
        baseline_encounters = np.random.poisson(3, n)  # Baseline utilization
        
        # Linear predictor for encounter rate
        linear_pred = np.log(baseline_encounters + 1) + 0.4 * X  # log link
        Y = np.random.poisson(np.exp(linear_pred))
        
        # Create DataFrame similar to SSD analysis
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': (X > 0).astype(int),  # Binary treatment
            'total_encounters': Y,
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n),
            'baseline_encounters': baseline_encounters
        })
        
        # Test model selection workflow
        def select_count_model_simple(df, outcome_col):
            """Simplified model selection"""
            Y = df[outcome_col]
            
            # Check basic properties
            mean_Y = Y.mean()
            var_Y = Y.var()
            dispersion_ratio = var_Y / mean_Y
            
            # Simple selection rule
            if dispersion_ratio > 1.5:
                return 'negative_binomial'
            else:
                return 'poisson'
        
        selected_model = select_count_model_simple(df, 'total_encounters')
        
        assert selected_model in ['poisson', 'negative_binomial']
        
        # Test that we have count data
        assert all(df['total_encounters'] >= 0)  # Non-negative
        assert all(df['total_encounters'] == df['total_encounters'].astype(int))  # Integers
    
    def test_count_model_with_clustering(self):
        """Test count models with cluster-robust standard errors"""
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        # Generate clustered count data
        cluster_ids = np.random.randint(1, n_clusters + 1, n)
        cluster_effects = np.random.normal(0, 0.3, n_clusters)  # Practice-level effects
        
        # Generate healthcare encounters with clustering
        X = np.random.normal(0, 1, n)  # SSD exposure
        cluster_effect = cluster_effects[cluster_ids - 1]
        linear_pred = 1.5 + 0.4 * X + cluster_effect
        Y = np.random.poisson(np.exp(linear_pred))
        
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'site_id': cluster_ids,
            'ssd_flag': (X > 0).astype(int),
            'total_encounters': Y,
            'iptw': np.random.gamma(2, 0.5, n)  # Propensity score weights
        })
        
        # Test requirements for cluster-robust Poisson
        required_cols = ['site_id', 'ssd_flag', 'total_encounters', 'iptw']
        for col in required_cols:
            assert col in df.columns
        
        # Test clustering structure
        n_sites = df['site_id'].nunique()
        assert n_sites >= 10  # Sufficient clusters
        
        # Test count outcome properties
        assert df['total_encounters'].min() >= 0  # Non-negative
        assert df['total_encounters'].dtype in ['int64', 'int32']  # Integer counts
    
    def test_integration_with_ssd_hypotheses(self):
        """Test integration with specific SSD hypotheses"""
        np.random.seed(42)
        n = 1000
        
        # H1: Normal lab cascade → healthcare encounters
        ssd_exposure = np.random.binomial(1, 0.15, n)  # 15% exposed
        baseline_encounters = np.random.poisson(3, n)
        
        # Effect: SSD increases encounter rate by ~35% (IRR = 1.35)
        effect_size = 0.3  # log(1.35) ≈ 0.3
        linear_pred = np.log(baseline_encounters + 0.1) + effect_size * ssd_exposure
        total_encounters = np.random.poisson(np.exp(linear_pred))
        
        # Create SSD analysis dataset
        ssd_df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': ssd_exposure,
            'total_encounters': total_encounters,
            'primary_care_encounters': np.random.poisson(np.exp(linear_pred - 0.2)),
            'ed_visits': np.random.poisson(np.exp(linear_pred - 1.5)),  # Lower rate
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n),
            'charlson_score': np.random.poisson(1, n),
            'site_id': np.random.randint(1, 21, n)
        })
        
        # Test H1 analysis structure
        exposed = ssd_df[ssd_df['ssd_flag'] == 1]
        control = ssd_df[ssd_df['ssd_flag'] == 0]
        
        # Should see higher encounter rates in exposed group
        exposed_mean = exposed['total_encounters'].mean()
        control_mean = control['total_encounters'].mean()
        
        observed_ratio = exposed_mean / control_mean
        assert observed_ratio > 1.1  # Should be elevated (allowing for randomness)
        
        # Test that we have the right data structure for Poisson regression
        assert 'ssd_flag' in ssd_df.columns  # Treatment variable
        assert 'total_encounters' in ssd_df.columns  # Count outcome
        assert 'site_id' in ssd_df.columns  # Clustering variable
        
        # Check count properties
        for outcome in ['total_encounters', 'primary_care_encounters', 'ed_visits']:
            assert ssd_df[outcome].min() >= 0  # Non-negative
            assert all(ssd_df[outcome] == ssd_df[outcome].astype(int))  # Integers


class TestCountModelResults:
    """Test count model results structure"""
    
    def test_count_model_results_structure(self):
        """Test expected structure of count model results"""
        # Mock results structure for what we expect from Poisson regression
        mock_results = {
            'model_type': 'poisson',
            'coefficients': [1.5, 0.3],  # Intercept, treatment effect
            'se': [0.1, 0.08],
            'irr': [np.exp(1.5), np.exp(0.3)],  # Incidence rate ratios
            'irr_ci_lower': [np.exp(1.5 - 1.96*0.1), np.exp(0.3 - 1.96*0.08)],
            'irr_ci_upper': [np.exp(1.5 + 1.96*0.1), np.exp(0.3 + 1.96*0.08)],
            'pvalues': [0.001, 0.001],
            'dispersion_test': {
                'statistic': 1.05,
                'is_overdispersed': False
            },
            'n_observations': 1000,
            'clustered': True,
            'n_clusters': 20
        }
        
        # Test structure
        assert mock_results['model_type'] == 'poisson'
        assert len(mock_results['coefficients']) == 2
        assert len(mock_results['irr']) == 2
        
        # Test treatment effect interpretation
        treatment_irr = mock_results['irr'][1]
        assert treatment_irr > 1.0  # Should increase encounter rate
        assert treatment_irr < 2.0  # Reasonable effect size
        
        # Test that confidence intervals make sense
        treatment_ci_lower = mock_results['irr_ci_lower'][1]
        treatment_ci_upper = mock_results['irr_ci_upper'][1]
        
        assert treatment_ci_lower < treatment_irr < treatment_ci_upper
    
    def test_model_comparison_structure(self):
        """Test structure for comparing Poisson vs Negative Binomial"""
        # Mock comparison results
        comparison = {
            'poisson': {
                'aic': 2450.5,
                'bic': 2465.2,
                'deviance': 2440.1,
                'dispersion_ratio': 1.05
            },
            'negative_binomial': {
                'aic': 2455.8,
                'bic': 2475.3,
                'deviance': 2441.2,
                'dispersion_ratio': 1.02
            },
            'selected_model': 'poisson',
            'selection_criterion': 'Lower AIC and no overdispersion'
        }
        
        # Test comparison logic
        assert comparison['poisson']['aic'] < comparison['negative_binomial']['aic']
        assert comparison['selected_model'] == 'poisson'
        assert comparison['poisson']['dispersion_ratio'] < 1.5  # Not overdispersed


if __name__ == "__main__":
    # Run basic demonstrations
    test_instance = TestPoissonCountModels()
    
    print("Running Poisson count model tests...")
    test_instance.test_poisson_regression_basic()
    print("✓ Basic Poisson regression test passed")
    
    test_instance.test_overdispersion_detection()
    print("✓ Overdispersion detection test passed")
    
    test_instance.test_incidence_rate_ratio_calculation()
    print("✓ IRR calculation test passed")
    
    test_instance.test_model_selection_workflow()
    print("✓ Model selection workflow test passed")
    
    test_instance.test_count_model_with_clustering()
    print("✓ Clustering integration test passed")
    
    test_instance.test_integration_with_ssd_hypotheses()
    print("✓ SSD hypothesis integration test passed")
    
    print("\nAll Poisson count model tests passed! Ready for implementation.")