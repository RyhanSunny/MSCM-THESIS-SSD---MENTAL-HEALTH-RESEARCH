#!/usr/bin/env python3
"""
Test suite for cluster-robust standard errors
Following TDD principles - tests written first per CLAUDE.md requirements
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cluster_robust_se import (
    calculate_cluster_robust_se,
    ClusterRobustRegression,
    cluster_robust_poisson,
    cluster_robust_logistic,
    validate_clustering_structure
)


class TestClusterRobustSE:
    """Test suite for cluster-robust standard errors implementation"""
    
    def test_clustering_structure_validation(self):
        """Test validation of clustering structure"""
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        # Valid clustering
        cluster_ids = np.random.randint(1, n_clusters + 1, n)
        result = validate_clustering_structure(cluster_ids)
        
        assert result['n_clusters'] == len(np.unique(cluster_ids))
        assert result['min_cluster_size'] >= 1
        assert result['max_cluster_size'] <= n
        assert result['is_valid'] is True
    
    def test_cluster_robust_vs_naive_se(self):
        """Test that cluster-robust SEs are larger than naive SEs with clustering"""
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        # Generate clustered data
        cluster_ids = np.random.randint(1, n_clusters + 1, n)
        cluster_effects = np.random.normal(0, 2, n_clusters)
        
        # Create data with cluster-level correlation
        X = np.random.normal(0, 1, n)
        cluster_effect = cluster_effects[cluster_ids - 1]
        Y = 2 + 1.5 * X + cluster_effect + np.random.normal(0, 1, n)
        
        # Fit models
        X_sm = sm.add_constant(X)
        
        # Naive model
        naive_model = sm.OLS(Y, X_sm).fit()
        naive_se = naive_model.bse[1]  # SE for X coefficient
        
        # Cluster-robust model
        robust_model = sm.OLS(Y, X_sm).fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
        robust_se = robust_model.bse[1]
        
        # Cluster-robust SE should be larger
        assert robust_se > naive_se
        
        # Test our wrapper function
        our_result = calculate_cluster_robust_se(X_sm, Y, cluster_ids, model_type='ols')
        assert abs(our_result['coef_se'][1] - robust_se) < 1e-6
    
    def test_poisson_cluster_robust(self):
        """Test cluster-robust standard errors for Poisson regression"""
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        # Generate clustered count data
        cluster_ids = np.random.randint(1, n_clusters + 1, n)
        X = np.random.normal(0, 1, n)
        
        # Poisson with clustering
        cluster_effects = np.random.normal(0, 0.5, n_clusters)
        linear_pred = 1 + 0.5 * X + cluster_effects[cluster_ids - 1]
        Y = np.random.poisson(np.exp(linear_pred))
        
        X_sm = sm.add_constant(X)
        
        # Fit Poisson with cluster-robust SE
        result = cluster_robust_poisson(X_sm, Y, cluster_ids)
        
        assert 'coef' in result
        assert 'coef_se' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'pvalues' in result
        
        # Check that we have coefficients for intercept and X
        assert len(result['coef']) == 2
        assert len(result['coef_se']) == 2
    
    def test_logistic_cluster_robust(self):
        """Test cluster-robust standard errors for logistic regression"""
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        # Generate clustered binary data
        cluster_ids = np.random.randint(1, n_clusters + 1, n)
        X = np.random.normal(0, 1, n)
        
        # Logistic with clustering
        cluster_effects = np.random.normal(0, 0.3, n_clusters)
        linear_pred = -0.5 + 0.8 * X + cluster_effects[cluster_ids - 1]
        prob = 1 / (1 + np.exp(-linear_pred))
        Y = np.random.binomial(1, prob)
        
        X_sm = sm.add_constant(X)
        
        # Fit logistic with cluster-robust SE
        result = cluster_robust_logistic(X_sm, Y, cluster_ids)
        
        assert 'coef' in result
        assert 'coef_se' in result
        assert 'odds_ratio' in result
        assert 'or_ci_lower' in result
        assert 'or_ci_upper' in result
        
        # Odds ratios should be positive
        assert all(or_val > 0 for or_val in result['odds_ratio'])
    
    def test_cluster_robust_regression_class(self):
        """Test the ClusterRobustRegression class interface"""
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        # Generate data
        cluster_ids = np.random.randint(1, n_clusters + 1, n)
        X = np.random.normal(0, 1, (n, 2))
        Y = 1 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.normal(0, 1, n)
        
        # Create regression object
        cr_reg = ClusterRobustRegression(model_type='ols')
        result = cr_reg.fit(X, Y, cluster_ids)
        
        assert hasattr(cr_reg, 'model_')
        assert hasattr(cr_reg, 'results_')
        assert 'coef' in result
        assert 'coef_se' in result
        
        # Test prediction
        X_new = np.random.normal(0, 1, (10, 2))
        predictions = cr_reg.predict(X_new)
        assert len(predictions) == 10
    
    def test_weighted_cluster_robust(self):
        """Test cluster-robust SE with inverse probability weights"""
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        # Generate data with weights
        cluster_ids = np.random.randint(1, n_clusters + 1, n)
        X = np.random.normal(0, 1, n)
        Y = 2 + 1.5 * X + np.random.normal(0, 1, n)
        weights = np.random.gamma(2, 0.5, n)  # IPTW-style weights
        
        X_sm = sm.add_constant(X)
        
        # Test weighted cluster-robust regression
        result = calculate_cluster_robust_se(
            X_sm, Y, cluster_ids, 
            model_type='ols', 
            weights=weights
        )
        
        assert 'coef' in result
        assert 'coef_se' in result
        assert 'weighted' in result
        assert result['weighted'] is True
    
    def test_insufficient_clusters_warning(self):
        """Test warning when number of clusters is too small"""
        np.random.seed(42)
        n = 100
        
        # Only 3 clusters (too few)
        cluster_ids = np.random.randint(1, 4, n)
        X = np.random.normal(0, 1, n)
        Y = 2 + 1.5 * X + np.random.normal(0, 1, n)
        
        X_sm = sm.add_constant(X)
        
        # Should warn about insufficient clusters
        with pytest.warns(UserWarning, match="few clusters"):
            result = calculate_cluster_robust_se(X_sm, Y, cluster_ids, model_type='ols')
        
        assert 'coef' in result  # Should still return results
    
    def test_integration_with_propensity_weights(self):
        """Test integration with propensity score weights from ps_match.py"""
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        # Simulate data similar to SSD analysis
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'site_id': np.random.randint(1, n_clusters + 1, n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n),
            'charlson_score': np.random.poisson(1, n),
            'total_encounters': np.random.poisson(8, n),
            'iptw': np.random.gamma(2, 0.5, n)  # From propensity score analysis
        })
        
        # Outcome model with clustering
        Y = df['total_encounters'].values
        X = sm.add_constant(df[['ssd_flag', 'age', 'sex_M', 'charlson_score']].values)
        cluster_ids = df['site_id'].values
        weights = df['iptw'].values
        
        # Fit weighted Poisson model with cluster-robust SE
        result = cluster_robust_poisson(X, Y, cluster_ids, weights=weights)
        
        # Should have results for all covariates
        assert len(result['coef']) == X.shape[1]
        assert len(result['coef_se']) == X.shape[1]
        
        # Treatment effect (ssd_flag) should be estimable
        treatment_coef = result['coef'][1]  # Assuming ssd_flag is second column
        treatment_se = result['coef_se'][1]
        
        assert not np.isnan(treatment_coef)
        assert not np.isnan(treatment_se)
        assert treatment_se > 0


class TestClusterDiagnostics:
    """Test cluster diagnostics and validation"""
    
    def test_cluster_size_distribution(self):
        """Test cluster size distribution analysis"""
        np.random.seed(42)
        n = 1000
        
        # Unbalanced clusters
        cluster_probs = np.array([0.1, 0.15, 0.25, 0.3, 0.2])
        cluster_ids = np.random.choice(range(1, 6), n, p=cluster_probs)
        
        result = validate_clustering_structure(cluster_ids)
        
        assert result['n_clusters'] == 5
        assert result['cluster_balance_ratio'] < 1.0  # Should detect imbalance
        assert 'cluster_sizes' in result
        
        # Check that largest cluster is much bigger than smallest
        cluster_sizes = result['cluster_sizes']
        assert max(cluster_sizes) / min(cluster_sizes) > 2
    
    def test_intracluster_correlation_estimate(self):
        """Test estimation of intracluster correlation coefficient"""
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        # Generate data with known ICC
        cluster_ids = np.random.randint(1, n_clusters + 1, n)
        
        # High ICC scenario
        cluster_effects = np.random.normal(0, 2, n_clusters)  # High between-cluster variance
        within_noise = np.random.normal(0, 0.5, n)  # Low within-cluster variance
        Y_high_icc = cluster_effects[cluster_ids - 1] + within_noise
        
        # Low ICC scenario  
        cluster_effects_low = np.random.normal(0, 0.5, n_clusters)  # Low between-cluster variance
        within_noise_high = np.random.normal(0, 2, n)  # High within-cluster variance
        Y_low_icc = cluster_effects_low[cluster_ids - 1] + within_noise_high
        
        result_high = validate_clustering_structure(cluster_ids, outcome=Y_high_icc)
        result_low = validate_clustering_structure(cluster_ids, outcome=Y_low_icc)
        
        # High ICC should be larger than low ICC
        if 'estimated_icc' in result_high and 'estimated_icc' in result_low:
            assert result_high['estimated_icc'] > result_low['estimated_icc']