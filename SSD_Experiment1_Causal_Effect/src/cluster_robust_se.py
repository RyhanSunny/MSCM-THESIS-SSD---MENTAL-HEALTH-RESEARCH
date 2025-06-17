#!/usr/bin/env python3
"""
Cluster-Robust Standard Errors Module

Implements cluster-robust standard errors for causal inference in multi-site data.
Addresses within-practice correlation following Cameron & Miller (2015) recommendations.

Key Features:
- Cluster-robust SE for OLS, Poisson, and Logistic regression
- Integration with inverse probability weights (IPTW)
- Cluster diagnostics and validation
- Support for statsmodels and econml workflows

Author: Ryhan Suny
Date: 2025-06-17
Version: 1.0.0

References:
- Cameron, A.C. & Miller, D.L. (2015). A practitioner's guide to cluster-robust inference. 
  Journal of Human Resources, 50(2), 317-372.
- Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data. MIT Press.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import warnings
from typing import Dict, Any, Union, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_clustering_structure(
    cluster_ids: Union[np.ndarray, pd.Series],
    outcome: Optional[np.ndarray] = None,
    min_clusters: int = 10
) -> Dict[str, Any]:
    """
    Validate clustering structure for cluster-robust inference.
    
    Args:
        cluster_ids: Cluster identifiers for each observation
        outcome: Optional outcome variable for ICC estimation
        min_clusters: Minimum recommended number of clusters
        
    Returns:
        Dictionary with clustering diagnostics
        
    References:
        Cameron & Miller (2015) recommend ≥10 clusters for reliable inference
    """
    if isinstance(cluster_ids, pd.Series):
        cluster_ids = cluster_ids.values
    
    cluster_ids = np.asarray(cluster_ids)
    n_obs = len(cluster_ids)
    
    # Basic cluster statistics
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    
    # Calculate cluster sizes
    cluster_sizes = []
    for cluster in unique_clusters:
        cluster_size = np.sum(cluster_ids == cluster)
        cluster_sizes.append(cluster_size)
    
    cluster_sizes = np.array(cluster_sizes)
    
    # Balance diagnostics
    min_cluster_size = np.min(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    mean_cluster_size = np.mean(cluster_sizes)
    cluster_balance_ratio = min_cluster_size / max_cluster_size
    
    # Validation checks
    sufficient_clusters = n_clusters >= min_clusters
    reasonable_balance = cluster_balance_ratio >= 0.1  # No cluster >10x others
    
    is_valid = sufficient_clusters and reasonable_balance
    
    # Estimate intracluster correlation if outcome provided
    estimated_icc = None
    if outcome is not None:
        try:
            estimated_icc = estimate_icc(outcome, cluster_ids)
        except Exception as e:
            logger.warning(f"Could not estimate ICC: {e}")
    
    # Warnings
    if not sufficient_clusters:
        logger.warning(f"Only {n_clusters} clusters detected. "
                      f"Cameron & Miller (2015) recommend ≥{min_clusters} for reliable inference.")
    
    if not reasonable_balance:
        logger.warning(f"Unbalanced clusters detected. "
                      f"Largest cluster is {max_cluster_size/min_cluster_size:.1f}x the smallest.")
    
    return {
        'n_observations': int(n_obs),
        'n_clusters': int(n_clusters),
        'cluster_sizes': cluster_sizes.tolist(),
        'min_cluster_size': int(min_cluster_size),
        'max_cluster_size': int(max_cluster_size),
        'mean_cluster_size': float(mean_cluster_size),
        'cluster_balance_ratio': float(cluster_balance_ratio),
        'sufficient_clusters': bool(sufficient_clusters),
        'reasonable_balance': bool(reasonable_balance),
        'is_valid': bool(is_valid),
        'estimated_icc': float(estimated_icc) if estimated_icc is not None else None,
        'timestamp': datetime.now().isoformat()
    }


def estimate_icc(outcome: np.ndarray, cluster_ids: np.ndarray) -> float:
    """
    Estimate intracluster correlation coefficient using ANOVA method.
    
    Args:
        outcome: Outcome variable
        cluster_ids: Cluster identifiers
        
    Returns:
        Estimated ICC
        
    Notes:
        ICC = (MSB - MSW) / (MSB + (n0-1)*MSW)
        where MSB = mean square between, MSW = mean square within, n0 = average cluster size
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({'outcome': outcome, 'cluster': cluster_ids})
    
    # Calculate ANOVA components
    overall_mean = df['outcome'].mean()
    n_total = len(df)
    n_clusters = df['cluster'].nunique()
    
    # Between-cluster sum of squares
    cluster_means = df.groupby('cluster')['outcome'].mean()
    cluster_sizes = df.groupby('cluster').size()
    ssb = np.sum(cluster_sizes * (cluster_means - overall_mean)**2)
    msb = ssb / (n_clusters - 1)
    
    # Within-cluster sum of squares
    ssw = 0
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]['outcome']
        cluster_mean = cluster_data.mean()
        ssw += np.sum((cluster_data - cluster_mean)**2)
    
    msw = ssw / (n_total - n_clusters)
    
    # Average cluster size (for unbalanced design)
    n0 = (n_total - np.sum(cluster_sizes**2) / n_total) / (n_clusters - 1)
    
    # ICC calculation
    if msb + (n0 - 1) * msw == 0:
        return 0.0
    
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    
    # ICC should be between 0 and 1
    icc = max(0, min(1, icc))
    
    return icc


def calculate_cluster_robust_se(
    X: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    model_type: str = 'ols',
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate cluster-robust standard errors for various model types.
    
    Args:
        X: Design matrix (with intercept if desired)
        y: Outcome variable
        cluster_ids: Cluster identifiers
        model_type: 'ols', 'poisson', or 'logistic'
        weights: Optional weights (e.g., IPTW weights)
        alpha: Significance level for confidence intervals
        
    Returns:
        Dictionary with coefficients, cluster-robust SEs, and CIs
    """
    # Validate inputs
    if len(X) != len(y) != len(cluster_ids):
        raise ValueError("X, y, and cluster_ids must have same length")
    
    if weights is not None and len(weights) != len(y):
        raise ValueError("weights must have same length as y")
    
    # Check cluster structure
    cluster_diagnostics = validate_clustering_structure(cluster_ids, y)
    
    if not cluster_diagnostics['sufficient_clusters']:
        warnings.warn(f"Only {cluster_diagnostics['n_clusters']} clusters detected. "
                     "Results may be unreliable with too few clusters.", 
                     UserWarning)
    
    # Fit model with cluster-robust standard errors
    try:
        if model_type.lower() == 'ols':
            if weights is not None:
                model = sm.WLS(y, X, weights=weights)
            else:
                model = sm.OLS(y, X)
            
            fitted_model = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
        elif model_type.lower() == 'poisson':
            if weights is not None:
                # Weighted Poisson regression
                model = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=weights)
            else:
                model = sm.GLM(y, X, family=sm.families.Poisson())
            
            fitted_model = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
        elif model_type.lower() == 'logistic':
            if weights is not None:
                model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
            else:
                model = sm.GLM(y, X, family=sm.families.Binomial())
            
            fitted_model = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
            
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # Extract results
        coef = fitted_model.params.values
        coef_se = fitted_model.bse.values
        pvalues = fitted_model.pvalues.values
        
        # Calculate confidence intervals
        ci_lower = fitted_model.conf_int(alpha=alpha).iloc[:, 0].values
        ci_upper = fitted_model.conf_int(alpha=alpha).iloc[:, 1].values
        
        # Additional results for specific model types
        additional_results = {}
        
        if model_type.lower() == 'poisson':
            # Incidence rate ratios
            additional_results['irr'] = np.exp(coef)
            additional_results['irr_ci_lower'] = np.exp(ci_lower)
            additional_results['irr_ci_upper'] = np.exp(ci_upper)
            
            # Overdispersion test
            pearson_residuals = fitted_model.resid_pearson
            dispersion_stat = np.sum(pearson_residuals**2) / fitted_model.df_resid
            additional_results['dispersion_statistic'] = float(dispersion_stat)
            additional_results['overdispersed'] = bool(dispersion_stat > 1.5)
            
        elif model_type.lower() == 'logistic':
            # Odds ratios
            additional_results['odds_ratio'] = np.exp(coef)
            additional_results['or_ci_lower'] = np.exp(ci_lower)
            additional_results['or_ci_upper'] = np.exp(ci_upper)
        
        results = {
            'coef': coef.tolist(),
            'coef_se': coef_se.tolist(),
            'pvalues': pvalues.tolist(),
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist(),
            'model_type': model_type,
            'weighted': weights is not None,
            'n_observations': int(len(y)),
            'n_clusters': int(cluster_diagnostics['n_clusters']),
            'cluster_diagnostics': cluster_diagnostics,
            'fitted_model': fitted_model,  # For further analysis if needed
            **additional_results
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error fitting {model_type} model with cluster-robust SE: {e}")
        raise


def cluster_robust_poisson(
    X: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Convenience function for Poisson regression with cluster-robust SEs.
    
    Particularly useful for count outcomes like healthcare encounters.
    """
    return calculate_cluster_robust_se(X, y, cluster_ids, 'poisson', weights, alpha)


def cluster_robust_logistic(
    X: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Convenience function for logistic regression with cluster-robust SEs.
    
    Particularly useful for binary outcomes like ED visits.
    """
    return calculate_cluster_robust_se(X, y, cluster_ids, 'logistic', weights, alpha)


class ClusterRobustRegression:
    """
    Scikit-learn style interface for cluster-robust regression.
    
    Useful for integration with ML pipelines while maintaining
    proper uncertainty quantification.
    """
    
    def __init__(self, model_type: str = 'ols', alpha: float = 0.05):
        """
        Initialize cluster-robust regression.
        
        Args:
            model_type: 'ols', 'poisson', or 'logistic'
            alpha: Significance level for confidence intervals
        """
        self.model_type = model_type
        self.alpha = alpha
        self.model_ = None
        self.results_ = None
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cluster_ids: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Fit cluster-robust regression model.
        
        Args:
            X: Design matrix (intercept will be added if not present)
            y: Outcome variable
            cluster_ids: Cluster identifiers
            weights: Optional weights
            
        Returns:
            Model results dictionary
        """
        # Add intercept if not present
        if not np.allclose(X[:, 0], 1):
            X = sm.add_constant(X)
        
        self.results_ = calculate_cluster_robust_se(
            X, y, cluster_ids, self.model_type, weights, self.alpha
        )
        
        self.model_ = self.results_['fitted_model']
        
        return self.results_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted model.
        
        Args:
            X: Design matrix for prediction
            
        Returns:
            Predicted values
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Add intercept if not present
        if not np.allclose(X[:, 0], 1):
            X = sm.add_constant(X)
        
        return self.model_.predict(X)
    
    def summary(self) -> str:
        """Return model summary string"""
        if self.results_ is None:
            return "Model not fitted"
        
        summary = f"Cluster-Robust {self.model_type.upper()} Regression Results\n"
        summary += "=" * 55 + "\n"
        summary += f"Number of observations: {self.results_['n_observations']}\n"
        summary += f"Number of clusters: {self.results_['n_clusters']}\n"
        summary += f"Weighted: {self.results_['weighted']}\n"
        summary += "\nCoefficients:\n"
        
        for i, (coef, se, pval) in enumerate(zip(
            self.results_['coef'],
            self.results_['coef_se'], 
            self.results_['pvalues']
        )):
            significance = ""
            if pval < 0.001:
                significance = "***"
            elif pval < 0.01:
                significance = "**"
            elif pval < 0.05:
                significance = "*"
            
            summary += f"  X{i}: {coef:8.4f} (SE: {se:6.4f}) {significance}\n"
        
        summary += "\nSignificance: *** p<0.001, ** p<0.01, * p<0.05\n"
        
        return summary


def main():
    """CLI interface for cluster-robust regression"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cluster-robust standard errors for causal inference"
    )
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with simulated data')
    
    args = parser.parse_args()
    
    if args.demo:
        # Generate demonstration data
        np.random.seed(42)
        n = 1000
        n_clusters = 20
        
        print("Generating clustered demonstration data...")
        
        # Clustered data
        cluster_ids = np.random.randint(1, n_clusters + 1, n)
        cluster_effects = np.random.normal(0, 1, n_clusters)
        
        X = np.random.normal(0, 1, n)
        cluster_effect = cluster_effects[cluster_ids - 1]
        y = 2 + 1.5 * X + cluster_effect + np.random.normal(0, 0.5, n)
        
        X_matrix = sm.add_constant(X)
        
        print("\nFitting cluster-robust OLS model...")
        result = calculate_cluster_robust_se(X_matrix, y, cluster_ids, 'ols')
        
        print(f"Number of clusters: {result['n_clusters']}")
        print(f"Coefficient: {result['coef'][1]:.4f}")
        print(f"Cluster-robust SE: {result['coef_se'][1]:.4f}")
        print(f"95% CI: [{result['ci_lower'][1]:.4f}, {result['ci_upper'][1]:.4f}]")
        
        # Compare with naive SE
        naive_model = sm.OLS(y, X_matrix).fit()
        naive_se = naive_model.bse[1]
        
        print(f"Naive SE: {naive_se:.4f}")
        print(f"SE inflation factor: {result['coef_se'][1] / naive_se:.2f}")
        
        print("\n✓ Demonstration completed successfully")


if __name__ == "__main__":
    main()