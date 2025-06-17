#!/usr/bin/env python3
"""
Poisson/Negative Binomial Count Models Module

Implements appropriate regression models for healthcare encounter count outcomes.
Addresses Cameron & Trivedi (2013) recommendations for count data analysis.

Key Features:
- Poisson regression for count outcomes
- Negative Binomial for overdispersed count data
- Overdispersion testing and model selection
- Incidence Rate Ratio (IRR) calculation
- Integration with cluster-robust standard errors

Author: Ryhan Suny
Date: 2025-06-17
Version: 1.0.0

References:
- Cameron, A.C. & Trivedi, P.K. (2013). Regression Analysis of Count Data. Cambridge.
- Hilbe, J.M. (2011). Negative Binomial Regression. Cambridge University Press.
- McCullagh, P. & Nelder, J.A. (1989). Generalized Linear Models. Chapman & Hall.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Union, Optional, Tuple, List
import logging
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CountModelResults:
    """Container for count model results"""
    
    def __init__(self, model_type: str, coefficients: np.ndarray, se: np.ndarray, 
                 pvalues: np.ndarray, n_obs: int, **kwargs):
        self.model_type = model_type
        self.coefficients = coefficients
        self.se = se
        self.pvalues = pvalues
        self.n_obs = n_obs
        
        # Additional results
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Calculate derived quantities
        self._calculate_irr()
        self._calculate_confidence_intervals()
    
    def _calculate_irr(self):
        """Calculate Incidence Rate Ratios"""
        self.irr = np.exp(self.coefficients)
    
    def _calculate_confidence_intervals(self, alpha: float = 0.05):
        """Calculate confidence intervals"""
        z_score = stats.norm.ppf(1 - alpha/2)
        
        # Coefficient CIs
        self.ci_lower = self.coefficients - z_score * self.se
        self.ci_upper = self.coefficients + z_score * self.se
        
        # IRR CIs
        self.irr_ci_lower = np.exp(self.ci_lower)
        self.irr_ci_upper = np.exp(self.ci_upper)
    
    def summary(self) -> str:
        """Return formatted summary"""
        summary = f"{self.model_type.title()} Regression Results\n"
        summary += "=" * 40 + "\n"
        summary += f"Number of observations: {self.n_obs}\n"
        
        if hasattr(self, 'n_clusters'):
            summary += f"Number of clusters: {self.n_clusters}\n"
        
        summary += "\nCoefficients:\n"
        for i, (coef, se, irr, pval) in enumerate(zip(
            self.coefficients, self.se, self.irr, self.pvalues
        )):
            sig = ""
            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            
            summary += f"  Coef{i}: {coef:8.4f} (SE: {se:6.4f}) IRR: {irr:6.3f} {sig}\n"
        
        if hasattr(self, 'dispersion_statistic'):
            summary += f"\nDispersion test: {self.dispersion_statistic:.3f}\n"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            'model_type': self.model_type,
            'coefficients': self.coefficients.tolist(),
            'se': self.se.tolist(),
            'irr': self.irr.tolist(),
            'irr_ci_lower': self.irr_ci_lower.tolist(),
            'irr_ci_upper': self.irr_ci_upper.tolist(),
            'pvalues': self.pvalues.tolist(),
            'n_observations': self.n_obs,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add optional attributes
        for attr in ['dispersion_statistic', 'n_clusters', 'clustered', 'overdispersed']:
            if hasattr(self, attr):
                result[attr] = getattr(self, attr)
        
        return result


class OverdispersionTest:
    """Test for overdispersion in count data"""
    
    @staticmethod
    def variance_to_mean_ratio(y: np.ndarray) -> float:
        """Simple variance-to-mean ratio test"""
        mean_y = np.mean(y)
        var_y = np.var(y, ddof=1)
        
        if mean_y == 0:
            return np.inf
        
        return var_y / mean_y
    
    @staticmethod
    def pearson_dispersion_test(y: np.ndarray, fitted_values: np.ndarray) -> Dict[str, float]:
        """Pearson chi-square dispersion test"""
        # Pearson residuals
        pearson_resid = (y - fitted_values) / np.sqrt(fitted_values)
        
        # Dispersion statistic
        dispersion_stat = np.sum(pearson_resid**2) / (len(y) - 2)  # Assuming 2 parameters
        
        # Under null hypothesis of Poisson, dispersion should be ~1
        # Values > 1.5 suggest overdispersion
        is_overdispersed = dispersion_stat > 1.5
        
        return {
            'dispersion_statistic': float(dispersion_stat),
            'is_overdispersed': bool(is_overdispersed),
            'pearson_residuals': pearson_resid
        }
    
    @staticmethod
    def cameron_trivedi_test(y: np.ndarray, fitted_values: np.ndarray) -> Dict[str, float]:
        """Cameron & Trivedi (1990) overdispersion test"""
        # Auxiliary regression: (y - mu)^2 - mu on mu
        auxiliary_y = (y - fitted_values)**2 - fitted_values
        auxiliary_x = fitted_values
        
        # Simple linear regression
        n = len(y)
        x_mean = np.mean(auxiliary_x)
        y_mean = np.mean(auxiliary_y)
        
        numerator = np.sum((auxiliary_x - x_mean) * (auxiliary_y - y_mean))
        denominator = np.sum((auxiliary_x - x_mean)**2)
        
        if denominator == 0:
            return {'test_statistic': 0, 'p_value': 1.0, 'is_overdispersed': False}
        
        slope = numerator / denominator
        
        # Standard error of slope
        residuals = auxiliary_y - (y_mean + slope * (auxiliary_x - x_mean))
        mse = np.sum(residuals**2) / (n - 2)
        se_slope = np.sqrt(mse / denominator)
        
        # Test statistic
        if se_slope == 0:
            t_stat = 0
        else:
            t_stat = slope / se_slope
        
        # P-value (one-tailed test)
        p_value = 1 - stats.t.cdf(t_stat, df=n-2)
        
        return {
            'test_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_overdispersed': bool(p_value < 0.05)
        }


class PoissonCountRegression:
    """Poisson regression for count data"""
    
    def __init__(self, cluster_col: Optional[str] = None):
        self.cluster_col = cluster_col
        self.fitted_model = None
        self.results = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None,
            cluster_ids: Optional[np.ndarray] = None) -> CountModelResults:
        """
        Fit Poisson regression model.
        
        Args:
            X: Design matrix (with intercept)
            y: Count outcome variable
            weights: Optional weights (e.g., IPTW)
            cluster_ids: Optional cluster identifiers
            
        Returns:
            CountModelResults object
        """
        try:
            # Try statsmodels if available
            import statsmodels.api as sm
            
            if weights is not None:
                model = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=weights)
            else:
                model = sm.GLM(y, X, family=sm.families.Poisson())
            
            if cluster_ids is not None:
                fitted_model = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
                clustered = True
                n_clusters = len(np.unique(cluster_ids))
            else:
                fitted_model = model.fit()
                clustered = False
                n_clusters = None
            
            # Extract results
            coefficients = fitted_model.params.values
            se = fitted_model.bse.values
            pvalues = fitted_model.pvalues.values
            fitted_values = fitted_model.fittedvalues.values
            
            # Overdispersion test
            dispersion_test = OverdispersionTest.pearson_dispersion_test(y, fitted_values)
            
            self.results = CountModelResults(
                model_type='poisson',
                coefficients=coefficients,
                se=se,
                pvalues=pvalues,
                n_obs=len(y),
                fitted_values=fitted_values,
                dispersion_statistic=dispersion_test['dispersion_statistic'],
                overdispersed=dispersion_test['is_overdispersed'],
                clustered=clustered,
                n_clusters=n_clusters,
                fitted_model=fitted_model
            )
            
        except ImportError:
            # Fallback to simple implementation
            logger.warning("Statsmodels not available, using simplified Poisson regression")
            self.results = self._fit_simple_poisson(X, y, weights, cluster_ids)
        
        self.fitted_model = self.results
        return self.results
    
    def _fit_simple_poisson(self, X: np.ndarray, y: np.ndarray, 
                           weights: Optional[np.ndarray] = None,
                           cluster_ids: Optional[np.ndarray] = None) -> CountModelResults:
        """Simplified Poisson regression using scipy optimization"""
        
        def poisson_loglik(params, X, y, weights=None):
            """Negative log-likelihood for Poisson regression"""
            linear_pred = X @ params
            mu = np.exp(linear_pred)
            
            # Avoid numerical issues
            mu = np.clip(mu, 1e-8, 1e8)
            
            # Log-likelihood
            loglik = y * np.log(mu) - mu - np.sum(np.log(np.maximum(np.arange(1, np.max(y)+1), 1)))
            
            if weights is not None:
                loglik = loglik * weights
            
            return -np.sum(loglik)
        
        # Initial values
        initial_params = np.zeros(X.shape[1])
        initial_params[0] = np.log(np.mean(y) + 0.1)  # Intercept
        
        # Optimize
        try:
            result = minimize(poisson_loglik, initial_params, args=(X, y, weights), 
                            method='BFGS')
            coefficients = result.x
            
            # Approximate standard errors using inverse Hessian
            if result.hess_inv is not None:
                se = np.sqrt(np.diag(result.hess_inv))
            else:
                se = np.full_like(coefficients, 0.1)  # Fallback
            
            # Calculate fitted values
            fitted_values = np.exp(X @ coefficients)
            
            # Simple p-values (assuming normal approximation)
            z_stats = coefficients / se
            pvalues = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
            
            # Overdispersion test
            dispersion_test = OverdispersionTest.variance_to_mean_ratio(y)
            
            return CountModelResults(
                model_type='poisson',
                coefficients=coefficients,
                se=se,
                pvalues=pvalues,
                n_obs=len(y),
                fitted_values=fitted_values,
                dispersion_statistic=dispersion_test,
                overdispersed=dispersion_test > 1.5,
                clustered=cluster_ids is not None,
                n_clusters=len(np.unique(cluster_ids)) if cluster_ids is not None else None
            )
            
        except Exception as e:
            logger.error(f"Poisson regression failed: {e}")
            # Return null results
            return CountModelResults(
                model_type='poisson',
                coefficients=np.zeros(X.shape[1]),
                se=np.ones(X.shape[1]),
                pvalues=np.ones(X.shape[1]),
                n_obs=len(y),
                fitted_values=np.ones_like(y, dtype=float),
                dispersion_statistic=1.0,
                overdispersed=False,
                clustered=False
            )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.results, 'fitted_model') and hasattr(self.results.fitted_model, 'predict'):
            return self.results.fitted_model.predict(X)
        else:
            return np.exp(X @ self.results.coefficients)


class NegativeBinomialCountRegression:
    """Negative Binomial regression for overdispersed count data"""
    
    def __init__(self, cluster_col: Optional[str] = None):
        self.cluster_col = cluster_col
        self.fitted_model = None
        self.results = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None,
            cluster_ids: Optional[np.ndarray] = None) -> CountModelResults:
        """Fit Negative Binomial regression model"""
        
        try:
            import statsmodels.api as sm
            
            if weights is not None:
                model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), freq_weights=weights)
            else:
                model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
            
            if cluster_ids is not None:
                fitted_model = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_ids})
                clustered = True
                n_clusters = len(np.unique(cluster_ids))
            else:
                fitted_model = model.fit()
                clustered = False
                n_clusters = None
            
            coefficients = fitted_model.params.values
            se = fitted_model.bse.values
            pvalues = fitted_model.pvalues.values
            fitted_values = fitted_model.fittedvalues.values
            
            self.results = CountModelResults(
                model_type='negative_binomial',
                coefficients=coefficients,
                se=se,
                pvalues=pvalues,
                n_obs=len(y),
                fitted_values=fitted_values,
                dispersion_statistic=1.0,  # NB handles overdispersion by design
                overdispersed=False,
                clustered=clustered,
                n_clusters=n_clusters,
                fitted_model=fitted_model
            )
            
        except ImportError:
            logger.warning("Statsmodels not available, falling back to Poisson")
            # Fall back to Poisson if NB not available
            poisson_model = PoissonCountRegression(self.cluster_col)
            self.results = poisson_model.fit(X, y, weights, cluster_ids)
            self.results.model_type = 'negative_binomial_fallback'
        
        self.fitted_model = self.results
        return self.results


def test_overdispersion(y: np.ndarray, fitted_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Test for overdispersion in count data"""
    
    # Simple variance-to-mean test
    var_mean_ratio = OverdispersionTest.variance_to_mean_ratio(y)
    
    results = {
        'variance_to_mean_ratio': var_mean_ratio,
        'simple_overdispersed': var_mean_ratio > 1.5
    }
    
    # More sophisticated tests if fitted values available
    if fitted_values is not None:
        pearson_test = OverdispersionTest.pearson_dispersion_test(y, fitted_values)
        cameron_trivedi_test = OverdispersionTest.cameron_trivedi_test(y, fitted_values)
        
        results.update({
            'pearson_dispersion': pearson_test,
            'cameron_trivedi_test': cameron_trivedi_test,
            'overdispersed': (pearson_test['is_overdispersed'] or 
                            cameron_trivedi_test['is_overdispersed'])
        })
    else:
        results['overdispersed'] = results['simple_overdispersed']
    
    return results


def select_count_model(X: np.ndarray, y: np.ndarray, 
                      weights: Optional[np.ndarray] = None,
                      cluster_ids: Optional[np.ndarray] = None) -> Tuple[str, CountModelResults]:
    """
    Select appropriate count model (Poisson vs Negative Binomial) based on overdispersion.
    
    Args:
        X: Design matrix
        y: Count outcome
        weights: Optional weights
        cluster_ids: Optional cluster identifiers
        
    Returns:
        Tuple of (selected_model_name, fitted_results)
    """
    logger.info("Testing for overdispersion to select count model")
    
    # Fit Poisson model first
    poisson_model = PoissonCountRegression()
    poisson_results = poisson_model.fit(X, y, weights, cluster_ids)
    
    # Check for overdispersion
    if hasattr(poisson_results, 'overdispersed') and poisson_results.overdispersed:
        logger.info("Overdispersion detected, fitting Negative Binomial model")
        
        nb_model = NegativeBinomialCountRegression()
        nb_results = nb_model.fit(X, y, weights, cluster_ids)
        
        return 'negative_binomial', nb_results
    else:
        logger.info("No overdispersion detected, using Poisson model")
        return 'poisson', poisson_results


def count_regression_analysis(df: pd.DataFrame, 
                             outcome_col: str,
                             treatment_col: str,
                             covariate_cols: List[str],
                             cluster_col: Optional[str] = None,
                             weight_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete count regression analysis workflow.
    
    Args:
        df: DataFrame with analysis data
        outcome_col: Name of count outcome column
        treatment_col: Name of treatment/exposure column
        covariate_cols: List of covariate column names
        cluster_col: Optional cluster column for robust SE
        weight_col: Optional weight column (e.g., IPTW)
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Running count regression analysis for {outcome_col}")
    
    # Prepare data
    y = df[outcome_col].values
    X_cols = [treatment_col] + covariate_cols
    X = df[X_cols].values
    
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])
    
    # Optional weights and clustering
    weights = df[weight_col].values if weight_col and weight_col in df.columns else None
    cluster_ids = df[cluster_col].values if cluster_col and cluster_col in df.columns else None
    
    # Validate count data
    if np.any(y < 0):
        raise ValueError("Count outcome must be non-negative")
    
    if not np.all(y == y.astype(int)):
        logger.warning("Outcome values are not integers, rounding to nearest integer")
        y = np.round(y).astype(int)
    
    # Model selection and fitting
    selected_model, results = select_count_model(X, y, weights, cluster_ids)
    
    # Compile analysis results
    analysis_results = {
        'outcome_variable': outcome_col,
        'treatment_variable': treatment_col,
        'covariates': covariate_cols,
        'selected_model': selected_model,
        'model_results': results.to_dict(),
        'treatment_effect': {
            'coefficient': float(results.coefficients[1]),  # Assuming treatment is first after intercept
            'se': float(results.se[1]),
            'irr': float(results.irr[1]),
            'irr_ci_lower': float(results.irr_ci_lower[1]),
            'irr_ci_upper': float(results.irr_ci_upper[1]),
            'p_value': float(results.pvalues[1])
        },
        'clustered': cluster_ids is not None,
        'weighted': weights is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Log key results
    treatment_irr = results.irr[1]
    treatment_ci_lower = results.irr_ci_lower[1]
    treatment_ci_upper = results.irr_ci_upper[1]
    
    logger.info(f"Treatment effect (IRR): {treatment_irr:.3f} "
               f"({treatment_ci_lower:.3f}, {treatment_ci_upper:.3f})")
    
    return analysis_results


def main():
    """CLI interface for count regression analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Poisson/Negative Binomial count regression for healthcare encounters"
    )
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with simulated healthcare data')
    
    args = parser.parse_args()
    
    if args.demo:
        print("Generating healthcare encounter count data...")
        
        np.random.seed(42)
        n = 1000
        
        # Simulate SSD analysis data
        ssd_exposure = np.random.binomial(1, 0.15, n)
        age = np.random.normal(50, 15, n)
        baseline_encounters = np.random.poisson(3, n)
        
        # Linear predictor: log(expected encounters)
        linear_pred = (np.log(baseline_encounters + 0.1) + 
                      0.3 * ssd_exposure +  # SSD effect (IRR ≈ 1.35)
                      0.01 * (age - 50))    # Age effect
        
        total_encounters = np.random.poisson(np.exp(linear_pred))
        
        # Create DataFrame
        df = pd.DataFrame({
            'ssd_flag': ssd_exposure,
            'total_encounters': total_encounters,
            'age': age,
            'baseline_encounters': baseline_encounters,
            'site_id': np.random.randint(1, 21, n)
        })
        
        print(f"Generated {n} patients with {df['ssd_flag'].sum()} SSD exposed")
        print(f"Mean encounters: SSD={df[df['ssd_flag']==1]['total_encounters'].mean():.1f}, "
              f"Control={df[df['ssd_flag']==0]['total_encounters'].mean():.1f}")
        
        # Run analysis
        print("\nRunning count regression analysis...")
        results = count_regression_analysis(
            df=df,
            outcome_col='total_encounters',
            treatment_col='ssd_flag',
            covariate_cols=['age', 'baseline_encounters'],
            cluster_col='site_id'
        )
        
        print(f"\nSelected model: {results['selected_model']}")
        print(f"Treatment IRR: {results['treatment_effect']['irr']:.3f} "
              f"({results['treatment_effect']['irr_ci_lower']:.3f}, "
              f"{results['treatment_effect']['irr_ci_upper']:.3f})")
        
        print("\n✓ Count regression demonstration completed successfully")


if __name__ == "__main__":
    main()