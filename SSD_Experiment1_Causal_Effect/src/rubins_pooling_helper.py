#!/usr/bin/env python3
"""
rubins_pooling_helper.py - Helper functions for Rubin's Rules implementation

Extracted from rubins_pooling_engine.py to comply with CLAUDE.md 50-line limit.

Author: Ryhan Suny
Date: June 30, 2025
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def compute_pooled_estimate(estimates: np.ndarray) -> float:
    """
    Compute pooled estimate Q̄ = m⁻¹∑Qₖ.
    
    Parameters:
    -----------
    estimates : np.ndarray
        Point estimates from m imputations
        
    Returns:
    --------
    float
        Pooled estimate
    """
    return np.mean(estimates)


def compute_variance_components(estimates: np.ndarray, 
                               variances: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute within, between, and total variance components.
    
    Following Rubin (1987):
    - Within-variance: Ū = m⁻¹∑Uₖ
    - Between-variance: B = (m-1)⁻¹∑(Qₖ-Q̄)²
    - Total variance: T = Ū + (1+m⁻¹)B
    
    Parameters:
    -----------
    estimates : np.ndarray
        Point estimates Qₖ from m imputations
    variances : np.ndarray
        Within-imputation variances Uₖ
        
    Returns:
    --------
    Tuple[float, float, float]
        (U_bar, B, T) - within, between, and total variance
    """
    m = len(estimates)
    Q_bar = np.mean(estimates)
    
    # Within-imputation variance (average of variances)
    U_bar = np.mean(variances)
    
    # Between-imputation variance
    B = np.sum((estimates - Q_bar) ** 2) / (m - 1) if m > 1 else 0.0
    
    # Total variance (Rubin's formula)
    T = U_bar + (1 + 1/m) * B
    
    return U_bar, B, T


def compute_missing_information_metrics(B: float, U_bar: float, 
                                       m: int) -> Tuple[float, float]:
    """
    Compute relative increase in variance and fraction of missing information.
    
    Parameters:
    -----------
    B : float
        Between-imputation variance
    U_bar : float
        Within-imputation variance
    m : int
        Number of imputations
        
    Returns:
    --------
    Tuple[float, float]
        (r, lambda_hat) - RIV and FMI
    """
    # Relative increase in variance due to missing data
    if U_bar > 0:
        r = (1 + 1/m) * B / U_bar
    else:
        r = np.inf if B > 0 else 0.0
    
    # Fraction of missing information (adjusted for finite samples)
    lambda_hat = (r + 2/(m + 1)) / (1 + r) if r != np.inf else 1.0
    
    return r, lambda_hat


def log_pooling_diagnostics(method: str, outcome: str, Q_bar: float,
                           se_pooled: float, ci_lower: float, ci_upper: float,
                           lambda_hat: float, r: float, 
                           validation: dict) -> None:
    """
    Log diagnostic information for pooled results.
    
    Parameters:
    -----------
    method : str
        Statistical method name
    outcome : str
        Outcome variable name
    Q_bar : float
        Pooled estimate
    se_pooled : float
        Pooled standard error
    ci_lower : float
        Lower CI bound
    ci_upper : float
        Upper CI bound
    lambda_hat : float
        Fraction of missing information
    r : float
        Relative increase in variance
    validation : dict
        Validation results
    """
    logger.info(f"Pooled {method} for {outcome}:")
    logger.info(f"  Estimate: {Q_bar:.4f} ± {se_pooled:.4f}")
    logger.info(f"  95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")
    logger.info(f"  FMI: {lambda_hat:.3f}, RIV: {r:.3f}")
    logger.info(f"  Validation: {validation}")


def compute_confidence_intervals(Q_bar: float, se_pooled: float, 
                               df: float, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute confidence intervals using t-distribution.
    
    Parameters:
    -----------
    Q_bar : float
        Pooled estimate
    se_pooled : float
        Pooled standard error
    df : float
        Degrees of freedom
    alpha : float
        Significance level
        
    Returns:
    --------
    Tuple[float, float]
        (ci_lower, ci_upper)
    """
    from scipy import stats
    
    if df > 0:
        t_critical = stats.t.ppf(1 - alpha/2, df)
    else:
        t_critical = stats.norm.ppf(1 - alpha/2)
        logger.warning("Using normal approximation due to invalid df")
    
    ci_lower = Q_bar - t_critical * se_pooled
    ci_upper = Q_bar + t_critical * se_pooled
    
    return ci_lower, ci_upper


def create_pooled_result(Q_bar: float, se_pooled: float, ci_lower: float, 
                        ci_upper: float, U_bar: float, B: float, T: float,
                        nu: float, lambda_hat: float, r: float, m: int,
                        method: str, outcome: str, alpha: float) -> dict:
    """
    Create pooled result dictionary from computed values.
    
    Returns dict instead of RubinsPooledResult to avoid circular import.
    """
    from datetime import datetime
    
    return {
        'estimate': float(Q_bar),
        'standard_error': float(se_pooled),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'within_variance': float(U_bar),
        'between_variance': float(B),
        'total_variance': float(T),
        'degrees_freedom': float(nu),
        'fmi': float(lambda_hat),
        'riv': float(r),
        'n_imputations': m,
        'method': method,
        'outcome': outcome,
        'alpha': alpha,
        'timestamp': datetime.now().isoformat()
    }