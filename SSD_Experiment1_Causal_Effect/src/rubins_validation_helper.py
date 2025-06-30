#!/usr/bin/env python3
"""
rubins_validation_helper.py - Validation helpers for Rubin's Rules

Extracted from rubins_pooling_engine.py to comply with CLAUDE.md 50-line limit.

Author: Ryhan Suny
Date: June 30, 2025
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def check_sufficient_imputations(n_imputations: int) -> bool:
    """Check if number of imputations is sufficient."""
    sufficient = n_imputations >= 5
    if not sufficient:
        logger.warning(f"Only {n_imputations} imputations available. "
                      f"Recommend m ≥ 5 for reliable inference.")
    return sufficient


def check_complete_data(estimates: np.ndarray, ses: np.ndarray) -> Dict[str, bool]:
    """Check for missing values in estimates and standard errors."""
    return {
        'complete_estimates': not np.any(np.isnan(estimates)),
        'complete_ses': not np.any(np.isnan(ses))
    }


def check_positive_ses(ses: np.ndarray) -> bool:
    """Check that all standard errors are positive."""
    positive = np.all(ses > 0)
    if not positive:
        logger.error("Negative or zero standard errors detected")
    return positive


def check_reasonable_variation(estimates: np.ndarray) -> bool:
    """Check for reasonable variation between imputations."""
    if np.mean(estimates) == 0:
        return True
    
    cv_estimates = np.std(estimates, ddof=1) / np.abs(np.mean(estimates))
    reasonable = cv_estimates < 0.5
    
    if not reasonable:
        logger.warning(f"High between-imputation variation (CV={cv_estimates:.3f}). "
                      f"Check imputation model specification.")
    return reasonable


def check_outliers(estimates: np.ndarray) -> bool:
    """Check for extreme outliers using Tukey's rule."""
    if len(estimates) < 3:
        return True
    
    q75, q25 = np.percentile(estimates, [75, 25])
    iqr = q75 - q25
    median = np.median(estimates)
    outlier_bounds = (median - 3*iqr, median + 3*iqr)
    
    return np.all((estimates >= outlier_bounds[0]) & 
                  (estimates <= outlier_bounds[1]))


def validate_critical_requirements(validation: Dict[str, bool]) -> bool:
    """Check if critical validation requirements are met."""
    critical_failures = [
        not validation['complete_estimates'],
        not validation['complete_ses'],
        not validation['positive_ses']
    ]
    return not any(critical_failures)