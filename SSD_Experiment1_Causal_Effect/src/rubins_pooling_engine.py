#!/usr/bin/env python3
"""
rubins_pooling_engine.py - Surgical Implementation of Rubin's Rules for Multiple Imputation

STATISTICAL FOUNDATION:
Rubin's Rules are the MANDATORY bookkeeping formulas for valid MI inference 
(Rubin 1987, Little & Rubin 2019). Without proper pooling, standard errors 
are under-estimated and Type-I error is inflated.

CORE FORMULAE (Following van Buuren notation):
1. Pooled estimate: Q̄ = m⁻¹∑Qₖ  
2. Within variance: Ū = m⁻¹∑Uₖ
3. Between variance: B = (m-1)⁻¹∑(Qₖ-Q̄)²
4. Total variance: T = Ū + B + B/m
5. Degrees of freedom: ν (Barnard-Rubin adjustment)

ARCHITECTURE INTEGRATION:
- Designed for existing TMLE/Double ML/Causal Forest estimators
- Maintains pipeline compatibility with 06_causal_estimators.py
- Supports multiple outcomes and methods simultaneously

Following CLAUDE.md requirements:
- TDD with comprehensive validation
- Functions ≤50 lines
- Complete type annotations and docstrings
- No overconfident assumptions

Author: Ryhan Suny (Toronto Metropolitan University)
Supervisor: Dr. Aziz Guergachi
Research Team: Car4Mind, University of Toronto
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from scipy import stats
import warnings

# Import helper functions
from rubins_pooling_helper import (
    compute_pooled_estimate,
    compute_variance_components,
    compute_missing_information_metrics,
    log_pooling_diagnostics,
    compute_confidence_intervals,
    create_pooled_result
)
from rubins_validation_helper import (
    check_sufficient_imputations,
    check_complete_data,
    check_positive_ses,
    check_reasonable_variation,
    check_outliers,
    validate_critical_requirements
)

# Configure logging following CLAUDE.md standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RubinsPooledResult:
    """
    Container for Rubin's Rules pooled multiple imputation results.
    
    Implements complete variance decomposition following Rubin (1987) and 
    Barnard & Rubin (1999) small-sample adjustment.
    
    Attributes:
    -----------
    estimate : float
        Pooled point estimate Q̄ = m⁻¹∑Qₖ
    standard_error : float
        Pooled standard error √T
    ci_lower : float
        Lower confidence interval bound
    ci_upper : float  
        Upper confidence interval bound
    within_variance : float
        Within-imputation variance Ū = m⁻¹∑Uₖ
    between_variance : float
        Between-imputation variance B = (m-1)⁻¹∑(Qₖ-Q̄)²
    total_variance : float
        Total variance T = Ū + B + B/m
    degrees_freedom : float
        Barnard-Rubin adjusted degrees of freedom
    fmi : float
        Fraction of Missing Information λ̂
    riv : float
        Relative Increase in Variance r
    n_imputations : int
        Number of imputations m
    alpha : float
        Significance level for confidence intervals
    method : str
        Estimation method (TMLE, Double ML, etc.)
    outcome : str
        Outcome variable name
    timestamp : str
        ISO timestamp of pooling
    """
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    within_variance: float
    between_variance: float
    total_variance: float
    degrees_freedom: float
    fmi: float  # Fraction of Missing Information
    riv: float  # Relative Increase in Variance
    n_imputations: int
    method: str
    outcome: str
    alpha: float = 0.05
    timestamp: str = ""


def validate_imputation_inputs(estimates: List[float], 
                             standard_errors: List[float]) -> Dict[str, bool]:
    """
    Validate inputs for Rubin's Rules following statistical best practices.
    
    CRITICAL ASSUMPTIONS (White, Royston & Wood 2011):
    1. Sufficient imputations (m ≥ 5, preferably m ≥ % missing)
    2. No missing estimates or standard errors  
    3. Positive standard errors
    4. Reasonable between-imputation variation (CV < 50%)
    5. No extreme outliers (>3 SD from mean)
    
    Parameters:
    -----------
    estimates : List[float]
        Point estimates from m imputed datasets
    standard_errors : List[float]
        Standard errors from m imputed datasets
        
    Returns:
    --------
    Dict[str, bool]
        Validation results for each assumption
        
    Raises:
    -------
    ValueError
        If basic input requirements not met
    """
    if len(estimates) != len(standard_errors):
        raise ValueError("Estimates and standard errors must have equal length")
    
    if len(estimates) < 2:
        raise ValueError("Minimum 2 imputations required for Rubin's Rules")
    
    estimates = np.array(estimates, dtype=float)
    ses = np.array(standard_errors, dtype=float)
    
    # Use helper functions for validation
    validation_results = {}
    validation_results['sufficient_imputations'] = check_sufficient_imputations(len(estimates))
    validation_results.update(check_complete_data(estimates, ses))
    validation_results['positive_ses'] = check_positive_ses(ses)
    validation_results['reasonable_variation'] = check_reasonable_variation(estimates)
    validation_results['no_extreme_outliers'] = check_outliers(estimates)
    
    return validation_results


def calculate_barnard_rubin_df(m: int, B: float, U_bar: float, 
                              n_obs: Optional[int] = None) -> float:
    """
    Calculate Barnard-Rubin adjusted degrees of freedom for small samples.
    
    Reference: Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of 
    freedom with multiple imputation. Biometrika, 86(4), 948-955.
    
    Parameters:
    -----------
    m : int
        Number of imputations
    B : float
        Between-imputation variance
    U_bar : float
        Average within-imputation variance
    n_obs : Optional[int]
        Number of observations (for finite sample adjustment)
        
    Returns:
    --------
    float
        Barnard-Rubin adjusted degrees of freedom
    """
    # Handle edge case where B = 0 (no between-imputation variance)
    if B == 0:
        # When there's no between-variance, df approaches infinity
        # But for finite samples, it's bounded by n-1
        if n_obs is None:
            return float('inf')
        else:
            # Return large but finite df
            return float(n_obs - 1)
    
    # Calculate relative increase in variance
    r = (1 + 1/m) * B / U_bar if U_bar > 0 else float('inf')
    
    # Calculate fraction of missing information
    lambda_hat = (r + 2/(m + 1)) / (1 + r) if r != float('inf') else 1.0
    
    # Old formula (Rubin 1987) - infinite population
    nu_old = (m - 1) * (1 + U_bar / ((1 + 1/m) * B)) ** 2
    
    # If no sample size provided, return old formula
    if n_obs is None:
        return nu_old
    
    # Observed data degrees of freedom (Barnard-Rubin adjustment)
    # This accounts for finite sample size
    nu_obs = 4 + (m - 4) * (1 + (1 - 2/m) * lambda_hat) ** 2
    
    # Barnard-Rubin combined formula
    nu_BR = 1 / (1/nu_old + 1/nu_obs)
    
    return nu_BR


def _pool_estimates_implementation(estimates: List[float], 
                                 standard_errors: List[float],
                                 method: str = "Unknown",
                                 outcome: str = "Unknown", 
                                 alpha: float = 0.05,
                                 n_obs: Optional[int] = None) -> RubinsPooledResult:
    """Core implementation of Rubin's Rules pooling (under 50 lines)."""
    # Validate inputs
    validation = validate_imputation_inputs(estimates, standard_errors)
    if not validate_critical_requirements(validation):
        raise ValueError(f"Critical validation failures: {validation}")
    
    # Convert and compute
    Q = np.array(estimates, dtype=float)
    U = np.array(standard_errors, dtype=float) ** 2
    m = len(Q)
    
    Q_bar = compute_pooled_estimate(Q)
    U_bar, B, T = compute_variance_components(Q, U)
    r, lambda_hat = compute_missing_information_metrics(B, U_bar, m)
    nu = calculate_barnard_rubin_df(m, B, U_bar, n_obs)
    
    se_pooled = np.sqrt(T)
    ci_lower, ci_upper = compute_confidence_intervals(Q_bar, se_pooled, nu, alpha)
    
    # Log diagnostics
    log_pooling_diagnostics(method, outcome, Q_bar, se_pooled, 
                           ci_lower, ci_upper, lambda_hat, r, validation)
    
    # Build result
    result_dict = create_pooled_result(
        Q_bar, se_pooled, ci_lower, ci_upper, U_bar, B, T,
        nu, lambda_hat, r, m, method, outcome, alpha
    )
    
    return RubinsPooledResult(**result_dict)


def pool_estimates_rubins_rules(estimates: List[float], 
                              standard_errors: List[float],
                              method: str = "Unknown",
                              outcome: str = "Unknown", 
                              alpha: float = 0.05,
                              n_obs: Optional[int] = None) -> RubinsPooledResult:
    """
    Pool estimates using Rubin's Rules with Barnard-Rubin small-sample adjustment.
    
    MATHEMATICAL IMPLEMENTATION:
    Following Rubin (1987) Chapter 3 and Barnard & Rubin (1999):
    
    1. Q̄ = m⁻¹∑Qₖ                    [Pooled estimate]
    2. Ū = m⁻¹∑Uₖ                    [Within-imputation variance]  
    3. B = (m-1)⁻¹∑(Qₖ-Q̄)²          [Between-imputation variance]
    4. T = Ū + (1+m⁻¹)B              [Total variance]
    5. r = (1+m⁻¹)B/Ū                [Relative increase in variance]
    6. λ̂ = (r+2/(m+1))/(1+r)         [Fraction of missing information]
    7. ν = (m-1)(1+Ū/((1+m⁻¹)B))²   [Barnard-Rubin degrees of freedom]
    
    Parameters:
    -----------
    estimates : List[float]
        Point estimates Qₖ from m imputed analyses
    standard_errors : List[float]  
        Standard errors from m imputed analyses
    method : str, default "Unknown"
        Statistical method used (TMLE, Double ML, etc.)
    outcome : str, default "Unknown"
        Outcome variable name
    alpha : float, default 0.05
        Significance level for confidence intervals
    n_obs : Optional[int], default None
        Number of observations for finite sample adjustment
        
    Returns:
    --------
    RubinsPooledResult
        Complete pooled results with variance decomposition
        
    Example:
    --------
    >>> estimates = [1.62, 1.58, 1.65, 1.61, 1.59]  # ATE from 5 imputations
    >>> ses = [0.035, 0.033, 0.037, 0.034, 0.036]    # Standard errors
    >>> result = pool_estimates_rubins_rules(estimates, ses, "TMLE", "total_encounters")
    >>> print(f"Pooled ATE: {result.estimate:.3f} (95% CI: {result.ci_lower:.3f}, {result.ci_upper:.3f})")
    """
    return _pool_estimates_implementation(
        estimates, standard_errors, method, outcome, alpha, n_obs
    )


def load_imputed_causal_estimates(results_dir: Union[str, Path],
                                pattern: str = "causal_estimates_imp*.json") -> Dict[str, Dict[str, List]]:
    """
    Load causal estimates from multiple imputed analyses.
    
    EXPECTED FILE FORMAT:
    {
        "outcome_name": {
            "method_name": {
                "estimate": float,
                "se": float,
                "ci_lower": float,
                "ci_upper": float
            }
        }
    }
    
    Parameters:
    -----------
    results_dir : Union[str, Path]
        Directory containing imputed results files
    pattern : str
        Glob pattern for imputed result files
        
    Returns:
    --------
    Dict[str, Dict[str, List]]
        Nested dict: {outcome: {method: [estimates]}, outcome: {method: [ses]}}
    """
    results_dir = Path(results_dir)
    estimate_data = {}
    se_data = {}
    
    # Find all matching result files
    result_files = sorted(results_dir.glob(pattern))
    
    if not result_files:
        logger.warning(f"No imputed result files found matching {pattern}")
        return {"estimates": estimate_data, "standard_errors": se_data}
    
    logger.info(f"Loading {len(result_files)} imputed result files")
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Parse nested structure: outcome -> method -> results
            for outcome, methods in data.items():
                if outcome not in estimate_data:
                    estimate_data[outcome] = {}
                    se_data[outcome] = {}
                
                if isinstance(methods, dict):
                    for method, results in methods.items():
                        if method not in estimate_data[outcome]:
                            estimate_data[outcome][method] = []
                            se_data[outcome][method] = []
                        
                        # Extract estimate and SE
                        estimate_data[outcome][method].append(results.get('estimate', np.nan))
                        se_data[outcome][method].append(results.get('se', np.nan))
                        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    return {"estimates": estimate_data, "standard_errors": se_data}


def pool_multiple_imputation_results(results_dir: Union[str, Path],
                                   output_file: Union[str, Path],
                                   alpha: float = 0.05) -> Dict[str, RubinsPooledResult]:
    """
    Pool all causal estimates using Rubin's Rules.
    
    INTEGRATION POINT: 
    This is the main function called by the pipeline to pool results
    from 06_causal_estimators.py across all imputed datasets.
    
    Parameters:
    -----------
    results_dir : Union[str, Path]
        Directory containing per-imputation results
    output_file : Union[str, Path] 
        Output file for pooled results
    alpha : float, default 0.05
        Significance level
        
    Returns:
    --------
    Dict[str, RubinsPooledResult]
        Pooled results by outcome-method combination
    """
    # Load imputed results
    data = load_imputed_causal_estimates(results_dir)
    estimates = data["estimates"]
    standard_errors = data["standard_errors"]
    
    if not estimates:
        raise ValueError("No imputed estimates found for pooling")
    
    pooled_results = {}
    
    # Pool each outcome-method combination
    for outcome in estimates:
        for method in estimates[outcome]:
            key = f"{outcome}_{method}"
            
            est_list = estimates[outcome][method]
            se_list = standard_errors[outcome][method]
            
            # Skip if insufficient data
            if len(est_list) < 2:
                logger.warning(f"Insufficient data for {key}, skipping")
                continue
            
            # Pool using Rubin's Rules
            pooled_results[key] = pool_estimates_rubins_rules(
                est_list, se_list, method, outcome, alpha
            )
    
    # Save results
    save_pooled_results(pooled_results, output_file)
    
    return pooled_results


def save_pooled_results(pooled_results: Dict[str, RubinsPooledResult],
                       output_file: Union[str, Path]) -> None:
    """
    Save pooled results to JSON file following pipeline conventions.
    
    Parameters:
    -----------
    pooled_results : Dict[str, RubinsPooledResult]
        Pooled results to save
    output_file : Union[str, Path]
        Output file path
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)
    
    # Convert to serializable format
    output_data = {}
    for key, result in pooled_results.items():
        output_data[key] = asdict(result)
    
    # Add metadata
    output_data["_metadata"] = {
        "pooling_method": "Rubin's Rules (1987) with Barnard-Rubin adjustment",
        "total_outcomes_pooled": len(pooled_results),
        "timestamp": datetime.now().isoformat(),
        "references": [
            "Rubin DB (1987). Multiple Imputation for Nonresponse in Surveys",
            "Barnard J, Rubin DB (1999). Small-sample degrees of freedom",
            "Little RJA, Rubin DB (2019). Statistical Analysis with Missing Data"
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Pooled results saved to {output_file}")


def main():
    """
    Command-line interface for Rubin's Rules pooling.
    
    Following CLAUDE.md requirement for functions ≤50 lines.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pool causal estimates using Rubin's Rules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--results-dir', default='results', 
                       help='Directory containing per-imputation results')
    parser.add_argument('--output-file', default='results/pooled_causal_estimates.json',
                       help='Output file for pooled results')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level for confidence intervals')
    parser.add_argument('--pattern', default='causal_estimates_imp*.json',
                       help='File pattern for imputed results')
    
    args = parser.parse_args()
    
    try:
        pooled_results = pool_multiple_imputation_results(
            args.results_dir, args.output_file, args.alpha
        )
        
        logger.info(f"SUCCESS: Pooled {len(pooled_results)} outcome-method combinations")
        
        # Summary statistics
        for key, result in pooled_results.items():
            print(f"\n{key}:")
            print(f"  Estimate: {result.estimate:.4f} ± {result.standard_error:.4f}")
            print(f"  95% CI: ({result.ci_lower:.4f}, {result.ci_upper:.4f})")
            print(f"  FMI: {result.fmi:.3f}")
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        raise


if __name__ == "__main__":
    main()