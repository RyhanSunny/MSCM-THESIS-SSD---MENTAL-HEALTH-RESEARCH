#!/usr/bin/env python3
"""
Weight Diagnostics Guard-Rails Module

Implements automated weight diagnostics checks for propensity score analysis.
Follows Austin (2011) recommendations for IPTW weight validation.

Key Features:
- Effective Sample Size (ESS) calculation using Kish formula
- Extreme weight detection (>10x median)
- Automated CI integration with JSON output
- Detailed logging for debugging

Author: Ryhan Suny
Date: 2025-06-17
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightDiagnosticsError(Exception):
    """Custom exception for weight diagnostics failures"""
    pass


def calculate_effective_sample_size(weights: np.ndarray) -> float:
    """
    Calculate Effective Sample Size using Kish formula.
    
    ESS = (Σw)² / Σw²
    
    Args:
        weights: Array of propensity score weights
        
    Returns:
        Effective sample size
        
    References:
        Kish, L. (1965). Survey Sampling. Wiley.
        Austin, P.C. (2011). Multivariate Behavioral Research, 46(3), 399-424.
    """
    if len(weights) == 0:
        raise ValueError("Weights array cannot be empty")
    
    if np.any(weights <= 0):
        raise ValueError("All weights must be positive")
    
    sum_weights = np.sum(weights)
    sum_weights_squared = np.sum(weights ** 2)
    
    if sum_weights_squared == 0:
        raise ValueError("Sum of squared weights cannot be zero")
    
    ess = (sum_weights ** 2) / sum_weights_squared
    
    logger.debug(f"ESS calculation: (Σw)²={(sum_weights**2):.2f}, Σw²={sum_weights_squared:.2f}, ESS={ess:.2f}")
    
    return ess


def check_weight_extremes(weights: np.ndarray, max_ratio: float = 10.0) -> Dict[str, Any]:
    """
    Check for extreme weights relative to median.
    
    Args:
        weights: Array of weights
        max_ratio: Maximum acceptable ratio relative to median
        
    Returns:
        Dictionary with extreme weight diagnostics
    """
    median_weight = np.median(weights)
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    
    max_weight_ratio = max_weight / median_weight if median_weight > 0 else np.inf
    min_weight_ratio = min_weight / median_weight if median_weight > 0 else 0
    
    # Count extreme weights
    n_extreme_high = np.sum(weights > median_weight * max_ratio)
    n_extreme_low = np.sum(weights < median_weight / max_ratio)
    n_extreme_total = n_extreme_high + n_extreme_low
    
    has_extreme_weights = max_weight_ratio > max_ratio
    
    return {
        'median_weight': float(median_weight),
        'max_weight': float(max_weight),
        'min_weight': float(min_weight),
        'max_weight_ratio': float(max_weight_ratio),
        'min_weight_ratio': float(min_weight_ratio),
        'n_extreme_high': int(n_extreme_high),
        'n_extreme_low': int(n_extreme_low),
        'n_extreme_weights': int(n_extreme_total),
        'has_extreme_weights': bool(has_extreme_weights),
        'extreme_weight_pct': float(n_extreme_total / len(weights) * 100)
    }


def validate_weight_diagnostics(
    weights: Union[np.ndarray, pd.Series],
    min_ess_ratio: float = 0.5,
    max_weight_ratio: float = 10.0,
    raise_on_failure: bool = True
) -> Dict[str, Any]:
    """
    Validate weight diagnostics against established thresholds.
    
    Implements Austin (2011) recommendations:
    - ESS should be > 50% of sample size
    - No weight should be > 10x median weight
    
    Args:
        weights: Propensity score weights
        min_ess_ratio: Minimum ESS as fraction of sample size
        max_weight_ratio: Maximum weight as multiple of median
        raise_on_failure: Whether to raise exception on validation failure
        
    Returns:
        Dictionary with validation results
        
    Raises:
        WeightDiagnosticsError: If validation fails and raise_on_failure=True
    """
    if isinstance(weights, pd.Series):
        weights = weights.values
    
    weights = np.asarray(weights)
    n = len(weights)
    
    if n == 0:
        raise ValueError("Cannot validate empty weights array")
    
    # Calculate diagnostics
    ess = calculate_effective_sample_size(weights)
    ess_ratio = ess / n
    
    extremes = check_weight_extremes(weights, max_weight_ratio)
    
    # Validation checks
    ess_check_passed = ess_ratio >= min_ess_ratio
    weight_check_passed = not extremes['has_extreme_weights']
    validation_passed = ess_check_passed and weight_check_passed
    
    # Prepare results
    results = {
        'n_observations': int(n),
        'ess': float(ess),
        'ess_ratio': float(ess_ratio),
        'min_ess_ratio_threshold': float(min_ess_ratio),
        'ess_check_passed': bool(ess_check_passed),
        'weight_check_passed': bool(weight_check_passed),
        'validation_passed': bool(validation_passed),
        'timestamp': datetime.now().isoformat(),
        **extremes
    }
    
    # Log results
    logger.info(f"Weight Diagnostics Summary:")
    logger.info(f"  Sample size: {n:,}")
    logger.info(f"  ESS: {ess:.1f} ({ess_ratio:.1%} of sample)")
    logger.info(f"  Max weight ratio: {extremes['max_weight_ratio']:.1f}x median")
    logger.info(f"  Extreme weights: {extremes['n_extreme_weights']} ({extremes['extreme_weight_pct']:.1f}%)")
    logger.info(f"  Validation: {'PASSED' if validation_passed else 'FAILED'}")
    
    # Raise exception if validation failed
    if not validation_passed and raise_on_failure:
        error_msg = []
        
        if not ess_check_passed:
            error_msg.append(f"Effective sample size too low: {ess:.1f} ({ess_ratio:.1%}) < {min_ess_ratio:.1%}")
        
        if not weight_check_passed:
            error_msg.append(f"Maximum weight too high: {extremes['max_weight_ratio']:.1f}x median > {max_weight_ratio}x")
        
        full_error = "Weight diagnostics validation failed:\n" + "\n".join(error_msg)
        raise WeightDiagnosticsError(full_error)
    
    return results


def save_weight_diagnostics(
    results: Dict[str, Any],
    output_path: Union[str, Path] = "results/weight_diagnostics.json"
) -> None:
    """
    Save weight diagnostics results to JSON file for CI integration.
    
    Args:
        results: Validation results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Weight diagnostics saved to {output_path}")


def load_weight_diagnostics(
    input_path: Union[str, Path] = "results/weight_diagnostics.json"
) -> Dict[str, Any]:
    """
    Load weight diagnostics results from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        Validation results dictionary
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Weight diagnostics file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    return results


def check_weight_diagnostics_ci() -> int:
    """
    CI integration function - returns exit code based on validation status.
    
    Returns:
        0 if validation passed, 1 if failed
    """
    try:
        results = load_weight_diagnostics()
        
        if results['validation_passed']:
            print("✓ Weight diagnostics validation PASSED")
            return 0
        else:
            print("✗ Weight diagnostics validation FAILED")
            print(f"  ESS: {results['ess']:.1f} ({results['ess_ratio']:.1%})")
            print(f"  Max weight ratio: {results['max_weight_ratio']:.1f}x")
            print(f"  Extreme weights: {results['n_extreme_weights']}")
            return 1
            
    except Exception as e:
        print(f"✗ Weight diagnostics check failed: {e}")
        return 1


def main():
    """CLI interface for weight diagnostics checking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check weight diagnostics validation")
    parser.add_argument('--matched-data', '-m',
                       help='Path to matched cohort data with weights')
    parser.add_argument('--output', '-o',
                       default="results/weight_diagnostics.json",
                       help='Output path for diagnostics JSON')
    parser.add_argument('--input', '-i', 
                       help='Input diagnostics file (for validation mode)')
    parser.add_argument('--ci', action='store_true',
                       help='CI mode - return exit code')
    
    args = parser.parse_args()
    
    if args.ci:
        exit_code = check_weight_diagnostics_ci()
        exit(exit_code)
    elif args.matched_data:
        # Compute diagnostics from matched data
        try:
            logger.info(f"Loading matched data from {args.matched_data}")
            df = pd.read_parquet(args.matched_data)
            
            # Check for weight columns
            weight_cols = [col for col in df.columns if 'weight' in col.lower() or col == 'ps_weight']
            if not weight_cols:
                raise ValueError("No weight columns found in matched data")
            
            weight_col = weight_cols[0]
            logger.info(f"Using weight column: {weight_col}")
            
            weights = df[weight_col].values
            
            # Calculate diagnostics
            diagnostics = validate_weight_diagnostics(weights)
            
            # Add metadata
            diagnostics['metadata'] = {
                'source_file': str(args.matched_data),
                'weight_column': weight_col,
                'n_observations': len(df),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results
            save_weight_diagnostics(diagnostics, args.output)
            
            # Print summary
            print(f"\nWeight Diagnostics Summary:")
            print(f"  ESS: {diagnostics['ess']:.2f} ({diagnostics['ess_ratio']*100:.1f}% of n={len(weights)})")
            print(f"  Max weight: {diagnostics['max_weight']:.4f}")
            print(f"  Extreme weights: {diagnostics['n_extreme_weights']} ({diagnostics['extreme_weight_pct']:.1f}%)")
            print(f"  Validation passed: {diagnostics['validation_passed']}")
            
        except Exception as e:
            logger.error(f"Error computing diagnostics: {e}")
            raise
    else:
        # Default validation mode
        input_file = args.input or "results/weight_diagnostics.json"
        try:
            results = load_weight_diagnostics(input_file)
            print(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Error: {e}")
            exit(1)



if __name__ == "__main__":
    main()