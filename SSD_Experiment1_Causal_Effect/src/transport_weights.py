#!/usr/bin/env python3
"""
transport_weights.py - Transportability weights for external validity

Calculates transportability weights for generalizing findings across populations
using ICES marginal distributions when available.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_transport_weights(study_data: pd.DataFrame,
                               target_marginals_path: Optional[Path] = None,
                               variables: Optional[list] = None) -> Dict[str, Any]:
    """
    Calculate transportability weights for external validity
    
    Parameters:
    -----------
    study_data : pd.DataFrame
        Study dataset with covariates
    target_marginals_path : Optional[Path]
        Path to ICES marginals CSV file  
    variables : Optional[list]
        Variables to use for transport weighting
        
    Returns:
    --------
    Dict[str, Any]
        Transport weights and diagnostics
    """
    logger.info("Computing transportability weights...")
    
    # Default marginals file path
    if target_marginals_path is None:
        target_marginals_path = Path("data/external/ices_marginals.csv")
    
    # Check if marginals file exists
    if not target_marginals_path.exists():
        logger.warning(f"ICES marginals file not found: {target_marginals_path}")
        logger.info("Returning skipped status for CI compatibility")
        
        # Return placeholder results for CI compatibility
        n = len(study_data)
        uniform_weights = np.ones(n)
        
        return {
            'status': 'skipped',
            'reason': 'ICES marginals file not available',
            'weights': uniform_weights,
            'effective_sample_size': n,
            'max_weight': 1.0,
            'mean_weight': 1.0,
            'file_path': str(target_marginals_path),
            'n_observations': n
        }
    
    # Load target population marginals
    try:
        target_marginals = pd.read_csv(target_marginals_path)
        logger.info(f"Loaded target marginals: {target_marginals.shape}")
    except Exception as e:
        logger.error(f"Failed to load marginals file: {e}")
        return _create_placeholder_results(study_data, "failed_to_load")
    
    # Default variables for transport weighting
    if variables is None:
        variables = ['age_group', 'sex', 'region', 'socioeconomic_quintile']
    
    # Check which variables are available
    available_vars = [var for var in variables if var in study_data.columns]
    if not available_vars:
        logger.warning("No transport variables available in study data")
        return _create_placeholder_results(study_data, "no_variables")
    
    logger.info(f"Using variables for transport weighting: {available_vars}")
    
    # Calculate study sample marginals
    study_marginals = {}
    for var in available_vars:
        study_prop = study_data[var].value_counts(normalize=True).sort_index()
        study_marginals[var] = study_prop
    
    # Calculate transport weights
    weights = np.ones(len(study_data))
    
    for var in available_vars:
        # Get target proportions for this variable from long format data
        var_target = target_marginals[target_marginals['variable'] == var]
        
        if len(var_target) > 0:
            # Create target proportions dictionary
            target_prop = dict(zip(var_target['category'], var_target['proportion']))
            study_prop = study_marginals[var]
            
            # Calculate weight for each observation
            for category in study_prop.index:
                if category in target_prop:
                    weight_ratio = target_prop[category] / study_prop[category]
                    mask = study_data[var] == category
                    weights[mask] *= weight_ratio
                    logger.debug(f"{var}={category}: target_prop={target_prop[category]:.3f}, study_prop={study_prop[category]:.3f}, ratio={weight_ratio:.3f}")
        else:
            logger.warning(f"Variable {var} not found in target marginals")
            
    # Calculate diagnostics
    ess = calculate_effective_sample_size(weights)
    max_weight = np.max(weights)
    mean_weight = np.mean(weights)
    
    results = {
        'status': 'completed',
        'weights': weights,
        'effective_sample_size': ess,
        'max_weight': max_weight,
        'mean_weight': mean_weight,
        'variables_used': available_vars,
        'n_observations': len(study_data),
        'target_marginals_file': str(target_marginals_path)
    }
    
    logger.info(f"Transport weights computed. ESS: {ess:.1f}, Max weight: {max_weight:.3f}")
    return results


def calculate_effective_sample_size(weights: np.ndarray) -> float:
    """
    Calculate effective sample size using Kish formula
    
    Parameters:
    -----------
    weights : np.ndarray
        Transport weights
        
    Returns:
    --------
    float
        Effective sample size
    """
    return (np.sum(weights)**2) / np.sum(weights**2)


def _create_placeholder_results(study_data: pd.DataFrame, reason: str) -> Dict[str, Any]:
    """Create placeholder results when transport weighting cannot be performed"""
    n = len(study_data)
    uniform_weights = np.ones(n)
    
    return {
        'status': 'skipped',
        'reason': reason,
        'weights': uniform_weights,
        'effective_sample_size': n,
        'max_weight': 1.0,
        'mean_weight': 1.0,
        'n_observations': n
    }


def validate_transport_weights(weights: np.ndarray, 
                              max_weight_threshold: float = 20.0,
                              min_ess_ratio: float = 0.1) -> Dict[str, Any]:
    """
    Validate transport weight quality
    
    Parameters:
    -----------
    weights : np.ndarray
        Transport weights to validate
    max_weight_threshold : float
        Maximum acceptable weight
    min_ess_ratio : float
        Minimum ESS ratio (ESS/N)
        
    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    n = len(weights)
    ess = calculate_effective_sample_size(weights)
    max_weight = np.max(weights)
    ess_ratio = ess / n
    
    validation = {
        'max_weight_ok': max_weight <= max_weight_threshold,
        'ess_ratio_ok': ess_ratio >= min_ess_ratio,
        'max_weight': max_weight,
        'ess_ratio': ess_ratio,
        'effective_sample_size': ess,
        'sample_size': n
    }
    
    validation['overall_quality'] = validation['max_weight_ok'] and validation['ess_ratio_ok']
    
    return validation


def create_example_ices_marginals(output_path: Path) -> None:
    """
    Create example ICES marginals file for testing
    
    Parameters:
    -----------
    output_path : Path
        Path where to save example marginals
    """
    logger.info(f"Creating example ICES marginals at: {output_path}")
    
    # Example marginal distributions for Ontario population
    marginals_data = {
        'variable': [
            'age_group', 'age_group', 'age_group', 'age_group', 'age_group',
            'sex', 'sex',
            'region', 'region', 'region', 'region',
            'socioeconomic_quintile', 'socioeconomic_quintile', 'socioeconomic_quintile', 
            'socioeconomic_quintile', 'socioeconomic_quintile'
        ],
        'category': [
            '18-34', '35-49', '50-64', '65-79', '80+',
            'female', 'male',
            'urban', 'suburban', 'rural', 'remote',
            'q1_lowest', 'q2', 'q3', 'q4', 'q5_highest'
        ],
        'proportion': [
            0.25, 0.22, 0.23, 0.20, 0.10,  # Age groups
            0.52, 0.48,  # Sex
            0.45, 0.35, 0.15, 0.05,  # Region
            0.20, 0.20, 0.20, 0.20, 0.20  # SES quintiles
        ]
    }
    
    marginals_df = pd.DataFrame(marginals_data)
    
    # Pivot to wide format for easier use
    wide_marginals = marginals_df.pivot(index='category', columns='variable', values='proportion').fillna(0)
    
    # Save both formats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    marginals_df.to_csv(output_path, index=False)
    
    logger.info(f"Example ICES marginals saved to: {output_path}")


def main():
    """Main execution for transport weights"""
    logger.info("Transportability weights module ready")
    
    print("Transport Weights Functions:")
    print("  - calculate_transport_weights() - Compute transportability weights")
    print("  - validate_transport_weights() - Validate weight quality")
    print("  - create_example_ices_marginals() - Create example marginals file")
    print("")
    print("Usage:")
    print("  weights = calculate_transport_weights(study_data)")
    print("  if weights['status'] == 'skipped':")
    print("      print('ICES marginals not available - using uniform weights')")
    
    # Check if example file should be created
    example_path = Path("data/external/ices_marginals.csv")
    if not example_path.exists():
        print(f"\nExample marginals file can be created at: {example_path}")
        print("Run: create_example_ices_marginals(Path('data/external/ices_marginals.csv'))")


if __name__ == "__main__":
    main()