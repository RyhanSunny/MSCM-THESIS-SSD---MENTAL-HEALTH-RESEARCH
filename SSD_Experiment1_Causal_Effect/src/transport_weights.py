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
import json
import argparse
from typing import Dict, Optional, Any
from datetime import datetime

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
        try:
            # Get target proportions for this variable from long format data
            var_target = target_marginals[target_marginals['variable'] == var]
        except KeyError:
            logger.warning(f"Column 'variable' not found in marginals file")
            return _create_placeholder_results(study_data, "invalid_marginals_format")
        
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
    """
    Create placeholder results when transport weighting cannot be performed
    
    Parameters:
    -----------
    study_data : pd.DataFrame
        Study dataset to generate uniform weights for
    reason : str
        Reason why transport weighting was skipped
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with uniform weights and placeholder statistics
    """
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


def main_cli(study_data_path: Path, 
             marginals_path: Optional[Path] = None,
             output_dir: Optional[Path] = None,
             variables: Optional[list] = None) -> Dict[str, Any]:
    """
    Command-line interface for transport weights analysis
    
    Parameters:
    -----------
    study_data_path : Path
        Path to CSV file with study data
    marginals_path : Optional[Path]
        Path to ICES marginals CSV file
    output_dir : Optional[Path]
        Output directory for results
    variables : Optional[list]
        Variables to use for transport weighting
        
    Returns:
    --------
    Dict[str, Any]
        Transport weight analysis results
    """
    logger.info("Starting transport weights CLI analysis...")
    
    # Set defaults
    if marginals_path is None:
        marginals_path = Path("data/external/ices_marginals.csv")
    if output_dir is None:
        output_dir = Path("results/transport_weights")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load study data
    try:
        study_data = pd.read_csv(study_data_path)
        logger.info(f"Loaded study data: {study_data.shape}")
    except Exception as e:
        logger.error(f"Failed to load study data: {e}")
        return {'status': 'error', 'message': str(e)}
    
    # Calculate transport weights
    results = calculate_transport_weights(
        study_data, marginals_path, variables
    )
    
    # Save results
    try:
        # Save weights to CSV
        weights_df = pd.DataFrame({
            'patient_id': range(len(results['weights'])),
            'transport_weight': results['weights']
        })
        weights_path = output_dir / 'transport_weights.csv'
        weights_df.to_csv(weights_path, index=False)
        logger.info(f"Weights saved: {weights_path}")
        
        # Save diagnostics to JSON
        diagnostics = {
            'status': results['status'],
            'effective_sample_size': results['effective_sample_size'],
            'max_weight': results['max_weight'],
            'mean_weight': results['mean_weight'],
            'n_observations': results['n_observations'],
            'analysis_timestamp': datetime.now().isoformat()
        }
        if 'reason' in results:
            diagnostics['reason'] = results['reason']
        if 'variables_used' in results:
            diagnostics['variables_used'] = results['variables_used']
            
        diag_path = output_dir / 'transport_diagnostics.json'
        with open(diag_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        logger.info(f"Diagnostics saved: {diag_path}")
        
        # Add file paths to results
        results['weights_file'] = str(weights_path)
        results['diagnostics_file'] = str(diag_path)
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        results['save_error'] = str(e)
    
    logger.info("Transport weights CLI analysis complete")
    return results


def run_transport_analysis(study_data_path: Path,
                          output_dir: Optional[Path] = None,
                          marginals_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run complete transport analysis workflow
    
    Parameters:
    -----------
    study_data_path : Path
        Path to study data CSV
    output_dir : Optional[Path]
        Output directory for results
    marginals_path : Optional[Path]
        Path to ICES marginals (optional)
        
    Returns:
    --------
    Dict[str, Any]
        Complete analysis results
    """
    logger.info("Running complete transport analysis workflow...")
    
    if output_dir is None:
        output_dir = Path("results/transport_analysis")
    
    # Run main analysis
    transport_results = main_cli(
        study_data_path=study_data_path,
        marginals_path=marginals_path,
        output_dir=output_dir
    )
    
    # Validate weights if analysis succeeded
    validation_results = None
    if transport_results['status'] in ['completed', 'skipped']:
        try:
            validation_results = validate_transport_weights(
                transport_results['weights']
            )
            logger.info(f"Weight validation: {validation_results['overall_quality']}")
        except Exception as e:
            logger.warning(f"Weight validation failed: {e}")
            validation_results = {'error': str(e)}
    
    # Create summary report
    try:
        report_content = f"""# Transport Weights Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Results

**Status:** {transport_results['status']}
**Sample Size:** {transport_results['n_observations']}
**Effective Sample Size:** {transport_results['effective_sample_size']:.1f}
**Maximum Weight:** {transport_results['max_weight']:.3f}
**Mean Weight:** {transport_results['mean_weight']:.3f}

"""
        
        if transport_results['status'] == 'skipped':
            report_content += f"""
## Note

Analysis was skipped due to: {transport_results.get('reason', 'Unknown reason')}
Uniform weights (all 1.0) were applied for compatibility.
"""
        elif transport_results['status'] == 'completed':
            report_content += f"""
## Transport Variables

Variables used for reweighting: {', '.join(transport_results.get('variables_used', []))}

## Quality Assessment

"""
            if validation_results:
                quality = validation_results.get('overall_quality', False)
                report_content += f"""
**Overall Quality:** {'✅ PASS' if quality else '⚠️ FAIL'}
**ESS Ratio:** {validation_results.get('ess_ratio', 0):.3f}
**Max Weight Check:** {'✅ PASS' if validation_results.get('max_weight_ok', False) else '⚠️ FAIL'}
"""
        
        report_content += "\n---\n*Generated by SSD Experiment 1 Transport Weights Module v4.0.0*"
        
        report_path = output_dir / 'transport_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        logger.info(f"Report saved: {report_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create report: {e}")
    
    return {
        'transport_results': transport_results,
        'validation_results': validation_results,
        'output_directory': str(output_dir)
    }


def main():
    """Main execution for transport weights"""
    parser = argparse.ArgumentParser(description='Transport weights for external validity')
    parser.add_argument('--study-data', type=Path,
                       help='Path to study data CSV file')
    parser.add_argument('--marginals', type=Path,
                       help='Path to ICES marginals CSV file')
    parser.add_argument('--output-dir', type=Path, default=Path('results/transport'),
                       help='Output directory for results')
    parser.add_argument('--variables', nargs='+',
                       help='Variables to use for transport weighting')
    parser.add_argument('--create-example', action='store_true',
                       help='Create example ICES marginals file')
    
    args = parser.parse_args()
    
    if args.create_example:
        example_path = Path("data/external/ices_marginals.csv")
        create_example_ices_marginals(example_path)
        print(f"Example ICES marginals created: {example_path}")
        return
    
    if args.study_data:
        results = main_cli(
            study_data_path=args.study_data,
            marginals_path=args.marginals,
            output_dir=args.output_dir,
            variables=args.variables
        )
        
        print(f"Transport weights analysis complete:")
        print(f"  Status: {results['status']}")
        print(f"  Sample size: {results['n_observations']}")
        print(f"  Effective sample size: {results['effective_sample_size']:.1f}")
        
        if results['status'] == 'skipped':
            print(f"  Reason: {results.get('reason', 'Unknown')}")
    else:
        logger.info("Transportability weights module ready")
        
        print("Transport Weights Functions:")
        print("  - calculate_transport_weights() - Compute transportability weights")
        print("  - validate_transport_weights() - Validate weight quality")
        print("  - create_example_ices_marginals() - Create example marginals file")
        print("  - main_cli() - Command-line interface")
        print("  - run_transport_analysis() - Complete workflow")
        print("")
        print("Usage:")
        print("  python3 transport_weights.py --study-data data.csv --marginals ices.csv")
        print("  python3 transport_weights.py --create-example")
        
        # Check if example file should be created
        example_path = Path("data/external/ices_marginals.csv")
        if not example_path.exists():
            print(f"\nExample marginals file can be created at: {example_path}")
            print("Run: python3 transport_weights.py --create-example")


if __name__ == "__main__":
    main()