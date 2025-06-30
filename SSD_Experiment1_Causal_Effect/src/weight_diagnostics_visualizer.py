#!/usr/bin/env python3
"""
weight_diagnostics_visualizer.py - Comprehensive weight diagnostics and Love plot generation

STATISTICAL PURPOSE:
1. Weight diagnostics assess IPTW quality and extreme weight detection
2. Love plots visualize covariate balance before/after weighting 
3. Propensity score density overlap checks for common support

INTEGRATION POINTS:
- Reads ps_matching_results.json for weight diagnostics
- Reads ps_weighted.parquet for covariate balance calculations
- Generates publication-ready figures following reporting standards

Following CLAUDE.md requirements:
- Evidence-based implementation with statistical rigor
- Functions ≤50 lines with comprehensive documentation
- TDD approach with input validation

Author: Ryhan Suny (Toronto Metropolitan University)
Supervisor: Dr. Aziz Guergachi
Research Team: Car4Mind, University of Toronto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set publication-ready style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})


def calculate_standardized_mean_difference(data: pd.DataFrame, 
                                        treatment_col: str,
                                        covariate_cols: List[str],
                                        weight_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate standardized mean differences for covariate balance assessment.
    
    STATISTICAL METHOD:
    SMD = (mean_treated - mean_control) / pooled_standard_deviation
    
    For weighted analysis:
    - Use weighted means and pooled standard deviations
    - Follow Austin (2011) recommendations for IPTW balance
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with treatment, covariates, and optional weights
    treatment_col : str
        Treatment indicator column (0/1)
    covariate_cols : List[str]
        List of covariate columns to assess
    weight_col : Optional[str]
        Weight column for weighted SMD calculation
        
    Returns:
    --------
    pd.DataFrame
        SMD results with columns: covariate, smd_unweighted, smd_weighted (if weights provided)
    """
    logger.info(f"Calculating SMD for {len(covariate_cols)} covariates")
    
    results = []
    
    for covariate in covariate_cols:
        if covariate not in data.columns:
            logger.warning(f"Covariate {covariate} not found in data, skipping")
            continue
        
        # Get treatment groups
        treated = data[data[treatment_col] == 1][covariate]
        control = data[data[treatment_col] == 0][covariate]
        
        # Unweighted SMD
        if len(treated) > 0 and len(control) > 0:
            mean_diff = treated.mean() - control.mean()
            pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
            smd_unweighted = mean_diff / pooled_sd if pooled_sd > 0 else 0.0
        else:
            smd_unweighted = np.nan
        
        # Weighted SMD (if weights provided)
        smd_weighted = np.nan
        if weight_col and weight_col in data.columns:
            weights = data[weight_col]
            treated_weights = weights[data[treatment_col] == 1]
            control_weights = weights[data[treatment_col] == 0]
            
            if len(treated_weights) > 0 and len(control_weights) > 0:
                # Weighted means
                mean_treated_w = np.average(treated, weights=treated_weights)
                mean_control_w = np.average(control, weights=control_weights)
                
                # Weighted variances
                var_treated_w = np.average((treated - mean_treated_w)**2, weights=treated_weights)
                var_control_w = np.average((control - mean_control_w)**2, weights=control_weights)
                
                # Weighted SMD
                mean_diff_w = mean_treated_w - mean_control_w
                pooled_sd_w = np.sqrt((var_treated_w + var_control_w) / 2)
                smd_weighted = mean_diff_w / pooled_sd_w if pooled_sd_w > 0 else 0.0
        
        results.append({
            'covariate': covariate,
            'smd_unweighted': smd_unweighted,
            'smd_weighted': smd_weighted
        })
    
    return pd.DataFrame(results)


def create_love_plot(smd_results: pd.DataFrame, 
                    output_path: Path,
                    title: str = "Covariate Balance Assessment") -> None:
    """
    Create Love plot showing standardized mean differences before/after weighting.
    
    DESIGN STANDARDS:
    - Follows Austin (2011) recommendations for balance assessment
    - SMD < 0.1 indicates good balance (vertical reference lines)
    - Color coding: before (red), after (blue)
    - Publication-ready formatting
    
    Parameters:
    -----------
    smd_results : pd.DataFrame
        SMD results from calculate_standardized_mean_difference()
    output_path : Path
        Output file path for the plot
    title : str
        Plot title
    """
    logger.info(f"Creating Love plot with {len(smd_results)} covariates")
    
    # Filter valid results
    valid_results = smd_results.dropna(subset=['smd_unweighted'])
    
    if len(valid_results) == 0:
        logger.warning("No valid SMD results for Love plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(valid_results) * 0.4)))
    
    # Plot settings
    y_positions = range(len(valid_results))
    covariate_labels = valid_results['covariate'].tolist()
    
    # Plot unweighted SMD (before)
    ax.scatter(valid_results['smd_unweighted'], y_positions, 
              color='red', alpha=0.7, s=60, label='Before weighting', marker='o')
    
    # Plot weighted SMD (after) if available
    if not valid_results['smd_weighted'].isna().all():
        ax.scatter(valid_results['smd_weighted'], y_positions,
                  color='blue', alpha=0.7, s=60, label='After weighting', marker='s')
    
    # Reference lines for good balance (|SMD| < 0.1)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.6, label='SMD = ±0.1')
    ax.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.6)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(covariate_labels, fontsize=10)
    ax.set_xlabel('Standardized Mean Difference', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Love plot saved to {output_path}")


def create_weight_distribution_plot(data: pd.DataFrame,
                                  weight_col: str,
                                  treatment_col: str,
                                  output_path: Path) -> None:
    """
    Create weight distribution diagnostic plots.
    
    DIAGNOSTIC CRITERIA:
    - Extreme weights: >10 or <0.1 (following Harder et al. 2010)
    - Effective sample size reduction
    - Distribution by treatment group
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with weights and treatment
    weight_col : str
        Weight column name
    treatment_col : str
        Treatment column name
    output_path : Path
        Output file path
    """
    logger.info("Creating weight distribution diagnostic plots")
    
    if weight_col not in data.columns:
        logger.error(f"Weight column {weight_col} not found in data")
        return
    
    weights = data[weight_col]
    treatment = data[treatment_col]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall weight distribution
    ax1.hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(weights.median(), color='red', linestyle='--', 
               label=f'Median = {weights.median():.3f}')
    ax1.axvline(weights.mean(), color='orange', linestyle='--',
               label=f'Mean = {weights.mean():.3f}')
    ax1.set_xlabel('IPTW Weight')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Weight Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Weight distribution by treatment group
    treated_weights = weights[treatment == 1]
    control_weights = weights[treatment == 0]
    
    ax2.hist(treated_weights, bins=30, alpha=0.6, color='red', 
            label=f'Treated (n={len(treated_weights)})', density=True)
    ax2.hist(control_weights, bins=30, alpha=0.6, color='blue', 
            label=f'Control (n={len(control_weights)})', density=True)
    ax2.set_xlabel('IPTW Weight')
    ax2.set_ylabel('Density')
    ax2.set_title('Weight Distribution by Treatment Group')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot by treatment
    weight_data = [control_weights, treated_weights]
    ax3.boxplot(weight_data, labels=['Control', 'Treated'])
    ax3.set_ylabel('IPTW Weight')
    ax3.set_title('Weight Distribution Box Plots')
    ax3.grid(True, alpha=0.3)
    
    # 4. Extreme weights identification
    extreme_threshold = 10  # Following literature recommendations
    extreme_weights = weights[(weights > extreme_threshold) | (weights < 1/extreme_threshold)]
    
    ax4.scatter(range(len(weights)), weights, alpha=0.5, s=10, color='gray')
    if len(extreme_weights) > 0:
        extreme_indices = weights[(weights > extreme_threshold) | (weights < 1/extreme_threshold)].index
        ax4.scatter(extreme_indices, weights.loc[extreme_indices], 
                   color='red', s=20, alpha=0.8, label=f'Extreme weights (n={len(extreme_weights)})')
    
    ax4.axhline(y=extreme_threshold, color='red', linestyle='--', alpha=0.7)
    ax4.axhline(y=1/extreme_threshold, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Observation Index')
    ax4.set_ylabel('IPTW Weight')
    ax4.set_title('Extreme Weight Detection')
    ax4.set_yscale('log')
    if len(extreme_weights) > 0:
        ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Weight distribution plots saved to {output_path}")


def create_propensity_score_overlap_plot(data: pd.DataFrame,
                                       ps_col: str,
                                       treatment_col: str,
                                       output_path: Path) -> None:
    """
    Create propensity score density overlap plots for common support assessment.
    
    STATISTICAL PURPOSE:
    - Assess overlap assumption for causal inference
    - Identify regions of poor common support
    - Following D'Agostino (1998) recommendations
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with propensity scores and treatment
    ps_col : str
        Propensity score column name
    treatment_col : str
        Treatment column name
    output_path : Path
        Output file path
    """
    logger.info("Creating propensity score overlap diagnostic plots")
    
    if ps_col not in data.columns:
        logger.warning(f"Propensity score column {ps_col} not found, skipping overlap plot")
        return
    
    ps_scores = data[ps_col]
    treatment = data[treatment_col]
    
    # Split by treatment
    ps_treated = ps_scores[treatment == 1]
    ps_control = ps_scores[treatment == 0]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 1. Density plot
    ax1.hist(ps_control, bins=50, alpha=0.6, color='blue', density=True, 
            label=f'Control (n={len(ps_control)})')
    ax1.hist(ps_treated, bins=50, alpha=0.6, color='red', density=True,
            label=f'Treated (n={len(ps_treated)})')
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Propensity Score Distribution by Treatment Group')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ps_data = [ps_control, ps_treated]
    ax2.boxplot(ps_data, labels=['Control', 'Treated'])
    ax2.set_ylabel('Propensity Score')
    ax2.set_title('Propensity Score Box Plots by Treatment Group')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Propensity score overlap plots saved to {output_path}")


def generate_comprehensive_diagnostics_report(data_path: Path,
                                            results_path: Path,
                                            treatment_col: str = 'ssd_flag',
                                            output_dir: Path = Path("figures")) -> Dict[str, Any]:
    """
    Generate comprehensive weight diagnostics and balance assessment report.
    
    INTEGRATION WORKFLOW:
    1. Load ps_weighted.parquet and ps_matching_results.json
    2. Calculate covariate balance (SMD)
    3. Generate Love plot, weight diagnostics, PS overlap plots
    4. Create summary statistics report
    
    Parameters:
    -----------
    data_path : Path
        Path to ps_weighted.parquet
    results_path : Path
        Path to ps_matching_results.json
    treatment_col : str
        Treatment column name
    output_dir : Path
        Output directory for figures
        
    Returns:
    --------
    Dict[str, Any]
        Summary diagnostics report
    """
    logger.info("Generating comprehensive weight diagnostics report")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    if not data_path.exists():
        raise FileNotFoundError(f"PS weighted data not found: {data_path}")
    
    data = pd.read_parquet(data_path)
    logger.info(f"Loaded data with {len(data)} observations")
    
    # Load PS matching results
    ps_results = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            ps_results = json.load(f)
        logger.info("Loaded PS matching results")
    else:
        logger.warning(f"PS matching results not found: {results_path}")
    
    # Identify covariate columns
    covariate_cols = [col for col in data.columns if 
                     col.endswith('_conf') or 
                     col in ['age', 'sex_M', 'charlson_score', 'baseline_encounters']]
    covariate_cols = [col for col in covariate_cols if col in data.columns]
    
    logger.info(f"Found {len(covariate_cols)} covariates for balance assessment")
    
    # Calculate SMD
    smd_results = calculate_standardized_mean_difference(
        data, treatment_col, covariate_cols, weight_col='iptw'
    )
    
    # Generate plots
    plots_generated = {}
    
    # 1. Love plot
    try:
        love_plot_path = output_dir / "love_plot_covariate_balance.png"
        create_love_plot(smd_results, love_plot_path)
        plots_generated['love_plot'] = str(love_plot_path)
    except Exception as e:
        logger.error(f"Failed to create Love plot: {e}")
    
    # 2. Weight distribution plots
    try:
        if 'iptw' in data.columns:
            weight_plot_path = output_dir / "weight_distribution_diagnostics.png"
            create_weight_distribution_plot(data, 'iptw', treatment_col, weight_plot_path)
            plots_generated['weight_diagnostics'] = str(weight_plot_path)
    except Exception as e:
        logger.error(f"Failed to create weight diagnostics: {e}")
    
    # 3. Propensity score overlap
    try:
        ps_cols = [col for col in data.columns if 'propensity' in col.lower() or col == 'ps']
        if ps_cols:
            ps_overlap_path = output_dir / "propensity_score_overlap.png"
            create_propensity_score_overlap_plot(data, ps_cols[0], treatment_col, ps_overlap_path)
            plots_generated['ps_overlap'] = str(ps_overlap_path)
    except Exception as e:
        logger.error(f"Failed to create PS overlap plot: {e}")
    
    # Generate summary report
    summary_report = {
        'diagnostics_timestamp': datetime.now().isoformat(),
        'data_summary': {
            'n_observations': len(data),
            'n_treated': int(data[treatment_col].sum()),
            'n_control': int(len(data) - data[treatment_col].sum()),
            'n_covariates_assessed': len(covariate_cols)
        },
        'balance_assessment': {
            'max_smd_unweighted': float(smd_results['smd_unweighted'].abs().max()),
            'max_smd_weighted': float(smd_results['smd_weighted'].abs().max()) if not smd_results['smd_weighted'].isna().all() else None,
            'n_covariates_unbalanced_before': int((smd_results['smd_unweighted'].abs() > 0.1).sum()),
            'n_covariates_unbalanced_after': int((smd_results['smd_weighted'].abs() > 0.1).sum()) if not smd_results['smd_weighted'].isna().all() else None
        },
        'weight_diagnostics': ps_results.get('weight_diagnostics', {}),
        'plots_generated': plots_generated,
        'covariate_details': smd_results.to_dict('records')
    }
    
    # Save summary report
    summary_path = output_dir / "weight_diagnostics_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    logger.info(f"Comprehensive diagnostics report saved to {summary_path}")
    logger.info(f"Generated {len(plots_generated)} diagnostic plots")
    
    return summary_report


def calculate_ess(weights: np.ndarray, return_diagnostics: bool = False) -> Any:
    """
    Calculate Effective Sample Size (ESS) for weighted analyses.
    
    FORMULA:
    ESS = n × sum(w)² / sum(w²)
    
    Where:
    - n = original sample size
    - w = weights (normalized or unnormalized)
    
    The ESS represents the approximate number of independent observations
    that would provide the same statistical precision as the weighted sample.
    
    Reference:
    - BMC Med Res Methodol (2024). "Three new methodologies for calculating ESS"
    - Austin PC & Stuart EA (2015). "Moving towards best practice..." Stat Med
    
    Parameters:
    -----------
    weights : np.ndarray
        Array of weights (must be positive)
    return_diagnostics : bool, default False
        If True, return dict with ESS and diagnostic info
        
    Returns:
    --------
    float or dict
        ESS value, or dict with diagnostics if requested
        
    Raises:
    -------
    ValueError
        If weights are invalid (empty, negative, all zero, or contain NaN)
    """
    # Convert to numpy array if needed
    if isinstance(weights, pd.Series):
        weights = weights.values
    weights = np.asarray(weights, dtype=float)
    
    # Validation
    if len(weights) == 0:
        raise ValueError("Weights array is empty")
    
    if np.any(np.isnan(weights)):
        raise ValueError("Weights contain NaN values")
    
    if np.any(weights < 0):
        raise ValueError("Weights contain negative values")
    
    if np.all(weights == 0):
        raise ValueError("All weights are zero")
    
    # Calculate ESS
    n = len(weights)
    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)
    
    ess = (sum_w**2) / sum_w2
    
    if not return_diagnostics:
        return ess
    
    # Return detailed diagnostics
    ess_ratio = ess / n
    warning = ess_ratio < 0.5  # Flag if ESS < 50% of original n
    
    return {
        'ess': ess,
        'ess_ratio': ess_ratio,
        'n_original': n,
        'sum_weights': sum_w,
        'sum_weights_squared': sum_w2,
        'warning': warning,
        'cv_weights': np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else np.inf
    }


def generate_weight_diagnostics(df: pd.DataFrame,
                              weight_col: str = 'ps_weight',
                              treatment_col: str = 'treatment') -> Dict[str, Any]:
    """
    Generate comprehensive weight diagnostics including ESS.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with weights and treatment
    weight_col : str
        Weight column name
    treatment_col : str
        Treatment column name
        
    Returns:
    --------
    Dict[str, Any]
        Diagnostic results including ESS by treatment group
    """
    # Overall ESS
    ess_overall = calculate_ess(df[weight_col])
    
    # ESS by treatment group
    treated_mask = df[treatment_col] == 1
    control_mask = df[treatment_col] == 0
    
    ess_treated = calculate_ess(df.loc[treated_mask, weight_col])
    ess_control = calculate_ess(df.loc[control_mask, weight_col])
    
    n_total = len(df)
    
    return {
        'ess_overall': ess_overall,
        'ess_treated': ess_treated,
        'ess_control': ess_control,
        'ess_ratio': ess_overall / n_total,
        'ess_ratio_treated': ess_treated / treated_mask.sum(),
        'ess_ratio_control': ess_control / control_mask.sum(),
        'n_total': n_total,
        'n_treated': treated_mask.sum(),
        'n_control': control_mask.sum()
    }


def main():
    """
    Command-line interface for weight diagnostics visualization.
    
    Following CLAUDE.md requirement for functions ≤50 lines.
    """
    parser = argparse.ArgumentParser(
        description="Generate comprehensive weight diagnostics and balance assessment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-file', default='data_derived/ps_weighted.parquet',
                       help='Path to PS weighted data file')
    parser.add_argument('--results-file', default='results/ps_matching_results.json',
                       help='Path to PS matching results file')
    parser.add_argument('--treatment-col', default='ssd_flag',
                       help='Treatment column name')
    parser.add_argument('--output-dir', default='figures',
                       help='Output directory for diagnostic plots')
    
    args = parser.parse_args()
    
    try:
        # Generate comprehensive diagnostics
        report = generate_comprehensive_diagnostics_report(
            data_path=Path(args.data_file),
            results_path=Path(args.results_file),
            treatment_col=args.treatment_col,
            output_dir=Path(args.output_dir)
        )
        
        # Print summary
        print("\n=== Weight Diagnostics Summary ===")
        print(f"Observations: {report['data_summary']['n_observations']:,}")
        print(f"Treated: {report['data_summary']['n_treated']:,}")
        print(f"Control: {report['data_summary']['n_control']:,}")
        print(f"Covariates assessed: {report['data_summary']['n_covariates_assessed']}")
        
        balance = report['balance_assessment']
        print(f"\nBalance Assessment:")
        print(f"  Max SMD before weighting: {balance['max_smd_unweighted']:.3f}")
        if balance['max_smd_weighted'] is not None:
            print(f"  Max SMD after weighting: {balance['max_smd_weighted']:.3f}")
        print(f"  Unbalanced covariates before: {balance['n_covariates_unbalanced_before']}")
        if balance['n_covariates_unbalanced_after'] is not None:
            print(f"  Unbalanced covariates after: {balance['n_covariates_unbalanced_after']}")
        
        weight_diag = report.get('weight_diagnostics', {})
        if weight_diag:
            print(f"\nWeight Diagnostics:")
            print(f"  ESS: {weight_diag.get('ess', 'N/A'):,.0f}")
            print(f"  ESS ratio: {weight_diag.get('ess_ratio', 'N/A'):.3f}")
            print(f"  Extreme weights: {weight_diag.get('n_extreme_weights', 'N/A')}")
        
        print(f"\nPlots generated: {len(report['plots_generated'])}")
        for plot_type, path in report['plots_generated'].items():
            print(f"  {plot_type}: {path}")
        print("================================\n")
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        raise


if __name__ == "__main__":
    main()