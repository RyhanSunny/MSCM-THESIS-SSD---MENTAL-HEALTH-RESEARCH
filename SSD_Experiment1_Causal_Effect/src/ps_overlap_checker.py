#!/usr/bin/env python3
"""
ps_overlap_checker.py - Propensity Score Overlap and Common Support Assessment

STATISTICAL PURPOSE:
Assess the overlap assumption for causal inference by examining:
1. Propensity score distributions between treatment groups
2. Common support regions and violations
3. Extreme propensity score detection (near 0 or 1)
4. Effective sample size under common support

THEORETICAL FOUNDATION:
- Common support: 0 < P(T=1|X) < 1 for all observed X
- Following D'Agostino (1998) and Rosenbaum & Rubin (1983)
- Trim extreme PS values following Crump et al. (2009)

Following CLAUDE.md requirements:
- Evidence-based statistical methods
- Functions ≤50 lines with comprehensive validation
- No assumptions without verification

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
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def assess_propensity_score_overlap(data: pd.DataFrame,
                                  ps_col: str,
                                  treatment_col: str,
                                  alpha: float = 0.1) -> Dict[str, Any]:
    """
    Assess propensity score overlap and common support violations.
    
    STATISTICAL CRITERIA:
    1. Extreme PS: < alpha/2 or > (1-alpha/2) following Crump et al. (2009)
    2. Non-overlapping regions: PS ranges with only treated or only control
    3. Density overlap assessment using Kolmogorov-Smirnov test
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with propensity scores and treatment
    ps_col : str
        Propensity score column name
    treatment_col : str
        Treatment indicator column (0/1)
    alpha : float
        Threshold for extreme PS detection
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive overlap assessment results
    """
    logger.info(f"Assessing PS overlap for {len(data)} observations")
    
    if ps_col not in data.columns:
        raise ValueError(f"Propensity score column {ps_col} not found")
    
    if treatment_col not in data.columns:
        raise ValueError(f"Treatment column {treatment_col} not found")
    
    ps_scores = data[ps_col].copy()
    treatment = data[treatment_col].copy()
    
    # Basic statistics
    ps_treated = ps_scores[treatment == 1]
    ps_control = ps_scores[treatment == 0]
    
    # 1. Extreme propensity score detection
    lower_threshold = alpha / 2
    upper_threshold = 1 - (alpha / 2)
    
    extreme_low = (ps_scores < lower_threshold).sum()
    extreme_high = (ps_scores > upper_threshold).sum()
    n_extreme = extreme_low + extreme_high
    
    # 2. Range assessment
    ps_range_treated = (ps_treated.min(), ps_treated.max())
    ps_range_control = (ps_control.min(), ps_control.max())
    overlap_range = (max(ps_range_treated[0], ps_range_control[0]),
                    min(ps_range_treated[1], ps_range_control[1]))
    
    # Check for non-overlapping regions
    has_overlap = overlap_range[0] < overlap_range[1]
    overlap_width = max(0, overlap_range[1] - overlap_range[0])
    
    # 3. Statistical tests for distribution differences
    ks_statistic, ks_pvalue = stats.ks_2samp(ps_treated, ps_control)
    
    # 4. Effective sample size under common support
    if has_overlap:
        in_overlap_treated = ((ps_treated >= overlap_range[0]) & 
                             (ps_treated <= overlap_range[1])).sum()
        in_overlap_control = ((ps_control >= overlap_range[0]) & 
                             (ps_control <= overlap_range[1])).sum()
        eff_sample_size = in_overlap_treated + in_overlap_control
    else:
        in_overlap_treated = 0
        in_overlap_control = 0
        eff_sample_size = 0
    
    # 5. Density overlap assessment (empirical)
    ps_bins = np.linspace(ps_scores.min(), ps_scores.max(), 50)
    hist_treated, _ = np.histogram(ps_treated, bins=ps_bins, density=True)
    hist_control, _ = np.histogram(ps_control, bins=ps_bins, density=True)
    
    # Calculate overlapping area (approximation)
    bin_width = ps_bins[1] - ps_bins[0]
    overlap_area = np.sum(np.minimum(hist_treated, hist_control)) * bin_width
    
    return {
        'assessment_timestamp': datetime.now().isoformat(),
        'basic_statistics': {
            'n_total': len(data),
            'n_treated': len(ps_treated),
            'n_control': len(ps_control),
            'ps_mean_treated': float(ps_treated.mean()),
            'ps_mean_control': float(ps_control.mean()),
            'ps_std_treated': float(ps_treated.std()),
            'ps_std_control': float(ps_control.std())
        },
        'extreme_ps_assessment': {
            'alpha_threshold': alpha,
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'n_extreme_low': int(extreme_low),
            'n_extreme_high': int(extreme_high),
            'n_extreme_total': int(n_extreme),
            'pct_extreme': float(n_extreme / len(data) * 100)
        },
        'overlap_assessment': {
            'ps_range_treated': ps_range_treated,
            'ps_range_control': ps_range_control,
            'overlap_range': overlap_range if has_overlap else None,
            'has_overlap': has_overlap,
            'overlap_width': float(overlap_width),
            'pct_overlap_width': float(overlap_width / (ps_scores.max() - ps_scores.min()) * 100) if ps_scores.max() > ps_scores.min() else 0
        },
        'common_support_ess': {
            'in_overlap_treated': int(in_overlap_treated),
            'in_overlap_control': int(in_overlap_control),
            'effective_sample_size': int(eff_sample_size),
            'ess_ratio': float(eff_sample_size / len(data)) if len(data) > 0 else 0
        },
        'distribution_tests': {
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'distributions_different': ks_pvalue < 0.05
        },
        'density_overlap': {
            'empirical_overlap_area': float(overlap_area),
            'theoretical_max_area': 1.0,
            'overlap_coefficient': float(min(overlap_area, 1.0))
        }
    }


def create_detailed_overlap_plot(data: pd.DataFrame,
                               ps_col: str,
                               treatment_col: str,
                               output_path: Path,
                               alpha: float = 0.1) -> None:
    """
    Create comprehensive propensity score overlap visualization.
    
    PLOT COMPONENTS:
    1. Density plots with shaded overlap regions
    2. Common support identification
    3. Extreme PS highlighting
    4. Statistical test results annotation
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with PS and treatment
    ps_col : str
        Propensity score column
    treatment_col : str
        Treatment column
    output_path : Path
        Output file path
    alpha : float
        Extreme PS threshold
    """
    logger.info("Creating detailed PS overlap visualization")
    
    ps_scores = data[ps_col]
    treatment = data[treatment_col]
    
    ps_treated = ps_scores[treatment == 1]
    ps_control = ps_scores[treatment == 0]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Density plot with overlap
    ax1.hist(ps_control, bins=50, alpha=0.6, color='blue', density=True, 
            label=f'Control (n={len(ps_control)})')
    ax1.hist(ps_treated, bins=50, alpha=0.6, color='red', density=True,
            label=f'Treated (n={len(ps_treated)})')
    
    # Mark extreme regions
    lower_thresh = alpha / 2
    upper_thresh = 1 - (alpha / 2)
    ax1.axvline(lower_thresh, color='orange', linestyle='--', alpha=0.8,
               label=f'Extreme PS thresholds (α={alpha})')
    ax1.axvline(upper_thresh, color='orange', linestyle='--', alpha=0.8)
    
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Propensity Score Distribution by Treatment Group')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plots for comparison
    ps_data = [ps_control, ps_treated]
    box_plot = ax2.boxplot(ps_data, labels=['Control', 'Treated'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    box_plot['boxes'][0].set_alpha(0.6)
    box_plot['boxes'][1].set_alpha(0.6)
    
    ax2.set_ylabel('Propensity Score')
    ax2.set_title('PS Distribution Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 3. Common support identification
    ps_range = np.linspace(ps_scores.min(), ps_scores.max(), 100)
    
    # Create histogram bins for support assessment
    bins = np.linspace(ps_scores.min(), ps_scores.max(), 20)
    hist_treated, bin_edges = np.histogram(ps_treated, bins=bins)
    hist_control, _ = np.histogram(ps_control, bins=bins)
    
    # Identify bins with both groups
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    common_support_bins = (hist_treated > 0) & (hist_control > 0)
    
    ax3.bar(bin_centers, hist_treated, alpha=0.6, color='red', 
           label='Treated', width=np.diff(bin_edges)[0])
    ax3.bar(bin_centers, hist_control, alpha=0.6, color='blue', 
           label='Control', width=np.diff(bin_edges)[0])
    
    # Highlight common support regions
    for i, (center, has_support) in enumerate(zip(bin_centers, common_support_bins)):
        if has_support:
            ax3.axvline(center, color='green', alpha=0.3, linewidth=2)
    
    ax3.set_xlabel('Propensity Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Common Support Assessment')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. QQ plot for distribution comparison
    from scipy.stats import probplot
    
    # Sample equal sizes for QQ plot
    min_size = min(len(ps_treated), len(ps_control))
    if min_size > 1000:  # Subsample for performance
        min_size = 1000
    
    ps_treated_sample = np.random.choice(ps_treated, min_size, replace=False)
    ps_control_sample = np.random.choice(ps_control, min_size, replace=False)
    
    # Sort for QQ plot
    ps_treated_sorted = np.sort(ps_treated_sample)
    ps_control_sorted = np.sort(ps_control_sample)
    
    ax4.scatter(ps_control_sorted, ps_treated_sorted, alpha=0.6, s=10)
    
    # Add diagonal line
    min_val = min(ps_scores.min(), ps_scores.min())
    max_val = max(ps_scores.max(), ps_scores.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    ax4.set_xlabel('Control PS Quantiles')
    ax4.set_ylabel('Treated PS Quantiles')
    ax4.set_title('Q-Q Plot: PS Distribution Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Detailed overlap plot saved to {output_path}")


def trim_extreme_propensity_scores(data: pd.DataFrame,
                                 ps_col: str,
                                 treatment_col: str,
                                 alpha: float = 0.1) -> pd.DataFrame:
    """
    Trim observations with extreme propensity scores following Crump et al. (2009).
    
    TRIMMING RULE:
    Remove observations with PS < alpha/2 or PS > (1-alpha/2)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Original dataset
    ps_col : str
        Propensity score column
    treatment_col : str
        Treatment column
    alpha : float
        Trimming threshold
        
    Returns:
    --------
    pd.DataFrame
        Trimmed dataset with improved common support
    """
    logger.info(f"Trimming extreme PS with alpha={alpha}")
    
    ps_scores = data[ps_col]
    
    # Define trimming bounds
    lower_bound = alpha / 2
    upper_bound = 1 - (alpha / 2)
    
    # Create trimming mask
    keep_mask = (ps_scores >= lower_bound) & (ps_scores <= upper_bound)
    
    # Apply trimming
    trimmed_data = data[keep_mask].copy()
    
    # Log trimming results
    n_original = len(data)
    n_trimmed = len(trimmed_data)
    n_removed = n_original - n_trimmed
    
    logger.info(f"Trimming results:")
    logger.info(f"  Original: {n_original:,} observations")
    logger.info(f"  Trimmed: {n_trimmed:,} observations")
    logger.info(f"  Removed: {n_removed:,} observations ({n_removed/n_original*100:.1f}%)")
    
    # Check treatment balance after trimming
    original_treated_pct = data[treatment_col].mean() * 100
    trimmed_treated_pct = trimmed_data[treatment_col].mean() * 100
    
    logger.info(f"  Treatment prevalence: {original_treated_pct:.1f}% → {trimmed_treated_pct:.1f}%")
    
    return trimmed_data


def generate_overlap_assessment_report(data_path: Path,
                                     ps_col: str = 'propensity_score',
                                     treatment_col: str = 'ssd_flag',
                                     alpha: float = 0.1,
                                     output_dir: Path = Path("results")) -> Dict[str, Any]:
    """
    Generate comprehensive propensity score overlap assessment report.
    
    WORKFLOW:
    1. Load data and assess current overlap
    2. Generate detailed diagnostic plots
    3. Recommend trimming if needed
    4. Create trimmed dataset assessment
    
    Parameters:
    -----------
    data_path : Path
        Path to dataset with propensity scores
    ps_col : str
        Propensity score column name
    treatment_col : str
        Treatment column name
    alpha : float
        Extreme PS threshold
    output_dir : Path
        Output directory
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive overlap assessment
    """
    logger.info(f"Generating overlap assessment report from {data_path}")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(data)} observations")
    
    # Check for PS column
    if ps_col not in data.columns:
        # Try to find PS column
        ps_candidates = [col for col in data.columns if 'propensity' in col.lower() or col == 'ps']
        if ps_candidates:
            ps_col = ps_candidates[0]
            logger.info(f"Using PS column: {ps_col}")
        else:
            raise ValueError(f"No propensity score column found. Available columns: {list(data.columns)}")
    
    # 1. Original overlap assessment
    original_assessment = assess_propensity_score_overlap(
        data, ps_col, treatment_col, alpha
    )
    
    # 2. Generate detailed plots
    overlap_plot_path = output_dir / "ps_overlap_detailed_assessment.png"
    create_detailed_overlap_plot(data, ps_col, treatment_col, overlap_plot_path, alpha)
    
    # 3. Trimming recommendation and implementation
    trimming_recommended = (
        original_assessment['extreme_ps_assessment']['pct_extreme'] > 5.0 or
        not original_assessment['overlap_assessment']['has_overlap'] or
        original_assessment['common_support_ess']['ess_ratio'] < 0.8
    )
    
    trimmed_assessment = None
    if trimming_recommended:
        logger.info("Trimming recommended based on overlap assessment")
        
        # Create trimmed dataset
        trimmed_data = trim_extreme_propensity_scores(data, ps_col, treatment_col, alpha)
        
        # Assess trimmed overlap
        trimmed_assessment = assess_propensity_score_overlap(
            trimmed_data, ps_col, treatment_col, alpha
        )
        
        # Generate trimmed plot
        trimmed_plot_path = output_dir / "ps_overlap_after_trimming.png"
        create_detailed_overlap_plot(trimmed_data, ps_col, treatment_col, trimmed_plot_path, alpha)
        
        # Save trimmed dataset
        trimmed_data_path = output_dir / "ps_weighted_trimmed.parquet"
        trimmed_data.to_parquet(trimmed_data_path)
        logger.info(f"Trimmed dataset saved to {trimmed_data_path}")
    
    # 4. Compile comprehensive report
    comprehensive_report = {
        'report_timestamp': datetime.now().isoformat(),
        'input_parameters': {
            'data_path': str(data_path),
            'ps_column': ps_col,
            'treatment_column': treatment_col,
            'alpha_threshold': alpha
        },
        'original_assessment': original_assessment,
        'trimming_analysis': {
            'trimming_recommended': trimming_recommended,
            'trimming_criteria': {
                'extreme_ps_pct_threshold': 5.0,
                'overlap_required': True,
                'min_ess_ratio': 0.8
            },
            'trimmed_assessment': trimmed_assessment
        },
        'diagnostic_plots': {
            'original_overlap_plot': str(overlap_plot_path),
            'trimmed_overlap_plot': str(output_dir / "ps_overlap_after_trimming.png") if trimming_recommended else None
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if original_assessment['extreme_ps_assessment']['pct_extreme'] > 10:
        comprehensive_report['recommendations'].append(
            f"HIGH PRIORITY: {original_assessment['extreme_ps_assessment']['pct_extreme']:.1f}% observations have extreme PS. Consider trimming."
        )
    
    if not original_assessment['overlap_assessment']['has_overlap']:
        comprehensive_report['recommendations'].append(
            "CRITICAL: No overlap between treatment groups. Causal inference not reliable."
        )
    
    if original_assessment['common_support_ess']['ess_ratio'] < 0.5:
        comprehensive_report['recommendations'].append(
            f"WARNING: Low effective sample size ratio ({original_assessment['common_support_ess']['ess_ratio']:.2f}). Limited common support."
        )
    
    if original_assessment['distribution_tests']['distributions_different']:
        comprehensive_report['recommendations'].append(
            f"INFO: PS distributions significantly different (KS p={original_assessment['distribution_tests']['ks_pvalue']:.4f}). Expected for good PS model."
        )
    
    # Save comprehensive report
    report_path = output_dir / "ps_overlap_comprehensive_report.json"
    with open(report_path, 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    
    logger.info(f"Comprehensive overlap report saved to {report_path}")
    
    return comprehensive_report


def main():
    """
    Command-line interface for propensity score overlap assessment.
    
    Following CLAUDE.md requirement for functions ≤50 lines.
    """
    parser = argparse.ArgumentParser(
        description="Assess propensity score overlap and common support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-file', default='data_derived/ps_weighted.parquet',
                       help='Path to data file with propensity scores')
    parser.add_argument('--ps-col', default='propensity_score',
                       help='Propensity score column name')
    parser.add_argument('--treatment-col', default='ssd_flag',
                       help='Treatment column name')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Extreme PS threshold (Crump et al. rule)')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for reports and plots')
    
    args = parser.parse_args()
    
    try:
        # Generate comprehensive overlap assessment
        report = generate_overlap_assessment_report(
            data_path=Path(args.data_file),
            ps_col=args.ps_col,
            treatment_col=args.treatment_col,
            alpha=args.alpha,
            output_dir=Path(args.output_dir)
        )
        
        # Print summary
        original = report['original_assessment']
        print("\n=== Propensity Score Overlap Assessment ===")
        print(f"Dataset: {args.data_file}")
        print(f"Observations: {original['basic_statistics']['n_total']:,}")
        print(f"Treated: {original['basic_statistics']['n_treated']:,}")
        print(f"Control: {original['basic_statistics']['n_control']:,}")
        
        print(f"\nExtreme PS Assessment (α={args.alpha}):")
        extreme = original['extreme_ps_assessment']
        print(f"  Extreme PS: {extreme['n_extreme_total']} ({extreme['pct_extreme']:.1f}%)")
        print(f"  Low PS (<{extreme['lower_threshold']:.3f}): {extreme['n_extreme_low']}")
        print(f"  High PS (>{extreme['upper_threshold']:.3f}): {extreme['n_extreme_high']}")
        
        print(f"\nOverlap Assessment:")
        overlap = original['overlap_assessment']
        print(f"  Has overlap: {overlap['has_overlap']}")
        if overlap['has_overlap']:
            print(f"  Overlap range: ({overlap['overlap_range'][0]:.3f}, {overlap['overlap_range'][1]:.3f})")
            print(f"  Overlap width: {overlap['pct_overlap_width']:.1f}% of PS range")
        
        print(f"\nCommon Support ESS:")
        ess = original['common_support_ess']
        print(f"  Effective sample size: {ess['effective_sample_size']:,}")
        print(f"  ESS ratio: {ess['ess_ratio']:.3f}")
        
        print(f"\nDistribution Tests:")
        dist = original['distribution_tests']
        print(f"  KS statistic: {dist['ks_statistic']:.4f}")
        print(f"  KS p-value: {dist['ks_pvalue']:.4f}")
        print(f"  Distributions different: {dist['distributions_different']}")
        
        if report['trimming_analysis']['trimming_recommended']:
            print(f"\nTrimming Analysis:")
            print(f"  Trimming recommended: YES")
            trimmed = report['trimming_analysis']['trimmed_assessment']
            if trimmed:
                print(f"  After trimming ESS ratio: {trimmed['common_support_ess']['ess_ratio']:.3f}")
                print(f"  After trimming extreme PS: {trimmed['extreme_ps_assessment']['pct_extreme']:.1f}%")
        
        print(f"\nRecommendations: {len(report['recommendations'])}")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nDiagnostic plots generated:")
        for plot_type, path in report['diagnostic_plots'].items():
            if path:
                print(f"  {plot_type}: {path}")
        print("==========================================\n")
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        raise


if __name__ == "__main__":
    main()