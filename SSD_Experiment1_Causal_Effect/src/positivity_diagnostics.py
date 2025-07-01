#!/usr/bin/env python3
"""
positivity_diagnostics.py - Document positivity violations and common support

Analyzes and documents propensity score overlap, positivity violations,
and weight trimming following Crump et al. (2009) rules.

Following CLAUDE.md requirements:
- TDD approach with comprehensive error handling
- Functions ≤50 lines
- Evidence-based implementation
- Version numbering and timestamps

Author: Ryhan Suny (Toronto Metropolitan University)
Version: 1.0
Date: 2025-07-01
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_positivity_violations(
    ps_data_path: Path = Path("data_derived/ps_matched.parquet"),
    output_dir: Path = Path("results")
) -> Dict:
    """
    Analyze positivity violations and common support region.
    
    Parameters:
    -----------
    ps_data_path : Path
        Path to propensity score data
    output_dir : Path
        Output directory for results
        
    Returns:
    --------
    Dict with positivity diagnostics
    """
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().isoformat()
    
    logger.info("Starting positivity violation analysis")
    
    # Load PS data
    try:
        df = pd.read_parquet(ps_data_path)
        logger.info(f"Loaded {len(df):,} patients with propensity scores")
    except FileNotFoundError:
        logger.error(f"PS data not found at {ps_data_path}")
        return {"error": "Data file not found"}
    
    # Separate by treatment
    treated = df[df['ssd_flag'] == 1]
    control = df[df['ssd_flag'] == 0]
    
    # Calculate diagnostics
    diagnostics = {
        "metadata": {
            "timestamp": timestamp,
            "n_total": len(df),
            "n_treated": len(treated),
            "n_control": len(control),
            "version": "1.0"
        }
    }
    
    # Analyze propensity score distributions
    ps_analysis = _analyze_ps_distributions(treated, control)
    diagnostics["propensity_scores"] = ps_analysis
    
    # Check common support
    common_support = _check_common_support(treated, control)
    diagnostics["common_support"] = common_support
    
    # Analyze weight distributions
    if 'iptw' in df.columns or 'weight' in df.columns:
        weight_col = 'iptw' if 'iptw' in df.columns else 'weight'
        weight_analysis = _analyze_weights(df, weight_col)
        diagnostics["weights"] = weight_analysis
    
    # Apply Crump rule and analyze trimming
    trimming_analysis = _apply_crump_trimming(df)
    diagnostics["trimming"] = trimming_analysis
    
    # Calculate effective sample size
    ess_analysis = _calculate_ess(df)
    diagnostics["effective_sample_size"] = ess_analysis
    
    # Generate visualizations
    _create_positivity_plots(df, output_dir)
    
    # Save results
    _save_positivity_results(diagnostics, output_dir)
    
    return diagnostics


def _analyze_ps_distributions(treated: pd.DataFrame, 
                             control: pd.DataFrame) -> Dict:
    """Analyze propensity score distributions (≤50 lines)."""
    ps_col = 'propensity_score'
    
    if ps_col not in treated.columns:
        return {"error": "No propensity score column found"}
    
    analysis = {
        "treated": {
            "mean": float(treated[ps_col].mean()),
            "std": float(treated[ps_col].std()),
            "min": float(treated[ps_col].min()),
            "max": float(treated[ps_col].max()),
            "q25": float(treated[ps_col].quantile(0.25)),
            "median": float(treated[ps_col].median()),
            "q75": float(treated[ps_col].quantile(0.75))
        },
        "control": {
            "mean": float(control[ps_col].mean()),
            "std": float(control[ps_col].std()),
            "min": float(control[ps_col].min()),
            "max": float(control[ps_col].max()),
            "q25": float(control[ps_col].quantile(0.25)),
            "median": float(control[ps_col].median()),
            "q75": float(control[ps_col].quantile(0.75))
        }
    }
    
    # Check for extreme propensity scores
    extreme_threshold = 0.1
    n_extreme_low_treated = (treated[ps_col] < extreme_threshold).sum()
    n_extreme_high_control = (control[ps_col] > (1 - extreme_threshold)).sum()
    
    analysis["extreme_scores"] = {
        "threshold": extreme_threshold,
        "treated_below_threshold": int(n_extreme_low_treated),
        "control_above_threshold": int(n_extreme_high_control),
        "pct_extreme_treated": float(n_extreme_low_treated / len(treated) * 100),
        "pct_extreme_control": float(n_extreme_high_control / len(control) * 100)
    }
    
    return analysis


def _check_common_support(treated: pd.DataFrame, 
                         control: pd.DataFrame) -> Dict:
    """Check common support region (≤50 lines)."""
    ps_col = 'propensity_score'
    
    # Find overlap region
    min_treated = treated[ps_col].min()
    max_treated = treated[ps_col].max()
    min_control = control[ps_col].min()
    max_control = control[ps_col].max()
    
    common_min = max(min_treated, min_control)
    common_max = min(max_treated, max_control)
    
    # Count patients in common support
    treated_in_support = ((treated[ps_col] >= common_min) & 
                         (treated[ps_col] <= common_max)).sum()
    control_in_support = ((control[ps_col] >= common_min) & 
                         (control[ps_col] <= common_max)).sum()
    
    return {
        "region": {
            "min": float(common_min),
            "max": float(common_max),
            "width": float(common_max - common_min)
        },
        "patients_in_support": {
            "treated": int(treated_in_support),
            "control": int(control_in_support),
            "total": int(treated_in_support + control_in_support),
            "pct_treated": float(treated_in_support / len(treated) * 100),
            "pct_control": float(control_in_support / len(control) * 100)
        },
        "positivity_satisfied": common_min < common_max
    }


def _analyze_weights(df: pd.DataFrame, weight_col: str) -> Dict:
    """Analyze weight distributions (≤50 lines)."""
    weights = df[weight_col]
    
    # Basic statistics
    analysis = {
        "column": weight_col,
        "statistics": {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "cv": float(weights.std() / weights.mean())  # Coefficient of variation
        },
        "percentiles": {
            "p1": float(weights.quantile(0.01)),
            "p5": float(weights.quantile(0.05)),
            "p10": float(weights.quantile(0.10)),
            "p90": float(weights.quantile(0.90)),
            "p95": float(weights.quantile(0.95)),
            "p99": float(weights.quantile(0.99))
        }
    }
    
    # Check for extreme weights
    extreme_threshold = 10
    n_extreme = (weights > extreme_threshold).sum()
    
    analysis["extreme_weights"] = {
        "threshold": extreme_threshold,
        "n_above_threshold": int(n_extreme),
        "pct_above_threshold": float(n_extreme / len(df) * 100),
        "max_weight": float(weights.max()),
        "patients_with_max": int((weights == weights.max()).sum())
    }
    
    # Weight stability
    analysis["stability"] = {
        "weight_range": float(weights.max() - weights.min()),
        "coefficient_of_variation": float(weights.std() / weights.mean()),
        "stable": weights.std() / weights.mean() < 2  # CV < 2 is stable
    }
    
    return analysis


def _apply_crump_trimming(df: pd.DataFrame) -> Dict:
    """Apply Crump et al. (2009) trimming rule (≤50 lines)."""
    ps_col = 'propensity_score'
    
    if ps_col not in df.columns:
        return {"error": "No propensity score column"}
    
    # Crump rule: trim if PS < 0.1 or PS > 0.9
    crump_lower = 0.1
    crump_upper = 0.9
    
    # Identify patients to trim
    trim_mask = (df[ps_col] < crump_lower) | (df[ps_col] > crump_upper)
    n_trimmed = trim_mask.sum()
    
    # Calculate trimming by group
    treated_trimmed = trim_mask & (df['ssd_flag'] == 1)
    control_trimmed = trim_mask & (df['ssd_flag'] == 0)
    
    trimming_results = {
        "crump_thresholds": {
            "lower": crump_lower,
            "upper": crump_upper
        },
        "n_trimmed": {
            "total": int(n_trimmed),
            "treated": int(treated_trimmed.sum()),
            "control": int(control_trimmed.sum()),
            "pct_total": float(n_trimmed / len(df) * 100)
        },
        "remaining_sample": {
            "n": int(len(df) - n_trimmed),
            "pct": float((len(df) - n_trimmed) / len(df) * 100)
        }
    }
    
    # Alternative optimal trimming (Crump et al. 2009)
    # α = 2 * (1/n_treated + 1/n_control)
    n_treated = (df['ssd_flag'] == 1).sum()
    n_control = (df['ssd_flag'] == 0).sum()
    alpha_optimal = 2 * (1/n_treated + 1/n_control)
    
    trimming_results["optimal_trimming"] = {
        "alpha": float(alpha_optimal),
        "lower_bound": float(alpha_optimal/2),
        "upper_bound": float(1 - alpha_optimal/2)
    }
    
    return trimming_results


def _calculate_ess(df: pd.DataFrame) -> Dict:
    """Calculate effective sample size (≤50 lines)."""
    weight_col = 'iptw' if 'iptw' in df.columns else 'weight'
    
    if weight_col not in df.columns:
        # If no weights, ESS = n
        return {
            "ess": len(df),
            "ess_ratio": 1.0,
            "no_weights": True
        }
    
    weights = df[weight_col]
    
    # ESS = (sum of weights)^2 / sum of squared weights
    ess = (weights.sum() ** 2) / (weights ** 2).sum()
    ess_ratio = ess / len(df)
    
    # Calculate by treatment group
    treated = df[df['ssd_flag'] == 1]
    control = df[df['ssd_flag'] == 0]
    
    w_treated = treated[weight_col]
    w_control = control[weight_col]
    
    ess_treated = (w_treated.sum() ** 2) / (w_treated ** 2).sum()
    ess_control = (w_control.sum() ** 2) / (w_control ** 2).sum()
    
    return {
        "overall": {
            "ess": float(ess),
            "n": len(df),
            "ess_ratio": float(ess_ratio),
            "pct_retained": float(ess_ratio * 100)
        },
        "by_treatment": {
            "treated": {
                "ess": float(ess_treated),
                "n": len(treated),
                "ess_ratio": float(ess_treated / len(treated))
            },
            "control": {
                "ess": float(ess_control),
                "n": len(control),
                "ess_ratio": float(ess_control / len(control))
            }
        },
        "sufficient": ess_ratio > 0.8  # ESS > 80% is good
    }


def _create_positivity_plots(df: pd.DataFrame, output_dir: Path):
    """Create positivity diagnostic plots (≤50 lines)."""
    ps_col = 'propensity_score'
    
    if ps_col not in df.columns:
        logger.warning("No propensity scores to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram of PS by treatment
    ax1 = axes[0, 0]
    treated = df[df['ssd_flag'] == 1][ps_col]
    control = df[df['ssd_flag'] == 0][ps_col]
    
    ax1.hist(control, bins=30, alpha=0.5, label='Control', density=True)
    ax1.hist(treated, bins=30, alpha=0.5, label='Treated', density=True)
    ax1.set_xlabel('Propensity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Propensity Score Distributions')
    ax1.legend()
    ax1.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='Crump bounds')
    ax1.axvline(0.9, color='red', linestyle='--', alpha=0.5)
    
    # 2. Box plots
    ax2 = axes[0, 1]
    df_plot = pd.DataFrame({
        'PS': df[ps_col],
        'Treatment': df['ssd_flag'].map({0: 'Control', 1: 'Treated'})
    })
    sns.boxplot(data=df_plot, x='Treatment', y='PS', ax=ax2)
    ax2.set_title('Propensity Score Box Plots')
    ax2.axhline(0.1, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(0.9, color='red', linestyle='--', alpha=0.5)
    
    # 3. Common support visualization
    ax3 = axes[1, 0]
    ax3.scatter(df[ps_col], df['ssd_flag'] + np.random.normal(0, 0.02, len(df)),
                alpha=0.3, s=10)
    ax3.set_xlabel('Propensity Score')
    ax3.set_ylabel('Treatment Status (jittered)')
    ax3.set_title('Common Support Region')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Control', 'Treated'])
    
    # 4. Weight distribution (if available)
    ax4 = axes[1, 1]
    if 'iptw' in df.columns or 'weight' in df.columns:
        weight_col = 'iptw' if 'iptw' in df.columns else 'weight'
        ax4.hist(df[weight_col], bins=50, edgecolor='black')
        ax4.set_xlabel('Weight')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Weight Distribution')
        ax4.axvline(10, color='red', linestyle='--', label='Threshold=10')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No weights available', ha='center', va='center')
        ax4.set_title('Weight Distribution')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'positivity_diagnostics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    logger.info(f"Positivity plots saved to {output_path}")


def _save_positivity_results(diagnostics: Dict, output_dir: Path):
    """Save positivity diagnostic results (≤50 lines)."""
    # Save JSON
    json_path = output_dir / "positivity_diagnostics.json"
    with open(json_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    # Create summary report
    report_lines = [
        "# Positivity Diagnostics Report",
        f"\nGenerated: {diagnostics['metadata']['timestamp']}",
        f"\n## Sample Size",
        f"- Total: {diagnostics['metadata']['n_total']:,}",
        f"- Treated: {diagnostics['metadata']['n_treated']:,}",
        f"- Control: {diagnostics['metadata']['n_control']:,}",
        "\n## Propensity Score Summary"
    ]
    
    if "propensity_scores" in diagnostics:
        ps = diagnostics["propensity_scores"]
        report_lines.extend([
            f"- Treated mean (SD): {ps['treated']['mean']:.3f} ({ps['treated']['std']:.3f})",
            f"- Control mean (SD): {ps['control']['mean']:.3f} ({ps['control']['std']:.3f})",
            f"- Extreme scores (<0.1 or >0.9):",
            f"  - Treated: {ps['extreme_scores']['pct_extreme_treated']:.1f}%",
            f"  - Control: {ps['extreme_scores']['pct_extreme_control']:.1f}%"
        ])
    
    if "common_support" in diagnostics:
        cs = diagnostics["common_support"]
        report_lines.extend([
            "\n## Common Support",
            f"- Region: [{cs['region']['min']:.3f}, {cs['region']['max']:.3f}]",
            f"- Width: {cs['region']['width']:.3f}",
            f"- Patients in support: {cs['patients_in_support']['pct_treated']:.1f}% treated, "
            f"{cs['patients_in_support']['pct_control']:.1f}% control",
            f"- Positivity satisfied: {'Yes' if cs['positivity_satisfied'] else 'No'}"
        ])
    
    if "effective_sample_size" in diagnostics:
        ess = diagnostics["effective_sample_size"]
        report_lines.extend([
            "\n## Effective Sample Size",
            f"- Overall ESS: {ess['overall']['ess']:.0f} ({ess['overall']['pct_retained']:.1f}%)",
            f"- Sufficient (>80%): {'Yes' if ess['sufficient'] else 'No'}"
        ])
    
    # Save report
    report_path = output_dir / "positivity_diagnostics_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Results saved to {json_path} and {report_path}")


if __name__ == "__main__":
    # Run positivity diagnostics
    diagnostics = analyze_positivity_violations()
    
    # Print summary
    print("\n✓ Positivity diagnostics complete")
    if "common_support" in diagnostics:
        cs = diagnostics["common_support"]
        print(f"  - Common support: [{cs['region']['min']:.3f}, {cs['region']['max']:.3f}]")
        print(f"  - Positivity satisfied: {'Yes' if cs['positivity_satisfied'] else 'No'}")
    
    if "effective_sample_size" in diagnostics:
        ess = diagnostics["effective_sample_size"]
        print(f"  - ESS: {ess['overall']['pct_retained']:.1f}% of original sample")
        print(f"  - Sufficient: {'Yes' if ess['sufficient'] else 'No'}")