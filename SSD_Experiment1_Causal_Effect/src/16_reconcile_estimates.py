#!/usr/bin/env python3
"""
16_reconcile_estimates.py - Estimate reconciliation rule for Week 5

Compares TMLE, DML, and Causal-Forest ATEs and flags discordant estimates
(>15% difference) as specified in MAX-EVAL §1.14.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import yaml
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def calculate_percentage_difference(baseline: float, comparison: float) -> float:
    """
    Calculate percentage difference between two estimates
    
    Parameters:
    -----------
    baseline : float
        Baseline estimate (denominator)
    comparison : float
        Comparison estimate (numerator)
        
    Returns:
    --------
    float
        Percentage difference
        
    Raises:
    -------
    ValueError
        If baseline is zero
    """
    if baseline == 0.0:
        raise ValueError("Cannot calculate percentage difference with zero baseline")
    
    return abs((comparison - baseline) / baseline) * 100


def compare_ate_estimates(estimates: Dict[str, Dict[str, float]], 
                         threshold: float = 15.0) -> Dict[str, Any]:
    """
    Compare ATE estimates across methods
    
    Parameters:
    -----------
    estimates : Dict[str, Dict[str, float]]
        Dictionary with method names as keys and estimate dicts as values
    threshold : float
        Threshold for flagging discordant estimates (default 15%)
        
    Returns:
    --------
    Dict[str, Any]
        Comparison results with status and pairwise comparisons
    """
    logger.info(f"Comparing ATE estimates across {len(estimates)} methods...")
    
    methods = list(estimates.keys())
    comparisons = []
    max_difference = 0.0
    
    # Pairwise comparisons
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            ate1 = estimates[method1]['ate']
            ate2 = estimates[method2]['ate']
            
            pct_diff = calculate_percentage_difference(ate1, ate2)
            
            comparison = {
                'method1': method1,
                'method2': method2,
                'ate1': ate1,
                'ate2': ate2,
                'percentage_diff': pct_diff,
                'exceeds_threshold': pct_diff > threshold
            }
            
            comparisons.append(comparison)
            max_difference = max(max_difference, pct_diff)
    
    # Determine overall status
    has_discordant = any(comp['exceeds_threshold'] for comp in comparisons)
    status = 'discordant' if has_discordant else 'concordant'
    
    result = {
        'status': status,
        'max_difference': max_difference,
        'threshold': threshold,
        'comparisons': comparisons,
        'has_discordant_pairs': has_discordant,
        'n_methods': len(methods),
        'n_comparisons': len(comparisons)
    }
    
    logger.info(f"ATE comparison complete. Status: {status}, Max difference: {max_difference:.2f}%")
    return result


def flag_discordant_estimates(comparison_result: Dict[str, Any], 
                             threshold: float = 15.0) -> Dict[str, Any]:
    """
    Flag discordant estimates based on comparison results
    
    Parameters:
    -----------
    comparison_result : Dict[str, Any]
        Results from compare_ate_estimates
    threshold : float
        Threshold for flagging (default 15%)
        
    Returns:
    --------
    Dict[str, Any]
        Flagging results
    """
    logger.info("Flagging discordant estimates...")
    
    discordant_pairs = [
        comp for comp in comparison_result['comparisons'] 
        if comp['percentage_diff'] > threshold
    ]
    
    flagged = len(discordant_pairs) > 0
    
    result = {
        'flagged': flagged,
        'threshold': threshold,
        'n_discordant_pairs': len(discordant_pairs),
        'discordant_pairs': discordant_pairs,
        'max_difference': comparison_result['max_difference']
    }
    
    if flagged:
        logger.warning(f"FLAGGED: {len(discordant_pairs)} discordant estimate pairs detected")
        for pair in discordant_pairs:
            logger.warning(f"  {pair['method1']} vs {pair['method2']}: {pair['percentage_diff']:.2f}% difference")
    else:
        logger.info("All estimates within acceptable range")
    
    return result


def load_estimate_results(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Load estimate results from YAML files
    
    Parameters:
    -----------
    results_dir : Path
        Directory containing estimate YAML files
        
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary of estimates by method
    """
    logger.info(f"Loading estimate results from: {results_dir}")
    
    estimates = {}
    
    # Expected file mappings
    file_mappings = {
        'tmle': ['tmle_results.yaml', 'tmle.yaml'],
        'dml': ['dml_results.yaml', 'dml.yaml'],
        'causal_forest': ['causal_forest_results.yaml', 'causal_forest.yaml', 'cf_results.yaml']
    }
    
    for method, filenames in file_mappings.items():
        loaded = False
        for filename in filenames:
            filepath = results_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        data = yaml.safe_load(f)
                    estimates[method] = data
                    logger.info(f"  Loaded {method} from {filename}")
                    loaded = True
                    break
                except Exception as e:
                    logger.warning(f"  Failed to load {filename}: {e}")
        
        if not loaded:
            logger.warning(f"  No results file found for {method}")
    
    logger.info(f"Loaded {len(estimates)} estimate sets")
    return estimates


def create_reconciliation_report(reconciliation_results: Dict[str, Any], 
                               output_dir: Path) -> Path:
    """
    Create reconciliation report in markdown format
    
    Parameters:
    -----------
    reconciliation_results : Dict[str, Any]
        Complete reconciliation results
    output_dir : Path
        Output directory for report
        
    Returns:
    --------
    Path
        Path to generated report
    """
    logger.info("Creating reconciliation report...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status = reconciliation_results['reconciliation_status']
    
    report_content = f"""# Estimate Reconciliation Report

Generated: {timestamp}
Status: **{status}**

## Executive Summary

This report compares Average Treatment Effect (ATE) estimates across three causal inference methods:
- TMLE (Targeted Maximum Likelihood Estimation)
- DML (Double Machine Learning) 
- Causal Forest

**Reconciliation Status:** {status}
**Maximum Difference:** {reconciliation_results['max_difference']:.2f}%
**Flagged:** {'YES' if reconciliation_results['flagged'] else 'NO'}

## Method Estimates

"""
    
    # Add estimate table
    estimates = reconciliation_results['estimates']
    report_content += "| Method | ATE | 95% CI | \n"
    report_content += "|--------|-----|--------|\n"
    
    for method, data in estimates.items():
        method_name = method.upper().replace('_', ' ')
        ate = data['ate']
        ci_lower = data.get('ci_lower', 'N/A')
        ci_upper = data.get('ci_upper', 'N/A')
        
        if ci_lower != 'N/A' and ci_upper != 'N/A':
            ci_text = f"({ci_lower:.3f}, {ci_upper:.3f})"
        else:
            ci_text = "N/A"
        
        report_content += f"| {method_name} | {ate:.3f} | {ci_text} |\n"
    
    # Add comparison details
    report_content += f"""
## Pairwise Comparisons

**Threshold:** {reconciliation_results.get('threshold', 15.0)}%

"""
    
    comparisons = reconciliation_results['comparison_results']['comparisons']
    report_content += "| Method 1 | Method 2 | Difference (%) | Exceeds Threshold |\n"
    report_content += "|----------|----------|----------------|-------------------|\n"
    
    for comp in comparisons:
        method1 = comp['method1'].upper().replace('_', ' ')
        method2 = comp['method2'].upper().replace('_', ' ')
        diff = comp['percentage_diff']
        exceeds = "⚠️ YES" if comp['exceeds_threshold'] else "✅ NO"
        
        report_content += f"| {method1} | {method2} | {diff:.2f}% | {exceeds} |\n"
    
    # Add interpretation
    if reconciliation_results['flagged']:
        report_content += """
## ⚠️ Action Required

**Discordant estimates detected.** Estimates differ by more than 15%, indicating:
1. Potential model misspecification
2. Different identification assumptions
3. Need for sensitivity analysis
4. Method-specific biases

**Recommendations:**
- Review model specifications
- Check covariate balance across methods
- Investigate overlap assumptions
- Consider ensemble averaging
"""
    else:
        report_content += """
## ✅ Reconciliation Passed

All estimates are within the acceptable 15% threshold, indicating:
- Consistent causal identification
- Robust findings across methods
- Appropriate model specifications

**Conclusion:** Estimates are concordant and suitable for reporting.
"""
    
    # Add technical details
    report_content += f"""
## Technical Details

- **Analysis Date:** {timestamp}
- **Methods Compared:** {', '.join(estimates.keys())}
- **Comparison Threshold:** {reconciliation_results.get('threshold', 15.0)}%
- **Total Comparisons:** {len(comparisons)}
- **Flagged Pairs:** {sum(1 for c in comparisons if c['exceeds_threshold'])}

---
*Generated by SSD Experiment 1 Causal Effect Pipeline v4.0.0*
"""
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'estimate_reconciliation_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Reconciliation report saved: {report_path}")
    return report_path


def reconcile_causal_estimates(results_dir: Path, 
                              output_dir: Optional[Path] = None,
                              threshold: float = 15.0) -> Dict[str, Any]:
    """
    Main reconciliation function for causal estimates
    
    Parameters:
    -----------
    results_dir : Path
        Directory containing estimate YAML files
    output_dir : Optional[Path]
        Output directory for reports (defaults to results_dir)
    threshold : float
        Threshold for flagging discordant estimates (default 15%)
        
    Returns:
    --------
    Dict[str, Any]
        Complete reconciliation results
        
    Raises:
    -------
    AssertionError
        If discordant estimates are detected (for CI failure)
    """
    logger.info("Starting causal estimate reconciliation...")
    
    if output_dir is None:
        output_dir = results_dir
    
    # Load estimates
    estimates = load_estimate_results(results_dir)
    
    if len(estimates) < 2:
        logger.warning(f"Only {len(estimates)} methods found. Need at least 2 for comparison.")
        return {
            'reconciliation_status': 'INSUFFICIENT_DATA',
            'flagged': False,
            'estimates': estimates,
            'message': f'Only {len(estimates)} methods available for comparison'
        }
    
    # Compare estimates
    comparison_results = compare_ate_estimates(estimates, threshold)
    
    # Flag discordant estimates
    flagging_results = flag_discordant_estimates(comparison_results, threshold)
    
    # Combine results
    reconciliation_results = {
        'reconciliation_status': 'FAIL' if flagging_results['flagged'] else 'PASS',
        'flagged': flagging_results['flagged'],
        'max_difference': comparison_results['max_difference'],
        'threshold': threshold,
        'estimates': estimates,
        'comparison_results': comparison_results,
        'flagging_results': flagging_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Create report
    report_path = create_reconciliation_report(reconciliation_results, output_dir)
    reconciliation_results['report_path'] = str(report_path)
    
    # Log final status
    status = reconciliation_results['reconciliation_status']
    logger.info(f"Reconciliation complete. Status: {status}")
    
    if status == 'FAIL':
        error_msg = (f"Discordant estimates detected! Max difference: "
                    f"{reconciliation_results['max_difference']:.2f}% > {threshold}%")
        logger.error(error_msg)
        
        # Raise assertion error for pytest/CI compatibility
        raise AssertionError(error_msg)
    
    return reconciliation_results


def main():
    """Main execution for estimate reconciliation"""
    logger.info("Estimate reconciliation module ready")
    
    print("Estimate Reconciliation Functions:")
    print("  - reconcile_causal_estimates() - Main reconciliation workflow")
    print("  - compare_ate_estimates() - Compare ATE estimates")
    print("  - flag_discordant_estimates() - Flag >15% differences")
    print("  - load_estimate_results() - Load from YAML files")
    print("  - create_reconciliation_report() - Generate markdown report")
    print("")
    print("Usage:")
    print("  results = reconcile_causal_estimates(Path('results/'))")
    print("  if results['reconciliation_status'] == 'FAIL':")
    print("      print('Discordant estimates detected!')")


if __name__ == "__main__":
    main()