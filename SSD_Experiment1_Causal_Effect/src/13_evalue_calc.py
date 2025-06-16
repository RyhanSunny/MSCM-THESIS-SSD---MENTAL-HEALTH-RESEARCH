#!/usr/bin/env python3
"""
13_evalue_calc.py - E-value calculation for unmeasured confounding

Calculates E-values to assess the robustness of causal estimates to
potential unmeasured confounding. E-value represents the minimum strength
of association that an unmeasured confounder would need to have with both
treatment and outcome to explain away the observed effect.

Hypothesis Support:
- H1: Healthcare utilization - Robustness to unmeasured confounders
- H2: Healthcare costs - Sensitivity analysis for cost estimates
- H3: Medication use - Unmeasured prescribing factors

Output:
- results/evalue_results.json: E-values for all estimates
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import load_config
from src.artefact_tracker import ArtefactTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_evalue(rr, ci_lower=None):
    """
    Calculate E-value for a risk ratio
    
    Parameters:
    -----------
    rr : float
        Risk ratio (or approximation from other effect measures)
    ci_lower : float, optional
        Lower bound of confidence interval
    
    Returns:
    --------
    e_value : float
        E-value for point estimate
    e_value_ci : float or None
        E-value for confidence interval
    """
    if rr < 1:
        # For protective effects, take reciprocal
        rr = 1 / rr
        if ci_lower is not None:
            ci_lower = 1 / ci_lower
    
    # E-value formula
    e_value = rr + np.sqrt(rr * (rr - 1))
    
    # E-value for CI
    e_value_ci = None
    if ci_lower is not None and ci_lower > 1:
        e_value_ci = ci_lower + np.sqrt(ci_lower * (ci_lower - 1))
    
    return e_value, e_value_ci

def convert_to_rr(estimate, effect_type='difference', baseline_rate=None):
    """
    Convert different effect measures to approximate risk ratio
    
    Parameters:
    -----------
    estimate : dict
        Effect estimate with 'estimate' and optionally 'ci_lower'
    effect_type : str
        Type of effect measure ('difference', 'odds_ratio', 'hazard_ratio')
    baseline_rate : float
        Baseline outcome rate for conversion
    
    Returns:
    --------
    rr : float
        Approximate risk ratio
    rr_lower : float or None
        Lower CI as risk ratio
    """
    if effect_type == 'hazard_ratio':
        # HR approximates RR for rare outcomes
        return estimate['estimate'], estimate.get('ci_lower')
    
    elif effect_type == 'odds_ratio':
        # OR approximates RR for rare outcomes
        return estimate['estimate'], estimate.get('ci_lower')
    
    elif effect_type == 'difference':
        if baseline_rate is None:
            # Assume a reasonable baseline rate
            baseline_rate = 10.0  # e.g., 10 events per unit
            logger.warning(f"No baseline rate provided, assuming {baseline_rate}")
        
        # Convert absolute difference to RR
        # RR = (baseline + difference) / baseline
        rr = 1 + estimate['estimate'] / baseline_rate
        
        rr_lower = None
        if 'ci_lower' in estimate:
            rr_lower = 1 + estimate['ci_lower'] / baseline_rate
        
        return rr, rr_lower
    
    else:
        raise ValueError(f"Unknown effect type: {effect_type}")

def calculate_observed_covariate_evalues(df, outcome_col, treatment_col, covariate_cols):
    """
    Calculate E-values for observed covariates to benchmark against
    """
    logger.info("Calculating E-values for observed covariates")
    
    covariate_evalues = {}
    
    for covar in covariate_cols[:10]:  # Limit to first 10 for efficiency
        if covar in df.columns:
            try:
                # Simple logistic regression for covariate association
                import statsmodels.api as sm
                
                # Association with treatment
                X_treatment = sm.add_constant(df[covar])
                y_treatment = df[treatment_col]
                model_treatment = sm.Logit(y_treatment, X_treatment).fit(disp=0)
                or_treatment = np.exp(model_treatment.params[1])
                
                # Association with outcome (controlling for treatment)
                X_outcome = sm.add_constant(df[[covar, treatment_col]])
                y_outcome = df[outcome_col]
                
                # Use appropriate model based on outcome type
                if y_outcome.dtype == bool or set(y_outcome.unique()) == {0, 1}:
                    model_outcome = sm.Logit(y_outcome, X_outcome).fit(disp=0)
                else:
                    model_outcome = sm.GLM(y_outcome, X_outcome, 
                                         family=sm.families.Poisson()).fit()
                
                or_outcome = np.exp(model_outcome.params[1])
                
                # Combined E-value for this covariate
                combined_association = or_treatment * or_outcome
                covar_evalue, _ = calculate_evalue(combined_association)
                
                covariate_evalues[covar] = {
                    'or_treatment': float(or_treatment),
                    'or_outcome': float(or_outcome),
                    'combined_association': float(combined_association),
                    'evalue': float(covar_evalue)
                }
                
            except Exception as e:
                logger.warning(f"Failed to calculate E-value for {covar}: {e}")
    
    return covariate_evalues

def main():
    parser = argparse.ArgumentParser(
        description="Calculate E-values for sensitivity to unmeasured confounding"
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without saving outputs')
    parser.add_argument('--treatment-col', default='ssd_flag', help='Treatment column name (default: ssd_flag)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Initialize tracker
    tracker = ArtefactTracker()
    tracker.track("script_start", {"script": "13_evalue_calc.py"})
    
    # Load ATE results
    ate_path = Path("results/ate_estimates.json")
    if not ate_path.exists():
        logger.error(f"ATE estimates not found at {ate_path}")
        logger.error("Please run 06_causal_estimators.py first")
        return
    
    with open(ate_path) as f:
        ate_results = json.load(f)
    
    # Load data for covariate E-values
    data_path = Path("data_derived/ps_weighted.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        
        # Get baseline outcome rate
        outcome_col = ate_results.get('outcome', 'total_encounters')
        treatment_col = args.treatment_col
        baseline_rate = df[df[treatment_col] == 0][outcome_col].mean()
        logger.info(f"Baseline {outcome_col} rate: {baseline_rate:.2f}")
    else:
        df = None
        # Use config-defined baseline rate or calculate from cohort report
        baseline_rate = config.get('analysis', {}).get('baseline_encounter_rate', None)
        if baseline_rate is None:
            logger.warning("No baseline rate available, results will be limited")
            baseline_rate = None
    
    # Calculate E-values for each estimate
    evalue_results = {
        'outcome': ate_results.get('outcome'),
        'baseline_rate': baseline_rate,
        'evalues_by_method': {},
        'timestamp': datetime.now().isoformat()
    }
    
    for estimate in ate_results.get('estimates', []):
        method = estimate['method']
        
        # Convert to RR
        rr, rr_lower = convert_to_rr(
            estimate, 
            effect_type='difference',
            baseline_rate=baseline_rate
        )
        
        # Calculate E-values
        e_value, e_value_ci = calculate_evalue(rr, rr_lower)
        
        evalue_results['evalues_by_method'][method] = {
            'effect_estimate': estimate['estimate'],
            'ci_lower': estimate.get('ci_lower'),
            'ci_upper': estimate.get('ci_upper'),
            'rr_approximation': float(rr),
            'global_evalue': float(e_value),
            'ci_evalue': float(e_value_ci) if e_value_ci else None
        }
        
        logger.info(f"{method} - E-value: {e_value:.2f}, CI E-value: {e_value_ci:.2f if e_value_ci else 'N/A'}")
    
    # Calculate E-values for observed covariates as benchmark
    if df is not None:
        covariate_cols = [col for col in df.columns if col.endswith('_conf') or 
                         col in ['age', 'sex_M', 'charlson_score']]
        covariate_cols = [col for col in covariate_cols if col in df.columns][:10]
        
        observed_evalues = calculate_observed_covariate_evalues(
            df, outcome_col, treatment_col, covariate_cols
        )
        
        evalue_results['observed_covariate_evalues'] = observed_evalues
        
        # Find maximum observed E-value
        if observed_evalues:
            max_observed = max(cov['evalue'] for cov in observed_evalues.values())
            evalue_results['max_observed_evalue'] = float(max_observed)
            logger.info(f"Maximum observed covariate E-value: {max_observed:.2f}")
    
    # Determine global E-value (most conservative)
    all_evalues = [
        res['global_evalue'] 
        for res in evalue_results['evalues_by_method'].values()
    ]
    if all_evalues:
        global_evalue = min(all_evalues)  # Most conservative
        evalue_results['global_evalue'] = float(global_evalue)
    
    # Save results
    if not args.dry_run:
        output_path = Path("results/evalue_results.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(evalue_results, f, indent=2)
        logger.info(f"Saved E-value results to {output_path}")
        
        # Track outputs
        tracker.track("output_generated", {
            "file": str(output_path),
            "global_evalue": global_evalue if 'global_evalue' in locals() else None
        })
        
        # Update study documentation
        if Path("scripts/update_study_doc.py").exists():
            import subprocess
            subprocess.run([
                "python", "scripts/update_study_doc.py",
                "--step", "E-value calculation complete",
                "--kv", f"global_evalue={global_evalue:.2f}" if 'global_evalue' in locals() else "global_evalue=NA"
            ])
    
    # Print summary
    print("\n=== E-Value Sensitivity Analysis ===")
    print(f"Outcome: {evalue_results.get('outcome')}")
    print(f"Baseline rate: {baseline_rate:.2f}")
    print("\nE-values by method:")
    for method, results in evalue_results['evalues_by_method'].items():
        print(f"  {method}:")
        print(f"    Effect: {results['effect_estimate']:.3f}")
        print(f"    E-value: {results['global_evalue']:.2f}")
        print(f"    CI E-value: {results['ci_evalue']:.2f if results['ci_evalue'] else 'N/A'}")
    
    if 'max_observed_evalue' in evalue_results:
        print(f"\nMax observed covariate E-value: {evalue_results['max_observed_evalue']:.2f}")
    
    if 'global_evalue' in evalue_results:
        print(f"\nGlobal E-value (most conservative): {evalue_results['global_evalue']:.2f}")
        print("\nInterpretation: An unmeasured confounder would need to be associated")
        print(f"with both treatment and outcome by a risk ratio of at least {evalue_results['global_evalue']:.2f}")
        print("to explain away the observed effect.")
    print("====================================\n")

if __name__ == "__main__":
    main()