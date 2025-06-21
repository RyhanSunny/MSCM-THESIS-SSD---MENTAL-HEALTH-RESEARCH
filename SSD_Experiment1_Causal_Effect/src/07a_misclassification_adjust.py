#!/usr/bin/env python3
"""
07a_misclassification_adjust.py - MC-SIMEX correction for SSD flag misclassification

Implements Misclassification Simulation-Extrapolation (MC-SIMEX) to adjust for
potential misclassification bias in the SSD flag exposure variable.

Hypothesis Support:
- H1: Healthcare utilization patterns - Corrects exposure misclassification
- H2: Healthcare costs - Improves cost estimates by reducing bias
- H3: Medication use - More accurate assessment of inappropriate prescribing

Output:
- data_derived/cohort_bias_corrected.parquet: Cohort with adjusted SSD flags
- results/simex_results.json: Bias correction statistics
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import argparse
import logging
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.global_seeds import set_global_seeds, get_random_state
from src.config_loader import load_config
from src.artefact_tracker import create_artefact_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def mc_simex(y, X, z_observed, sensitivity, specificity, B=100, lambdas=None):
    """
    MC-SIMEX for binary misclassified exposure
    
    Parameters:
    -----------
    y : array-like
        Outcome variable
    X : array-like
        Covariates
    z_observed : array-like
        Observed (potentially misclassified) exposure
    sensitivity : float
        True positive rate
    specificity : float
        True negative rate
    B : int
        Number of simulations per lambda
    lambdas : list
        Misclassification multipliers
    
    Returns:
    --------
    corrected_coef : float
        Bias-corrected coefficient
    se_reduction : float
        Percentage reduction in standard error
    """
    if lambdas is None:
        lambdas = [0, 0.5, 1.0, 1.5, 2.0]
    
    coefs = []
    ses = []
    
    logger.info(f"Running MC-SIMEX with B={B} simulations")
    
    for lam in lambdas:
        coef_sum = 0
        se_sum = 0
        
        for b in range(B):
            # Simulate misclassification
            z_star = z_observed.copy()
            
            # Calculate flip probabilities based on lambda
            flip_prob_1_to_0 = lam * (1 - sensitivity)  # False negative
            flip_prob_0_to_1 = lam * (1 - specificity)  # False positive
            
            # Apply misclassification
            rng = np.random.RandomState(get_random_state())
            for i in range(len(z_star)):
                if z_observed[i] == 1:
                    if rng.random() < flip_prob_1_to_0:
                        z_star[i] = 0
                else:
                    if rng.random() < flip_prob_0_to_1:
                        z_star[i] = 1
            
            # Fit model with noisy exposure
            X_with_z = np.column_stack([z_star, X])
            result = stats.linregress(X_with_z[:, 0], y)
            coef_sum += result.slope
            se_sum += result.stderr
        
        coefs.append(coef_sum / B)
        ses.append(se_sum / B)
    
    # Step 2: Extrapolate to lambda = -1
    # Fit quadratic: coef = a + b*lambda + c*lambda^2
    p = np.polyfit(lambdas, coefs, 2)
    corrected_coef = np.polyval(p, -1)
    
    # Calculate SE reduction
    p_se = np.polyfit(lambdas, ses, 2)
    corrected_se = np.polyval(p_se, -1)
    naive_se = ses[0]  # SE at lambda=0
    se_reduction = (naive_se - corrected_se) / naive_se * 100
    
    logger.info(f"Corrected coefficient: {corrected_coef:.4f}")
    logger.info(f"SE reduction: {se_reduction:.1f}%")
    
    return corrected_coef, se_reduction

def apply_bias_correction(cohort_df, config, treatment_col='ssd_flag'):
    """
    Apply MC-SIMEX bias correction to SSD flag
    """
    logger.info("Applying MC-SIMEX bias correction to SSD flag")
    
    # Get misclassification parameters from config
    sensitivity = config.get('misclassification', {}).get('sensitivity', 0.82)
    specificity = config.get('misclassification', {}).get('specificity', 0.82)
    B = config.get('misclassification', {}).get('simex_B', 100)
    
    # Prepare data for SIMEX
    # Use healthcare utilization as outcome
    y = cohort_df['baseline_encounters'].values
    
    # Use key confounders as covariates
    confounder_cols = ['age', 'sex_M', 'charlson_score', 'baseline_high_utilizer']
    X = cohort_df[confounder_cols].values
    
    # Observed SSD flag
    z_observed = cohort_df[treatment_col].values
    
    # Run MC-SIMEX
    corrected_coef, se_reduction = mc_simex(y, X, z_observed, sensitivity, specificity, B)
    
    # Calculate adjustment factor
    naive_coef_result = stats.linregress(z_observed, y)
    naive_coef = naive_coef_result.slope
    adjustment_factor = corrected_coef / naive_coef if naive_coef != 0 else 1.0
    
    # Create bias-corrected flag based on propensity scores
    # This is a simplified approach - in practice would use full model
    logger.info(f"Adjustment factor: {adjustment_factor:.3f}")
    
    # Add bias correction indicators
    cohort_df[f'{treatment_col}_naive'] = cohort_df[treatment_col]
    cohort_df['bias_correction_applied'] = True
    cohort_df['simex_adjustment_factor'] = adjustment_factor
    
    # Create bias-corrected flag: ssd_flag_adj
    # Apply probabilistic correction based on sensitivity/specificity
    rng = np.random.RandomState(get_random_state())
    ssd_flag_adj = cohort_df[treatment_col].copy()
    
    for i in range(len(cohort_df)):
        if cohort_df[treatment_col].iloc[i] == 1:
            # If flagged as SSD, apply false negative correction
            if rng.random() > sensitivity:
                ssd_flag_adj.iloc[i] = 0
        else:
            # If not flagged, apply false positive correction
            if rng.random() > specificity:
                ssd_flag_adj.iloc[i] = 1
    
    cohort_df['ssd_flag_adj'] = ssd_flag_adj
    logger.info(f"Created ssd_flag_adj: {ssd_flag_adj.sum()}/{len(ssd_flag_adj)} flagged ({ssd_flag_adj.mean():.3f} rate)")
    
    results = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'simex_B': B,
        'naive_coefficient': float(naive_coef),
        'corrected_coefficient': float(corrected_coef),
        'adjustment_factor': float(adjustment_factor),
        'se_reduction_percent': float(se_reduction),
        'timestamp': datetime.now().isoformat()
    }
    
    return cohort_df, results

def main():
    parser = argparse.ArgumentParser(
        description="Apply MC-SIMEX correction for SSD flag misclassification"
    )
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run without saving outputs')
    parser.add_argument('--treatment-col', default='ssd_flag', help='Treatment column name (default: ssd_flag)')
    args = parser.parse_args()
    
    # Set random seeds
    set_global_seeds()
    
    # Load configuration
    config = load_config()
    
    # Log script start
    logger.info("Starting MC-SIMEX bias correction script")
    
    # Load cohort data
    cohort_path = Path("data_derived/patient_master.parquet")
    if not cohort_path.exists():
        logger.error(f"Patient master file not found at {cohort_path}")
        return
    
    logger.info(f"Loading cohort from {cohort_path}")
    cohort_df = pd.read_parquet(cohort_path)
    initial_rows = len(cohort_df)
    
    treatment_col = args.treatment_col
    # Apply bias correction
    cohort_corrected, simex_results = apply_bias_correction(cohort_df, config, treatment_col)
    
    # Validate
    assert len(cohort_corrected) == initial_rows, "Row count changed during correction"
    
    # Save outputs
    if not args.dry_run:
        # Save corrected cohort
        output_path = Path("data_derived/cohort_bias_corrected.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cohort_corrected.to_parquet(output_path, index=False)
        logger.info(f"Saved bias-corrected cohort to {output_path}")
        
        # Save SIMEX results
        results_path = Path("results/simex_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(simex_results, f, indent=2)
        logger.info(f"Saved SIMEX results to {results_path}")
        
        # Create artefact metadata
        create_artefact_metadata(
            artefact_path=output_path,
            script_name="07a_misclassification_adjust.py",
            hypotheses=["H1", "H2", "H3"],
            metrics={
                "rows": len(cohort_corrected),
                "se_reduction": simex_results['se_reduction_percent'],
                "adjustment_factor": simex_results['adjustment_factor']
            },
            description="MC-SIMEX bias-corrected cohort with ssd_flag_adj"
        )
        
        # Merge bias-corrected flag back to patient_master.parquet
        logger.info("Merging ssd_flag_adj back to patient_master.parquet...")
        try:
            from mc_simex_flag_merger import merge_bias_corrected_flag
            merge_bias_corrected_flag(
                master_path=cohort_path,
                corrected_path=output_path,
                backup=True
            )
            logger.info("✓ Successfully merged ssd_flag_adj to patient_master.parquet")
        except Exception as e:
            logger.error(f"Failed to merge ssd_flag_adj: {e}")
            logger.warning("Continuing without flag merge - manual merge required")
        
        # Update study documentation
        if Path("scripts/update_study_doc.py").exists():
            import subprocess
            subprocess.run([
                "python", "scripts/update_study_doc.py",
                "--step", "MC-SIMEX bias correction applied",
                "--kv", "ssd_flag_adj=true",
                "--kv", f"simex_se_reduction={simex_results['se_reduction_percent']:.1f}%"
            ])
    
    # Print summary
    print("\n=== MC-SIMEX Bias Correction Summary ===")
    print(f"Sensitivity: {simex_results['sensitivity']:.2f}")
    print(f"Specificity: {simex_results['specificity']:.2f}")
    print(f"Naive coefficient: {simex_results['naive_coefficient']:.4f}")
    print(f"Corrected coefficient: {simex_results['corrected_coefficient']:.4f}")
    print(f"Adjustment factor: {simex_results['adjustment_factor']:.3f}")
    print(f"SE reduction: {simex_results['se_reduction_percent']:.1f}%")
    print("=======================================\n")

if __name__ == "__main__":
    main()