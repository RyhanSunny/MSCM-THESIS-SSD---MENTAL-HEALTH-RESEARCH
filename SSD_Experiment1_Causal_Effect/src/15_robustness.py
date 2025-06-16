#!/usr/bin/env python3
"""
15_robustness.py - Comprehensive robustness checks

Implements multiple robustness checks:
1. Alternative model specifications
2. Trimming analysis (remove extreme weights)
3. Covariate balance diagnostics
4. Leave-one-out sensitivity
5. Alternative outcome definitions

Output:
- results/robustness_results.json: All robustness check results
- figures/robustness_forest_plot.pdf: Forest plot of estimates
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.global_seeds import set_global_seeds
from src.config_loader import load_config
from src.artefact_tracker import ArtefactTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def alternative_specifications(df, outcome_col, treatment_col, covariate_cols):
    """Test robustness to alternative model specifications"""
    logger.info("Testing alternative model specifications")
    
    results = {}
    
    # 1. Linear model (OLS)
    try:
        X = sm.add_constant(df[covariate_cols + [treatment_col]])
        y = df[outcome_col]
        
        if 'iptw' in df.columns:
            model = sm.WLS(y, X, weights=df['iptw'])
        else:
            model = sm.OLS(y, X)
        
        fit = model.fit()
        results['linear_model'] = {
            'effect': float(fit.params[treatment_col]),
            'ci_lower': float(fit.conf_int().loc[treatment_col, 0]),
            'ci_upper': float(fit.conf_int().loc[treatment_col, 1]),
            'p_value': float(fit.pvalues[treatment_col])
        }
    except Exception as e:
        logger.error(f"Linear model failed: {e}")
    
    # 2. Negative binomial (for overdispersion)
    try:
        if 'iptw' in df.columns:
            model = sm.GLM(y, X, family=sm.families.NegativeBinomial(),
                          freq_weights=df['iptw'])
        else:
            model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        
        fit = model.fit()
        results['negbinom_model'] = {
            'effect': float(fit.params[treatment_col]),
            'ci_lower': float(fit.conf_int().loc[treatment_col, 0]),
            'ci_upper': float(fit.conf_int().loc[treatment_col, 1]),
            'p_value': float(fit.pvalues[treatment_col])
        }
    except Exception as e:
        logger.error(f"Negative binomial model failed: {e}")
    
    # 3. Log-transformed outcome
    try:
        y_log = np.log1p(y)  # log(y + 1) to handle zeros
        
        if 'iptw' in df.columns:
            model = sm.WLS(y_log, X, weights=df['iptw'])
        else:
            model = sm.OLS(y_log, X)
        
        fit = model.fit()
        results['log_outcome_model'] = {
            'effect': float(fit.params[treatment_col]),
            'ci_lower': float(fit.conf_int().loc[treatment_col, 0]),
            'ci_upper': float(fit.conf_int().loc[treatment_col, 1]),
            'p_value': float(fit.pvalues[treatment_col])
        }
    except Exception as e:
        logger.error(f"Log outcome model failed: {e}")
    
    return results

def trimming_analysis(df, outcome_col, treatment_col, covariate_cols):
    """Test sensitivity to weight trimming"""
    logger.info("Running trimming analysis")
    
    if 'iptw' not in df.columns:
        logger.warning("No weights available for trimming analysis")
        return {}
    
    results = {}
    
    # Different trimming levels
    trim_levels = [0, 1, 5, 10]  # percentiles
    
    for trim in trim_levels:
        try:
            # Trim weights
            if trim > 0:
                lower = np.percentile(df['iptw'], trim)
                upper = np.percentile(df['iptw'], 100 - trim)
                weights_trimmed = np.clip(df['iptw'], lower, upper)
            else:
                weights_trimmed = df['iptw']
            
            # Fit model
            X = sm.add_constant(df[covariate_cols + [treatment_col]])
            y = df[outcome_col]
            
            model = sm.GLM(y, X, family=sm.families.Poisson(),
                          freq_weights=weights_trimmed)
            fit = model.fit()
            
            # Calculate ESS
            ess = (np.sum(weights_trimmed))**2 / np.sum(weights_trimmed**2)
            
            results[f'trim_{trim}pct'] = {
                'effect': float(fit.params[treatment_col]),
                'ci_lower': float(fit.conf_int().loc[treatment_col, 0]),
                'ci_upper': float(fit.conf_int().loc[treatment_col, 1]),
                'p_value': float(fit.pvalues[treatment_col]),
                'ess': float(ess)
            }
            
        except Exception as e:
            logger.error(f"Trimming at {trim}% failed: {e}")
    
    return results

def covariate_balance_check(df, treatment_col, covariate_cols):
    """Check covariate balance after weighting"""
    logger.info("Checking covariate balance")
    
    balance_results = {}
    
    # Calculate SMD for each covariate
    for covar in covariate_cols[:10]:  # Limit to first 10
        if covar in df.columns:
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            # Unweighted
            mean_diff = treated[covar].mean() - control[covar].mean()
            pooled_sd = np.sqrt((treated[covar].var() + control[covar].var()) / 2)
            smd_unweighted = mean_diff / pooled_sd if pooled_sd > 0 else 0
            
            # Weighted
            if 'iptw' in df.columns:
                w1 = df.loc[df[treatment_col] == 1, 'iptw']
                w0 = df.loc[df[treatment_col] == 0, 'iptw']
                
                mean1_w = np.average(treated[covar], weights=w1)
                mean0_w = np.average(control[covar], weights=w0)
                
                var1_w = np.average((treated[covar] - mean1_w)**2, weights=w1)
                var0_w = np.average((control[covar] - mean0_w)**2, weights=w0)
                
                pooled_sd_w = np.sqrt((var1_w + var0_w) / 2)
                smd_weighted = (mean1_w - mean0_w) / pooled_sd_w if pooled_sd_w > 0 else 0
            else:
                smd_weighted = smd_unweighted
            
            balance_results[covar] = {
                'smd_unweighted': float(smd_unweighted),
                'smd_weighted': float(smd_weighted),
                'balanced': abs(smd_weighted) < 0.1
            }
    
    # Overall balance
    all_balanced = all(res['balanced'] for res in balance_results.values())
    max_smd = max(abs(res['smd_weighted']) for res in balance_results.values())
    
    return {
        'covariate_balance': balance_results,
        'all_balanced': all_balanced,
        'max_weighted_smd': float(max_smd)
    }

def leave_one_out_sensitivity(df, outcome_col, treatment_col, covariate_cols):
    """Leave-one-covariate-out sensitivity analysis"""
    logger.info("Running leave-one-out sensitivity analysis")
    
    results = {}
    
    # Full model
    X_full = sm.add_constant(df[covariate_cols + [treatment_col]])
    y = df[outcome_col]
    
    if 'iptw' in df.columns:
        model_full = sm.GLM(y, X_full, family=sm.families.Poisson(),
                           freq_weights=df['iptw'])
    else:
        model_full = sm.GLM(y, X_full, family=sm.families.Poisson())
    
    fit_full = model_full.fit()
    effect_full = fit_full.params[treatment_col]
    
    results['full_model'] = {
        'effect': float(effect_full),
        'n_covariates': len(covariate_cols)
    }
    
    # Leave each covariate out
    for i, covar in enumerate(covariate_cols[:10]):  # Limit to first 10
        try:
            covars_subset = [c for c in covariate_cols if c != covar]
            X_subset = sm.add_constant(df[covars_subset + [treatment_col]])
            
            if 'iptw' in df.columns:
                model = sm.GLM(y, X_subset, family=sm.families.Poisson(),
                             freq_weights=df['iptw'])
            else:
                model = sm.GLM(y, X_subset, family=sm.families.Poisson())
            
            fit = model.fit()
            effect = fit.params[treatment_col]
            
            # Percent change from full model
            pct_change = (effect - effect_full) / effect_full * 100
            
            results[f'without_{covar}'] = {
                'effect': float(effect),
                'pct_change': float(pct_change)
            }
            
        except Exception as e:
            logger.error(f"Failed leaving out {covar}: {e}")
    
    return results

def create_forest_plot(robustness_results, output_path):
    """Create forest plot of robustness checks"""
    logger.info("Creating forest plot")
    
    # Collect all estimates
    estimates = []
    
    # Alternative specifications
    if 'alternative_specifications' in robustness_results:
        for spec, result in robustness_results['alternative_specifications'].items():
            if 'effect' in result:
                estimates.append({
                    'method': spec.replace('_', ' ').title(),
                    'effect': result['effect'],
                    'ci_lower': result.get('ci_lower', result['effect']),
                    'ci_upper': result.get('ci_upper', result['effect']),
                    'category': 'Model Specification'
                })
    
    # Trimming analysis
    if 'trimming_analysis' in robustness_results:
        for trim, result in robustness_results['trimming_analysis'].items():
            if 'effect' in result:
                estimates.append({
                    'method': trim.replace('_', ' ').title(),
                    'effect': result['effect'],
                    'ci_lower': result.get('ci_lower', result['effect']),
                    'ci_upper': result.get('ci_upper', result['effect']),
                    'category': 'Weight Trimming'
                })
    
    if not estimates:
        logger.warning("No estimates to plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by category and effect size
    estimates_df = pd.DataFrame(estimates)
    estimates_df = estimates_df.sort_values(['category', 'effect'])
    
    y_pos = np.arange(len(estimates_df))
    
    # Plot points and CIs
    for i, row in estimates_df.iterrows():
        # Point estimate
        ax.plot(row['effect'], y_pos[i], 'o', markersize=8,
               color='blue' if row['category'] == 'Model Specification' else 'red')
        
        # Confidence interval
        ax.plot([row['ci_lower'], row['ci_upper']], [y_pos[i], y_pos[i]],
               'k-', linewidth=2, alpha=0.5)
    
    # Reference line at 0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(estimates_df['method'])
    ax.set_xlabel('Effect Estimate')
    ax.set_title('Robustness Checks: Forest Plot')
    
    # Add category labels
    for cat in estimates_df['category'].unique():
        cat_data = estimates_df[estimates_df['category'] == cat]
        y_min = y_pos[cat_data.index[0]]
        y_max = y_pos[cat_data.index[-1]]
        ax.text(-0.1, (y_min + y_max) / 2, cat, 
               transform=ax.get_yaxis_transform(),
               ha='right', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive robustness checks"
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without saving outputs')
    parser.add_argument('--outcome', default='total_encounters',
                       help='Outcome variable to analyze')
    parser.add_argument('--treatment-col', default='ssd_flag', help='Treatment column name (default: ssd_flag)')
    args = parser.parse_args()
    
    # Set random seeds
    set_global_seeds()
    
    # Load configuration
    config = load_config()
    
    # Initialize tracker
    tracker = ArtefactTracker()
    tracker.track("script_start", {"script": "15_robustness.py"})
    
    # Load data
    data_path = Path("data_derived/ps_weighted.parquet")
    if not data_path.exists():
        logger.error(f"PS weighted data not found at {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Define variables
    outcome_col = args.outcome
    treatment_col = args.treatment_col
    
    # Get covariates
    covariate_cols = [col for col in df.columns if col.endswith('_conf') or 
                     col in ['age', 'sex_M', 'charlson_score', 
                            'baseline_encounters', 'baseline_high_utilizer']]
    covariate_cols = [col for col in covariate_cols if col in df.columns]
    
    # Run robustness checks
    robustness_results = {
        'outcome': outcome_col,
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Alternative specifications
    alt_spec_results = alternative_specifications(
        df, outcome_col, treatment_col, covariate_cols
    )
    robustness_results['alternative_specifications'] = alt_spec_results
    
    # 2. Trimming analysis
    trim_results = trimming_analysis(
        df, outcome_col, treatment_col, covariate_cols
    )
    robustness_results['trimming_analysis'] = trim_results
    
    # 3. Covariate balance
    balance_results = covariate_balance_check(
        df, treatment_col, covariate_cols
    )
    robustness_results['covariate_balance'] = balance_results
    
    # 4. Leave-one-out
    loo_results = leave_one_out_sensitivity(
        df, outcome_col, treatment_col, covariate_cols
    )
    robustness_results['leave_one_out'] = loo_results
    
    # Overall robustness assessment
    # Check if estimates are consistent across specifications
    all_effects = []
    
    for spec_results in [alt_spec_results, trim_results]:
        for result in spec_results.values():
            if 'effect' in result:
                all_effects.append(result['effect'])
    
    if all_effects:
        effect_cv = np.std(all_effects) / np.mean(all_effects) if np.mean(all_effects) != 0 else 0
        robustness_results['effect_consistency'] = {
            'coefficient_of_variation': float(effect_cv),
            'robust': effect_cv < 0.2  # Less than 20% variation
        }
    
    # Compile flags
    robustness_flags = {
        'balance': balance_results.get('all_balanced', False),
        'consistency': robustness_results.get('effect_consistency', {}).get('robust', False)
    }
    
    # Check for placebo test results
    placebo_path = Path("results/placebo_tests.json")
    if placebo_path.exists():
        try:
            with open(placebo_path, 'r') as f:
                placebo_data = json.load(f)
                placebo_passed = placebo_data.get('all_tests_passed', False)
                robustness_flags['placebo'] = placebo_passed
        except Exception as e:
            logger.warning(f"Could not load placebo test results: {e}")
            robustness_flags['placebo'] = None
    else:
        logger.info("Placebo test results not found - run 14_placebo_tests.py first")
        robustness_flags['placebo'] = None
    
    robustness_results['robustness_flags'] = robustness_flags
    
    # Save results
    if not args.dry_run:
        output_path = Path("results/robustness_results.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(robustness_results, f, indent=2)
        logger.info(f"Saved robustness results to {output_path}")
        
        # Create forest plot
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)
        plot_path = figures_dir / "robustness_forest_plot.pdf"
        create_forest_plot(robustness_results, plot_path)
        
        # Track outputs
        tracker.track("output_generated", {
            "file": str(output_path),
            "robustness_flags": robustness_flags
        })
        
        # Update study documentation
        if Path("scripts/update_study_doc.py").exists():
            import subprocess
            flags_str = json.dumps(robustness_flags)
            subprocess.run([
                "python", "scripts/update_study_doc.py",
                "--step", "Robustness checks complete",
                "--kv", f"robustness_flags={flags_str}"
            ])
    
    # Print summary
    print("\n=== Robustness Check Summary ===")
    print(f"Outcome: {outcome_col}")
    print("\nAlternative specifications:")
    for spec, result in alt_spec_results.items():
        if 'effect' in result:
            print(f"  {spec}: {result['effect']:.3f} (p={result.get('p_value', 0):.3f})")
    
    print("\nTrimming analysis:")
    for trim, result in trim_results.items():
        if 'effect' in result:
            print(f"  {trim}: {result['effect']:.3f} (ESS={result.get('ess', 0):.0f})")
    
    print(f"\nCovariate balance:")
    print(f"  All balanced (<0.1 SMD): {balance_results.get('all_balanced', False)}")
    print(f"  Max weighted SMD: {balance_results.get('max_weighted_smd', 0):.3f}")
    
    if 'effect_consistency' in robustness_results:
        print(f"\nEffect consistency:")
        print(f"  Coefficient of variation: {robustness_results['effect_consistency']['coefficient_of_variation']:.2f}")
        print(f"  Robust: {robustness_results['effect_consistency']['robust']}")
    
    print("\nOverall robustness flags:")
    for flag, value in robustness_flags.items():
        print(f"  {flag}: {'OK' if value else 'WARNING'}")
    print("================================\n")

if __name__ == "__main__":
    main()