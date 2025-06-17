#!/usr/bin/env python3
"""
12_temporal_adjust.py - Segmented regression for temporal adjustment

Implements segmented regression analysis to account for temporal trends and
COVID-19 impact on SSD outcomes. Optionally supports Marginal Structural Models.

Hypothesis Support:
- H1: Healthcare utilization - Adjust for temporal trends
- H5-MH: Effect modification in MH subgroups - Control for temporal confounding

Output:
- results/segmented_regression.pkl: Model results
- figures/temporal_trends.pdf: Visualization of trends
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
import pickle
import json
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

def prepare_temporal_data(df, config):
    """Prepare data for temporal analysis"""
    logger.info("Preparing temporal data")
    
    # Ensure index_date is datetime
    df['index_date'] = pd.to_datetime(df['index_date'])
    
    # Create time variables
    df['month'] = df['index_date'].dt.to_period('M')
    df['year_month'] = df['index_date'].dt.strftime('%Y-%m')
    
    # Calculate time since study start
    study_start = pd.Timestamp(config['temporal']['reference_date'])
    df['months_since_start'] = (
        (df['index_date'] - study_start).dt.days / 30.44
    ).round().astype(int)
    
    # COVID-19 indicator (March 2020 onwards)
    covid_start = pd.Timestamp('2020-03-01')
    df['post_covid'] = (df['index_date'] >= covid_start).astype(int)
    
    # Time since COVID start (for slope change)
    df['months_since_covid'] = np.where(
        df['post_covid'] == 1,
        ((df['index_date'] - covid_start).dt.days / 30.44).round(),
        0
    ).astype(int)
    
    return df

def run_segmented_regression(df, outcome_col, treatment_col='ssd_flag', 
                           use_weights=True, family='poisson'):
    """
    Run segmented regression analysis
    
    Model: outcome ~ time + post_covid + time_post_covid + treatment + 
                    treatment*post_covid + covariates
    """
    logger.info(f"Running segmented regression for {outcome_col}")
    
    # Prepare design matrix
    X_vars = [
        'months_since_start',           # Linear time trend
        'post_covid',                   # Level change at COVID
        'months_since_covid',           # Slope change after COVID
        treatment_col,                  # Treatment effect
        'treatment_covid_interaction',  # Differential COVID impact
        'age',                          # Covariates
        'sex_M',
        'charlson_score'
    ]
    
    # Create interaction term
    df['treatment_covid_interaction'] = df[treatment_col] * df['post_covid']
    
    # Select complete cases
    analysis_df = df[X_vars + [outcome_col]].dropna()
    
    if use_weights and 'iptw' in df.columns:
        weights = df.loc[analysis_df.index, 'iptw']
    else:
        weights = None
    
    # Add constant
    X = sm.add_constant(analysis_df[X_vars])
    y = analysis_df[outcome_col]
    
    # Choose model family
    if family == 'poisson':
        model_family = sm.families.Poisson()
    elif family == 'negbinom':
        model_family = sm.families.NegativeBinomial()
    else:
        model_family = sm.families.Gaussian()
    
    # Fit model
    if weights is not None:
        model = sm.GLM(y, X, family=model_family, freq_weights=weights)
    else:
        model = sm.GLM(y, X, family=model_family)
    
    results = model.fit()
    
    # Extract key parameters
    params = {
        'time_trend': results.params.get('months_since_start', 0),
        'covid_level_shift': results.params.get('post_covid', 0),
        'covid_slope_change': results.params.get('months_since_covid', 0),
        'treatment_effect': results.params.get(treatment_col, 0),
        'treatment_covid_interaction': results.params.get('treatment_covid_interaction', 0)
    }
    
    # Calculate confidence intervals
    conf_int = results.conf_int()
    
    return results, params, conf_int

def create_temporal_plot(df, outcome_col, treatment_col='ssd_flag', 
                        predictions=None, output_path=None):
    """Create visualization of temporal trends"""
    logger.info("Creating temporal trends visualization")
    
    # Aggregate by month and treatment
    monthly = df.groupby(['year_month', treatment_col]).agg({
        outcome_col: ['mean', 'count', 'std']
    }).reset_index()
    
    monthly.columns = ['year_month', treatment_col, 'mean', 'n', 'std']
    monthly['se'] = monthly['std'] / np.sqrt(monthly['n'])
    monthly['date'] = pd.to_datetime(monthly['year_month'] + '-01')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot observed data
    for treatment in [0, 1]:
        data = monthly[monthly[treatment_col] == treatment]
        label = 'SSD' if treatment == 1 else 'Control'
        color = 'red' if treatment == 1 else 'blue'
        
        ax.errorbar(data['date'], data['mean'], yerr=1.96*data['se'],
                   marker='o', label=f'{label} (observed)', 
                   color=color, alpha=0.6, capsize=5)
    
    # Add COVID line
    covid_date = pd.Timestamp('2020-03-01')
    ax.axvline(x=covid_date, color='gray', linestyle='--', 
              alpha=0.5, label='COVID-19 start')
    
    # Format plot
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Mean {outcome_col}')
    ax.set_title('Temporal Trends in Outcome')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def run_msm_analysis(df, config, demo_mode=False):
    """
    Marginal Structural Model analysis for time-varying confounding
    
    Parameters:
    -----------
    df : DataFrame
        Longitudinal data
    config : dict
        Configuration parameters
    demo_mode : bool
        If True, run with synthetic demo data
        
    Returns:
    --------
    dict
        MSM analysis results
    """
    logger.info("Running MSM analysis for time-varying confounding")
    
    if demo_mode or len(df) < 1000:
        logger.info("Running MSM demo with simulated longitudinal data")
        
        # Create synthetic longitudinal data for demo
        np.random.seed(42)
        n_patients = min(1000, len(df))
        n_timepoints = 6  # 6 months of follow-up
        
        # Simulate longitudinal data
        demo_data = []
        for patient_id in range(n_patients):
            baseline_risk = np.random.uniform(0.1, 0.9)
            
            for t in range(n_timepoints):
                # Time-varying confounders
                stress_level = baseline_risk + np.random.normal(0, 0.1)
                severity = baseline_risk + np.random.normal(0, 0.1) + t * 0.02
                
                # Time-varying treatment (SSD status)
                ssd_prob = np.clip(severity * 0.6 + stress_level * 0.4, 0, 1)
                ssd_flag = np.random.binomial(1, ssd_prob)
                
                # Outcome influenced by treatment and confounders
                encounter_rate = (ssd_flag * 0.3 + severity * 0.4 + 
                                stress_level * 0.2 + np.random.normal(0, 0.1))
                encounters = np.random.poisson(np.clip(encounter_rate * 3, 0, 10))
                
                demo_data.append({
                    'patient_id': patient_id,
                    'time_period': t,
                    'stress_level': stress_level,
                    'severity': severity,
                    'ssd_flag': ssd_flag,
                    'encounters': encounters,
                    'baseline_risk': baseline_risk
                })
        
        demo_df = pd.DataFrame(demo_data)
        
        # Simple MSM analysis using IPTW
        results = run_iptw_msm(demo_df)
        results['demo_mode'] = True
        results['n_patients'] = n_patients
        results['n_timepoints'] = n_timepoints
        
    else:
        logger.info("Running MSM with provided longitudinal data")
        
        # For real data, implement simplified MSM
        if 'time_period' not in df.columns:
            logger.warning("No time_period column found, creating from index_date")
            if 'index_date' in df.columns:
                df['time_period'] = pd.to_datetime(df['index_date']).dt.to_period('M').astype(str)
            else:
                logger.error("Cannot create time periods, using single time point")
                df['time_period'] = 0
        
        results = run_iptw_msm(df)
        results['demo_mode'] = False
    
    # Save results
    output_path = Path("results/msm_demo.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"MSM results saved to {output_path}")
    
    return results

def run_iptw_msm(df):
    """
    Run simplified MSM using Inverse Probability of Treatment Weighting
    """
    logger.info("Running IPTW-based MSM analysis")
    
    try:
        # Group by patient and calculate mean effects
        if 'patient_id' in df.columns:
            patient_data = df.groupby('patient_id').agg({
                'ssd_flag': 'mean',
                'encounters': 'mean',
                'stress_level': 'mean' if 'stress_level' in df.columns else lambda x: 0.5,
                'severity': 'mean' if 'severity' in df.columns else lambda x: 0.5
            }).reset_index()
        else:
            # Single time point data
            patient_data = df.copy()
            patient_data['patient_id'] = range(len(patient_data))
        
        # Simple propensity score for treatment
        from sklearn.linear_model import LogisticRegression
        
        X_vars = ['stress_level', 'severity'] if all(col in patient_data.columns for col in ['stress_level', 'severity']) else []
        
        if X_vars:
            X = patient_data[X_vars].fillna(0.5)
            y = patient_data['ssd_flag'].fillna(0)
            
            # Fit propensity score model
            ps_model = LogisticRegression(random_state=42)
            ps_model.fit(X, y)
            ps_scores = ps_model.predict_proba(X)[:, 1]
        else:
            # Use marginal probability if no confounders
            ps_scores = np.full(len(patient_data), patient_data['ssd_flag'].mean())
        
        # Calculate IPTW weights
        weights = np.where(
            patient_data['ssd_flag'] == 1,
            1 / ps_scores,
            1 / (1 - ps_scores)
        )
        
        # Trim extreme weights
        weights = np.clip(weights, 0.1, 10)
        
        # Weighted outcome analysis
        treated_outcome = np.average(
            patient_data[patient_data['ssd_flag'] == 1]['encounters'],
            weights=weights[patient_data['ssd_flag'] == 1]
        )
        
        control_outcome = np.average(
            patient_data[patient_data['ssd_flag'] == 0]['encounters'],
            weights=weights[patient_data['ssd_flag'] == 0]
        )
        
        ate = treated_outcome - control_outcome
        
        results = {
            'method': 'IPTW-MSM',
            'treated_mean': float(treated_outcome),
            'control_mean': float(control_outcome),
            'ate': float(ate),
            'n_treated': int((patient_data['ssd_flag'] == 1).sum()),
            'n_control': int((patient_data['ssd_flag'] == 0).sum()),
            'mean_ps_score': float(ps_scores.mean()),
            'ps_model_features': X_vars,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        logger.info(f"MSM ATE: {ate:.3f} (Treated: {treated_outcome:.3f}, Control: {control_outcome:.3f})")
        
        return results
        
    except Exception as e:
        logger.error(f"MSM analysis failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def main():
    parser = argparse.ArgumentParser(
        description="Temporal adjustment using segmented regression"
    )
    parser.add_argument('--msm', action='store_true',
                       help='Run MSM analysis')
    parser.add_argument('--demo', action='store_true',
                       help='Run in demo mode with synthetic data')
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
    tracker.track("script_start", {"script": "12_temporal_adjust.py"})
    
    # Load data
    data_path = Path("data_derived/ps_weighted.parquet")
    if not data_path.exists():
        logger.error(f"PS weighted data not found at {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Prepare temporal variables
    df = prepare_temporal_data(df, config)
    
    # Run segmented regression
    treatment_col = args.treatment_col
    results, params, conf_int = run_segmented_regression(
        df, args.outcome, treatment_col=treatment_col, use_weights=True
    )
    
    # Extract key results
    covid_shift = params['covid_level_shift']
    interaction = params['treatment_covid_interaction']
    
    logger.info(f"COVID level shift: β={covid_shift:.3f}")
    logger.info(f"Treatment*COVID interaction: β={interaction:.3f}")
    
    # Run MSM if requested
    msm_results = None
    if args.msm or args.demo:
        msm_results = run_msm_analysis(df, config, demo_mode=args.demo)
    
    # Create visualization
    if not args.dry_run:
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)
        plot_path = figures_dir / f"temporal_trends_{args.outcome}.png"
        create_temporal_plot(df, args.outcome, treatment_col=treatment_col, output_path=plot_path)
    
    # Save results
    if not args.dry_run:
        # Save regression results
        results_path = Path("results/segmented_regression.pkl")
        results_path.parent.mkdir(exist_ok=True)
        results.save(str(results_path))
        logger.info(f"Saved regression results to {results_path}")
        
        # Save summary
        summary = {
            'outcome': args.outcome,
            'n_observations': len(df),
            'time_trend_beta': float(params['time_trend']),
            'covid_level_shift_beta': float(covid_shift),
            'covid_slope_change_beta': float(params['covid_slope_change']),
            'treatment_effect_beta': float(params['treatment_effect']),
            'treatment_covid_interaction_beta': float(interaction),
            'msm_results': msm_results,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = Path("results/temporal_adjustment_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Track outputs
        tracker.track("output_generated", {
            "file": str(results_path),
            "covid_shift": covid_shift,
            "interaction": interaction
        })
        
        # Update study documentation
        if Path("scripts/update_study_doc.py").exists():
            import subprocess
            subprocess.run([
                "python", "scripts/update_study_doc.py",
                "--step", "Segmented regression complete",
                "--kv", f"covid_level_shift=β={covid_shift:.3f}"
            ])
    
    # Print summary
    print("\n=== Segmented Regression Summary ===")
    print(f"Outcome analyzed: {args.outcome}")
    print(f"N observations: {len(df):,}")
    print("\nKey coefficients:")
    print(f"  Time trend: β={params['time_trend']:.4f}")
    print(f"  COVID level shift: β={covid_shift:.3f}")
    print(f"  COVID slope change: β={params['covid_slope_change']:.4f}")
    print(f"  Treatment effect: β={params['treatment_effect']:.3f}")
    print(f"  Treatment*COVID: β={interaction:.3f}")
    if args.msm:
        print(f"\nMSM status: {msm_results['status']}")
    print("=====================================\n")

if __name__ == "__main__":
    main()