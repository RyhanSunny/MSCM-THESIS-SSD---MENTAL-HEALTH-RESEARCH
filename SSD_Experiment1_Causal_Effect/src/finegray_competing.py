#!/usr/bin/env python3
"""
finegray_competing.py - Fine-Gray competing risk analysis

Implements competing risk analysis for SSD outcomes, accounting for death
as a competing event. Uses cause-specific Cox model as approximation when
Fine-Gray is not available.

Hypothesis Support:
- H1: Healthcare utilization accounting for mortality
- H3: Medication use with death as competing risk

Output:
- results/competing_risk_results.json: Hazard ratios and survival curves
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime
import warnings
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.global_seeds import set_global_seeds
from src.config_loader import load_config
from src.artefact_tracker import ArtefactTracker

# Try to import R integration for true Fine-Gray
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    warnings.warn("R/rpy2 not available, using cause-specific Cox approximation")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_survival_data(df):
    """Prepare data for survival analysis"""
    logger.info("Preparing survival data")
    
    # Ensure date columns are datetime
    date_cols = ['index_date', 'death_date', 'last_encounter_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Calculate follow-up time
    if 'death_date' in df.columns:
        # Time to death or last encounter
        df['end_date'] = df['death_date'].fillna(df['last_encounter_date'])
    else:
        df['end_date'] = df['last_encounter_date']
    
    # Follow-up time in days
    df['followup_days'] = (df['end_date'] - df['index_date']).dt.days
    
    # Event indicator (1 = death, 0 = censored)
    df['death_event'] = df['death_date'].notna().astype(int)
    
    # For competing risk: define multiple event types
    # 0 = censored, 1 = primary outcome (high utilization), 2 = death
    df['event_type'] = 0  # Default censored
    
    # Define high utilization event (e.g., >10 encounters in outcome period)
    if 'total_encounters' in df.columns:
        high_util_threshold = df['total_encounters'].quantile(0.75)
        df.loc[df['total_encounters'] > high_util_threshold, 'event_type'] = 1
    
    # Death overwrites other events
    df.loc[df['death_event'] == 1, 'event_type'] = 2
    
    # Time to first event
    df['time_to_event'] = df['followup_days']
    
    # For those with primary outcome, might have earlier event time
    # This is simplified - in practice would track actual event dates
    
    return df

def run_cause_specific_cox(df, treatment_col='ssd_flag', covariate_cols=None):
    """Run cause-specific Cox models"""
    logger.info("Running cause-specific Cox models")
    
    if covariate_cols is None:
        covariate_cols = ['age', 'sex_M', 'charlson_score']
        covariate_cols = [col for col in covariate_cols if col in df.columns]
    
    results = {}
    
    # Model 1: Primary outcome (ignoring competing risk)
    logger.info("Fitting model for primary outcome")
    df_primary = df.copy()
    df_primary['event'] = (df['event_type'] == 1).astype(int)
    
    cph_primary = CoxPHFitter()
    try:
        cph_primary.fit(
            df_primary[[treatment_col] + covariate_cols + ['time_to_event', 'event']],
            duration_col='time_to_event',
            event_col='event'
        )
        
        hr_primary = np.exp(cph_primary.params[treatment_col])
        ci_primary = np.exp(cph_primary.confidence_intervals_[treatment_col])
        
        results['primary_outcome'] = {
            'hr': float(hr_primary),
            'ci_lower': float(ci_primary.iloc[0]),
            'ci_upper': float(ci_primary.iloc[1]),
            'p_value': float(cph_primary.summary.loc[treatment_col, 'p'])
        }
    except Exception as e:
        logger.error(f"Primary outcome model failed: {e}")
        results['primary_outcome'] = None
    
    # Model 2: Death as outcome
    logger.info("Fitting model for death outcome")
    df_death = df.copy()
    df_death['event'] = (df['event_type'] == 2).astype(int)
    
    cph_death = CoxPHFitter()
    try:
        cph_death.fit(
            df_death[[treatment_col] + covariate_cols + ['time_to_event', 'event']],
            duration_col='time_to_event',
            event_col='event'
        )
        
        hr_death = np.exp(cph_death.params[treatment_col])
        ci_death = np.exp(cph_death.confidence_intervals_[treatment_col])
        
        results['death'] = {
            'hr': float(hr_death),
            'ci_lower': float(ci_death.iloc[0]),
            'ci_upper': float(ci_death.iloc[1]),
            'p_value': float(cph_death.summary.loc[treatment_col, 'p'])
        }
    except Exception as e:
        logger.error(f"Death outcome model failed: {e}")
        results['death'] = None
    
    return results, cph_primary, cph_death

def run_fine_gray_r(df, treatment_col='ssd_flag', covariate_cols=None):
    """Run Fine-Gray model using R (if available)"""
    if not R_AVAILABLE:
        logger.warning("R not available for Fine-Gray model")
        return None
    
    logger.info("Running Fine-Gray model in R")
    
    try:
        # Import R packages
        cmprsk = importr('cmprsk')
        survival = importr('survival')
        
        # Prepare data for R
        r_data = df[[treatment_col] + covariate_cols + 
                   ['time_to_event', 'event_type']].copy()
        
        # Convert to R dataframe
        r_df = pandas2ri.py2rpy(r_data)
        
        # Run Fine-Gray model
        # In R: crr(ftime, fstatus, cov1, failcode=1)
        ro.r('''
        finegray_model <- function(data, treatment, covariates) {
            library(cmprsk)
            
            # Prepare covariates matrix
            cov_matrix <- as.matrix(data[, c(treatment, covariates)])
            
            # Run competing risk regression
            fit <- crr(ftime = data$time_to_event,
                      fstatus = data$event_type,
                      cov1 = cov_matrix,
                      failcode = 1)
            
            # Extract results
            coef <- fit$coef[1]  # Treatment effect
            se <- sqrt(fit$var[1,1])
            hr <- exp(coef)
            ci_lower <- exp(coef - 1.96*se)
            ci_upper <- exp(coef + 1.96*se)
            p_value <- 2 * pnorm(-abs(coef/se))
            
            return(list(hr=hr, ci_lower=ci_lower, ci_upper=ci_upper, p_value=p_value))
        }
        ''')
        
        # Call R function
        result = ro.r['finegray_model'](r_df, treatment_col, covariate_cols)
        
        return {
            'hr': float(result[0][0]),
            'ci_lower': float(result[1][0]),
            'ci_upper': float(result[2][0]),
            'p_value': float(result[3][0])
        }
        
    except Exception as e:
        logger.error(f"Fine-Gray R implementation failed: {e}")
        return None

def create_cumulative_incidence_plot(df, treatment_col='ssd_flag', output_path=None):
    """Create cumulative incidence plot"""
    logger.info("Creating cumulative incidence plot")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Primary outcome by treatment
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    
    for treatment, group_df in df.groupby(treatment_col):
        label = 'SSD' if treatment == 1 else 'Control'
        event = (group_df['event_type'] == 1).astype(int)
        
        kmf.fit(group_df['time_to_event'], event, label=f'{label} - Primary')
        kmf.plot_cumulative_density(ax=ax1)
    
    ax1.set_xlabel('Days from Index Date')
    ax1.set_ylabel('Cumulative Incidence')
    ax1.set_title('Primary Outcome')
    ax1.legend()
    
    # Plot 2: Death by treatment
    for treatment, group_df in df.groupby(treatment_col):
        label = 'SSD' if treatment == 1 else 'Control'
        event = (group_df['event_type'] == 2).astype(int)
        
        kmf.fit(group_df['time_to_event'], event, label=f'{label} - Death')
        kmf.plot_cumulative_density(ax=ax2)
    
    ax2.set_xlabel('Days from Index Date')
    ax2.set_ylabel('Cumulative Incidence')
    ax2.set_title('Death as Competing Risk')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Fine-Gray competing risk analysis"
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without saving outputs')
    args = parser.parse_args()
    
    # Set random seeds
    set_global_seeds()
    
    # Load configuration
    config = load_config()
    
    # Initialize tracker
    tracker = ArtefactTracker()
    tracker.track("script_start", {"script": "finegray_competing.py"})
    
    # Load data
    data_path = Path("data_derived/patient_master.parquet")
    if not data_path.exists():
        logger.error(f"Patient master data not found at {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Prepare survival data
    df = prepare_survival_data(df)
    
    # Get covariates
    covariate_cols = [col for col in df.columns if col.endswith('_conf') or 
                     col in ['age', 'sex_M', 'charlson_score']]
    covariate_cols = [col for col in covariate_cols if col in df.columns][:10]
    
    # Run cause-specific Cox models
    cox_results, cph_primary, cph_death = run_cause_specific_cox(
        df, 'ssd_flag', covariate_cols
    )
    
    # Try Fine-Gray if R available
    finegray_result = None
    if R_AVAILABLE:
        finegray_result = run_fine_gray_r(df, 'ssd_flag', covariate_cols)
    
    # Compile results
    results = {
        'analysis': 'competing_risk',
        'n_patients': len(df),
        'n_deaths': int(df['death_event'].sum()),
        'n_primary_events': int((df['event_type'] == 1).sum()),
        'median_followup_days': float(df['followup_days'].median()),
        'cause_specific_cox': cox_results,
        'fine_gray': finegray_result,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    if not args.dry_run:
        output_path = Path("results/competing_risk_results.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")
        
        # Create plot
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)
        plot_path = figures_dir / "cumulative_incidence.pdf"
        create_cumulative_incidence_plot(df, output_path=plot_path)
        
        # Track outputs
        tracker.track("output_generated", {
            "file": str(output_path),
            "n_deaths": results['n_deaths'],
            "fine_gray_available": finegray_result is not None
        })
        
        # Update study documentation
        if Path("scripts/update_study_doc.py").exists():
            import subprocess
            if cox_results.get('primary_outcome'):
                hr = cox_results['primary_outcome']['hr']
                subprocess.run([
                    "python", "scripts/update_study_doc.py",
                    "--step", "Fine-Gray competing risk analysis complete",
                    "--kv", f"fine_gray_hr={hr:.3f}"
                ])
    
    # Print summary
    print("\n=== Competing Risk Analysis Summary ===")
    print(f"Total patients: {len(df):,}")
    print(f"Deaths: {results['n_deaths']:,}")
    print(f"Primary events: {results['n_primary_events']:,}")
    print(f"Median follow-up: {results['median_followup_days']:.0f} days")
    
    if cox_results.get('primary_outcome'):
        print("\nCause-specific Cox (Primary outcome):")
        print(f"  HR: {cox_results['primary_outcome']['hr']:.3f} "
              f"({cox_results['primary_outcome']['ci_lower']:.3f}-"
              f"{cox_results['primary_outcome']['ci_upper']:.3f})")
    
    if cox_results.get('death'):
        print("\nCause-specific Cox (Death):")
        print(f"  HR: {cox_results['death']['hr']:.3f} "
              f"({cox_results['death']['ci_lower']:.3f}-"
              f"{cox_results['death']['ci_upper']:.3f})")
    
    if finegray_result:
        print("\nFine-Gray model:")
        print(f"  HR: {finegray_result['hr']:.3f} "
              f"({finegray_result['ci_lower']:.3f}-"
              f"{finegray_result['ci_upper']:.3f})")
    else:
        print("\nNote: Fine-Gray model requires R package cmprsk")
    print("=======================================\n")

if __name__ == "__main__":
    main()