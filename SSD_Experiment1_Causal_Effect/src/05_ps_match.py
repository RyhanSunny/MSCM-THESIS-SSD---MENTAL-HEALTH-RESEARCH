#!/usr/bin/env python3
"""
05_ps_match.py - Propensity Score Matching with GPU-accelerated XGBoost

Implements propensity score estimation and matching for causal inference.
Uses GPU-accelerated XGBoost for high-dimensional covariate adjustment.

Hypothesis Support:
- H1: Healthcare utilization - PS adjustment for confounding
- H2: Healthcare costs - Balanced comparison groups
- H3: Medication use - Unbiased treatment effect estimation
- H5: Health anxiety mediation - Covariate balance for mediation analysis

Output:
- data_derived/ps_weighted.parquet: Data with propensity scores and weights
- data_derived/ps_matched.parquet: 1:1 matched cohort
- figures/love_plot.pdf: Covariate balance visualization
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime
import warnings
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.global_seeds import set_global_seeds, get_random_state
from src.config_loader import load_config
from src.artefact_tracker import ArtefactTracker

# Try to import tableone, fall back if not available
try:
    from tableone import TableOne
    TABLEONE_AVAILABLE = True
except ImportError:
    TABLEONE_AVAILABLE = False
    warnings.warn("tableone not available, will use custom SMD calculation")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_smd(df, var, treatment_col='ssd_flag', weights=None):
    """Calculate Standardized Mean Difference"""
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    if weights is not None:
        # Weighted means and SDs
        w1 = weights[df[treatment_col] == 1]
        w0 = weights[df[treatment_col] == 0]
        
        mean1 = np.average(treated[var], weights=w1)
        mean0 = np.average(control[var], weights=w0)
        
        var1 = np.average((treated[var] - mean1)**2, weights=w1)
        var0 = np.average((control[var] - mean0)**2, weights=w0)
    else:
        mean1 = treated[var].mean()
        mean0 = control[var].mean()
        var1 = treated[var].var()
        var0 = control[var].var()
    
    pooled_sd = np.sqrt((var1 + var0) / 2)
    
    if pooled_sd == 0:
        return 0
    
    return (mean1 - mean0) / pooled_sd

def train_propensity_model(X, y, config):
    """Train XGBoost model for propensity scores"""
    logger.info("Training propensity score model")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': config.get('ps_model', {}).get('max_depth', 6),
        'eta': config.get('ps_model', {}).get('learning_rate', 0.1),
        'subsample': config.get('ps_model', {}).get('subsample', 0.8),
        'colsample_bytree': config.get('ps_model', {}).get('colsample_bytree', 0.8),
        'seed': 42
    }
    
    # Check for GPU availability
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
            logger.info("GPU detected, using GPU acceleration")
        else:
            params['tree_method'] = 'hist'
            logger.info("No GPU detected, using CPU")
    except:
        params['tree_method'] = 'hist'
        logger.info("GPU check failed, using CPU")
    
    # Train model
    num_rounds = config.get('ps_model', {}).get('num_rounds', 100)
    model = xgb.train(params, dtrain, num_boost_round=num_rounds)
    
    # Get propensity scores
    ps = model.predict(dtrain)
    
    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, ps)
    logger.info(f"Propensity score model AUC: {auc:.3f}")
    
    return model, ps, auc

def calculate_weights(ps, treatment, config):
    """Calculate inverse probability of treatment weights (IPTW)"""
    logger.info("Calculating IPTW weights")
    
    # Basic IPTW
    iptw = np.where(treatment == 1, 1/ps, 1/(1-ps))
    
    # Stabilized weights
    p_treatment = treatment.mean()
    sw = np.where(treatment == 1, p_treatment/ps, (1-p_treatment)/(1-ps))
    
    # Trim weights at percentiles to avoid extreme values
    trim_lower = config.get('ps_trimming', {}).get('lower', 1)
    trim_upper = config.get('ps_trimming', {}).get('upper', 99)
    
    iptw_trimmed = np.clip(iptw, np.percentile(iptw, trim_lower), 
                          np.percentile(iptw, trim_upper))
    sw_trimmed = np.clip(sw, np.percentile(sw, trim_lower), 
                        np.percentile(sw, trim_upper))
    
    # Calculate effective sample size (ESS)
    ess_iptw = (np.sum(iptw_trimmed))**2 / np.sum(iptw_trimmed**2)
    ess_sw = (np.sum(sw_trimmed))**2 / np.sum(sw_trimmed**2)
    
    logger.info(f"ESS (IPTW): {ess_iptw:.0f}")
    logger.info(f"ESS (Stabilized): {ess_sw:.0f}")
    
    # Use stabilized weights by default
    weights = sw_trimmed
    ess = ess_sw
    
    return weights, ess

def perform_matching(ps, treatment, config):
    """Perform 1:1 propensity score matching with caliper"""
    logger.info("Performing 1:1 propensity score matching")
    
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    
    # Fit nearest neighbors on control units
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(ps[control_idx].reshape(-1, 1))
    
    # Find matches for treated units
    distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
    
    # Apply caliper
    caliper = config.get('ps_matching', {}).get('caliper', 0.05)
    caliper_threshold = caliper * ps.std()
    
    matched_pairs = []
    for i, treated in enumerate(treated_idx):
        if distances[i][0] <= caliper_threshold:
            control = control_idx[indices[i][0]]
            matched_pairs.append((treated, control))
    
    logger.info(f"Matched {len(matched_pairs)} pairs out of {len(treated_idx)} treated units")
    logger.info(f"Matching rate: {len(matched_pairs)/len(treated_idx)*100:.1f}%")
    
    return matched_pairs

def create_love_plot(smd_before, smd_after, output_path):
    """Create Love plot for covariate balance"""
    logger.info("Creating Love plot")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort variables by absolute SMD before matching
    vars_sorted = sorted(smd_before.keys(), 
                        key=lambda x: abs(smd_before[x]), 
                        reverse=True)
    
    y_pos = np.arange(len(vars_sorted))
    
    # Plot SMDs
    before_vals = [smd_before[var] for var in vars_sorted]
    after_vals = [smd_after[var] for var in vars_sorted]
    
    ax.scatter(before_vals, y_pos, color='red', alpha=0.6, s=100, label='Before weighting')
    ax.scatter(after_vals, y_pos, color='blue', alpha=0.6, s=100, label='After weighting')
    
    # Connect before and after points
    for i, var in enumerate(vars_sorted):
        ax.plot([smd_before[var], smd_after[var]], [i, i], 
               'gray', alpha=0.3, linewidth=1)
    
    # Add reference lines
    ax.axvline(x=0.1, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.1, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars_sorted)
    ax.set_xlabel('Standardized Mean Difference')
    ax.set_title('Covariate Balance: Love Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set x limits
    max_abs_smd = max(max(abs(min(before_vals)), max(before_vals)), 0.5)
    ax.set_xlim(-max_abs_smd*1.1, max_abs_smd*1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Love plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Propensity score estimation and matching"
    )
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run without saving outputs')
    parser.add_argument('--matching-only', action='store_true',
                       help='Only perform matching (skip weighting)')
    args = parser.parse_args()
    
    # Set random seeds
    set_global_seeds()
    
    # Load configuration
    config = load_config()
    
    # Initialize tracker
    tracker = ArtefactTracker()
    tracker.track("script_start", {"script": "05_ps_match.py"})
    
    # Load data
    master_path = Path("data_derived/patient_master.parquet")
    if not master_path.exists():
        logger.error(f"Patient master file not found at {master_path}")
        return
    
    logger.info(f"Loading data from {master_path}")
    df = pd.read_parquet(master_path)
    initial_rows = len(df)
    
    # Define treatment
    treatment_col = 'ssd_flag'
    y = df[treatment_col].values
    
    # Define covariates (all confounder columns)
    covar_cols = [col for col in df.columns if col.endswith('_conf') or 
                  col in ['age', 'sex_M', 'charlson_score', 'baseline_encounters',
                         'baseline_high_utilizer']]
    
    # Remove any columns that don't exist
    covar_cols = [col for col in covar_cols if col in df.columns]
    
    logger.info(f"Using {len(covar_cols)} covariates for propensity score model")
    
    X = df[covar_cols].fillna(0).values  # Simple imputation for now
    
    # Train propensity score model
    model, ps, auc = train_propensity_model(X, y, config)
    
    # Add propensity scores to dataframe
    df['propensity_score'] = ps
    
    # Calculate weights
    weights, ess = calculate_weights(ps, y, config)
    df['iptw'] = weights
    
    # Check covariate balance
    logger.info("Checking covariate balance")
    
    # Calculate SMD before weighting
    smd_before = {}
    smd_after = {}
    
    for col in covar_cols[:20]:  # Limit to first 20 for visualization
        if col in df.columns:
            smd_before[col] = calculate_smd(df, col, treatment_col)
            smd_after[col] = calculate_smd(df, col, treatment_col, weights)
    
    # Maximum SMD after weighting
    max_smd_after = max(abs(smd) for smd in smd_after.values())
    logger.info(f"Maximum SMD after weighting: {max_smd_after:.3f}")
    
    # Create Love plot
    if not args.dry_run:
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)
        love_plot_path = figures_dir / "love_plot.pdf"
        create_love_plot(smd_before, smd_after, love_plot_path)
    
    # Perform 1:1 matching if requested
    matched_df = None
    if not args.matching_only:
        matched_pairs = perform_matching(ps, y, config)
        
        # Create matched dataset
        matched_indices = []
        for treated, control in matched_pairs:
            matched_indices.extend([treated, control])
        
        matched_df = df.iloc[matched_indices].copy()
        matched_df['match_id'] = np.repeat(range(len(matched_pairs)), 2)
    
    # Save outputs
    if not args.dry_run:
        # Save weighted dataset
        weighted_path = Path("data_derived/ps_weighted.parquet")
        df.to_parquet(weighted_path, index=False)
        logger.info(f"Saved weighted data to {weighted_path}")
        
        # Save matched dataset if created
        if matched_df is not None:
            matched_path = Path("data_derived/ps_matched.parquet")
            matched_df.to_parquet(matched_path, index=False)
            logger.info(f"Saved matched data to {matched_path}")
        
        # Save PS model
        model_path = Path("models/ps_xgboost.json")
        model_path.parent.mkdir(exist_ok=True)
        model.save_model(str(model_path))
        
        # Save summary statistics
        results = {
            'n_total': int(initial_rows),
            'n_treated': int(y.sum()),
            'n_control': int(len(y) - y.sum()),
            'ps_model_auc': float(auc),
            'ess': float(ess),
            'max_smd_before': float(max(abs(smd) for smd in smd_before.values())),
            'max_smd_after': float(max_smd_after),
            'n_matched_pairs': len(matched_pairs) if matched_df is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = Path("results/ps_matching_results.json")
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Track outputs
        tracker.track("output_generated", {
            "weighted_file": str(weighted_path),
            "ess": ess,
            "max_smd": max_smd_after
        })
        
        # Update study documentation
        if Path("scripts/update_study_doc.py").exists():
            import subprocess
            subprocess.run([
                "python", "scripts/update_study_doc.py",
                "--step", "GPU PS weighting complete",
                "--kv", f"ess={int(ess)}",
                "--kv", f"max_post_weight_smd={max_smd_after:.3f}",
                "--kv", f"love_plot_path={love_plot_path}"
            ])
    
    # Print summary
    print("\n=== Propensity Score Analysis Summary ===")
    print(f"Total patients: {initial_rows:,}")
    print(f"Treated (SSD): {y.sum():,}")
    print(f"Control: {len(y) - y.sum():,}")
    print(f"PS model AUC: {auc:.3f}")
    print(f"Effective sample size: {ess:,.0f}")
    print(f"Max SMD before weighting: {max(abs(smd) for smd in smd_before.values()):.3f}")
    print(f"Max SMD after weighting: {max_smd_after:.3f}")
    if matched_df is not None:
        print(f"Matched pairs: {len(matched_pairs):,}")
    print("=========================================\n")

if __name__ == "__main__":
    main()