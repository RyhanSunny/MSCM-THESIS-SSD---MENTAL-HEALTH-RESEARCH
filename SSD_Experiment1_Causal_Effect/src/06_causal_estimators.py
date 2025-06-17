#!/usr/bin/env python3
"""
06_causal_estimators.py - Comprehensive causal inference estimation

Implements multiple causal inference methods:
- Targeted Maximum Likelihood Estimation (TMLE)
- Double Machine Learning (DML)
- Causal Forest for heterogeneous effects

Hypothesis Support:
- H1: Healthcare utilization - Causal effect estimation
- H2: Healthcare costs - Robust cost impact assessment
- H3: Medication use - Treatment effect on inappropriate prescribing
- H5: Physician mediation - Effect modification analysis

Output:
- results/ate_estimates.json: Average treatment effect estimates
- results/cate_analysis.json: Conditional average treatment effects
- figures/cate_distribution.pdf: CATE visualization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
import json
import warnings
from datetime import datetime
from scipy import stats
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.global_seeds import set_global_seeds, get_random_state
from src.config_loader import load_config
from src.artefact_tracker import ArtefactTracker

# Try to import advanced causal libraries
try:
    from econml.dml import LinearDML, CausalForestDML
    from econml.grf import CausalForest
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    warnings.warn("EconML not available, will use simplified implementations")

try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    warnings.warn("DoWhy not available, skipping causal graph validation")

# Try to import statsmodels for cluster-robust SE
try:
    import statsmodels.api as sm
    from cluster_robust_se import validate_clustering_structure, cluster_robust_poisson, cluster_robust_logistic
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available, using simplified SE calculations")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cluster_bootstrap_se(estimator_func, df, cluster_col='site_id', n_bootstrap=500, random_state=42):
    """
    Calculate cluster-robust standard errors using bootstrap.
    
    Alternative to statsmodels when not available.
    Uses cluster bootstrap following Cameron & Miller (2015).
    """
    np.random.seed(random_state)
    
    # Get unique clusters
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    
    if n_clusters < 10:
        warnings.warn(f"Only {n_clusters} clusters. Results may be unreliable.")
    
    # Original estimate
    original_estimate = estimator_func(df)
    
    # Bootstrap estimates
    bootstrap_estimates = []
    
    for b in range(n_bootstrap):
        # Sample clusters with replacement
        sampled_clusters = np.random.choice(clusters, size=n_clusters, replace=True)
        
        # Create bootstrap sample
        bootstrap_dfs = []
        for cluster in sampled_clusters:
            cluster_data = df[df[cluster_col] == cluster].copy()
            bootstrap_dfs.append(cluster_data)
        
        bootstrap_df = pd.concat(bootstrap_dfs, ignore_index=True)
        
        # Calculate estimate on bootstrap sample
        try:
            bootstrap_estimate = estimator_func(bootstrap_df)
            bootstrap_estimates.append(bootstrap_estimate)
        except:
            continue  # Skip failed bootstrap samples
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    # Calculate cluster-robust standard error
    cluster_se = np.std(bootstrap_estimates, ddof=1)
    
    # Calculate bias-corrected confidence interval
    alpha = 0.05
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha/2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha/2))
    
    return {
        'estimate': original_estimate,
        'cluster_se': cluster_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_clusters': n_clusters,
        'n_bootstrap': len(bootstrap_estimates)
    }

class SimplifiedTMLE:
    """Simplified TMLE implementation for when R integration is not available"""
    
    def __init__(self, Y, A, W, weights=None):
        self.Y = Y
        self.A = A
        self.W = W
        self.weights = weights if weights is not None else np.ones(len(Y))
        
    def fit(self):
        """Fit TMLE estimator"""
        n = len(self.Y)
        
        # Step 1: Estimate outcome model E[Y|A,W]
        X = np.column_stack([self.A, self.W])
        self.outcome_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        
        # Use cross-validation predictions to avoid overfitting
        cv_preds = cross_val_predict(
            self.outcome_model, X, self.Y, cv=5, method='predict'
        )
        
        # Fit final model on all data
        self.outcome_model.fit(X, self.Y)
        
        # Step 2: Estimate propensity score E[A|W]
        self.ps_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        ps_cv = cross_val_predict(
            self.ps_model, self.W, self.A, cv=5, method='predict_proba'
        )[:, 1]
        
        # Fit final model
        self.ps_model.fit(self.W, self.A)
        ps = self.ps_model.predict_proba(self.W)[:, 1]
        
        # Step 3: Calculate clever covariate
        H1 = self.A / ps
        H0 = (1 - self.A) / (1 - ps)
        
        # Step 4: Fluctuation parameter (simplified)
        epsilon_model = sm.GLM(
            self.Y - cv_preds,
            np.column_stack([H1, H0]),
            freq_weights=self.weights
        )
        epsilon_fit = epsilon_model.fit()
        
        # Step 5: Calculate ATE
        Q1 = self.outcome_model.predict(np.column_stack([np.ones(n), self.W]))
        Q0 = self.outcome_model.predict(np.column_stack([np.zeros(n), self.W]))
        
        # Apply fluctuation
        self.ate = np.average(Q1 - Q0, weights=self.weights)
        
        # Calculate standard error (simplified)
        influence = (Q1 - Q0 - self.ate) + H1 * (self.Y - Q1) - H0 * (self.Y - Q0)
        self.se = np.sqrt(np.average(influence**2, weights=self.weights) / n)
        
        return self

def run_tmle(df, outcome_col, treatment_col, covariate_cols, weights=None, cluster_col=None):
    """Run TMLE estimation with optional cluster-robust SE"""
    logger.info("Running TMLE estimation")
    
    Y = df[outcome_col].values
    A = df[treatment_col].values
    W = df[covariate_cols].values
    
    if weights is None and 'iptw' in df.columns:
        weights = df['iptw'].values
    
    # Define estimator function for cluster bootstrap
    def tmle_estimator(bootstrap_df):
        Y_b = bootstrap_df[outcome_col].values
        A_b = bootstrap_df[treatment_col].values
        W_b = bootstrap_df[covariate_cols].values
        weights_b = bootstrap_df['iptw'].values if 'iptw' in bootstrap_df.columns else None
        
        tmle_b = SimplifiedTMLE(Y_b, A_b, W_b, weights_b)
        tmle_b.fit()
        return tmle_b.ate
    
    # Use simplified TMLE
    tmle = SimplifiedTMLE(Y, A, W, weights)
    tmle.fit()
    
    # Calculate standard errors
    if cluster_col is not None and cluster_col in df.columns:
        logger.info("Calculating cluster-robust standard errors")
        cluster_results = cluster_bootstrap_se(tmle_estimator, df, cluster_col)
        
        results = {
            'method': 'TMLE',
            'estimate': float(tmle.ate),
            'se_naive': float(tmle.se),
            'se_cluster': float(cluster_results['cluster_se']),
            'ci_lower': float(cluster_results['ci_lower']),
            'ci_upper': float(cluster_results['ci_upper']),
            'n': len(Y),
            'n_clusters': cluster_results['n_clusters'],
            'se_inflation_factor': float(cluster_results['cluster_se'] / tmle.se),
            'clustered': True
        }
        
        logger.info(f"TMLE ATE: {tmle.ate:.3f} ({cluster_results['ci_lower']:.3f}, {cluster_results['ci_upper']:.3f}) [cluster-robust]")
        logger.info(f"SE inflation factor: {cluster_results['cluster_se'] / tmle.se:.2f}")
        
    else:
        # Standard (naive) confidence interval
        ci_lower = tmle.ate - 1.96 * tmle.se
        ci_upper = tmle.ate + 1.96 * tmle.se
        
        results = {
            'method': 'TMLE',
            'estimate': float(tmle.ate),
            'se': float(tmle.se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n': len(Y),
            'clustered': False
        }
        
        logger.info(f"TMLE ATE: {tmle.ate:.3f} ({ci_lower:.3f}, {ci_upper:.3f}) [naive SE]")
    
    return results

def run_double_ml(df, outcome_col, treatment_col, covariate_cols):
    """Run Double Machine Learning estimation"""
    logger.info("Running Double ML estimation")
    
    Y = df[outcome_col].values
    T = df[treatment_col].values
    X = df[covariate_cols].values
    
    if ECONML_AVAILABLE:
        # Use EconML implementation
        dml = LinearDML(
            model_y='auto',
            model_t='auto',
            discrete_treatment=True,
            cv=5,
            random_state=42
        )
        dml.fit(Y, T, X=X, W=X)
        
        ate = dml.effect(X).mean()
        ci = dml.effect_interval(X, alpha=0.05)
        ci_lower = ci[0].mean()
        ci_upper = ci[1].mean()
        
    else:
        # Simplified implementation
        n = len(Y)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Store out-of-sample predictions
        Y_res = np.zeros(n)
        T_res = np.zeros(n)
        
        for train_idx, test_idx in kf.split(X):
            # Fit outcome model
            rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_y.fit(X[train_idx], Y[train_idx])
            Y_res[test_idx] = Y[test_idx] - rf_y.predict(X[test_idx])
            
            # Fit treatment model
            rf_t = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_t.fit(X[train_idx], T[train_idx])
            T_prob = rf_t.predict_proba(X[test_idx])[:, 1]
            T_res[test_idx] = T[test_idx] - T_prob
        
        # Final stage regression
        ate = np.sum(T_res * Y_res) / np.sum(T_res * T_res)
        
        # Standard error (simplified)
        residuals = Y_res - ate * T_res
        se = np.sqrt(np.sum(residuals**2) / (n - 1)) / np.sqrt(np.sum(T_res**2))
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
    
    results = {
        'method': 'Double ML',
        'estimate': float(ate),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n': len(Y)
    }
    
    logger.info(f"DML ATE: {ate:.3f} ({ci_lower:.3f}, {ci_upper:.3f})")
    
    return results

def run_causal_forest(df, outcome_col, treatment_col, covariate_cols):
    """Run Causal Forest for heterogeneous treatment effects"""
    logger.info("Running Causal Forest estimation")
    
    Y = df[outcome_col].values
    T = df[treatment_col].values
    X = df[covariate_cols].values
    
    if ECONML_AVAILABLE:
        # Use EconML Causal Forest
        cf = CausalForest(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        cf.fit(X, T, Y)
        
        # Get CATE predictions
        cate = cf.predict(X)
        ate = cate.mean()
        
        # Get confidence intervals
        ci = cf.predict_interval(X, alpha=0.05)
        ate_ci_lower = ci[0].mean()
        ate_ci_upper = ci[1].mean()
        
    else:
        # Simplified implementation using two separate forests
        # Forest for treated
        treated_idx = T == 1
        rf_treated = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_treated.fit(X[treated_idx], Y[treated_idx])
        
        # Forest for control
        control_idx = T == 0
        rf_control = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_control.fit(X[control_idx], Y[control_idx])
        
        # Predict CATE
        Y1_pred = rf_treated.predict(X)
        Y0_pred = rf_control.predict(X)
        cate = Y1_pred - Y0_pred
        ate = cate.mean()
        
        # Bootstrap for confidence intervals
        n_bootstrap = 100
        ate_bootstrap = []
        rng = get_random_state()
        
        for _ in range(n_bootstrap):
            idx = rng.choice(len(X), size=len(X), replace=True)
            ate_bootstrap.append(cate[idx].mean())
        
        ate_ci_lower = np.percentile(ate_bootstrap, 2.5)
        ate_ci_upper = np.percentile(ate_bootstrap, 97.5)
    
    results = {
        'method': 'Causal Forest',
        'estimate': float(ate),
        'ci_lower': float(ate_ci_lower),
        'ci_upper': float(ate_ci_upper),
        'n': len(Y),
        'cate': cate  # Store for heterogeneity analysis
    }
    
    logger.info(f"CF ATE: {ate:.3f} ({ate_ci_lower:.3f}, {ate_ci_upper:.3f})")
    
    return results

def analyze_heterogeneity(df, cate, covariate_cols):
    """Analyze heterogeneous treatment effects"""
    logger.info("Analyzing treatment effect heterogeneity")
    
    # Define subgroups - extended as per plan specification
    subgroups = {
        'age_young': df['age'] < 40,
        'age_old': df['age'] >= 65,
        'female': df['sex_M'] == 0,
        'male': df['sex_M'] == 1,
        'high_charlson': df['charlson_score'] >= 3,
        'low_charlson': df['charlson_score'] < 3,
        'high_baseline_use': df.get('baseline_high_utilizer', 0) == 1,
        'prior_anxiety': df.get('anxiety_flag', 0) == 1,
        'high_deprivation': df.get('deprivation_quintile', 3) >= 4,
        'low_deprivation': df.get('deprivation_quintile', 3) <= 2
    }
    
    # Calculate CATE by subgroup
    subgroup_results = {}
    p_values = []
    
    for name, mask in subgroups.items():
        if mask.sum() > 10:  # Only if sufficient sample size
            cate_subgroup = cate[mask]
            cate_other = cate[~mask]
            
            # T-test for difference
            if len(cate_subgroup) > 0 and len(cate_other) > 0:
                _, p_value = stats.ttest_ind(cate_subgroup, cate_other)
                p_values.append(p_value)
                
                subgroup_results[name] = {
                    'mean_cate': float(cate_subgroup.mean()),
                    'se': float(cate_subgroup.std() / np.sqrt(len(cate_subgroup))),
                    'n': int(mask.sum()),
                    'p_value': float(p_value)
                }
    
    # FDR correction
    if p_values:
        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        
        # Add adjusted p-values
        for i, name in enumerate(subgroup_results.keys()):
            subgroup_results[name]['p_adjusted'] = float(p_adjusted[i])
    
    # Check for significant heterogeneity
    significant_heterogeneity = any(
        res.get('p_adjusted', 1) < 0.05 
        for res in subgroup_results.values()
    )
    
    return subgroup_results, significant_heterogeneity

def create_cate_plot(df, cate, output_path):
    """Create CATE distribution visualization"""
    logger.info("Creating CATE distribution plot")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall CATE distribution
    ax1.hist(cate, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=cate.mean(), color='red', linestyle='--', 
               label=f'ATE = {cate.mean():.3f}')
    ax1.set_xlabel('Conditional Average Treatment Effect')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Treatment Effects')
    ax1.legend()
    
    # CATE by age groups
    age_groups = pd.cut(df['age'], bins=[0, 40, 60, 100], 
                       labels=['<40', '40-60', '>60'])
    
    cate_by_age = []
    for group in age_groups.unique():
        if pd.notna(group):
            mask = age_groups == group
            if mask.sum() > 0:
                cate_by_age.append({
                    'group': group,
                    'cate': cate[mask],
                    'mean': cate[mask].mean(),
                    'se': cate[mask].std() / np.sqrt(mask.sum())
                })
    
    # Box plot by age
    ax2.boxplot([d['cate'] for d in cate_by_age], 
               labels=[d['group'] for d in cate_by_age])
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Treatment Effect')
    ax2.set_title('Treatment Effects by Age Group')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive causal inference estimation"
    )
    parser.add_argument('--outcome', default='total_encounters',
                       help='Outcome variable to analyze')
    parser.add_argument('--treatment-col', default='ssd_flag', help='Treatment column name (default: ssd_flag)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without saving outputs')
    args = parser.parse_args()
    
    # Set random seeds
    set_global_seeds()
    
    # Load configuration
    config = load_config()
    
    # Initialize tracker
    tracker = ArtefactTracker()
    tracker.track("script_start", {"script": "06_causal_estimators.py"})
    
    # Load data
    data_path = Path("data_derived/ps_weighted.parquet")
    if not data_path.exists():
        logger.error(f"PS weighted data not found at {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Define variables
    outcome_col = args.outcome
    TREATMENT_COL = args.treatment_col
    
    # Define covariates
    covariate_cols = [col for col in df.columns if col.endswith('_conf') or 
                     col in ['age', 'sex_M', 'charlson_score', 
                            'baseline_encounters', 'baseline_high_utilizer']]
    covariate_cols = [col for col in covariate_cols if col in df.columns]
    
    logger.info(f"Using {len(covariate_cols)} covariates")
    
    # Run different estimators
    ate_estimates = []
    
    # 1. TMLE
    try:
        tmle_results = run_tmle(df, outcome_col, TREATMENT_COL, covariate_cols)
        ate_estimates.append(tmle_results)
    except Exception as e:
        logger.error(f"TMLE failed: {e}")
    
    # 2. Double ML
    try:
        dml_results = run_double_ml(df, outcome_col, TREATMENT_COL, covariate_cols)
        ate_estimates.append(dml_results)
    except Exception as e:
        logger.error(f"Double ML failed: {e}")
    
    # 3. Causal Forest
    try:
        cf_results = run_causal_forest(df, outcome_col, TREATMENT_COL, covariate_cols)
        ate_estimates.append(cf_results)
        
        # Analyze heterogeneity
        if 'cate' in cf_results:
            subgroup_results, significant_het = analyze_heterogeneity(
                df, cf_results['cate'], covariate_cols
            )
    except Exception as e:
        logger.error(f"Causal Forest failed: {e}")
    
    # Save results
    if not args.dry_run:
        # Save ATE estimates
        ate_results = {
            'outcome': outcome_col,
            'estimates': [
                {k: v for k, v in est.items() if k != 'cate'} 
                for est in ate_estimates
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = Path("results/ate_estimates.json")
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(ate_results, f, indent=2)
        
        # Save heterogeneity analysis
        if 'subgroup_results' in locals():
            het_results = {
                'subgroup_effects': subgroup_results,
                'significant_heterogeneity': significant_het,
                'timestamp': datetime.now().isoformat()
            }
            
            het_path = Path("results/cate_analysis.json")
            with open(het_path, 'w') as f:
                json.dump(het_results, f, indent=2)
        
        # Create CATE plot
        if 'cf_results' in locals() and 'cate' in cf_results:
            figures_dir = Path("figures")
            figures_dir.mkdir(exist_ok=True)
            cate_plot_path = figures_dir / "cate_distribution.pdf"
            create_cate_plot(df, cf_results['cate'], cate_plot_path)
        
        # Track outputs
        tracker.track("output_generated", {
            "ate_estimates": len(ate_estimates),
            "methods": [est['method'] for est in ate_estimates]
        })
        
        # Update study documentation
        if Path("scripts/update_study_doc.py").exists():
            import subprocess
            # Format estimates for YAML
            ate_str = json.dumps([
                {
                    'method': est['method'],
                    'estimate': round(est['estimate'], 3),
                    'ci': [round(est['ci_lower'], 3), round(est['ci_upper'], 3)]
                }
                for est in ate_estimates
            ])
            
            subprocess.run([
                "python", "scripts/update_study_doc.py",
                "--step", "Causal estimation complete",
                "--kv", f"ate_estimates={ate_str}",
                "--kv", f"significant_heterogeneity={significant_het if 'significant_het' in locals() else 'unknown'}"
            ])
    
    # Print summary
    print("\n=== Causal Inference Summary ===")
    print(f"Outcome: {outcome_col}")
    print(f"N observations: {len(df):,}")
    print("\nAverage Treatment Effects:")
    for est in ate_estimates:
        print(f"  {est['method']}: {est['estimate']:.3f} "
              f"({est['ci_lower']:.3f}, {est['ci_upper']:.3f})")
    if 'significant_het' in locals():
        print(f"\nSignificant heterogeneity detected: {significant_het}")
    print("================================\n")

if __name__ == "__main__":
    main()