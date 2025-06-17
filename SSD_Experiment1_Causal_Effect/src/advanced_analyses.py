#!/usr/bin/env python3
"""
advanced_analyses.py - H4-H6 Advanced Causal Inference Analyses

Implements mediation analysis, effect modification via causal forest, 
and G-computation for intervention simulation as required for Week 4.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediationAnalysis:
    """
    H4: Mediation analysis for causal pathways
    
    Tests whether psychiatric referral patterns mediate the relationship
    between SSD exposure and mental health service utilization.
    """
    
    def __init__(self, outcome_type: str = 'continuous'):
        """
        Initialize mediation analysis
        
        Parameters:
        -----------
        outcome_type : str
            Type of outcome ('continuous' or 'binary')
        """
        self.outcome_type = outcome_type
        self.results = {}
        
    def fit_mediation_models(self, 
                           X: pd.DataFrame, 
                           treatment: pd.Series,
                           mediator: pd.Series, 
                           outcome: pd.Series,
                           covariates: List[str]) -> Dict[str, Any]:
        """
        Fit mediation analysis models using Baron & Kenny approach
        
        Parameters:
        -----------
        X : pd.DataFrame
            Covariate data
        treatment : pd.Series
            Treatment/exposure variable
        mediator : pd.Series
            Mediator variable (e.g., psychiatric referrals)
        outcome : pd.Series
            Outcome variable
        covariates : List[str]
            List of covariate column names
            
        Returns:
        --------
        Dict[str, Any]
            Mediation analysis results
        """
        logger.info("Performing mediation analysis...")
        
        # Create analysis dataset
        data = X[covariates].copy()
        data['treatment'] = treatment
        data['mediator'] = mediator  
        data['outcome'] = outcome
        
        # Remove missing values
        data_clean = data.dropna()
        logger.info(f"Analysis sample: {len(data_clean):,} observations")
        
        # Step 1: Treatment -> Outcome (total effect)
        model_total = self._fit_outcome_model(
            data_clean[covariates + ['treatment']], 
            data_clean['outcome']
        )
        if hasattr(model_total, 'coef_'):
            # Get treatment coefficient (last column)
            total_effect = model_total.coef_[-1] if len(model_total.coef_) > 1 else model_total.coef_[0]
        else:
            total_effect = np.nan
        
        # Step 2: Treatment -> Mediator
        model_mediator = LinearRegression()
        model_mediator.fit(
            data_clean[covariates + ['treatment']], 
            data_clean['mediator']
        )
        a_path = model_mediator.coef_[-1]  # Treatment coefficient
        
        # Step 3: Treatment + Mediator -> Outcome (direct effect)
        model_direct = self._fit_outcome_model(
            data_clean[covariates + ['treatment', 'mediator']], 
            data_clean['outcome']
        )
        if hasattr(model_direct, 'coef_') and len(model_direct.coef_) >= 2:
            direct_effect = model_direct.coef_[-2]  # Treatment coef
            b_path = model_direct.coef_[-1]  # Mediator coef
        else:
            direct_effect = np.nan
            b_path = np.nan
        
        # Calculate mediation effects
        indirect_effect = a_path * b_path
        proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0
        
        # Sobel test for significance
        sobel_se = np.sqrt((b_path**2 * np.var(data_clean['mediator'])) + 
                          (a_path**2 * np.var(data_clean['outcome'])))
        sobel_z = indirect_effect / sobel_se if sobel_se > 0 else 0
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
        
        results = {
            'total_effect': total_effect,
            'direct_effect': direct_effect, 
            'indirect_effect': indirect_effect,
            'proportion_mediated': proportion_mediated,
            'a_path': a_path,
            'b_path': b_path,
            'sobel_z': sobel_z,
            'sobel_p': sobel_p,
            'n_obs': len(data_clean),
            'significant_mediation': sobel_p < 0.05
        }
        
        logger.info(f"Mediation results: Indirect effect = {indirect_effect:.3f}, "
                   f"Proportion mediated = {proportion_mediated:.1%}, "
                   f"Sobel p = {sobel_p:.3f}")
        
        self.results = results
        return results
    
    def _fit_outcome_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit appropriate model based on outcome type"""
        if self.outcome_type == 'binary':
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            # For logistic regression, coef_ is 2D, flatten it
            if hasattr(model, 'coef_') and model.coef_.ndim > 1:
                model.coef_ = model.coef_.flatten()
        else:
            model = LinearRegression()
            model.fit(X, y)
        
        return model


class CausalForestAnalysis:
    """
    H5: Heterogeneous treatment effect estimation via Causal Forest
    
    Identifies patient subgroups with differential response to SSD exposure
    using modified random forest approach.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize causal forest analysis
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in forest
        random_state : int
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.tau_forest = None
        self.results = {}
        
    def estimate_heterogeneous_effects(self,
                                     X: pd.DataFrame,
                                     treatment: pd.Series, 
                                     outcome: pd.Series,
                                     covariates: List[str]) -> Dict[str, Any]:
        """
        Estimate conditional average treatment effects using causal forest
        
        Parameters:
        -----------
        X : pd.DataFrame
            Covariate data
        treatment : pd.Series
            Binary treatment indicator
        outcome : pd.Series
            Outcome variable
        covariates : List[str]
            Covariates for heterogeneity analysis
            
        Returns:
        --------
        Dict[str, Any]
            Heterogeneous treatment effect results
        """
        logger.info("Estimating heterogeneous treatment effects...")
        
        # Create analysis dataset
        data = X[covariates].copy()
        data['treatment'] = treatment
        data['outcome'] = outcome
        data_clean = data.dropna()
        
        # Split into treatment groups
        treated = data_clean[data_clean['treatment'] == 1]
        control = data_clean[data_clean['treatment'] == 0]
        
        logger.info(f"Treatment groups: {len(treated):,} treated, {len(control):,} control")
        
        # Estimate propensity scores
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(data_clean[covariates], data_clean['treatment'])
        propensity_scores = ps_model.predict_proba(data_clean[covariates])[:, 1]
        
        # Causal forest approximation using separate outcome models
        # Fit outcome models for each treatment group
        y1_model = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        y0_model = RandomForestRegressor(
            n_estimators=self.n_estimators, 
            random_state=self.random_state
        )
        
        # Train models
        y1_model.fit(treated[covariates], treated['outcome'])
        y0_model.fit(control[covariates], control['outcome'])
        
        # Predict potential outcomes for all individuals
        y1_pred = y1_model.predict(data_clean[covariates])
        y0_pred = y0_model.predict(data_clean[covariates])
        
        # Calculate individual treatment effects (ITE)
        tau_estimates = y1_pred - y0_pred
        
        # Analyze heterogeneity
        tau_mean = np.mean(tau_estimates)
        tau_std = np.std(tau_estimates)
        tau_q25, tau_q75 = np.percentile(tau_estimates, [25, 75])
        
        # Identify high/low response groups
        high_responders = tau_estimates > np.percentile(tau_estimates, 75)
        low_responders = tau_estimates < np.percentile(tau_estimates, 25)
        
        # Feature importance for heterogeneity
        importance_treated = y1_model.feature_importances_
        importance_control = y0_model.feature_importances_
        
        # Variable importance for treatment effect heterogeneity
        het_importance = np.abs(importance_treated - importance_control)
        het_ranking = pd.DataFrame({
            'variable': covariates,
            'het_importance': het_importance
        }).sort_values('het_importance', ascending=False)
        
        results = {
            'tau_estimates': tau_estimates,
            'tau_mean': tau_mean,
            'tau_std': tau_std, 
            'tau_q25': tau_q25,
            'tau_q75': tau_q75,
            'high_responders': high_responders,
            'low_responders': low_responders,
            'n_high_responders': np.sum(high_responders),
            'n_low_responders': np.sum(low_responders),
            'het_ranking': het_ranking,
            'propensity_scores': propensity_scores,
            'y1_pred': y1_pred,
            'y0_pred': y0_pred
        }
        
        logger.info(f"Heterogeneous effects: Mean τ = {tau_mean:.3f} ± {tau_std:.3f}")
        logger.info(f"High responders: {np.sum(high_responders):,} ({np.mean(high_responders)*100:.1f}%)")
        logger.info(f"Top heterogeneity variables: {', '.join(het_ranking.head(3)['variable'].tolist())}")
        
        self.results = results
        return results


class GComputationAnalysis:
    """
    H6: G-computation for intervention simulation
    
    Simulates population-level effects of hypothetical interventions
    on SSD exposure patterns.
    """
    
    def __init__(self):
        """Initialize G-computation analysis"""
        self.outcome_model = None
        self.results = {}
        
    def simulate_interventions(self,
                             X: pd.DataFrame,
                             treatment: pd.Series,
                             outcome: pd.Series, 
                             covariates: List[str],
                             intervention_scenarios: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate population outcomes under different intervention scenarios
        
        Parameters:
        -----------
        X : pd.DataFrame
            Covariate data
        treatment : pd.Series
            Current treatment patterns
        outcome : pd.Series
            Observed outcomes
        covariates : List[str]
            Baseline covariates
        intervention_scenarios : Dict[str, float]
            Scenarios to simulate {name: treatment_probability}
            
        Returns:
        --------
        Dict[str, Any]
            G-computation simulation results
        """
        logger.info("Running G-computation intervention simulations...")
        
        # Create analysis dataset
        data = X[covariates].copy()
        data['treatment'] = treatment
        data['outcome'] = outcome
        data_clean = data.dropna()
        
        # Fit outcome model
        outcome_model = self._fit_outcome_model(
            data_clean[covariates + ['treatment']], 
            data_clean['outcome']
        )
        
        # Simulate interventions
        simulation_results = {}
        
        for scenario_name, intervention_prob in intervention_scenarios.items():
            logger.info(f"Simulating {scenario_name} (P(treatment) = {intervention_prob:.2f})")
            
            # Generate intervention assignments
            n_obs = len(data_clean)
            intervention_treatment = np.random.binomial(1, intervention_prob, n_obs)
            
            # Create intervention dataset
            intervention_data = data_clean[covariates].copy()
            intervention_data['treatment'] = intervention_treatment
            
            # Predict outcomes under intervention
            intervention_outcomes = outcome_model.predict(intervention_data)
            
            # Calculate population statistics
            mean_outcome = np.mean(intervention_outcomes)
            outcome_var = np.var(intervention_outcomes)
            
            # Calculate risk difference vs. observed
            observed_mean = np.mean(data_clean['outcome'])
            risk_difference = mean_outcome - observed_mean
            
            # Bootstrap confidence interval
            n_bootstrap = 100
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                boot_indices = np.random.choice(n_obs, n_obs, replace=True)
                boot_treatment = np.random.binomial(1, intervention_prob, n_obs)
                boot_data = data_clean.iloc[boot_indices][covariates].copy()
                boot_data['treatment'] = boot_treatment
                boot_outcomes = outcome_model.predict(boot_data)
                bootstrap_means.append(np.mean(boot_outcomes))
            
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            
            simulation_results[scenario_name] = {
                'intervention_prob': intervention_prob,
                'mean_outcome': mean_outcome,
                'outcome_variance': outcome_var,
                'risk_difference': risk_difference,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_treated': np.sum(intervention_treatment),
                'treatment_rate': np.mean(intervention_treatment)
            }
            
            logger.info(f"{scenario_name}: Mean outcome = {mean_outcome:.3f} "
                       f"(95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        
        # Calculate intervention comparisons
        scenario_names = list(intervention_scenarios.keys())
        if len(scenario_names) >= 2:
            base_scenario = scenario_names[0]
            comparisons = {}
            
            for scenario in scenario_names[1:]:
                diff = (simulation_results[scenario]['mean_outcome'] - 
                       simulation_results[base_scenario]['mean_outcome'])
                comparisons[f"{scenario}_vs_{base_scenario}"] = diff
            
            simulation_results['comparisons'] = comparisons
        
        results = {
            'simulations': simulation_results,
            'outcome_model': outcome_model,
            'n_observations': len(data_clean),
            'baseline_outcome': observed_mean
        }
        
        self.results = results
        return results
    
    def _fit_outcome_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit outcome prediction model"""
        # Use random forest for flexible modeling
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model


def run_advanced_analyses(cohort_df: pd.DataFrame,
                         exposure_col: str,
                         outcome_col: str, 
                         mediator_col: str,
                         covariates: List[str],
                         output_dir: Path) -> Dict[str, Any]:
    """
    Run all H4-H6 advanced analyses
    
    Parameters:
    -----------
    cohort_df : pd.DataFrame
        Analysis cohort
    exposure_col : str
        SSD exposure column name
    outcome_col : str
        Primary outcome column name
    mediator_col : str
        Mediator variable column name
    covariates : List[str]
        Baseline covariates
    output_dir : Path
        Output directory for results
        
    Returns:
    --------
    Dict[str, Any]
        Combined results from all analyses
    """
    logger.info("Running H4-H6 advanced causal analyses...")
    
    # Prepare data
    analysis_data = cohort_df[covariates + [exposure_col, outcome_col, mediator_col]].copy()
    analysis_data = analysis_data.dropna()
    
    logger.info(f"Advanced analysis sample: {len(analysis_data):,} patients")
    
    combined_results = {}
    
    # H4: Mediation Analysis
    try:
        logger.info("=== H4: Mediation Analysis ===")
        mediation = MediationAnalysis(outcome_type='continuous')
        h4_results = mediation.fit_mediation_models(
            X=analysis_data,
            treatment=analysis_data[exposure_col],
            mediator=analysis_data[mediator_col],
            outcome=analysis_data[outcome_col],
            covariates=covariates
        )
        combined_results['h4_mediation'] = h4_results
        
    except Exception as e:
        logger.error(f"H4 mediation analysis failed: {e}")
        combined_results['h4_mediation'] = {'error': str(e)}
    
    # H5: Causal Forest Analysis
    try:
        logger.info("=== H5: Causal Forest Analysis ===")
        causal_forest = CausalForestAnalysis()
        h5_results = causal_forest.estimate_heterogeneous_effects(
            X=analysis_data,
            treatment=analysis_data[exposure_col],
            outcome=analysis_data[outcome_col],
            covariates=covariates
        )
        combined_results['h5_causal_forest'] = h5_results
        
    except Exception as e:
        logger.error(f"H5 causal forest analysis failed: {e}")
        combined_results['h5_causal_forest'] = {'error': str(e)}
    
    # H6: G-computation
    try:
        logger.info("=== H6: G-computation Analysis ===")
        
        # Define intervention scenarios
        current_exposure_rate = analysis_data[exposure_col].mean()
        intervention_scenarios = {
            'status_quo': current_exposure_rate,
            'reduce_50_percent': current_exposure_rate * 0.5,
            'universal_screening': 0.9,
            'eliminate_exposure': 0.1
        }
        
        gcomp = GComputationAnalysis()
        h6_results = gcomp.simulate_interventions(
            X=analysis_data,
            treatment=analysis_data[exposure_col],
            outcome=analysis_data[outcome_col],
            covariates=covariates,
            intervention_scenarios=intervention_scenarios
        )
        combined_results['h6_gcomputation'] = h6_results
        
    except Exception as e:
        logger.error(f"H6 G-computation analysis failed: {e}")
        combined_results['h6_gcomputation'] = {'error': str(e)}
    
    # Save results
    results_path = output_dir / 'advanced_analyses_results.pkl'
    
    try:
        import pickle
        with open(results_path, 'wb') as f:
            pickle.dump(combined_results, f)
        logger.info(f"Advanced analysis results saved: {results_path}")
    except ImportError:
        logger.warning("Pickle not available - saving summary only")
        
    # Create summary report
    create_advanced_analysis_report(combined_results, output_dir)
    
    return combined_results


def create_advanced_analysis_report(results: Dict[str, Any], output_dir: Path):
    """Create summary report for advanced analyses"""
    
    report_content = f"""# Advanced Causal Analyses Report (H4-H6)
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## H4: Mediation Analysis

"""
    
    if 'h4_mediation' in results and 'error' not in results['h4_mediation']:
        med_results = results['h4_mediation']
        report_content += f"""### Results
- **Total Effect**: {med_results.get('total_effect', 'N/A'):.3f}
- **Direct Effect**: {med_results.get('direct_effect', 'N/A'):.3f}  
- **Indirect Effect**: {med_results.get('indirect_effect', 'N/A'):.3f}
- **Proportion Mediated**: {med_results.get('proportion_mediated', 0)*100:.1f}%
- **Sobel Test p-value**: {med_results.get('sobel_p', 'N/A'):.3f}
- **Significant Mediation**: {'Yes' if med_results.get('significant_mediation', False) else 'No'}

"""
    else:
        report_content += "Analysis failed or incomplete.\n\n"
    
    report_content += """## H5: Causal Forest (Heterogeneous Effects)

"""
    
    if 'h5_causal_forest' in results and 'error' not in results['h5_causal_forest']:
        cf_results = results['h5_causal_forest']
        report_content += f"""### Results
- **Mean Treatment Effect**: {cf_results.get('tau_mean', 'N/A'):.3f} ± {cf_results.get('tau_std', 'N/A'):.3f}
- **Effect Range**: {cf_results.get('tau_q25', 'N/A'):.3f} to {cf_results.get('tau_q75', 'N/A'):.3f} (IQR)
- **High Responders**: {cf_results.get('n_high_responders', 'N/A'):,} patients
- **Low Responders**: {cf_results.get('n_low_responders', 'N/A'):,} patients

### Top Variables for Heterogeneity
"""
        if 'het_ranking' in cf_results:
            top_vars = cf_results['het_ranking'].head(5)
            for _, row in top_vars.iterrows():
                report_content += f"- {row['variable']}: {row['het_importance']:.3f}\n"
        
        report_content += "\n"
    else:
        report_content += "Analysis failed or incomplete.\n\n"
    
    report_content += """## H6: G-computation (Intervention Simulation)

"""
    
    if 'h6_gcomputation' in results and 'error' not in results['h6_gcomputation']:
        gc_results = results['h6_gcomputation']
        if 'simulations' in gc_results:
            for scenario, sim_result in gc_results['simulations'].items():
                if scenario != 'comparisons':
                    report_content += f"""### {scenario.replace('_', ' ').title()}
- **Intervention Rate**: {sim_result.get('intervention_prob', 'N/A'):.1%}
- **Mean Outcome**: {sim_result.get('mean_outcome', 'N/A'):.3f}
- **95% CI**: ({sim_result.get('ci_lower', 'N/A'):.3f}, {sim_result.get('ci_upper', 'N/A'):.3f})
- **Risk Difference**: {sim_result.get('risk_difference', 'N/A'):.3f}

"""
    else:
        report_content += "Analysis failed or incomplete.\n\n"
    
    report_content += """
## Clinical Interpretation

These advanced analyses provide insights into:
1. **Causal pathways** through which SSD exposure affects outcomes
2. **Patient heterogeneity** in treatment response  
3. **Population-level impact** of potential interventions

Results should be interpreted in context of study limitations and clinical plausibility.
"""
    
    # Save report
    report_path = output_dir / 'advanced_analyses_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Advanced analysis report saved: {report_path}")


def main():
    """Main execution for advanced analyses"""
    logger.info("Advanced causal analysis module ready")
    
    print("H4-H6 Advanced Analysis Functions:")
    print("  - MediationAnalysis() - Causal pathway analysis")
    print("  - CausalForestAnalysis() - Heterogeneous treatment effects")
    print("  - GComputationAnalysis() - Intervention simulation")
    print("  - run_advanced_analyses() - Execute all H4-H6 analyses")


if __name__ == "__main__":
    main()