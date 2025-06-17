#!/usr/bin/env python3
"""
test_advanced_analyses.py - Tests for H4-H6 advanced causal analyses

Tests mediation analysis, causal forest, and G-computation implementations
for Week 4 advanced analysis requirements.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAdvancedAnalyses:
    """Test H4-H6 advanced causal analysis implementations"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n = 500
        
        # Generate covariates
        age = np.random.normal(45, 15, n)
        sex = np.random.binomial(1, 0.6, n)  # 60% female
        comorbidity = np.random.poisson(2, n)
        
        # Generate treatment (exposure) based on covariates
        treatment_logit = -1 + 0.02*age + 0.5*sex + 0.1*comorbidity + np.random.normal(0, 0.5, n)
        treatment_prob = 1 / (1 + np.exp(-treatment_logit))
        treatment = np.random.binomial(1, treatment_prob, n)
        
        # Generate mediator (psychiatric referrals) based on treatment and covariates
        mediator_logit = -0.5 + 1.2*treatment + 0.01*age + 0.3*sex + np.random.normal(0, 0.3, n)
        mediator_prob = 1 / (1 + np.exp(-mediator_logit))
        mediator = np.random.binomial(1, mediator_prob, n)
        
        # Generate outcome based on treatment, mediator, and covariates
        outcome = (2 + 0.8*treatment + 1.5*mediator + 0.02*age + 0.5*sex + 
                  0.1*comorbidity + np.random.normal(0, 1, n))
        
        return pd.DataFrame({
            'patient_id': range(n),
            'age': age,
            'sex': sex,
            'comorbidity_count': comorbidity,
            'ssd_exposure': treatment,
            'psychiatric_referrals': mediator,
            'mh_encounters': outcome
        })
    
    def test_mediation_analysis_basic(self, sample_data):
        """Test basic mediation analysis functionality"""
        from advanced_analyses import MediationAnalysis
        
        mediation = MediationAnalysis(outcome_type='continuous')
        
        results = mediation.fit_mediation_models(
            X=sample_data,
            treatment=sample_data['ssd_exposure'],
            mediator=sample_data['psychiatric_referrals'],
            outcome=sample_data['mh_encounters'],
            covariates=['age', 'sex', 'comorbidity_count']
        )
        
        # Check required outputs
        assert 'total_effect' in results
        assert 'direct_effect' in results
        assert 'indirect_effect' in results
        assert 'proportion_mediated' in results
        assert 'sobel_p' in results
        assert 'significant_mediation' in results
        
        # Check reasonable values
        assert isinstance(results['total_effect'], float)
        assert isinstance(results['indirect_effect'], float)
        assert 0 <= abs(results['proportion_mediated']) <= 2  # Allow for some noise
        assert 0 <= results['sobel_p'] <= 1
    
    def test_mediation_analysis_binary_outcome(self, sample_data):
        """Test mediation analysis with binary outcome"""
        from advanced_analyses import MediationAnalysis
        
        # Convert outcome to binary
        binary_outcome = (sample_data['mh_encounters'] > sample_data['mh_encounters'].median()).astype(int)
        
        mediation = MediationAnalysis(outcome_type='binary')
        
        results = mediation.fit_mediation_models(
            X=sample_data,
            treatment=sample_data['ssd_exposure'],
            mediator=sample_data['psychiatric_referrals'],
            outcome=binary_outcome,
            covariates=['age', 'sex', 'comorbidity_count']
        )
        
        assert 'total_effect' in results
        assert 'indirect_effect' in results
        assert results['n_obs'] > 0
    
    def test_causal_forest_analysis(self, sample_data):
        """Test causal forest heterogeneous effects analysis"""
        from advanced_analyses import CausalForestAnalysis
        
        causal_forest = CausalForestAnalysis(n_estimators=50, random_state=42)
        
        results = causal_forest.estimate_heterogeneous_effects(
            X=sample_data,
            treatment=sample_data['ssd_exposure'],
            outcome=sample_data['mh_encounters'],
            covariates=['age', 'sex', 'comorbidity_count']
        )
        
        # Check required outputs
        assert 'tau_estimates' in results
        assert 'tau_mean' in results
        assert 'tau_std' in results
        assert 'high_responders' in results
        assert 'low_responders' in results
        assert 'het_ranking' in results
        
        # Check array dimensions
        assert len(results['tau_estimates']) == len(sample_data)
        assert len(results['high_responders']) == len(sample_data)
        assert len(results['low_responders']) == len(sample_data)
        
        # Check heterogeneity ranking structure
        het_ranking = results['het_ranking']
        assert 'variable' in het_ranking.columns
        assert 'het_importance' in het_ranking.columns
        assert len(het_ranking) == 3  # Number of covariates
    
    def test_gcomputation_analysis(self, sample_data):
        """Test G-computation intervention simulation"""
        from advanced_analyses import GComputationAnalysis
        
        gcomp = GComputationAnalysis()
        
        intervention_scenarios = {
            'status_quo': 0.3,
            'reduced_exposure': 0.1,
            'increased_exposure': 0.6
        }
        
        results = gcomp.simulate_interventions(
            X=sample_data,
            treatment=sample_data['ssd_exposure'],
            outcome=sample_data['mh_encounters'],
            covariates=['age', 'sex', 'comorbidity_count'],
            intervention_scenarios=intervention_scenarios
        )
        
        # Check overall structure
        assert 'simulations' in results
        assert 'outcome_model' in results
        assert 'n_observations' in results
        assert 'baseline_outcome' in results
        
        # Check simulation results for each scenario
        simulations = results['simulations']
        for scenario in intervention_scenarios.keys():
            assert scenario in simulations
            sim_result = simulations[scenario]
            
            assert 'mean_outcome' in sim_result
            assert 'ci_lower' in sim_result
            assert 'ci_upper' in sim_result
            assert 'risk_difference' in sim_result
            
            # Check confidence interval ordering
            assert sim_result['ci_lower'] <= sim_result['mean_outcome'] <= sim_result['ci_upper']
    
    def test_run_advanced_analyses_integration(self, sample_data, tmp_path):
        """Test integrated execution of all H4-H6 analyses"""
        from advanced_analyses import run_advanced_analyses
        
        results = run_advanced_analyses(
            cohort_df=sample_data,
            exposure_col='ssd_exposure',
            outcome_col='mh_encounters',
            mediator_col='psychiatric_referrals',
            covariates=['age', 'sex', 'comorbidity_count'],
            output_dir=tmp_path
        )
        
        # Check all analyses completed
        assert 'h4_mediation' in results
        assert 'h5_causal_forest' in results
        assert 'h6_gcomputation' in results
        
        # Check no analysis failed with errors
        for analysis_name, analysis_results in results.items():
            assert 'error' not in analysis_results, f"{analysis_name} failed with error"
        
        # Check report generation
        report_path = tmp_path / 'advanced_analyses_report.md'
        assert report_path.exists(), "Analysis report not generated"
        
        # Check report content
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        assert 'H4: Mediation Analysis' in report_content
        assert 'H5: Causal Forest' in report_content
        assert 'H6: G-computation' in report_content
    
    def test_mediation_robustness_missing_data(self, sample_data):
        """Test mediation analysis handles missing data appropriately"""
        from advanced_analyses import MediationAnalysis
        
        # Introduce missing data
        sample_with_missing = sample_data.copy()
        missing_indices = np.random.choice(len(sample_data), size=50, replace=False)
        sample_with_missing.loc[missing_indices, 'age'] = np.nan
        
        mediation = MediationAnalysis()
        
        results = mediation.fit_mediation_models(
            X=sample_with_missing,
            treatment=sample_with_missing['ssd_exposure'],
            mediator=sample_with_missing['psychiatric_referrals'],
            outcome=sample_with_missing['mh_encounters'],
            covariates=['age', 'sex', 'comorbidity_count']
        )
        
        # Should handle missing data by excluding
        assert results['n_obs'] == len(sample_data) - 50
        assert 'total_effect' in results
    
    def test_causal_forest_feature_importance(self, sample_data):
        """Test causal forest produces reasonable feature importance rankings"""
        from advanced_analyses import CausalForestAnalysis
        
        # Add noise variables to test ranking
        sample_enhanced = sample_data.copy()
        sample_enhanced['noise1'] = np.random.normal(0, 1, len(sample_data))
        sample_enhanced['noise2'] = np.random.normal(0, 1, len(sample_data))
        
        causal_forest = CausalForestAnalysis(n_estimators=30)
        
        results = causal_forest.estimate_heterogeneous_effects(
            X=sample_enhanced,
            treatment=sample_enhanced['ssd_exposure'],
            outcome=sample_enhanced['mh_encounters'],
            covariates=['age', 'sex', 'comorbidity_count', 'noise1', 'noise2']
        )
        
        het_ranking = results['het_ranking']
        
        # Check that noise variables tend to rank lower
        # (This is probabilistic, so we don't enforce strict ordering)
        assert len(het_ranking) == 5
        assert all(var in het_ranking['variable'].values for var in ['age', 'sex', 'comorbidity_count', 'noise1', 'noise2'])
    
    def test_gcomputation_scenario_comparison(self, sample_data):
        """Test G-computation scenario comparisons"""
        from advanced_analyses import GComputationAnalysis
        
        gcomp = GComputationAnalysis()
        
        intervention_scenarios = {
            'baseline': 0.2,
            'intervention': 0.8
        }
        
        results = gcomp.simulate_interventions(
            X=sample_data,
            treatment=sample_data['ssd_exposure'],
            outcome=sample_data['mh_encounters'],
            covariates=['age', 'sex', 'comorbidity_count'],
            intervention_scenarios=intervention_scenarios
        )
        
        # Check comparison calculations
        simulations = results['simulations']
        assert 'comparisons' in simulations
        
        comparisons = simulations['comparisons']
        assert 'intervention_vs_baseline' in comparisons
        
        # Intervention should have different outcome than baseline
        baseline_outcome = simulations['baseline']['mean_outcome']
        intervention_outcome = simulations['intervention']['mean_outcome']
        assert baseline_outcome != intervention_outcome
    
    def test_advanced_analyses_edge_cases(self, tmp_path):
        """Test advanced analyses with edge cases"""
        from advanced_analyses import run_advanced_analyses
        
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'patient_id': [1, 2, 3, 4, 5],
            'age': [30, 40, 50, 60, 70],
            'sex': [0, 1, 0, 1, 0],
            'ssd_exposure': [0, 1, 0, 1, 0],
            'psychiatric_referrals': [0, 0, 1, 1, 0],
            'mh_encounters': [1.0, 2.0, 1.5, 3.0, 1.2]
        })
        
        # Should handle small sample gracefully
        results = run_advanced_analyses(
            cohort_df=minimal_data,
            exposure_col='ssd_exposure',
            outcome_col='mh_encounters',
            mediator_col='psychiatric_referrals',
            covariates=['age', 'sex'],
            output_dir=tmp_path
        )
        
        # All analyses should attempt to run (may have errors due to small sample)
        assert 'h4_mediation' in results
        assert 'h5_causal_forest' in results
        assert 'h6_gcomputation' in results
        
        # Report should still be generated
        report_path = tmp_path / 'advanced_analyses_report.md'
        assert report_path.exists()