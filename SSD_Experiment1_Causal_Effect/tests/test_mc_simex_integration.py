#!/usr/bin/env python3
"""
test_mc_simex_integration.py - Tests for MC-SIMEX integration with pipeline

Tests the integration of bias-corrected flags with propensity score matching
and causal estimation modules.

Author: Ryhan Suny
Date: 2025-06-17
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import load_config
# Import with importlib to handle numeric prefix
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("misclassification_adjust", 
    str(Path(__file__).parent.parent / "src" / "07a_misclassification_adjust.py"))
misclassification_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(misclassification_module)
apply_bias_correction = misclassification_module.apply_bias_correction
from utils.global_seeds import set_global_seeds

class TestMCSimexIntegration:
    """Test MC-SIMEX integration with pipeline"""
    
    @pytest.fixture
    def sample_cohort(self):
        """Create sample cohort for testing"""
        set_global_seeds(42)
        n = 1000
        
        # Create synthetic cohort
        data = {
            'patient_id': range(n),
            'age': np.random.normal(45, 15, n),
            'sex_M': np.random.binomial(1, 0.5, n),
            'charlson_score': np.random.poisson(1.5, n),
            'baseline_encounters': np.random.poisson(3, n),
            'baseline_high_utilizer': np.random.binomial(1, 0.2, n),
            'ssd_flag': np.random.binomial(1, 0.15, n)
        }
        
        df = pd.DataFrame(data)
        # Ensure age and charlson_score are reasonable
        df['age'] = np.clip(df['age'], 18, 95)
        df['charlson_score'] = np.clip(df['charlson_score'], 0, 10)
        
        return df
    
    @pytest.fixture
    def config_with_bias_correction(self):
        """Config with bias correction enabled"""
        config = {
            'mc_simex': {
                'enabled': True,
                'sensitivity': 0.82,
                'specificity': 0.82,
                'bootstrap_samples': 10,  # Reduced for testing
                'use_bias_corrected_flag': True
            },
            'misclassification': {
                'sensitivity': 0.82,
                'specificity': 0.82,
                'simex_B': 10
            }
        }
        return config
    
    @pytest.fixture
    def config_without_bias_correction(self):
        """Config with bias correction disabled"""
        config = {
            'mc_simex': {
                'enabled': True,
                'sensitivity': 0.82,
                'specificity': 0.82,
                'bootstrap_samples': 10,
                'use_bias_corrected_flag': False
            },
            'misclassification': {
                'sensitivity': 0.82,
                'specificity': 0.82,
                'simex_B': 10
            }
        }
        return config
    
    def test_mc_simex_creates_bias_corrected_flag(self, sample_cohort, config_with_bias_correction):
        """Test that MC-SIMEX creates ssd_flag_adj column"""
        # Apply bias correction
        corrected_df, results = apply_bias_correction(
            sample_cohort, config_with_bias_correction
        )
        
        # Check that ssd_flag_adj was created
        assert 'ssd_flag_adj' in corrected_df.columns
        assert 'ssd_flag_naive' in corrected_df.columns
        assert 'bias_correction_applied' in corrected_df.columns
        
        # Check that flags are different (should be with random correction)
        original_sum = sample_cohort['ssd_flag'].sum()
        corrected_sum = corrected_df['ssd_flag_adj'].sum()
        
        # Flags should be different due to bias correction
        # Given sensitivity/specificity of 0.82, we expect some changes
        assert original_sum != corrected_sum, "Bias correction should change the flag counts"
        
        # Check results structure
        assert 'sensitivity' in results
        assert 'specificity' in results
        assert results['sensitivity'] == 0.82
        assert results['specificity'] == 0.82
    
    def test_flag_selection_logic(self, sample_cohort):
        """Test logic for selecting between original and bias-corrected flags"""
        # Test with bias correction enabled
        config_enabled = {'mc_simex': {'use_bias_corrected_flag': True}}
        
        # Add ssd_flag_adj to dataframe
        sample_cohort['ssd_flag_adj'] = 1 - sample_cohort['ssd_flag']  # Flip for testing
        
        # Logic should select ssd_flag_adj
        treatment_col = get_treatment_column(sample_cohort, config_enabled)
        assert treatment_col == 'ssd_flag_adj'
        
        # Test with bias correction disabled
        config_disabled = {'mc_simex': {'use_bias_corrected_flag': False}}
        treatment_col = get_treatment_column(sample_cohort, config_disabled)
        assert treatment_col == 'ssd_flag'
    
    def test_estimates_differ_with_correction(self, sample_cohort, config_with_bias_correction, config_without_bias_correction):
        """Test that estimates differ when using bias-corrected flag"""
        # Apply bias correction to create ssd_flag_adj
        corrected_df, _ = apply_bias_correction(sample_cohort, config_with_bias_correction)
        
        # Simulate simple treatment effect estimation
        # Original flag
        original_treated = corrected_df[corrected_df['ssd_flag'] == 1]['baseline_encounters'].mean()
        original_control = corrected_df[corrected_df['ssd_flag'] == 0]['baseline_encounters'].mean()
        original_effect = original_treated - original_control
        
        # Bias-corrected flag  
        corrected_treated = corrected_df[corrected_df['ssd_flag_adj'] == 1]['baseline_encounters'].mean()
        corrected_control = corrected_df[corrected_df['ssd_flag_adj'] == 0]['baseline_encounters'].mean()
        corrected_effect = corrected_treated - corrected_control
        
        # Effects should be different (unless by extreme chance)
        assert abs(original_effect - corrected_effect) > 0.01  # Some meaningful difference
    
    def test_missing_ssd_flag_adj_handling(self, sample_cohort):
        """Test handling when ssd_flag_adj is requested but doesn't exist"""
        config = {'mc_simex': {'use_bias_corrected_flag': True}}
        
        # Should fall back to ssd_flag if ssd_flag_adj doesn't exist
        treatment_col = get_treatment_column(sample_cohort, config)
        assert treatment_col == 'ssd_flag'
    
    def test_integration_with_ps_matching(self, sample_cohort, config_with_bias_correction):
        """Test integration with propensity score matching"""
        # This would test the actual integration with 05_ps_match.py
        # For now, just test the logic of treatment column selection
        
        # Apply bias correction
        corrected_df, _ = apply_bias_correction(sample_cohort, config_with_bias_correction)
        
        # Test treatment column selection
        treatment_col = get_treatment_column(corrected_df, config_with_bias_correction)
        assert treatment_col == 'ssd_flag_adj'
        
        # Check that the selected column exists and has valid values
        assert treatment_col in corrected_df.columns
        assert corrected_df[treatment_col].isin([0, 1]).all()
        assert corrected_df[treatment_col].sum() > 0  # Should have some treated units


def get_treatment_column(df, config):
    """
    Helper function to determine which treatment column to use
    This mimics the logic that should be in 05_ps_match.py and 06_causal_estimators.py
    """
    use_bias_corrected = config.get('mc_simex', {}).get('use_bias_corrected_flag', False)
    
    if use_bias_corrected and 'ssd_flag_adj' in df.columns:
        return 'ssd_flag_adj'
    else:
        return 'ssd_flag'


if __name__ == "__main__":
    pytest.main([__file__])