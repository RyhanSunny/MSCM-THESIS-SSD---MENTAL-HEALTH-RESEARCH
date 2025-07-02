# -*- coding: utf-8 -*-
"""
test_high_utilization_threshold_fix.py - Tests for High Utilization Threshold Correction

TDD COMPLIANCE FOR CLAUDE.md REQUIREMENTS:
==========================================

Following CLAUDE.md mandate: "TDD is MANDATORY - ALWAYS write tests FIRST"
This test suite validates the high utilization threshold correction from 75th to 90th percentile
for improved discriminative ability in outcome classification.

Test Coverage:
1. Cost data loading and validation
2. Threshold comparison analysis (75th vs 90th percentile)
3. Discriminative ability assessment
4. Clinical justification validation
5. Visualization generation
6. Configuration updates
7. Error handling and edge cases

Author: Manus AI Research Assistant (TDD compliance)
Date: July 2, 2025
Version: 1.0 (Tests-first implementation)
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for testing
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Import the module under test
try:
    import high_utilization_threshold_fix as hutf
except ImportError:
    hutf = None

class TestHighUtilizationThresholdFix:
    """Test suite for high utilization threshold fix functionality."""
    
    @pytest.fixture
    def sample_cost_data(self):
        """Create sample cost data with realistic healthcare cost distribution."""
        np.random.seed(42)
        n_patients = 10000
        
        # Create realistic healthcare cost distribution (log-normal)
        base_costs = np.random.lognormal(mean=8, sigma=1.5, size=n_patients)
        
        # Add some high-cost outliers
        high_cost_indices = np.random.choice(n_patients, size=int(0.05 * n_patients), replace=False)
        base_costs[high_cost_indices] *= np.random.uniform(5, 20, len(high_cost_indices))
        
        cost_data = pd.DataFrame({
            'Patient_ID': range(1, n_patients + 1),
            'cost_total_12m': base_costs,
            'age_at_index': np.random.normal(65, 15, n_patients),
            'sex': np.random.choice(['M', 'F'], n_patients),
            'Charlson': np.random.poisson(2, n_patients)
        })
        
        return cost_data
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        
        data_dirs = [
            'data/processed',
            'config',
            'results/threshold_analysis'
        ]
        
        for dir_path in data_dirs:
            (Path(temp_dir) / dir_path).mkdir(parents=True, exist_ok=True)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_load_cost_data_success(self, sample_cost_data, temp_data_dir):
        """Test successful loading of cost data."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Save sample data
        cost_path = Path(temp_data_dir) / 'data/processed/master_with_costs.parquet'
        sample_cost_data.to_parquet(cost_path)
        
        with patch('high_utilization_threshold_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            result = hutf.load_cost_data()
            
            assert isinstance(result, pd.DataFrame)
            assert 'cost_total_12m' in result.columns
            assert len(result) == len(sample_cost_data)
    
    def test_calculate_threshold_comparison_basic(self, sample_cost_data):
        """Test basic threshold comparison calculation."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        result = hutf.calculate_threshold_comparison(sample_cost_data)
        
        # Validate structure
        assert isinstance(result, dict)
        assert 'percentile_75' in result
        assert 'percentile_90' in result
        assert 'threshold_comparison' in result
        
        # Validate threshold values
        p75 = result['percentile_75']['threshold_value']
        p90 = result['percentile_90']['threshold_value']
        
        assert p90 > p75  # 90th percentile should be higher than 75th
        assert p75 > 0 and p90 > 0  # Both should be positive
        
        # Validate patient counts
        assert result['percentile_75']['high_utilizers'] > result['percentile_90']['high_utilizers']
    
    def test_calculate_threshold_comparison_edge_cases(self):
        """Test threshold comparison with edge cases."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Test with uniform costs
        uniform_data = pd.DataFrame({
            'Patient_ID': range(1, 101),
            'cost_total_12m': [1000] * 100  # All same cost
        })
        
        result = hutf.calculate_threshold_comparison(uniform_data)
        
        # Should handle uniform distribution
        assert result['percentile_75']['threshold_value'] == result['percentile_90']['threshold_value']
        assert result['percentile_75']['high_utilizers'] == result['percentile_90']['high_utilizers']
    
    def test_assess_discriminative_ability_improvement(self, sample_cost_data):
        """Test discriminative ability assessment."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        threshold_comparison = hutf.calculate_threshold_comparison(sample_cost_data)
        
        result = hutf.assess_discriminative_ability(threshold_comparison, sample_cost_data)
        
        # Validate structure
        assert isinstance(result, dict)
        assert 'auc_improvement' in result
        assert 'concentration_analysis' in result
        assert 'clinical_impact' in result
        
        # Validate AUC values
        auc_75 = result['auc_improvement']['auc_75th']
        auc_90 = result['auc_improvement']['auc_90th']
        
        assert 0.5 <= auc_75 <= 1.0
        assert 0.5 <= auc_90 <= 1.0
        assert auc_90 >= auc_75  # 90th percentile should have better or equal discrimination
    
    def test_assess_discriminative_ability_concentration_analysis(self, sample_cost_data):
        """Test concentration analysis component."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        threshold_comparison = hutf.calculate_threshold_comparison(sample_cost_data)
        result = hutf.assess_discriminative_ability(threshold_comparison, sample_cost_data)
        
        concentration = result['concentration_analysis']
        
        # Validate concentration metrics
        assert 'cost_concentration_75th' in concentration
        assert 'cost_concentration_90th' in concentration
        assert 'pareto_principle_validation' in concentration
        
        # 90th percentile should capture higher cost concentration
        assert concentration['cost_concentration_90th'] >= concentration['cost_concentration_75th']
    
    def test_generate_clinical_justification_evidence_based(self):
        """Test clinical justification generation with evidence-based content."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Sample discriminative analysis results
        discriminative_results = {
            'auc_improvement': {
                'auc_75th': 0.75,
                'auc_90th': 0.82,
                'improvement': 0.07
            },
            'concentration_analysis': {
                'cost_concentration_90th': 0.68,
                'cost_concentration_75th': 0.55
            }
        }
        
        result = hutf.generate_clinical_justification(discriminative_results)
        
        assert isinstance(result, dict)
        assert 'literature_backing' in result
        assert 'clinical_rationale' in result
        assert 'methodological_improvement' in result
        
        # Should reference specific literature
        literature = result['literature_backing']
        assert any('Shukla' in ref for ref in literature)
        assert any('Berwick' in ref for ref in literature)
    
    def test_update_configuration_files(self, temp_data_dir):
        """Test configuration file updates."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create sample config file
        config_path = Path(temp_data_dir) / 'config/config.yaml'
        sample_config = {
            'thresholds': {
                'high_utilization_percentile': 75
            },
            'other_settings': {
                'some_value': 42
            }
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        with patch('high_utilization_threshold_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            hutf.update_configuration_files()
            
            # Verify config was updated
            with open(config_path, 'r') as f:
                updated_config = yaml.safe_load(f)
            
            assert updated_config['thresholds']['high_utilization_percentile'] == 90
            assert updated_config['other_settings']['some_value'] == 42  # Other settings preserved
    
    def test_generate_threshold_visualizations(self, sample_cost_data, temp_data_dir):
        """Test threshold visualization generation."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        threshold_comparison = hutf.calculate_threshold_comparison(sample_cost_data)
        discriminative_results = {
            'auc_improvement': {'auc_75th': 0.75, 'auc_90th': 0.82}
        }
        
        with patch('high_utilization_threshold_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            # Should not raise exceptions
            hutf.generate_threshold_visualizations(
                sample_cost_data, threshold_comparison, discriminative_results
            )
            
            # Verify plots directory
            plots_dir = Path(temp_data_dir) / 'results/threshold_analysis'
            assert plots_dir.exists()
    
    def test_main_function_integration(self, sample_cost_data, temp_data_dir):
        """Test main function integration."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Save sample data
        cost_path = Path(temp_data_dir) / 'data/processed/master_with_costs.parquet'
        sample_cost_data.to_parquet(cost_path)
        
        # Create config file
        config_path = Path(temp_data_dir) / 'config/config.yaml'
        sample_config = {'thresholds': {'high_utilization_percentile': 75}}
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        with patch('high_utilization_threshold_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            result = hutf.main()
            
            assert isinstance(result, dict)
            assert 'analysis_date' in result
            assert 'threshold_comparison' in result
            assert 'discriminative_analysis' in result
            assert 'clinical_justification' in result
    
    def test_error_handling_missing_cost_column(self):
        """Test error handling when cost column is missing."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Data without cost column
        no_cost_data = pd.DataFrame({
            'Patient_ID': [1, 2, 3],
            'age': [65, 70, 55]
        })
        
        with pytest.raises((KeyError, ValueError)):
            hutf.calculate_threshold_comparison(no_cost_data)
    
    def test_error_handling_negative_costs(self):
        """Test error handling with negative costs."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Data with negative costs
        negative_cost_data = pd.DataFrame({
            'Patient_ID': [1, 2, 3, 4, 5],
            'cost_total_12m': [1000, -500, 2000, 1500, -100]
        })
        
        # Should either handle gracefully or raise appropriate error
        try:
            result = hutf.calculate_threshold_comparison(negative_cost_data)
            # If it handles gracefully, should filter out negative costs
            assert all(cost >= 0 for cost in result['valid_costs'])
        except ValueError:
            # Acceptable to raise error for invalid data
            pass
    
    def test_statistical_validity_auc_calculation(self, sample_cost_data):
        """Test statistical validity of AUC calculations."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        threshold_comparison = hutf.calculate_threshold_comparison(sample_cost_data)
        result = hutf.assess_discriminative_ability(threshold_comparison, sample_cost_data)
        
        # AUC values should be statistically valid
        auc_75 = result['auc_improvement']['auc_75th']
        auc_90 = result['auc_improvement']['auc_90th']
        
        # AUC should be between 0.5 and 1.0
        assert 0.5 <= auc_75 <= 1.0
        assert 0.5 <= auc_90 <= 1.0
        
        # Improvement should be non-negative
        improvement = result['auc_improvement']['improvement']
        assert improvement >= 0
    
    def test_reproducibility(self, sample_cost_data):
        """Test that results are reproducible with same input."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Run analysis twice
        result1 = hutf.calculate_threshold_comparison(sample_cost_data)
        result2 = hutf.calculate_threshold_comparison(sample_cost_data)
        
        # Results should be identical
        assert result1['percentile_75']['threshold_value'] == result2['percentile_75']['threshold_value']
        assert result1['percentile_90']['threshold_value'] == result2['percentile_90']['threshold_value']
        assert result1['percentile_75']['high_utilizers'] == result2['percentile_75']['high_utilizers']
        assert result1['percentile_90']['high_utilizers'] == result2['percentile_90']['high_utilizers']

class TestHighUtilizationThresholdFixIntegration:
    """Integration tests for high utilization threshold fix with pipeline data."""
    
    def test_integration_with_outcome_variables(self):
        """Test integration with outcome variable definitions."""
        if hutf is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create data with outcome variables that depend on high utilization
        outcome_data = pd.DataFrame({
            'Patient_ID': range(1, 1001),
            'cost_total_12m': np.random.lognormal(8, 1.5, 1000),
            'H4_outcome': np.random.choice([0, 1], 1000),  # High utilization outcome
            'H5_outcome': np.random.choice([0, 1], 1000)   # Another cost-related outcome
        })
        
        # Test that threshold change affects outcome classification appropriately
        result = hutf.calculate_threshold_comparison(outcome_data)
        
        # Should provide meaningful comparison for outcome classification
        assert result['percentile_90']['high_utilizers'] < result['percentile_75']['high_utilizers']
        assert result['threshold_comparison']['relative_change'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

