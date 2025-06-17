#!/usr/bin/env python3
"""
test_reconcile_estimates.py - Tests for estimate reconciliation rule

Tests for comparing TMLE, DML, and Causal-Forest ATEs with 15% threshold flagging.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch, MagicMock

# Import the module we're testing
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from reconcile_estimates import (
    compare_ate_estimates,
    load_estimate_results,
    calculate_percentage_difference,
    flag_discordant_estimates,
    reconcile_causal_estimates,
    create_reconciliation_report
)


class TestATEComparison:
    """Test ATE comparison functionality"""
    
    def test_calculate_percentage_difference_basic(self):
        """Test basic percentage difference calculation"""
        # Test case: 20% difference
        result = calculate_percentage_difference(1.0, 1.2)
        assert abs(result - 20.0) < 0.01
        
        # Test case: identical estimates
        result = calculate_percentage_difference(1.5, 1.5)
        assert result == 0.0
        
        # Test case: negative difference
        result = calculate_percentage_difference(1.2, 1.0)
        assert abs(result - 16.67) < 0.01
    
    def test_calculate_percentage_difference_edge_cases(self):
        """Test edge cases for percentage difference"""
        # Test zero baseline
        with pytest.raises(ValueError, match="Cannot calculate percentage difference with zero baseline"):
            calculate_percentage_difference(0.0, 1.0)
        
        # Test negative values
        result = calculate_percentage_difference(-1.0, -1.2)
        assert abs(result - 20.0) < 0.01
    
    def test_compare_ate_estimates_concordant(self):
        """Test comparison with concordant estimates (within 15%)"""
        estimates = {
            'tmle': {'ate': 1.0, 'ci_lower': 0.8, 'ci_upper': 1.2},
            'dml': {'ate': 1.1, 'ci_lower': 0.9, 'ci_upper': 1.3},
            'causal_forest': {'ate': 0.95, 'ci_lower': 0.75, 'ci_upper': 1.15}
        }
        
        result = compare_ate_estimates(estimates)
        
        assert result['status'] == 'concordant'
        assert result['max_difference'] < 15.0
        assert len(result['comparisons']) == 3  # 3 pairwise comparisons
        assert not result['has_discordant_pairs']
    
    def test_compare_ate_estimates_discordant(self):
        """Test comparison with discordant estimates (>15% difference)"""
        estimates = {
            'tmle': {'ate': 1.0, 'ci_lower': 0.8, 'ci_upper': 1.2},
            'dml': {'ate': 1.5, 'ci_lower': 1.3, 'ci_upper': 1.7},  # 50% difference
            'causal_forest': {'ate': 0.95, 'ci_lower': 0.75, 'ci_upper': 1.15}
        }
        
        result = compare_ate_estimates(estimates)
        
        assert result['status'] == 'discordant'
        assert result['max_difference'] > 15.0
        assert result['has_discordant_pairs']
        assert any(comp['percentage_diff'] > 15.0 for comp in result['comparisons'])
    
    def test_flag_discordant_estimates(self):
        """Test flagging functionality"""
        # Create mock comparison results
        comparison_result = {
            'status': 'discordant',
            'max_difference': 25.0,
            'comparisons': [
                {'method1': 'tmle', 'method2': 'dml', 'percentage_diff': 25.0},
                {'method1': 'tmle', 'method2': 'causal_forest', 'percentage_diff': 5.0},
                {'method1': 'dml', 'method2': 'causal_forest', 'percentage_diff': 20.0}
            ]
        }
        
        result = flag_discordant_estimates(comparison_result, threshold=15.0)
        
        assert result['flagged']
        assert result['threshold'] == 15.0
        assert len(result['discordant_pairs']) == 2  # tmle-dml and dml-causal_forest


class TestEstimateLoading:
    """Test loading estimates from YAML files"""
    
    def test_load_estimate_results_success(self):
        """Test successful loading of estimate results"""
        # Create temporary YAML files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create mock YAML files
            tmle_data = {'ate': 1.0, 'ci_lower': 0.8, 'ci_upper': 1.2, 'method': 'TMLE'}
            dml_data = {'ate': 1.1, 'ci_lower': 0.9, 'ci_upper': 1.3, 'method': 'DML'}
            cf_data = {'ate': 0.95, 'ci_lower': 0.75, 'ci_upper': 1.15, 'method': 'CausalForest'}
            
            with open(tmpdir / 'tmle_results.yaml', 'w') as f:
                yaml.dump(tmle_data, f)
            with open(tmpdir / 'dml_results.yaml', 'w') as f:
                yaml.dump(dml_data, f)
            with open(tmpdir / 'causal_forest_results.yaml', 'w') as f:
                yaml.dump(cf_data, f)
            
            result = load_estimate_results(tmpdir)
            
            assert 'tmle' in result
            assert 'dml' in result
            assert 'causal_forest' in result
            assert result['tmle']['ate'] == 1.0
            assert result['dml']['ate'] == 1.1
            assert result['causal_forest']['ate'] == 0.95
    
    def test_load_estimate_results_missing_files(self):
        """Test handling of missing estimate files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Only create TMLE file
            tmle_data = {'ate': 1.0, 'ci_lower': 0.8, 'ci_upper': 1.2}
            with open(tmpdir / 'tmle_results.yaml', 'w') as f:
                yaml.dump(tmle_data, f)
            
            result = load_estimate_results(tmpdir)
            
            assert 'tmle' in result
            assert 'dml' not in result
            assert 'causal_forest' not in result


class TestReconciliationWorkflow:
    """Test full reconciliation workflow"""
    
    def test_reconcile_causal_estimates_concordant(self):
        """Test full reconciliation with concordant estimates"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create mock concordant estimates
            estimates = {
                'tmle': {'ate': 1.0, 'ci_lower': 0.8, 'ci_upper': 1.2},
                'dml': {'ate': 1.1, 'ci_lower': 0.9, 'ci_upper': 1.3},
                'causal_forest': {'ate': 0.95, 'ci_lower': 0.75, 'ci_upper': 1.15}
            }
            
            # Save as YAML files
            for method, data in estimates.items():
                with open(tmpdir / f"{method}_results.yaml", 'w') as f:
                    yaml.dump(data, f)
            
            result = reconcile_causal_estimates(tmpdir, output_dir=tmpdir)
            
            assert result['reconciliation_status'] == 'PASS'
            assert not result['flagged']
            assert result['max_difference'] < 15.0
            
            # Check report was created
            assert (tmpdir / 'estimate_reconciliation_report.md').exists()
    
    def test_reconcile_causal_estimates_discordant(self):
        """Test full reconciliation with discordant estimates"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create mock discordant estimates
            estimates = {
                'tmle': {'ate': 1.0, 'ci_lower': 0.8, 'ci_upper': 1.2},
                'dml': {'ate': 1.5, 'ci_lower': 1.3, 'ci_upper': 1.7},  # 50% difference
                'causal_forest': {'ate': 0.95, 'ci_lower': 0.75, 'ci_upper': 1.15}
            }
            
            # Save as YAML files
            for method, data in estimates.items():
                with open(tmpdir / f"{method}_results.yaml", 'w') as f:
                    yaml.dump(data, f)
            
            # Should raise assertion error for discordant estimates
            with pytest.raises(AssertionError, match="Discordant estimates detected"):
                reconcile_causal_estimates(tmpdir, output_dir=tmpdir)


class TestReportGeneration:
    """Test reconciliation report generation"""
    
    def test_create_reconciliation_report(self):
        """Test reconciliation report creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Mock reconciliation results
            reconciliation_results = {
                'reconciliation_status': 'PASS',
                'flagged': False,
                'max_difference': 10.5,
                'estimates': {
                    'tmle': {'ate': 1.0, 'ci_lower': 0.8, 'ci_upper': 1.2},
                    'dml': {'ate': 1.1, 'ci_lower': 0.9, 'ci_upper': 1.3},
                    'causal_forest': {'ate': 0.95, 'ci_lower': 0.75, 'ci_upper': 1.15}
                },
                'comparison_results': {
                    'comparisons': [
                        {'method1': 'tmle', 'method2': 'dml', 'percentage_diff': 10.0, 'exceeds_threshold': False},
                        {'method1': 'tmle', 'method2': 'causal_forest', 'percentage_diff': 5.3, 'exceeds_threshold': False},
                        {'method1': 'dml', 'method2': 'causal_forest', 'percentage_diff': 15.8, 'exceeds_threshold': True}
                    ]
                }
            }
            
            report_path = create_reconciliation_report(reconciliation_results, tmpdir)
            
            assert report_path.exists()
            assert report_path.name == 'estimate_reconciliation_report.md'
            
            # Check report content
            content = report_path.read_text()
            assert 'Estimate Reconciliation Report' in content
            assert 'PASS' in content
            assert 'TMLE' in content
            assert 'DML' in content
            assert 'Causal Forest' in content


if __name__ == "__main__":
    pytest.main([__file__])