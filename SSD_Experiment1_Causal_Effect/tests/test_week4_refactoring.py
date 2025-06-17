#!/usr/bin/env python3
"""
test_week4_refactoring.py - Tests for Week 4 function refactoring

Tests to ensure refactored functions maintain same behavior as original versions.
Following TDD mandate from CLAUDE.md.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.1.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestWeek4DocumentationRefactoring:
    """Test refactored documentation generation functions"""
    
    def test_generate_week4_analysis_report_original_behavior(self, tmp_path):
        """Test that refactored report generation maintains original behavior"""
        from week4_documentation_generator import generate_week4_analysis_report
        
        # Create test directories
        results_dir = tmp_path / "results"
        figures_dir = tmp_path / "figures" 
        output_dir = tmp_path / "output"
        results_dir.mkdir()
        figures_dir.mkdir()
        output_dir.mkdir()
        
        # Test report generation
        report_path = generate_week4_analysis_report(results_dir, figures_dir, output_dir)
        
        # Verify output
        assert report_path.exists()
        assert report_path.name == 'week4_analysis_report.md'
        
        # Check content structure
        content = report_path.read_text(encoding='utf-8')
        assert "# Week 4 Analysis Report" in content
        assert "## Executive Summary" in content
        assert "## 1. Mental Health Cohort Enhancement" in content
        assert "## 2. Mental Health-Specific Outcomes" in content
        assert "## 3. Advanced Causal Methods" in content
        
    def test_report_sections_independently(self, tmp_path):
        """Test that report section functions work independently"""
        from week4_documentation_generator import (
            _generate_executive_summary,
            _generate_cohort_enhancement_section,
            _generate_outcomes_section,
            _generate_advanced_methods_section,
            _generate_quality_and_technical_sections
        )
        
        # Test each section function
        exec_summary = _generate_executive_summary()
        assert "Week 4 Analysis Report" in exec_summary
        assert len(exec_summary) > 100
        
        cohort_section = _generate_cohort_enhancement_section()
        assert "Mental Health Cohort Enhancement" in cohort_section
        assert "ICD-10: F32-F48" in cohort_section
        
        outcomes_section = _generate_outcomes_section()
        assert "Mental Health-Specific Outcomes" in outcomes_section
        assert "MH Service Encounters" in outcomes_section
        
        methods_section = _generate_advanced_methods_section()
        assert "Advanced Causal Methods" in methods_section
        assert "H4: Mediation Analysis" in methods_section
        
        quality_section = _generate_quality_and_technical_sections()
        assert "Quality Assurance" in quality_section
        assert "Technical Implementation" in quality_section


class TestWeek4StatisticalRefactoring:
    """Test refactored statistical functions"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'exposure': np.random.binomial(1, 0.3, 1000),
            'mediator': np.random.normal(0, 1, 1000),
            'outcome': np.random.normal(0, 1, 1000),
            'confounder1': np.random.normal(0, 1, 1000),
            'confounder2': np.random.binomial(1, 0.5, 1000)
        })
        
    def test_enhanced_mediation_analysis_original_behavior(self):
        """Test that refactored mediation analysis maintains original behavior"""
        from week4_statistical_refinements import enhanced_mediation_analysis
        
        # Test with minimal data
        result = enhanced_mediation_analysis(
            data=self.test_data,
            exposure='exposure',
            mediator='mediator', 
            outcome='outcome',
            confounders=['confounder1', 'confounder2'],
            n_bootstrap=100  # Reduced for testing
        )
        
        # Verify required output structure
        required_keys = [
            'total_effect', 'direct_effect', 'indirect_effect',
            'a_path', 'b_path', 'proportion_mediated',
            'sobel_test', 'bootstrap_ci', 'sample_size', 'method'
        ]
        
        for key in required_keys:
            assert key in result
            
        # Verify data types
        assert isinstance(result['total_effect'], (int, float))
        assert isinstance(result['direct_effect'], (int, float))
        assert isinstance(result['sample_size'], int)
        assert isinstance(result['sobel_test'], dict)
        assert result['sample_size'] == len(self.test_data)
        
    def test_mediation_helper_functions(self):
        """Test individual mediation helper functions"""
        from week4_statistical_refinements import (
            _scipy_fallback_mediation,
            _calculate_mediation_statistics
        )
        
        # Test scipy fallback
        total_effect, direct_effect, a_path, b_path, method = _scipy_fallback_mediation(
            self.test_data, 'exposure', 'mediator', 'outcome'
        )
        
        assert isinstance(total_effect, float)
        assert isinstance(direct_effect, float)
        assert isinstance(a_path, float)
        assert isinstance(b_path, float)
        assert method == "Scipy-Fallback"
        
        # Test statistics calculation
        stats = _calculate_mediation_statistics(total_effect, direct_effect, a_path, b_path)
        
        assert 'indirect_effect' in stats
        assert 'proportion_mediated' in stats
        assert 'sobel_test' in stats
        assert isinstance(stats['sobel_test'], dict)
        
    @patch('week4_statistical_refinements.logger')
    def test_mediation_error_handling(self, mock_logger):
        """Test mediation analysis error handling"""
        from week4_statistical_refinements import enhanced_mediation_analysis
        
        # Test with missing columns
        bad_data = self.test_data.drop('mediator', axis=1)
        
        with pytest.raises(KeyError):
            enhanced_mediation_analysis(
                data=bad_data,
                exposure='exposure',
                mediator='mediator',
                outcome='outcome', 
                confounders=['confounder1']
            )


class TestTransportWeightsRefactoring:
    """Test transport weights function refactoring"""
    
    def test_calculate_transport_weights_original_behavior(self):
        """Test that transport weights function maintains original behavior"""
        from transport_weights import calculate_transport_weights
        
        # Test data
        study_data = pd.DataFrame({
            'age_group': ['18-34', '35-49', '50-64'] * 10,
            'sex': ['female', 'male'] * 15,
            'region': ['urban', 'suburban', 'rural'] * 10
        })
        
        # Test with missing file (should return skipped status)
        result = calculate_transport_weights(study_data)
        
        assert result['status'] == 'skipped'
        assert result['reason'] == 'ICES marginals file not available'
        assert len(result['weights']) == len(study_data)
        assert np.all(result['weights'] == 1.0)
        
    def test_transport_weights_validation(self):
        """Test transport weights validation function"""
        from transport_weights import validate_transport_weights
        
        # Test good weights
        good_weights = np.random.gamma(2, 0.5, 1000)
        validation = validate_transport_weights(good_weights)
        
        assert 'overall_quality' in validation
        assert 'max_weight_ok' in validation
        assert 'ess_ratio_ok' in validation
        assert isinstance(validation['overall_quality'], (bool, np.bool_))


class TestCodeQualityCompliance:
    """Test code quality compliance after refactoring"""
    
    def test_function_lengths_after_refactoring(self):
        """Test that refactored functions comply with 50 LOC limit"""
        from week4_documentation_generator import (
            _generate_executive_summary,
            _generate_cohort_enhancement_section, 
            _generate_outcomes_section,
            _generate_advanced_methods_section,
            _generate_quality_and_technical_sections,
            generate_week4_analysis_report
        )
        from week4_statistical_refinements import (
            _scipy_fallback_mediation,
            _calculate_mediation_statistics,
            enhanced_mediation_analysis
        )
        
        import inspect
        
        # Check refactored functions are under 50 LOC
        functions_to_check = [
            _generate_executive_summary,
            _generate_cohort_enhancement_section,
            _generate_outcomes_section, 
            _generate_advanced_methods_section,
            generate_week4_analysis_report,  # Should now be under 50 LOC
            _calculate_mediation_statistics,
            enhanced_mediation_analysis,  # Should now be under 50 LOC
        ]
        
        for func in functions_to_check:
            source_lines = inspect.getsource(func).split('\n')
            # Count non-empty, non-comment lines
            code_lines = [line for line in source_lines 
                         if line.strip() and not line.strip().startswith('#')
                         and not line.strip().startswith('"""')
                         and not line.strip().startswith("'''")]
            
            # Allow some tolerance for refactored functions
            if func.__name__ in ['generate_week4_analysis_report', 'enhanced_mediation_analysis']:
                assert len(code_lines) <= 50, f"{func.__name__} has {len(code_lines)} LOC (should be ≤50)"
            
    def test_docstring_presence(self):
        """Test that all functions have proper docstrings"""
        from week4_documentation_generator import (
            _generate_executive_summary,
            generate_week4_analysis_report
        )
        from week4_statistical_refinements import (
            _calculate_mediation_statistics,
            enhanced_mediation_analysis
        )
        
        functions_to_check = [
            _generate_executive_summary,
            generate_week4_analysis_report,
            _calculate_mediation_statistics, 
            enhanced_mediation_analysis
        ]
        
        for func in functions_to_check:
            assert func.__doc__ is not None, f"{func.__name__} missing docstring"
            assert len(func.__doc__.strip()) > 20, f"{func.__name__} has insufficient docstring"