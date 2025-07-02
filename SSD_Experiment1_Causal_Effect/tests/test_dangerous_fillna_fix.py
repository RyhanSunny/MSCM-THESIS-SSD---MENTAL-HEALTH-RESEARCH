# -*- coding: utf-8 -*-
"""
test_dangerous_fillna_fix.py - Tests for Dangerous fillna Pattern Correction

TDD COMPLIANCE FOR CLAUDE.md REQUIREMENTS:
==========================================

Following CLAUDE.md mandate: "TDD is MANDATORY - ALWAYS write tests FIRST"
This test suite validates the systematic correction of dangerous fillna assumptions
throughout the codebase to prevent bias in causal estimates.

Test Coverage:
1. Codebase scanning for dangerous patterns
2. Severity assessment of fillna patterns
3. Evidence-based alternative generation
4. Implementation plan creation
5. Automated fix script generation
6. Clinical justification validation
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
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys

# Add src to path for testing
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Import the module under test
try:
    import dangerous_fillna_fix as dff
except ImportError:
    dff = None

class TestDangerousFillnaFix:
    """Test suite for dangerous fillna fix functionality."""
    
    @pytest.fixture
    def sample_python_files(self, temp_data_dir):
        """Create sample Python files with various fillna patterns."""
        src_dir = Path(temp_data_dir) / 'src'
        src_dir.mkdir(exist_ok=True)
        
        # File with dangerous fillna patterns
        dangerous_file = src_dir / 'dangerous_script.py'
        dangerous_content = '''
import pandas as pd

# Dangerous patterns
df['visit_count'] = df['visit_count'].fillna(0)  # Assumes no visits
df['diagnosis_flag'] = df['diagnosis_flag'].fillna(0.5)  # Nonsensical for binary
df['cost_total'] = df['cost_total'].fillna(0)  # Assumes no costs
df['charlson_score'] = df['charlson_score'].fillna(-1)  # Arbitrary negative

# Some safe patterns
df['optional_field'] = df['optional_field'].fillna("unknown")
'''
        
        with open(dangerous_file, 'w') as f:
            f.write(dangerous_content)
        
        # File with safe patterns
        safe_file = src_dir / 'safe_script.py'
        safe_content = '''
import pandas as pd

# Safe patterns
df['category'] = df['category'].fillna("other")
df = df.dropna()  # Complete case analysis
'''
        
        with open(safe_file, 'w') as f:
            f.write(safe_content)
        
        return [dangerous_file, safe_file]
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        
        data_dirs = [
            'src',
            'results/fillna_fix_scripts'
        ]
        
        for dir_path in data_dirs:
            (Path(temp_dir) / dir_path).mkdir(parents=True, exist_ok=True)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_scan_codebase_for_fillna_basic(self, sample_python_files, temp_data_dir):
        """Test basic codebase scanning for fillna patterns."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            result = dff.scan_codebase_for_fillna()
            
            # Validate structure
            assert isinstance(result, dict)
            
            # Should find dangerous patterns
            assert 'fillna(0)' in result
            assert 'fillna(0.5)' in result
            assert 'fillna(-1)' in result
            
            # Validate pattern detection
            fillna_0_issues = result['fillna(0)']
            assert len(fillna_0_issues) >= 2  # visit_count and cost_total
            
            fillna_05_issues = result['fillna(0.5)']
            assert len(fillna_05_issues) >= 1  # diagnosis_flag
    
    def test_scan_codebase_no_python_files(self, temp_data_dir):
        """Test scanning when no Python files exist."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Empty src directory
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            result = dff.scan_codebase_for_fillna()
            
            # Should return empty results
            assert isinstance(result, dict)
            for pattern_issues in result.values():
                assert len(pattern_issues) == 0
    
    def test_assess_fillna_severity_critical_cases(self):
        """Test severity assessment for critical fillna patterns."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Critical cases
        critical_cases = [
            "df['visit_count'].fillna(0)",
            "patient_data['cost_total'].fillna(0)",
            "cohort['utilization_count'].fillna(0)",
            "binary_flag.fillna(0.5)"
        ]
        
        for case in critical_cases:
            severity = dff.assess_fillna_severity(case, 'fillna(0)' if '(0)' in case else 'fillna(0.5)')
            assert severity == 'CRITICAL'
    
    def test_assess_fillna_severity_moderate_cases(self):
        """Test severity assessment for moderate fillna patterns."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Moderate cases
        moderate_cases = [
            "df['confounder_var'].fillna(0)",
            "adjustment_data['covariate'].fillna(1)"
        ]
        
        for case in moderate_cases:
            severity = dff.assess_fillna_severity(case, 'fillna(0)')
            assert severity in ['MODERATE', 'HIGH']  # Could be either depending on context
    
    def test_assess_fillna_severity_low_cases(self):
        """Test severity assessment for low-impact fillna patterns."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Low impact cases
        low_cases = [
            "df['optional_field'].fillna('')",
            "metadata['description'].fillna('unknown')"
        ]
        
        for case in low_cases:
            severity = dff.assess_fillna_severity(case, 'fillna("")')
            assert severity == 'LOW'
    
    def test_determine_appropriate_alternative_count_variables(self):
        """Test alternative determination for count variables."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        count_line = "df['visit_count'].fillna(0)"
        alternative = dff.determine_appropriate_alternative(count_line, 'fillna(0)', 'CRITICAL')
        
        assert isinstance(alternative, dict)
        assert alternative['method'] == 'Conditional Mean Imputation'
        assert 'group-specific means' in alternative['implementation']
        assert 'code_example' in alternative
        assert 'rationale' in alternative
    
    def test_determine_appropriate_alternative_binary_variables(self):
        """Test alternative determination for binary variables."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        binary_line = "df['diagnosis_flag'].fillna(0.5)"
        alternative = dff.determine_appropriate_alternative(binary_line, 'fillna(0.5)', 'CRITICAL')
        
        assert isinstance(alternative, dict)
        assert alternative['method'] == 'Logistic Regression Imputation'
        assert 'logistic regression' in alternative['implementation']
        assert 'LogisticRegression' in alternative['code_example']
    
    def test_determine_appropriate_alternative_cost_variables(self):
        """Test alternative determination for cost variables."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        cost_line = "df['total_cost'].fillna(0)"
        alternative = dff.determine_appropriate_alternative(cost_line, 'fillna(0)', 'CRITICAL')
        
        assert isinstance(alternative, dict)
        assert alternative['method'] == 'Multiple Imputation with Predictive Mean Matching'
        assert 'MICE' in alternative['implementation']
        assert 'miceforest' in alternative['code_example']
    
    def test_generate_evidence_based_alternatives_integration(self, sample_python_files, temp_data_dir):
        """Test evidence-based alternatives generation."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            dangerous_patterns = dff.scan_codebase_for_fillna()
            alternatives = dff.generate_evidence_based_alternatives(dangerous_patterns)
            
            assert isinstance(alternatives, dict)
            
            # Should have alternatives for each pattern found
            for pattern, issues in dangerous_patterns.items():
                if issues:  # If pattern was found
                    assert pattern in alternatives
                    pattern_alternatives = alternatives[pattern]
                    
                    for alt in pattern_alternatives:
                        assert 'recommended_alternative' in alt
                        assert 'implementation_priority' in alt
                        assert 'clinical_justification' in alt
    
    def test_get_implementation_priority_mapping(self):
        """Test implementation priority mapping."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        priority_map = {
            'CRITICAL': 'IMMEDIATE',
            'HIGH': 'HIGH',
            'MODERATE': 'MODERATE',
            'LOW': 'LOW'
        }
        
        for severity, expected_priority in priority_map.items():
            priority = dff.get_implementation_priority(severity)
            assert expected_priority in priority
    
    def test_get_clinical_justification_content(self):
        """Test clinical justification content generation."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Test different variable types
        test_cases = [
            ("df['visit_count'].fillna(0)", {'method': 'Conditional Mean Imputation'}),
            ("df['diagnosis_flag'].fillna(0.5)", {'method': 'Logistic Regression Imputation'}),
            ("df['total_cost'].fillna(0)", {'method': 'Predictive Mean Matching'})
        ]
        
        for line_content, alternative in test_cases:
            justification = dff.get_clinical_justification(line_content, alternative)
            
            assert isinstance(justification, str)
            assert len(justification) > 50  # Substantial content
            assert any(word in justification.lower() for word in ['clinical', 'patient', 'bias', 'imputation'])
    
    def test_generate_implementation_plan_prioritization(self):
        """Test implementation plan generation and prioritization."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Sample alternatives with different priorities
        sample_alternatives = {
            'fillna(0)': [
                {
                    'original_issue': {'severity': 'CRITICAL'},
                    'implementation_priority': 'IMMEDIATE (affects primary variables)',
                    'recommended_alternative': {'method': 'Conditional Mean Imputation'}
                },
                {
                    'original_issue': {'severity': 'MODERATE'},
                    'implementation_priority': 'MODERATE (affects adjustment)',
                    'recommended_alternative': {'method': 'MICE'}
                }
            ],
            'fillna(0.5)': [
                {
                    'original_issue': {'severity': 'CRITICAL'},
                    'implementation_priority': 'IMMEDIATE (affects primary variables)',
                    'recommended_alternative': {'method': 'Logistic Regression Imputation'}
                }
            ]
        }
        
        plan = dff.generate_implementation_plan(sample_alternatives)
        
        assert isinstance(plan, dict)
        assert 'phase_1_immediate' in plan
        assert 'phase_2_high_priority' in plan
        assert 'phase_3_moderate_priority' in plan
        assert 'phase_4_low_priority' in plan
        
        # Immediate phase should have critical fixes
        immediate_fixes = plan['phase_1_immediate']['fixes']
        assert len(immediate_fixes) == 2  # Two critical issues
    
    def test_create_fix_scripts_generation(self, temp_data_dir):
        """Test automated fix script generation."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Sample implementation plan
        sample_plan = {
            'phase_1_immediate': {
                'description': 'Fix critical issues',
                'timeline': 'Today',
                'impact': 'Prevents bias',
                'fixes': [
                    {
                        'original_issue': {
                            'file': 'test.py',
                            'line_number': 10,
                            'line_content': "df['count'].fillna(0)"
                        },
                        'recommended_alternative': {
                            'method': 'Conditional Mean Imputation',
                            'code_example': 'Use group means'
                        },
                        'clinical_justification': 'Prevents bias'
                    }
                ]
            }
        }
        
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            dff.create_fix_scripts(sample_plan)
            
            # Should create script files
            scripts_dir = Path(temp_data_dir) / 'results/fillna_fix_scripts'
            assert scripts_dir.exists()
    
    def test_generate_clinical_recommendations_high_risk(self):
        """Test clinical recommendations for high-risk scenarios."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # High-risk implementation plan
        high_risk_plan = {
            'phase_1_immediate': {'fixes': [1, 2, 3]},  # 3 immediate fixes
            'phase_2_high_priority': {'fixes': [1, 2]},  # 2 high priority fixes
            'phase_3_moderate_priority': {'fixes': [1]},
            'phase_4_low_priority': {'fixes': []}
        }
        
        recommendations = dff.generate_clinical_recommendations(high_risk_plan)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('URGENT' in rec for rec in recommendations)
        assert any('3 critical' in rec for rec in recommendations)
    
    def test_generate_clinical_recommendations_low_risk(self):
        """Test clinical recommendations for low-risk scenarios."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Low-risk implementation plan
        low_risk_plan = {
            'phase_1_immediate': {'fixes': []},
            'phase_2_high_priority': {'fixes': []},
            'phase_3_moderate_priority': {'fixes': [1]},
            'phase_4_low_priority': {'fixes': [1, 2]}
        }
        
        recommendations = dff.generate_clinical_recommendations(low_risk_plan)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert not any('URGENT' in rec for rec in recommendations)
        assert any('METHODOLOGICAL IMPROVEMENT' in rec for rec in recommendations)
    
    def test_main_function_integration(self, sample_python_files, temp_data_dir):
        """Test main function integration."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            result = dff.main()
            
            assert isinstance(result, dict)
            assert 'analysis_date' in result
            assert 'scan_summary' in result
            assert 'dangerous_patterns' in result
            assert 'evidence_based_alternatives' in result
            assert 'implementation_plan' in result
            assert 'clinical_recommendations' in result
            assert 'thesis_defensibility' in result
    
    def test_error_handling_no_src_directory(self, temp_data_dir):
        """Test error handling when src directory doesn't exist."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Remove src directory
        src_dir = Path(temp_data_dir) / 'src'
        if src_dir.exists():
            shutil.rmtree(src_dir)
        
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            result = dff.scan_codebase_for_fillna()
            
            # Should handle gracefully
            assert isinstance(result, dict)
    
    def test_error_handling_unreadable_files(self, temp_data_dir):
        """Test error handling with unreadable files."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create src directory with unreadable file
        src_dir = Path(temp_data_dir) / 'src'
        src_dir.mkdir(exist_ok=True)
        
        unreadable_file = src_dir / 'unreadable.py'
        unreadable_file.touch()
        
        # Mock file reading to raise exception
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            with patch('builtins.open', side_effect=PermissionError("Access denied")):
                result = dff.scan_codebase_for_fillna()
                
                # Should handle gracefully and continue with other files
                assert isinstance(result, dict)
    
    def test_edge_case_empty_files(self, temp_data_dir):
        """Test handling of empty Python files."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create empty Python file
        src_dir = Path(temp_data_dir) / 'src'
        src_dir.mkdir(exist_ok=True)
        
        empty_file = src_dir / 'empty.py'
        empty_file.touch()
        
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            result = dff.scan_codebase_for_fillna()
            
            # Should handle empty files gracefully
            assert isinstance(result, dict)
    
    def test_edge_case_commented_fillna(self, temp_data_dir):
        """Test handling of commented fillna patterns."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Create file with commented fillna
        src_dir = Path(temp_data_dir) / 'src'
        src_dir.mkdir(exist_ok=True)
        
        commented_file = src_dir / 'commented.py'
        commented_content = '''
# This is commented out: df['count'].fillna(0)
# Another comment: df['flag'].fillna(0.5)
df['actual'].fillna(0)  # This should be detected
'''
        
        with open(commented_file, 'w') as f:
            f.write(commented_content)
        
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            result = dff.scan_codebase_for_fillna()
            
            # Should only detect non-commented patterns
            fillna_0_issues = result.get('fillna(0)', [])
            assert len(fillna_0_issues) == 1  # Only the non-commented one
    
    def test_statistical_validity_checks(self):
        """Test statistical validity of recommendations."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Test that recommendations are statistically sound
        test_line = "df['visit_count'].fillna(0)"
        alternative = dff.determine_appropriate_alternative(test_line, 'fillna(0)', 'CRITICAL')
        
        # Should recommend valid statistical methods
        valid_methods = [
            'Conditional Mean Imputation',
            'Logistic Regression Imputation',
            'Multiple Imputation',
            'Predictive Mean Matching'
        ]
        
        assert alternative['method'] in valid_methods
    
    def test_reproducibility(self, sample_python_files, temp_data_dir):
        """Test that results are reproducible with same input."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        with patch('dangerous_fillna_fix.Path') as mock_path:
            mock_path.return_value = Path(temp_data_dir)
            
            # Run analysis twice
            result1 = dff.scan_codebase_for_fillna()
            result2 = dff.scan_codebase_for_fillna()
            
            # Results should be identical
            assert result1.keys() == result2.keys()
            
            for pattern in result1:
                assert len(result1[pattern]) == len(result2[pattern])

class TestDangerousFillnaFixIntegration:
    """Integration tests for dangerous fillna fix with real pipeline code."""
    
    def test_integration_with_actual_pipeline_patterns(self):
        """Test integration with actual pipeline fillna patterns."""
        if dff is None:
            pytest.skip("Module not implemented yet (TDD)")
        
        # Simulate actual pipeline patterns found in codebase
        pipeline_patterns = {
            'fillna(0)': [
                {
                    'file': 'src/01_cohort_builder.py',
                    'line_content': 'charlson_scores = charlson_scores.fillna(0)',
                    'severity': 'MODERATE'
                },
                {
                    'file': 'src/02_exposure_flag.py',
                    'line_content': 'lab_counts.fillna(0)',
                    'severity': 'CRITICAL'
                }
            ],
            'fillna(0.5)': [
                {
                    'file': 'src/12_temporal_adjust.py',
                    'line_content': 'X = patient_data[X_vars].fillna(0.5)',
                    'severity': 'CRITICAL'
                }
            ]
        }
        
        alternatives = dff.generate_evidence_based_alternatives(pipeline_patterns)
        
        # Should provide appropriate alternatives for each pattern
        assert 'fillna(0)' in alternatives
        assert 'fillna(0.5)' in alternatives
        
        # Critical patterns should get immediate priority
        for pattern_alts in alternatives.values():
            for alt in pattern_alts:
                if alt['original_issue']['severity'] == 'CRITICAL':
                    assert 'IMMEDIATE' in alt['implementation_priority']

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

