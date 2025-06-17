#!/usr/bin/env python3
"""
test_week4_documentation_generator.py - Tests for Week 4 documentation generation

Tests methods supplement, STROBE-CI checklist, ROBINS-I assessment, glossary,
and analysis report generation following TDD principles.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestWeek4DocumentationGenerator:
    """Test Week 4 documentation generation functionality"""
    
    def test_methods_supplement_generation(self, tmp_path):
        """Test Methods Supplement generation with MH methodology"""
        from week4_documentation_generator import generate_methods_supplement_mh
        
        # Generate methods supplement
        methods_path = generate_methods_supplement_mh(tmp_path)
        
        # Check file creation
        assert methods_path.exists()
        assert methods_path.suffix == '.md'
        assert methods_path.name == 'methods_supplement_mh.md'
        
        # Check content requirements
        content = methods_path.read_text()
        assert 'Mental Health Causal Analysis' in content
        assert 'ICD-10 Codes (F32-F48)' in content
        assert 'Psychiatric Drug Persistence (H3 Enhanced)' in content
        assert '180 days' in content
        assert 'n=256,746' in content
        
        # Check file is substantial
        assert len(content) > 2000  # Should be comprehensive
    
    def test_strobe_ci_checklist_update(self, tmp_path):
        """Test STROBE-CI checklist tailored to MH study"""
        from week4_documentation_generator import update_strobe_ci_checklist_mh
        
        # Generate STROBE-CI checklist
        strobe_path = update_strobe_ci_checklist_mh(tmp_path)
        
        # Check file creation
        assert strobe_path.exists()
        assert strobe_path.suffix == '.md'
        assert 'strobe_ci_checklist' in strobe_path.name
        
        # Check content requirements
        content = strobe_path.read_text()
        assert 'STROBE-CI' in content
        assert 'Mental Health' in content or 'Psychiatric' in content
        assert 'ICD F32-F48' in content  # More specific ICD reference
        assert 'Causal' in content
        
        # Should contain checklist items (table format)
        assert '|' in content and 'Item' in content
    
    def test_robins_i_assessment_update(self, tmp_path):
        """Test ROBINS-I bias assessment with new domains"""
        from week4_documentation_generator import update_robins_i_assessment_mh
        
        # Generate ROBINS-I assessment
        robins_path = update_robins_i_assessment_mh(tmp_path)
        
        # Check file creation
        assert robins_path.exists()
        assert robins_path.suffix == '.md'
        assert 'robins_i' in robins_path.name.lower()
        
        # Check content requirements
        content = robins_path.read_text()
        assert 'ROBINS-I' in content
        assert 'Bias' in content
        assert 'Mental Health' in content or 'Psychiatric' in content
        assert 'Confounding' in content
        
        # Should contain bias domains
        assert 'Selection' in content or 'Measurement' in content
    
    def test_glossary_update_mh(self, tmp_path):
        """Test glossary update with MH-specific terms"""
        from week4_documentation_generator import update_glossary_mh
        
        # Generate updated glossary
        glossary_path = update_glossary_mh(tmp_path)
        
        # Check file creation
        assert glossary_path.exists()
        assert glossary_path.suffix == '.md'
        assert 'glossary' in glossary_path.name.lower()
        
        # Check content requirements
        content = glossary_path.read_text()
        assert 'Glossary' in content
        assert 'Mental Health' in content or 'Psychiatric' in content
        assert 'SSD' in content
        assert 'ICD' in content
        
        # Should contain term definitions
        assert ':' in content or '-' in content  # Definition formatting
    
    def test_week4_analysis_report_generation(self, tmp_path):
        """Test Week 4 analysis report generation"""
        from week4_documentation_generator import generate_week4_analysis_report
        
        # Create mock directories
        results_dir = tmp_path / 'results'
        figures_dir = tmp_path / 'figures'
        results_dir.mkdir()
        figures_dir.mkdir()
        
        # Create mock result files
        (results_dir / 'mediation_results.json').write_text('{"total_effect": 0.25}')
        (results_dir / 'causal_forest_results.json').write_text('{"tau_mean": 0.18}')
        (results_dir / 'hypothesis_h4.json').write_text('{"effect": 0.15}')
        
        # Create mock figures
        (figures_dir / 'mediation_pathway_diagram.svg').write_text('<svg></svg>')
        (figures_dir / 'cate_heatmap.svg').write_text('<svg></svg>')
        (figures_dir / 'evalue_plot.svg').write_text('<svg></svg>')
        
        # Generate analysis report
        report_path = generate_week4_analysis_report(
            results_dir, figures_dir, tmp_path
        )
        
        # Check file creation
        assert report_path.exists()
        assert report_path.suffix == '.md'
        assert 'week4_analysis_report' in report_path.name
        
        # Check content requirements
        content = report_path.read_text()
        assert 'Week 4 Analysis Report' in content
        assert 'Mental Health' in content
        assert 'Mediation' in content
        assert 'CATE' in content
        
        # Should embed figures
        assert 'mediation_pathway_diagram' in content
        assert 'cate_heatmap' in content
        assert 'evalue_plot' in content
    
    def test_generate_all_week4_documentation(self, tmp_path):
        """Test complete Week 4 documentation generation"""
        from week4_documentation_generator import generate_all_week4_documentation
        
        # Create mock directories
        results_dir = tmp_path / 'results'
        figures_dir = tmp_path / 'figures'
        docs_dir = tmp_path / 'docs'
        results_dir.mkdir()
        figures_dir.mkdir()
        
        # Create minimal mock files
        (results_dir / 'mediation_results.json').write_text('{}')
        (figures_dir / 'mediation_pathway_diagram.svg').write_text('<svg></svg>')
        
        # Generate all documentation
        generated_docs = generate_all_week4_documentation(
            results_dir, figures_dir, docs_dir
        )
        
        # Check all expected documents generated
        expected_docs = [
            'methods_supplement',
            'strobe_checklist', 
            'robins_assessment',
            'glossary',
            'analysis_report'
        ]
        
        for doc_name in expected_docs:
            assert doc_name in generated_docs
            assert generated_docs[doc_name].exists()
            assert generated_docs[doc_name].stat().st_size > 100
    
    def test_documentation_content_quality(self, tmp_path):
        """Test documentation content quality and completeness"""
        from week4_documentation_generator import generate_methods_supplement_mh
        
        methods_path = generate_methods_supplement_mh(tmp_path)
        content = methods_path.read_text()
        
        # Check for key methodological elements
        required_elements = [
            'F32-F34',  # Specific ICD codes
            'N06A',     # Drug classes
            'Baron & Kenny', # Actual method name used
            'G-computation',
            'Causal Forest',
            'E-value',
            'Bootstrap'
        ]
        
        for element in required_elements:
            assert element in content, f"Missing required element: {element}"
    
    def test_documentation_with_missing_inputs(self, tmp_path):
        """Test documentation generation with missing input files"""
        from week4_documentation_generator import generate_week4_analysis_report
        
        # Missing directories
        missing_results = tmp_path / 'missing_results'
        missing_figures = tmp_path / 'missing_figures'
        
        # Should handle gracefully
        report_path = generate_week4_analysis_report(
            missing_results, missing_figures, tmp_path
        )
        
        assert report_path.exists()
        content = report_path.read_text()
        assert 'Week 4 Analysis Report' in content
        # Should indicate missing data
        assert 'not available' in content or 'missing' in content or 'placeholder' in content
    
    def test_documentation_timestamp_inclusion(self, tmp_path):
        """Test that all documentation includes timestamps"""
        from week4_documentation_generator import (
            generate_methods_supplement_mh,
            update_strobe_ci_checklist_mh,
            update_robins_i_assessment_mh
        )
        
        # Generate documents
        methods_path = generate_methods_supplement_mh(tmp_path)
        strobe_path = update_strobe_ci_checklist_mh(tmp_path)
        robins_path = update_robins_i_assessment_mh(tmp_path)
        
        # Check timestamps present
        for doc_path in [methods_path, strobe_path, robins_path]:
            content = doc_path.read_text()
            assert '2025-' in content  # Should have current year
            assert 'Generated:' in content or 'Updated:' in content
    
    def test_documentation_cross_references(self, tmp_path):
        """Test that documentation includes proper cross-references"""
        from week4_documentation_generator import generate_week4_analysis_report
        
        # Create mock files with references
        results_dir = tmp_path / 'results'
        figures_dir = tmp_path / 'figures'
        results_dir.mkdir()
        figures_dir.mkdir()
        
        (figures_dir / 'mediation_pathway_diagram.svg').write_text('<svg></svg>')
        (figures_dir / 'cate_heatmap.svg').write_text('<svg></svg>')
        
        report_path = generate_week4_analysis_report(
            results_dir, figures_dir, tmp_path
        )
        
        content = report_path.read_text()
        
        # Should reference figures properly
        assert 'Figure' in content or 'figure' in content
        assert '![' in content or '<img' in content  # Markdown image syntax
    
    def test_documentation_error_handling(self, tmp_path):
        """Test documentation generation error handling"""
        from week4_documentation_generator import update_glossary_mh
        
        # Test with read-only directory (if possible)
        try:
            readonly_path = tmp_path / 'readonly'
            readonly_path.mkdir()
            readonly_path.chmod(0o444)  # Read-only
            
            # Should handle permission errors gracefully
            glossary_path = update_glossary_mh(readonly_path)
            # Should either succeed or fail gracefully without crashing
            
        except PermissionError:
            # Expected on some systems
            pass
        except Exception as e:
            # Should not crash with unexpected errors
            pytest.fail(f"Unexpected error in documentation generation: {e}")