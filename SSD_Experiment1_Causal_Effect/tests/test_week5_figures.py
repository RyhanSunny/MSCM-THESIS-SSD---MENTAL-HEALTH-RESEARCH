#!/usr/bin/env python3
"""
test_week5_figures.py - Tests for Week 5 remaining figures

Tests for Week 5 Task F: Selection & cost-effectiveness figures
Covers selection diagram and cost-effectiveness plane generation.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestSelectionDiagram:
    """Test selection diagram generation"""
    
    def test_create_selection_diagram(self):
        """Test selection diagram creation with CONSORT flow"""
        from week5_figures import create_selection_diagram
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'selection_diagram.svg'
            
            # Test with mock cohort numbers
            cohort_stats = {
                'total_patients': 350000,
                'after_age_filter': 280000,
                'after_observation_filter': 250025,
                'mental_health_subset': 45000,
                'exposed_group': 12500,
                'control_group': 37500,
                'matched_pairs': 8500
            }
            
            result_path = create_selection_diagram(
                cohort_stats=cohort_stats,
                output_path=output_path
            )
            
            assert result_path.exists(), "Selection diagram SVG not created"
            assert result_path.suffix == '.svg', "Output should be SVG format"
            
            # Check file content has expected elements
            content = result_path.read_text()
            assert 'CONSORT' in content or 'Selection' in content
            assert '350000' in content  # Total patients
            assert '250025' in content  # Final cohort
    
    def test_selection_diagram_with_missing_stats(self):
        """Test selection diagram handles missing statistics gracefully"""
        from week5_figures import create_selection_diagram
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'selection_diagram.svg'
            
            # Test with minimal stats
            cohort_stats = {
                'total_patients': 250000,
                'final_cohort': 200000
            }
            
            result_path = create_selection_diagram(
                cohort_stats=cohort_stats,
                output_path=output_path
            )
            
            assert result_path.exists(), "Should create diagram even with minimal stats"
    
    def test_selection_diagram_png_output(self):
        """Test selection diagram can generate PNG format"""
        from week5_figures import create_selection_diagram
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'selection_diagram.png'
            
            cohort_stats = {'total_patients': 250000, 'final_cohort': 200000}
            
            result_path = create_selection_diagram(
                cohort_stats=cohort_stats,
                output_path=output_path,
                format='png'
            )
            
            assert result_path.exists()
            assert result_path.suffix == '.png'


class TestCostEffectivenessPlane:
    """Test cost-effectiveness plane generation"""
    
    def test_create_cost_effectiveness_plane(self):
        """Test cost-effectiveness plane creation for H6"""
        from week5_figures import create_cost_effectiveness_plane
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'cost_plane.svg'
            
            # Mock intervention scenarios
            scenarios = [
                {'name': 'Baseline', 'cost': 0, 'effectiveness': 0, 'quadrant': 'origin'},
                {'name': 'Enhanced Screening', 'cost': 150, 'effectiveness': 0.12, 'quadrant': 'NE'},
                {'name': 'Reduced Testing', 'cost': -75, 'effectiveness': -0.05, 'quadrant': 'SW'},
                {'name': 'AI-Assisted Diagnosis', 'cost': 200, 'effectiveness': 0.18, 'quadrant': 'NE'},
                {'name': 'Telemedicine', 'cost': -50, 'effectiveness': 0.08, 'quadrant': 'NW'}
            ]
            
            result_path = create_cost_effectiveness_plane(
                scenarios=scenarios,
                output_path=output_path,
                willingness_to_pay=500  # $/QALY threshold
            )
            
            assert result_path.exists(), "Cost-effectiveness plane SVG not created"
            assert result_path.suffix == '.svg', "Output should be SVG format"
            
            # Check file content has expected elements
            content = result_path.read_text()
            assert 'Cost' in content or 'Effectiveness' in content
            assert 'Enhanced Screening' in content
    
    def test_cost_effectiveness_plane_quadrants(self):
        """Test cost-effectiveness plane correctly identifies quadrants"""
        from week5_figures import create_cost_effectiveness_plane, _classify_quadrant
        
        # Test quadrant classification
        assert _classify_quadrant(100, 0.1) == 'NE'  # More costly, more effective
        assert _classify_quadrant(-50, 0.1) == 'NW'  # Less costly, more effective (dominant)
        assert _classify_quadrant(100, -0.05) == 'SE'  # More costly, less effective (dominated)
        assert _classify_quadrant(-50, -0.05) == 'SW'  # Less costly, less effective
    
    def test_cost_effectiveness_icer_calculation(self):
        """Test ICER (Incremental Cost-Effectiveness Ratio) calculation"""
        from week5_figures import calculate_icer
        
        # Test basic ICER calculation
        icer = calculate_icer(
            delta_cost=100,
            delta_effectiveness=0.2
        )
        assert icer == 500, f"Expected ICER=500, got {icer}"
        
        # Test with zero effectiveness difference
        icer = calculate_icer(
            delta_cost=100,
            delta_effectiveness=0.0
        )
        assert icer == float('inf'), "Should return infinity for zero effectiveness difference"
        
        # Test with negative effectiveness difference
        icer = calculate_icer(
            delta_cost=100,
            delta_effectiveness=-0.1
        )
        assert icer == -1000, "Should handle negative effectiveness differences"
    
    def test_cost_effectiveness_with_confidence_intervals(self):
        """Test cost-effectiveness plane with uncertainty ellipses"""
        from week5_figures import create_cost_effectiveness_plane
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'cost_plane_ci.svg'
            
            scenarios = [
                {
                    'name': 'Intervention',
                    'cost': 150, 'effectiveness': 0.12,
                    'cost_ci_lower': 100, 'cost_ci_upper': 200,
                    'effectiveness_ci_lower': 0.08, 'effectiveness_ci_upper': 0.16
                }
            ]
            
            result_path = create_cost_effectiveness_plane(
                scenarios=scenarios,
                output_path=output_path,
                include_confidence_intervals=True
            )
            
            assert result_path.exists()


class TestFigureIntegration:
    """Test figure integration into documentation"""
    
    def test_embed_figures_in_week4_docs(self):
        """Test embedding new figures into Week 4 documentation"""
        from week5_figures import embed_figures_in_documentation
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock figures
            selection_path = Path(tmpdir) / 'selection_diagram.svg'
            cost_path = Path(tmpdir) / 'cost_plane.svg'
            
            selection_path.write_text('<svg>Mock Selection Diagram</svg>')
            cost_path.write_text('<svg>Mock Cost Plane</svg>')
            
            # Create mock documentation file
            doc_path = Path(tmpdir) / 'week4_analysis_report.md'
            doc_content = """# Week 4 Analysis Report

## Figures

### Existing Figures
- DAG diagram
- Love plot

### New Figures Section
<!-- WEEK5_FIGURES_PLACEHOLDER -->

## Analysis Results
"""
            doc_path.write_text(doc_content)
            
            # Test embedding
            updated_doc = embed_figures_in_documentation(
                doc_path=doc_path,
                selection_diagram_path=selection_path,
                cost_plane_path=cost_path
            )
            
            # Check that figures were embedded
            updated_content = updated_doc.read_text()
            assert 'selection_diagram.svg' in updated_content
            assert 'cost_plane.svg' in updated_content
            assert 'Selection Diagram' in updated_content
            assert 'Cost-Effectiveness' in updated_content
    
    def test_figure_list_update_in_readme(self):
        """Test updating figure list in README"""
        from week5_figures import update_figure_list_in_readme
        
        with tempfile.TemporaryDirectory() as tmpdir:
            readme_path = Path(tmpdir) / 'README.md'
            readme_content = """# SSD Analysis Project

## Generated Figures

- dag.svg - Causal directed acyclic graph
- love_plot.svg - Covariate balance assessment
- forest_plot.svg - Treatment effect estimates

<!-- FIGURE_LIST_END -->

## Analysis
"""
            readme_path.write_text(readme_content)
            
            new_figures = [
                'selection_diagram.svg - Patient selection flowchart',
                'cost_plane.svg - Cost-effectiveness analysis'
            ]
            
            updated_readme = update_figure_list_in_readme(
                readme_path=readme_path,
                new_figures=new_figures
            )
            
            updated_content = updated_readme.read_text()
            assert 'selection_diagram.svg' in updated_content
            assert 'cost_plane.svg' in updated_content
            assert 'Patient selection flowchart' in updated_content


if __name__ == "__main__":
    pytest.main([__file__])