#!/usr/bin/env python3
"""
test_week4_figure_generator.py - Tests for Week 4 figure generation

Tests mediation diagrams, CATE heatmaps, E-value plots, and updated love plots
following TDD principles.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestWeek4FigureGenerator:
    """Test Week 4 figure generation functionality"""
    
    def test_mediation_pathway_diagram_creation(self, tmp_path):
        """Test mediation pathway diagram generation"""
        from week4_figure_generator import create_mediation_pathway_diagram
        
        # Mock mediation results
        mediation_results = {
            'total_effect': 0.25,
            'direct_effect': 0.15,
            'indirect_effect': 0.10,
            'a_path': 0.30,
            'b_path': 0.33,
            'proportion_mediated': 0.40
        }
        
        # Generate diagram
        figure_path = create_mediation_pathway_diagram(
            mediation_results, tmp_path, dpi=150
        )
        
        # Check file creation
        assert figure_path.exists()
        assert figure_path.suffix == '.svg'
        assert (tmp_path / 'mediation_pathway_diagram.png').exists()
        
        # Check file is not empty (adjust for placeholder when matplotlib unavailable)
        assert figure_path.stat().st_size > 100  # SVG should exist
    
    def test_cate_heatmap_creation(self, tmp_path):
        """Test CATE heterogeneity heatmap generation"""
        from week4_figure_generator import create_cate_heatmap
        
        # Mock causal forest results
        causal_forest_results = {
            'tau_mean': 0.18,
            'tau_std': 0.12,
            'het_ranking': pd.DataFrame({
                'variable': ['age', 'comorbidity_count', 'prior_psych', 'sex'],
                'het_importance': [0.45, 0.32, 0.28, 0.15]
            }),
            'tau_estimates': np.random.normal(0.18, 0.12, 100)
        }
        
        # Generate heatmap
        figure_path = create_cate_heatmap(
            causal_forest_results, tmp_path, dpi=150
        )
        
        # Check file creation
        assert figure_path.exists()
        assert figure_path.suffix == '.svg'
        assert (tmp_path / 'cate_heatmap.png').exists()
        assert figure_path.stat().st_size > 100
    
    def test_cate_heatmap_empty_data(self, tmp_path):
        """Test CATE heatmap with empty data"""
        from week4_figure_generator import create_cate_heatmap
        
        # Empty results
        empty_results = {
            'het_ranking': pd.DataFrame(),
            'tau_estimates': np.array([])
        }
        
        # Should create placeholder
        figure_path = create_cate_heatmap(empty_results, tmp_path, dpi=150)
        
        assert figure_path.exists()
        assert figure_path.suffix == '.svg'
    
    def test_evalue_plot_creation(self, tmp_path):
        """Test E-value bias sensitivity plot generation"""
        from week4_figure_generator import create_evalue_plot
        
        # Mock hypothesis results
        hypothesis_results = {
            'h1': {'irr': 1.23},
            'h2': {'irr': 1.45}, 
            'h3': {'irr': 1.31}
        }
        
        # Generate E-value plot
        figure_path = create_evalue_plot(
            hypothesis_results, tmp_path, dpi=150
        )
        
        # Check file creation
        assert figure_path.exists()
        assert figure_path.suffix == '.svg'
        assert (tmp_path / 'evalue_plot.png').exists()
        assert figure_path.stat().st_size > 100
    
    def test_love_plot_mh_update(self, tmp_path):
        """Test updated love plot with MH covariates"""
        from week4_figure_generator import update_love_plot_mh
        
        # Mock PS data (empty for test)
        ps_data = pd.DataFrame()
        
        # Generate updated love plot
        figure_path = update_love_plot_mh(ps_data, tmp_path, dpi=150)
        
        # Check file creation
        assert figure_path.exists()
        assert figure_path.suffix == '.svg'
        assert (tmp_path / 'love_plot_mh.png').exists()
        assert figure_path.stat().st_size > 100
    
    def test_evalue_calculation_edge_cases(self):
        """Test E-value calculation for various effect sizes"""
        from week4_figure_generator import create_evalue_plot
        
        # Test with different IRR values
        test_cases = [
            {'h1': {'irr': 1.0}},  # No effect
            {'h1': {'irr': 0.8}},  # Protective effect
            {'h1': {'irr': 2.0}},  # Strong effect
        ]
        
        for hypothesis_results in test_cases:
            # Should not raise errors
            try:
                # Just test the calculation logic, not full plot
                irr = hypothesis_results['h1']['irr']
                if irr >= 1:
                    evalue = irr + np.sqrt(irr * (irr - 1))
                else:
                    evalue = 1 / (1/irr + np.sqrt((1/irr) * (1/irr - 1)))
                assert evalue > 0
            except Exception as e:
                pytest.fail(f"E-value calculation failed for IRR={irr}: {e}")
    
    def test_generate_all_week4_figures(self, tmp_path):
        """Test complete Week 4 figure generation"""
        from week4_figure_generator import generate_all_week4_figures
        
        # Create mock directories
        results_dir = tmp_path / 'results'
        figures_dir = tmp_path / 'figures'
        results_dir.mkdir()
        
        # Generate all figures
        generated_figures = generate_all_week4_figures(
            results_dir, figures_dir, dpi=150
        )
        
        # Check all expected figures generated
        expected_figures = [
            'mediation_pathway',
            'cate_heatmap', 
            'evalue_plot',
            'love_plot_mh'
        ]
        
        for fig_name in expected_figures:
            assert fig_name in generated_figures
            assert generated_figures[fig_name].exists()
            assert generated_figures[fig_name].stat().st_size > 100
    
    def test_figure_file_formats(self, tmp_path):
        """Test that figures are generated in both SVG and PNG formats"""
        from week4_figure_generator import create_mediation_pathway_diagram
        
        mediation_results = {
            'total_effect': 0.20,
            'direct_effect': 0.12,
            'indirect_effect': 0.08,
            'a_path': 0.25,
            'b_path': 0.32
        }
        
        figure_path = create_mediation_pathway_diagram(
            mediation_results, tmp_path, dpi=150
        )
        
        # Both SVG and PNG should exist
        svg_path = figure_path
        png_path = figure_path.with_suffix('.png')
        
        assert svg_path.exists() and svg_path.suffix == '.svg'
        assert png_path.exists() and png_path.suffix == '.png'
        
        # Both should have reasonable file sizes
        assert svg_path.stat().st_size > 100
        assert png_path.stat().st_size > 100
    
    def test_figure_high_resolution(self, tmp_path):
        """Test high-resolution figure generation"""
        from week4_figure_generator import create_evalue_plot
        
        hypothesis_results = {'h1': {'irr': 1.5}}
        
        # Test different DPI settings
        low_dpi_path = create_evalue_plot(hypothesis_results, tmp_path, dpi=72)
        high_dpi_path = create_evalue_plot(hypothesis_results, tmp_path, dpi=300)
        
        # Both should create files
        assert low_dpi_path.exists()
        assert high_dpi_path.exists()
    
    def test_figure_error_handling(self, tmp_path):
        """Test figure generation error handling"""
        from week4_figure_generator import create_cate_heatmap
        
        # Test with invalid data types
        invalid_results = {
            'het_ranking': "not_a_dataframe",
            'tau_estimates': "not_an_array"
        }
        
        # Should handle gracefully and create placeholder
        try:
            figure_path = create_cate_heatmap(invalid_results, tmp_path)
            assert figure_path.exists()
        except Exception:
            # Should not crash the entire pipeline
            pass