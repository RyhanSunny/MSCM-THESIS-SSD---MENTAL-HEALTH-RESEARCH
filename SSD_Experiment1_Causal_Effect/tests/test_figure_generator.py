#!/usr/bin/env python3
"""
Test suite for figure generator
Following TDD principles - tests written first per CLAUDE.md requirements
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from figure_generator import SSDFigureGenerator


class TestSSDFigureGenerator:
    """Test suite for figure generation"""
    
    def test_init_creates_figures_directory(self):
        """Test that initialization creates figures directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            figures_dir = Path(tmp_dir) / "test_figures"
            generator = SSDFigureGenerator(figures_dir)
            
            assert figures_dir.exists()
            assert figures_dir.is_dir()
    
    def test_generate_causal_dag(self):
        """Test DAG generation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SSDFigureGenerator(Path(tmp_dir))
            generator.generate_causal_dag()
            
            # Check that DAG file was created
            dag_files = list(Path(tmp_dir).glob('dag*'))
            assert len(dag_files) > 0
            assert any(f.suffix in ['.svg', '.pdf'] for f in dag_files)
    
    def test_generate_love_plot(self):
        """Test Love plot generation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SSDFigureGenerator(Path(tmp_dir))
            generator.generate_love_plot()
            
            # Check that Love plot was created
            love_plot = Path(tmp_dir) / 'love_plot.svg'
            assert love_plot.exists()
            assert love_plot.stat().st_size > 0
    
    def test_generate_forest_plot(self):
        """Test forest plot generation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SSDFigureGenerator(Path(tmp_dir))
            
            # Create mock results for testing
            import json
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            mock_h1 = {
                'description': 'Normal lab cascade → healthcare encounters',
                'treatment_effect': {
                    'irr': 1.25,
                    'irr_ci_lower': 1.10,
                    'irr_ci_upper': 1.42,
                    'p_value': 0.001
                }
            }
            
            # Generate forest plot
            generator.generate_forest_plot()
            
            # Check that forest plot was created
            forest_plot = Path(tmp_dir) / 'forest_plot.svg'
            assert forest_plot.exists()
    
    def test_generate_consort_flowchart(self):
        """Test CONSORT flowchart generation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SSDFigureGenerator(Path(tmp_dir))
            generator.generate_consort_flowchart()
            
            # Check that flowchart was created
            flowchart = Path(tmp_dir) / 'consort_flowchart.svg'
            assert flowchart.exists()
            assert flowchart.stat().st_size > 0
    
    def test_generate_all_figures(self):
        """Test generating all figures at once"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SSDFigureGenerator(Path(tmp_dir))
            figures = generator.generate_all_figures()
            
            # Should generate 4 figures
            assert len(figures) >= 4
            
            # Check specific files exist
            expected_files = [
                'dag.svg',  # or dag (graphviz adds extension)
                'love_plot.svg',
                'forest_plot.svg',
                'consort_flowchart.svg'
            ]
            
            generated_names = [f.name for f in figures]
            for expected in expected_files:
                # Check if file exists (graphviz might not have .svg in name)
                assert any(expected in name or expected.replace('.svg', '') in name 
                          for name in generated_names)
    
    def test_figure_quality_checks(self):
        """Test that figures meet quality requirements"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SSDFigureGenerator(Path(tmp_dir))
            
            # Generate a test figure
            generator.generate_love_plot()
            
            # Check file size (should be reasonable)
            love_plot = Path(tmp_dir) / 'love_plot.svg'
            file_size_kb = love_plot.stat().st_size / 1024
            
            # SVG should be between 1KB and 1MB
            assert 1 < file_size_kb < 1024
            
            # Check that it's actually an SVG
            with open(love_plot, 'r') as f:
                content = f.read(100)
                assert '<svg' in content or '<?xml' in content


class TestFigureIntegration:
    """Test integration with Week 2 requirements"""
    
    def test_figures_use_actual_results(self):
        """Test that forest plot uses actual H1-H3 results when available"""
        # This would be tested with actual results files
        pass
    
    def test_figure_formatting_compliance(self):
        """Test figures meet publication standards"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = SSDFigureGenerator(Path(tmp_dir))
            
            # Generate figures
            generator.generate_love_plot()
            generator.generate_forest_plot()
            
            # Check that files are vector format (SVG)
            for fig in Path(tmp_dir).glob('*.svg'):
                assert fig.suffix == '.svg'
                
                # Basic check that it's valid SVG
                with open(fig, 'r') as f:
                    content = f.read()
                    assert content.strip().endswith('</svg>')


if __name__ == "__main__":
    # Run basic demonstrations
    test_instance = TestSSDFigureGenerator()
    
    print("Running figure generator tests...")
    test_instance.test_generate_causal_dag()
    print("✓ DAG generation test passed")
    
    test_instance.test_generate_love_plot()
    print("✓ Love plot generation test passed")
    
    test_instance.test_generate_forest_plot()
    print("✓ Forest plot generation test passed")
    
    test_instance.test_generate_consort_flowchart()
    print("✓ CONSORT flowchart test passed")
    
    test_instance.test_generate_all_figures()
    print("✓ All figures generation test passed")
    
    print("\nAll figure generator tests passed! Ready for implementation.")