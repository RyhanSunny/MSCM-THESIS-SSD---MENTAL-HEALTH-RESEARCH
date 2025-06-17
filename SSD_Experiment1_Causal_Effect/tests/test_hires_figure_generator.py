#!/usr/bin/env python3
"""
test_hires_figure_generator.py - Tests for high-resolution figure generation

Tests that figures are generated at ≥300 DPI in both vector (SVG) and raster (PNG) formats
for manuscript submission requirements.
"""

import pytest
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestHiresFigureGenerator:
    """Test high-resolution figure generation for manuscript submission"""
    
    def test_hires_directory_creation(self, tmp_path):
        """Test that hires directory is created"""
        from hires_figure_generator import HiresFigureGenerator
        
        generator = HiresFigureGenerator(output_dir=tmp_path)
        generator.create_hires_directory()
        
        assert (tmp_path / "figures" / "hires").exists()
        assert (tmp_path / "figures" / "hires").is_dir()
    
    def test_dag_generation_high_resolution(self, tmp_path):
        """Test DAG is generated at high resolution"""
        from hires_figure_generator import HiresFigureGenerator
        
        generator = HiresFigureGenerator(output_dir=tmp_path)
        
        # Test DAG generation
        dag_files = generator.generate_causal_dag()
        
        assert len(dag_files) == 2  # SVG and PNG
        assert any(f.suffix == '.svg' for f in dag_files)
        assert any(f.suffix == '.png' for f in dag_files)
        
        # Check files exist
        for f in dag_files:
            assert f.exists()
    
    def test_selection_diagram_generation(self, tmp_path):
        """Test STROBE selection diagram generation"""
        from hires_figure_generator import HiresFigureGenerator
        
        generator = HiresFigureGenerator(output_dir=tmp_path)
        
        # Test selection diagram
        diagram_files = generator.generate_selection_diagram()
        
        assert len(diagram_files) == 2  # SVG and PNG
        assert all(f.parent.name == 'hires' for f in diagram_files)
    
    @patch('matplotlib.pyplot.savefig')
    def test_dpi_settings(self, mock_savefig, tmp_path):
        """Test that figures are saved with correct DPI"""
        from hires_figure_generator import HiresFigureGenerator
        
        generator = HiresFigureGenerator(output_dir=tmp_path)
        generator._save_figure_hires(MagicMock(), "test_figure")
        
        # Check savefig was called with correct DPI
        png_calls = [c for c in mock_savefig.call_args_list 
                     if 'png' in str(c)]
        assert any('dpi=300' in str(c) for c in png_calls)
    
    def test_all_figures_converted(self, tmp_path):
        """Test that all existing figures are converted to high-res"""
        from hires_figure_generator import HiresFigureGenerator
        
        # Create mock existing figures
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        existing_figures = ['dag.svg', 'forest_plot.svg', 
                          'love_plot.svg', 'consort_flowchart.svg']
        for fig in existing_figures:
            (figures_dir / fig).touch()
        
        generator = HiresFigureGenerator(output_dir=tmp_path)
        converted = generator.convert_existing_figures()
        
        # Check all figures have hires versions
        assert len(converted) >= len(existing_figures)
        
        hires_dir = figures_dir / "hires"
        for fig in existing_figures:
            base_name = fig.replace('.svg', '')
            assert (hires_dir / f"{base_name}.png").exists() or \
                   (hires_dir / f"{base_name}_hires.png").exists()
    
    def test_metadata_preservation(self, tmp_path):
        """Test that figure metadata is preserved"""
        from hires_figure_generator import HiresFigureGenerator
        
        generator = HiresFigureGenerator(output_dir=tmp_path)
        
        # Generate figure with metadata
        metadata = {
            'title': 'Test Figure',
            'dpi': 300,
            'format': ['svg', 'png'],
            'description': 'High-resolution test figure'
        }
        
        files = generator.generate_figure_with_metadata(
            'test_figure', metadata
        )
        
        # Check metadata file exists
        metadata_file = tmp_path / "figures" / "hires" / "test_figure.metadata.json"
        assert metadata_file.exists()
    
    def test_bundle_creation(self, tmp_path):
        """Test creation of figures bundle for submission"""
        from hires_figure_generator import HiresFigureGenerator
        
        generator = HiresFigureGenerator(output_dir=tmp_path)
        
        # Create bundle
        bundle_path = generator.create_figures_bundle()
        
        assert bundle_path.exists()
        assert bundle_path.suffix == '.zip'
        assert 'figures_bundle' in bundle_path.name
    
    def test_figure_validation(self, tmp_path):
        """Test validation of generated figures"""
        from hires_figure_generator import HiresFigureGenerator
        
        generator = HiresFigureGenerator(output_dir=tmp_path)
        
        # Generate test figures
        generator.generate_all_hires_figures()
        
        # Validate
        validation_results = generator.validate_figures()
        
        assert validation_results['all_valid']
        assert all(r['dpi'] >= 300 for r in validation_results['figures'] 
                  if r['format'] == 'png')