#!/usr/bin/env python3
"""
test_conceptual_framework.py - Tests for conceptual framework generator

Following TDD approach per CLAUDE.md requirements.

Author: Ryhan Suny
Date: 2025-07-01
"""

import unittest
from pathlib import Path
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from conceptual_framework_generator import (
    create_conceptual_framework,
    _add_conceptual_boxes,
    _add_conceptual_arrows,
    _add_framework_legend
)


class TestConceptualFramework(unittest.TestCase):
    """Test conceptual framework generation."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("test_figures")
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        # Remove test figures
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()
    
    def test_create_conceptual_framework(self):
        """Test main framework creation."""
        # Create framework
        output_path = create_conceptual_framework(self.test_dir)
        
        # Check file exists
        self.assertTrue(Path(output_path).exists())
        
        # Check SVG and PDF versions
        svg_path = Path(output_path)
        pdf_path = svg_path.with_suffix('.pdf')
        self.assertTrue(svg_path.exists())
        self.assertTrue(pdf_path.exists())
        
        # Check file size (should be non-empty)
        self.assertGreater(svg_path.stat().st_size, 1000)
        self.assertGreater(pdf_path.stat().st_size, 1000)
    
    def test_framework_components(self):
        """Test that all required components are present."""
        # This would require parsing SVG, simplified for now
        output_path = create_conceptual_framework(self.test_dir)
        
        # Read SVG content
        with open(output_path, 'r') as f:
            svg_content = f.read()
        
        # Check for key text elements
        required_texts = [
            'Mental Health',
            'Normal Lab',
            'Referrals',
            'Psychotropic',
            'SSDSI',
            'Healthcare',
            'H1', 'H2', 'H3', 'H4', 'H5'
        ]
        
        for text in required_texts:
            self.assertIn(text, svg_content, 
                         f"Missing required text: {text}")
    
    def test_file_naming(self):
        """Test that files are named with timestamp."""
        output_path = create_conceptual_framework(self.test_dir)
        filename = Path(output_path).name
        
        # Check format: conceptual_framework_YYYYMMDD_HHMMSS.svg
        self.assertTrue(filename.startswith('conceptual_framework_'))
        self.assertTrue(filename.endswith('.svg'))
        
        # Extract timestamp part
        timestamp_part = filename.replace('conceptual_framework_', '')\
                                .replace('.svg', '')
        self.assertEqual(len(timestamp_part), 15)  # YYYYMMDD_HHMMSS


if __name__ == '__main__':
    unittest.main()