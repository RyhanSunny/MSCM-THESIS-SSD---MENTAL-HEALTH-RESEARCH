#!/usr/bin/env python3
"""
test_documentation_generator.py - Tests for supplementary documentation generation

Tests generation of Methods supplement, STROBE-CI checklist, ROBINS-I assessment,
and Glossary for manuscript submission.
"""

import pytest
from pathlib import Path
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestDocumentationGenerator:
    """Test supplementary documentation generation"""
    
    def test_methods_supplement_generation(self, tmp_path):
        """Test Methods supplement generation in LaTeX and Markdown"""
        from documentation_generator import DocumentationGenerator
        
        generator = DocumentationGenerator(output_dir=tmp_path)
        
        # Generate methods supplement
        methods_files = generator.generate_methods_supplement()
        
        assert len(methods_files) == 2  # LaTeX and Markdown
        assert any(f.suffix == '.tex' for f in methods_files)
        assert any(f.suffix == '.md' for f in methods_files)
        
        # Check content includes key sections
        md_file = next(f for f in methods_files if f.suffix == '.md')
        content = md_file.read_text()
        
        assert "Study Design" in content
        assert "Statistical Analysis" in content
        assert "Causal Inference Methods" in content
        assert "Code Snippets" in content
    
    def test_strobe_checklist_generation(self, tmp_path):
        """Test STROBE-CI checklist with line references"""
        from documentation_generator import DocumentationGenerator
        
        generator = DocumentationGenerator(output_dir=tmp_path)
        
        # Generate STROBE checklist
        strobe_file = generator.generate_strobe_checklist()
        
        assert strobe_file.exists()
        assert strobe_file.suffix == '.md'
        
        content = strobe_file.read_text()
        assert "STROBE-CI Checklist" in content
        assert "Line" in content  # Line references
        assert "Page" in content  # Page references
    
    def test_robins_i_assessment(self, tmp_path):
        """Test ROBINS-I bias assessment form generation"""
        from documentation_generator import DocumentationGenerator
        
        generator = DocumentationGenerator(output_dir=tmp_path)
        
        # Generate ROBINS-I assessment
        robins_file = generator.generate_robins_i_assessment()
        
        assert robins_file.exists()
        
        content = robins_file.read_text()
        assert "ROBINS-I" in content
        assert "Bias due to confounding" in content
        assert "Bias in selection" in content
        assert "Bias in classification" in content
    
    def test_glossary_generation(self, tmp_path):
        """Test glossary generation and relocation"""
        from documentation_generator import DocumentationGenerator
        
        generator = DocumentationGenerator(output_dir=tmp_path)
        
        # Generate glossary
        glossary_file = generator.generate_glossary()
        
        assert glossary_file.exists()
        assert glossary_file.parent.name == 'docs'
        assert glossary_file.name == 'Glossary.md'
        
        content = glossary_file.read_text()
        assert "Glossary" in content
        assert "SSD" in content
        assert "IPTW" in content
        assert "TMLE" in content
    
    def test_code_snippet_embedding(self, tmp_path):
        """Test that code snippets are properly embedded"""
        from documentation_generator import DocumentationGenerator
        
        generator = DocumentationGenerator(output_dir=tmp_path)
        
        # Create mock code file
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        code_file = src_dir / "example.py"
        code_file.write_text("""
def calculate_iptw(ps):
    '''Calculate inverse probability weights'''
    return 1 / ps
""")
        
        # Generate methods with code embedding
        methods_files = generator.generate_methods_supplement()
        
        # Check code is embedded
        md_file = next(f for f in methods_files if f.suffix == '.md')
        content = md_file.read_text()
        
        assert "```python" in content or "calculate_iptw" in content
    
    def test_latex_compilation_ready(self, tmp_path):
        """Test that LaTeX files are compilation-ready"""
        from documentation_generator import DocumentationGenerator
        
        generator = DocumentationGenerator(output_dir=tmp_path)
        
        # Generate LaTeX methods
        methods_files = generator.generate_methods_supplement()
        tex_file = next(f for f in methods_files if f.suffix == '.tex')
        
        content = tex_file.read_text()
        
        # Check LaTeX structure
        assert "\\documentclass" in content
        assert "\\begin{document}" in content
        assert "\\end{document}" in content
        assert "\\section" in content
    
    def test_cross_reference_validation(self, tmp_path):
        """Test that cross-references are valid"""
        from documentation_generator import DocumentationGenerator
        
        generator = DocumentationGenerator(output_dir=tmp_path)
        
        # Generate all documentation
        all_docs = generator.generate_all_documentation()
        
        # Validate cross-references
        validation = generator.validate_cross_references()
        
        assert validation['valid']
        assert len(validation['broken_refs']) == 0
    
    def test_bundle_documentation(self, tmp_path):
        """Test bundling all documentation"""
        from documentation_generator import DocumentationGenerator
        
        generator = DocumentationGenerator(output_dir=tmp_path)
        
        # Generate and bundle
        bundle_path = generator.create_documentation_bundle()
        
        assert bundle_path.exists()
        assert bundle_path.suffix == '.zip'
        assert 'documentation_bundle' in bundle_path.name