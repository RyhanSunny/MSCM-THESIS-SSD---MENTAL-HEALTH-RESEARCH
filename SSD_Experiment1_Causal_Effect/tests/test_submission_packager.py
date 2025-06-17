#!/usr/bin/env python3
"""
test_submission_packager.py - Tests for submission package creation

Tests bundling of all artifacts for manuscript submission and OSF upload.
"""

import pytest
from pathlib import Path
import sys
import os
import zipfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSubmissionPackager:
    """Test submission package creation"""
    
    def test_submission_package_creation(self, tmp_path):
        """Test creation of complete submission package"""
        from submission_packager import SubmissionPackager
        
        packager = SubmissionPackager(output_dir=tmp_path)
        
        # Create mock artifacts
        (tmp_path / "figures").mkdir()
        (tmp_path / "figures" / "dag.svg").touch()
        (tmp_path / "tables").mkdir()
        (tmp_path / "tables" / "main_results.md").touch()
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "Methods_Supplement.md").touch()
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "hypothesis_h1.json").touch()
        
        # Create package
        package_path = packager.create_submission_package()
        
        assert package_path.exists()
        assert package_path.suffix == '.zip'
        assert 'SSD_Week3' in package_path.name
    
    def test_package_contents(self, tmp_path):
        """Test that package contains all required components"""
        from submission_packager import SubmissionPackager
        
        packager = SubmissionPackager(output_dir=tmp_path)
        
        # Create complete mock structure
        structure = {
            'figures/hires/dag.svg': 'DAG content',
            'tables/main_results.md': 'Results table',
            'docs/Methods_Supplement.md': 'Methods',
            'results/hypothesis_h1.json': '{"irr": 1.2}',
            'submission_package/figures_bundle.zip': 'figures',
            'src/weight_diagnostics.py': 'code'
        }
        
        for path, content in structure.items():
            file_path = tmp_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
        
        # Create package
        package_path = packager.create_submission_package()
        
        # Check contents
        with zipfile.ZipFile(package_path, 'r') as zf:
            file_list = zf.namelist()
            
            assert any('figures/' in f for f in file_list)
            assert any('tables/' in f for f in file_list)  
            assert any('docs/' in f for f in file_list)
            assert any('results/' in f for f in file_list)
            assert any('code/' in f for f in file_list)
    
    def test_osf_upload_script(self, tmp_path):
        """Test OSF upload script generation"""
        from submission_packager import SubmissionPackager
        
        packager = SubmissionPackager(output_dir=tmp_path)
        
        # Generate upload script
        script_path = packager.generate_osf_upload_script()
        
        assert script_path.exists()
        
        content = script_path.read_text()
        assert 'osf' in content.lower()
        assert 'upload' in content.lower()
        assert executable_permissions(script_path)
    
    def test_manifest_generation(self, tmp_path):
        """Test manifest file generation"""
        from submission_packager import SubmissionPackager
        
        packager = SubmissionPackager(output_dir=tmp_path)
        
        # Create test files
        test_files = ['fig1.svg', 'table1.md', 'methods.tex']
        for fname in test_files:
            (tmp_path / fname).touch()
        
        # Generate manifest
        manifest = packager.generate_manifest(test_files)
        
        assert 'files' in manifest
        assert len(manifest['files']) == len(test_files)
        assert 'generated' in manifest
        assert 'total_size' in manifest
    
    def test_checksum_validation(self, tmp_path):
        """Test checksum generation for integrity"""
        from submission_packager import SubmissionPackager
        
        packager = SubmissionPackager(output_dir=tmp_path)
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Generate checksums
        checksums = packager.generate_checksums([test_file])
        
        assert test_file.name in checksums
        assert len(checksums[test_file.name]) == 64  # SHA256 length
    
    def test_readme_generation(self, tmp_path):
        """Test README generation for submission package"""
        from submission_packager import SubmissionPackager
        
        packager = SubmissionPackager(output_dir=tmp_path)
        
        # Generate README
        readme_path = packager.generate_package_readme()
        
        assert readme_path.exists()
        
        content = readme_path.read_text()
        assert "SSD Causal Analysis" in content
        assert "Manuscript Submission" in content
        assert "Week 3" in content
    
    def test_docker_save(self, tmp_path):
        """Test Docker image export"""
        from submission_packager import SubmissionPackager
        
        packager = SubmissionPackager(output_dir=tmp_path)
        
        # Mock Docker save (real would require Docker)
        docker_path = packager.save_docker_image()
        
        # Should create placeholder or skip if Docker unavailable
        assert docker_path is None or docker_path.exists()
    
    def test_complete_packaging_workflow(self, tmp_path):
        """Test complete packaging workflow"""
        from submission_packager import SubmissionPackager
        
        packager = SubmissionPackager(output_dir=tmp_path)
        
        # Create full mock environment
        dirs = ['figures/hires', 'tables', 'docs', 'results', 'src']
        for d in dirs:
            (tmp_path / d).mkdir(parents=True, exist_ok=True)
            (tmp_path / d / 'dummy.txt').touch()
        
        # Run complete workflow
        results = packager.package_for_submission()
        
        assert 'package_path' in results
        assert 'manifest' in results
        assert 'checksums' in results
        assert 'osf_script' in results


def executable_permissions(file_path):
    """Check if file has executable permissions"""
    return os.access(file_path, os.X_OK)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])