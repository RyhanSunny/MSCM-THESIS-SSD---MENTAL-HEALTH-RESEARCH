"""
Test for Makefile installation target compliance.

Following TDD principles as per CLAUDE.md requirements.
"""

import os
import subprocess
import pytest
from pathlib import Path


class TestMakefileInstallTarget:
    """Test Makefile install target uses environment.yml instead of requirements.txt."""
    
    def test_makefile_install_target_exists(self):
        """Test that install target exists in Makefile."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        assert makefile_path.exists(), "Makefile not found"
        
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        assert ".PHONY: install" in content, "install target not found in Makefile"
        assert "install:" in content, "install target definition not found"
    
    def test_makefile_install_uses_environment_yml(self):
        """Test that install target uses conda env create from environment.yml."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Extract install target section
        lines = content.split('\n')
        install_start = None
        install_lines = []
        
        for i, line in enumerate(lines):
            if line.strip() == "install:":
                install_start = i
                continue
            elif install_start is not None:
                # Stop at next target or end of file
                if line.startswith('.PHONY:') or (line and not line.startswith('\t')):
                    break
                install_lines.append(line)
        
        assert install_start is not None, "install target not found"
        install_content = '\n'.join(install_lines)
        
        # Should use conda environment.yml, not pip requirements.txt
        assert "environment.yml" in install_content, "install target should reference environment.yml"
        assert "conda env create" in install_content or "mamba env create" in install_content, \
            "install target should use conda/mamba env create"
        
        # Should NOT use pip requirements.txt 
        assert "requirements.txt" not in install_content, \
            "install target should not use requirements.txt (use environment.yml instead)"
    
    def test_environment_yml_exists(self):
        """Test that environment.yml file exists."""
        env_path = Path(__file__).parent.parent / "environment.yml"
        assert env_path.exists(), "environment.yml file not found"
    
    def test_environment_yml_has_conda_format(self):
        """Test that environment.yml has proper conda environment format."""
        env_path = Path(__file__).parent.parent / "environment.yml"
        
        with open(env_path, 'r') as f:
            content = f.read()
        
        # Should have conda environment format
        assert "name:" in content, "environment.yml should specify environment name"
        assert "channels:" in content, "environment.yml should specify channels"
        assert "dependencies:" in content, "environment.yml should specify dependencies"


# Run the failing test to ensure TDD compliance
if __name__ == "__main__":
    test = TestMakefileInstallTarget()
    try:
        test.test_makefile_install_uses_environment_yml()
        print("❌ TEST SHOULD FAIL (TDD requirement)")
    except AssertionError as e:
        print(f"✓ Test fails as expected: {e}")
        print("Now implementing fix...")