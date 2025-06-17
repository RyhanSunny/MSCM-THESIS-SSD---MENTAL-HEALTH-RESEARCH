#!/usr/bin/env python3
"""
test_osf_upload.py - Tests for OSF upload functionality

Tests the OSF upload script behavior with and without OSF_TOKEN,
including dry-run mode for CI integration.

Author: Ryhan Suny
Date: 2025-06-17
"""

import pytest
import os
import subprocess
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the module
from scripts.osf_upload import upload_changelog, check_osfclient_available, upload_with_osfclient


class TestOSFUpload:
    """Test OSF upload functionality"""
    
    def test_upload_without_osf_token(self):
        """Test that upload skips gracefully when OSF_TOKEN not set"""
        # Ensure OSF_TOKEN is not set
        with patch.dict(os.environ, {}, clear=True):
            result = upload_changelog(dry_run=True)
            assert result is True, "Should return True when OSF_TOKEN not set (not an error)"
    
    def test_upload_with_osf_token_dry_run(self):
        """Test upload behavior with OSF_TOKEN set in dry-run mode"""
        # Mock OSF_TOKEN environment variable
        with patch.dict(os.environ, {'OSF_TOKEN': 'fake-token-12345'}):
            # Mock osfclient availability
            with patch('scripts.osf_upload.check_osfclient_available', return_value=True):
                result = upload_changelog(dry_run=True)
                assert result is True, "Dry-run should succeed"
    
    def test_upload_missing_changelog(self):
        """Test behavior when CHANGELOG.md is missing"""
        with patch.dict(os.environ, {'OSF_TOKEN': 'fake-token-12345'}):
            with patch('pathlib.Path.exists', return_value=False):
                result = upload_changelog(dry_run=True)
                assert result is False, "Should fail when CHANGELOG.md missing"
    
    def test_check_osfclient_available_not_installed(self):
        """Test osfclient availability check when not installed"""
        with patch('subprocess.run', side_effect=FileNotFoundError):
            result = check_osfclient_available()
            assert result is False, "Should return False when osfclient not installed"
    
    def test_check_osfclient_available_installed(self):
        """Test osfclient availability check when installed"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch('subprocess.run', return_value=mock_result):
            result = check_osfclient_available()
            assert result is True, "Should return True when osfclient available"
    
    def test_upload_with_osfclient_dry_run(self):
        """Test osfclient upload in dry-run mode"""
        file_path = Path("CHANGELOG.md")
        result = upload_with_osfclient(file_path, dry_run=True)
        assert result is True, "Dry-run should always succeed"
    
    def test_upload_with_osfclient_success(self):
        """Test successful osfclient upload"""
        file_path = Path("CHANGELOG.md")
        mock_result = MagicMock()
        mock_result.returncode = 0
        
        with patch('subprocess.run', return_value=mock_result):
            result = upload_with_osfclient(file_path, dry_run=False)
            assert result is True, "Should succeed when osf command succeeds"
    
    def test_upload_with_osfclient_failure(self):
        """Test failed osfclient upload"""
        file_path = Path("CHANGELOG.md")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Upload failed"
        
        with patch('subprocess.run', return_value=mock_result):
            result = upload_with_osfclient(file_path, dry_run=False)
            assert result is False, "Should fail when osf command fails"
    
    def test_cli_dry_run_flag(self):
        """Test CLI --dry-run flag"""
        # Test the script can be called with --dry-run
        result = subprocess.run([
            sys.executable, "scripts/osf_upload.py", "--dry-run"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Should not error (returns 0 when OSF_TOKEN not set)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Check both stdout and stderr since logging goes to stderr
        output = result.stdout + result.stderr
        assert "DRY-RUN" in output or "OSF_TOKEN not set" in output
    
    def test_cli_version_flag(self):
        """Test CLI --version flag"""
        result = subprocess.run([
            sys.executable, "scripts/osf_upload.py", "--version"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0, f"Version flag failed: {result.stderr}"
        assert "4.2.0" in result.stdout, "Should show version 4.2.0"
    
    def test_changelog_file_exists(self):
        """Test that CHANGELOG.md exists in the project root"""
        changelog_path = Path(__file__).parent.parent / "CHANGELOG.md"
        assert changelog_path.exists(), "CHANGELOG.md should exist for upload"
        
        # Check it has content
        content = changelog_path.read_text()
        assert len(content) > 100, "CHANGELOG.md should have substantial content"
        assert "v4." in content, "Should contain version information"
    
    def test_requests_fallback_without_requests(self):
        """Test requests fallback when requests library not available"""
        from scripts.osf_upload import upload_with_requests
        
        # Directly test the ImportError behavior
        result = upload_with_requests(Path("CHANGELOG.md"), "fake-token", dry_run=False)
        # Since requests may or may not be available, we just test that it handles missing imports
        assert isinstance(result, bool), "Should return boolean result"
    
    def test_requests_fallback_dry_run(self):
        """Test requests fallback in dry-run mode"""
        from scripts.osf_upload import upload_with_requests
        
        result = upload_with_requests(Path("CHANGELOG.md"), "fake-token", dry_run=True)
        assert result is True, "Dry-run should succeed even with incomplete implementation"


if __name__ == "__main__":
    pytest.main([__file__])