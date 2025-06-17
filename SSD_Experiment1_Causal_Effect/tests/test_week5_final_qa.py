#!/usr/bin/env python3
"""
test_week5_final_qa.py - Tests for Week 5 final QA & release

Tests for Week 5 Task G: Final QA & release v4.1.0
Validates complete pipeline quality gates and version release readiness.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.1.0
"""

import pytest
import subprocess
import sys
from pathlib import Path
import yaml
import json
import re


class TestWeek5FinalQA:
    """Test final QA validation pipeline"""
    
    def test_all_week5_tests_pass(self):
        """Test that all Week 5 tests pass"""
        # Test reconcile estimates
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_reconcile_estimates.py', '-v'
        ], capture_output=True, text=True, cwd=Path.cwd())
        assert result.returncode == 0, f"Reconcile tests failed: {result.stdout}"
        
        # Test transport weights
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_transport_weights_enhanced.py', '-v'
        ], capture_output=True, text=True, cwd=Path.cwd())
        assert result.returncode == 0, f"Transport tests failed: {result.stdout}"
        
        # Test figures
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_week5_figures.py', '-v'
        ], capture_output=True, text=True, cwd=Path.cwd())
        assert result.returncode == 0, f"Figures tests failed: {result.stdout}"
    
    def test_week5_deliverables_present(self):
        """Test that all Week 5 deliverables are present"""
        # Core modules
        assert Path('src/16_reconcile_estimates.py').exists()
        assert Path('src/transport_weights.py').exists() 
        assert Path('src/week5_figures.py').exists()
        assert Path('src/retrain_autoencoder.py').exists()
        assert Path('src/power_analysis_sync.py').exists()
        
        # Generated figures
        assert Path('figures/selection_diagram.svg').exists()
        assert Path('figures/cost_plane.svg').exists()
        
        # Test files
        assert Path('tests/test_reconcile_estimates.py').exists()
        assert Path('tests/test_transport_weights_enhanced.py').exists()
        assert Path('tests/test_week5_figures.py').exists()
    
    def test_power_analysis_consistency(self):
        """Test power analysis consistency between YAML and blueprint"""
        # Check latest YAML file
        results_dir = Path('results')
        yaml_files = list(results_dir.glob('study_documentation_*.yaml'))
        assert yaml_files, "No study documentation YAML files found"
        
        latest_yaml = sorted(yaml_files)[-1]
        with open(latest_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Check power analysis section exists and is consistent
        assert 'power_analysis' in yaml_data
        params = yaml_data['power_analysis']['parameters']
        
        # Check relative risk is present (blueprint specifies RR 1.05)
        assert 'relative_risk' in params
        assert params['relative_risk'] == 1.05
        
        # Check that inconsistent effect_size was removed
        assert 'effect_size' not in params
        
        # Check sync timestamp exists
        assert 'last_sync' in params
        assert 'sync_source' in params
    
    def test_version_bump_ready(self):
        """Test that project is ready for v4.1.0 version bump"""
        # Check current version indicators
        makefile_path = Path('Makefile')
        if makefile_path.exists():
            content = makefile_path.read_text()
            # Should have week5 targets
            assert 'week5-validation' in content or 'Week 5' in content
    
    def test_documentation_completeness(self):
        """Test documentation is complete"""
        # Check Week 4 report has figures embedded
        week4_report = Path('docs/week4/week4_analysis_report.md')
        if week4_report.exists():
            content = week4_report.read_text()
            assert 'selection_diagram.svg' in content
            assert 'cost_plane.svg' in content
            assert 'Week 5 Additional Figures' in content


class TestCodeQuality:
    """Test code quality standards"""
    
    def test_no_syntax_errors(self):
        """Test Python syntax is valid"""
        src_files = list(Path('src').glob('*.py'))
        for file_path in src_files:
            try:
                with open(file_path, 'r') as f:
                    compile(f.read(), file_path, 'exec')
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {file_path}: {e}")
    
    def test_imports_work(self):
        """Test that key modules can be imported"""
        sys.path.insert(0, str(Path('src')))
        
        try:
            import week5_figures
            import transport_weights
            # These should import without errors
        except ImportError as e:
            pytest.fail(f"Import error: {e}")
        finally:
            sys.path.pop(0)


class TestPipelineIntegration:
    """Test pipeline integration"""
    
    def test_makefile_targets_exist(self):
        """Test that Makefile has required targets"""
        makefile_path = Path('Makefile')
        if not makefile_path.exists():
            pytest.skip("Makefile not found")
        
        content = makefile_path.read_text()
        
        # Should have basic targets
        required_targets = ['help', 'all', 'clean']
        for target in required_targets:
            assert f'{target}:' in content or f'.PHONY: {target}' in content
    
    def test_docker_compatibility(self):
        """Test Docker compatibility indicators"""
        dockerfile = Path('Dockerfile')
        if dockerfile.exists():
            content = dockerfile.read_text()
            # Should have Python environment
            assert 'python' in content.lower() or 'conda' in content.lower()


if __name__ == "__main__":
    pytest.main([__file__])