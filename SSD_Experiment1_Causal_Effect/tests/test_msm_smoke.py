#!/usr/bin/env python3
"""
test_msm_smoke.py - Tests for MSM smoke test functionality

Tests that the MSM smoke test runs successfully in under 30 seconds
and produces the expected results.

Author: Ryhan Suny
Date: 2025-06-17
"""

import pytest
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import sys
import subprocess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class TestMSMSmoke:
    """Test MSM smoke test functionality"""
    
    def test_longitudinal_demo_data_creation(self):
        """Test that toy longitudinal data can be created"""
        # Run the creation script
        result = subprocess.run([
            sys.executable, "create_longitudinal_demo.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0, f"Demo data creation failed: {result.stderr}"
        
        # Check that file was created
        demo_path = Path(__file__).parent.parent / "tests/data/longitudinal_demo.parquet"
        assert demo_path.exists(), "Demo data file not created"
        
        # Check data structure
        df = pd.read_parquet(demo_path)
        assert len(df) == 6000, f"Expected 6000 rows, got {len(df)}"
        assert df['patient_id'].nunique() == 1000, f"Expected 1000 patients, got {df['patient_id'].nunique()}"
        
        required_columns = ['patient_id', 'time_period', 'ssd_flag', 'encounters', 'stress_level', 'severity']
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check data quality
        assert df['ssd_flag'].isin([0, 1]).all(), "SSD flag should be binary"
        assert (df['encounters'] >= 0).all(), "Encounters should be non-negative"
        assert df.groupby('patient_id').size().iloc[0] == 6, "Should have 6 time points per patient"
    
    def test_msm_demo_functionality(self):
        """Test that MSM demo runs successfully"""
        # Run the MSM demo
        result = subprocess.run([
            sys.executable, "test_msm_demo.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0, f"MSM demo failed: {result.stderr}"
        assert "✅ MSM demo completed successfully" in result.stdout
        
        # Check results file
        results_path = Path(__file__).parent.parent / "results/msm_demo.json"
        assert results_path.exists(), "MSM demo results file not created"
        
        with open(results_path) as f:
            results = json.load(f)
        
        # Check results structure
        assert results['status'] == 'completed', f"MSM demo failed: {results}"
        assert 'ate' in results, "ATE not found in results"
        assert 'treated_mean' in results, "Treated mean not found in results"
        assert 'control_mean' in results, "Control mean not found in results"
        
        # Check that ATE is reasonable (should show some effect)
        ate = results['ate']
        assert -10 <= ate <= 10, f"ATE {ate} seems unreasonable"
        
        # Check sample sizes
        assert results['n_treated'] > 0, "No treated patients found"
        assert results['n_control'] > 0, "No control patients found"
    
    def test_msm_smoke_test_runtime(self):
        """Test that MSM smoke test completes within time limit"""
        start_time = time.time()
        
        # Run make target
        result = subprocess.run([
            "make", "msm_smoke_test"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        runtime = time.time() - start_time
        
        # Should complete successfully
        assert result.returncode == 0, f"Make target failed: {result.stderr}"
        assert "✓ MSM smoke test completed successfully!" in result.stdout
        
        # Should complete within 30 seconds
        assert runtime < 30, f"MSM smoke test took {runtime:.1f}s, should be < 30s"
    
    def test_msm_demo_deterministic(self):
        """Test that MSM demo produces deterministic results"""
        # Run MSM demo twice
        results1 = subprocess.run([
            sys.executable, "test_msm_demo.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Read first results
        results_path = Path(__file__).parent.parent / "results/msm_demo.json"
        with open(results_path) as f:
            results_data1 = json.load(f)
        
        # Run again
        results2 = subprocess.run([
            sys.executable, "test_msm_demo.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Read second results
        with open(results_path) as f:
            results_data2 = json.load(f)
        
        # Results should be identical (due to random seed)
        assert results_data1['ate'] == results_data2['ate'], "Results should be deterministic"
        assert results_data1['treated_mean'] == results_data2['treated_mean'], "Treated mean should be identical"
        assert results_data1['control_mean'] == results_data2['control_mean'], "Control mean should be identical"
    
    def test_make_target_exists(self):
        """Test that msm_smoke_test make target exists"""
        # Check that target is defined in Makefile
        makefile_path = Path(__file__).parent.parent / "Makefile"
        with open(makefile_path) as f:
            makefile_content = f.read()
        
        assert "msm_smoke_test:" in makefile_content, "msm_smoke_test target not found in Makefile"
        assert "MSM smoke test" in makefile_content, "MSM smoke test description not found"
        
        # Test help/list targets
        result = subprocess.run([
            "make", "--dry-run", "msm_smoke_test"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Should not error (target exists)
        assert result.returncode == 0, "Make target not properly defined"


if __name__ == "__main__":
    pytest.main([__file__])