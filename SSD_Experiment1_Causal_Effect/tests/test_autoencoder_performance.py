#!/usr/bin/env python3
"""
test_autoencoder_performance.py - Tests for autoencoder performance requirements

Tests that the autoencoder can achieve AUROC ≥ 0.70 as required by Week 6 deliverables.

Author: Ryhan Suny
Date: 2025-06-17
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.simple_autoencoder_retrain import run_autoencoder_simulation, load_and_prepare_data

class TestAutoencoderPerformance:
    """Test autoencoder performance requirements"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_autoencoder_achieves_target_auroc_synthetic(self, temp_output_dir):
        """Test that autoencoder achieves target AUROC with synthetic data"""
        # Create synthetic data with clear signal
        input_path = Path("nonexistent.parquet")  # Force synthetic data
        
        result = run_autoencoder_simulation(
            input_path=input_path,
            output_dir=temp_output_dir,
            target_auroc=0.7,
            max_trials=3
        )
        
        # Check that target was achieved
        assert result['success'], f"Failed to achieve target AUROC. Got {result['best_auroc']:.3f}"
        assert result['best_auroc'] >= 0.7, f"AUROC {result['best_auroc']:.3f} below target 0.7"
        
        # Check output files exist
        assert (temp_output_dir / 'autoencoder_config.json').exists()
        assert (temp_output_dir / 'autoencoder_performance_report.md').exists()
        
        # Check config file content
        with open(temp_output_dir / 'autoencoder_config.json') as f:
            config = json.load(f)
        
        assert config['target_achieved'] is True
        assert config['best_auroc'] >= 0.7
    
    def test_autoencoder_achieves_target_auroc_real_data(self, temp_output_dir):
        """Test that autoencoder achieves target AUROC with real cohort data"""
        input_path = Path("data_derived/patient_master.parquet")
        
        if not input_path.exists():
            pytest.skip("Real data not available, skipping real data test")
        
        result = run_autoencoder_simulation(
            input_path=input_path,
            output_dir=temp_output_dir,
            target_auroc=0.7,
            max_trials=3
        )
        
        # Check that target was achieved
        assert result['success'], f"Failed to achieve target AUROC with real data. Got {result['best_auroc']:.3f}"
        assert result['best_auroc'] >= 0.7, f"AUROC {result['best_auroc']:.3f} below target 0.7"
        
        # Test AUROC should be realistic (not suspiciously perfect)
        assert result['best_auroc'] <= 0.95, f"AUROC {result['best_auroc']:.3f} suspiciously high"
        
        # Test and training performance should be similar
        if 'test_auroc' in result:
            auroc_diff = abs(result['best_auroc'] - result['test_auroc'])
            assert auroc_diff <= 0.1, f"Large gap between train/test AUROC: {auroc_diff:.3f}"
    
    def test_data_loading_and_preparation(self):
        """Test data loading handles both real and synthetic data"""
        # Test with nonexistent file (should create synthetic)
        X_synth, y_synth = load_and_prepare_data(Path("nonexistent.parquet"))
        
        assert X_synth.shape[0] > 0, "No synthetic samples generated"
        assert X_synth.shape[1] >= 10, f"Too few features: {X_synth.shape[1]}"
        assert len(y_synth) == X_synth.shape[0], "Target length mismatch"
        assert set(np.unique(y_synth)) == {0, 1}, f"Invalid target values: {np.unique(y_synth)}"
        
        # Test with real data if available
        real_data_path = Path("data_derived/patient_master.parquet")
        if real_data_path.exists():
            X_real, y_real = load_and_prepare_data(real_data_path)
            
            assert X_real.shape[0] > 0, "No real samples loaded"
            assert X_real.shape[1] >= 10, f"Too few real features: {X_real.shape[1]}"
            assert len(y_real) == X_real.shape[0], "Real target length mismatch"
            assert set(np.unique(y_real)) == {0, 1}, f"Invalid real target values: {np.unique(y_real)}"
    
    def test_performance_graceful_degradation(self, temp_output_dir):
        """Test that system handles when target is not achieved"""
        # Set an impossibly high target
        input_path = Path("nonexistent.parquet")
        
        result = run_autoencoder_simulation(
            input_path=input_path,
            output_dir=temp_output_dir,
            target_auroc=0.99,  # Impossibly high
            max_trials=2
        )
        
        # Should not crash, but may not achieve target
        assert 'best_auroc' in result
        assert 'success' in result
        assert result['best_auroc'] >= 0.5, "AUROC should be better than random"
        
        # Output files should still be created
        assert (temp_output_dir / 'autoencoder_config.json').exists()
        assert (temp_output_dir / 'autoencoder_performance_report.md').exists()
    
    def test_ci_runtime_performance(self, temp_output_dir):
        """Test that autoencoder runs quickly enough for CI"""
        import time
        
        input_path = Path("nonexistent.parquet")  # Use synthetic for speed
        
        start_time = time.time()
        result = run_autoencoder_simulation(
            input_path=input_path,
            output_dir=temp_output_dir,
            target_auroc=0.7,
            max_trials=2  # Limited trials for CI speed
        )
        runtime = time.time() - start_time
        
        # Should complete within reasonable time for CI
        assert runtime < 30, f"Runtime {runtime:.1f}s too long for CI"
        
        # Should still achieve target despite limited trials
        assert result['best_auroc'] >= 0.7, f"AUROC {result['best_auroc']:.3f} below target despite CI constraints"


if __name__ == "__main__":
    pytest.main([__file__])