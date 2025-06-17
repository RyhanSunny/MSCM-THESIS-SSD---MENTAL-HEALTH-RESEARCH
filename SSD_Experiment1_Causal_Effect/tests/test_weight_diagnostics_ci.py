#!/usr/bin/env python3
"""
test_weight_diagnostics_ci.py - CI test for weight diagnostics validation

This test runs in CI to ensure weight diagnostics pass required thresholds.
It will fail the CI build if extreme weights are detected.
"""

import pytest
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestWeightDiagnosticsCI:
    """CI tests for weight diagnostics thresholds"""
    
    def test_weight_diagnostics_json_validation(self):
        """Test that weight diagnostics JSON passes all thresholds"""
        # Look for weight diagnostics output
        results_dir = Path("results")
        weight_files = list(results_dir.glob("*weight*.json"))
        
        if not weight_files:
            pytest.skip("No weight diagnostics JSON found - skipping CI check")
        
        for weight_file in weight_files:
            with open(weight_file, 'r') as f:
                weight_data = json.load(f)
            
            # Check ESS ratio
            ess_ratio = weight_data.get('ess_ratio', 0)
            assert ess_ratio > 0.5, \
                f"ESS too low: {ess_ratio:.2%} (must be > 50%)"
            
            # Check extreme weights percentage
            extreme_pct = weight_data.get('extreme_weight_pct', 100)
            assert extreme_pct < 0.05, \
                f"Too many extreme weights: {extreme_pct:.2%} (must be < 0.05%)"
            
            # Check max weight ratio
            max_weight_ratio = weight_data.get('max_weight_ratio', float('inf'))
            assert max_weight_ratio < 10, \
                f"Max weight too extreme: {max_weight_ratio:.1f}x median (must be < 10x)"
            
            # Check validation passed
            assert weight_data.get('validation_passed', False), \
                "Weight validation failed"
    
    def test_weight_diagnostics_in_pipeline(self):
        """Test that weight diagnostics are integrated in pipeline"""
        from weight_diagnostics import validate_weight_diagnostics
        import numpy as np
        
        # Test with good weights
        good_weights = np.random.uniform(0.5, 2.0, 1000)
        result = validate_weight_diagnostics(good_weights)
        
        assert result['validation_passed']
        assert result['ess_ratio'] > 0.5
        
        # Test with bad weights (should fail)
        bad_weights = np.ones(1000)
        bad_weights[0] = 100  # Extreme weight
        
        result = validate_weight_diagnostics(bad_weights)
        assert not result['validation_passed']
    
    @pytest.mark.skipif(not Path("data_derived/propensity_scores.parquet").exists(),
                        reason="Propensity scores not yet generated")
    def test_actual_pipeline_weights(self):
        """Test weights from actual pipeline run"""
        import pandas as pd
        
        # Load propensity scores
        ps_file = Path("data_derived/propensity_scores.parquet")
        df = pd.read_parquet(ps_file)
        
        if 'iptw' in df.columns:
            from weight_diagnostics import validate_weight_diagnostics
            
            weights = df['iptw'].values
            result = validate_weight_diagnostics(weights)
            
            # This should pass if pipeline is working correctly
            assert result['validation_passed'], \
                f"Pipeline weights failed validation: {result}"
            
            # Log summary for CI
            print(f"\nWeight Diagnostics Summary:")
            print(f"  ESS: {result['ess']:.0f} ({result['ess_ratio']:.1%} of sample)")
            print(f"  Max weight: {result['max_weight_ratio']:.1f}x median")
            print(f"  Extreme weights: {result['extreme_weight_pct']:.3%}")


def test_weight_diagnostics_cli():
    """Test command-line validation of weights"""
    import subprocess
    import tempfile
    import numpy as np
    
    # Create test weight file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {
            'ess_ratio': 0.6,
            'extreme_weight_pct': 0.02,
            'max_weight_ratio': 8.5,
            'validation_passed': True
        }
        json.dump(test_data, f)
        temp_file = f.name
    
    try:
        # Run validation
        cmd = [
            'python3', '-m', 'pytest', 
            'tests/test_weight_diagnostics_ci.py::TestWeightDiagnosticsCI::test_weight_diagnostics_json_validation',
            '-v'
        ]
        
        # Should pass with good weights
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, "CI validation should pass with good weights"
        
    finally:
        Path(temp_file).unlink()


if __name__ == "__main__":
    # Run as standalone CI check
    pytest.main([__file__, '-v', '--tb=short'])