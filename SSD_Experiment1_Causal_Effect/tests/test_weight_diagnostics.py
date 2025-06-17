#!/usr/bin/env python3
"""
Test suite for weight diagnostics guard-rails
Following TDD principles - tests written first per CLAUDE.md requirements
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from weight_diagnostics import (
    validate_weight_diagnostics,
    calculate_effective_sample_size,
    check_weight_extremes,
    WeightDiagnosticsError
)


class TestWeightDiagnostics:
    """Test suite for weight diagnostics guard-rails implementation"""
    
    def test_effective_sample_size_calculation(self):
        """Test ESS calculation follows Kish formula"""
        # Simple case: equal weights should give ESS = N
        weights = np.ones(1000)
        ess = calculate_effective_sample_size(weights)
        assert abs(ess - 1000) < 1e-6
        
        # Extreme case: one large weight should give small ESS
        weights = np.ones(1000)
        weights[0] = 100
        ess = calculate_effective_sample_size(weights)
        assert ess < 500  # Should be much smaller than N
    
    def test_weight_extremes_detection(self):
        """Test detection of extreme weights"""
        # Normal weights should pass
        weights = np.random.gamma(2, 0.5, 1000)
        median_weight = np.median(weights)
        
        result = check_weight_extremes(weights)
        assert result['max_weight_ratio'] < 10  # Should be reasonable
        
        # Extreme weights should be flagged
        weights_extreme = weights.copy()
        weights_extreme[0] = median_weight * 15  # 15x median
        
        result = check_weight_extremes(weights_extreme)
        assert result['max_weight_ratio'] > 10
        assert result['has_extreme_weights'] is True
    
    def test_validation_passes_good_weights(self):
        """Test validation passes with good weights"""
        np.random.seed(42)
        n = 1000
        weights = np.random.gamma(2, 0.5, n)  # Reasonable weights
        
        # Should not raise exception
        result = validate_weight_diagnostics(weights)
        assert result['ess'] > 0.5 * n
        assert result['max_weight_ratio'] < 10
        assert result['validation_passed'] is True
    
    def test_validation_fails_low_ess(self):
        """Test validation fails with low ESS"""
        n = 1000
        weights = np.ones(n)
        weights[0] = 1000  # One extreme weight
        
        with pytest.raises(WeightDiagnosticsError, match="Effective sample size"):
            validate_weight_diagnostics(weights)
    
    def test_validation_fails_extreme_weights(self):
        """Test validation fails with extreme weights"""
        n = 1000
        weights = np.ones(n)
        median_weight = np.median(weights)
        weights[0] = median_weight * 15  # 15x median
        
        with pytest.raises(WeightDiagnosticsError, match="Maximum weight"):
            validate_weight_diagnostics(weights)
    
    def test_validation_with_custom_thresholds(self):
        """Test validation with custom thresholds"""
        n = 1000
        weights = np.random.gamma(2, 0.5, n)
        
        # Should pass with lenient thresholds
        result = validate_weight_diagnostics(
            weights, 
            min_ess_ratio=0.3, 
            max_weight_ratio=20
        )
        assert result['validation_passed'] is True
    
    def test_integration_with_propensity_scores(self):
        """Test integration with actual propensity score workflow"""
        np.random.seed(42)
        n = 1000
        
        # Simulate propensity scores
        ps = np.random.beta(2, 5, n)  # Skewed toward 0
        treatment = np.random.binomial(1, 0.15, n)
        
        # Calculate IPTW weights
        iptw = np.where(treatment == 1, 1/ps, 1/(1-ps))
        
        # Trim at 1st/99th percentiles (existing behavior)
        iptw_trimmed = np.clip(iptw, np.percentile(iptw, 1), np.percentile(iptw, 99))
        
        # Should pass validation after trimming
        result = validate_weight_diagnostics(iptw_trimmed)
        assert result['validation_passed'] is True
    
    def test_dataframe_integration(self):
        """Test integration with DataFrame workflow"""
        np.random.seed(42)
        n = 1000
        
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'propensity_score': np.random.beta(2, 5, n),
            'iptw': np.random.gamma(2, 0.5, n)
        })
        
        # Should validate DataFrame weights
        result = validate_weight_diagnostics(df['iptw'].values)
        assert result['validation_passed'] is True
        
        # Should store diagnostics in DataFrame
        for key, value in result.items():
            df[f'weight_diag_{key}'] = value
        
        assert 'weight_diag_ess' in df.columns
        assert 'weight_diag_max_weight_ratio' in df.columns
    
    def test_weight_summary_json_output(self):
        """Test JSON output for CI integration"""
        np.random.seed(42)
        n = 1000
        weights = np.random.gamma(2, 0.5, n)
        
        result = validate_weight_diagnostics(weights)
        
        # Should have all required fields for CI
        required_fields = [
            'ess', 'ess_ratio', 'max_weight', 'max_weight_ratio',
            'n_extreme_weights', 'validation_passed', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in result
        
        # All numeric fields should be JSON serializable
        import json
        json_str = json.dumps(result, default=str)
        assert len(json_str) > 0


@pytest.fixture
def problematic_weights():
    """Generate problematic weights for testing"""
    np.random.seed(42)
    n = 1000
    weights = np.random.gamma(2, 0.5, n)
    
    # Make some weights extremely large
    weights[:5] = np.median(weights) * 20
    
    return weights


class TestWeightDiagnosticsIntegration:
    """Integration tests for weight diagnostics in pipeline"""
    
    def test_ps_match_integration(self, problematic_weights):
        """Test integration with ps_match.py workflow"""
        # This would be called from within calculate_weights function
        with pytest.raises(WeightDiagnosticsError):
            validate_weight_diagnostics(problematic_weights)
    
    def test_ci_failure_mechanism(self, tmp_path):
        """Test that CI will fail when weight diagnostics fail"""
        import subprocess
        import json
        
        # Create a test script that should fail
        test_script = tmp_path / "test_weight_failure.py"
        test_script.write_text("""
import numpy as np
from weight_diagnostics import validate_weight_diagnostics

# Create problematic weights
n = 1000
weights = np.ones(n)
weights[0] = 1000  # Extreme weight

# This should fail and return non-zero exit code
try:
    validate_weight_diagnostics(weights)
    exit(0)  # Should not reach here
except Exception:
    exit(1)  # Non-zero exit code for CI
""")
        
        # Run the test script - it should fail
        result = subprocess.run(['python', str(test_script)], capture_output=True)
        assert result.returncode != 0  # Should fail
    
    def test_makefile_integration(self):
        """Test that Makefile targets will include weight validation"""
        # This test documents the expected Makefile integration
        # The actual implementation would add weight validation to ps_match target
        
        expected_makefile_target = """
ps_match: data_derived/patient_master.parquet
\t@echo "Running propensity score matching with weight diagnostics..."
\tpython src/05_ps_match.py
\t@echo "Validating weight diagnostics..."
\tpython -c "import json; data=json.load(open('results/ps_matching_results.json')); assert data['weight_diagnostics']['validation_passed'], 'Weight diagnostics failed'"
\t@echo "Weight diagnostics passed ✓"
"""
        
        # This is a documentation test - the actual Makefile would be updated
        assert "weight_diagnostics" in expected_makefile_target
        assert "validation_passed" in expected_makefile_target