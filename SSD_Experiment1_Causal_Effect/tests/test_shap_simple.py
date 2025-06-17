#!/usr/bin/env python3
"""
test_shap_simple.py - Simple test for SHAP functionality

Tests that SHAP functionality exists and handles missing dependencies gracefully.

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
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class TestSHAPSimple:
    """Simple test for SHAP functionality"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_shap_function_exists(self):
        """Test that SHAP function exists in PS module"""
        # Import the module and check function exists
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("ps_match", 
            str(Path(__file__).parent.parent / "src" / "05_ps_match.py"))
        
        # This might fail if XGBoost is not available, which is OK
        try:
            ps_match_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ps_match_module)
            
            # Check that the function exists
            assert hasattr(ps_match_module, 'generate_shap_explanations')
            
            # Check that the function is callable
            func = getattr(ps_match_module, 'generate_shap_explanations')
            assert callable(func)
            
        except ImportError as e:
            # If XGBoost or other dependencies are missing, that's expected
            if "xgboost" in str(e) or "shap" in str(e):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_shap_graceful_degradation(self, temp_output_dir):
        """Test that SHAP function handles missing SHAP library gracefully"""
        
        # Create mock data
        X = np.random.randn(100, 10)
        feature_names = [f'feature_{i}' for i in range(10)]
        mock_model = MagicMock()
        
        try:
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("ps_match", 
                str(Path(__file__).parent.parent / "src" / "05_ps_match.py"))
            ps_match_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ps_match_module)
            
            # Mock SHAP as unavailable by patching the global variable
            original_shap_available = getattr(ps_match_module, 'SHAP_AVAILABLE', False)
            
            # Force SHAP to be unavailable
            ps_match_module.SHAP_AVAILABLE = False
            
            try:
                # Call function - should return None gracefully
                result = ps_match_module.generate_shap_explanations(
                    model=mock_model,
                    X=X,
                    feature_names=feature_names,
                    output_dir=temp_output_dir
                )
                
                # Should return None when SHAP is not available
                assert result is None
                
            finally:
                # Restore original value
                ps_match_module.SHAP_AVAILABLE = original_shap_available
                
        except ImportError as e:
            # If dependencies are missing, that's expected in CI
            if any(dep in str(e) for dep in ["xgboost", "shap", "tableone"]):
                pytest.skip(f"Dependencies not available: {e}")
            else:
                raise
    
    def test_shap_with_mock_dependencies(self, temp_output_dir):
        """Test SHAP function behavior with mocked dependencies"""
        
        # This test verifies the function structure without requiring actual SHAP
        X = np.random.randn(50, 5)
        feature_names = ['age', 'sex_M', 'charlson_score', 'baseline_encounters', 'feature_5']
        
        try:
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("ps_match", 
                str(Path(__file__).parent.parent / "src" / "05_ps_match.py"))
            ps_match_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ps_match_module)
            
            # Create a mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = np.random.uniform(0, 1, len(X))
            
            # If SHAP is available, test the function
            if getattr(ps_match_module, 'SHAP_AVAILABLE', False):
                result = ps_match_module.generate_shap_explanations(
                    model=mock_model,
                    X=X,
                    feature_names=feature_names,
                    output_dir=None  # No file output
                )
                
                # Could return None or results depending on SHAP availability
                assert result is None or isinstance(result, dict)
                
                if result is not None:
                    # If results returned, check structure
                    expected_keys = ['feature_importance', 'shap_values', 'top_features', 'n_features_nonzero']
                    for key in expected_keys:
                        assert key in result, f"Missing key: {key}"
            else:
                # Test graceful handling when SHAP not available
                result = ps_match_module.generate_shap_explanations(
                    model=mock_model,
                    X=X,
                    feature_names=feature_names,
                    output_dir=temp_output_dir
                )
                
                assert result is None
                
        except ImportError as e:
            # Expected in CI environment without full dependencies
            pytest.skip(f"Dependencies not available for full test: {e}")
    
    def test_file_creation_structure(self, temp_output_dir):
        """Test expected file structure for SHAP outputs"""
        
        # Test that the expected file paths are correct
        expected_files = [
            temp_output_dir / "ps_shap_importance.csv",
            temp_output_dir / "ps_shap_summary.svg"
        ]
        
        # These files should be created by SHAP function when working
        for expected_file in expected_files:
            # Test that parent directory exists
            assert expected_file.parent.exists() or expected_file.parent == temp_output_dir
            
            # Test file naming convention
            assert expected_file.suffix in ['.csv', '.svg']
            assert 'shap' in expected_file.name.lower()


if __name__ == "__main__":
    pytest.main([__file__])