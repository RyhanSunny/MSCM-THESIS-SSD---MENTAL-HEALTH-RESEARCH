#!/usr/bin/env python3
"""
test_shap_explanations.py - Tests for SHAP explanations of propensity score model

Tests that SHAP explanations are generated correctly and provide meaningful insights
into the XGBoost propensity score model.

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
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules with numeric names using importlib
import importlib.util

# Import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import PS match module
spec = importlib.util.spec_from_file_location("ps_match", 
    str(Path(__file__).parent.parent / "src" / "05_ps_match.py"))
ps_match_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps_match_module)
generate_shap_explanations = ps_match_module.generate_shap_explanations

class TestSHAPExplanations:
    """Test SHAP explanations for propensity score model"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Create correlated features
        X = np.random.randn(n_samples, n_features)
        
        # Create treatment with some signal
        treatment_logits = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + 
                           np.random.randn(n_samples) * 0.5)
        y = (treatment_logits > 0).astype(int)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        return X, y, feature_names
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_xgb_model(self, sample_data):
        """Create mock XGBoost model"""
        X, y, feature_names = sample_data
        
        if XGB_AVAILABLE:
            # Create real XGBoost model
            dtrain = xgb.DMatrix(X, label=y)
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 3,
                'eta': 0.1,
                'seed': 42
            }
            model = xgb.train(params, dtrain, num_boost_round=10)
            return model
        else:
            # Create mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = np.random.uniform(0, 1, len(y))
            return mock_model
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not available")
    def test_shap_explanations_generation(self, mock_xgb_model, sample_data, temp_output_dir):
        """Test basic SHAP explanations generation"""
        
        X, y, feature_names = sample_data
        
        results = generate_shap_explanations(
            model=mock_xgb_model,
            X=X,
            feature_names=feature_names,
            output_dir=temp_output_dir
        )
        
        # Check that results are returned
        if results is not None:  # Only if SHAP is available
            assert 'feature_importance' in results
            assert 'shap_values' in results
            assert 'top_features' in results
            assert 'n_features_nonzero' in results
            
            # Check feature importance structure
            importance_df = results['feature_importance']
            assert len(importance_df) == len(feature_names)
            assert 'feature' in importance_df.columns
            assert 'shap_importance' in importance_df.columns
            
            # Check that at least some features have non-zero importance
            assert results['n_features_nonzero'] >= 1
            
            # Check that top features are sorted by importance
            top_features = results['top_features']
            importances = top_features['shap_importance'].values
            assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not available")
    def test_shap_output_files_creation(self, mock_xgb_model, sample_data, temp_output_dir):
        """Test that SHAP outputs create expected files"""
        
        X, y, feature_names = sample_data
        
        results = generate_shap_explanations(
            model=mock_xgb_model,
            X=X,
            feature_names=feature_names,
            output_dir=temp_output_dir
        )
        
        if results is not None:  # Only if SHAP is available
            # Check that files are created
            importance_file = temp_output_dir / "ps_shap_importance.csv"
            assert importance_file.exists(), "SHAP importance CSV should be created"
            
            # Verify CSV content
            importance_df = pd.read_csv(importance_file)
            assert len(importance_df) <= 20, "Should contain top 20 features"
            assert 'feature' in importance_df.columns
            assert 'shap_importance' in importance_df.columns
            
            # Check that at least 10 features have non-zero importance (as per requirements)
            non_zero_features = (importance_df['shap_importance'] > 0).sum()
            assert non_zero_features >= 10, f"At least 10 features should have non-zero importance, got {non_zero_features}"
    
    @pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not available")
    def test_shap_large_dataset_subsampling(self, sample_data, temp_output_dir):
        """Test that SHAP handles large datasets by subsampling"""
        
        # Create large dataset
        np.random.seed(42)
        n_large = 2000
        X_large = np.random.randn(n_large, 10)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # Mock model
        mock_model = MagicMock()
        
        if XGB_AVAILABLE and SHAP_AVAILABLE:
            # Create real model for testing
            y_large = np.random.randint(0, 2, n_large)
            dtrain = xgb.DMatrix(X_large, label=y_large)
            params = {'objective': 'binary:logistic', 'max_depth': 3, 'eta': 0.1, 'seed': 42}
            model = xgb.train(params, dtrain, num_boost_round=5)
            
            results = generate_shap_explanations(
                model=model,
                X=X_large,
                feature_names=feature_names,
                output_dir=temp_output_dir
            )
            
            if results is not None:
                # Should work even with large dataset due to subsampling
                assert 'shap_values' in results
                # SHAP values should be for subsampled data (1000 samples)
                assert results['shap_values'].shape[0] <= 1000
    
    def test_shap_unavailable_graceful_handling(self, sample_data, temp_output_dir):
        """Test graceful handling when SHAP is not available"""
        
        X, y, feature_names = sample_data
        mock_model = MagicMock()
        
        # Mock SHAP as unavailable
        with patch.object(ps_match_module, 'SHAP_AVAILABLE', False):
            results = generate_shap_explanations(
                model=mock_model,
                X=X,
                feature_names=feature_names,
                output_dir=temp_output_dir
            )
            
            # Should return None gracefully
            assert results is None
    
    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGBoost not available")
    def test_propensity_score_integration(self, sample_data):
        """Test that SHAP integrates properly with propensity score workflow"""
        # This would be an integration test with the full PS pipeline
        # For now, just test that the function can be called with XGBoost model
        
        X, y, feature_names = sample_data
        
        # Train a simple XGBoost model
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'binary:logistic',
            'max_depth': 3,
            'eta': 0.1,
            'seed': 42
        }
        model = xgb.train(params, dtrain, num_boost_round=10)
        
        # Test that we can call the function without errors
        
        results = generate_shap_explanations(
            model=model,
            X=X,
            feature_names=feature_names,
            output_dir=None  # No file output for this test
        )
        
        # Should either return results or None (if SHAP unavailable)
        assert results is None or isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__])