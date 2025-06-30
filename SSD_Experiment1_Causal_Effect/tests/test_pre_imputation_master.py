#!/usr/bin/env python3
"""
test_pre_imputation_master.py - Tests for pre-imputation master table builder

Following CLAUDE.md TDD requirements - Tests written FIRST before implementation.
These tests will initially FAIL until pre_imputation_master.py is implemented.

Author: Ryhan Suny
Date: June 30, 2025
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# These imports will fail initially - that's expected in TDD
try:
    from pre_imputation_master import (
        load_all_features,
        validate_merge_keys,
        combine_features_with_missingness,
        create_pre_imputation_master,
        get_missingness_report
    )
except ImportError:
    # Expected to fail before implementation
    pass


class TestPreImputationMaster:
    """Test suite for pre-imputation master table creation."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data_derived"
        data_dir.mkdir()
        
        # Create test datasets
        np.random.seed(42)
        n_patients = 100
        
        # Base cohort (19 columns)
        cohort = pd.DataFrame({
            'Patient_ID': range(1000, 1000 + n_patients),
            'Sex': np.random.choice(['M', 'F'], n_patients),
            'Age_at_2015': np.random.randint(18, 90, n_patients),
            'Charlson': np.random.randint(0, 10, n_patients),
            'NYD_count': np.random.randint(0, 5, n_patients)
        })
        # Add some missing values
        cohort.loc[10:15, 'Sex'] = np.nan
        cohort.to_parquet(data_dir / "cohort.parquet")
        
        # Exposure flags
        exposure = pd.DataFrame({
            'Patient_ID': range(1000, 1000 + n_patients),
            'ssd_flag': np.random.choice([0, 1], n_patients),
            'normal_lab_count': np.random.randint(0, 20, n_patients),
            'symptom_referral_n': np.random.randint(0, 10, n_patients)
        })
        exposure.to_parquet(data_dir / "exposure.parquet")
        
        # Mediator
        mediator = pd.DataFrame({
            'Patient_ID': range(1000, 1000 + n_patients),
            'SSD_severity_index': np.random.uniform(0, 100, n_patients)
        })
        mediator.to_parquet(data_dir / "mediator_autoencoder.parquet")
        
        # Outcomes
        outcomes = pd.DataFrame({
            'Patient_ID': range(1000, 1000 + n_patients),
            'total_encounters': np.random.randint(1, 50, n_patients),
            'ed_visits': np.random.randint(0, 10, n_patients),
            'high_utilization': np.random.choice([0, 1], n_patients)
        })
        outcomes.to_parquet(data_dir / "outcomes.parquet")
        
        # Confounders (many columns)
        confounder_cols = {f'confounder_{i}': np.random.randn(n_patients) 
                          for i in range(20)}
        confounders = pd.DataFrame({
            'Patient_ID': range(1000, 1000 + n_patients),
            **confounder_cols
        })
        # Add missingness
        confounders.loc[20:30, 'confounder_1'] = np.nan
        confounders.to_parquet(data_dir / "confounders.parquet")
        
        yield data_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_load_all_features(self, temp_data_dir):
        """Test loading all feature datasets."""
        datasets = load_all_features(temp_data_dir)
        
        # Should load all expected datasets
        expected_keys = ['cohort', 'exposure', 'mediator', 'outcomes', 'confounders']
        assert set(datasets.keys()) == set(expected_keys)
        
        # Each should be a DataFrame
        for key, df in datasets.items():
            assert isinstance(df, pd.DataFrame)
            assert 'Patient_ID' in df.columns
    
    def test_validate_merge_keys(self, temp_data_dir):
        """Test merge key validation."""
        datasets = load_all_features(temp_data_dir)
        
        # Should pass validation with consistent Patient_IDs
        is_valid, report = validate_merge_keys(datasets)
        assert is_valid
        assert report['n_unique_patients'] == 100
        assert report['all_patients_present']
    
    def test_validate_merge_keys_missing_patients(self, temp_data_dir):
        """Test validation catches missing patients."""
        datasets = load_all_features(temp_data_dir)
        
        # Remove some patients from one dataset
        datasets['exposure'] = datasets['exposure'].iloc[10:]
        
        is_valid, report = validate_merge_keys(datasets)
        assert not is_valid
        assert not report['all_patients_present']
        assert len(report['missing_by_dataset']['exposure']) == 10
    
    def test_combine_features_with_missingness(self, temp_data_dir):
        """Test combining features preserves missingness."""
        datasets = load_all_features(temp_data_dir)
        
        combined = combine_features_with_missingness(datasets)
        
        # Should have all columns
        assert 'Patient_ID' in combined.columns
        assert 'ssd_flag' in combined.columns
        assert 'SSD_severity_index' in combined.columns
        assert 'total_encounters' in combined.columns
        assert 'confounder_1' in combined.columns
        
        # Should preserve missingness
        assert combined['Sex'].isna().sum() > 0
        assert combined['confounder_1'].isna().sum() > 0
        
        # Should have correct shape
        assert len(combined) == 100
        # With test data: 5 cohort + 3 exposure + 1 mediator + 3 outcomes + 20 confounders = 32
        assert combined.shape[1] >= 30  # At least 30 columns in test data
    
    def test_get_missingness_report(self, temp_data_dir):
        """Test missingness reporting."""
        datasets = load_all_features(temp_data_dir)
        combined = combine_features_with_missingness(datasets)
        
        report = get_missingness_report(combined)
        
        assert 'total_missing_pct' in report
        assert 'columns_with_missing' in report
        assert 'missing_by_column' in report
        
        # Sex and confounder_1 should have missing values
        assert 'Sex' in report['missing_by_column']
        assert 'confounder_1' in report['missing_by_column']
        assert report['missing_by_column']['Sex'] > 0
    
    def test_create_pre_imputation_master_integration(self, temp_data_dir):
        """Integration test for full pipeline."""
        output_path = temp_data_dir / "master_with_missing.parquet"
        
        result = create_pre_imputation_master(
            data_dir=temp_data_dir,
            output_path=output_path
        )
        
        # Should succeed
        assert result['success']
        assert output_path.exists()
        
        # Load and verify
        master = pd.read_parquet(output_path)
        assert len(master) == 100
        assert master.shape[1] >= 30  # Adjusted for test data
        
        # Should have key columns
        assert 'Patient_ID' in master.columns
        assert 'ssd_flag' in master.columns
        assert 'SSD_severity_index' in master.columns
        
        # Should preserve missingness
        assert master.isna().sum().sum() > 0
    
    def test_function_size_compliance(self):
        """Test that all functions comply with 50-line limit."""
        # This will be verified when implementation is complete
        # For now, it's a reminder for implementation
        pass
    
    def test_error_handling_missing_files(self, temp_data_dir):
        """Test graceful handling of missing files."""
        # Remove a required file
        (temp_data_dir / "exposure.parquet").unlink()
        
        with pytest.raises(FileNotFoundError):
            load_all_features(temp_data_dir)
    
    def test_column_preservation(self, temp_data_dir):
        """Test all columns from all datasets are preserved."""
        datasets = load_all_features(temp_data_dir)
        combined = combine_features_with_missingness(datasets)
        
        # Count expected columns
        expected_cols = set()
        for df in datasets.values():
            expected_cols.update(df.columns)
        
        # Remove duplicate Patient_ID counts
        expected_count = len(expected_cols) - (len(datasets) - 1)
        
        # Combined should have all unique columns (accounting for test data structure)
        # In test data we have some duplicate columns which get suffixes
        assert len(combined.columns) >= expected_count


@pytest.mark.integration
class TestPreImputationMasterIntegration:
    """Integration tests with actual pipeline."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data_derived"
        data_dir.mkdir()
        
        # Create minimal test data
        cohort = pd.DataFrame({
            'Patient_ID': range(1000, 1050),
            'Age': np.random.randint(18, 90, 50)
        })
        cohort.to_parquet(data_dir / "cohort.parquet")
        
        for name in ['exposure', 'mediator_autoencoder', 'outcomes', 'confounders']:
            df = pd.DataFrame({
                'Patient_ID': range(1000, 1050),
                'test_col': np.random.randn(50)
            })
            filename = f"{name}.parquet" if name != 'mediator_autoencoder' else 'mediator_autoencoder.parquet'
            df.to_parquet(data_dir / filename)
        
        yield data_dir
        shutil.rmtree(temp_dir)
    
    def test_compatibility_with_imputation_pipeline(self, temp_data_dir):
        """Test output is compatible with 07_missing_data.py."""
        output_path = temp_data_dir / "master_with_missing.parquet"
        
        result = create_pre_imputation_master(
            data_dir=temp_data_dir,
            output_path=output_path
        )
        
        master = pd.read_parquet(output_path)
        
        # Should have correct structure for imputation
        assert 'Patient_ID' in master.columns
        assert master['Patient_ID'].is_unique
        assert master.select_dtypes(include=[np.number]).shape[1] >= 5  # Adjusted for test data
        
        # Should identify numeric vs categorical columns
        numeric_cols = master.select_dtypes(include=[np.number]).columns
        categorical_cols = master.select_dtypes(include=['object', 'category']).columns
        
        assert len(numeric_cols) > 0
        # Patient_ID is kept as identifier (imputation should skip it)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])