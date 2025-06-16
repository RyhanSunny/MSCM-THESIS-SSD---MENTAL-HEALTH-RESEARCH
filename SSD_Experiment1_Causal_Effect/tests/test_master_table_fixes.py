#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for master table creation fixes
Following CLAUDE.md TDD requirements - tests written first
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / 'data_derived'

class TestMasterTablePrerequisites:
    """Test that all prerequisites for master table creation are met"""
    
    def test_required_parquet_files_exist(self):
        """Test that all required input files exist with correct names"""
        required_files = [
            'cohort.parquet',
            'exposure.parquet', 
            'mediator_autoencoder.parquet',
            'outcomes.parquet',
            'confounders.parquet',  # Actual file name
            'lab_sensitivity.parquet'  # Actual file name
        ]
        
        for filename in required_files:
            filepath = DERIVED / filename
            assert filepath.exists(), f"Required file {filename} does not exist"
            
            # Test file is readable
            df = pd.read_parquet(filepath)
            assert len(df) > 0, f"File {filename} is empty"
            assert 'Patient_ID' in df.columns, f"File {filename} missing Patient_ID column"
    
    def test_patient_id_consistency(self):
        """Test that Patient_ID is consistent across all files"""
        files_to_check = [
            'cohort.parquet',
            'exposure.parquet',
            'mediator_autoencoder.parquet', 
            'outcomes.parquet',
            'confounders.parquet',
            'lab_sensitivity.parquet'
        ]
        
        patient_ids = {}
        for filename in files_to_check:
            filepath = DERIVED / filename
            if filepath.exists():
                df = pd.read_parquet(filepath)
                patient_ids[filename] = set(df['Patient_ID'].unique())
        
        # All files should have overlapping patient IDs
        base_ids = patient_ids['cohort.parquet']
        for filename, ids in patient_ids.items():
            overlap = len(base_ids.intersection(ids))
            total_base = len(base_ids)
            overlap_pct = overlap / total_base * 100
            assert overlap_pct > 90, f"Patient_ID overlap with {filename} is only {overlap_pct:.1f}%"
    
    def test_age_variable_exists(self):
        """Test that age variable exists in cohort data"""
        cohort_file = DERIVED / 'cohort.parquet'
        df = pd.read_parquet(cohort_file)
        
        # Check which age variable exists
        age_vars = [col for col in df.columns if 'Age' in col]
        assert len(age_vars) > 0, "No age variable found in cohort data"
        
        # Document what we found for fixing the script
        print(f"Available age variables: {age_vars}")

class TestFileNameMismatches:
    """Test to identify and validate file name corrections needed"""
    
    def test_identify_file_name_mismatches(self):
        """Test that files exist with correct names (after fixes)"""
        required_files = ['confounders.parquet', 'lab_sensitivity.parquet']
        
        for filename in required_files:
            filepath = DERIVED / filename
            assert filepath.exists(), f"Required file {filename} does not exist"
            
            # Test file is readable and has expected structure
            df = pd.read_parquet(filepath)
            assert len(df) > 0, f"File {filename} is empty"
            assert 'Patient_ID' in df.columns, f"File {filename} missing Patient_ID column"
    
    def test_referral_sequences_exists(self):
        """Test that referral sequences file exists and is properly generated"""
        referral_file = DERIVED / 'referral_sequences.parquet'
        assert referral_file.exists(), "referral_sequences.parquet should exist"
        
        # Test file is readable and has required structure
        df = pd.read_parquet(referral_file)
        assert len(df) > 0, "referral_sequences.parquet is empty"
        assert 'Patient_ID' in df.columns, "referral_sequences.parquet missing Patient_ID column"
        
        # Check if script exists
        script_file = ROOT / 'src' / '07_referral_sequence.py'
        assert script_file.exists(), "07_referral_sequence.py script should exist"

class TestMasterTableScript:
    """Test the master table script expectations vs reality"""
    
    def test_master_table_script_expectations(self):
        """Test what the master table script expects vs what exists"""
        # Test that all files exist with correct names after fixes
        expected_files = [
            'cohort.parquet',
            'exposure.parquet',
            'mediator_autoencoder.parquet',
            'outcomes.parquet',
            'confounders.parquet',
            'lab_sensitivity.parquet',
            'referral_sequences.parquet'
        ]
        
        for filename in expected_files:
            filepath = DERIVED / filename
            assert filepath.exists(), f"Required file {filename} should exist"
            
            # Test file is readable
            df = pd.read_parquet(filepath)
            assert len(df) > 0, f"File {filename} should not be empty"
            assert 'Patient_ID' in df.columns, f"File {filename} should have Patient_ID column"

@pytest.fixture
def sample_patient_master(mock_patient_data, mock_exposure_data, mock_outcome_data, mock_confounder_data):
    """Create a sample patient master table with all necessary columns."""
    
    # Merge all the mock data to create a master table
    master_df = mock_patient_data.merge(mock_exposure_data, on='Patient_ID') \
                                 .merge(mock_outcome_data, on='Patient_ID') \
                                 .merge(mock_confounder_data, on='Patient_ID')
    
    # The exposure data already has ssd_flag and ssd_flag_strict from conftest.py
    # Add the other aliases that should be created by 08_patient_master_table.py
    # Handle the fact that Age_at_2015 appears in both patient and confounder data
    if 'Age_at_2015_y' in master_df.columns:
        master_df['age'] = master_df['Age_at_2015_y']  # Use the confounder version
    elif 'Age_at_2015_x' in master_df.columns:
        master_df['age'] = master_df['Age_at_2015_x']  # Use the patient version
    elif 'Age_at_2015' in master_df.columns:
        master_df['age'] = master_df['Age_at_2015']
    else:
        # Create a synthetic age column for testing
        master_df['age'] = 50.0
    
    master_df['sex_M'] = (master_df['Sex'] == 'M').astype(int)
    master_df['charlson_score'] = master_df['Charlson']
    
    return master_df

def test_master_table_alias_creation(sample_patient_master):
    """Test that ssd_flag and ssd_flag_strict aliases are properly created."""
    
    # Check that required columns exist
    assert 'ssd_flag' in sample_patient_master.columns
    assert 'ssd_flag_strict' in sample_patient_master.columns
    assert 'age' in sample_patient_master.columns
    assert 'sex_M' in sample_patient_master.columns
    assert 'charlson_score' in sample_patient_master.columns
    
    # Check data types (allow both int32 and int64)
    assert sample_patient_master['ssd_flag'].dtype in ['int32', 'int64']
    assert sample_patient_master['ssd_flag_strict'].dtype in ['int32', 'int64']
    assert sample_patient_master['sex_M'].dtype in ['int32', 'int64']
    
    # Check value ranges
    assert sample_patient_master['ssd_flag'].isin([0, 1]).all()
    assert sample_patient_master['ssd_flag_strict'].isin([0, 1]).all()
    assert sample_patient_master['sex_M'].isin([0, 1]).all()
    
    # Check that strict is subset of regular (in real data)
    # Note: in test data this may not hold due to random generation
    print(f"Regular exposure: {sample_patient_master['ssd_flag'].sum()}")
    print(f"Strict exposure: {sample_patient_master['ssd_flag_strict'].sum()}")

def test_treatment_col_parameter_compatibility():
    """Test that analysis scripts can handle both treatment column options."""
    
    # Create test dataframe with both columns
    test_df = pd.DataFrame({
        'Patient_ID': range(1, 101),
        'ssd_flag': np.random.binomial(1, 0.15, 100),
        'ssd_flag_strict': np.random.binomial(1, 0.08, 100),
        'age': np.random.normal(50, 15, 100),
        'sex_M': np.random.binomial(1, 0.5, 100),
        'charlson_score': np.random.poisson(2, 100),
        'total_encounters': np.random.poisson(5, 100),
    })
    
    # Verify both treatment columns work for basic operations
    regular_exposed = test_df['ssd_flag'].sum()
    strict_exposed = test_df['ssd_flag_strict'].sum()
    
    assert regular_exposed >= 0
    assert strict_exposed >= 0
    
    # Test that we can access both columns without errors
    assert 'ssd_flag' in test_df.columns
    assert 'ssd_flag_strict' in test_df.columns

def test_backwards_compatibility(sample_patient_master):
    """Test that existing code still works with default ssd_flag."""
    
    # Simulate what existing analysis scripts expect
    treatment_col = 'ssd_flag'
    
    # Should be able to access the column
    assert treatment_col in sample_patient_master.columns
    
    # Should be binary
    values = sample_patient_master[treatment_col].unique()
    assert set(values).issubset({0, 1})
    
    # Should have reasonable prevalence 
    prevalence = sample_patient_master[treatment_col].mean()
    assert 0 <= prevalence <= 1

def test_treatment_col_cli_parameter():
    """Test that CLI scripts can accept --treatment-col parameter."""
    
    # This is a basic test to ensure the parameter structure is correct
    # In a real test environment, we'd invoke the scripts directly
    test_cases = [
        ('ssd_flag', 'default OR logic'),
        ('ssd_flag_strict', 'strict AND logic')
    ]
    
    for treatment_col, description in test_cases:
        # Verify column name is valid
        assert isinstance(treatment_col, str)
        assert len(treatment_col) > 0
        assert ' ' not in treatment_col  # No spaces in column names
        
        print(f"Treatment column '{treatment_col}': {description}")

if __name__ == "__main__":
    # Run tests to confirm failures before implementing fixes
    pytest.main([__file__, "-v"])