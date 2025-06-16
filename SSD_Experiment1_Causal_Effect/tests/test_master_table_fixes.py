#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for master table creation fixes
Following CLAUDE.md TDD requirements - tests written first
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
SRC = (Path(__file__).resolve().parents[2] / "src").as_posix()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

ROOT = Path(__file__).resolve().parents[2]
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
        """Identify mismatches between expected and actual file names"""
        expected_files = {
            'confounder_flag.parquet': 'confounders.parquet',
            'lab_flag.parquet': 'lab_sensitivity.parquet'
        }
        
        for expected, actual in expected_files.items():
            expected_path = DERIVED / expected
            actual_path = DERIVED / actual
            
            # Expected file should NOT exist
            assert not expected_path.exists(), f"Unexpected file {expected} exists"
            
            # Actual file SHOULD exist  
            assert actual_path.exists(), f"Required file {actual} does not exist"
    
    def test_referral_sequences_missing(self):
        """Test that referral sequences file is missing and needs generation"""
        referral_file = DERIVED / 'referral_sequences.parquet'
        assert not referral_file.exists(), "referral_sequences.parquet should not exist yet"
        
        # Check if script exists to generate it
        script_file = ROOT / 'src' / '07_referral_sequence.py'
        assert script_file.exists(), "07_referral_sequence.py script should exist"

class TestMasterTableScript:
    """Test the master table script expectations vs reality"""
    
    def test_master_table_script_expectations(self):
        """Test what the master table script expects vs what exists"""
        # This test documents the mismatches for fixing
        script_expectations = {
            'confounder_flag.parquet': 'confounders.parquet', 
            'lab_flag.parquet': 'lab_sensitivity.parquet',
            'referral_sequences.parquet': None  # Missing, needs generation
        }
        
        for expected, actual in script_expectations.items():
            expected_path = DERIVED / expected
            
            if actual is None:
                # File needs to be generated
                assert not expected_path.exists(), f"{expected} should not exist until generated"
            else:
                # File exists with different name
                actual_path = DERIVED / actual
                assert not expected_path.exists(), f"Script expects {expected} but it doesn't exist"
                assert actual_path.exists(), f"Actual file {actual} should exist"

if __name__ == "__main__":
    # Run tests to confirm failures before implementing fixes
    pytest.main([__file__, "-v"])