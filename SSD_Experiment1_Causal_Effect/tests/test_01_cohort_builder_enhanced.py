#!/usr/bin/env python3
"""
Test Suite for Enhanced SSD Cohort Builder with NYD Enhancements

Following TDD Methodology (CLAUDE.md Requirements):
1. Write failing tests FIRST
2. Implement minimal code to make tests pass  
3. Refactor while keeping tests green

Author: Ryhan Suny
Date: January 2025
Version: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import module under test (will be created)
import importlib.util

class TestNYDBodyPartMapping:
    """Test NYD ICD code to body part mapping functionality"""
    
    def test_nyd_body_part_mapping_creation(self):
        """Test that NYD body part mapping includes clinical categories"""
        # This test will fail initially - we need to create the module
        try:
            spec = importlib.util.spec_from_file_location(
                "cohort_builder_enhanced",
                Path(__file__).parent.parent / 'src' / '01_cohort_builder_enhanced.py'
            )
            cohort_enhanced = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cohort_enhanced)
            
            nyd_mapping = cohort_enhanced.create_nyd_body_part_mapping()
            
            # Test required body part categories
            assert 'General' in nyd_mapping.values(), "Missing General NYD category"
            assert 'Mental/Behavioral' in nyd_mapping.values(), "Missing Mental/Behavioral category"
            assert 'Neurological' in nyd_mapping.values(), "Missing Neurological category"
            assert 'Cardiovascular' in nyd_mapping.values(), "Missing Cardiovascular category"
            assert 'Respiratory' in nyd_mapping.values(), "Missing Respiratory category"
            
            # Test specific ICD codes
            assert '799.9' in nyd_mapping, "Missing 799.9 (Other ill-defined conditions)"
            assert nyd_mapping['799.9'] == 'General', "Incorrect mapping for 799.9"
            
        except (FileNotFoundError, ModuleNotFoundError):
            pytest.fail("Module 01_cohort_builder_enhanced.py not found - this test should fail first in TDD")

    def test_nyd_body_part_comprehensive_coverage(self):
        """Test that NYD mapping covers all major body systems"""
        try:
            spec = importlib.util.spec_from_file_location(
                "cohort_builder_enhanced",
                Path(__file__).parent.parent / 'src' / '01_cohort_builder_enhanced.py'
            )
            cohort_enhanced = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cohort_enhanced)
            
            nyd_mapping = cohort_enhanced.create_nyd_body_part_mapping()
            
            # Should have comprehensive coverage
            assert len(nyd_mapping) >= 15, f"Should have at least 15 NYD codes, got {len(nyd_mapping)}"
            
            # Test body system categories
            body_systems = set(nyd_mapping.values())
            required_systems = {
                'General', 'Mental/Behavioral', 'Neurological', 
                'Cardiovascular', 'Respiratory', 'Gastrointestinal',
                'Musculoskeletal', 'Dermatological', 'Genitourinary'
            }
            
            missing_systems = required_systems - body_systems
            assert len(missing_systems) == 0, f"Missing body systems: {missing_systems}"
            
        except (FileNotFoundError, ModuleNotFoundError):
            pytest.fail("Module not found - failing as expected in TDD cycle")

class TestNYDBinaryFlags:
    """Test NYD binary flag generation"""
    
    def test_nyd_binary_flag_calculation(self):
        """Test that NYD binary flags are calculated correctly"""
        # Mock data for testing
        mock_nyd_data = pd.DataFrame({
            'Patient_ID': [1, 1, 2, 3, 3, 3],
            'ICD_code': ['799.9', 'V71.0', '799.9', 'V71.1', '780.9', '799.9'],
            'body_part': ['General', 'Mental/Behavioral', 'General', 'Neurological', 'General', 'General']
        })
        
        mock_cohort = pd.DataFrame({
            'Patient_ID': [1, 2, 3, 4],
            'NYD_count': [2, 1, 3, 0]
        })
        
        try:
            spec = importlib.util.spec_from_file_location(
                "cohort_builder_enhanced",
                Path(__file__).parent.parent / 'src' / '01_cohort_builder_enhanced.py'
            )
            cohort_enhanced = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cohort_enhanced)
            
            enhanced_cohort = cohort_enhanced.add_nyd_binary_flags(mock_cohort, mock_nyd_data)
            
            # Test binary flag columns exist
            assert 'NYD_yn' in enhanced_cohort.columns, "Missing NYD_yn binary flag"
            assert 'NYD_general_yn' in enhanced_cohort.columns, "Missing NYD_general_yn flag"
            assert 'NYD_mental_yn' in enhanced_cohort.columns, "Missing NYD_mental_yn flag"
            assert 'NYD_neuro_yn' in enhanced_cohort.columns, "Missing NYD_neuro_yn flag"
            
            # Test binary flag logic
            assert enhanced_cohort.loc[enhanced_cohort['Patient_ID'] == 1, 'NYD_yn'].iloc[0] == 1, "Patient 1 should have NYD_yn=1"
            assert enhanced_cohort.loc[enhanced_cohort['Patient_ID'] == 4, 'NYD_yn'].iloc[0] == 0, "Patient 4 should have NYD_yn=0"
            assert enhanced_cohort.loc[enhanced_cohort['Patient_ID'] == 1, 'NYD_mental_yn'].iloc[0] == 1, "Patient 1 should have mental NYD"
            assert enhanced_cohort.loc[enhanced_cohort['Patient_ID'] == 3, 'NYD_neuro_yn'].iloc[0] == 1, "Patient 3 should have neuro NYD"
            
        except (FileNotFoundError, ModuleNotFoundError):
            pytest.fail("Module not found - expected failure in TDD")

    def test_nyd_binary_flags_edge_cases(self):
        """Test NYD binary flags handle edge cases correctly"""
        # Empty NYD data
        empty_nyd = pd.DataFrame(columns=['Patient_ID', 'ICD_code', 'body_part'])
        mock_cohort = pd.DataFrame({
            'Patient_ID': [1, 2, 3],
            'NYD_count': [0, 0, 0]
        })
        
        try:
            spec = importlib.util.spec_from_file_location(
                "cohort_builder_enhanced",
                Path(__file__).parent.parent / 'src' / '01_cohort_builder_enhanced.py'
            )
            cohort_enhanced = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cohort_enhanced)
            
            enhanced_cohort = cohort_enhanced.add_nyd_binary_flags(mock_cohort, empty_nyd)
            
            # All patients should have NYD_yn=0
            assert all(enhanced_cohort['NYD_yn'] == 0), "Empty NYD data should result in all NYD_yn=0"
            assert all(enhanced_cohort['NYD_general_yn'] == 0), "Empty NYD data should result in all body part flags=0"
            
        except (FileNotFoundError, ModuleNotFoundError):
            pytest.fail("Module not found - expected in TDD cycle")

class TestEnhancedCohortBuilder:
    """Test complete enhanced cohort building functionality"""
    
    def test_enhanced_cohort_integration(self):
        """Test full integration of enhanced cohort building"""
        mock_base_cohort = pd.DataFrame({
            'Patient_ID': [1, 2, 3, 4],
            'Age_at_2015': [45, 32, 67, 28],
            'NYD_count': [2, 0, 1, 0],
            'IndexDate_lab': pd.to_datetime(['2015-01-01'] * 4)
        })
        
        mock_nyd_records = pd.DataFrame({
            'Patient_ID': [1, 1, 3],
            'ICD_code': ['799.9', 'V71.0', '780.9'],
            'diagnosis_date': pd.to_datetime(['2017-06-01', '2017-08-01', '2017-09-01'])
        })
        
        try:
            spec = importlib.util.spec_from_file_location(
                "cohort_builder_enhanced",
                Path(__file__).parent.parent / 'src' / '01_cohort_builder_enhanced.py'
            )
            cohort_enhanced = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cohort_enhanced)
            
            enhanced_cohort = cohort_enhanced.build_enhanced_cohort(mock_base_cohort, mock_nyd_records)
            
            # Check structure
            assert len(enhanced_cohort) == len(mock_base_cohort), "Enhanced cohort should maintain patient count"
            assert 'NYD_yn' in enhanced_cohort.columns, "Missing enhanced NYD columns"
            assert 'NYD_body_part_summary' in enhanced_cohort.columns, "Missing body part summary"
            
            # Check data types
            assert enhanced_cohort['NYD_yn'].dtype in ['int64', 'int32'], "NYD_yn should be integer"
            
        except (FileNotFoundError, ModuleNotFoundError):
            pytest.fail("Module not found - expected failure in TDD")

class TestNYDValidationReporting:
    """Test NYD enhancement validation and reporting"""
    
    def test_nyd_enhancement_report_generation(self):
        """Test that NYD enhancement generates proper validation report"""
        mock_enhanced_cohort = pd.DataFrame({
            'Patient_ID': [1, 2, 3, 4],
            'NYD_count': [2, 0, 1, 0],
            'NYD_yn': [1, 0, 1, 0],
            'NYD_general_yn': [1, 0, 1, 0],
            'NYD_mental_yn': [1, 0, 0, 0],
            'NYD_neuro_yn': [0, 0, 0, 0]
        })
        
        try:
            spec = importlib.util.spec_from_file_location(
                "cohort_builder_enhanced",
                Path(__file__).parent.parent / 'src' / '01_cohort_builder_enhanced.py'
            )
            cohort_enhanced = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cohort_enhanced)
            
            report_content = cohort_enhanced.create_nyd_enhancement_report(mock_enhanced_cohort)
            
            # Check report content
            assert "NYD Enhancement Report" in report_content, "Missing report title"
            assert "Binary Flags Added" in report_content, "Missing binary flags section"
            assert "Body Part Distribution" in report_content, "Missing body part analysis"
            assert str(mock_enhanced_cohort['NYD_yn'].sum()) in report_content, "Missing actual counts"
            
        except (FileNotFoundError, ModuleNotFoundError):
            pytest.fail("Module not found - expected in TDD")

class TestNYDPerformanceAndEdgeCases:
    """Test NYD enhancement performance and edge case handling"""
    
    def test_nyd_large_dataset_performance(self):
        """Test NYD enhancement with large datasets"""
        # Create large mock dataset
        large_cohort = pd.DataFrame({
            'Patient_ID': list(range(10000)),
            'NYD_count': np.random.randint(0, 5, 10000),
            'Age_at_2015': np.random.randint(18, 80, 10000)
        })
        
        large_nyd = pd.DataFrame({
            'Patient_ID': np.random.choice(range(10000), 5000),
            'ICD_code': np.random.choice(['799.9', 'V71.0', '780.9'], 5000),
            'diagnosis_date': pd.to_datetime(['2017-01-01'] * 5000)
        })
        
        try:
            spec = importlib.util.spec_from_file_location(
                "cohort_builder_enhanced",
                Path(__file__).parent.parent / 'src' / '01_cohort_builder_enhanced.py'
            )
            cohort_enhanced = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cohort_enhanced)
            
            import time
            start_time = time.time()
            enhanced_large_cohort = cohort_enhanced.build_enhanced_cohort(large_cohort, large_nyd)
            execution_time = time.time() - start_time
            
            # Performance check
            assert execution_time < 15, f"Performance test failed: took {execution_time:.2f} seconds for 10k patients"
            assert len(enhanced_large_cohort) == len(large_cohort), "Large dataset processing should maintain patient count"
            
        except (FileNotFoundError, ModuleNotFoundError):
            pytest.fail("Module not found - expected in TDD")

    def test_nyd_invalid_data_handling(self):
        """Test handling of invalid or missing NYD data"""
        invalid_cohort = pd.DataFrame({
            'Patient_ID': [1, 2, 3],
            'NYD_count': [np.nan, -1, 0]  # Invalid values
        })
        
        invalid_nyd = pd.DataFrame({
            'Patient_ID': [1, 2],
            'ICD_code': ['INVALID', None],  # Invalid codes
            'diagnosis_date': [pd.NaT, pd.to_datetime('2017-01-01')]
        })
        
        try:
            spec = importlib.util.spec_from_file_location(
                "cohort_builder_enhanced",
                Path(__file__).parent.parent / 'src' / '01_cohort_builder_enhanced.py'
            )
            cohort_enhanced = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cohort_enhanced)
            
            # Should handle gracefully without crashing
            enhanced_cohort = cohort_enhanced.build_enhanced_cohort(invalid_cohort, invalid_nyd)
            
            # Check that it processed without errors
            assert len(enhanced_cohort) == len(invalid_cohort), "Should handle invalid data gracefully"
            assert 'NYD_yn' in enhanced_cohort.columns, "Should still create required columns"
            
        except (FileNotFoundError, ModuleNotFoundError):
            pytest.fail("Module not found - expected in TDD")

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"]) 