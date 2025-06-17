#!/usr/bin/env python3
"""
test_mh_exposure_enhanced.py - Tests for enhanced mental health exposure definitions

Tests 180-day drug persistence, enhanced drug classes, and psychiatric referral patterns
for Week 4 enhanced exposure analysis.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMHExposureEnhanced:
    """Test enhanced mental health exposure definitions"""
    
    def test_enhanced_drug_mapping(self):
        """Test enhanced ATC drug code mapping"""
        from mh_exposure_enhanced import create_enhanced_drug_mapping
        
        drug_mapping = create_enhanced_drug_mapping()
        
        # Check enhanced drug classes are included
        assert 'N06A' in drug_mapping  # Antidepressants
        assert 'N03A' in drug_mapping  # Anticonvulsants  
        assert 'N05A' in drug_mapping  # Antipsychotics
        
        # Check specific subcategories
        assert 'N06AB' in drug_mapping  # SSRI antidepressants
        assert 'N03AE' in drug_mapping  # Benzodiazepine anticonvulsants
        assert 'N05AA' in drug_mapping  # Phenothiazine antipsychotics
        
        # Verify classifications
        assert drug_mapping['N06A'] == 'antidepressants'
        assert drug_mapping['N03A'] == 'anticonvulsants'
        assert drug_mapping['N05A'] == 'antipsychotics'
    
    def test_180_day_drug_persistence(self):
        """Test 180-day drug persistence calculation"""
        from mh_exposure_enhanced import calculate_enhanced_drug_persistence
        
        # Create test prescription data
        test_prescriptions = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1, 1, 1, 2, 2],
            'drug_code': ['N06AB06'] * 9,
            'date_prescribed': pd.to_datetime([
                '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', 
                '2023-05-01', '2023-06-01', '2023-07-01',  # Patient 1: 7 months continuous
                '2023-01-01', '2023-08-01'                 # Patient 2: large gap
            ]),
            'days_supply': [30] * 9
        })
        
        persistence = calculate_enhanced_drug_persistence(test_prescriptions, 180)
        
        # Patient 1 should have 180-day persistence (4 months continuous)
        patient_1 = persistence[persistence['patient_id'] == 1].iloc[0]
        assert patient_1['persistent_180'] == True
        assert patient_1['prescription_count'] >= 2
        
        # Patient 2 should not have persistence (large gap)
        patient_2 = persistence[persistence['patient_id'] == 2].iloc[0]
        assert patient_2['persistent_180'] == False
    
    def test_psychiatric_referral_patterns(self):
        """Test enhanced psychiatric referral pattern identification"""
        from mh_exposure_enhanced import identify_enhanced_referral_patterns
        
        test_referrals = pd.DataFrame({
            'patient_id': [1, 1, 2, 3, 3, 4],
            'referral_specialty': [
                'Psychiatry', 'Psychology',      # Patient 1: multiple types
                'Psychiatry',                    # Patient 2: single
                'Psychiatry', 'Mental Health',   # Patient 3: multiple types
                'Cardiology'                     # Patient 4: non-psychiatric
            ],
            'referral_status': [
                'pending', 'completed',
                'pending', 
                'pending', 'cancelled',
                'completed'
            ]
        })
        
        patterns = identify_enhanced_referral_patterns(test_referrals)
        
        # Check pattern identification
        assert len(patterns) >= 3  # At least 3 patients with psychiatric referrals
        
        # Patient 1 should have multiple specialties
        patient_1 = patterns[patterns['patient_id'] == 1].iloc[0]
        assert patient_1['multiple_psychiatric_specialties'] == True
        assert patient_1['has_psychiatry_referral'] == True
        assert patient_1['has_psychology_referral'] == True
        
        # Patient 3 should show referral loop pattern (multiple unresolved)
        patient_3 = patterns[patterns['patient_id'] == 3].iloc[0]
        assert patient_3['referral_loop_pattern'] == True
    
    def test_enhanced_exposure_flag_creation(self):
        """Test creation of enhanced exposure flags"""
        from mh_exposure_enhanced import create_enhanced_exposure_flags
        
        # Create test data
        cohort_df = pd.DataFrame({
            'patient_id': [1, 2, 3, 4],
            'age': [30, 40, 50, 60]
        })
        
        prescriptions_df = pd.DataFrame({
            'patient_id': [1, 1, 1, 1, 1, 1, 1, 3, 3],
            'drug_code': ['N06AB06'] * 7 + ['N05AA01', 'N05AA01'],
            'date_prescribed': pd.to_datetime([
                '2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01',
                '2023-05-01', '2023-06-01', '2023-07-01',  # Patient 1: 7 months persistent
                '2023-01-01', '2023-02-01'                 # Patient 3: not persistent
            ]),
            'days_supply': [30] * 9
        })
        
        referrals_df = pd.DataFrame({
            'patient_id': [2, 2, 4],
            'referral_specialty': ['Psychiatry', 'Psychology', 'Cardiology'],
            'referral_status': ['pending', 'pending', 'completed']
        })
        
        lab_df = pd.DataFrame()  # Empty for this test
        
        enhanced_cohort = create_enhanced_exposure_flags(
            cohort_df, prescriptions_df, referrals_df, lab_df
        )
        
        # Check enhanced exposure columns
        assert 'h1_normal_lab_cascade' in enhanced_cohort.columns
        assert 'h2_psychiatric_referral_loop' in enhanced_cohort.columns
        assert 'h3_drug_persistence_180' in enhanced_cohort.columns
        assert 'ssd_exposure_enhanced' in enhanced_cohort.columns
        
        # Patient 1 should have H3 exposure (drug persistence)
        patient_1 = enhanced_cohort[enhanced_cohort['patient_id'] == 1].iloc[0]
        assert patient_1['h3_drug_persistence_180'] == True
        assert patient_1['ssd_exposure_enhanced'] == True
        
        # Patient 2 should have H2 exposure (referral loop)
        patient_2 = enhanced_cohort[enhanced_cohort['patient_id'] == 2].iloc[0]
        assert patient_2['h2_psychiatric_referral_loop'] == True
        assert patient_2['ssd_exposure_enhanced'] == True
    
    def test_drug_classification_edge_cases(self):
        """Test drug classification with edge cases"""
        from mh_exposure_enhanced import calculate_enhanced_drug_persistence
        
        # Test with various drug codes
        edge_case_prescriptions = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'drug_code': ['N06AX01', 'N99Z99', np.nan],  # Valid, invalid, missing
            'date_prescribed': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01']),
            'days_supply': [30, 30, 30]
        })
        
        persistence = calculate_enhanced_drug_persistence(edge_case_prescriptions, 180)
        
        # Should handle edge cases gracefully
        assert len(persistence) >= 0  # No crashes
        # Patient with valid psychotropic should be included
        assert any(persistence['patient_id'] == 1)
    
    def test_referral_specialty_classification(self):
        """Test psychiatric specialty classification robustness"""
        from mh_exposure_enhanced import identify_enhanced_referral_patterns
        
        test_referrals = pd.DataFrame({
            'patient_id': [1, 2, 3, 4, 5],
            'referral_specialty': [
                'PSYCHIATRY',           # Case variations
                'behavioral health',    # Alternate naming
                'substance abuse',      # Specific subspecialty  
                'Emergency Medicine',   # Non-psychiatric
                None                    # Missing
            ],
            'referral_status': ['pending'] * 5
        })
        
        patterns = identify_enhanced_referral_patterns(test_referrals)
        
        # Should identify first 3 as psychiatric
        psychiatric_patients = patterns[patterns['total_psychiatric_referrals'] > 0]
        assert len(psychiatric_patients) == 3
        assert set(psychiatric_patients['patient_id']) == {1, 2, 3}
    
    def test_persistence_threshold_sensitivity(self):
        """Test sensitivity to different persistence thresholds"""
        from mh_exposure_enhanced import calculate_enhanced_drug_persistence
        
        # Create borderline case (exactly 180 days)
        test_prescriptions = pd.DataFrame({
            'patient_id': [1, 1],
            'drug_code': ['N06AB06', 'N06AB06'],
            'date_prescribed': pd.to_datetime(['2023-01-01', '2023-06-01']),  # 151 days apart
            'days_supply': [150, 30]  # Total coverage exactly 180 days
        })
        
        # Test with 180-day threshold
        persistence_180 = calculate_enhanced_drug_persistence(test_prescriptions, 180)
        
        # Test with stricter 200-day threshold
        persistence_200 = calculate_enhanced_drug_persistence(test_prescriptions, 200)
        
        # Should meet 180-day but not 200-day threshold
        assert persistence_180[persistence_180['patient_id'] == 1]['persistent_180'].iloc[0] == True
        assert persistence_200[persistence_200['patient_id'] == 1]['persistent_200'].iloc[0] == False
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        from mh_exposure_enhanced import create_enhanced_exposure_flags
        
        # Empty datasets
        empty_cohort = pd.DataFrame({'patient_id': [], 'age': []})
        empty_prescriptions = pd.DataFrame({
            'patient_id': [], 'drug_code': [], 
            'date_prescribed': [], 'days_supply': []
        })
        empty_referrals = pd.DataFrame({
            'patient_id': [], 'referral_specialty': [], 'referral_status': []
        })
        empty_lab = pd.DataFrame()
        
        # Should handle empty data gracefully
        result = create_enhanced_exposure_flags(
            empty_cohort, empty_prescriptions, empty_referrals, empty_lab
        )
        
        assert len(result) == 0
        assert 'ssd_exposure_enhanced' in result.columns