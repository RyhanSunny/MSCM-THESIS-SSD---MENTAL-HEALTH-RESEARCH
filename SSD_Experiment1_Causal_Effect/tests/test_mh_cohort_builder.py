#!/usr/bin/env python3
"""
test_mh_cohort_builder.py - Tests for mental health-specific cohort building

Tests implementation of mental health ICD filtering and enhanced exposure definitions
following TDD principles.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMentalHealthCohortBuilder:
    """Test mental health cohort building enhancements"""
    
    def test_mental_health_icd_filter(self):
        """Test filtering by mental health ICD codes"""
        from mh_cohort_builder import filter_mental_health_patients
        
        # Create test data with mixed diagnoses
        test_df = pd.DataFrame({
            'patient_id': [1, 2, 3, 4, 5],
            'diagnosis_codes': [
                'F32.1',    # Major depressive disorder
                'I10',      # Hypertension (not MH)
                '296.2',    # Major depressive episode
                'F41.1',    # Generalized anxiety disorder
                'K59.1'     # Diarrhea (not MH)
            ]
        })
        
        # Filter for mental health patients
        mh_patients = filter_mental_health_patients(test_df)
        
        # Should include patients 1, 3, 4 (MH diagnoses)
        assert len(mh_patients) == 3
        assert set(mh_patients['patient_id']) == {1, 3, 4}
    
    def test_mental_health_icd_patterns(self):
        """Test comprehensive mental health ICD pattern matching"""
        from mh_cohort_builder import is_mental_health_diagnosis
        
        # Test F codes (ICD-10)
        assert is_mental_health_diagnosis('F32.1')  # Depression
        assert is_mental_health_diagnosis('F41.9')  # Anxiety
        assert is_mental_health_diagnosis('F43.2')  # PTSD
        assert is_mental_health_diagnosis('F48.0')  # Neurasthenia
        
        # Test 296.* codes (ICD-9)
        assert is_mental_health_diagnosis('296.2')  # Major depression
        assert is_mental_health_diagnosis('296.80') # Bipolar
        
        # Test 300.* codes (ICD-9)  
        assert is_mental_health_diagnosis('300.0')  # Anxiety states
        assert is_mental_health_diagnosis('300.4')  # Dysthymic disorder
        
        # Test non-MH codes
        assert not is_mental_health_diagnosis('I10')    # Hypertension
        assert not is_mental_health_diagnosis('K59.1')  # GI
        assert not is_mental_health_diagnosis('N18.6')  # CKD
    
    def test_drug_persistence_180_day(self):
        """Test 180-day drug persistence calculation"""
        from mh_cohort_builder import calculate_drug_persistence_180
        
        # Create test prescription data
        test_prescriptions = pd.DataFrame({
            'patient_id': [1, 1, 1, 2, 2],
            'drug_code': ['N06AB', 'N06AB', 'N06AB', 'N06AB', 'N03AF'],
            'date_prescribed': pd.to_datetime([
                '2023-01-01', '2023-02-01', '2023-04-01',  # Patient 1: gap in March
                '2023-01-01', '2023-07-01'                  # Patient 2: large gap
            ]),
            'days_supply': [30, 30, 30, 30, 30]
        })
        
        persistence = calculate_drug_persistence_180(test_prescriptions)
        
        # Patient 1 should not have 180-day persistence (gap too large)
        # Patient 2 should not have 180-day persistence (massive gap)
        assert persistence[persistence['patient_id'] == 1]['persistent_180'].iloc[0] == False
        assert persistence[persistence['patient_id'] == 2]['persistent_180'].iloc[0] == False
    
    def test_enhanced_drug_classes(self):
        """Test inclusion of enhanced drug classes N06A, N03A, N05A"""
        from mh_cohort_builder import classify_psychotropic_drugs
        
        test_drugs = [
            'N06AB06',  # Sertraline (antidepressant)
            'N03AE01',  # Clonazepam (anticonvulsant/anxiolytic)
            'N05AB02',  # Fluphenazine (antipsychotic)
            'N02BA01',  # Aspirin (not psychotropic)
            'N06AF01'   # Moclobemide (MAOI)
        ]
        
        classifications = classify_psychotropic_drugs(test_drugs)
        
        # Should classify first 3 as psychotropic
        assert classifications['N06AB06'] == 'antidepressant'
        assert classifications['N03AE01'] == 'anticonvulsant_anxiolytic'
        assert classifications['N05AB02'] == 'antipsychotic'
        assert classifications['N02BA01'] == 'not_psychotropic'
        assert classifications['N06AF01'] == 'antidepressant'
    
    def test_psychiatric_referral_logic(self):
        """Test psychiatric referral identification"""
        from mh_cohort_builder import identify_psychiatric_referrals
        
        test_referrals = pd.DataFrame({
            'patient_id': [1, 2, 3, 4],
            'referral_specialty': [
                'Psychiatry',
                'Psychology', 
                'Mental Health',
                'Cardiology'
            ],
            'referral_status': ['pending', 'completed', 'pending', 'completed']
        })
        
        psych_referrals = identify_psychiatric_referrals(test_referrals)
        
        # Should identify first 3 as psychiatric referrals
        assert len(psych_referrals) == 3
        assert set(psych_referrals['patient_id']) == {1, 2, 3}
    
    def test_cohort_size_validation(self):
        """Test that mental health cohort meets expected size"""
        from mh_cohort_builder import validate_mh_cohort_size
        
        # Create test cohort with large enough size and high MH percentage
        test_cohort = pd.DataFrame({
            'patient_id': range(250000),  # Above minimum threshold
            'is_mental_health': [True] * 200000 + [False] * 50000  # 80% MH patients
        })
        
        validation = validate_mh_cohort_size(test_cohort)
        
        assert validation['total_patients'] == 250000
        assert validation['mh_patients'] == 200000
        assert validation['mh_percentage'] == 80.0
        assert validation['meets_minimum'] == True
    
    def test_integration_with_existing_pipeline(self, tmp_path):
        """Test integration with existing cohort builder"""
        from mh_cohort_builder import enhance_existing_cohort
        
        # Create mock existing cohort
        existing_cohort = pd.DataFrame({
            'patient_id': [1, 2, 3, 4, 5],
            'age': [25, 35, 45, 55, 65],
            'sex': ['F', 'M', 'F', 'M', 'F']
        })
        
        # Create mock diagnosis data with more MH patients
        diagnosis_data = pd.DataFrame({
            'patient_id': [1, 2, 3, 4, 5],
            'diagnosis_codes': ['F32.1', 'F41.1', 'F43.2', '296.2', 'F48.0']  # All MH codes
        })
        
        # Enhance cohort
        enhanced_cohort = enhance_existing_cohort(
            existing_cohort, 
            diagnosis_data,
            tmp_path
        )
        
        # Should add mental health flags
        assert 'is_mental_health' in enhanced_cohort.columns
        assert 'mh_diagnosis_category' in enhanced_cohort.columns
        
        # Function filters to MH patients only, so all should be MH
        assert len(enhanced_cohort) == 5  # All 5 patients have MH diagnoses
        assert enhanced_cohort['is_mental_health'].all()  # All should be True
        
        # Check specific patients exist
        patient_ids = set(enhanced_cohort['patient_id'])
        assert 1 in patient_ids
        assert 2 in patient_ids