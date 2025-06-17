#!/usr/bin/env python3
"""
test_mh_outcomes.py - Tests for mental health-specific outcome flags

Tests MH encounter identification and psychiatric ED visit flagging
as required by H1-H3 hypotheses.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestMentalHealthOutcomes:
    """Test mental health-specific outcome identification"""
    
    def test_mental_health_encounter_identification(self):
        """Test identification of mental health service encounters"""
        from mh_outcomes import identify_mh_encounters
        
        # Create test encounter data
        test_encounters = pd.DataFrame({
            'encounter_id': [1, 2, 3, 4, 5],
            'patient_id': [1, 1, 2, 3, 3],
            'provider_specialty': [
                'Psychiatry',
                'Family Medicine', 
                'Psychology',
                'Emergency Medicine',
                'Mental Health'
            ],
            'diagnosis_codes': [
                'F32.1',    # Depression
                'I10',      # Hypertension  
                'F41.1',    # Anxiety
                'S72.001A', # Fracture
                'F43.2'     # PTSD
            ],
            'encounter_date': pd.to_datetime([
                '2023-01-01', '2023-01-15', '2023-02-01',
                '2023-02-15', '2023-03-01'
            ])
        })
        
        mh_encounters = identify_mh_encounters(test_encounters)
        
        # Should identify encounters 1, 3, 5 as mental health
        assert len(mh_encounters) == 3
        assert set(mh_encounters['encounter_id']) == {1, 3, 5}
    
    def test_psychiatric_ed_visit_identification(self):
        """Test identification of psychiatric emergency department visits"""
        from mh_outcomes import identify_psychiatric_ed_visits
        
        # Create test ED visit data
        test_ed_visits = pd.DataFrame({
            'encounter_id': [1, 2, 3, 4, 5],
            'patient_id': [1, 1, 2, 3, 4],
            'encounter_type': ['Emergency'] * 5,
            'diagnosis_codes': [
                'F32.1',        # Depression in ED
                'S72.001A',     # Fracture in ED
                'F43.2,F41.1',  # PTSD + Anxiety in ED
                'I46.9',        # Cardiac arrest
                'F20.9'         # Schizophrenia in ED
            ],
            'discharge_disposition': [
                'Psychiatric unit',
                'Home',
                'Mental health facility', 
                'ICU',
                'Psychiatric hospital'
            ]
        })
        
        psych_ed = identify_psychiatric_ed_visits(test_ed_visits)
        
        # Should identify encounters 1, 3, 5 as psychiatric ED
        assert len(psych_ed) == 3
        assert set(psych_ed['encounter_id']) == {1, 3, 5}
    
    def test_mh_encounter_counting(self):
        """Test counting of mental health encounters per patient"""
        from mh_outcomes import count_mh_encounters_by_patient
        
        test_encounters = pd.DataFrame({
            'encounter_id': range(1, 11),
            'patient_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
            'is_mh_encounter': [True, True, False, True, False, True, True, True, False, False]
        })
        
        mh_counts = count_mh_encounters_by_patient(test_encounters)
        
        # Expected counts: Patient 1=2, Patient 2=1, Patient 3=3, Patient 4=0
        assert mh_counts[mh_counts['patient_id'] == 1]['mh_encounters_count'].iloc[0] == 2
        assert mh_counts[mh_counts['patient_id'] == 2]['mh_encounters_count'].iloc[0] == 1
        assert mh_counts[mh_counts['patient_id'] == 3]['mh_encounters_count'].iloc[0] == 3
        assert mh_counts[mh_counts['patient_id'] == 4]['mh_encounters_count'].iloc[0] == 0
    
    def test_provider_specialty_classification(self):
        """Test provider specialty-based MH classification"""
        from mh_outcomes import classify_provider_specialty
        
        specialties = [
            'Psychiatry',
            'Psychology', 
            'Mental Health',
            'Behavioral Health',
            'Family Medicine',
            'Emergency Medicine',
            'Psychiatric Nursing',
            'Addiction Medicine'
        ]
        
        classifications = [classify_provider_specialty(s) for s in specialties]
        
        # First 4 and last 2 should be mental health
        expected = [True, True, True, True, False, False, True, True]
        assert classifications == expected
    
    def test_mental_health_diagnosis_in_encounter(self):
        """Test diagnosis-based MH encounter identification"""
        from mh_outcomes import has_mh_diagnosis
        
        diagnosis_strings = [
            'F32.1',                    # Single MH diagnosis
            'I10,F41.1,K59.00',       # Mixed with MH
            'S72.001A,I46.9',         # No MH diagnoses
            'F43.2;F41.1;F32.1',     # Multiple MH diagnoses
            '',                        # Empty
            'F48.0'                   # Somatoform disorder
        ]
        
        results = [has_mh_diagnosis(d) for d in diagnosis_strings]
        expected = [True, True, False, True, False, True]
        
        assert results == expected
    
    def test_psychiatric_ed_criteria(self):
        """Test comprehensive psychiatric ED visit criteria"""
        from mh_outcomes import meets_psychiatric_ed_criteria
        
        test_cases = [
            {
                'encounter_type': 'Emergency',
                'diagnosis_codes': 'F32.1',
                'discharge_disposition': 'Home',
                'provider_specialty': 'Emergency Medicine'
            },
            {
                'encounter_type': 'Emergency', 
                'diagnosis_codes': 'S72.001A',
                'discharge_disposition': 'Psychiatric unit',
                'provider_specialty': 'Emergency Medicine'
            },
            {
                'encounter_type': 'Emergency',
                'diagnosis_codes': 'F41.1', 
                'discharge_disposition': 'Mental health facility',
                'provider_specialty': 'Psychiatry'
            },
            {
                'encounter_type': 'Inpatient',
                'diagnosis_codes': 'F32.1',
                'discharge_disposition': 'Home', 
                'provider_specialty': 'Psychiatry'
            }
        ]
        
        results = [meets_psychiatric_ed_criteria(**case) for case in test_cases]
        
        # Cases 1, 2, 3 should qualify as psychiatric ED
        # Case 4 should not (not emergency encounter)
        expected = [True, True, True, False]
        assert results == expected
    
    def test_outcome_flag_integration(self, tmp_path):
        """Test integration with existing outcome flagging"""
        from mh_outcomes import enhance_outcome_flags
        
        # Create test data
        encounters = pd.DataFrame({
            'encounter_id': [1, 2, 3, 4],
            'patient_id': [1, 1, 2, 2],
            'encounter_type': ['Emergency', 'Outpatient', 'Emergency', 'Inpatient'],
            'provider_specialty': ['Psychiatry', 'Family Medicine', 'Emergency Medicine', 'Psychology'],
            'diagnosis_codes': ['F32.1', 'I10', 'F41.1', 'F43.2'],
            'encounter_date': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15'])
        })
        
        cohort = pd.DataFrame({
            'patient_id': [1, 2],
            'age': [35, 45],
            'sex': ['F', 'M']
        })
        
        enhanced_cohort = enhance_outcome_flags(cohort, encounters, tmp_path)
        
        # Should add MH outcome columns
        assert 'mh_encounters_count' in enhanced_cohort.columns
        assert 'psychiatric_ed_visit' in enhanced_cohort.columns
        
        # Check specific values
        patient_1 = enhanced_cohort[enhanced_cohort['patient_id'] == 1].iloc[0]
        patient_2 = enhanced_cohort[enhanced_cohort['patient_id'] == 2].iloc[0]
        
        assert patient_1['mh_encounters_count'] >= 1  # At least the psychiatry encounter
        assert patient_1['psychiatric_ed_visit'] == True  # Emergency + F32.1
        assert patient_2['mh_encounters_count'] >= 1  # Psychology encounter
        assert patient_2['psychiatric_ed_visit'] == True  # Emergency + F41.1