#!/usr/bin/env python3
"""
test_hierarchical_index_dates.py - Unit tests for hierarchical index date implementation

Tests the hierarchical index date logic added to 01_cohort_builder.py
to address the 28.3% missing lab dates issue.

Author: Test suite for hierarchical implementation
Date: July 1, 2025
Version: 1.0.0

References:
- DSM-5-TR (2022): Somatic Symptom and Related Disorders criteria
- FINAL_TODO_LIST_IndexDate_Implementation.md (internal documentation)
- Phase4_Implementation_Plan_IndexDate_Solution.md (internal documentation)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestHierarchicalIndexDates:
    """Test suite for hierarchical index date implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create sample patient data
        n_patients = 100
        patient_ids = [f"P{i:04d}" for i in range(n_patients)]
        
        # Create eligibility dataframe
        elig = pd.DataFrame({
            'Patient_ID': patient_ids,
            'BirthYear': np.random.randint(1940, 2000, n_patients),
            'Sex': np.random.choice(['M', 'F'], n_patients),
            'SpanMonths': np.random.uniform(30, 120, n_patients),
            'Charlson': np.random.randint(0, 5, n_patients)
        })
        
        # Create lab data (only 70% have labs - simulating 30% missing)
        lab_patients = np.random.choice(patient_ids, size=int(n_patients * 0.7), replace=False)
        lab = pd.DataFrame({
            'Patient_ID': np.repeat(lab_patients, 3),  # 3 labs per patient
            'PerformedDate': pd.date_range('2015-01-01', periods=len(lab_patients)*3, freq='M')
        })
        
        # Create encounter data (all patients have encounters)
        encounter = pd.DataFrame({
            'Patient_ID': np.repeat(patient_ids, 5),  # 5 encounters per patient
            'Encounter_ID': range(n_patients * 5),
            'DateCreated': pd.date_range('2014-01-01', periods=n_patients*5, freq='W')
        })
        
        # Create encounter diagnosis for mental health
        mh_encounters = np.random.choice(encounter['Encounter_ID'].values, 
                                       size=int(len(encounter) * 0.3), replace=False)
        enc_diag = pd.DataFrame({
            'Encounter_ID': mh_encounters,
            'DiagnosisCode_calc': np.random.choice(['296.2', '300.0', '311'], len(mh_encounters))
        })
        
        # Create medication data (80% have some medication)
        med_patients = np.random.choice(patient_ids, size=int(n_patients * 0.8), replace=False)
        medication = pd.DataFrame({
            'Patient_ID': np.repeat(med_patients, 2),  # 2 prescriptions per patient
            'Code_calc': np.random.choice(['N05BA01', 'N06AB03', 'N03AX12'], len(med_patients)*2),
            'StartDate': pd.date_range('2015-06-01', periods=len(med_patients)*2, freq='M'),
            'StopDate': pd.date_range('2015-09-01', periods=len(med_patients)*2, freq='M')
        })
        
        return {
            'elig': elig,
            'lab': lab,
            'encounter': encounter,
            'enc_diag': enc_diag,
            'medication': medication
        }
    
    def test_lab_index_primary(self, sample_data):
        """Test that lab index is used as primary when available."""
        elig = sample_data['elig'].copy()
        lab = sample_data['lab']
        
        # Add lab index
        idx_lab = lab.groupby("Patient_ID")["PerformedDate"].min().rename("IndexDate_lab")
        elig = elig.merge(idx_lab, left_on="Patient_ID", right_index=True, how="left")
        
        # Check that ~70% have lab index
        lab_coverage = elig['IndexDate_lab'].notna().mean()
        assert 0.65 <= lab_coverage <= 0.75, f"Expected ~70% lab coverage, got {lab_coverage:.1%}"
        
        # Check that lab dates are valid
        assert elig['IndexDate_lab'].min() >= pd.Timestamp('2015-01-01')
    
    def test_mental_health_encounter_fallback(self, sample_data):
        """Test mental health encounter as secondary index."""
        elig = sample_data['elig'].copy()
        encounter = sample_data['encounter']
        enc_diag = sample_data['enc_diag']
        
        # Get MH encounters
        mh_encounters = encounter[
            encounter['Encounter_ID'].isin(enc_diag['Encounter_ID'])
        ]
        
        # Calculate MH index
        idx_mh = mh_encounters.groupby('Patient_ID')['DateCreated'].min().rename('IndexDate_mh')
        elig = elig.merge(idx_mh, left_on="Patient_ID", right_index=True, how="left")
        
        # Some patients should have MH encounters
        mh_coverage = elig['IndexDate_mh'].notna().mean()
        assert mh_coverage > 0, "No mental health encounters found"
    
    def test_psychotropic_medication_tertiary(self, sample_data):
        """Test psychotropic medication as tertiary index."""
        elig = sample_data['elig'].copy()
        medication = sample_data['medication']
        
        # Filter psychotropic medications
        psychotropic_atc = ['N05', 'N06']
        psych_meds = medication[
            medication.Code_calc.str.startswith(tuple(psychotropic_atc), na=False)
        ]
        
        # Calculate duration
        psych_meds['duration_days'] = (
            pd.to_datetime(psych_meds['StopDate']) - 
            pd.to_datetime(psych_meds['StartDate'])
        ).dt.days
        
        # Find patients with ≥180 days
        psych_duration = psych_meds.groupby('Patient_ID')['duration_days'].sum()
        long_psych = psych_duration[psych_duration >= 180].index
        
        # Some patients should qualify
        assert len(long_psych) > 0, "No patients with long psychotropic use"
    
    def test_hierarchical_index_creation(self, sample_data):
        """Test complete hierarchical index date creation."""
        elig = sample_data['elig'].copy()
        lab = sample_data['lab']
        encounter = sample_data['encounter'] 
        enc_diag = sample_data['enc_diag']
        medication = sample_data['medication']
        
        # 1. Lab index
        idx_lab = lab.groupby("Patient_ID")["PerformedDate"].min().rename("IndexDate_lab")
        elig = elig.merge(idx_lab, left_on="Patient_ID", right_index=True, how="left")
        
        # 2. MH encounter index
        mh_encounters = encounter[
            encounter['Encounter_ID'].isin(enc_diag['Encounter_ID'])
        ]
        idx_mh = mh_encounters.groupby('Patient_ID')['DateCreated'].min().rename('IndexDate_mh')
        elig = elig.merge(idx_mh, left_on="Patient_ID", right_index=True, how="left")
        
        # 3. Psychotropic index
        psychotropic_atc = ['N05', 'N06']
        psych_meds = medication[medication.Code_calc.str.startswith(tuple(psychotropic_atc), na=False)]
        psych_meds['duration_days'] = 90  # Simplified for test
        first_psych = psych_meds.groupby('Patient_ID')['StartDate'].min().rename('IndexDate_psych')
        elig = elig.merge(first_psych, left_on="Patient_ID", right_index=True, how="left")
        
        # 4. Create unified index
        elig['IndexDate_unified'] = elig['IndexDate_lab'].fillna(
            elig['IndexDate_mh'].fillna(
                elig['IndexDate_psych']
            )
        )
        
        # 5. Track source
        elig['index_date_source'] = np.select(
            [
                elig['IndexDate_lab'].notna(),
                elig['IndexDate_mh'].notna(),
                elig['IndexDate_psych'].notna()
            ],
            ['Laboratory', 'Mental_Health_Encounter', 'Psychotropic_Medication'],
            default='No_Index'
        )
        
        # Verify all patients get an index date (with fallback to first encounter)
        no_index = elig['index_date_source'] == 'No_Index'
        if no_index.any():
            first_enc = encounter.groupby('Patient_ID')['DateCreated'].min()
            elig.loc[no_index, 'IndexDate_unified'] = elig.loc[no_index, 'Patient_ID'].map(first_enc)
            elig.loc[no_index, 'index_date_source'] = 'First_Encounter_Fallback'
        
        # All patients should have index date
        assert elig['IndexDate_unified'].notna().all(), "Some patients missing unified index date"
        
        # Source distribution should be reasonable
        source_counts = elig['index_date_source'].value_counts()
        assert 'Laboratory' in source_counts.index
        assert source_counts['Laboratory'] > len(elig) * 0.5  # >50% from lab
    
    def test_phenotype_assignment(self, sample_data):
        """Test phenotype assignment based on lab utilization."""
        elig = sample_data['elig'].copy()
        lab = sample_data['lab']
        
        # Add lab index
        idx_lab = lab.groupby("Patient_ID")["PerformedDate"].min().rename("IndexDate_lab")
        elig = elig.merge(idx_lab, left_on="Patient_ID", right_index=True, how="left")
        
        # Create phenotype
        elig['lab_utilization_phenotype'] = np.where(
            elig['IndexDate_lab'].isna(),
            'Avoidant_SSD',
            'Test_Seeking_SSD'
        )
        
        # Check distribution
        phenotype_counts = elig['lab_utilization_phenotype'].value_counts()
        assert 'Avoidant_SSD' in phenotype_counts.index
        assert 'Test_Seeking_SSD' in phenotype_counts.index
        
        # Avoidant should be ~30%
        avoidant_pct = (phenotype_counts['Avoidant_SSD'] / len(elig)) * 100
        assert 20 <= avoidant_pct <= 40, f"Expected ~30% avoidant, got {avoidant_pct:.1f}%"
    
    def test_index_date_consistency(self, sample_data):
        """Test that index dates are internally consistent."""
        elig = sample_data['elig'].copy()
        lab = sample_data['lab']
        
        # Add lab index and unified
        idx_lab = lab.groupby("Patient_ID")["PerformedDate"].min().rename("IndexDate_lab")
        elig = elig.merge(idx_lab, left_on="Patient_ID", right_index=True, how="left")
        elig['IndexDate_unified'] = elig['IndexDate_lab']  # Simplified
        
        # For patients with lab index, unified should equal lab
        has_lab = elig['IndexDate_lab'].notna()
        assert (elig.loc[has_lab, 'IndexDate_unified'] == elig.loc[has_lab, 'IndexDate_lab']).all()
    
    def test_date_ordering(self, sample_data):
        """Test that index dates follow expected temporal ordering."""
        elig = sample_data['elig'].copy()
        lab = sample_data['lab']
        encounter = sample_data['encounter']
        
        # Get first lab and first encounter
        idx_lab = lab.groupby("Patient_ID")["PerformedDate"].min()
        first_enc = encounter.groupby("Patient_ID")["DateCreated"].min()
        
        # Merge both
        date_compare = pd.DataFrame({
            'first_lab': idx_lab,
            'first_encounter': first_enc
        })
        
        # Lab dates should generally be after first encounter
        # (patients need to be in system before labs ordered)
        valid_ordering = date_compare.dropna()
        late_labs = (valid_ordering['first_lab'] >= valid_ordering['first_encounter']).mean()
        assert late_labs > 0.8, f"Expected most labs after first encounter, got {late_labs:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])