#!/usr/bin/env python3
"""
Tests for Sequential Pathway Analysis module.
"""

import pandas as pd
import sys
from pathlib import Path
import pytest

# Dynamically load module
MODULE_PATH = Path(__file__).parent.parent / 'src' / '08_sequential_pathway_analysis.py'
if MODULE_PATH.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location('sequential', MODULE_PATH)
    sequential = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequential)
    SSDSequentialAnalyzer = sequential.SSDSequentialAnalyzer
else:
    pytest.skip('Sequential module not available', allow_module_level=True)


def create_test_analyzer():
    cohort = pd.DataFrame({'Patient_ID': [1, 2, 3]})

    health = pd.DataFrame({
        'Patient_ID': [1, 1, 2],
        'Condition': ['NYD', 'Anxiety', 'NYD'],
        'Date': pd.to_datetime(['2020-01-01', '2020-05-01', '2020-01-01'])
    })

    lab = pd.DataFrame({
        'Patient_ID': [1, 1, 1, 2],
        'Result': ['normal', 'normal', 'normal', 'normal'],
        'Date': pd.to_datetime(['2020-01-05', '2020-01-10', '2020-02-01', '2020-02-01'])
    })

    referral = pd.DataFrame({
        'Patient_ID': [1, 1],
        'Type': ['medical', 'psychiatric'],
        'Date': pd.to_datetime(['2020-03-01', '2020-06-01'])
    })

    exposure = pd.DataFrame({
        'Patient_ID': [1],
        'Date': pd.to_datetime(['2020-07-01'])
    })

    return SSDSequentialAnalyzer(cohort, health, lab, referral, exposure)


def test_analyze_cohort_results():
    analyzer = create_test_analyzer()
    df = analyzer.analyze_cohort()

    assert len(df) == 3
    assert set(df.columns) == {
        'patient_id', 'pathway_stage', 'complete_pathway',
        'nyd_to_ssd_days', 'stages_completed', 'bottleneck_stage',
        'probability', 'interval_days'
    }

    patient1 = df[df['patient_id'] == 1].iloc[0]
    assert bool(patient1['complete_pathway']) is True
    assert patient1['pathway_stage'] == 7
    assert patient1['nyd_to_ssd_days'] == 182
    assert patient1['probability'] == 1.0

    patient2 = df[df['patient_id'] == 2].iloc[0]
    assert patient2['pathway_stage'] == 1
    assert bool(patient2['complete_pathway']) is False
    assert patient2['probability'] == pytest.approx(1/7, rel=0.01)

    patient3 = df[df['patient_id'] == 3].iloc[0]
    assert patient3['pathway_stage'] == 0
    assert bool(patient3['complete_pathway']) is False
    assert patient3['probability'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

