import pandas as pd
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parent.parent / 'src' / '09_felipe_patient_table.py'
spec = importlib.util.spec_from_file_location('felipe', MODULE_PATH)
felipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(felipe)


def test_generate_felipe_table_basic():
    cohort = pd.DataFrame({
        'Patient_ID': [1, 2],
        'Age_at_2018': [40, 60],
        'NYD_count': [1, 0]
    })
    referral = pd.DataFrame({
        'Patient_ID': [1],
        'Name_calc': ['Psychiatry Clinic'],
        'Type': ['psychiatric']
    })

    table = felipe.generate_felipe_table(cohort, referral)

    assert list(table.columns) == ['PID', 'age', 'NYD_yn', 'referred_to_psy_yn']
    assert table.loc[0, 'referred_to_psy_yn'] == 1
    assert table.loc[1, 'referred_to_psy_yn'] == 0
    assert table.loc[0, 'NYD_yn'] == 1
    assert table.loc[1, 'NYD_yn'] == 0
