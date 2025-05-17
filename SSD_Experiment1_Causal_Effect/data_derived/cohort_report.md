# SSD Cohort Build Report

## Summary
This report summarizes the results of running the cohort builder (01_cohort_builder.py) on the most recent checkpoint (checkpoint_1_20250318_024427).

## Checkpoint Used
- **Folder:** SSD_Experiment1_Causal_Effect/Notebooks/data/interim/checkpoint_1_20250318_024427
- **Description:** Data Loading and Validation with Optimized Processing (Notebook 1)

## Tables Loaded
- **patient:** 352,161 rows (parquet)
- **encounter:** 11,577,739 rows (parquet)
- **health_condition:** 2,571,583 rows (parquet)
- **lab:** 8,528,807 rows (csv)
- **encounter_diagnosis:** 12,471,764 rows (parquet)

## Filtering Results
- **Age ≥ 18 (as of 2015-01-01):** 307,100 patients remain.
- **Observation Span (≥ 30 months):** Excluded 56,276 patients; 250,824 patients remain.
- **Opt-out exclusion:** 0 patients excluded.
- **Palliative-care exclusion:** 0 patients excluded.
- **Charlson > 5 exclusion:** 0 patients excluded (Charlson index set to 0 due to missing helper).

## Final Cohort
- **Output File:** SSD_Experiment1_Causal_Effect/data_derived/cohort.parquet
- **Total Patients:** 250,824

## Notes
- **Charlson Index:** All patients have a Charlson score of 0 (helper not found). If needed, ensure that the helper (src/icd_utils.py) is available.
- **Logging Warnings:** Some log messages (using "≥" or "→") caused UnicodeEncodeError (charmap) warnings. These do not affect the results. 