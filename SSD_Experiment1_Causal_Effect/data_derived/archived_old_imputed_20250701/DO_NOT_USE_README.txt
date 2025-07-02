ARCHIVED: 2025-07-01 13:49

This directory contains the OLD/INCOMPLETE imputation approach and should NOT be used.

Issues with this old approach:
- Only 5 imputations (not the required 30)
- Only imputed 19 cohort columns (missing 54 other important variables)
- Used outdated pipeline that imputed BEFORE merging all features

The CORRECT imputed data is in: data_derived/imputed_master/
- Contains 30 imputations
- Imputes full 73-column master table
- Uses corrected pipeline that merges all features BEFORE imputation

DO NOT USE THESE FILES FOR ANALYSIS!
