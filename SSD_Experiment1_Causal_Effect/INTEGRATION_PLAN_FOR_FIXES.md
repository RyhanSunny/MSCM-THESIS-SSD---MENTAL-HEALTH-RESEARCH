# Integration Plan for Pipeline Fixes
*Created: July 2, 2025*

## Overview
This document provides a step-by-step plan to fix the critical issues identified in the SSD causal analysis pipeline.

## Issues Summary

### 1. **Critical: Wrong Treatment Variable** ❌
- Current: `exposure_flag` with AND logic (186 patients)
- Should be: Either OR logic or test H1/H2/H3 separately
- Files affected:
  - `src/imputed_causal_wrapper.py` (line 23)
  - Notebook Step 2 (exposure generation)
  - All downstream analyses

### 2. **Format String Error** ❌
- File: `src/imputed_causal_pipeline_progress.py` (line 81)
- Already fixed in: `src/imputed_causal_pipeline_progress_FIXED.py`

### 3. **PS Matching Balance Issue** ⚠️
- Max SMD: 0.699 (should be <0.1)
- Root cause: Only 186 treated units makes matching impossible

### 4. **Incomplete Imputation** ⚠️
- 70,764 missing values remain after imputation
- May need investigation but not critical if missingness is in non-essential variables

## Step-by-Step Integration Plan

### Phase 1: Fix Exposure Definition (CRITICAL)

#### Step 1.1: Re-run Exposure Generation
```bash
# In notebook, modify Step 2:
result = run_pipeline_script("02_exposure_flag.py",
                           args="--logic both",  # Generate both OR and AND versions
                           description="Exposure Flag Generation (Both Logics)")
```

#### Step 1.2: Update Wrapper to Use Correct Treatment
Create `src/imputed_causal_wrapper_FIXED.py`:
```python
# Line 23, change from:
treatment_col = 'exposure_flag'  # OLD - AND logic
# To one of:
treatment_col = 'ssd_flag_adj'   # Option 1: MC-SIMEX adjusted (45,264 patients)
treatment_col = 'H1_normal_labs'  # Option 2: Test H1 separately (111,794 patients)
```

#### Step 1.3: Create Multi-Hypothesis Wrapper
Create `src/imputed_causal_wrapper_multi_hypothesis.py`:
```python
def run_causal_estimation_multi_hypothesis(df, imp_num):
    """Run causal estimation for each hypothesis separately"""
    results = {}
    
    for hypothesis, treatment_col in [
        ('H1', 'H1_normal_labs'),
        ('H2', 'H2_referral_loop'), 
        ('H3', 'H3_drug_persistence'),
        ('Combined_OR', 'exposure_flag'),  # If using OR logic
        ('Adjusted', 'ssd_flag_adj')
    ]:
        if treatment_col in df.columns:
            results[hypothesis] = run_causal_estimation_on_imputation(
                df, imp_num, treatment_col=treatment_col
            )
    
    return results
```

### Phase 2: Fix Script Errors

#### Step 2.1: Replace Pipeline Progress Script
```bash
# Backup original
cp src/imputed_causal_pipeline_progress.py src/imputed_causal_pipeline_progress.BACKUP

# Replace with fixed version
cp src/imputed_causal_pipeline_progress_FIXED.py src/imputed_causal_pipeline_progress.py
```

#### Step 2.2: Fix Placebo Test Arguments
In notebook Step 14.1-14.5, use the corrected cell from `/fixes/CORRECTED_PLACEBO_TEST_CELL.py`

### Phase 3: Re-run Affected Steps

#### Step 3.1: Re-run from Step 8 (Pre-Imputation Master)
Since exposure definition affects the master table:
1. Step 8: Pre-Imputation Master Assembly
2. Step 9: Multiple Imputation (if exposure is used in imputation model)
3. Step 10: Master Patient Table

#### Step 3.2: Re-run Causal Analysis Chain
1. Step 12: Sequential Analysis
2. Step 13: PS Matching (should see better balance with more exposed)
3. Step 14: Causal Estimation on Imputations
4. Step 14.1-14.5: Additional Analyses
5. Step 15: Rubin's Pooling

### Phase 4: Validation

#### Step 4.1: Verify Exposure Counts
```python
# Check in notebook after Step 2
import pandas as pd
exp = pd.read_parquet('data_derived/exposure_or.parquet')
print(f"OR logic exposed: {exp['exposure_flag'].sum()}")
print(f"H1 exposed: {exp['H1_normal_labs'].sum()}")
print(f"H2 exposed: {exp['H2_referral_loop'].sum()}")
print(f"H3 exposed: {exp['H3_drug_persistence'].sum()}")
```

#### Step 4.2: Verify PS Balance
After Step 13, check:
```python
results = pd.read_json('results/ps_matching_results.json')
print(f"Max SMD after matching: {results['max_smd_after']}")
# Should be < 0.1
```

#### Step 4.3: Verify Causal Estimates
After Step 15, check:
- TMLE should not be 0.0
- Confidence intervals should be reasonable
- Results should differ between hypotheses

## Implementation Priority

1. **IMMEDIATE**: Fix treatment variable (Phase 1)
2. **HIGH**: Fix script errors (Phase 2)
3. **MEDIUM**: Re-run pipeline (Phase 3)
4. **LOW**: Address imputation completeness

## Expected Outcomes After Fixes

1. **Exposure Counts**:
   - H1: ~111,794 exposed (44.7%)
   - H2: ~1,655 exposed (0.7%)
   - H3: ~55,695 exposed (22.3%)
   - OR logic: ~140,000 exposed
   - Adjusted: ~45,264 exposed

2. **PS Matching**:
   - Max SMD < 0.1 for H1 and H3
   - H2 may still have balance issues due to small N

3. **Causal Estimates**:
   - Non-zero TMLE estimates
   - Reasonable confidence intervals
   - Different effects for each hypothesis

## Quick Start Commands

```bash
# 1. Fix the wrapper
cp src/imputed_causal_wrapper.py src/imputed_causal_wrapper.BACKUP
# Edit line 23 to use 'ssd_flag_adj' or create multi-hypothesis version

# 2. Fix the pipeline progress
cp src/imputed_causal_pipeline_progress_FIXED.py src/imputed_causal_pipeline_progress.py

# 3. Re-run from notebook
# Start from Step 2 with --logic both
```

## Monitoring Progress

Track these metrics:
1. Exposed count matches expectations
2. PS balance achieved (SMD < 0.1)
3. All 30 imputations complete successfully
4. Pooled estimates have reasonable CIs
5. Placebo tests pass

## Risk Mitigation

- **Backup all current results** before re-running
- **Test on single imputation** before running all 30
- **Monitor memory usage** during imputation
- **Save intermediate results** frequently

This plan addresses all critical issues while respecting the CLAUDE.md requirements for thorough checking and the methodology blueprint's hypothesis structure.