# Critical Pipeline Issues Analysis
*Based on actual execution logs and data inspection - July 2, 2025*

## Executive Summary
The pipeline has several critical issues that violate the study design specified in the methodology blueprint. The most severe is using AND logic for exposure definition, resulting in only 186 exposed patients instead of the expected tens of thousands.

## 1. CRITICAL: Wrong Exposure Definition ❌

### Issue
The pipeline uses `ssd_flag` with AND logic, requiring patients to meet ALL three criteria:
- H1: ≥3 normal labs AND
- H2: ≥2 specialist referrals AND  
- H3: ≥180 days psychotropic medication

### Impact
- **Expected exposed patients**: ~140,000 (OR logic) or test separately
- **Actual exposed patients**: 186 (0.07% of cohort)
- **Statistical power**: Essentially zero
- **TMLE estimate**: 0.0 (likely due to no variation)

### Root Cause
In Step 2, the notebook runs:
```python
run_pipeline_script("02_exposure_flag.py", args="--logic and")
```

### Fix Required
Either:
1. Change to `--logic or` to get ~140,000 exposed patients
2. Test each hypothesis separately using H1_normal_labs, H2_referral_loop, H3_drug_persistence
3. Use the MC-SIMEX adjusted flag: `ssd_flag_adj` (45,264 patients)

## 2. Incomplete Imputation ⚠️

### Issue
After 6.5 hours of imputation, the log shows:
```
INFO: Remaining missing values: 70764
```

### Impact
- Imputation did not fully complete
- May affect downstream analyses
- Violates assumption of no missing data

### Fix Required
- Investigate why imputation failed to complete
- Consider using more robust imputation method
- May need to exclude variables with extreme missingness

## 3. Covariate Balance Failure ❌

### Issue
After PS matching:
```
Maximum SMD after weighting: 0.699
```

### Impact
- SMD > 0.1 indicates poor balance
- 0.699 is extremely high, suggesting matching failed
- Causal estimates may be biased

### Root Cause
With only 186 treated units, finding good matches is nearly impossible

## 4. Script Invocation Errors ⚠️

### Issues Found
1. **14_placebo_tests.py**: Called with wrong arguments
   - Expected: `--n-iterations 100 --treatment-col ssd_flag`
   - Provided: `--data-dir data_derived/imputed_master --config config/analysis_config.json`

2. **Format string error** in imputed_causal_pipeline_progress.py:
   ```python
   print(f"    - {method}: {estimate:.4f if isinstance(estimate, (int, float)) else estimate}")
   ```
   Should be:
   ```python
   print(f"    - {method}: {estimate:.4f}" if isinstance(estimate, (int, float)) else f"    - {method}: {estimate}")
   ```

## 5. Hypothesis Misalignment ❌

### Issue
The methodology blueprint clearly states H1, H2, and H3 are separate hypotheses to be tested independently, but the pipeline combines them.

### Expected Analysis Structure
```
H1 Analysis: Normal labs → Healthcare utilization
H2 Analysis: Referral loops → Healthcare utilization  
H3 Analysis: Drug persistence → Healthcare utilization
H4 Analysis: Mediation through SSDSI
H5 Analysis: Effect modification by subgroups
```

### Actual Analysis
Single analysis with AND(H1, H2, H3) → Healthcare utilization

## Recommendations

### Immediate Actions
1. **Re-run Step 2** with `--logic or` or modify to test hypotheses separately
2. **Fix format string** in imputed_causal_pipeline_progress.py
3. **Update causal pipeline** to use correct treatment variable
4. **Re-run Steps 13-15** after fixing exposure definition

### Code Changes Needed

#### In notebook Step 2:
```python
# Option 1: Use OR logic
result = run_pipeline_script("02_exposure_flag.py", 
                           args="--logic or",
                           description="Exposure Flag Generation (OR logic)")

# Option 2: Generate both for separate testing
result = run_pipeline_script("02_exposure_flag.py", 
                           args="--logic both",
                           description="Exposure Flag Generation (Both logics)")
```

#### In causal estimation:
```python
# Test each hypothesis
for hyp, treatment in [("H1", "H1_normal_labs"), 
                       ("H2", "H2_referral_loop"), 
                       ("H3", "H3_drug_persistence")]:
    run_causal_analysis(treatment_col=treatment, output_prefix=hyp)
```

## Validation After Fixes

Confirm:
1. Exposed count matches blueprint expectations
2. PS matching achieves SMD < 0.1 for all covariates
3. TMLE produces non-zero estimates
4. Placebo tests run successfully
5. Each hypothesis is tested separately

## References
- Methodology Blueprint: Lines 11-13 specify separate hypothesis testing
- CLAUDE.md: Emphasizes checking implementation thoroughly (line 90)
- Actual data: data_derived/patient_master.parquet contains all necessary columns