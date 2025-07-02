# Corrected Pipeline Execution Plan

## Critical Issues to Fix:

1. **Exposure Flag Issue**: 
   - exposure_or.parquet has correct OR logic (142,769 exposed = 57%)
   - patient_master.parquet has wrong AND logic (186 exposed = 0.07%)
   - Need to rebuild patient_master with correct exposure flag

2. **MC-SIMEX Parameters**:
   - Sensitivity = 0.78, Specificity = 0.71
   - These create attenuation bias (underestimate effects)
   - Adjustment factor = 3.038x

3. **Multiple Imputation**:
   - Must preserve correct exposure flag through imputation
   - Should impute on 142,769 exposed patients, not 186

## Corrected Execution Order:

### Step 1: Fix patient_master.parquet
```python
# Load correct exposure from exposure_or.parquet
exposure_df = pd.read_parquet('data_derived/exposure_or.parquet')
master_df = pd.read_parquet('data_derived/patient_master.parquet')

# Replace exposure_flag with correct OR logic
master_df = master_df.drop('exposure_flag', axis=1)
master_df = master_df.merge(
    exposure_df[['Patient_ID', 'exposure_flag', 'H1_normal_labs', 'H2_referral_loop', 'H3_drug_persistence']],
    on='Patient_ID',
    how='left'
)

# Verify
print(f"Corrected exposure: {master_df['exposure_flag'].sum()} ({master_df['exposure_flag'].mean()*100:.1f}%)")
# Should show ~142,769 (57%)

# Save corrected master
master_df.to_parquet('data_derived/patient_master_corrected.parquet')
```

### Step 2: Re-run Steps 6-15 with corrected data
- Step 6: Propensity scores with 57% exposed
- Step 7: Matching/weighting 
- Step 7a: MC-SIMEX adjustment (sensitivity=0.78, specificity=0.71)
- Steps 8-9: Balance checks
- Step 10: Multiple imputation on corrected data
- Steps 11-15: Causal estimation

### Expected Results:
- ~142,769 exposed patients (57%) instead of 186 (0.07%)
- Much higher statistical power
- Meaningful effect estimates
- MC-SIMEX will amplify effects by ~3x due to misclassification