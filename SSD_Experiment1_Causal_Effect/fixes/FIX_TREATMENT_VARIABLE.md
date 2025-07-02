# Critical Fix: Treatment Variable Issue

## Problem Identified
The causal analysis is using `ssd_flag` which has only 186 exposed patients (0.07%) due to AND logic combining all three hypotheses. This is incorrect based on the methodology blueprint.

## Current State
- `ssd_flag`: 186 patients (H1 AND H2 AND H3)
- `ssd_flag_adj`: 45,264 patients (MC-SIMEX adjusted)
- Individual hypotheses:
  - `H1_normal_labs`: 111,794 patients (44.7%)
  - `H2_referral_loop`: 1,655 patients (0.7%)
  - `H3_drug_persistence`: 55,695 patients (22.3%)

## Required Fix

### Option 1: Use MC-SIMEX Adjusted Flag
Change the causal pipeline to use `ssd_flag_adj` instead of `ssd_flag`:

```python
# In the causal estimation scripts, change:
treatment_col = "ssd_flag"  # OLD - only 186 patients
# To:
treatment_col = "ssd_flag_adj"  # NEW - 45,264 patients
```

### Option 2: Test Hypotheses Separately (Recommended)
Create three separate causal analyses:

```python
# Run causal analysis for each hypothesis
for hypothesis, treatment_col in [
    ("H1", "H1_normal_labs"),     # 111,794 patients
    ("H2", "H2_referral_loop"),    # 1,655 patients  
    ("H3", "H3_drug_persistence")  # 55,695 patients
]:
    run_causal_analysis(treatment_col=treatment_col, 
                       output_prefix=f"causal_results_{hypothesis}")
```

## Implementation Steps

1. **Update 06_causal_estimators.py** to accept treatment column as parameter
2. **Update imputed_causal_pipeline.py** to use correct treatment column
3. **Update notebook** to specify correct treatment column
4. **Re-run Steps 14-15** with correct treatment variable

## Expected Impact
- Proper statistical power for H1 and H3
- H2 may still have limited power due to only 1,655 exposed patients
- Results will align with the hypotheses as specified in the blueprint

## Validation
After fix, verify:
- Exposed count matches expected values
- Propensity scores have good overlap
- Causal estimates are reasonable (not 0.0 for TMLE)