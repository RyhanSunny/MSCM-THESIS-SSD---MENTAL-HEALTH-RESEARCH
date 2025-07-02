# Next Steps Summary - SSD Pipeline Fixes
*Created: July 2, 2025*

## Quick Summary
The pipeline completed but with a critical flaw: only 186 exposed patients instead of tens of thousands. This must be fixed immediately.

## Files Created for You

### 1. Analysis Documents
- `CRITICAL_PIPELINE_ISSUES_ANALYSIS.md` - Detailed analysis of all issues
- `INTEGRATION_PLAN_FOR_FIXES.md` - Step-by-step fix plan
- `docs/TRANSPORT_WEIGHTS_DOCUMENTATION.md` - Explanation of transport weights

### 2. Fixed Scripts
- `src/imputed_causal_pipeline_progress_FIXED.py` - Fixes format string error
- `src/imputed_causal_wrapper_multi_hypothesis.py` - Tests each hypothesis separately
- `fixes/CORRECTED_PLACEBO_TEST_CELL.py` - Corrected placebo test code

### 3. Implementation Tools
- `QUICK_FIX_IMPLEMENTATION.py` - Run this to apply fixes automatically
- `fixes/FIX_TREATMENT_VARIABLE.md` - Detailed treatment variable fix

## Immediate Actions (In Order)

### 1. Apply Quick Fixes (5 minutes)
```bash
python QUICK_FIX_IMPLEMENTATION.py
```
This will:
- Backup original files
- Fix the format string error
- Update wrapper to use ssd_flag_adj
- Verify your data files

### 2. Re-run Step 2 in Notebook (10 minutes)
Change the notebook Step 2 from:
```python
# OLD - gives only 186 exposed
result = run_pipeline_script("02_exposure_flag.py", args="--logic and")
```
To:
```python
# NEW - generates both OR and AND versions
result = run_pipeline_script("02_exposure_flag.py", args="--logic both")
```

### 3. Re-run Critical Steps (2-3 hours)
Re-run these steps in order:
- Step 8: Pre-Imputation Master Assembly
- Step 13: PS Matching (should see better balance)
- Step 14: Causal Estimation (will use fixed wrapper)
- Step 15: Rubin's Pooling

### 4. Verify Success
Check that:
- ✓ Exposed count is ~45,000 (not 186)
- ✓ PS matching achieves SMD < 0.1
- ✓ TMLE gives non-zero estimates
- ✓ Results vary between H1, H2, H3

## Alternative: Test Hypotheses Separately (Recommended)

Instead of using combined exposure, test each hypothesis as intended:

```python
# In a new notebook cell after Step 14:
from src.imputed_causal_wrapper_multi_hypothesis import run_multi_hypothesis_estimation

# Test all hypotheses on first imputation
df = pd.read_parquet("data_derived/imputed_master/master_imputed_1.parquet")
results = run_multi_hypothesis_estimation(df, imp_num=1)

# View results
for hyp, res in results['hypotheses'].items():
    if res.get('n_treated'):
        print(f"\n{hyp}: {res['n_treated']} treated")
        for est in res.get('estimates', []):
            print(f"  {est['method']}: {est.get('estimate', 'Failed')}")
```

## Why This Matters

### Current State (Wrong)
- Testing: H1 AND H2 AND H3 (all must be true)
- Result: 186 patients (0.07%)
- Power: Essentially zero
- Estimates: TMLE = 0.0

### After Fix (Correct)
- Testing: H1, H2, H3 separately OR combined with OR logic
- Result: 
  - H1: 111,794 patients (44.7%)
  - H2: 1,655 patients (0.7%)
  - H3: 55,695 patients (22.3%)
  - Adjusted: 45,264 patients (18.1%)
- Power: Adequate for H1 and H3
- Estimates: Meaningful causal effects

## Risk Mitigation

1. **Backup Current Results**
   ```bash
   cp -r results results_backup_20250702
   ```

2. **Test on Single Imputation First**
   Before running all 30 imputations, test with just one

3. **Monitor Progress**
   The fixed pipeline shows progress and time estimates

## Support

If issues arise:
1. Check error files in `results/imputed_causal_results/causal_error_imp*.txt`
2. Verify treatment columns exist in your data
3. Ensure sufficient memory for imputation (needs ~16GB)

## Expected Timeline

- Quick fixes: 5 minutes
- Re-run exposure generation: 10 minutes
- Re-run PS matching: 15 minutes
- Re-run causal estimation (30 imputations): 90-120 minutes
- Total: ~2.5 hours

After these fixes, your causal estimates will be valid and aligned with the methodology blueprint.