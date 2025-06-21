# MC-SIMEX Status

**Date**: June 21, 2025  
**Status**: DISABLED  
**Author**: Ryhan Suny

## Why MC-SIMEX is Currently Disabled

MC-SIMEX (Misclassification Simulation-Extrapolation) is implemented in our pipeline but intentionally disabled because:

1. **No Clinical Validation Data**: The sensitivity and specificity parameters are marked as `NEEDS_CLINICAL_VALIDATION` in `config.yaml`
2. **Conservative Approach**: Without validated parameters, we present uncorrected estimates which are biased toward the null (underestimate true effects)
3. **Ready for Future Use**: Once clinical chart review validates our SSD phenotype algorithm, we can:
   - Update `mc_simex.sensitivity` and `mc_simex.specificity` with real values
   - Set `use_bias_corrected_flag: true`
   - Re-run the pipeline to get bias-corrected estimates

## Impact on Results

Current estimates (IRR 1.35-1.50) likely underestimate true effects by 10-20% due to exposure misclassification.

## How to Enable When Ready

1. Complete clinical validation (see `Notebooks/SSD_Phenotype_Validation.ipynb`)
2. Update `config/config.yaml`:
   ```yaml
   mc_simex:
     enabled: true
     sensitivity: 0.82  # Your validated value
     specificity: 0.82  # Your validated value
     use_bias_corrected_flag: true
   ```
3. Run: `make misclassification mc-simex-merge`
4. Re-run analyses with corrected exposure 