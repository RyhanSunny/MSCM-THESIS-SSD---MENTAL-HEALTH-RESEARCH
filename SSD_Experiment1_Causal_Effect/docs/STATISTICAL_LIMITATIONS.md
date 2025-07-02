# Statistical Limitations Documentation

**Date**: July 1, 2025  
**Author**: Ryhan Suny  
**Version**: 1.1.0  

## Overview

This document outlines known statistical limitations in our causal analysis pipeline and provides recommendations for future improvements.

## 1. MC-SIMEX Variance Pooling Limitation

### Current Implementation
The MC-SIMEX (Monte Carlo Simulation-Extrapolation) correction for measurement error currently operates on a single dataset rather than being integrated with multiple imputation.

### Statistical Impact
- **Variance Underestimation**: The current approach does not account for between-imputation variance in the measurement error correction
- **Conservative Estimates**: Standard errors from MC-SIMEX may be underestimated by approximately 10-20% depending on the degree of missingness

### Technical Details
```
Current approach:
1. Run multiple imputation → m datasets
2. Select one imputed dataset
3. Apply MC-SIMEX correction
4. Use MC-SIMEX corrected estimates

Ideal approach (not implemented):
1. Run multiple imputation → m datasets
2. Apply MC-SIMEX to each imputed dataset
3. Pool MC-SIMEX results using Rubin's Rules
4. Account for two levels of variance
```

### Recommendation
Future work should implement a two-level variance approach:
- Level 1: MC-SIMEX simulation variance
- Level 2: Between-imputation variance
- Combined using modified Rubin's Rules for nested uncertainty

### References
- Carroll, R. J., Ruppert, D., Stefanski, L. A., & Crainiceanu, C. M. (2006). *Measurement error in nonlinear models*
- Blackwell, M., Honaker, J., & King, G. (2017). "A unified approach to measurement error and missing data"

## 2. Weight Truncation Not Yet Implemented

### Current State
Propensity score weights are not truncated, which may lead to instability with extreme weights.

### Recommendation
Implement Crump et al. (2009) trimming rule: exclude units with weights > 10.

## 3. Sensitivity Analysis Framework

### Current State
Limited sensitivity analyses for unmeasured confounding.

### Recommendation
Implement E-value calculations and formal sensitivity analysis framework.

## 4. H2 Hypothesis Implementation Limitation

### Current Implementation
The H2 hypothesis as specified by Dr. Karim requires tracking patients with "unresolved" specialist referrals (NYD → referred to specialist → "no diagnosis of that body part"). However, the CPCSSN referral table lacks Status/Resolution fields necessary to verify whether referrals were resolved.

### Statistical Impact
- **Measurement Error**: Current implementation counts ANY specialist referrals with symptom codes (ICD-9: 780-799), not specifically unresolved ones
- **Sample Size**: Only 1,536 patients (0.6%) meet current criteria vs potentially 500-2,000 with enhanced criteria
- **Validity**: May include resolved referrals, diluting the effect estimate

### Current Workaround
Three-tier proxy implementation available:
- **Tier 1 (Basic)**: ≥2 referrals with symptom codes (current implementation)
- **Tier 2 (Enhanced)**: NYD codes (780-799) + ≥2 specialist referrals  
- **Tier 3 (Full Proxy)**: NYD + ≥3 normal labs + repeated same-specialty referrals

### Recommendation
1. Report all three tiers in sensitivity analysis
2. Use Tier 2 as primary analysis (balances validity and sample size)
3. Advocate for Status/Resolution fields in future CPCSSN updates
4. Consider linking to specialist EMR data for outcome tracking

### References
- Dr. Karim Keshavjee: Discussion on potential causal chains in SSD (personal communication)
- DSM-5-TR (2022): Somatic Symptom and Related Disorders criteria

## Version Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-06-30 | Initial documentation of MC-SIMEX limitation |
| 1.1.0 | 2025-07-01 | Added H2 hypothesis implementation limitation |

---
*Note: This document should be updated as limitations are addressed or new ones are discovered.*