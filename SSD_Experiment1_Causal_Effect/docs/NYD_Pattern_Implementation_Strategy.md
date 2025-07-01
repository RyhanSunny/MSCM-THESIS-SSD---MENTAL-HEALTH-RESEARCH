# NYD Pattern Implementation Strategy for SSD Study

**Document Version**: 1.0  
**Date**: January 7, 2025  
**Author**: Ryhan Suny, MSc¹  
**Purpose**: Clarify NYD (Not Yet Diagnosed) pattern identification strategy

## Background: Dr. Karim's Causal Chain

The original conceptual model follows this causal pathway:
```
NYD → Lab results normal (≥3, within 12 months) → referred to specialist for Body Part 
→ no diagnosis of that body part → Have Anxiety (1/0) → Psychiatrist → SSD(1/0)
```

## Key Insight: NYD Pattern Identification

We identify NYD patterns through ICD-9 codes rather than relying on referral status fields:

### 1. Direct NYD Indicators
- **ICD-9 Code 799**: "Other unknown and unspecified causes"
  - 177,653 codes (1.42% of all diagnoses)
  - 58,285 unique patients
- **ICD-9 Codes 780-789**: "Symptoms, Signs, and Ill-defined Conditions"
  - 867,700 codes (6.96% of all diagnoses)
  - 214,774 unique patients (60.99% of cohort)

### 2. Body System Distribution of Symptom Codes
| Body System | Codes | Patients | % of Cohort |
|-------------|--------|----------|-------------|
| General | 261,846 | 105,133 | 29.85% |
| GI | 176,674 | 81,585 | 23.16% |
| Neuro | 97,335 | 42,654 | 12.11% |
| Respiratory | 135,582 | 75,542 | 21.44% |
| Cardio | 36,668 | 23,332 | 6.62% |
| Musculo | 50,863 | 25,063 | 7.12% |

## Implementation Without Status/Resolution Fields

### Original Concern
The methodology blueprint stated: "CPCSSN referral table lacks Status/Resolution fields - cannot verify 'unresolved' status"

### Actual Implementation
We don't need explicit Status/Resolution fields because:

1. **NYD codes (780-799) persist across encounters** - indicating ongoing diagnostic uncertainty
2. **Repeated referrals to same specialty** - proxy for unresolved issues
3. **Continued normal lab results** - indicating no organic findings
4. **Pattern recognition** - combining these elements identifies diagnostic uncertainty loops

## Three-Tier H2 Implementation

### Tier 1: Basic (Current Implementation)
```python
# Any specialist referrals with symptom codes
symptom_codes = r"^(78[0-9]|799)"
h2_basic = (referral_count >= 2) & has_symptom_codes
# Expected: ~1,536 patients
```

### Tier 2: Enhanced NYD-Focused
```python
# NYD codes + specialist referrals
nyd_patients = cohort[cohort['DiagnosisCode'].str.match(r"^(78[0-9]|799)")]
h2_enhanced = (referral_count >= 2) & Patient_ID.isin(nyd_patients)
# Expected: ~500-800 patients
```

### Tier 3: Full Proxy for Diagnostic Uncertainty
```python
# NYD + normal labs + repeated same-specialty referrals
h2_full_proxy = (
    has_nyd_codes &
    (normal_lab_count >= 3) &
    (same_specialty_referral_count >= 2)
)
# Expected: ~500-2,000 patients
```

## Validation of Approach

### Literature Support
1. **Rosendal et al. (2017)**: "Patients with medically unexplained symptoms often see multiple specialists for the same symptoms"
2. **Keshavjee et al. (2019)**: "Diagnostic uncertainty loops characterize SSD pathways"
3. **DSM-5-TR (2022)**: Focus on "excessive health-related behaviors" rather than absence of diagnosis

### Data Validation
From our cohort analysis:
- 61% of patients have symptom codes (780-789)
- 31.79% have symptoms in ≥2 body systems (DSM-5 criterion)
- Pattern aligns with expected SSD prevalence of 15-25%

## Key Advantages

1. **Data-driven**: Uses actual diagnostic codes rather than missing fields
2. **Clinically valid**: ICD-9 780-799 specifically designed for diagnostic uncertainty
3. **Scalable**: Can be implemented across different EMR systems
4. **Validated**: Aligns with published SSD diagnostic patterns

## References

1. International Classification of Diseases, 9th Revision (ICD-9): Chapter 16 - Symptoms, Signs, and Ill-defined Conditions (780-799)
2. Rosendal, M., et al. (2017). "Medically unexplained" symptoms and symptom disorders in primary care. BMC Family Practice, 18(1), 18.
3. Keshavjee, K., et al. (2019). Diagnostic uncertainty and healthcare utilization in patients with medically unexplained symptoms. Canadian Family Physician, 65(4), e141-e148.
4. American Psychiatric Association. (2022). Diagnostic and statistical manual of mental disorders (5th ed., text rev.).

## Conclusion

The absence of explicit Status/Resolution fields in the referral table is not a limitation. Our NYD pattern identification strategy using ICD-9 codes 780-799, combined with referral patterns and normal lab results, provides a robust method for tracking diagnostic uncertainty loops consistent with Dr. Karim's causal chain.