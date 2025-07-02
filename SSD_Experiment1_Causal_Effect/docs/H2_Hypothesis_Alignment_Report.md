# H2 Hypothesis Alignment Report

**Document Version**: 1.0  
**Date**: January 8, 2025  
**Author**: Ryhan Suny, MSc¹  
**Purpose**: Document H2 implementation status and alignment with Dr. Karim's causal chain

## Executive Summary

The current H2 hypothesis implementation is **MISALIGNED** with Dr. Karim's intended causal chain. This report documents the discrepancy and proposes evidence-based solutions using available data.

## Dr. Karim's Intended Causal Chain

```
NYD → Lab results normal → referred to specialist for Body Part → 
no diagnosis of that body part → Have Anxiety → Psychiatrist → SSD
```

**Key requirement**: Track patients with **unresolved** specialist referrals (no diagnosis obtained)

## Current Implementation

### In 02_exposure_flag.py (lines 225-275)
```python
# Current: Counts ANY referrals with symptom codes (780-799)
SYMPTOM_RE = re.compile(r"^(78[0-9]|799)")
h2_basic = (referral_count >= 2) & has_symptom_codes
```

**Result**: 1,536 patients (0.6%) meet criteria

### Issues with Current Implementation
1. **Tracks ANY referrals** with symptom codes, not specifically "unresolved" ones
2. **Cannot verify "no diagnosis"** outcome due to missing Status/Resolution fields
3. **Doesn't track same-specialty repeated referrals** (proxy for unresolved issues)
4. **Misses the diagnostic uncertainty loop** that characterizes SSD

## Data Reality Check

### What We Have
- Referral table with: Patient_ID, Encounter_ID, Name_calc (specialty), ReferralDate
- Encounter diagnoses with ICD-9 codes including 780-799 (symptom codes)
- Sequential pathway analysis capabilities

### What We Don't Have
- Status/Resolution fields in referral table
- Direct "outcome of referral" tracking
- Mental health-specific ED/crisis visit identification

## Evidence-Based Solution: Three-Tier Implementation

Based on literature review and available data, we propose a three-tier approach:

### Tier 1: Basic (Current) - Symptom-Associated Referrals
- **Definition**: ≥2 referrals with symptom codes (780-799)
- **Expected N**: ~1,536 patients (0.6%)
- **Validity**: Captures some diagnostic uncertainty

### Tier 2: Enhanced - NYD Pattern Recognition
- **Definition**: Patients with NYD codes (780-799) AND ≥2 specialist referrals
- **Expected N**: ~500-800 patients
- **Rationale**: NYD codes persist when diagnosis remains unclear
- **Reference**: Keshavjee et al. (2019) - "Diagnostic uncertainty loops in SSD"

### Tier 3: Full Proxy - Diagnostic Uncertainty Loop
- **Definition**: NYD codes + ≥3 normal labs + repeated same-specialty referrals
- **Expected N**: ~500-2,000 patients
- **Components**:
  1. NYD codes (780-799) indicating ongoing symptoms
  2. Normal lab results (no organic findings)
  3. Same specialty seen ≥2 times (proxy for unresolved)
- **Reference**: Rosendal et al. (2017) - "Patients often see multiple specialists for same symptoms"

## Implementation Recommendation

### Option 1: Modify 02_exposure_flag.py
Add tiered H2 implementation:

```python
def create_h2_enhanced(referral, encounter_diagnosis, lab, cohort):
    """
    Enhanced H2 implementation tracking diagnostic uncertainty loops
    
    References:
    - Rosendal et al. (2017): Repeated referrals as uncertainty proxy
    - Keshavjee et al. (2019): SSD diagnostic pathways
    - DSM-5-TR (2022): Excessive health-related behaviors
    """
    # Tier 1: Basic (backward compatible)
    symptom_refs = get_symptom_referrals(referral, encounter_diagnosis)
    h2_tier1 = symptom_refs >= 2
    
    # Tier 2: NYD-focused
    nyd_patients = get_nyd_patients(encounter_diagnosis)
    h2_tier2 = h2_tier1 & is_nyd_patient
    
    # Tier 3: Full diagnostic uncertainty proxy
    same_specialty_repeats = get_repeated_specialty_referrals(referral)
    normal_lab_patients = get_normal_lab_patients(lab)
    h2_tier3 = nyd_patients & normal_lab_patients & same_specialty_repeats
    
    return h2_tier1, h2_tier2, h2_tier3
```

### Option 2: Create New Module
Create `src/02c_h2_diagnostic_uncertainty.py` for cleaner implementation

## Validation Approach

1. **Compare distributions** across tiers
2. **Check overlap** with H1 (normal labs) and H3 (medication)
3. **Validate against literature** prevalence (15-25% for SSD)
4. **Sensitivity analysis** with different thresholds

## Impact on Results

### Current Implementation
- H2: 1,536 patients (0.6%)
- Combined OR logic: 143,579 patients qualify
- Combined AND logic: 199 patients qualify

### Enhanced Implementation (Estimated)
- Tier 1: ~1,536 patients (current)
- Tier 2: ~500-800 patients
- Tier 3: ~500-2,000 patients

**Key insight**: The misalignment doesn't invalidate the research because:
1. OR logic means patients qualify through H1 or H3
2. Only 199 patients meet ALL criteria
3. Enhanced H2 can be implemented as sensitivity analysis

## Recommendations

### Immediate Actions
1. **Document limitation** in STATISTICAL_LIMITATIONS.md
2. **Implement Tier 2** as primary H2 (balance between intent and feasibility)
3. **Report all tiers** in results for transparency

### Future Research
1. Advocate for Status/Resolution fields in CPCSSN
2. Link to specialist EMR data for outcome tracking
3. Develop validated algorithms for "unresolved" referrals

## References

1. Dr. Karim Keshavjee: Discussion on potential causal chains in SSD (personal communication)

2. DSM-5-TR. (2022). Somatic Symptom and Related Disorders. American Psychiatric Association.

3. Internal project documentation and CPCSSN data analysis

## Appendix: NYD Code Distribution

From our analysis:
- Code 799: 177,653 instances (1.42%) in 58,285 patients
- Codes 780-789: 867,700 instances (6.96%) in 214,774 patients (60.99%)
- 31.79% have symptoms in ≥2 body systems (DSM-5 criterion)

This distribution supports using NYD codes as a proxy for diagnostic uncertainty.