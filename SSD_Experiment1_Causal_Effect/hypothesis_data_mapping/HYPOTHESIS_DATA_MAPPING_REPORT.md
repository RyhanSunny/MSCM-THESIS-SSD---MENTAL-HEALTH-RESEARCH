# Hypothesis-to-Data Mapping Report
**Author:** Ryhan Suny  
**Date:** 2025-05-26  
**Institution:** Toronto Metropolitan University  
**Research Team:** Car4Mind  

## Executive Summary

This report provides a comprehensive analysis of data availability for testing the 9 hypotheses in the SSD causal effect study. We analyzed the checkpoint data (`checkpoint_1_20250318_024427`) containing 11 tables with over 40 million records to assess feasibility of hypothesis testing.

### Key Findings:
- **2 hypotheses (H8, H9)** have 100% data availability
- **3 hypotheses (H1, H3, H6)** have 66-70% data availability  
- **4 hypotheses (H2, H4, H5, H7)** have <50% data availability due to critical missing data

## Data Inventory Summary

| Table | Records | Key Finding |
|-------|---------|------------|
| patient | 352,161 | Basic demographics available |
| patient_demographic | 352,220 | **CRITICAL: ResidencePostalCode 0% complete** |
| encounter | 11,577,739 | Excellent temporal coverage |
| encounter_diagnosis | 12,471,764 | ICD-9 codes available |
| health_condition | 2,571,583 | 91,409 patients with MH diagnoses |
| lab | 8,528,807 | Normal range detection implemented |
| medication | 7,706,628 | ATC codes and duration available |
| referral | 1,141,061 | Specialty information available |
| family_history | 325,202 | Family health data |
| medical_procedure | 1,203,002 | Procedure tracking |
| risk_factor | 603,298 | Risk factor documentation |

## Hypothesis-by-Hypothesis Analysis

### ✅ H1: SSD Patterns → Healthcare Utilization (70% Available)
**Data Required:**
- Lab results with normal ranges ✓ PARTIAL
- Referral data with specialties ✓ YES  
- Medication data with duration ✓ YES
- Encounter utilization metrics ✓ YES
- ED visit identification ⚠️ NEEDS DERIVATION

**Status:** FEASIBLE with ED derivation

### ⚠️ H2: Severity → Healthcare Costs (50% Available)
**Data Required:**
- Healthcare cost data ✗ NO
- Billing/claims data ✗ NO
- Resource utilization ✓ YES (proxy)
- Severity index ✓ YES (autoencoder)

**Status:** PARTIALLY FEASIBLE using utilization as cost proxy

### ✅ H3: Provider Changes (67% Available)
**Data Required:**
- Provider IDs ✓ YES (56.5% see multiple providers)
- Temporal sequences ✓ YES
- Provider specialty ✗ NO (use referral specialty as proxy)

**Status:** FEASIBLE with limitations

### ❌ H4: Psychological Mediation (0% Available)
**Data Required:**
- Mental health diagnoses ✓ YES! (91,409 patients with ICD-9 290-319)
- Anxiety/depression codes ✓ YES! (35,425 anxiety, 36,807 depression)
- Psychological assessments ✗ NO

**Status:** FEASIBLE (updated from initial assessment)

### ⚠️ H5: Health Anxiety Mediation (17% Available)
**Data Required:**
- Health anxiety indicators ✗ NO
- Frequent health concerns ✓ PARTIAL (visit frequency)
- Hypochondriasis codes ⚠️ NEEDS CHECK

**Status:** LIMITED FEASIBILITY

### ✅ H6: Diagnostic Uncertainty (67% Available)
**Data Required:**
- NYD codes ✗ NO (0 found - config mismatch?)
- Diagnostic changes ✓ YES (trackable)
- Provider notes ✗ NO

**Status:** PARTIALLY FEASIBLE

### ❌ H7: Socioeconomic Moderation (17% Available)
**Data Required:**
- Postal codes ✗ NO (0% complete)
- Income/education ✓ PARTIAL (1.4% education, 7.6% occupation)
- Deprivation index ✗ NO (requires postal codes)

**Status:** NOT FEASIBLE without postal codes

### ✅ H8: Multimorbidity Patterns (100% Available)
**Data Required:**
- Charlson comorbidity ✓ YES (calculable)
- Chronic conditions ✓ YES (2.6M records)
- Disease clustering ✓ YES

**Status:** FULLY FEASIBLE

### ✅ H9: Temporal Utilization (100% Available)
**Data Required:**
- Longitudinal data ✓ YES (2010-2015 primary)
- Pre/post periods ✓ YES
- Time series ✓ YES

**Status:** FULLY FEASIBLE

## Critical Data Gaps

### 1. Postal Codes (Impact: H7)
- Field exists but 0% populated
- **Prevents:** Pampalon deprivation index calculation
- **Alternative:** Use available SES indicators (limited coverage)

### 2. Cost/Billing Data (Impact: H2)
- No cost or claims tables
- **Alternative:** Develop utilization-based cost proxy using:
  - Encounter counts × average costs
  - Procedure counts × procedure costs
  - Medication days × drug costs

### 3. NYD Codes (Impact: H6)
- Zero records found despite config definition
- **Action needed:** Verify ICD code format/version

### 4. Emergency Department Visits
- Not explicitly coded in EncounterType
- Found 1,918 "ER Visit" types
- Found 5,279 encounters with ED keywords
- **Action needed:** Develop ED identification algorithm

## Recommendations

### Immediate Actions:
1. **Fix H2 referral counting bug** (identified separately)
2. **Derive ED visits** using encounter types and reason keywords
3. **Verify NYD code format** - may need ICD-9 versions

### Data Enhancement Options:
1. **Postal code linkage** - Check if available in another dataset
2. **Cost estimation** - Develop utilization-to-cost mapping
3. **Provider specialty** - Link via referral patterns

### Hypothesis Prioritization:
**Tier 1 (Fully Feasible):**
- H8: Multimorbidity patterns
- H9: Temporal utilization

**Tier 2 (Feasible with modifications):**
- H1: SSD patterns → utilization
- H3: Provider changes
- H4: Psychological factors (MH diagnoses available!)

**Tier 3 (Limited feasibility):**
- H2: Severity → costs (proxy only)
- H6: Diagnostic uncertainty

**Tier 4 (Not currently feasible):**
- H5: Health anxiety (no direct measures)
- H7: Socioeconomic moderation (no postal codes)

## Unexpected Findings

1. **Mental health data IS available!** Initial scan for F-codes (ICD-10) found none, but ICD-9 codes (290-319) identify 91,409 patients with mental health conditions
2. **Provider changes are common** - 56.5% of patients see multiple providers
3. **Temporal coverage has outliers** - Some encounters dated 1800 (data quality issue)

## Conclusion

The checkpoint data provides sufficient information to test 5 of 9 hypotheses with full or high confidence. Two additional hypotheses can be tested with proxy measures. Only 2 hypotheses (H5, H7) face significant data limitations.

The discovered availability of mental health diagnoses (ICD-9) significantly improves feasibility for H4 (psychological mediation), which was initially assessed as not feasible.