# Research Pre-registration Protocol: Negative Lab Cascade Effect on Healthcare Utilization
Version: 1.0
Date: February 24, 2025
Study ID: SSD-CPCSSN-2025-001

## Primary Research Question
Does a cascade of negative laboratory results causally increase future healthcare utilization through increased severity of somatic symptom disorder (SSD)?

## Hypotheses
1. Primary (H1): Multiple negative laboratory results (≥3 in 12 months) causally increase subsequent healthcare utilization, mediated by increased SSD severity.

2. Secondary:
   - H2a: The indirect effect (via SSD severity) will be larger than the direct effect
   - H2b: The effect will be stronger in patients with pre-existing anxiety/depression
   - H2c: The effect will vary by patient age and sex

## Variables
1. Treatment (X):
   - Definition: Three or more normal laboratory results within 12-month window
   - Measurement: Binary (0/1)
   - Timing: Assessed during months 6-18 of study period

2. Mediator (M):
   - Construct: SSD Severity Score
   - Components:
     a) Count of symptom-based ICD-9 codes (780-789)
     b) Frequency of symptom-focused visits
     c) Presence of anxiety/depression diagnoses
   - Measurement: Continuous score (0-100)
   - Validation: Factor analysis + expert review

3. Outcome (Y):
   - Primary: Total healthcare encounters in 12-month follow-up
   - Secondary: Emergency department visits, specialist referrals
   - Measurement: Count data

## Confounders
1. Pre-specified confounders:
   - Age
   - Sex
   - Pre-existing conditions
   - Baseline healthcare utilization
   - Socioeconomic indicators (postal code based)

2. Sensitivity analyses will assess unmeasured confounding

## Statistical Analysis Plan
1. Primary Analysis:
   - Causal mediation analysis using DoWhy framework
   - Estimation of direct and indirect effects
   - Bootstrap confidence intervals (5000 resamples)

2. Secondary Analyses:
   - Stratified analyses by age, sex, anxiety/depression status
   - Sensitivity analyses for unmeasured confounding
   - Alternative definitions of exposure/outcome

## Sample Size & Power
- Required sample: 8,000 patients (4,000 per group)
- Power: 80% to detect mediator effect of 0.2 SD
- Accounting for 20% attrition

## Success Criteria
1. Statistical:
   - Primary: p < 0.05 for mediation effect
   - Secondary: Clear dose-response relationship
   
2. Clinical:
   - Minimum 20% increase in utilization
   - Consistent effects across subgroups

## Quality Control
1. Data validation protocol
2. Code review process
3. Sensitivity analysis framework
4. Missing data handling strategy

## Timeline
- Data extraction: March 2025
- Analysis: April-May 2025
- Manuscript: June-July 2025