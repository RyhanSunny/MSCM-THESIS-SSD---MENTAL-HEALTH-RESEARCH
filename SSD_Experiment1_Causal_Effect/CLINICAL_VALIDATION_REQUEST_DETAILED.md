# Clinical Validation Request: SSD Phenotype Algorithm
## MC-SIMEX Bias Correction for Causal Inference

**Project**: Somatic Symptom Disorder (SSD) Causal Analysis  
**PI**: Ryhan Suny, Toronto Metropolitan University  
**Date**: June 2025  
**Required Timeline**: 1-2 weeks  

---

## 1. Executive Summary

We need clinical validation to calculate the **actual accuracy** of our SSD detection algorithm. Currently, we're using placeholder values from literature that may not apply to our Canadian primary care population. This validation is **critical** for producing unbiased causal effect estimates.

---

## 2. Why This Validation is Essential

### 2.1 The Problem: Misclassification Bias

When we use Electronic Medical Records (EMR) to identify SSD patients, we inevitably make errors:
- **False Positives**: Flagging patients as SSD who don't actually have it
- **False Negatives**: Missing true SSD patients

These errors create **misclassification bias** in our causal estimates. As shown by Lash et al. (2014)¹, even modest misclassification can severely bias results:
- 80% sensitivity/specificity → up to 40% bias in effect estimates
- 90% sensitivity/specificity → still 15-20% bias

### 2.2 The Solution: MC-SIMEX (Misclassification Simulation-Extrapolation)

MC-SIMEX, developed by Cook & Stefanski (1994)² and extended by Küchenhoff et al. (2006)³, corrects for known misclassification by:
1. Simulating increasing levels of misclassification
2. Extrapolating back to zero misclassification
3. Providing bias-corrected estimates

**But it requires knowing our actual sensitivity and specificity!**

### 2.3 Why Literature Values Don't Work

We're currently using 82% sensitivity/specificity from U.S. EMR studies, but:
- Different healthcare systems have different coding practices
- Canadian primary care may have unique patterns
- Our specific algorithm may perform differently
- Population characteristics affect accuracy

**Bottom line**: We need to validate on OUR data with OUR population.

---

## 3. Sample Size Justification

### Why 200 Patients?

Based on statistical power calculations for diagnostic accuracy studies⁴:

**Formula**: n = Z²α/2 × p(1-p) / d²

Where:
- Z = 1.96 (95% confidence level)
- p = expected proportion (0.80 for 80% accuracy)
- d = precision (0.05 for ±5% margin)

**Ideal sample**: 246 per group (492 total)  
**Practical sample**: 100 per group (200 total)

This gives us:
- 95% CI width of ±8% (acceptable for MC-SIMEX)
- Sufficient precision for bias correction
- Feasible for manual review
- Consistent with similar studies (Newton et al., 2013)⁵

---

## 4. What You (Developer) Will Do

### Step 1: Generate Validation Sample (30 minutes)
```python
# Run the notebook to create stratified sample
cd Notebooks
jupyter notebook SSD_Phenotype_Validation.ipynb

# This will:
# 1. Randomly select 100 patients our algorithm flagged as SSD
# 2. Randomly select 100 patients our algorithm flagged as non-SSD
# 3. Extract key features for clinical review
```

### Step 2: Prepare Clinical Review File
You'll create a CSV file with these columns:

```csv
validation_id | patient_id | algorithm_ssd_flag | age | sex | normal_lab_count | symptom_referrals | anxiolytic_days | encounters_per_year | charlson_score | CLINICIAN_SSD_DIAGNOSIS | CONFIDENCE | NOTES
1 | P12345 | 1 | 45 | F | 8 | 3 | 240 | 35 | 2 | [EMPTY] | [EMPTY] | [EMPTY]
2 | P23456 | 0 | 62 | M | 1 | 0 | 0 | 8 | 4 | [EMPTY] | [EMPTY] | [EMPTY]
...
```

### Step 3: After Clinical Review
```python
# Load the completed CSV with clinician's diagnoses
# Run validation metrics calculation
# This will output:
# - Sensitivity: P(Algorithm=1 | Clinician says SSD=1)
# - Specificity: P(Algorithm=0 | Clinician says SSD=0)
# - Positive/Negative Predictive Values
# - 95% Confidence Intervals
```

---

## 5. What the Clinician Will Do

### For Each Patient Record:

**Review the provided data**:
- Demographics (age, sex)
- Clinical patterns:
  - Number of normal lab results despite symptoms
  - Referrals to specialists for unexplained symptoms
  - Days on anxiolytics/analgesics without clear indication
  - Annual healthcare encounters
  - Comorbidity burden (Charlson score)

**Apply DSM-5 Criteria**⁶:
- **Criterion A**: One or more somatic symptoms causing distress
- **Criterion B**: Excessive thoughts, feelings, or behaviors about symptoms
- **Criterion C**: Persistent state (>6 months)

**Make a diagnosis**:
- **YES**: Patient meets DSM-5 criteria for SSD
- **NO**: Patient does not meet criteria
- **UNCERTAIN**: Insufficient information or borderline case

**Document confidence** (0-100%):
- 90-100%: Very confident
- 70-89%: Moderately confident
- <70%: Low confidence (may need discussion)

**Add notes** for:
- Uncertain cases
- Alternative diagnoses considered
- Missing information needed

---

## 6. Example Clinical Review Cases

### Example 1: Clear Positive
```
Age: 42F
Normal labs: 12 (CBC, metabolic panel, thyroid, etc.)
Referrals: 5 (neurology, rheumatology, cardiology)
Anxiolytic use: 365 days
Annual visits: 52

Clinician diagnosis: YES (high confidence)
Note: "Classic SSD - multiple unexplained symptoms, excessive testing, high anxiety"
```

### Example 2: Clear Negative
```
Age: 68M
Normal labs: 2
Referrals: 1 (routine diabetic eye exam)
Anxiolytic use: 0 days
Annual visits: 4

Clinician diagnosis: NO (high confidence)
Note: "Appropriate healthcare use for documented diabetes"
```

### Example 3: Uncertain Case
```
Age: 55F
Normal labs: 5
Referrals: 2 (GI for documented IBS)
Anxiolytic use: 90 days (post-surgical)
Annual visits: 18

Clinician diagnosis: UNCERTAIN (60% confidence)
Note: "Some features present but may be explained by IBS and surgical recovery"
```

---

## 7. Timeline and Deliverables

### Developer Tasks (Your Work):
**Day 1**: 
- Generate validation sample (30 min)
- Create review CSV file (30 min)
- Send to clinician with instructions

**Day 7-8** (after clinical review):
- Process completed CSV (30 min)
- Calculate validation metrics (30 min)
- Update config files with real values (15 min)

### Clinician Tasks:
**Days 2-6**:
- Review 200 patients (~40 per day, 5-10 min each)
- Complete diagnosis columns
- Return annotated CSV

### Final Deliverables:
1. Validated sensitivity/specificity metrics
2. Updated `config.yaml` with real values
3. Validation report with confidence intervals
4. Ready for MC-SIMEX bias correction

---

## 8. Impact on Results

### With Current Placeholder Values (82%/82%):
- Unknown bias in causal estimates
- Reviewers will question validity
- Results may be completely wrong

### With Validated Values:
- Accurate bias correction via MC-SIMEX
- Defensible, publishable results
- Known confidence in findings

### Expected Outcomes:
If validation shows:
- **High accuracy (>85%)**: Minimal bias correction needed
- **Moderate accuracy (70-85%)**: Significant correction, but valid results
- **Low accuracy (<70%)**: May need to refine algorithm before proceeding

---

## 9. References

1. Lash TL, Fox MP, Fink AK. *Applying Quantitative Bias Analysis to Epidemiologic Data*. Springer; 2014. [doi:10.1007/978-0-387-87959-8](https://doi.org/10.1007/978-0-387-87959-8)

2. Cook JR, Stefanski LA. Simulation-extrapolation estimation in parametric measurement error models. *J Am Stat Assoc*. 1994;89(428):1314-1328. [doi:10.1080/01621459.1994.10476871](https://doi.org/10.1080/01621459.1994.10476871)

3. Küchenhoff H, Mwalili SM, Lesaffre E. A general method for dealing with misclassification in regression: the misclassification SIMEX. *Biometrics*. 2006;62(1):85-96. [doi:10.1111/j.1541-0420.2005.00396.x](https://doi.org/10.1111/j.1541-0420.2005.00396.x)

4. Buderer NM. Statistical methodology: I. Incorporating the prevalence of disease into the sample size calculation for sensitivity and specificity. *Acad Emerg Med*. 1996;3(9):895-900. [doi:10.1111/j.1553-2712.1996.tb03538.x](https://doi.org/10.1111/j.1553-2712.1996.tb03538.x)

5. Newton KM, Peissig PL, Kho AN, et al. Validation of electronic medical record-based phenotyping algorithms. *J Am Med Inform Assoc*. 2013;20(e1):e147-e154. [doi:10.1136/amiajnl-2012-000896](https://doi.org/10.1136/amiajnl-2012-000896)

6. American Psychiatric Association. *Diagnostic and Statistical Manual of Mental Disorders* (5th ed., text rev.). 2022. [doi:10.1176/appi.books.9780890425787](https://doi.org/10.1176/appi.books.9780890425787)

---

## 10. Contact Information

**Principal Investigator**: Ryhan Suny  
**Email**: sajibrayhan.suny@torontomu.ca  
**Supervisor**: Dr. Aziz Guergachi  
**Institution**: Toronto Metropolitan University  

For questions about:
- Clinical criteria: Contact psychiatrist consultant
- Technical implementation: Contact Ryhan
- Statistical methods: Contact biostatistician consultant

---

## Appendix: Quick Reference for Clinicians

### DSM-5 SSD Diagnostic Checklist

☐ **A. Somatic Symptoms** (≥1 required)
- Distressing physical symptoms
- Disrupts daily functioning

☐ **B. Excessive Response** (≥1 required)
- Disproportionate worry about symptom seriousness
- High health anxiety
- Excessive time/energy on symptoms

☐ **C. Duration**
- Symptomatic state >6 months

**All criteria must be met for SSD diagnosis**

### Red Flags to Consider
- Undiagnosed medical conditions
- Appropriate anxiety about serious illness
- Cultural factors in symptom expression
- Iatrogenic anxiety from medical uncertainty

---

**Document Version**: 1.0  
**Created**: June 2025  
**Next Update**: After validation completion