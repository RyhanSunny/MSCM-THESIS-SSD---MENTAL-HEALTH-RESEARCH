# SSD Phenotype Validation Implementation Guide

## Overview

The MC-SIMEX bias correction in our pipeline requires sensitivity and specificity values for the SSD phenotype algorithm. This guide explains how to obtain these values through validation.

## Why We Need Validation

- **Current State**: The pipeline has placeholder values (sensitivity=0.82, specificity=0.82)
- **Goal**: Calculate actual sensitivity/specificity by comparing our algorithm against clinical judgment
- **Impact**: More accurate bias correction → better causal effect estimates

## Quick Start: Automated Validation Process

### Step 1: Generate Validation Sample
```bash
cd SSD_Experiment1_Causal_Effect
python src/generate_validation_sample.py --n_positive=100 --n_negative=100
```

This creates `data_derived/validation_sample.csv` with 200 patients.

### Step 2: AI-Assisted Review
```python
# Using OpenAI GPT-4
import openai
openai.api_key = "your-key"

# Or using local LLM for privacy
# pip install ollama
# ollama pull medllama2
```

### Step 3: Run Validation Notebook
```bash
jupyter notebook Notebooks/SSD_Phenotype_Validation_with_AI.ipynb
```

## DSM-5 Criteria for SSD (300.82)

### A. Somatic Symptoms
One or more somatic symptoms that are distressing or result in significant disruption of daily life.

**In our data, look for:**
- Multiple normal lab tests (≥3) despite symptom complaints
- Symptoms across multiple body systems
- ICD-9 codes: 780-799 (symptoms without clear diagnosis)

### B. Excessive Response
Excessive thoughts, feelings, or behaviors related to the somatic symptoms, manifested by at least one:
1. Disproportionate and persistent thoughts about symptom seriousness
2. Persistently high anxiety about health/symptoms  
3. Excessive time/energy devoted to symptoms

**In our data, look for:**
- High healthcare utilization (>90th percentile visits)
- Multiple specialist referrals for symptoms
- Prolonged anxiolytic use (>180 days)
- Prolonged analgesic use without clear pain diagnosis

### C. Persistence
Although any one somatic symptom may not be continuously present, the state of being symptomatic is persistent (typically >6 months).

**In our data, look for:**
- Pattern spans multiple quarters
- Repeated encounters with similar complaints

## Manual Review Process

### For Each Patient in Validation Sample:

1. **Review Key Features**:
   ```
   - Normal lab count: [number]
   - Symptom referrals: [number]  
   - Anxiolytic days: [number]
   - Analgesic days: [number]
   - Annual encounters: [number]
   - High utilizer flag: [Y/N]
   ```

2. **Apply DSM-5 Criteria**:
   - [ ] Criterion A met? (somatic symptoms)
   - [ ] Criterion B met? (excessive response)
   - [ ] Criterion C met? (persistence >6 months)
   - [ ] All criteria met = SSD diagnosis

3. **Document Decision**:
   ```csv
   validation_id,patient_id,ssd_flag,meets_dsm5_criteria,confidence,notes
   1,P12345,1,1,85,"Multiple unexplained symptoms, high utilization"
   2,P67890,0,0,90,"Symptoms explained by documented arthritis"
   ```

## Calculating Metrics

Once reviews are complete:

```python
from sklearn.metrics import confusion_matrix

# Load results
results_df = pd.read_csv('ssd_validation_reviews.csv')

# Calculate confusion matrix
y_true = results_df['meets_dsm5_criteria']  # Clinical judgment
y_pred = results_df['ssd_flag']             # Algorithm

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Calculate metrics
sensitivity = tp / (tp + fn)  # True positive rate
specificity = tn / (tn + fp)  # True negative rate
ppv = tp / (tp + fp)         # Positive predictive value
npv = tn / (tn + fn)         # Negative predictive value

print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
```

## Updating Pipeline Configuration

1. **Update config.yaml**:
   ```yaml
   mc_simex:
     sensitivity: 0.875  # Your calculated value
     specificity: 0.823  # Your calculated value
     validation_date: "2025-01-20"
     validation_sample_size: 200
   ```

2. **Run MC-SIMEX correction**:
   ```bash
   python src/07a_misclassification_adjust.py
   ```

3. **Enable bias-corrected flag**:
   ```yaml
   mc_simex:
     use_bias_corrected_flag: true
   ```

4. **Re-run analysis**:
   ```bash
   python src/05_ps_match.py
   python src/06_causal_estimators.py
   ```

## Common Pitfalls to Avoid

### False Positives (Algorithm says SSD, but doesn't meet criteria):
- Appropriate treatment for documented anxiety/depression
- Undiagnosed medical conditions (early autoimmune, endocrine disorders)
- Cultural differences in symptom expression
- Medication use for legitimate chronic pain

### False Negatives (Algorithm misses SSD):
- Patients who avoid healthcare despite symptoms
- Focus on single symptom rather than multiple
- Good coping mechanisms masking dysfunction
- Incomplete data capture

## Validation Checklist

- [ ] Generated balanced sample (100 positive, 100 negative)
- [ ] Reviewed all 200 cases using DSM-5 criteria
- [ ] Flagged uncertain cases for senior clinician review
- [ ] Calculated sensitivity and specificity
- [ ] Updated config.yaml with results
- [ ] Ran MC-SIMEX to generate ssd_flag_adj
- [ ] Documented validation process and decisions

## Expected Outcomes

Typical validation results for EMR-based SSD algorithms:
- Sensitivity: 0.70-0.85 (catches most true cases)
- Specificity: 0.80-0.90 (few false positives)
- PPV: 0.60-0.80 (depends on prevalence)
- NPV: 0.85-0.95 (good at ruling out)

If your metrics fall outside these ranges, consider:
1. Reviewing your algorithm criteria
2. Expanding validation sample size
3. Getting second opinions on edge cases

## References

1. DSM-5-TR (2022). Somatic Symptom and Related Disorders, pp. 309-327
2. Dimsdale et al. (2013). Somatic Symptom Disorder: An important change in DSM. J Psychosom Res.
3. Häuser et al. (2022). Validation of ICD-11 chronic primary pain diagnoses. Pain.

## Contact for Clinical Questions

For questions about DSM-5 criteria application or edge cases:
- Principal Investigator: [Name]
- Clinical Advisor: [Name]
- Study Psychiatrist: [Name] 