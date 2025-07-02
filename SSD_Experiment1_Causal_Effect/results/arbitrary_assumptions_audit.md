# Audit of Arbitrary Assumptions and Default Values in SSD Pipeline

Generated: 2025-01-02

## Summary
This document catalogs all arbitrary assumptions, default values, and "magic numbers" found in the SSD pipeline codebase that may lack evidence-based justification.

## 1. Missing Data Handling

### 1.1 fillna() with Hardcoded Values

| File | Line | Code | Assumption | Justification Needed |
|------|------|------|------------|---------------------|
| `01_cohort_builder.py` | 179 | `elig["SpanMonths"].fillna(0)` | Missing span = 0 months | Why not exclude patients with missing span? |
| `01_cohort_builder.py` | 214 | `.dt.days.fillna(30)` | Missing medication duration = 30 days | Arbitrary 1-month assumption |
| `01_cohort_builder.py` | 284 | `charlson_scores.fillna(0)` | Missing Charlson = 0 | Assumes healthy if no data |
| `02_exposure_flag.py` | 579 | `med.StopDate.fillna(med.StartDate + pd.Timedelta(days=30))` | Missing stop date = start + 30 days | Arbitrary 1-month refill assumption |
| `02_exposure_flag.py` | 604 | `.fillna(0)` | Missing exposure criteria = 0 | Assumes no exposure if missing |
| `03_mediator_autoencoder.py` | 155, 242 | `.fillna(0)` | Missing features = 0 | May bias autoencoder |
| `05_ps_match.py` | 437 | `X = df[covar_cols].fillna(0).values` | Missing covariates = 0 | "Simple imputation for now" - needs proper handling |
| `12_temporal_adjust.py` | 306 | `X = patient_data[X_vars].fillna(0.5)` | Missing temporal features = 0.5 | Arbitrary midpoint assumption |

## 2. Time-Based Thresholds

### 2.1 Duration Thresholds

| Threshold | Value | Location | Evidence/Justification |
|-----------|-------|----------|----------------------|
| Minimum observation period | 30 months | `SPAN_REQ_MONTHS` | Arbitrary, why not 24 or 36? |
| Drug persistence (original) | 90 days | Various | Clinical guidelines unclear |
| Drug persistence (enhanced) | 180 days | `MIN_DRUG_DAYS` | "Felipe Enhancement" - needs clinical validation |
| Medication refill assumption | 30 days | `02_exposure_flag.py:579` | Arbitrary single refill |
| Maximum gap between prescriptions | 30 days | Multiple files | No evidence provided |
| Early death window | 30 days | `death_rates_analysis.py:185` | Arbitrary definition |
| Exposure window | 365 days | Various | Why exactly 1 year? |

## 3. Quantile-Based Thresholds

### 3.1 High Utilization Definition

| File | Definition | Justification Needed |
|------|------------|---------------------|
| `03_mediator_autoencoder.py:355` | Top 75th percentile of visits | Why 75th and not 80th or 90th? |
| `04_outcome_flag.py:254` | Top 75th percentile of encounters | Arbitrary quartile cutoff |
| `05_confounder_flag.py:200` | Top 75th percentile baseline encounters | No clinical basis |
| `14_mediation_analysis.py:442` | SSDSI > 75th percentile for intervention | Why this specific cutoff? |

## 4. Clinical Criteria Thresholds

### 4.1 Exposure Criteria

| Criterion | Threshold | Location | Evidence |
|-----------|-----------|----------|----------|
| Minimum normal labs | 3 | `MIN_NORMAL_LABS` | Why 3 and not 2 or 4? |
| Minimum symptom referrals | 2 | `MIN_SYMPTOM_REFERRALS` | Arbitrary cutoff |
| Maximum Charlson score | 5 | `CHARLSON_MAX` | Why exclude >5? |
| Minimum age | 18 | `MIN_AGE` | Standard but affects generalizability |

## 5. Statistical/Technical Parameters

### 5.1 Autoencoder Configuration

| Parameter | Default | Justification Needed |
|-----------|---------|---------------------|
| Input features | 56 | Based on what? |
| Encoding dimension | 16 | Arbitrary compression |
| Hidden dimension | 32 | No optimization shown |
| Regularization | 1e-5 | Standard but not tuned |
| Epochs | 100 | May need more/less |
| Batch size | 256 | Standard but arbitrary |
| Validation split | 0.2 | Standard 80/20 but why? |
| Early stopping patience | 10 | Arbitrary patience |

### 5.2 Multiple Imputation

| Parameter | Value | Location | Evidence |
|-----------|-------|----------|----------|
| Number of imputations | 5 | `m_imputations` | Rubin's minimum, but may need more |
| Maximum missing percent | 5.0% | `max_missing_percent` | Very restrictive threshold |

## 6. Lab Test Normal Ranges

| Test | Normal Range | File | Clinical Validation |
|------|--------------|------|-------------------|
| Vitamin B12 | >= 133 pmol/L | `lab_utils.py` | Needs lab-specific ranges |
| Ferritin | M: 24-336, F: 11-307 μg/L | `lab_utils.py` | Age/sex specific but simplified |
| TSH | 0.32-4.0 mIU/L | `lab_utils.py` | Standard but varies by lab |
| Vitamin D | >= 75 nmol/L | `lab_utils.py:108` | Optimal vs sufficient debate |

## 7. Cost-Effectiveness Assumptions

| Assumption | Value | Location | Basis |
|------------|-------|----------|-------|
| Enhanced screening cost | $150 | `week5_figures.py:580` | Arbitrary |
| Reduced testing savings | -$75 | `week5_figures.py:581` | Arbitrary |
| AI-assisted diagnosis cost | $200 | `week5_figures.py:582` | No source |
| Telemedicine savings | -$50 | `week5_figures.py:583` | Estimated |

## 8. Other Arbitrary Values

### 8.1 Miscellaneous

| Value | Context | Location | Issue |
|-------|---------|----------|-------|
| Random seed | 42 | Multiple files | Standard but affects reproducibility |
| AUROC fallback | 0.65-0.75 | `retrain_autoencoder.py:324` | Random range when calculation fails |
| Effective sample size | 750 | `tests/conftest.py:213` | Test fixture but arbitrary |
| Lab coverage expectation | 65-75% | `test_hierarchical_index_dates.py:98` | Arbitrary range |

## Recommendations

1. **Clinical Validation Required**:
   - All duration thresholds (30, 90, 180 days) need clinical literature support
   - Lab normal ranges should be institution-specific
   - High utilization definitions need healthcare system context

2. **Statistical Justification Needed**:
   - Quantile cutoffs (75th percentile) should be data-driven
   - Missing data handling strategy needs formal justification
   - Autoencoder hyperparameters need optimization study

3. **Documentation Improvements**:
   - Each arbitrary value should have a comment explaining its source
   - Create a configuration validation module to flag unjustified defaults
   - Maintain a "Clinical Assumptions" document for review

4. **Sensitivity Analysis**:
   - Test robustness to different thresholds
   - Report how results change with different assumptions
   - Consider multiple scenarios for key parameters

5. **Evidence Collection**:
   - Literature review for clinical thresholds
   - Consult domain experts for validation
   - Use data-driven methods where possible (e.g., optimal cutoffs from ROC curves)