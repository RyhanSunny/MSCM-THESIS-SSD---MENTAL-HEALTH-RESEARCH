# Pipeline Fallback Values Audit
## Critical Review of Arbitrary Assumptions

### 1. **MEDICATION DURATION - 30 DAYS** ❌
**Location**: `01_cohort_builder.py:214`
```python
).dt.days.fillna(30)  # Default 30 days if missing
```
**Problem**: No clinical justification for 30 days
**Why it matters**: Affects psychotropic medication exposure classification (H3)
**Evidence needed**: 
- Standard prescription durations from CPCSSN data
- Literature on typical psychotropic prescription patterns
- Consider using mode/median from actual data instead

### 2. **OBSERVATION PERIOD - 30 MONTHS** ❓
**Location**: `01_cohort_builder.py` (eligibility criteria)
**Problem**: Arbitrary threshold for minimum observation time
**Why it matters**: Excludes 56,276 patients (18.3%)
**Alternative**: 
- 24 months (2 years) - standard in epidemiology
- 36 months (3 years) - for chronic conditions
- Data-driven: based on median follow-up time

### 3. **LAB NORMAL THRESHOLD - 3 LABS** ❌
**Location**: `02_exposure_flag.py` (H1 hypothesis)
**Current**: ≥3 normal labs in 12 months
**Problem**: No citation for why 3, not 2 or 4
**Impact**: 111,794 patients (44.7%) classified as exposed
**Evidence needed**: 
- Literature on diagnostic uncertainty patterns
- Sensitivity analysis with 2, 3, 4, 5 thresholds

### 4. **DRUG PERSISTENCE - 180 DAYS** ❓
**Location**: `02_exposure_flag.py` (H3 hypothesis)
**History**: Enhanced from 90 days to 180 days
**Problem**: Both values seem arbitrary
**Alternative**: 
- WHO definition of chronic therapy (>90 days)
- FDA chronic use definition (>6 months)
- Literature-based thresholds for psychotropic adherence

### 5. **PRESCRIPTION GAP - 30 DAYS** ❌
**Location**: `02_exposure_flag.py`
**Problem**: Maximum gap between prescriptions
**Why problematic**: Different drugs have different refill patterns
**Better approach**: Drug-specific grace periods based on:
- Days supply
- Drug class (SSRIs vs benzodiazepines)
- Clinical guidelines

### 6. **HIGH UTILIZATION - 75TH PERCENTILE** ❌
**Location**: Multiple files (outcomes, confounders)
**Problem**: No justification for 75th vs 80th, 90th percentile
**Impact**: Defines "high utilizers" arbitrarily
**Alternative**: 
- Clinical definition (e.g., >X visits/year)
- Cost-based threshold
- Statistical outlier detection (>2 SD)

### 7. **CHARLSON CUTOFF - SCORE > 5** ❓
**Location**: `01_cohort_builder.py`
**Problem**: Excludes 799 patients
**Question**: Why 5? Literature uses various cutoffs (3, 4, 6)
**Need**: Sensitivity analysis or literature justification

### 8. **MISSING DATA IMPUTATION - VARIOUS** ❌
**Locations**: Throughout pipeline
- `fillna(0)` for counts - assumes no events (dangerous)
- `fillna(0.5)` for binary variables - nonsensical
- Mean/mode imputation without considering MAR/MCAR/MNAR

### 9. **INDEX DATE FALLBACK - FIRST ENCOUNTER** ❓
**Location**: `01_cohort_builder.py:268`
**Problem**: For 33,208 patients with no lab/MH/drug index
**Issue**: First encounter may be unrelated to SSD
**Alternative**: Exclude these patients or use different phenotype

### 10. **AUTOENCODER PARAMETERS** ❌
**Location**: `03_mediator_autoencoder.py`
- Encoding dimension: 16
- Hidden layer: 32
- Epochs: 50
**Problem**: No hyperparameter tuning mentioned
**Impact**: Affects SSDSI quality

### 11. **LAB NORMAL RANGES** ❌
**Location**: `lab_utils.py`
**Problem**: Fixed ranges, not institution-specific
**Example**: Glucose 70-100 mg/dL (varies by lab)
**Solution**: Use CPCSSN lab-specific reference ranges

### 12. **TIME WINDOWS - ARBITRARY** ❌
Various exposure/outcome windows:
- 6 months, 12 months, 18 months, 24 months
- No clinical rationale provided
- Should align with disease natural history

## RECOMMENDATIONS

### Immediate Actions:
1. **Document all assumptions** with literature citations
2. **Sensitivity analyses** for key thresholds
3. **Data-driven defaults** where possible

### For Thesis Defense:
1. Create supplementary table of all assumptions
2. Justify each with literature or acknowledge as limitation
3. Show robustness through sensitivity analyses

### Code Changes Needed:
```python
# Instead of:
).dt.days.fillna(30)  # Default 30 days if missing

# Use:
# Calculate median duration from complete cases
median_duration = psych_meds[psych_meds['duration_days'].notna()]['duration_days'].median()
).dt.days.fillna(median_duration)  # Use data-driven default
```

### Missing Clinical Justifications:
1. Why 2 referrals for H2? (Currently 1,655 patients, 0.7%)
2. Why OR logic vs AND logic for exposure?
3. Why 2015-2016 exposure window?
4. Why 2016-2017 outcome window?

### Statistical Concerns:
- Multiple testing without FDR correction in some places
- No power calculations for rare outcomes
- Arbitrary significance levels (0.05) without adjustment

This audit reveals significant gaps in clinical justification throughout the pipeline. Each arbitrary choice potentially affects the validity of causal inference.