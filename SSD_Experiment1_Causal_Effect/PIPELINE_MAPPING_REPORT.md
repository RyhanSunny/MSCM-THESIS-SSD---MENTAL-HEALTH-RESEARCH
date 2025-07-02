# SSD Pipeline Complete Mapping Report
**Generated**: 2025-07-02
**Author**: Claude (AI Assistant)
**Purpose**: Comprehensive analysis of the SSD research pipeline execution, data flow, and critical issues

## Executive Summary

This report provides a complete mapping of the SSD (Somatic Symptom Disorder) research pipeline, analyzing the actual implementation against the documented methodology. The pipeline processes 250,025 mental health patients to identify SSD patterns and measure causal effects on healthcare utilization.

### Critical Findings:
1. **MAJOR BUG**: Exposure flag incorrectly stores AND logic (186 patients) instead of OR logic (142,769 patients)
2. **DATA FLOW ISSUE**: Pipeline execution order causes data overwrite
3. **INCOMPLETE RESULTS**: Pooled causal estimates are empty despite 30 imputation runs
4. **STRING FORMATTING ERROR**: Affects progress display but not core results

## Research Context

### Study Question
"What is the causal effect of SSD-consistent care patterns on healthcare utilization and costs in mental health patients?"

### Hypotheses (H1-H6)
- **H1**: Normal lab results → increased healthcare utilization
- **H2**: Symptom-based referral loops → increased costs (DATA LIMITATION: Cannot identify MH crisis)
- **H3**: Prolonged psychotropic use → higher utilization
- **H4**: Mediation through psychological factors
- **H5**: Health anxiety as mediator
- **H6**: Physician communication factors

### Population
- Initial: 256,746 mental health patients
- After eligibility: 250,025 patients
- Study period: 2013-2016 (exposure: 2015-2016, outcome: 2016-2017)

## Pipeline Architecture (12 Phases, 26 Steps)

### PHASE 1: Data Foundation (Steps 1-3)
**Purpose**: Establish cohort and exposure definitions

#### Step 1: Cohort Definition (`01_cohort_definition.py`)
- **Input**: Raw checkpoint data from `checkpoint_1_20250318_024427/`
- **Process**: 
  - Apply inclusion/exclusion criteria
  - Implement hierarchical index date (lab → MH encounter → psychotropic)
  - Handle 28.3% missing lab dates issue
- **Output**: `cohort.parquet` (250,025 patients)
- **Status**: ✓ Completed successfully

#### Step 2: Exposure Flag (`02_exposure_flag.py`) ⚠️ CRITICAL BUG
- **Input**: `cohort.parquet`, lab/referral/medication tables
- **Process**: 
  - H1: ≥3 normal labs (111,794 patients)
  - H2: ≥2 symptom referrals (1,655 patients) 
  - H3: ≥180 days psychotropics (55,695 patients)
  - OR logic: Any criterion = exposed
  - AND logic: All criteria = exposed
- **Bug**: Script runs twice (OR then AND), final file has AND logic (186) not OR (142,769)
- **Output**: `exposure.parquet` (INCORRECT: 186 instead of 142,769)

#### Step 3: Master Table (`03_master_table.py`)
- **Input**: All derived tables
- **Process**: Merge demographics, conditions, utilization
- **Output**: `patient_master.parquet` (250,025 patients, 186 exposed)
- **Note**: Propagates the exposure flag bug

### PHASE 2: Feature Engineering (Steps 4-6)

#### Step 4: SSDSI Autoencoder (`04_ssdsi_autoencoder.py`)
- **Purpose**: Create severity index from symptoms
- **Method**: Dimensionality reduction on H1-H3 features
- **Output**: `ssdsi_scores.parquet`

#### Step 5: Covariates (`05_prepare_causal_covariates.py`)
- **Categories**: Demographics, clinical, healthcare history
- **Output**: `covariates_for_causal.parquet`

#### Step 6: PS Matching (`06_ps_matching_xgboost.py`)
- **Model**: XGBoost for propensity scores
- **Issue**: Uses incorrect exposure (186 patients)
- **Output**: `ps_scores.parquet`, matched pairs

### PHASE 3: Outcome Preparation (Steps 7-8)

#### Step 7: Outcomes (`07_prepare_outcomes.py`)
- **Metrics**: Visit counts, costs, ER visits
- **Window**: 12 months post-exposure
- **Output**: `outcomes_for_causal.parquet`

#### Step 8: Validation (`08_validate_causal_data.py`)
- **Checks**: Overlap, balance, missing data
- **Output**: Validation reports

### PHASE 4: Causal Analysis (Steps 9-12)

#### Step 9: TMLE (`09_tmle_analysis.py`)
- **Method**: Targeted Maximum Likelihood
- **Result**: ATE = 0.0 (likely due to small exposed n=186)

#### Step 10: Double ML (`10_double_ml_analysis.py`)
- **Method**: Double Machine Learning
- **Result**: ATE = 4.38 (SE=0.57)

#### Step 11: Causal Forest (`11_causal_forest_analysis.py`)
- **Method**: Generalized Random Forests
- **Result**: ATE = 6.05 (SE=0.01)

#### Step 12: MC-SIMEX (`12_mcsimex_bias_correction.py`)
- **Purpose**: Correct measurement error
- **Status**: Completed

### PHASE 5: Advanced Analytics (Steps 13-16)

#### Step 13: Temporal (`13_temporal_analysis.py`)
- **Analysis**: Time-varying effects
- **Output**: Temporal patterns

#### Step 14: Subgroups (`14_subgroup_analysis.py`)
- **Groups**: Age, gender, diagnoses
- **Output**: Heterogeneous effects

#### Step 15: Multiple Imputations (`15_*_imputed_*.py`)
- **Scripts**:
  - `15_1_run_multiple_imputations.py`: Create 30 imputations
  - `15_2_run_imputed_data_validation.py`: Validate each
  - `15_3_run_imputed_causal_pipeline.py`: Run causal analysis
  - `15_4_pool_causal_estimates.py`: Pool using Rubin's Rules
  - `15_5_multiple_testing_correction.py`: FDR correction
- **Issue**: String formatting error but results generated
- **Output**: 30 result files (but pooling empty)

#### Step 16: Sensitivity (`16_sensitivity_analysis.py`)
- **Tests**: Unmeasured confounding
- **Methods**: E-value, tipping point

### PHASE 6: Reporting (Steps 17-26)
- Visualization, documentation, reproducibility

## Critical Issues Identified

### 1. Exposure Flag Overwrite Bug
**Location**: `02_exposure_flag.py` lines 680-682
**Issue**: When run with `--logic and`, overwrites the OR logic file
**Impact**: All downstream analyses use 186 exposed instead of 142,769
**Evidence**:
```python
# Notebook runs twice:
run_pipeline_script("02_exposure_flag.py", args="--logic or")   # Creates 142,769
run_pipeline_script("02_exposure_flag.py", args="--logic and")  # Overwrites with 186
```

### 2. Empty Pooled Results
**File**: `results/pooled_causal_estimates.json`
**Content**: Only metadata, no pooled estimates
**Cause**: Likely format mismatch in input files

### 3. String Formatting Error
**Location**: `imputed_causal_pipeline_progress.py` line 81
**Error**: Invalid f-string format specifier
**Impact**: Progress display broken but results still generated

## Data Flow Summary

```
Raw Data (checkpoint) 
    ↓
Cohort (250,025) 
    ↓
Exposure Flag (BUG: 186 not 142,769)
    ↓
Master Table + Features
    ↓
PS Matching (on wrong exposure)
    ↓
Causal Analysis (TMLE=0, DML=4.38, CF=6.05)
    ↓
30 Imputations 
    ↓
Pooled Results (EMPTY)
```

## Recommendations

### Immediate Actions:
1. **FIX EXPOSURE BUG**: Modify notebook to save OR/AND to separate files
2. **RE-RUN PIPELINE**: With correct exposure flag (142,769 patients)
3. **DEBUG POOLING**: Check input format for Rubin's Rules
4. **FIX STRING FORMAT**: Remove conditional in f-string

### Code Fix for Exposure:
```python
# Instead of overwriting, save to different files:
result_or = run_pipeline_script("02_exposure_flag.py", args="--logic both")
# This saves both exposure_or.parquet and exposure_and.parquet
```

### Validation Steps:
1. Verify exposure.parquet has 142,769 exposed
2. Check PS matching uses correct exposure
3. Confirm pooled estimates are populated
4. Validate causal estimates make clinical sense

## Conclusion

The SSD pipeline is architecturally sound with sophisticated causal inference methods. However, a critical bug in the exposure flag generation causes the entire analysis to run on only 186 patients (meeting ALL criteria) instead of 142,769 patients (meeting ANY criteria). This explains why TMLE shows zero effect and why results may not reflect the true causal relationships. The pipeline must be re-run with the corrected exposure definition to produce valid results for the research hypotheses.