# SSD Causal Analysis Pipeline - Source Code

This directory contains the main analysis scripts for the Somatic Symptom Disorder (SSD) causal analysis project.

## Scripts Overview

### Data Preparation Pipeline
- **`01_cohort_builder.py`** - Build eligible cohort from CPCSSN data
- **`02_exposure_flag.py`** - Generate SSD pattern exposure flags
- **`03_mediator_autoencoder.py`** - Build SSD Severity Index using autoencoder
- **`04_outcome_flag.py`** - Calculate healthcare utilization outcomes
- **`05_confounder_flag.py`** - Extract baseline confounders
- **`06_lab_flag.py`** - Process laboratory result patterns
- **`07_missing_data.py`** - Handle missing data with MICE
- **`07_referral_sequence.py`** - Analyze referral patterns
- **`07a_misclassification_adjust.py`** - MC-SIMEX bias correction
- **`08_patient_master_table.py`** - Merge all datasets

### Causal Analysis Pipeline
- **`05_ps_match.py`** - Propensity score estimation and matching
- **`06_causal_estimators.py`** - TMLE, Double ML, Causal Forest
- **`14_mediation_analysis.py`** - DoWhy-based mediation analysis (H4, H5, H6)
- **`12_temporal_adjust.py`** - Temporal confounding adjustment
- **`13_evalue_calc.py`** - E-value sensitivity analysis
- **`14_placebo_tests.py`** - Placebo outcome tests
- **`15_robustness.py`** - Comprehensive robustness checks

### Supporting Analysis
- **`death_rates_analysis.py`** - Crude death rate calculations
- **`finegray_competing.py`** - Competing risk analysis

### Utilities
- **`config_loader.py`** - Configuration management
- **`artefact_tracker.py`** - Analysis tracking
- **`icd_utils.py`** - ICD code utilities

## Treatment Column Options

All causal analysis scripts accept a `--treatment-col` parameter to specify which SSD definition to use:

- **`ssd_flag` (default)**: OR logic - DSM-5 Somatic Symptom Disorder proxy
- **`ssd_flag_strict`**: AND logic - DSM-IV Somatoform Disorders proxy

## Usage

### Individual Scripts
```bash
# Default (OR logic / DSM-5 proxy)
python src/05_ps_match.py
python src/06_causal_estimators.py

# Strict (AND logic / DSM-IV proxy)  
python src/05_ps_match.py --treatment-col ssd_flag_strict
python src/06_causal_estimators.py --treatment-col ssd_flag_strict
python src/14_mediation_analysis.py --treatment-col ssd_flag_strict
```

### Full Pipeline
```bash
make all                           # Run complete pipeline (OR logic)
make all TREATMENT_COL=ssd_flag_strict  # Run with strict definition
make cohort exposure               # Run specific stages
make causal                        # Run causal analysis only
make causal TREATMENT_COL=ssd_flag_strict  # Causal analysis with strict definition
```

### Available Scripts with --treatment-col Support
- `05_ps_match.py` - Propensity score matching
- `06_causal_estimators.py` - TMLE, Double ML, Causal Forest
- `14_mediation_analysis.py` - **CORRECTED**: Mental health analysis suite (H4-H6)
- `07a_misclassification_adjust.py` - Exposure misclassification adjustment
- `12_temporal_adjust.py` - Temporal confounding adjustment
- `13_evalue_calc.py` - E-value sensitivity analysis
- `14_placebo_tests.py` - Placebo outcome tests
- `15_robustness.py` - Comprehensive robustness checks
- `death_rates_analysis.py` - Death rate calculations
- `finegray_competing.py` - Competing risk analysis

### Hypothesis-Specific Analysis Scripts
**H4 (Mediation)**: `14_mediation_analysis.py --hypothesis H4`
- Tests SSD severity index mediation in homogeneous MH population

**H5 (Effect Modification)**: `14_mediation_analysis.py --hypothesis H5`  
- Tests interaction effects in high-risk MH subgroups (anxiety, age <40, female, substance use)

**H6 (Intervention)**: `14_mediation_analysis.py --hypothesis H6`
- G-computation for integrated MH-PC care intervention effects

---

*Source code documentation - Last updated: 2025-06-16*