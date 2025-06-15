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

## Usage

### Individual Scripts
```bash
python src/01_cohort_builder.py
python src/02_exposure_flag.py
```

### Full Pipeline
```bash
make all                    # Run complete pipeline
make cohort exposure        # Run specific stages
make causal                 # Run causal analysis only
```

---

*Source code documentation - Last updated: 2025-05-25*