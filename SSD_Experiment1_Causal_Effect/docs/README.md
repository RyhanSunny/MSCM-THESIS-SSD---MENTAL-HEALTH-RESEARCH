# SSD Causal Analysis Documentation

This directory contains documentation for the Somatic Symptom Disorder (SSD) causal analysis project.

## Contents

- **[power_poisson.md](power_poisson.md)**: Detailed power analysis calculations for Poisson count outcomes
- **methodology_notes.md**: Additional methodological considerations and implementation details
- **data_dictionary.md**: Comprehensive data dictionary for all variables
- **causal_dag.md**: Causal directed acyclic graph (DAG) documentation

## Project Overview

This project investigates the causal relationship between patterns of healthcare utilization (repeated normal diagnostic tests, unresolved specialist referrals, and persistent medication use) and subsequent healthcare encounters using data from the Canadian Primary Care Sentinel Surveillance Network (CPCSSN).

### Key Features

- **Robust causal inference**: Multiple estimators (TMLE, Double ML, Causal Forest)
- **Comprehensive sensitivity analysis**: E-values, placebo tests, misclassification adjustment
- **Temporal adjustment**: COVID-19 period segmentation and time-varying confounding
- **Effect modification**: Heterogeneous treatment effects across subgroups
- **Reproducible pipeline**: Docker containerization and DVC versioning

### Study Hypotheses

1. **H1**: ≥3 normal lab panels increase primary-care encounters (IRR ≈ 1.25–1.35)
2. **H2**: ≥2 unresolved referrals predict new psychotropic prescriptions (OR ≈ 1.40–1.60)
3. **H3**: >90 days medication predicts ED visits (aOR ≈ 1.30 anxiolytic, 1.20 analgesic)
4. **H4**: SSD Severity Index mediates ≥50% of total causal effect
5. **H5**: Effects strengthen in younger females, high deprivation, prior anxiety
6. **H6**: High-SSDSI patients: intervention reduces utilization ≥20%

## Quick Start

1. **Setup environment**:
   ```bash
   conda activate base
   pip install -r requirements.txt
   ```

2. **Run pipeline**:
   ```bash
   make all
   ```

3. **Generate reports**:
   ```bash
   make reporting
   ```

## Analysis Pipeline

The analysis follows a structured pipeline with dependency management:

```
01_cohort_builder → 02_exposure_flag → 03_mediator_autoencoder
                                   ↓
08_patient_master_table ← 05_confounder_flag ← 04_outcome_flag
                                   ↓
05_ps_match → 06_causal_estimators → 13_evalue_calc
                                   ↓
                              15_robustness → 18_reporting
```

## Data Sources

- **Primary**: CPCSSN checkpoint `checkpoint_1_20250318_024427`
- **Tables**: patient, encounter, health_condition, lab, medication, referral
- **Study period**: 2015-01-01 to 2017-12-31
- **Sample size**: 250,025 eligible patients

## Repository Structure

```
├── src/                    # Main analysis scripts
├── config/                 # Configuration files
├── data_derived/          # Processed datasets
├── results/               # Analysis results
├── figures/               # Generated plots
├── reports/               # R Markdown reports
├── tests/                 # Test suite
├── docs/                  # Documentation
├── utils/                 # Utility functions
├── Notebooks/             # Jupyter notebooks
├── requirements.txt       # Python dependencies
├── Makefile              # Build automation
├── dvc.yaml              # Data pipeline
└── Dockerfile            # Container specification
```

## Key Configuration

All analysis parameters are centralized in `config/config.yaml`:

- **Temporal windows**: Reference, exposure, and outcome periods
- **Cohort criteria**: Age, observation time, exclusions
- **Exposure definition**: Lab counts, referral thresholds, medication duration
- **Statistical parameters**: Random seeds, model hyperparameters

## Quality Assurance

The project implements comprehensive QA:

- **Data validation**: Integrity checks, missing data assessment
- **Propensity score diagnostics**: Balance assessment, overlap evaluation
- **Causal assumption testing**: Placebo tests, sensitivity analyses
- **Reproducibility**: Global random seeds, version control
- **Automated testing**: Unit tests for key functions

## Computational Requirements

- **Memory**: 32GB RAM recommended
- **GPU**: NVIDIA GPU with 6GB+ VRAM (optional, for XGBoost acceleration)
- **Storage**: ~50GB for full pipeline outputs
- **Software**: Python 3.12+, R 4.3+, Docker (optional)

## Results Summary

*Results will be populated upon pipeline completion*

- **Primary finding**: TBD
- **Effect size**: TBD
- **Confidence interval**: TBD
- **E-value**: TBD
- **Robustness**: TBD

## Contact

- **Author**: Ryhan Suny
- **Institution**: Toronto Metropolitan University
- **Team**: Car4Mind Research Team, University of Toronto
- **Supervisor**: Dr. Aziz Guergachi
- **Email**: sajibrayhan.suny@torontomu.ca

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Citation

If you use this work, please cite:

```
Suny, R., Guergachi, A. (2025). Causal Effects of Somatic Symptom Disorder Patterns 
on Healthcare Utilization: A Population-Based Study Using CPCSSN Data. 
Toronto Metropolitan University.
```

---

*Documentation last updated: 2025-05-25*