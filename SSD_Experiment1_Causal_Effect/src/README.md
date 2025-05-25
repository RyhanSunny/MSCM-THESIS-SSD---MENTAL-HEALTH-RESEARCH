# SSD Causal Analysis - Source Code

This directory contains the implementation of the SSD causal analysis pipeline as described in the methodology blueprint.

## Script-to-Hypothesis Mapping

| Script | Hypotheses Supported | Purpose |
|--------|---------------------|---------|
| 01_cohort_builder.py | H1-H6, RQ | Builds base cohort for all analyses |
| 02_exposure_flag.py | H1-H6 | Identifies SSD-pattern treatment group |
| 03_mediator_autoencoder.py | H4, H5, H6 | Creates psychological distress mediator |
| 04_outcome_flag.py | H1, H2, H3, RQ | Captures healthcare utilization & cost outcomes |
| 05_confounder_flag.py | H1-H6, RQ | Extracts confounders for causal adjustment |
| 06_lab_flag.py | RQ, H1, H5 | Analyzes lab utilization patterns |

### Research Question & Hypotheses Reference
- **RQ**: What is the prevalence of SSD in primary care?
- **H1**: SSD patients have higher healthcare utilization
- **H2**: SSD patients incur higher medical costs
- **H3**: SSD patients receive more inappropriate medications
- **H4**: Effects mediated through psychological distress
- **H5**: Effects mediated through health anxiety
- **H6**: Effects mediated through physician factors

## Modules

### 01_cohort_builder.py

This module implements the cohort selection logic for the SSD study. It:

1. Loads the necessary data files from the prepared dataset
2. Applies inclusion criteria:
   - Age ≥ 18 years
   - At least 30 consecutive months of electronic records before censor date
   - Has lab records for index date definition
3. Identifies patients with SSD based on the expanded criteria:
   - Patients with explicit SSD diagnosis codes
   - Patients with "Not Yet Diagnosed" (NYD) codes (799.9, V71.x) AND multi-system symptoms AND ≥3 normal lab results
4. Outputs a comprehensive patient-level dataset with all relevant features for downstream analysis

#### Key Features

- Detection of NYD (Not Yet Diagnosed) codes
- Body system categorization to identify multi-system complaints
- Normal lab result identification using multiple approaches
- SSD patient flagging based on expanded criteria

#### Output

The module produces a `cohort.parquet` file in the `data` directory containing all patients with flags indicating SSD status and associated features.

## Study Documentation

Each script automatically updates the study documentation YAML file upon completion using `scripts/update_study_doc.py`. This ensures:
- Provenance tracking of all artefacts
- Mapping of outputs to hypotheses
- Logging of key metrics and results
- Timestamping of pipeline execution

To view the current study documentation:
```bash
cat results/study_documentation_*.yaml | head -50
```

## Next Steps

After the cohort is built, proceed to the next modules in the pipeline:

1. `02_exposure_flag.py` - Implements the SSD-Pattern Flag
2. `03_mediator_autoencoder.py` - Builds the SSD Severity Index
3. `04_covariates.py` - Creates confounding variables and additional flags
4. `05_ps_match.py` - Performs propensity score matching

## Usage

To run the cohort builder:

```
python src/01_cohort_builder.py
```

This will process all data files and generate the cohort file.

## Licence
Code: MIT © 2025 <Ryhan Suny>.
Data: CPCSSN; redistribution prohibited under DSA #2025-TMU-SSD.

## Data Source and Provenance

The pipeline uses the most recent checkpoint folder in `Notebooks/data/interim/` (e.g., `checkpoint_1_20250318_024427`), which contains the full prepared data tables (not the 100k sample). These tables are generated from the raw CPCSSN extracts and processed through the data preparation notebooks. The 100k_sample is for development/testing only and should not be used for production analyses.

**To use the full dataset:**
- Ensure the latest checkpoint contains the full prepared data files (see below for table list).
- The pipeline scripts will automatically select the most recent checkpoint.

## Checkpoint Management
- Checkpoints are created by running the data loading and validation notebook(s) in `Notebooks/`.
- Each checkpoint folder contains a README and metadata.json documenting the data provenance, processing steps, and table row counts.
- For reproducibility, always document which checkpoint was used for each analysis. 