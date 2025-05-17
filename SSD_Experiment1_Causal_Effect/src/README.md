# SSD Causal Analysis - Source Code

This directory contains the implementation of the SSD causal analysis pipeline as described in the methodology blueprint.

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