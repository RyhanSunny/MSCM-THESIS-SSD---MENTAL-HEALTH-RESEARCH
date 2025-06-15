# Somatic Symptom Disorder Causal Analysis

This repository contains code for a graduate thesis investigating how repeated diagnostics, unresolved referrals, and medication persistence impact healthcare utilization in patients with suspected Somatic Symptom Disorder (SSD), and whether a severity index mediates these effects.

The project uses electronic medical records from the Canadian Primary Care Sentinel Surveillance Network (CPCSSN) to build patient cohorts, derive exposures, and estimate causal effects of repeated diagnostic tests, referrals, and medication patterns.
Raw CPCSSN data are not included in this repository due to privacy restrictions; only derived tables and configuration files are provided.

## Directory overview

- **SSD_Experiment1_Causal_Effect/** – main analysis code, documentation and Makefile
- **prepared_data/** – processed CPCSSN tables used by the pipeline
- **outputs/** – generated figures and intermediate results
- various `.py` and `.ipynb` files at the repository root demonstrate exploratory data preparation and analysis steps

## Key documentation

- [Experiment preregistration](SSD_Exp1_Preregistration.md)
- [Documentation index](SSD_Experiment1_Causal_Effect/docs/README.md)
- [Implementation status](SSD_Experiment1_Causal_Effect/IMPLEMENTATION_STATUS_FINAL.md)
- [Hypotheses report](SSD_Experiment1_Causal_Effect/SSD_Hypotheses_Report.md)
- [Methods blueprint](SSD_Experiment1_Causal_Effect/SSD THESIS final METHODOLOGIES blueprint (1).md)

## Running the pipeline

The full pipeline is managed by a Makefile located in `SSD_Experiment1_Causal_Effect/`. Navigate to that directory and run:

```bash
make all
```

This sequentially executes cohort construction, exposure flagging, mediator derivation, propensity score matching, causal estimators and robustness checks. Other targets such as `make reporting` generate final reports.


## Environment setup

Create a Python environment and install dependencies:

```bash
make -C SSD_Experiment1_Causal_Effect install
```

The `requirements.txt` file includes `pandas`, `numpy`, `pyarrow`, `scikit-learn`, and `pytest` to support the unit tests and analysis scripts.

## Reference date decision

The available CPCSSN checkpoint was generated with a 2015 baseline. To remain consistent with the derived cohort stored in `data_derived/`, the configuration has been aligned to `2015-01-01`. Exposure and outcome windows now span 2015–2017.

## Data gaps and proxies

Postal codes and detailed billing data are not available in the dataset. Costs are estimated using Ontario schedule of benefits values defined in `config.yaml`. Emergency department visits are flagged using an encounter-type keyword algorithm (e.g. `emerg`, `urgent`, `ER visit`).

