# SSD Experiment 1: Causal Analysis Pipeline

This repository contains the analysis code for investigating how **repeated normal lab results**, **unresolved specialist referrals**, and **persistent medication use** may causally increase healthcare utilization in patients with Somatic Symptom Disorder (SSD). We also evaluate a composite severity index as a potential mediator or modifier of this relationship. The project is part of the MScM thesis work at Toronto Metropolitan University.

## Project Goals

* Determine whether repeated normal lab results, unresolved referrals, and prolonged medication exposure drive additional healthcare utilization.
* Measure how these exposures influence the composite SSD severity index.
* Demonstrate a reproducible causal analysis framework for primary care data.

## Environment Setup

Use a Python virtual environment and install the required dependencies. The
commands below appear exactly as listed in `PIPELINE_STATUS_2025-05-25.md`
(lines 70‑75). Change into the `SSD_Experiment1_Causal_Effect` directory first
and then run:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Executing the Pipeline

All pipeline scripts live in `SSD_Experiment1_Causal_Effect/src/`. After activating the environment, run the scripts sequentially:

1. `02_exposure_flag.py`
2. `03_mediator_autoencoder.py`
3. `04_outcome_flag.py`
4. `05_confounder_flag.py`
5. `06_lab_flag.py`
6. `07_missing_data.py`
7. `07_referral_sequence.py`
8. `08_patient_master_table.py`
9. Continue with the Phase 4–6 scripts (`05_ps_match.py`, `12_temporal_adjust.py`, `06_causal_estimators.py`, `13_evalue_calc.py`, `14_placebo_tests.py`, `15_robustness.py`)

Lines 90‑95 of `PIPELINE_STATUS_2025-05-25.md` highlighted that a root `README.md` was missing and documented the above execution order.

You can also use the provided `Makefile` in `SSD_Experiment1_Causal_Effect` to run grouped targets (`make all`, `make cohort`, `make causal`, etc.).

## Outputs

The raw prepared tables live in `prepared_data/` while intermediate cohort
outputs and analysis artefacts are saved under
`SSD_Experiment1_Causal_Effect/data_derived/` and
`SSD_Experiment1_Causal_Effect/results/`. Configuration options are controlled
through `SSD_Experiment1_Causal_Effect/config/config.yaml`.

---
*Documentation created May 2025 to address missing README noted in the pipeline status report.*
