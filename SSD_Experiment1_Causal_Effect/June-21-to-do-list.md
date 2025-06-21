# June 21 To-Do List

The following tasks consolidate the outstanding issues documented across the assessment and project files. Each item includes a reference to the source lines for context.

- [ ] **Merge misclassification-corrected flag**
  - The MC-SIMEX step produces `cohort_bias_corrected.parquet` but the flag never flows back to `patient_master.parquet`.
  - Source: lines 33–52 in `assesment as of June 21 2025.md`.
- [ ] **Remove socioeconomic variables and synthetic marginals**
  - Transport weights and deprivation analyses rely on unavailable SES data; `ices_marginals.csv` contains placeholder quintiles.
  - Source: lines 132–216 in `assesment as of June 21 2025.md`.
- [ ] **Update documentation to reflect proxy cost data and SES limitations**
  - Mark cost estimates as proxies and clarify missing SES information.
  - Source: lines 207–211 in `assesment as of June 21 2025.md`.
- [ ] **Run production readiness check and clinical validations**
  - Execute `prepare_for_production.py` and send `CLINICAL_VALIDATION_REQUEST.md` for chart review of the SSD phenotype, drug persistence thresholds, ICD code mappings, utilization cut‑offs, and ≥3 normal labs.
  - Source: `COMPREHENSIVE_PROJECT_MAP.md` lines 227–234 and `PRODUCTION_READINESS_CHECKLIST.json` lines 2–33.
- [ ] **Fix Makefile and ensure environment setup**
  - `week5-validation` now appears in the `all` target, but the `install` recipe still uses `requirements.txt`.
  - Modify the `install` target to create the conda environment from `environment.yml`.
  - Source: `COMPREHENSIVE_PROJECT_MAP.md` lines 227–231 and `SSD THESIS final METHODOLOGIES blueprint (1).md` lines 74–78.
- [x] **Complete remaining pipeline modules**
  - `07_missing_data.py`, `07a_misclassification_adjust.py`, `05_ps_match.py`, and `06_causal_estimators.py` are implemented with full logic.
  - Source: `SSD THESIS final METHODOLOGIES blueprint (1).md` lines 46–68.
- [x] **Ensure weight diagnostics, clustered SEs, Poisson regression, multiple imputation, and temporal checks are active**
  - Verified in the code base: `weight_diagnostics` in `05_ps_match.py`, cluster‑robust SEs in `06_causal_estimators.py`, Rubin-style imputation in `07_missing_data.py`, and segmented regression in `12_temporal_adjust.py`.
  - Source: `JUNE-16-MAX-EVAL.md` lines 82–104 and 478–516.
- [x] **Improve autoencoder performance toward AUROC ≥0.7**
  - `simple_autoencoder_retrain.py` and its tests show AUROC >0.84.
  - Source: `COMPREHENSIVE_PROJECT_MAP.md` line 214.
- [x] **Align power analysis narrative with YAML values**
  - `power_analysis_sync.py` and tests confirm YAML parameters match the documentation.
  - Source: `JUNE-16-MAX-EVAL.md` lines 315–330.
- [ ] **Acquire real ICES population marginals (optional)**
  - Synthetic data limits external validity; request official age, sex, and region frequencies.
  - Source: `assesment as of June 21 2025.md` lines 189–217 and `PRODUCTION_READINESS_CHECKLIST.json` lines 34–41.
- [ ] **Prepare manuscript materials (figures, tables, STROBE checklist)**
  - After final analyses, generate the required output for a Q1 journal submission.
  - Source: `JUNE-16-MAX-EVAL.md` lines 620–664.

*Note: Do not introduce synthetic or dummy data while completing these tasks. The goal is a production‑ready analysis using real CPCSSN records.*
