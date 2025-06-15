# Pipeline Execution Plan - Remaining Tasks

## Current Status
- Cohort already built with 250,025 patients (created May 16, 2025)
- Note: Cohort and config now aligned to reference date **2015-01-01**
- Python environment needed to run scripts (requirements.txt exists)
- Makefile available with targets for running the pipeline
- Reference date reconciliation pending (2015 vs 2018)
  - Follow-up required with team to finalize

## Phase 1 - Infrastructure (Remaining)
1. **1.1 Cohort Builder Review** ✓
   - Cohort exists with 250,025 patients
   - Temporal discrepancy resolved: config updated to 2015
   - No cohort rebuild required


2. **1.2 Missing Data Engine** (07_missing_data.py)
   - Script exists
   - Needs to be run
   - Will implement miceforest imputation

3. **1.3 Lab Normal Helper** ✓ COMPLETED
   - helpers/lab_utils.py created
   - 06_lab_flag.py updated to use it

4. **1.4 Drug Code Manifest**
   - Need to check if code_lists/drug_atc.csv exists
   - Create if missing with ATC codes from config

## Phase 2 - Data Preparation (Remaining)
1. **2.1 Exposure Flag** (02_exposure_flag.py)
   - Script exists
   - Needs to be run to generate exposure flags
   
2. **2.2 Mediator Autoencoder** (03_mediator_autoencoder.py)
   - Script exists
   - Needs to be run to generate severity index

3. **2.3 Outcome Counter** (04_outcome_flag.py)
   - Script exists, updated with config costs
   - Needs to be run

4. **2.4 Covariate Matrix** ✓ COMPLETED
   - 05_confounder_flag.py implemented
   - Needs to be run

5. **2.5 Referral Sequence** (07_referral_sequence.py)
   - Script exists
   - Needs to be run

6. **2.6 Lab Sensitivity** ✓ COMPLETED
   - 06_lab_flag.py implemented
   - Needs to be run

## Phase 3 - Causal Analysis (Remaining)
1. **3.1 MC-SIMEX** ✓ COMPLETED
   - 07a_misclassification_adjust.py created
   - Needs to be run after exposure flag

2. **3.2 Patient Master** (08_patient_master_table.py)
   - Script exists
   - Needs to be run after all flags generated

## Phase 4 - Matching/Weighting
- All scripts completed, need to be run after data prep

## Phase 5 - Estimation
- All scripts completed, need to be run after matching

## Phase 6 - Sensitivity
- All scripts completed, need to be run after estimation

## Execution Order
1. Temporal alignment resolved (reference date 2015-01-01)
2. Set up Python environment
3. Run Phase 1-2 scripts in order
4. Continue with Phase 3-6

## Environment Setup Needed
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```