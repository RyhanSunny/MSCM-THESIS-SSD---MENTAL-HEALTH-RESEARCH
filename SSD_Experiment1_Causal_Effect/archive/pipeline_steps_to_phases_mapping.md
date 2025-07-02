# Pipeline Steps to Phases Mapping

**Date**: December 2024  
**Purpose**: Comprehensive mapping of 26 pipeline steps to 12 execution phases  
**Notebook**: SSD_Complete_Pipeline_Analysis_v2_CLEAN.ipynb

## Summary

The SSD pipeline contains **26 distinct steps** distributed across **12 phases** in the execution notebook. The clean notebook has been properly reordered with all phases appearing sequentially from 1-12.

## Detailed Mapping

### PHASE 1: Setup and Configuration
*No numbered steps - preliminary setup only*
- Environment setup
- Path configuration  
- Git tracking
- Configuration loading

### PHASE 2: Data Preparation (Steps 1-7)
1. **Step 1**: Cohort Construction (`01_cohort_builder.py`)
2. **Step 2**: Exposure Flags (`02_exposure_flag.py`)
3. **Step 3**: Mediator Autoencoder (`03_mediator_autoencoder.py`)
4. **Step 4**: Outcomes Generation (`04_outcome_flag.py`)
5. **Step 5**: Confounders Matrix (`05_confounder_flag.py`)
6. **Step 6**: Lab Flags (`06_lab_flag.py`)
7. **Step 7**: Referral Sequences (`07_referral_sequence.py`)

### PHASE 3: Pre-Imputation Integration (Step 8)
8. **Step 8**: Pre-Imputation Master Assembly (`pre_imputation_master.py`)

### PHASE 4: Multiple Imputation (Step 9)
9. **Step 9**: Multiple Imputation with m=30 (`07b_missing_data_master.py`)

### PHASE 5: Bias Correction (Steps 10-11)
10. **Step 10**: MC-SIMEX Misclassification Adjustment (`07a_misclassification_adjust.py`)
11. **Step 11**: Master Table Integration (`08_patient_master_table.py`)

### PHASE 6: Primary Causal Analysis (Steps 12-16)
12. **Step 12**: Sequential Analysis (`08_sequential_pathway_analysis.py`)
13. **Step 13**: Propensity Score Matching (`05_ps_match.py`)
14. **Step 14**: Causal Estimation on All Imputations (`imputed_causal_pipeline.py`)
15. **Step 15**: Rubin's Rules Pooling (`rubins_pooling_engine.py`)
16. **Step 16**: Mediation Analysis (`14_mediation_analysis.py`)

### PHASE 7: Sensitivity Analyses (Steps 17-21)
17. **Step 17**: Temporal Adjustment (`12_temporal_adjust.py`)
18. **Step 18**: E-value Calculation (`13_evalue_calc.py`)
19. **Step 19**: Competing Risk Analysis (`finegray_competing.py`)
20. **Step 20**: Death Rates Analysis (`death_rates_analysis.py`)
21. **Step 21**: Robustness Checks (`15_robustness.py`)

### PHASE 8: Validation Weeks (Steps 22-26)
22. **Step 22**: Week 1 Validation (`week1-validation` target)
23. **Step 23**: Week 2 All Analyses (`week2-all` target)
24. **Step 24**: Week 3 All Analyses (`week3-all` target)
25. **Step 25**: Week 4 All Analyses (`week4-all` target)
26. **Step 26**: Week 5 Final Validation (`week5-validation` target)

### PHASE 9: Hypothesis Testing & Results
*Uses results from previous steps - no new numbered steps*
- Tests hypotheses H1-H6 using pooled results

### PHASE 10: Visualization Suite
*Post-processing phase - no new numbered steps*
- Generates publication-quality figures from results

### PHASE 11: Tables for Manuscript
*Post-processing phase - no new numbered steps*
- Creates formatted tables from results

### PHASE 12: Final Compilation
*Summary phase - no new numbered steps*
- Executive summary
- Git documentation
- Archive creation

## Verification Against Makefile

The Makefile `all` target includes these components:
```
cohort, exposure, mediator, outcomes, confounders, lab, referral,
pre-imputation-master, missing-master, misclassification, master,
sequential, ps, causal-mi, pool-mi, mediation, temporal, evalue,
competing-risk, death-rates, robustness, week1-validation,
week2-all, week3-all, week4-all, week5-validation
```

**Count**: 26 components ✓

## Key Observations

1. **Steps 1-21**: Core pipeline execution steps with specific Python scripts
2. **Steps 22-26**: Validation week targets that run comprehensive test suites
3. **Phases 9-12**: Post-processing phases that analyze and present results from earlier steps
4. **Critical New Steps**: 
   - Step 8 (pre-imputation master) - Added June 29, 2025
   - Step 14 (causal on all imputations) - Runs on all 30 datasets
   - Step 15 (Rubin's pooling) - Includes Barnard-Rubin adjustment

## Execution Dependencies

- Steps 1-7 must run sequentially to build feature sets
- Step 8 integrates all features before imputation
- Step 9 creates 30 imputed datasets
- Steps 10-11 apply corrections and create final master
- Steps 12-16 perform causal analyses
- Steps 17-21 test robustness
- Steps 22-26 validate the entire pipeline

This mapping confirms that all 26 steps from the Makefile are properly distributed across the 12 phases in the execution notebook.