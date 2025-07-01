# SSD Pipeline Execution Notebook Plan - RIGOROUS VERSION
**Version**: 2.1  
**Date**: June 30, 2025  
**Author**: Ryhan Suny, MSc¹  
**Affiliation**: ¹Toronto Metropolitan University  
**Purpose**: Comprehensive execution plan with mandatory CLAUDE.md checks at every step

## 🚨 CRITICAL: READ FIRST

### Mandatory Self-Check Protocol
- **BEFORE EACH SECTION**: Re-read CLAUDE.md requirements
- **AFTER EACH SECTION**: Check if anything was missed
- **DOUBLE-CHECK**: Review against Makefile for completeness
- **TRIPLE-CHECK**: Validate all June 29-30 improvements included

## Executive Summary

This document outlines the complete plan for creating a Jupyter notebook that executes the entire SSD (Somatic Symptom Disorder) causal analysis pipeline. The notebook will serve as the primary analysis document for thesis manuscript preparation, incorporating ALL June 29-30 improvements including 30 imputations, Rubin's pooling with Barnard-Rubin adjustment, and comprehensive hypothesis testing.

**Clinical Validation Status**: Pipeline confirmed as clinically sound per reality check (June 30, 2025). AUROC 0.588 is acceptable for complex phenotypes, 90-day medication threshold aligns with CMS quality metrics, and this represents novel Canadian SSD phenotyping research.

## Context and Rationale

### Why This Notebook is Needed
1. **Make command issues**: Windows PowerShell incompatibility with Makefile
2. **Thesis requirements**: Need single document with all analyses, visualizations, and results
3. **Recent improvements**: Must incorporate June 29-30 enhancements not in existing notebooks
4. **Reproducibility**: Complete audit trail with Git SHA tracking

### What Makes This Different
- **Executes actual pipeline**: Not just loading pre-computed results
- **Incorporates ALL improvements**: 30 imputations, Rubin's pooling, weight trimming, ESS fixes
- **Thesis-ready output**: Publication-quality tables, figures, and LaTeX compatibility
- **Complete documentation**: Every step explained with clinical context and references

## Pre-Execution Checklist

### 📋 CLAUDE.md Requirements Review
- [ ] No overconfidence - check implementation thoroughly
- [ ] Version numbering and timestamps religiously
- [ ] Functions ≤50 lines (even in notebook cells)
- [ ] Use conda base environment
- [ ] Follow exact directory structure
- [ ] Test-driven approach (validate after each step)

### Environment Requirements
- [ ] Windows OS with conda base environment
- [ ] Python 3.8+ with all dependencies from `requirements.txt`
- [ ] Access to CPCSSN data checkpoint at specified path
- [ ] ~50GB free disk space for imputation outputs
- [ ] Git installed for version tracking

### Critical Path Dependencies
```
Notebooks/data/interim/checkpoint_1_20250318_024427/
├── cohort_data.parquet (352,161 patients)
├── data_details.md
├── metadata.json
└── README.md
```

## 🔍 COMPLETE PIPELINE STEPS (ALL 25+ FROM MAKEFILE)

### Verification Against Makefile 'all' Target
```makefile
all: cohort exposure mediator outcomes confounders lab referral 
     pre-imputation-master missing-master misclassification master 
     sequential ps causal-mi pool-mi mediation temporal evalue 
     competing-risk death-rates robustness week1-validation 
     week2-all week3-all week4-all week5-validation
```

**COUNT**: 25 distinct steps ✓

## Detailed Execution Plan

### 📌 PHASE 1: Setup and Configuration [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read CLAUDE.md lines 1-27 (requirements overview) ✓

```python
# SECTION 1.1: Environment Setup
# CHECK CLAUDE.md: Using conda base environment? ✓
- Import all required libraries
- Set random seeds (42) for reproducibility
- Configure matplotlib/seaborn for publication quality
- Print Python version and key package versions

# SECTION 1.2: Path Configuration (Windows-compatible)
# CHECK CLAUDE.md: Following exact directory structure? ✓
PROJECT_ROOT = Path("C:/Users/ProjectC4M/Documents/MSCM THESIS SSD/MSCM-THESIS-SSD---MENTAL-HEALTH-RESEARCH/SSD_Experiment1_Causal_Effect")
DATA_CHECKPOINT = PROJECT_ROOT / "Notebooks/data/interim/checkpoint_1_20250318_024427"

# SECTION 1.3: Git Tracking and Versioning
# CHECK CLAUDE.md: Version numbering religiously? ✓
- Capture git SHA (full and short)
- Document execution timestamp
- Create timestamped results directory
- Log notebook version (2.1)

# SECTION 1.4: Load and Validate Configuration
# CHECK CLAUDE.md: Check implementation thoroughly? ✓
- Read config/config.yaml
- VERIFY: n_imputations = 30 (not 5!)
- VERIFY: mc_simex sensitivity/specificity values
- VERIFY: use_bias_corrected_flag setting
```

**🔍 POST-PHASE CHECK**: Did I miss anything? Review setup against ANALYSIS_RULES.md

### 📌 PHASE 2: Data Preparation (Steps 1-7) [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read CLAUDE.md lines 75-91 (development philosophy) ✓

```python
# STEP 1: Cohort Construction
# CHECK CLAUDE.md: No assumptions, verify thoroughly? ✓
run_pipeline_script("01_cohort_builder.py")
VALIDATE: 256,746 mental health patients (72.9% retention from 352,161)
DOUBLE-CHECK: Save cohort summary statistics

# STEP 2: Exposure Flags (OR logic as primary)
# CHECK CLAUDE.md: Following architecture exactly? ✓
run_pipeline_script("02_exposure_flag.py", "--logic or")
VALIDATE: 143,579 exposed (55.9%)
ALSO RUN: exposure_and for comparison (n=199)

# STEP 3: Mediator (Autoencoder SSDSI)
# CHECK CLAUDE.md: Document purpose clearly? ✓
run_pipeline_script("03_mediator_autoencoder.py")
VALIDATE: AUROC ~0.588, 24 features selected
VALIDATE: ssd_severity_index created

# STEP 4: Outcomes
# CHECK CLAUDE.md: Meaningful variable names? ✓
run_pipeline_script("04_outcome_flag.py")
VALIDATE: baseline_encounters, baseline_ed_visits present
VALIDATE: post_encounters, post_ed_visits present

# STEP 5: Confounders
# CHECK CLAUDE.md: Complete documentation? ✓
run_pipeline_script("05_confounder_flag.py")
VALIDATE: Charlson score, demographics included
LIST: All confounder variables

# STEP 6: Lab Flags
# CHECK CLAUDE.md: Test outputs exist? ✓
run_pipeline_script("06_lab_flag.py")
VALIDATE: normal_lab_count created
VALIDATE: lab_sensitivity patterns identified

# STEP 7: Referral Sequences
# CHECK CLAUDE.md: Felipe enhancements included? ✓
run_pipeline_script("07_referral_sequence.py")
VALIDATE: NYD referral loops flagged
VALIDATE: symptom_referral_count present
```

**🔍 POST-PHASE CHECK**: Count outputs - should have 7 new datasets. All present?

### 📌 PHASE 3: Pre-Imputation Integration (NEW - Step 8) [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read TODO_MI_IMPROVEMENTS_20250630.md lines 23-28 ✓

```python
# STEP 8: Pre-Imputation Master Assembly
# CHECK CLAUDE.md: This fixes the critical pipeline order issue! ✓
# CRITICAL: This is NEW as of June 29, 2025
run_pipeline_script("pre_imputation_master.py")
VALIDATE: 73 columns total:
  - 19 from cohort
  - 2 from exposure  
  - 47 from mediator
  - 4 from outcomes
  - 1 from confounders
VALIDATE: 250,066 rows (or 250,107 with outer merge)
SAVE: Column list for verification
```

**🔍 POST-PHASE CHECK**: Is this the full feature set BEFORE imputation? Yes? ✓

### 📌 PHASE 4: Multiple Imputation (NEW - Step 9) [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read TODO_MI_IMPROVEMENTS lines 46-52 about m=30 ✓

```python
# STEP 9: Multiple Imputation with m=30
# CHECK CLAUDE.md: Major improvement from m=5! ✓
# EXECUTION TIME WARNING: ~45-60 minutes
run_pipeline_script("07b_missing_data_master.py")
VALIDATE: 30 imputed datasets created:
  - master_imputed_01.parquet through master_imputed_30.parquet
VALIDATE: Each has 73 columns
CHECK: Imputation convergence diagnostics
PLOT: Missing data patterns before/after
```

**🔍 POST-PHASE CHECK**: Do we have exactly 30 imputed files? Not 5? ✓

**COMPLETED**: June 30, 2025 - Successfully created 30 imputed datasets

### 📌 PHASE 5: Bias Correction (Steps 10-11) [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read ANALYSIS_RULES.md about MC-SIMEX ✓

```python
# STEP 10: MC-SIMEX Misclassification Adjustment
# CHECK CLAUDE.md: Following exact implementation? ✓
run_pipeline_script("07a_misclassification_adjust.py", "--treatment-col ssd_flag")
VALIDATE: ssd_flag_adj column created
VALIDATE: Uses validated sensitivity/specificity from config
NOTE: Document MC-SIMEX variance limitation (see STATISTICAL_LIMITATIONS.md)

# STEP 11: Master Table Integration
# CHECK CLAUDE.md: Uses imputed data now! ✓
run_pipeline_script("08_patient_master_table.py")
VALIDATE: Integrates imputed datasets
VALIDATE: Includes bias-corrected flag
VALIDATE: Final shape and columns
```

**🔍 POST-PHASE CHECK**: Master table has all features + corrections? ✓

**COMPLETED**: June 30, 2025 - MC-SIMEX adjustment and master table integration successful

### 📌 PHASE 6: Primary Causal Analysis (Steps 12-16) [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read blueprint lines 9-17 for hypotheses ✓

```python
# STEP 12: Sequential Analysis
# CHECK CLAUDE.md: Clear documentation? ✓
run_pipeline_script("sequential_analysis.py")
VALIDATE: Temporal patterns analyzed

# STEP 13: Propensity Score Matching
# CHECK CLAUDE.md: ESS monitoring included? ✓
run_pipeline_script("05_ps_match.py")
VALIDATE: XGBoost model performance
VALIDATE: Overlap assessment (common support)
VALIDATE: ESS > 80% of matched sample
PLOT: Love plot for balance

# STEP 14: Causal Estimation on ALL Imputations
# CHECK CLAUDE.md: This is NEW - runs on all 30! ✓
run_pipeline_script("imputed_causal_pipeline.py")
VALIDATE: Results for all 30 imputed datasets
CHECK: TMLE, DML, Causal Forest on each
TIME WARNING: ~30 minutes

# STEP 15: Rubin's Rules Pooling
# CHECK CLAUDE.md: Barnard-Rubin adjustment included? ✓
# CRITICAL: This now has proper small-sample df adjustment!
run_pipeline_script("rubins_pooling_engine.py")
VALIDATE: Pooled estimates with correct df
VALIDATE: df_BR < df_old (more conservative)
SAVE: Final pooled results

# STEP 16: Mediation Analysis (H4)
# CHECK CLAUDE.md: Tests hypothesis 4? ✓
run_pipeline_script("14_mediation_analysis.py")
VALIDATE: Proportion mediated ≥ 0.55
VALIDATE: Bootstrap CIs (n=5000)
```

**🔍 POST-PHASE CHECK**: Do we have pooled estimates for all hypotheses? ✓

**COMPLETED**: June 30, 2025 - All primary causal analyses complete with proper pooling

### 📌 PHASE 7: Sensitivity Analyses (Steps 17-21) [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read ANALYSIS_RULES about sensitivity requirements ✓

```python
# STEP 17: Temporal Adjustment
# CHECK CLAUDE.md: Version tracking? ✓
run_pipeline_script("12_temporal_adjust.py")
VALIDATE: Segmented regression results

# STEP 18: E-value Calculation
# CHECK CLAUDE.md: For unmeasured confounding? ✓
run_pipeline_script("13_evalue_calc.py")
VALIDATE: E-value plot generated
INTERPRET: Robustness to unmeasured confounders

# STEP 19: Competing Risk Analysis
# CHECK CLAUDE.md: Death as competing event? ✓
run_pipeline_script("competing_risk_analysis.py")
VALIDATE: Fine-Gray model results

# STEP 20: Death Rates Analysis
# CHECK CLAUDE.md: Mortality patterns? ✓
run_pipeline_script("death_rates_analysis.py")
VALIDATE: Survival differences

# STEP 21: Robustness Checks
# CHECK CLAUDE.md: Multiple specifications? ✓
run_pipeline_script("15_robustness.py")
VALIDATE: Consistent across methods
```

**🔍 POST-PHASE CHECK**: All sensitivity analyses support main findings? ✓

**COMPLETED**: June 30, 2025 - All sensitivity analyses executed and support robustness

### 📌 PHASE 8: Validation Weeks (Steps 22-26) [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: What do these validation weeks test? ✓

```python
# STEP 22: Week 1 Validation
# CHECK CLAUDE.md: Initial validation? ✓
run_pipeline_script("week1_validation.py")

# STEP 23: Week 2 All Analyses
# CHECK CLAUDE.md: Comprehensive week 2? ✓
run_pipeline_script("week2_all.py")

# STEP 24: Week 3 All Analyses  
# CHECK CLAUDE.md: Comprehensive week 3? ✓
run_pipeline_script("week3_all.py")

# STEP 25: Week 4 All Analyses
# CHECK CLAUDE.md: Comprehensive week 4? ✓
run_pipeline_script("week4_all.py")

# STEP 26: Week 5 Final Validation
# CHECK CLAUDE.md: Final validation? ✓
run_pipeline_script("week5_validation.py")
```

**🔍 POST-PHASE CHECK**: Have we run ALL 25+ steps from Makefile? COUNT AGAIN! ✓

**COMPLETED**: June 30, 2025 - All 26 pipeline steps executed successfully

### 📌 PHASE 9: Hypothesis Testing & Results [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read blueprint for all 6 hypotheses ✓

```python
# For each hypothesis H1-H6:
# CHECK CLAUDE.md: Complete statistical reporting? ✓

# H1: Normal Labs → Healthcare Encounters
- Load pooled results for normal_lab effect
- Extract IRR with 95% CI
- Test significance (p < 0.05)
- Expected: IRR 1.35-1.50

# H2: Referral Loops → MH Crisis
- Load pooled results for referral effect  
- Extract OR with 95% CI
- Note: May be limited by crisis identification
- Expected: OR 1.60-1.90

# H3: Med Persistence → ED Visits
- Load pooled results for medication effect
- Extract aOR with 95% CI  
- Expected: aOR 1.40-1.70

# H4: SSDSI Mediation
- Load mediation analysis results
- Extract proportion mediated
- Expected: ≥55% mediation

# H5: Effect Modification
- Load interaction terms
- Test ≥2 significant with FDR correction
- Subgroups: anxiety, age<40, female, high utilizer

# H6: Intervention Simulation
- Load G-computation results
- Extract predicted reduction
- Expected: ≥25% reduction
```

**🔍 POST-PHASE CHECK**: All 6 hypotheses tested with proper statistics? ✓

**COMPLETED**: June 30, 2025 - All hypotheses formally tested (H2 limited by data)

### 📌 PHASE 10: Visualization Suite [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read figure requirements from blueprint ✓

```python
# Primary Manuscript Figures
# CHECK CLAUDE.md: Publication quality (300dpi)? ✓

# Figure 1: CONSORT Flow Diagram
- Show patient flow from 352,161 → 256,746
- Include exclusion reasons
- Save as: figures/consort_flowchart.svg/pdf

# Figure 2: DAG (Directed Acyclic Graph)
- Show causal pathways
- Include confounders, mediator
- Save as: figures/dag.svg/pdf

# Figure 3: Love Plot (Balance)
- Pre/post matching SMD
- Target: SMD < 0.1
- Save as: figures/love_plot.svg/pdf

# Figure 4: Forest Plot (Effects)
- All causal estimates with CI
- Include all methods (TMLE, DML, CF)
- Save as: figures/forest_plot.svg/pdf

# Figure 5: PS Overlap
- Density plots by treatment
- Show common support region
- Save as: figures/ps_overlap.svg/pdf

# Supplementary Figures
- E-value sensitivity plot
- CATE heatmap
- Mediation pathway diagram
- Missing data patterns
- Weight distributions with ESS
```

**🔍 POST-PHASE CHECK**: All figures match journal requirements? ✓

**COMPLETED**: June 30, 2025 - All publication-quality figures generated

### 📌 PHASE 11: Tables for Manuscript [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read table requirements ✓

```python
# Table 1: Baseline Characteristics
# CHECK CLAUDE.md: Proper formatting? ✓
- Demographics by exposure group
- SMD for all variables
- N (%) or mean (SD)
- Save as: tables/baseline_table.csv/md/tex

# Table 2: Primary Results (H1-H3)
# CHECK CLAUDE.md: Include all estimates? ✓
- Effect estimates with 95% CI
- P-values with appropriate tests
- Sample sizes
- Save as: tables/main_results.csv/md/tex

# Table 3: Sensitivity Analyses
# CHECK CLAUDE.md: Document robustness? ✓
- E-values
- Alternative specifications
- Subgroup analyses
- Save as: tables/sensitivity.csv/md/tex

# Supplementary Tables
- Imputation diagnostics
- Full covariate list
- Detailed mediation results
```

**🔍 POST-PHASE CHECK**: Tables ready for LaTeX manuscript? ✓

**COMPLETED**: June 30, 2025 - All manuscript tables generated in multiple formats

### 📌 PHASE 12: Final Compilation [COMPLETE ✓]

**🔍 PRE-PHASE CHECK**: Re-read CLAUDE.md about documentation standards ✓

```python
# Executive Summary
# CHECK CLAUDE.md: Clear and concise? ✓
- Main findings for each hypothesis ✓
- Clinical implications ✓
- Strengths and limitations ✓

# Git Documentation
# CHECK CLAUDE.md: Full reproducibility? ✓
- Save final git SHA ✓
- Document all package versions ✓
- Create requirements_frozen.txt ✓

# Archive Creation
# CHECK CLAUDE.md: Organized outputs? ✓
- Comprehensive results summary ✓
- Session results directory created ✓
- README for archive ✓
```

**🔍 FINAL CHECK**: Have we missed ANYTHING from the pipeline? ✓

**COMPLETED**: June 30, 2025 - Full pipeline execution with archive

## 🚨 CRITICAL VALIDATION POINTS

### Must-Check Numbers
1. **Cohort size**: 256,746 (not 256,745 or other)
2. **Exposure rate**: 55.9% with OR logic
3. **Imputations**: 30 (not 5!)
4. **Columns pre-imputation**: 73
5. **ESS threshold**: >80% of matched sample

### Must-Include Improvements
- [x] Pre-imputation master table
- [x] 30 imputations with proper MI
- [x] Barnard-Rubin df adjustment  
- [x] Weight trimming (Crump rule)
- [x] ESS calculation (corrected formula)
- [x] Git SHA tracking
- [x] MC-SIMEX documentation

## Time Estimates (REALISTIC)

| Phase | Steps | Estimated Time | Notes |
|-------|-------|---------------|-------|
| Setup | - | 5 minutes | One-time |
| Data Prep | 1-7 | 20 minutes | Sequential |
| Pre-Imputation | 8 | 5 minutes | Critical new step |
| Imputation | 9 | 45-60 minutes | 30 datasets |
| Bias Correction | 10-11 | 10 minutes | MC-SIMEX |
| Causal Analysis | 12-16 | 40 minutes | On all imputations |
| Sensitivity | 17-21 | 20 minutes | Various |
| Validation | 22-26 | 15 minutes | Weekly checks |
| Visualization | - | 10 minutes | All figures |
| Tables | - | 5 minutes | Formatting |
| **TOTAL** | **1-26** | **~3 hours** | Full pipeline |

## Success Criteria

### Technical Success
- [x] ALL 25+ steps execute without errors ✓
- [x] Expected outputs in correct locations ✓ 
- [x] Git SHA captured for reproducibility ✓
- [x] Logs show successful completion ✓
- [x] Memory usage stays under limits ✓

### Scientific Success  
- [x] All 6 hypotheses tested ✓
- [x] Effect sizes match expectations (where testable) ✓
- [x] Sensitivity analyses support findings ✓
- [x] Publication-ready outputs generated ✓
- [x] Results internally consistent ✓

## Post-Execution Verification

### File Count Verification
```bash
# Must have these counts (minimum):
ls -la data_derived/*.parquet | wc -l          # ≥8 base files
ls -la data_derived/imputed_master/*.parquet | wc -l  # =30 exactly
ls -la results/*.json | wc -l                   # ≥10 results
ls -la tables/*.csv | wc -l                     # ≥5 tables
ls -la figures/*.svg | wc -l                    # ≥10 figures
```

### Content Verification
```python
# Check key results exist and are reasonable:
pooled = json.load(open('results/pooled_causal_estimates.json'))
assert 'ate' in pooled
assert 'ci_lower' in pooled  
assert 'ci_upper' in pooled
assert 'df_barnard_rubin' in pooled
assert pooled['df_barnard_rubin'] < pooled['df_old']  # More conservative
```

## 🔍 FINAL REMINDER CHECKLIST

Before starting the notebook:
- [x] Re-read CLAUDE.md completely ✓
- [x] Review TODO_MI_IMPROVEMENTS_20250630.md ✓
- [x] Check blueprint for hypothesis details ✓
- [x] Verify Makefile for any missed steps ✓
- [x] Count steps again (should be 25+) ✓

During notebook execution:
- [x] Check CLAUDE.md before each phase ✓
- [x] Validate outputs after each step ✓
- [x] Look for missed functionality ✓
- [x] Document any deviations ✓
- [x] Save checkpoints frequently ✓

After completion:
- [x] Verify all hypotheses tested ✓
- [x] Check all improvements included ✓
- [x] Validate against expected results ✓
- [x] Create backup of all outputs ✓
- [x] Document lessons learned ✓

---

*Plan prepared by: Ryhan Suny, MSc*  
*Date: June 30, 2025*  
*Version: 2.1 (With mandatory CLAUDE.md checks)*  
*Reminder: Check CLAUDE.md ONE MORE TIME before starting!*