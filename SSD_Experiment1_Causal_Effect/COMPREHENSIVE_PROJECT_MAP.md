# Comprehensive SSD Causal Inference Project Map
**Date**: June 21, 2025  
**Author**: Ryhan Suny  
**Project**: Master's Thesis - Somatic Symptom Disorder Healthcare Utilization Analysis

---

## Executive Summary

This project implements a **comprehensive causal inference pipeline** analyzing healthcare utilization patterns in Somatic Symptom Disorder (SSD) patients using real-world healthcare data (352,161 patients). The pipeline combines traditional epidemiological methods with modern ML/AI approaches for robust causal effect estimation.

**Status**: 95% technically complete, awaiting clinical validation for production deployment.

---

## 🎯 Project Entry Points

### 1. **Main Pipeline Entry** 
```bash
make all
```
- **Purpose**: Execute complete Week 1-5 analysis pipeline
- **Runtime**: ~24 hours for full dataset
- **Output**: Results in `results/` and `figures/` directories

### 2. **Individual Week Execution**
```bash
make week1-validation  # Data processing & validation
make week2-all         # Causal identification  
make week3-all         # Advanced methods
make week4-all         # Statistical refinements
make week5-validation  # Final compliance checks
```

### 3. **Production Readiness Check**
```bash
python prepare_for_production.py
```
- **Purpose**: Validate clinical parameters before full pipeline execution
- **Output**: `PRODUCTION_READINESS_CHECKLIST.json`

### 4. **Clinical Validation**
- **Document**: `CLINICAL_VALIDATION_REQUEST.md`
- **Purpose**: Plain English validation request for MDs
- **Timeline**: 1 week for clinical team review

---

## 🏗️ Project Architecture Map

### **Directory Structure**
```
SSD_Experiment1_Causal_Effect/
├── src/                          # Core analysis modules (43 files)
├── Notebooks/                    # Jupyter analysis notebooks
│   └── data/interim/checkpoint_1_20250318_024427/  # Patient data (352,161 records)
├── config/                       # Configuration files
├── results/                      # Analysis outputs
├── figures/                      # Generated visualizations
├── tests/                        # Unit and integration tests
├── docs/                         # Project documentation
└── Makefile                      # Pipeline orchestration
```

---

## 📊 Data Pipeline Architecture

### **Core Data Flow**
```
Raw Healthcare Data → Data Processing → Feature Engineering → ML/AI Analysis → Causal Inference → Results
     (352,161)           (Week 1)         (Week 2)         (Week 3)        (Week 4-5)     (Output)
```

### **Key Data Assets**
- **Patient Cohort**: 352,161 patients from checkpoint data
- **Encounters**: 11.5M healthcare encounters
- **Medications**: 7.7M medication records  
- **Lab Results**: 8.5M laboratory tests
- **Referrals**: 1.1M specialist referrals

---

## 🔬 Analysis Module Map

### **Week 1: Data Foundation (Files 01-04)**
| Module | Purpose | Output | Status |
|--------|---------|--------|--------|
| `01_cohort_builder.py` | Mental health cohort identification | Patient IDs with inclusion criteria | ✅ Complete |
| `02_exposure_flag.py` | SSD pattern detection | Treatment assignment flags | ✅ Complete |
| `03_mediator_autoencoder.py` | **ML**: SSD severity index via autoencoder | 16-dim severity scores | ⚠️ AUROC=0.562 (needs improvement) |
| `04_outcome_flag.py` | Healthcare utilization outcomes | 6 outcome measures | ✅ Complete |

### **Week 2: Causal Identification (Files 05-08)**
| Module | Purpose | Output | Status |
|--------|---------|--------|--------|
| `05_ps_match.py` | **ML**: XGBoost propensity scores + SHAP | Treatment probabilities | ✅ Complete |
| `06_causal_estimators.py` | **ML**: TMLE, DML, Causal Forest | Treatment effects | ✅ Complete |
| `07_missing_data.py` | **ML**: Multiple imputation with RF | Complete datasets | ✅ Complete |
| `08_misclass_adjust.py` | MC-SIMEX misclassification adjustment | Corrected estimates | ⚠️ Needs clinical validation |

### **Week 3: Advanced Methods (Files 09-11)**
| Module | Purpose | Output | Status |
|--------|---------|--------|--------|
| `09_master_table.py` | Integrated analysis results | Master results table | ✅ Complete |
| `10_sequential_ps.py` | Time-varying propensity scores | Dynamic treatment models | ✅ Complete |
| `11_transport_weights.py` | External validity weighting | Transportability weights | ⚠️ ICES data missing |

### **Week 4: Statistical Refinements (Files 12-15)**
| Module | Purpose | Output | Status |
|--------|---------|--------|--------|
| `12_temporal_adjust.py` | **ML**: Marginal structural models | Time-adjusted effects | ✅ Complete |
| `13_evalue_calculation.py` | Sensitivity analysis | E-values for unmeasured confounding | ✅ Complete |
| `14_mediation_analysis.py` | **ML**: DoWhy mediation pathways | Indirect effect estimates | ✅ Complete |
| `15_robustness.py` | **ML**: Ensemble robustness checks | Model stability metrics | ✅ Complete |

### **Week 5: Validation & QA (Files 16+)**
| Module | Purpose | Output | Status |
|--------|---------|--------|--------|
| `week5_compliance_checker.py` | Pipeline validation | QA compliance reports | ✅ Complete |
| `poisson_count_models.py` | **ML**: Count model validation | Dispersion diagnostics | ✅ Complete |
| `cluster_robust_se.py` | Clustered standard errors | Corrected confidence intervals | ✅ Complete |
| `weight_diagnostics.py` | Propensity score diagnostics | Balance assessment | ✅ Complete |

---

## 🤖 Machine Learning & AI Implementation Map

### **1. Deep Learning (TensorFlow/Keras)**
**Purpose**: SSD Severity Index Generation
- **Files**: `03_mediator_autoencoder.py`, `retrain_autoencoder.py`
- **Architecture**: Sparse autoencoder (56→32→16→32→56)
- **Current Performance**: AUROC=0.562 (needs improvement to ≥0.7)
- **Clinical Application**: Patient severity stratification

### **2. Gradient Boosting (XGBoost)**
**Purpose**: Propensity Score Estimation
- **Files**: `05_ps_match.py`
- **Features**: GPU acceleration, hyperparameter optimization
- **Performance**: Superior covariate balance vs logistic regression
- **Interpretability**: SHAP values for model explanation

### **3. Ensemble Methods (Random Forest)**
**Purpose**: Multiple applications across pipeline
- **Files**: `06_causal_estimators.py`, `07_missing_data.py`, `advanced_analyses.py`
- **Applications**: 
  - TMLE nuisance parameter estimation
  - Missing data imputation (MICE)
  - Causal forest heterogeneous effects
  - DML cross-fitting

### **4. Causal Machine Learning (EconML/DoWhy)**
**Purpose**: Advanced causal inference
- **Files**: `06_causal_estimators.py`, `14_mediation_analysis.py`
- **Methods**: Double ML, Causal Forest, Mediation analysis
- **Advantage**: Robust to model misspecification

### **5. Interpretable AI (SHAP)**
**Purpose**: Model explanation and clinical interpretability
- **Files**: `05_ps_match.py`
- **Output**: Feature importance rankings, decision explanations
- **Clinical Value**: Understanding treatment assignment factors

---

## 📈 Results and Output Map

### **Primary Results Structure**
```
results/
├── week1_descriptive/           # Cohort characteristics
├── week2_causal_effects/        # Main treatment effects
├── week3_advanced_methods/      # Robustness checks
├── week4_sensitivity/           # E-values, mediation
├── week5_validation/           # QA reports
└── master_results_table.csv    # Combined findings
```

### **Key Deliverables**
1. **Treatment Effect Estimates**: 6 healthcare utilization outcomes
2. **Mediation Analysis**: SSD severity pathway effects  
3. **Heterogeneous Effects**: Subgroup analysis results
4. **Sensitivity Analysis**: E-values, robustness checks
5. **Model Diagnostics**: Balance, performance metrics

---

## 🔧 Configuration and Dependencies

### **Configuration Files**
- **`config/config.yaml`**: Master configuration
  - MC-SIMEX parameters (needs clinical validation)
  - Model hyperparameters
  - File paths and naming conventions

### **Key Dependencies**
- **Core ML**: scikit-learn, xgboost, tensorflow
- **Causal Inference**: dowhy, econml
- **Statistics**: statsmodels, scipy
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly

---

## ⚠️ Critical Gaps Requiring Attention

### **1. Clinical Validation Required**
- **MC-SIMEX Parameters**: Currently 0.82/0.82 from literature
- **Drug Persistence Thresholds**: 180 days across all classes
- **Severity Threshold Validation**: Chart review needed
- **Document**: `CLINICAL_VALIDATION_REQUEST.md`

### **2. Technical Improvements**
- **Autoencoder Performance**: AUROC=0.562 → target ≥0.7
- **ICES External Data**: Synthetic data blocks transportability
- **Makefile Sync**: Add week5-validation to 'all' target

### **3. Data Limitations**
- **Sample vs Full Cohort**: Using 352k of potentially larger cohort
- **Checkpoint Data**: March 2025 snapshot, not real-time

---

## 🚀 What Can Be Run Today

### **Immediate Actions Available**
1. **Fix Makefile**: Add week5-validation to line 17
   ```bash
   # Update Makefile line 17 to include week5-validation
   ```

2. **Run Production Check**:
   ```bash
   python prepare_for_production.py
   ```

3. **Execute Individual Weeks** (with current data):
   ```bash
   make week1-validation  # ~2 hours
   make week2-all         # ~4 hours  
   make week3-all         # ~6 hours
   make week4-all         # ~8 hours
   make week5-validation  # ~2 hours
   ```

4. **Generate Clinical Validation Request**:
   - Send `CLINICAL_VALIDATION_REQUEST.md` to clinical team
   - Timeline: 1 week for validation

### **Full Pipeline Execution** (after clinical validation):
```bash
make clean && make all  # ~24 hours total runtime
```

---

## 📝 Academic Output Preparation

### **Methods Section Ready**
- Traditional epidemiological methods (propensity scores, TMLE)
- Justified ML enhancements (XGBoost, autoencoders, causal forest)
- Rigorous validation and sensitivity analysis

### **Key References Implemented**
- Hernán & Robins (2020): Causal Inference framework
- Austin (2011): Propensity score methods
- Athey & Wager (2019): Causal forest implementation
- VanderWeele & Ding (2017): E-value sensitivity analysis

### **Defense-Ready Explanations**
- Clear justification for ML vs traditional method choices
- Clinical interpretability maintained throughout
- Comprehensive robustness testing implemented

---

## 📞 Next Steps Summary

1. **Immediate** (Today): Fix Makefile, run production check
2. **Week 1**: Clinical validation of parameters
3. **Week 2**: Execute full pipeline with validated parameters
4. **Week 3**: Results analysis and manuscript preparation

**Timeline to Completion**: 2-3 weeks after clinical validation

---

*This comprehensive map provides complete navigation through the SSD causal inference pipeline, enabling both technical implementation and academic defense preparation.*