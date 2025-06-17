# RYHAN HUMAN ATTENTION TASKS - Production Pipeline Validation

**🚨 CRITICAL NOTICE**: Production-grade pipeline per CLAUDE.md - NO simulated data, placeholders, or fabricated values allowed.

**Date Created**: June 17, 2025  
**Author**: Claude AI Assistant  
**Priority**: CRITICAL - Production System Blockers  
**Status**: COMPREHENSIVE VALIDATION REQUIRED

---

## 📋 EXECUTIVE SUMMARY

**Total Critical Issues Found**: 47 areas requiring human validation  
**Production Blockers**: 12 critical items  
**Clinical Validation Required**: 23 items  
**Data Integration Issues**: 8 items  
**Technical Validation**: 4 items  

**⚠️ PRODUCTION READINESS**: Currently 0% - Cannot proceed without human validation

---

## 🚨 CRITICAL PRODUCTION BLOCKERS

### **BLOCKER 1: MC-SIMEX Validation Parameters**
**Status**: ❌ **PLACEHOLDER VALUES IN PRODUCTION**  
**Impact**: All bias-corrected analyses invalid  
**Location**: `config/config.yaml` lines 97-98

#### Current State:
```yaml
mc_simex:
  sensitivity: 0.82  # ❌ PLACEHOLDER FROM LITERATURE
  specificity: 0.82  # ❌ PLACEHOLDER FROM LITERATURE
```

#### Required Actions:
1. **Execute SSD Phenotype Validation Notebook**
   - **File**: `Notebooks/SSD_Phenotype_Validation.ipynb`
   - **Process**: Follow `Notebooks/ssd_phenotyping_validation_process.png`
   - **Requirement**: Clinical expert chart review of 200 patients

2. **Clinical Expert Review Protocol**
   ```
   Step 1: Generate validation sample
   ✓ Run notebook cells 1-5 to create stratified sample
   ✓ Export: ssd_validation_reviews.csv
   
   Step 2: Expert clinical review (2-3 days)
   ✓ Recruit psychiatrist/psychologist with SSD expertise
   ✓ Apply DSM-5 criteria to each case
   ✓ Document reasoning for uncertain cases
   
   Step 3: Calculate real metrics
   ✓ Update notebook with expert decisions
   ✓ Run cells 11-13 for final metrics
   ✓ Replace placeholder values in config.yaml
   ```

3. **Validation Quality Gates**
   - Sensitivity ≥ 0.70 (acceptable threshold)
   - Specificity ≥ 0.80 (preferred threshold)
   - Inter-rater reliability if multiple reviewers
   - Confidence intervals width ≤ 0.10

#### Expected Timeline: 1-2 weeks

---

### **BLOCKER 2: ICES External Validity Data**
**Status**: ❌ **STUB DATA IN PRODUCTION**  
**Impact**: Cannot assess external validity  
**Location**: `data/external/ices_marginals.csv`

#### Current State:
```csv
# ❌ FAKE/ROUNDED PROPORTIONS
age_group,18-34,0.25
age_group,35-49,0.22
# ... clearly fabricated data
```

#### Required Actions:
1. **Data Access Negotiation**
   ```
   Step 1: ICES contact and data request
   ✓ Contact ICES Data & Analytic Services
   ✓ Submit formal data sharing agreement
   ✓ Specify: age-sex-Charlson marginal frequencies
   
   Step 2: Legal/ethics clearance
   ✓ REB approval for external data linkage
   ✓ Data sharing agreement execution
   ✓ Privacy impact assessment
   
   Step 3: Data integration
   ✓ Receive real ICES marginal frequencies
   ✓ Replace stub file
   ✓ Test transport_weights.py functionality
   ```

2. **Alternative Solution**
   ```
   If ICES data unavailable:
   ✓ Document limitation in manuscript
   ✓ Use published Ontario demographic data
   ✓ Perform sensitivity analysis on assumptions
   ```

#### Expected Timeline: 2-6 months (depends on data access)

---

### **BLOCKER 3: Full Dataset Integration**
**Status**: ❌ **USING CHECKPOINT SAMPLE DATA**  
**Impact**: All results based on subset, not full cohort  
**Location**: Using `checkpoint_1_20250318_024427/`

#### Current State:
- Pipeline runs on sample/checkpoint data
- Full CPCSSN dataset not integrated
- Unknown performance at production scale

#### Required Actions:
1. **Production Data Infrastructure**
   ```
   Step 1: Data environment setup
   ✓ Configure conda base environment as per CLAUDE.md
   ✓ Ensure adequate memory (>32GB recommended)
   ✓ Verify storage capacity (>500GB)
   
   Step 2: Full dataset acquisition
   ✓ Obtain complete CPCSSN dataset
   ✓ Verify patient count ≥ 250,000
   ✓ Validate data completeness per data_details.md
   
   Step 3: Pipeline scaling validation
   ✓ Test 01_cohort_builder.py on full dataset
   ✓ Monitor memory usage and performance
   ✓ Validate all downstream modules
   ```

2. **Performance Validation**
   ```
   Execute full pipeline:
   make clean && make all
   
   Validate:
   ✓ No memory errors
   ✓ Processing time < 24 hours
   ✓ All outputs generated
   ✓ Result consistency with sample data
   ```

#### Expected Timeline: 1 week

---

## 🏥 CLINICAL VALIDATION REQUIREMENTS

### **CLINICAL-1: Mental Health ICD Code Validation**
**Expert Required**: Psychiatrist  
**Status**: ❌ **REQUIRES CLINICAL REVIEW**

#### Scope:
**File**: `src/mh_cohort_builder.py` lines 53-98
```python
# Current mappings - NEED EXPERT VALIDATION
'depression': ['F32', 'F33', 'F34', '296.2', '296.3']
'anxiety': ['F40', 'F41', 'F42', '300.0*']  
'ptsd': ['F43', '308.*', '309.*']
'bipolar': ['F31', '296.0', '296.1', '296.4']
```

#### Validation Protocol:
```
Step 1: Expert code review (1 day)
✓ Review ICD-9/10 mapping completeness
✓ Validate against current DSM-5-TR criteria
✓ Check for missing/inappropriate codes
✓ Assess sensitivity vs specificity trade-offs

Step 2: Population validation (2 days)  
✓ Sample 100 patients with each code category
✓ Manual chart review for diagnostic accuracy
✓ Calculate positive predictive value by code
✓ Recommend adjustments if PPV < 80%

Step 3: Implementation
✓ Update code mappings per expert recommendations
✓ Re-run cohort builder with validated codes
✓ Document clinical rationale for code selection
```

#### Deliverables:
- [ ] Expert-validated ICD code mappings
- [ ] Clinical rationale documentation
- [ ] Updated `mh_cohort_builder.py`
- [ ] Validation report with PPV/sensitivity metrics

---

### **CLINICAL-2: Drug Persistence Thresholds**
**Expert Required**: Psychiatrist + Clinical Pharmacist  
**Status**: ❌ **CLINICAL ASSUMPTIONS NEED VALIDATION**

#### Scope:
**Current**: 180-day persistence threshold (enhanced from 90)  
**Issue**: No clinical validation for SSD population

#### Validation Protocol:
```
Step 1: Literature review (2 days)
✓ Review SSD medication persistence studies
✓ Compare with anxiety/depression persistence
✓ Assess 90 vs 180-day appropriateness
✓ Consider medication-specific variations

Step 2: Clinical expert consultation (1 day)
✓ Psychiatrist: Clinical appropriateness
✓ Pharmacist: Prescription pattern analysis
✓ Consider patient adherence patterns
✓ Assess different drug classes (N06A, N03A, N05A)

Step 3: Data-driven validation (3 days)
✓ Analyze actual prescription patterns in cohort
✓ Plot distribution of treatment episodes
✓ Identify natural break points
✓ Compare outcomes by persistence definition
```

#### Critical Questions:
- Is 180 days clinically meaningful for SSD?
- Should persistence vary by medication class?
- How to handle dose changes/titration periods?
- What constitutes "inappropriate" vs "appropriate" use?

---

### **CLINICAL-3: Laboratory Normal Range Validation**
**Expert Required**: Clinical Laboratory Medicine Specialist  
**Status**: ⚠️ **USING GENERIC RANGES**

#### Scope:
**File**: `src/helpers/lab_utils.py` lines 43-114  
**Issue**: Generic normal ranges may not reflect local lab standards

#### Validation Protocol:
```
Step 1: Local lab standard review (1 day)
✓ Obtain current normal ranges from CPCSSN labs
✓ Compare with implemented ranges
✓ Identify discrepancies > 10%
✓ Assess clinical significance of differences

Step 2: Age/sex stratification review (2 days)
✓ Validate age-specific ranges (e.g., creatinine)
✓ Review sex-specific ranges (e.g., hemoglobin)
✓ Consider pregnancy exclusions
✓ Assess pediatric vs adult thresholds

Step 3: Clinical context validation (1 day)
✓ Review "normal" definition for SSD context
✓ Consider subclinical abnormalities
✓ Assess impact on exposure definition
✓ Validate reference lab methodology
```

---

### **CLINICAL-4: Healthcare Utilization Thresholds**
**Expert Required**: Health Services Researcher + Emergency Medicine  
**Status**: ❌ **ARBITRARY PERCENTILE THRESHOLDS**

#### Scope:
**Current**: 75th percentile for "high utilization"  
**Issue**: No clinical validation for SSD population

#### Validation Protocol:
```
Step 1: Clinical appropriateness review (2 days)
✓ Review SSD utilization patterns literature
✓ Compare with other chronic conditions
✓ Assess emergency vs planned care patterns
✓ Validate psychiatric ED visit definition

Step 2: Threshold optimization (3 days)
✓ Plot utilization distributions
✓ Test multiple percentile thresholds
✓ Assess clinical outcome correlations
✓ Optimize sensitivity/specificity for SSD

Step 3: Utilization pathway validation (2 days)
✓ Map typical SSD patient journey
✓ Validate referral pattern logic
✓ Assess appropriateness of care classification
✓ Review "healthcare cascade" hypothesis
```

---

## 💊 MEDICATION AND CLINICAL CRITERIA

### **MED-1: Enhanced Drug Class Validation**
**Expert Required**: Psychiatrist + Clinical Pharmacist  
**Clinical Decision**: Dr. Felipe enhancement needs validation

#### Scope:
**File**: `config/config.yaml` lines 43-45
```yaml
# Enhanced drug classes - NEED CLINICAL VALIDATION
antidepressant: ["N06A"]    # All antidepressants
anticonvulsant: ["N03A"]    # Gabapentin/pregabalin
antipsychotic: ["N05A"]     # Off-label anxiety use
```

#### Validation Protocol:
```
Step 1: Clinical appropriateness (2 days)
✓ Review off-label use patterns for SSD
✓ Validate N03A (anticonvulsants) for anxiety/pain
✓ Assess N05A (antipsychotics) appropriateness
✓ Consider dosing patterns vs indication

Step 2: Prescription pattern analysis (3 days)
✓ Analyze actual prescribing in cohort
✓ Identify appropriate vs inappropriate use
✓ Assess duration patterns by drug class
✓ Validate persistence definitions by class

Step 3: Clinical outcome correlation (2 days)
✓ Assess outcomes by drug class
✓ Validate association with SSD symptoms
✓ Review adverse effect patterns
✓ Confirm clinical hypothesis validity
```

---

## 🔬 STATISTICAL AND METHODOLOGICAL VALIDATION

### **STAT-1: Causal Inference Method Validation**
**Expert Required**: Causal Inference Specialist/Biostatistician  
**Status**: ❌ **COMPLEX METHODS NEED EXPERT REVIEW**

#### Scope:
**Files**: `src/06_causal_estimators.py`, `src/14_mediation_analysis.py`

#### Validation Protocol:
```
Step 1: Method appropriateness (3 days)
✓ Review TMLE implementation for binary treatment
✓ Validate Double ML parameter tuning
✓ Assess Causal Forest configuration
✓ Review mediation analysis assumptions

Step 2: Assumption validation (2 days)
✓ Assess positivity assumption
✓ Review exchangeability assumptions
✓ Validate consistency assumption
✓ Check temporal ordering

Step 3: Sensitivity analysis review (2 days)
✓ Review E-value calculations
✓ Assess MC-SIMEX implementation
✓ Validate bias analysis approaches
✓ Review multiple testing corrections
```

---

### **STAT-2: Sample Size and Power Validation**
**Expert Required**: Biostatistician  
**Status**: ⚠️ **POWER CALCULATIONS NEED VERIFICATION**

#### Scope:
**Files**: `config/config.yaml`, power analysis documentation

#### Validation Protocol:
```
Step 1: Power calculation review (2 days)
✓ Validate effect size assumptions (0.2 vs 1.05 RR)
✓ Review sample size justification
✓ Assess multiple testing impact
✓ Validate minimum detectable effect

Step 2: Design efficiency (1 day)
✓ Review propensity score efficiency
✓ Assess matching vs weighting
✓ Validate bootstrap sample sizes
✓ Review confidence interval coverage
```

---

## 📊 DATA QUALITY AND INTEGRATION

### **DATA-1: NYD Code Investigation**
**Expert Required**: Clinical Informaticist + General Internist  
**Status**: ❌ **UNEXPECTEDLY LOW NYD CODE FREQUENCY**

#### Issue:
**Current**: Only 17 NYD codes found (expected more)  
**Location**: H2 referral loop analysis

#### Investigation Protocol:
```
Step 1: Code completeness review (2 days)
✓ Review NYD code list in config.yaml
✓ Cross-reference with CPCSSN coding practices
✓ Check for alternative "symptom" codes
✓ Assess coding variation across sites

Step 2: Clinical pattern analysis (3 days)
✓ Manual review of low-frequency finding
✓ Assess clinical plausibility
✓ Review alternative symptom identification
✓ Validate referral pattern logic

Step 3: Methodology adjustment (1 day)
✓ Consider broader symptom code inclusion
✓ Assess impact on exposure definition
✓ Update NYD identification if needed
✓ Document rationale for code selection
```

---

## 🎯 HYPOTHESIS-SPECIFIC VALIDATIONS

### **H1: Diagnostic Cascade Validation**
**Expert Required**: General Internist + Health Services Researcher

#### Clinical Assumptions to Validate:
```
1. Normal labs → increased healthcare seeking
2. 3+ normal labs = clinically meaningful threshold  
3. Healthcare "cascade" hypothesis validity
4. Temporal ordering of events
```

#### Validation Steps:
```
Step 1: Clinical pathway review (2 days)
✓ Map typical SSD diagnostic journey
✓ Validate "normal lab" significance
✓ Assess healthcare seeking behavior
✓ Review cascade hypothesis literature

Step 2: Threshold optimization (2 days)
✓ Test 2, 3, 4, 5+ normal lab thresholds
✓ Assess clinical outcome correlations
✓ Optimize sensitivity/specificity
✓ Validate against expert judgment
```

---

### **H2: Specialist Referral Loop Validation**
**Expert Required**: Specialist + Health Services Researcher

#### Clinical Assumptions to Validate:
```
1. NYD codes capture "unexplained symptoms"
2. Referral loops indicate inappropriate care
3. 2+ referrals = clinically meaningful threshold
4. Mental health crisis services definition
```

---

### **H3: Medication Persistence Validation**
**Expert Required**: Psychiatrist + Clinical Pharmacist

#### Clinical Assumptions to Validate:
```
1. 180-day persistence threshold appropriateness
2. "Inappropriate" vs "appropriate" use definition
3. Emergency department visit causality
4. Medication cessation patterns
```

---

### **H4-H6: Advanced Analysis Validation**
**Expert Required**: Causal Inference Specialist + Psychiatrist

#### Assumptions Requiring Validation:
```
H4 Mediation:
- SSDSI mediates treatment effect
- Sequential ignorability assumptions
- Mediation pathway clinical plausibility

H5 Effect Modification:
- Subgroup definitions clinical validity
- Interaction biological plausibility
- Multiple testing correction adequacy

H6 Intervention Simulation:
- G-computation assumptions
- Intervention feasibility
- Cost-effectiveness model validity
```

---

## 📋 STEP-BY-STEP EXECUTION PLAN

### **PHASE 1: CRITICAL BLOCKERS (Week 1-2)**

#### **Week 1: Essential Validations**

**Monday-Tuesday: MC-SIMEX Validation**
```
Day 1 (Monday):
□ 9:00 AM: Execute Notebooks/SSD_Phenotype_Validation.ipynb
□ 11:00 AM: Generate 200-patient validation sample
□ 2:00 PM: Export ssd_validation_reviews.csv
□ 4:00 PM: Contact clinical expert for review

Day 2 (Tuesday):
□ 9:00 AM: Clinical expert review session (4 hours)
□ 2:00 PM: Input expert decisions to notebook
□ 4:00 PM: Calculate final sensitivity/specificity
□ 5:00 PM: Update config/config.yaml with real values
```

**Wednesday-Thursday: Full Dataset Integration**
```
Day 3 (Wednesday):
□ 9:00 AM: Setup production data environment
□ 11:00 AM: Acquire complete CPCSSN dataset
□ 2:00 PM: Test 01_cohort_builder.py scaling
□ 4:00 PM: Monitor memory/performance

Day 4 (Thursday):
□ 9:00 AM: Execute full pipeline: make clean && make all
□ 12:00 PM: Validate all outputs generated
□ 3:00 PM: Compare results with sample data
□ 5:00 PM: Document performance metrics
```

**Friday: Clinical Code Validation**
```
Day 5 (Friday):
□ 9:00 AM: Psychiatrist consultation (2 hours)
□ 11:00 AM: Review ICD code mappings
□ 2:00 PM: Validate drug class definitions
□ 4:00 PM: Update code mappings per expert input
□ 5:00 PM: Re-run cohort builder with validated codes
```

#### **Week 2: Extended Validations**

**Monday-Tuesday: Laboratory and Clinical Thresholds**
```
Days 8-9: Lab specialist + clinical expert reviews
□ Normal range validation
□ Healthcare utilization threshold optimization
□ Charlson comorbidity weight validation
```

**Wednesday-Thursday: Statistical Method Review**
```
Days 10-11: Biostatistician consultation
□ Causal inference method validation
□ Power calculation verification
□ Sensitivity analysis review
```

**Friday: Documentation and Integration**
```
Day 12:
□ Compile all validation reports
□ Update pipeline with validated parameters
□ Final end-to-end test
□ Prepare production readiness report
```

### **PHASE 2: HYPOTHESIS-SPECIFIC VALIDATION (Week 3-4)**

#### **Week 3: Clinical Hypothesis Validation**
```
Monday: H1 Diagnostic Cascade
Tuesday: H2 Specialist Referral Loops  
Wednesday: H3 Medication Persistence
Thursday: H4-H6 Advanced Analyses
Friday: Integration and Testing
```

#### **Week 4: Final Validation and Documentation**
```
Monday-Tuesday: ICES data access (if available)
Wednesday-Thursday: Transport weight implementation
Friday: Final validation report and sign-off
```

---

## 🏆 SUCCESS CRITERIA AND SIGN-OFF

### **PRODUCTION READINESS CHECKLIST**

#### **Critical Requirements (Must Complete All):**
- [ ] **Real MC-SIMEX metrics** from clinical validation (sensitivity ≥ 0.70, specificity ≥ 0.80)
- [ ] **Full dataset integration** completed without errors
- [ ] **Clinical expert sign-off** on ICD codes, drug definitions, and thresholds
- [ ] **End-to-end pipeline execution** successful on production data
- [ ] **No placeholder values** remaining in configuration files
- [ ] **Statistical methods validated** by qualified biostatistician

#### **Quality Gates:**
- [ ] All tests pass on production data
- [ ] Performance acceptable for production scale
- [ ] Documentation complete and validated
- [ ] Clinical assumptions documented and approved
- [ ] Ethical approvals current and sufficient

### **EXPERT SIGN-OFF REQUIREMENTS**

**Required Expert Signatures:**
```
□ Psychiatrist: _______________ Date: _______
   (ICD codes, drug persistence, SSD criteria)

□ Clinical Pharmacist: _______________ Date: _______
   (Medication definitions, persistence thresholds)

□ Biostatistician: _______________ Date: _______
   (Statistical methods, power calculations)

□ Health Services Researcher: _______________ Date: _______
   (Utilization thresholds, outcome definitions)

□ Clinical Laboratory Specialist: _______________ Date: _______
   (Laboratory normal ranges, reference standards)
```

---

## 📞 EXPERT CONTACT PROTOCOL

### **Finding Qualified Experts**

#### **Academic Contacts:**
```
Psychiatry/Psychology:
- Canadian Psychiatric Association
- Local university psychiatry departments
- Somatic symptom disorder specialists

Biostatistics:
- Canadian Society for Epidemiology and Biostatistics
- University biostatistics departments
- Causal inference specialists

Clinical Laboratory:
- Canadian Society for Medical Laboratory Science
- Hospital laboratory directors
- Provincial lab medicine specialists
```

#### **Consultation Structure:**
```
Initial Contact:
□ Brief project overview (1 page)
□ Specific validation scope
□ Time commitment estimate
□ Compensation discussion

Validation Session:
□ 2-4 hour focused review
□ Structured questionnaire
□ Decision documentation
□ Follow-up recommendations

Final Review:
□ Implementation validation
□ Final sign-off
□ Publication acknowledgment
```

---

## 🚨 RISK ASSESSMENT AND MITIGATION

### **HIGH-RISK SCENARIOS**

#### **Risk 1: Expert Unavailability**
**Probability**: Medium  
**Impact**: High  
**Mitigation**:
- Identify 2-3 backup experts per domain
- Offer flexible scheduling and remote consultation
- Consider interim literature-based validation

#### **Risk 2: ICES Data Access Denied**
**Probability**: High  
**Impact**: Medium  
**Mitigation**:
- Use published Ontario demographic data
- Perform sensitivity analysis on assumptions
- Document limitation explicitly

#### **Risk 3: Low Validation Metrics**
**Probability**: Medium  
**Impact**: High  
**Mitigation**:
- Refine phenotype algorithm if sensitivity < 0.70
- Document limitations if metrics cannot be improved
- Consider alternative bias correction approaches

---

## 💰 BUDGET AND RESOURCE REQUIREMENTS

### **Expert Consultation Costs**
```
Psychiatrist (8 hours): $2,000-3,000
Biostatistician (8 hours): $1,500-2,500  
Clinical Pharmacist (4 hours): $800-1,200
Lab Specialist (4 hours): $800-1,200
Health Services Researcher (4 hours): $800-1,200

Total Expert Consultation: $5,900-9,100
```

### **Infrastructure Requirements**
```
Computing:
- Memory: 32GB+ RAM for full dataset
- Storage: 500GB+ for data processing
- Processing: Multi-core CPU (8+ cores recommended)

Data Access:
- ICES data sharing agreement costs
- Legal review for data sharing
- Potential data hosting fees
```

### **Timeline and Personnel**
```
Total Time Requirement: 6-8 weeks
Ryhan Time Commitment: 160-200 hours
Expert Time Requirement: 32-40 hours
Administrative Time: 20-30 hours
```

---

## 📝 DOCUMENTATION REQUIREMENTS

### **Final Deliverables**

#### **Validation Reports:**
- [ ] `ssd_phenotype_validation_report.pdf`
- [ ] `clinical_expert_validation_summary.pdf`
- [ ] `statistical_methods_validation_report.pdf`
- [ ] `data_quality_assessment_report.pdf`

#### **Updated Configuration:**
- [ ] `config/config.yaml` (no placeholder values)
- [ ] `src/validated_clinical_parameters.py`
- [ ] `docs/clinical_validation_documentation.md`

#### **Production Readiness:**
- [ ] `production_readiness_assessment.pdf`
- [ ] `expert_sign_off_documentation.pdf`
- [ ] `validation_limitations_and_assumptions.md`

---

## 🎯 IMMEDIATE NEXT STEPS

### **TODAY (Priority 1):**
1. **Review this comprehensive task list** with research team
2. **Identify and contact clinical experts** for MC-SIMEX validation
3. **Schedule Week 1 validation sessions** with confirmed experts
4. **Prepare validation notebook** for expert review

### **THIS WEEK (Priority 2):**
1. **Begin MC-SIMEX clinical validation** (most critical blocker)
2. **Setup production data environment** for full dataset integration
3. **Initiate expert recruitment** for other validation domains
4. **Create validation tracking spreadsheet** for progress monitoring

### **CRITICAL MILESTONE:**
**No production use or publication submission until all Priority 1 blockers resolved**

---

**Final Note**: This pipeline represents substantial work and sophisticated methodology, but production readiness requires human validation of all clinical assumptions and parameters. The comprehensive task list above provides the roadmap to convert from development framework to validated production system.

**Contact for Questions**: See research team lead for expert recruitment and validation session coordination.

---

**Document Version**: 2.0 Unified  
**Last Updated**: June 17, 2025  
**Status**: Ready for Implementation  
**Next Review**: After Phase 1 completion