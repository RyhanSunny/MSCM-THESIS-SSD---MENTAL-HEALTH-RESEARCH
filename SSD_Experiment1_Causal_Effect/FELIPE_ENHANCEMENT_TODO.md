# SSD Causal Effect Study: Dr. Felipe Enhancement Implementation Plan

**Author**: Ryhan Suny  
**Affiliation**: Toronto Metropolitan University  
**Date**: January 7, 2025  
**Status**: READY FOR IMPLEMENTATION

---

## **EXECUTIVE SUMMARY**

This document provides a structured implementation plan to address specific gaps identified in the current SSD causal effect study implementation versus Dr. Felipe's clinical recommendations. The plan is organized into manageable phases with clear connections to research hypotheses and specific code implementations.

**Current Implementation Status**: 80-85% complete  
**Missing Components**: 4 core enhancements + 1 critical decision  
**Estimated Implementation Time**: 3-4 weeks  

---

## **CRITICAL DECISION REQUIRED FIRST**

### ⚠️ **BLOCKER: Exposure Definition Resolution**

**Issue**: 721x difference between implemented OR logic vs blueprint AND logic
- **OR Logic**: 143,579 patients (55.9%) - Currently implemented
- **AND Logic**: 199 patients (0.08%) - Blueprint specification

**Files Affected**:
- `src/02_exposure_flag.py` lines 345-356
- All downstream analyses (scripts 03-18)

**Decision Required**:
- [ ] **Option A**: Keep OR logic (document clinical justification)
- [ ] **Option B**: Switch to AND logic (accept severely limited sample)
- [ ] **Option C**: Dual analysis approach (both as primary/sensitivity)

**Impact on Research Questions**:
- **H1-H3**: Exposure definition fundamentally changes treatment group
- **H4-H6**: Mediator analysis sample size affects statistical power
- **RQ**: Overall causal inference validity depends on appropriate exposure definition

**Timeline**: **MUST BE RESOLVED WEEK 1** before any other enhancements

---

## **PHASE 1: CORE MISSING FEATURES (WEEKS 1-2)**

### 📋 **TASK 1.1: Enhanced Medication Tracking**

**Research Hypothesis Connection**: **H3** (Drug Persistence Patterns)  
**Current Gap**: Missing antidepressants (N06A), anticonvulsants (N03A), antipsychotics (N05A)  
**Clinical Rationale**: Dr. Felipe emphasized these are core SSD medications

#### **Implementation Steps**:

**Step 1.1.1**: Update ATC code configuration
- [ ] **File**: `code_lists/drug_atc.csv`
- [ ] **Action**: Add missing drug classes:
```csv
atc_code,drug_class,description
N06A1,antidepressants,Tricyclic antidepressants
N06A2,antidepressants,SSRI antidepressants  
N06A3,antidepressants,SNRI antidepressants
N06A4,antidepressants,Other antidepressants
N03A1,anticonvulsants,Phenytoin and derivatives
N03A2,anticonvulsants,Carbamazepine and derivatives
N03AX,anticonvulsants,Other anticonvulsants (gabapentin/pregabalin)
N05A1,antipsychotics,Typical antipsychotics
N05A2,antipsychotics,Atypical antipsychotics
N05A3,antipsychotics,Lithium
N05A4,antipsychotics,Other antipsychotics
```

**Step 1.1.2**: Extend drug duration threshold
- [ ] **File**: `config/config.yaml` line 37
- [ ] **Change**: `min_drug_days: 90` → `min_drug_days: 180`
- [ ] **Rationale**: Dr. Felipe specified "six months or more"

**Step 1.1.3**: Validate enhanced drug capture
- [ ] **Create**: `analysis/medication_enhancement_validation.py`
- [ ] **Purpose**: Compare before/after drug capture rates
- [ ] **Expected Impact**: 30-50% increase in H3 qualifying patients

#### **Code Implementation**:
```python
# File: analysis/medication_enhancement_validation.py
def validate_enhanced_medication_tracking():
    # Load original exposure data
    exposure_original = pd.read_parquet('data_derived/exposure.parquet')
    
    # Re-run with enhanced drug codes
    enhanced_exposure = run_enhanced_drug_analysis()
    
    # Compare H3 drug persistence rates
    original_h3 = exposure_original['H3_drug_persistence'].sum()
    enhanced_h3 = enhanced_exposure['H3_drug_persistence'].sum()
    
    print(f"Original H3 patients: {original_h3:,}")
    print(f"Enhanced H3 patients: {enhanced_h3:,}")
    print(f"Improvement: {(enhanced_h3/original_h3-1)*100:.1f}%")
```

---

### 📋 **TASK 1.2: Psychiatric Referral Pathway Analysis**

**Research Hypothesis Connection**: **H2** (Referral Loop Patterns) + **RQ** (Healthcare utilization patterns)  
**Current Gap**: General referral analysis exists, psychiatric-specific tracking missing  
**Clinical Rationale**: Dr. Felipe identified psychiatrist referral as key diagnostic pathway

#### **Implementation Steps**:

**Step 1.2.1**: Create psychiatrist identification logic
- [ ] **File**: `src/07_referral_sequence_enhanced.py`
- [ ] **Action**: Extend existing referral analysis with psychiatric specialization

```python
# Addition to src/07_referral_sequence_enhanced.py
def identify_psychiatric_referrals(referrals):
    """Identify psychiatrist/mental health referrals"""
    psych_keywords = [
        'psychiatr', 'mental health', 'psych', 'behavioral health',
        'addiction', 'substance', 'counsell', 'therapy'
    ]
    pattern = '|'.join(psych_keywords)
    return referrals[
        referrals['Name_calc'].str.contains(pattern, case=False, na=False)
    ]

def identify_medical_specialists(referrals):
    """Identify non-psychiatric medical specialists"""
    medical_specialties = [
        'cardio', 'gastro', 'neuro', 'orthop', 'rheuma', 'endocrin',
        'pulmon', 'nephro', 'oncol', 'dermat', 'urol', 'ophthal'
    ]
    pattern = '|'.join(medical_specialties)
    medical_refs = referrals[
        referrals['Name_calc'].str.contains(pattern, case=False, na=False)
    ]
    # Exclude general practice
    return medical_refs[
        ~medical_refs['Name_calc'].str.contains('family|general|gp', case=False, na=False)
    ]
```

**Step 1.2.2**: Track dual referral patterns
- [ ] **Output**: Add columns to `data_derived/referral_sequences.parquet`:
  - `psych_referral_flag`: Binary flag for any psychiatric referral
  - `medical_specialist_count`: Count of non-psychiatric specialist referrals
  - `dual_pathway_flag`: Both medical and psychiatric referrals
  - `psych_after_medical_flag`: Psychiatric referral following medical specialists

**Step 1.2.3**: Integration with exposure criteria
- [ ] **File**: `src/02_exposure_flag.py` lines 220-267 (H2 criterion section)
- [ ] **Enhancement**: Separate medical vs psychiatric referral loops in H2 calculation

#### **Expected Impact**:
- **H2 Refinement**: More clinically meaningful referral loop detection
- **RQ Enhancement**: Better understanding of healthcare utilization patterns
- **Clinical Validity**: Aligns with Dr. Felipe's SSD diagnostic pathway insights

---

### 📋 **TASK 1.3: NYD (Not Yet Diagnosed) Enhancement**

**Research Hypothesis Connection**: **H1** (Normal Lab Patterns) + **H2** (Referral Patterns)  
**Current Status**: Basic NYD count exists, missing binary flags and body part tracking  
**Clinical Rationale**: Dr. Felipe wants NYD as entry point to SSD pathway analysis

#### **Implementation Steps**:

**Step 1.3.1**: Create NYD binary flags and body part mapping
- [ ] **File**: `src/01_cohort_builder_enhanced.py`
- [ ] **Action**: Extend existing NYD counting with detailed classification

```python
# Addition to src/01_cohort_builder_enhanced.py
NYD_BODY_PART_MAPPING = {
    '799.9': 'General/Unspecified',
    'V71.0': 'Mental/Behavioral',
    'V71.1': 'Neurological',
    'V71.2': 'Cardiovascular', 
    'V71.3': 'Gastrointestinal',
    'V71.4': 'Musculoskeletal',
    'V71.5': 'Genitourinary',
    'V71.6': 'Respiratory',
    'V71.7': 'Dermatological',
    'V71.8': 'Endocrine',
    'V71.9': 'Other specified'
}

def enhance_nyd_analysis(health_condition):
    """Create enhanced NYD flags and body part tracking"""
    nyd_analysis = {}
    
    for patient_id in health_condition['Patient_ID'].unique():
        patient_hc = health_condition[health_condition['Patient_ID'] == patient_id]
        nyd_codes = patient_hc[patient_hc['DiagnosisCode_calc'].isin(NYD_CODES)]
        
        if len(nyd_codes) > 0:
            nyd_analysis[patient_id] = {
                'nyd_binary': 1,
                'nyd_count': len(nyd_codes),
                'primary_body_part': get_primary_body_part(nyd_codes),
                'body_parts_list': list(nyd_codes['DiagnosisCode_calc'].map(NYD_BODY_PART_MAPPING).unique())
            }
        else:
            nyd_analysis[patient_id] = {
                'nyd_binary': 0,
                'nyd_count': 0,
                'primary_body_part': 'None',
                'body_parts_list': []
            }
    
    return pd.DataFrame.from_dict(nyd_analysis, orient='index')
```

**Step 1.3.2**: Integration with confounders
- [ ] **File**: `src/05_confounder_flag.py` lines 158-160
- [ ] **Action**: Replace simple NYD_count with enhanced NYD analysis

---

## **PHASE 2: SEQUENTIAL PATHWAY ANALYSIS (WEEKS 3-4)**

### 📋 **TASK 2.1: Implement Dr. Felipe's Causal Chain**

**Research Hypothesis Connection**: **ALL (H1-H6) + RQ**  
**Missing Component**: NYD → Normal Labs → Specialist → No Diagnosis → Anxiety → Psychiatrist → SSD sequence  
**Clinical Significance**: This is Dr. Felipe's core insight about SSD diagnostic journey

#### **Implementation Steps**:

**Step 2.1.1**: Create sequential pathway analyzer
- [ ] **New File**: `src/08_sequential_pathway_analysis.py`
- [ ] **Purpose**: Implement the complete causal chain detection

```python
# File: src/08_sequential_pathway_analysis.py
class SSDSequentialAnalyzer:
    """
    Implements Dr. Felipe's sequential causal chain:
    NYD → Normal Labs → Specialist → No Diagnosis → Anxiety → Psychiatrist → SSD
    """
    
    def __init__(self, cohort, health_condition, lab, referral, exposure):
        self.cohort = cohort
        self.health_condition = health_condition  
        self.lab = lab
        self.referral = referral
        self.exposure = exposure
        
        # Define temporal windows
        self.pathway_window = 24  # months for complete pathway
        self.lab_window = 12      # months after NYD for normal labs
        self.referral_window = 18 # months for specialist referrals
    
    def detect_complete_pathway(self, patient_id):
        """Detect if patient follows complete NYD->SSD pathway"""
        timeline = self.build_patient_timeline(patient_id)
        
        # Step 1: NYD diagnosis (entry point)
        nyd_dates = self.get_nyd_diagnoses(patient_id, timeline)
        if not nyd_dates:
            return self.create_pathway_result(patient_id, stage=0)
        
        # Step 2: Normal lab results within 12 months of NYD
        normal_labs = self.get_normal_labs_after_nyd(patient_id, nyd_dates, timeline)
        if len(normal_labs) < 3:  # Require ≥3 normal labs
            return self.create_pathway_result(patient_id, stage=1)
        
        # Step 3: Medical specialist referrals
        med_referrals = self.get_medical_specialist_referrals(patient_id, timeline)
        if not med_referrals:
            return self.create_pathway_result(patient_id, stage=2)
        
        # Step 4: No conclusive medical diagnosis
        inconclusive = self.assess_inconclusive_workup(patient_id, med_referrals, timeline)
        if not inconclusive:
            return self.create_pathway_result(patient_id, stage=3)
        
        # Step 5: Anxiety/depression emergence
        anxiety_emergence = self.detect_anxiety_after_workup(patient_id, timeline)
        if not anxiety_emergence:
            return self.create_pathway_result(patient_id, stage=4)
        
        # Step 6: Psychiatrist referral
        psych_referral = self.get_psychiatrist_referral(patient_id, timeline)
        if not psych_referral:
            return self.create_pathway_result(patient_id, stage=5)
        
        # Step 7: SSD diagnosis or exposure flag
        ssd_outcome = self.assess_ssd_outcome(patient_id)
        
        return self.create_pathway_result(
            patient_id, 
            stage=7 if ssd_outcome else 6,
            complete_pathway=ssd_outcome
        )
```

**Step 2.1.2**: Pathway validation and reporting
- [ ] **Output**: `data_derived/sequential_pathways.parquet`
- [ ] **Columns**:
  - `patient_id`: Patient identifier
  - `pathway_stage`: Highest stage reached (0-7)
  - `complete_pathway`: Boolean for full NYD→SSD completion
  - `nyd_to_ssd_days`: Duration of complete pathway
  - `stages_completed`: List of completed pathway stages
  - `bottleneck_stage`: Where pathway typically stops

---

### 📋 **TASK 2.2: Dr. Felipe Patient Characteristics Table**

**Research Hypothesis Connection**: **Clinical Validation of All Hypotheses**  
**Missing Component**: Unified table in Dr. Felipe's specified format  
**Purpose**: Clinical interpretability and case study analysis

#### **Implementation Steps**:

**Step 2.2.1**: Create unified patient table
- [ ] **New File**: `src/09_felipe_patient_table.py`
- [ ] **Format**: `PID, age, sex, NYD (y/n), body part, referred to psy (y/n), other (y/n), SSD (1/0), Number of specialist referrals`

```python
# File: src/09_felipe_patient_table.py
def create_felipe_patient_table():
    """
    Create patient characteristics table in Dr. Felipe's requested format
    """
    # Load all required data
    cohort = pd.read_parquet('data_derived/cohort.parquet')
    exposure = pd.read_parquet('data_derived/exposure.parquet')
    confounders = pd.read_parquet('data_derived/confounders.parquet')
    referral_sequences = pd.read_parquet('data_derived/referral_sequences.parquet')
    nyd_enhanced = pd.read_parquet('data_derived/nyd_enhanced.parquet')
    psych_referrals = pd.read_parquet('data_derived/psychiatric_referrals.parquet')
    
    # Build Felipe table
    felipe_table = pd.DataFrame()
    felipe_table['PID'] = cohort['Patient_ID']
    felipe_table['age'] = cohort['Age_at_2018']
    felipe_table['sex'] = cohort['Sex_clean']
    felipe_table['NYD_yn'] = nyd_enhanced['nyd_binary']
    felipe_table['body_part'] = nyd_enhanced['primary_body_part']
    felipe_table['referred_to_psy_yn'] = psych_referrals['psych_referral_flag']
    felipe_table['referred_to_other_yn'] = psych_referrals['medical_referral_flag']
    felipe_table['SSD_flag'] = exposure['exposure_flag']
    felipe_table['num_specialist_referrals'] = referral_sequences['sequence_length']
    
    return felipe_table
```

**Step 2.2.2**: Clinical validation analysis
- [ ] **Output**: `analysis/felipe_table_analysis.py`
- [ ] **Purpose**: Generate clinical insights from the unified table

---

## **PHASE 3: INTEGRATION AND VALIDATION (WEEKS 5-6)**

### 📋 **TASK 3.1: Comprehensive Enhancement Validation**

**Research Hypothesis Connection**: **Validation of ALL Hypotheses**  
**Purpose**: Ensure enhancements improve clinical validity without breaking existing analyses

#### **Implementation Steps**:

**Step 3.1.1**: Before/after comparison analysis
- [ ] **File**: `analysis/enhancement_validation.py`
- [ ] **Comparisons**:
  - H1: Normal lab patterns (original vs enhanced)
  - H2: Referral patterns (general vs psychiatric-specific)
  - H3: Drug persistence (original vs expanded medication classes)
  - H4-H6: Mediator analysis impact
  - RQ: Overall causal inference validity

**Step 3.1.2**: Clinical meaningfulness assessment
- [ ] **File**: `analysis/clinical_validity_assessment.py`
- [ ] **Metrics**:
  - Effect sizes before/after enhancement
  - Clinical coherence of identified patterns
  - Alignment with published SSD literature

---

### 📋 **TASK 3.2: Research Paper Integration**

**Purpose**: Organize enhancements for clear research communication

#### **Implementation Steps**:

**Step 3.2.1**: Create enhancement documentation
- [ ] **File**: `docs/felipe_enhancements_methods.md`
- [ ] **Sections**:
  - Enhanced medication tracking methodology
  - Psychiatric referral pathway analysis
  - Sequential diagnostic journey mapping
  - Clinical validation results

**Step 3.2.2**: Update study protocol
- [ ] **File**: Update main research protocol with enhancement rationale
- [ ] **Justification**: Connect each enhancement to clinical SSD patterns

---

## **EXECUTION TIMELINE**

| Week | Focus | Key Deliverables | Dependencies |
|------|-------|------------------|--------------|
| 1 | **CRITICAL DECISION** | Exposure definition resolution | Team meeting required |
| 2 | Phase 1 Tasks | Enhanced medication tracking, psychiatric referrals, NYD flags | Exposure decision |
| 3 | Phase 2 Task 2.1 | Sequential pathway analysis | Phase 1 complete |
| 4 | Phase 2 Task 2.2 | Felipe patient table | Sequential analysis |
| 5 | Phase 3 Task 3.1 | Comprehensive validation | All enhancements complete |
| 6 | Phase 3 Task 3.2 | Documentation and integration | Validation complete |

---

## **SUCCESS CRITERIA**

### **Technical Success**:
- [ ] All enhanced datasets generated without errors
- [ ] Before/after validation shows expected improvements
- [ ] No regression in existing functionality

### **Clinical Success**:
- [ ] Enhanced medication tracking captures additional SSD-relevant prescriptions
- [ ] Psychiatric referral patterns align with clinical SSD pathways
- [ ] Sequential analysis identifies meaningful diagnostic journeys

### **Research Success**:
- [ ] Enhancements strengthen hypothesis testing validity
- [ ] Clinical insights support research question investigation
- [ ] Implementation aligns with Dr. Felipe's clinical expertise

---

## **RISK MITIGATION**

### **Technical Risks**:
- **Data incompatibility**: Test all enhancements on subset before full implementation
- **Performance issues**: Monitor processing time with enhanced feature sets
- **Memory constraints**: Implement chunked processing for large datasets

### **Research Risks**:
- **Over-engineering**: Validate that enhancements meaningfully improve clinical insights
- **Scope creep**: Stick to Dr. Felipe's specific recommendations
- **Timeline pressure**: Prioritize high-impact enhancements if time constraints arise

---

## **FINAL CHECKLIST**

### **Before Starting**:
- [ ] Exposure definition decision documented
- [ ] Development environment setup validated
- [ ] Backup of current data_derived files created

### **During Implementation**:
- [ ] Each task validated before proceeding to next
- [ ] Regular progress documentation
- [ ] Clinical rationale maintained for all changes

### **Upon Completion**:
- [ ] All enhanced datasets generated
- [ ] Comprehensive validation completed
- [ ] Documentation updated
- [ ] Research paper sections drafted

---

**This implementation plan transforms Dr. Felipe's clinical insights into specific, executable enhancements that strengthen the SSD causal effect study's clinical validity and research impact.**