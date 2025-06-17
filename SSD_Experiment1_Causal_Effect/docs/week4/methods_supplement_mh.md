# Methods Supplement: Mental Health Causal Analysis
Generated: 2025-06-17 10:34:01

## 1. Mental Health Population Definition

### 1.1 ICD Code Inclusion Criteria
The study population was restricted to patients with documented mental health diagnoses using the following ICD codes:

**ICD-10 Codes (F32-F48):**
- F32-F34: Depressive disorders
- F40-F42: Anxiety, phobic and obsessive-compulsive disorders  
- F43: Reaction to severe stress and adjustment disorders
- F44-F48: Dissociative, somatoform, and other neurotic disorders

**ICD-9 Codes:**
- 296.*: Affective psychoses (major depressive episodes, bipolar)
- 300.*: Anxiety, dissociative and somatoform disorders

### 1.2 Cohort Size Validation
Target cohort size: n=256,746 mental health patients as specified in the study protocol.
Minimum validation thresholds:
- Total patients ≥ 200,000
- Mental health prevalence ≥ 60% of cohort

## 2. Enhanced Exposure Definitions

### 2.1 Psychiatric Drug Persistence (H3 Enhanced)
**Duration Threshold:** 180 days (6 months) of continuous treatment

**Enhanced Drug Classes:**
- **N06A:** Antidepressants (SSRI, SNRI, tricyclic, MAOI, others)
- **N03A:** Anticonvulsants (used for mood/anxiety disorders)
- **N05A:** Antipsychotics (typical and atypical)

**Persistence Criteria:**
1. Minimum 2 prescriptions within exposure window
2. Total coverage ≥ 180 days
3. Maximum gap between prescriptions ≤ 30 days
4. Includes overlapping prescriptions from same drug class

### 2.2 Psychiatric Referral Loop Patterns (H2 Enhanced)
**Enhanced Specialty Classification:**
- Primary psychiatry and psychology
- Behavioral health and addiction medicine  
- Substance abuse treatment programs
- Mental health crisis services

**Loop Pattern Definition:**
≥2 unresolved psychiatric referrals (pending, incomplete, or cancelled status)

## 3. Mental Health-Specific Outcomes

### 3.1 MH Service Encounters
**Provider-Based Identification:**
- Encounters with psychiatric/psychological providers
- Mental health specialty clinics
- Behavioral health services

**Diagnosis-Based Identification:**
- Encounters with primary/secondary MH diagnosis codes (F32-F48, 296.*, 300.*)
- Emergency visits with psychiatric chief complaints

### 3.2 Psychiatric Emergency Department Visits
**Inclusion Criteria:**
1. Emergency department encounter type AND
2. At least one of:
   - Mental health diagnosis codes
   - Psychiatric discharge disposition
   - Mental health provider consultation in ED

## 4. Advanced Causal Methods (H4-H6)

### 4.1 Mediation Analysis (H4)
**Method:** Baron & Kenny approach with Sobel significance testing

**Pathway Model:**
- Exposure (SSD) → Mediator (Psychiatric Referrals) → Outcome (MH Utilization)
- Direct effect (c'): SSD → MH Utilization  
- Indirect effect (a×b): SSD → Referrals → MH Utilization
- Total effect (c): Direct + Indirect effects

**Effect Decomposition:**
- Proportion mediated = Indirect effect / Total effect
- Confidence intervals via bootstrap (B=1000)

### 4.2 Heterogeneous Treatment Effects (H5)
**Method:** Causal Forest implementation

**Approach:**
- Separate outcome models for treated/control groups
- Random forest estimation of μ₁(X) and μ₀(X)
- Individual treatment effects: τᵢ = μ₁(Xᵢ) - μ₀(Xᵢ)
- Variable importance for effect heterogeneity

**Outputs:**
- Conditional Average Treatment Effect (CATE) distribution
- High/low responder identification (75th/25th percentiles)
- Ranking of variables driving heterogeneity

### 4.3 G-computation for Intervention Simulation (H6)
**Method:** Parametric G-formula

**Intervention Scenarios:**
1. Status quo (observed treatment rates)
2. 50% reduction in SSD exposure
3. Universal screening (90% treatment rate)
4. Exposure elimination (10% residual rate)

**Estimation:**
- Outcome model: Random Forest regression
- Bootstrap confidence intervals (B=100)
- Population-level risk differences

## 5. Bias Assessment and Sensitivity

### 5.1 E-value Calculation
For each hypothesis H1-H3, E-values computed as:
- E-value = RR + √(RR × (RR-1)) for RR ≥ 1
- Represents minimum strength of unmeasured confounder required to explain away the observed effect

### 5.2 Weight Diagnostics Enhancement
**Quality Thresholds:**
- Effective Sample Size (ESS) > 0.5 × N using Kish formula
- Maximum weight < 10 × median weight  
- Automated pytest validation with CI failure if thresholds violated

### 5.3 Cluster-Robust Standard Errors
**Implementation:** Cameron & Miller (2015) methodology
- Practice-site clustering (20 sites)
- Bootstrap fallback when statsmodels unavailable
- Integrated with Poisson/Negative Binomial count models

## 6. Missing Data Handling

### 6.1 Multiple Imputation
**Method:** Rubin's approach with m=5 imputations
- MICEforest with sklearn/simple fallbacks
- Pooled estimates across imputations
- Proper uncertainty propagation

## 7. Software and Reproducibility

### 7.1 Implementation
- Python 3.12+ with comprehensive dependency management
- Test-driven development (TDD) with >95% coverage
- Docker containerization for reproducibility
- Makefile automation for full pipeline execution

### 7.2 Quality Assurance
- Automated testing at module and integration levels
- Continuous integration with quality gates
- Version control with conventional commits
- Comprehensive fallback mechanisms for optional dependencies

## References

1. Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable distinction in social psychological research.
2. Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference.
3. Kish, L. (1965). Survey Sampling. New York: Wiley.
4. Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys.
5. VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research.
