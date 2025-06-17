# Glossary: Mental Health Causal Analysis Terms

## A

**ATC Codes (Enhanced):** Anatomical Therapeutic Chemical classification system with enhanced psychiatric drug classes:
- N06A: Antidepressants (SSRI, SNRI, tricyclic, MAOI)
- N03A: Anticonvulsants (used for mood/anxiety disorders)  
- N05A: Antipsychotics (typical and atypical)

## C

**CATE (Conditional Average Treatment Effect):** Individual-level treatment effects estimated via causal forest, representing heterogeneity in response to SSD exposure across patient characteristics.

**Causal Forest:** Machine learning method for estimating heterogeneous treatment effects by fitting separate outcome models for treated and control groups using random forests.

**Cluster-Robust Standard Errors:** Statistical method accounting for correlation within practice sites (20 sites in this study) using Cameron & Miller (2015) methodology.

## E

**Effective Sample Size (ESS):** Kish's measure of the effective number of independent observations after propensity score weighting: ESS = (Σwᵢ)² / Σwᵢ²

**E-value:** Minimum strength of association (on risk ratio scale) that an unmeasured confounder would need with both exposure and outcome to explain away an observed effect.

## G

**G-computation:** Parametric implementation of the G-formula for simulating population-level effects of hypothetical interventions on exposure patterns.

## I

**ICD-10 Mental Health Codes (F32-F48):**
- F32-F34: Depressive disorders
- F40-F42: Anxiety and phobic disorders
- F43: Stress and adjustment disorders  
- F44-F48: Dissociative and somatoform disorders

**ICD-9 Mental Health Codes:**
- 296.*: Affective psychoses (depression, bipolar)
- 300.*: Anxiety and neurotic disorders

**IPTW (Inverse Probability of Treatment Weighting):** Propensity score method creating a pseudo-population where treatment assignment is independent of measured confounders.

## M

**Mediation Analysis:** Statistical method decomposing total treatment effect into direct and indirect (mediated) pathways using Baron & Kenny approach with Sobel significance testing.

**Mental Health Service Encounters:** Healthcare visits identified by either:
- Provider specialty (psychiatry, psychology, behavioral health)
- Primary/secondary mental health diagnosis codes
- Mental health facility location

**Multiple Imputation:** Missing data method using m=5 imputations with Rubin's pooling rules for uncertainty propagation.

## P

**Psychiatric Emergency Department Visits:** ED encounters meeting criteria:
1. Emergency encounter type AND
2. Mental health diagnosis OR psychiatric discharge disposition OR MH provider consultation

**Psychiatric Referral Loop:** Pattern of ≥2 unresolved psychiatric referrals (pending, incomplete, cancelled status) indicating care coordination challenges.

## S

**SSD (Somatic Symptom Disorder) Exposure:** Three-part definition:
- H1: Normal laboratory cascade despite symptom presentation
- H2: Psychiatric referral loop patterns  
- H3: Enhanced psychiatric drug persistence (≥180 days)

**STROBE-CI:** Strengthening the Reporting of Observational Studies in Epidemiology - Causal Inference extension for reporting causal studies.

## T

**Target Trial Emulation:** Framework for designing observational studies to emulate hypothetical randomized trials, making causal assumptions explicit.

**Temporal Ordering Validation:** Hill's temporality criterion verification ensuring exposure precedes outcome for all patients in causal pathway.

## W

**Weight Diagnostics:** Quality assessment of propensity score weights:
- ESS > 0.5 × N (effective sample size threshold)
- Maximum weight < 10 × median weight (extreme weight threshold)
- Automated pytest validation with CI failure triggers

---

## Statistical Notation

- **τ(X):** Individual treatment effect as function of covariates X
- **μ₁(X), μ₀(X):** Outcome models for treated and control groups
- **E[Y(1) - Y(0)]:** Average treatment effect (ATE)
- **E[Y(1) - Y(0)|X]:** Conditional average treatment effect (CATE)
- **a, b, c, c':** Mediation pathway coefficients (treatment→mediator, mediator→outcome, total effect, direct effect)

## Software and Implementation

- **Python 3.12+:** Primary analysis environment
- **TDD (Test-Driven Development):** Red-green-refactor development cycle
- **Docker:** Containerization for reproducibility
- **Makefile:** Pipeline automation and quality gates
- **pytest:** Testing framework with >95% coverage requirement

---

*This glossary should be referenced throughout the manuscript and supplementary materials. Terms are defined in the context of this specific mental health causal inference study.*
