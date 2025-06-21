# Study Limitations and Mitigation Strategies

**Document Version**: 1.0  
**Date**: June 21, 2025  
**Authors**: SSD Mental Health Research Team  
**Purpose**: Manuscript limitations section for Q1 journal submission

---

## Overview

Our study examining the causal relationship between somatic symptom disorder (SSD) patterns and healthcare utilization in mental health populations has several important limitations. We present these transparently along with our mitigation strategies to ensure readers can appropriately interpret our findings.

## 1. Measurement Error in SSD Phenotype Classification

### Limitation
Our algorithmic definition of SSD patterns—based on repeated normal laboratory results, unresolved specialist referrals, and persistent psychotropic medication use—has not been validated against gold-standard clinical diagnoses. Without clinical chart review, we cannot determine the sensitivity and specificity of our exposure classification.

### Impact
Misclassification of exposure status typically biases effect estimates toward the null hypothesis (Hernán & Robins, 2020). If our algorithm has 82% sensitivity and specificity as estimated from similar studies, approximately 18% of patients are misclassified. This means our observed incidence rate ratios (IRRs) of 1.35-1.50 likely underestimate the true causal effects.

### Our Mitigation Strategy
We implemented MC-SIMEX (Misclassification Simulation-Extrapolation) infrastructure in our analytical pipeline. This statistical method can correct for known misclassification bias by:
1. Adding incremental amounts of classification error to the data
2. Observing how effect estimates change with increasing error
3. Extrapolating back to zero classification error

However, without validated sensitivity/specificity parameters from our specific population, we present uncorrected estimates as our primary results. We acknowledge this leads to conservative (underestimated) effect sizes. Our code is structured to immediately incorporate bias correction once clinical validation data become available (`use_bias_corrected_flag` in configuration).

## 2. Limited External Validity Assessment

### Limitation
We cannot formally assess whether our findings from the Canadian Primary Care Sentinel Surveillance Network (CPCSSN) generalize to the broader Ontario population. The Institute for Clinical Evaluative Sciences (ICES) population marginal distributions needed for transportability analysis were not available due to data sharing constraints.

### Impact
Our results apply specifically to patients receiving care at CPCSSN-affiliated practices. These practices may differ from the general Ontario population in important ways:
- Geographic distribution (potentially more urban)
- Socioeconomic status of patients
- Practice patterns and resources
- Patient health-seeking behaviors

### Our Mitigation Strategy
We developed a complete transportability weights framework that can calculate inverse-odds weights to rebalance our sample to match external population distributions. When ICES marginal distributions are unavailable, our pipeline:
1. Returns uniform weights (all patients weighted equally)
2. Documents this limitation in output files
3. Continues analysis without claiming provincial generalizability

We characterize our study population thoroughly in Table 1, allowing readers to assess similarity to their populations of interest. Future researchers can apply our open-source pipeline to their local data to assess whether similar patterns emerge.

## 3. Literature-Based Clinical Parameters

### Limitation
Our operational definitions rely on thresholds derived from literature rather than empirically validated in our population:
- "≥3 normal laboratory results" as indicative of diagnostic uncertainty
- "≥90 days of continuous psychotropic medication" as persistent use
- "≥2 unresolved specialist referrals" as care fragmentation

These thresholds, while clinically plausible, may not optimally capture SSD patterns in our specific mental health population.

### Impact
Using non-optimized thresholds could lead to:
- Exposure misclassification (addressed in Limitation 1)
- Reduced statistical power to detect true effects
- Difficulty comparing results across different healthcare systems

### Our Mitigation Strategy
We conduct comprehensive sensitivity analyses varying each threshold:
- Laboratory results: ≥2 vs ≥3 vs ≥4 normal results
- Medication persistence: 90 vs 180 days (with enhanced module testing 180-day threshold per expert consultation)
- Referral patterns: ≥1 vs ≥2 vs ≥3 unresolved referrals

We report whether our conclusions are robust to these variations. Additionally, we parameterized all thresholds in our configuration files, allowing other researchers to adapt definitions to their clinical contexts.

## 4. Additional Methodological Considerations

### 4.1 Temporal Dynamics During COVID-19
Our study period (2015-2020) includes the early COVID-19 pandemic. We address this through segmented regression analysis with a level-shift at March 2020, though only 3 months of pandemic data limits our ability to model pandemic-specific trends.

### 4.2 Mental Health Population Specificity
By restricting to patients with mental health diagnoses (ICD codes F32-F48, 296.*, 300.*), we enhance internal validity but limit generalizability to patients without documented mental health conditions. This design choice reflects our focus on understanding SSD patterns within an already vulnerable population.

### 4.3 Administrative Data Constraints
Electronic health record data may incompletely capture:
- Care received outside CPCSSN practices
- Over-the-counter medication use
- Symptom severity and functional impairment

We acknowledge these as inherent limitations of administrative data research.

## Strengths Balancing These Limitations

Despite these limitations, our study has several methodological strengths:

1. **Causal inference framework**: We employ state-of-the-art methods including propensity score weighting, doubly robust estimation, and extensive sensitivity analyses

2. **Transparent methodology**: All code is open-source with version control, containerized environments, and comprehensive documentation

3. **Multiple hypothesis testing**: We pre-specified six hypotheses with appropriate multiple testing corrections (Benjamini-Hochberg FDR)

4. **Robust statistical approach**: We use appropriate models for count data (Poisson/Negative Binomial), cluster-robust standard errors, and multiple imputation for missing data

## Conclusions Regarding Limitations

These limitations primarily affect the precision and generalizability of our estimates rather than their validity. Our conservative approach—presenting uncorrected estimates, acknowledging uncertainty in external validity, and testing multiple threshold definitions—ensures our conclusions are scientifically defensible while highlighting areas for future research.

We encourage replication studies with:
- Clinical validation of SSD phenotypes
- Access to population-level data for transportability assessment  
- Prospectively defined and validated clinical thresholds

Our open-source analytical pipeline (available at [repository DOI]) facilitates such replications and extensions.

---

## References

Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399-424.

Hernán, M. A., & Robins, J. M. (2020). *Causal inference: What if*. Chapman & Hall/CRC.

VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.

---

*This document represents a complete and transparent assessment of study limitations as of June 21, 2025. Updates will be tracked in version control as clinical validation data or external population distributions become available.*