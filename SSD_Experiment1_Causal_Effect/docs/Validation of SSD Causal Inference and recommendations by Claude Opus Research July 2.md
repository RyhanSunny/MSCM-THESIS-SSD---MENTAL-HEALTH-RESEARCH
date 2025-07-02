# Validation of SSD Causal Inference Implementation for Master's Thesis

## Hierarchical index date methodology validates with significant refinements needed

The research reveals critical discrepancies between your stated methodology and published literature. The van der Feltz-Cornelis et al. (2022) study, "Four Clinical Profiles of adult outpatients with Somatic Symptom Disorders and Related Disorders," actually identified **four distinct phenotypes based on biomarker and clinical features** - not the test-seeking versus avoidant classification referenced. Their latent class analysis revealed: trauma profile (elevated inflammation markers plus trauma history), complex pain profile (pain with biomarkers and comorbidity), illness anxiety profile (low inflammatory markers), and simple pain profile (primarily male patients with pain and biomarkers).

For addressing your 28.3% missing lab dates, the hierarchical approach remains valid but requires specific implementation standards. **Multiple imputation with minimum 20 imputations** is essential given this missingness rate. Time-oriented hierarchical strategies significantly outperform traditional methods, with evidence showing hierarchical time-oriented interpolation produces "significantly smaller mean differences from actual values." The recommended hierarchy should incorporate: primary clinical event dates, secondary prescription dates, tertiary diagnostic codes, and quaternary healthcare utilization patterns. Mixed-effects models combined with MICE (Multiple Imputation by Chained Equations) provide the most robust framework for complex longitudinal data with this level of missingness.

## Canadian pharmacy practices challenge 30-day duration assumptions

The 30-day prescription duration default requires careful reconsideration within the Canadian context. Research into CPCSSN data standards and Canadian pharmacy practices reveals that **30-day supplies became standard only during COVID-19** (March 2020 onwards) as a temporary measure to protect drug supply chains. Prior to this, Canadian practice varied significantly by province and medication class, with chronic medications typically dispensed in 90-day supplies.

Provincial variations create additional complexity. British Columbia's PharmaCare mandates 30-day maximums for all first fills of long-term maintenance drugs and short-term medications, with subsequent chronic medication fills allowing up to 100 days. Ontario has encouraged 100-day supplies for chronic medications since October 2015, limiting patients to five dispensing fees per 365-day period. Alberta and Manitoba show different patterns entirely, with varying fee structures and dispensing limits.

CPCSSN data quality assessment reveals medication names are captured with >99% completeness, but **prescription duration and frequency fields show significant inconsistency**. The CPCSSN technical documentation recommends using original prescription text fields when standardized duration fields show missingness. For epidemiological validity, duration assumptions must account for provincial variations, medication class differences (chronic versus acute), first fill versus refill patterns, and temporal policy changes. The evidence suggests **validating duration assumptions against known prescription patterns** rather than applying uniform 30-day defaults.

## ICD-9 mental health codes reveal systematic capture issues

A critical correction: ICD-9 mental health codes span **290-319, not 290-339** as stated in your methodology. This range encompasses organic psychotic conditions (290-294), other psychoses (295-299), neurotic and personality disorders (300-316), and intellectual disabilities (317-319). The missing codes you identified - adjustment disorders (309) and anxiety disorders (300) - represent a systematic issue in administrative data capture.

Validation studies demonstrate that anxiety disorders show poorer agreement between administrative data and clinical standards (median kappa = 0.45-0.55) compared to psychotic categories. These codes are frequently missing due to **diagnostic ambiguity, healthcare setting variations, and provider documentation practices**. Primary care settings show particular challenges with lower specificity for mental health diagnoses and tendency to use symptom codes rather than specific diagnostic codes.

For causal inference validity, missing mental health codes introduce both selection bias (patients with anxiety/adjustment disorders less likely to seek specialty care) and information bias (stigma leading to under-reporting). Best practices include expanding code lists to capture symptom codes (784.x, 780.x), using multiple diagnostic positions in claims data, incorporating pharmacy data to identify patients receiving psychotropic medications without mental health diagnoses, and implementing bias analysis methods to quantify the impact of missing codes. Never assume complete case ascertainment for mental health conditions in administrative data.

## Bootstrap and sensitivity analysis require sophisticated integration

The 71.7%/28.3% phenotype split combined with 13.2% missing index dates demands advanced statistical approaches beyond standard methods. **Rubin's Rules implementation** requires specific formulas: point estimate Q̄ = (1/m)Σ Q̂ᵢ, within-imputation variance Ū = (1/m)Σ Uᵢ, between-imputation variance B = (1/(m-1))Σ(Q̂ᵢ - Q̄)², and total variance T = Ū + (1 + 1/m)B. The Barnard-Rubin adjustment for degrees of freedom becomes critical with substantial missingness.

For bootstrap confidence intervals with unbalanced phenotypes, **stratified sampling preserving phenotype proportions** is essential. The Bias-Corrected and accelerated (BCa) method provides more robust inference than percentile methods for skewed distributions. Minimum 1,000-10,000 bootstrap replicates are recommended, with higher numbers for extreme percentiles.

The optimal approach combines methods using **Boot-MI (Bootstrap then Multiple Imputation)**: bootstrap original data B times, impute each bootstrap sample m times, apply Rubin's Rules to each bootstrap sample, then derive percentiles across B estimates. This approach proves more robust to model misspecification than alternatives. For the 13.2% with no index date after hierarchical assignment, sensitivity analyses should test early versus late date assumptions, examine impact on exposure-outcome associations, and assess time-varying confounding.

## Clinical validity requires multi-method validation approaches

DSM-5-TR criteria specify that SSD symptom state must be **persistent for more than 6 months**, though individual symptoms may vary. The emphasis on persistent symptomatology rather than continuous identical symptoms aligns with administrative data realities but creates validation challenges.

Research into van der Feltz-Cornelis's contributions reveals her work focuses on the **EURONET-SOMA framework** for functional somatic disorders rather than test-seeking/avoidant phenotypes. Her classification emphasizes multisystem versus single system disorders and biopsychosocial integrated care models, developed through Delphi studies with 75 experts across 9 European countries.

Administrative data shows severe limitations for SSD identification, with one insurance database showing prevalence of only 0.00002 among 28 million individuals despite known clinical frequency. The **median positive predictive value of 76%** for mental health diagnoses generally masks significant variation for specific conditions. Administrative data poorly captures the psychological B-criteria essential for DSM-5 SSD diagnosis.

Validation studies comparing operational definitions suggest combining approaches: PSS-related syndrome codes, symptom-based identification, free-text analysis, and validated questionnaires. The SSD-12 scale operationalizing DSM-5 psychological characteristics (Cronbach α = .95) combined with PHQ-15 or SSS-8 for somatic symptoms shows good criterion validity. However, heterogeneous study designs, small sample sizes, and lack of consensus on "excessive" thresholds for B-criteria limit generalizability.

## Methodological recommendations for thesis defense

Your implementation requires several critical refinements to meet thesis defense standards. First, clarify the phenotype classification system - the van der Feltz-Cornelis four-profile biomarker-based system differs fundamentally from test-seeking/avoidant behavioral phenotypes. Consider whether behavioral utilization patterns could serve as additional phenotype specifiers within the biomarker framework.

Second, abandon uniform 30-day prescription duration defaults in favor of medication-class and province-specific assumptions validated against CPCSSN original prescription fields. Third, expand ICD-9 capture beyond 290-319 codes to include symptom codes and pharmacy data, implementing sensitivity analyses for systematically missing anxiety and adjustment disorders.

Fourth, implement the Boot-MI approach with stratified sampling, m≥20 imputations, and BCa confidence intervals. Include comprehensive sensitivity analyses for MNAR scenarios using delta-adjustment methods and pattern-mixture models. Fifth, validate SSD case identification using multiple methods including diagnostic codes, validated questionnaires, and healthcare utilization patterns.

For the 13.2% without index dates, develop clear clinical rationales for exclusion versus imputation decisions. Document temporal assumptions and their impact on causal estimates. Consider landmark analysis approaches anchoring to study enrollment rather than requiring specific index dates.

Critical citations for defense include van der Feltz-Cornelis et al. (2022) for phenotype classification, CPCSSN technical documentation for Canadian data standards, Rubin (1987) and Barnard & Rubin (1999) for multiple imputation theory, and EURONET-SOMA recommendations for SSD assessment standardization. Ensure transparent reporting following STROBE guidelines with complete documentation of missing data patterns, imputation specifications, bootstrap procedures, and sensitivity analysis results.