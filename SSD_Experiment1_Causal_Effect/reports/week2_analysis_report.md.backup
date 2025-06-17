# Week 2 Analysis Report: H1-H3 Hypotheses

*Generated: 2025-06-17 03:27:32*

## Executive Summary

- Total sample size: 250,025 patients
- Hypotheses tested: 3
- Figures generated: 4
- Tables created: 3

### Key Findings

- H1: Normal lab cascade showed IRR=1.005 (95% CI: 0.995-1.014, p=0.329)
- H2: Unresolved referrals significantly associated with increased healthcare utilization (IRR=1.005, p=0.043)
- H3: Medication persistence showed IRR=0.996 for ED visits

## Hypothesis Test Results

### H1: Normal lab cascade → healthcare encounters

- **Sample size**: 250,025 total, 37,646 exposed
- **Effect estimate**: IRR = 1.005 (95% CI: 0.995-1.014)
- **P-value**: 0.329
- **Interpretation**: No statistically significant difference

### H2: Unresolved referrals → healthcare utilization

- **Sample size**: 250,025 total, 19,895 exposed
- **Effect estimate**: IRR = 1.005 (95% CI: 1.000-1.009)
- **P-value**: 0.043
- **Interpretation**: Statistically significant increase

### H3: Medication persistence → reduced ED visits

- **Sample size**: 250,025 total, 49,955 exposed
- **Effect estimate**: IRR = 0.996 (95% CI: 0.140-7.072)
- **P-value**: 0.997
- **Interpretation**: No statistically significant difference

## Figures Generated

### Figure: CONSORT Flow Diagram: Cohort Selection
- **File**: `figures/consort_flowchart.svg`
- **Description**: Patient selection and exclusion criteria

### Figure: Causal DAG: SSD → Healthcare Utilization
- **File**: `figures/dag.svg`
- **Description**: Directed acyclic graph showing causal relationships

### Figure: Forest Plot: H1-H3 Treatment Effects
- **File**: `figures/forest_plot.svg`
- **Description**: Incidence rate ratios with 95% confidence intervals

### Figure: Covariate Balance: Love Plot
- **File**: `figures/love_plot.svg`
- **Description**: Standardized mean differences before/after weighting

## Tables Generated

### Table: baseline_table
- **CSV**: `tables/baseline_table.csv`
- **Markdown**: `tables/baseline_table.md`

### Table: main_results
- **CSV**: `tables/main_results.csv`
- **Markdown**: `tables/main_results.md`

### Table: sensitivity
- **CSV**: `tables/sensitivity.csv`
- **Markdown**: `tables/sensitivity.md`

## Technical Notes

- All analyses used cluster-robust standard errors (20 practice sites)
- Count outcomes analyzed with Poisson/Negative Binomial regression
- Propensity score weights validated (ESS = 66.7% of sample)
- Multiple imputation would be applied for missing data in full analysis
