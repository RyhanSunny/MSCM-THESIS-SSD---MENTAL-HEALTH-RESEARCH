# Supplementary Methods

## Study Design

We conducted a retrospective cohort study using the Canadian Primary Care Sentinel Surveillance Network (CPCSSN) database. The study followed STROBE-CI reporting guidelines for causal inference studies.

### Population

The study population consisted of primary care patients aged ≥18 years with at least 30 months of follow-up data. We excluded patients with:
- Charlson comorbidity index >5
- Palliative care codes
- Opt-out status

Final cohort: N=250,025 patients

## Exposure Definition

Somatic symptom disorder (SSD) patterns were identified using a multi-criteria algorithm:

1. **Normal laboratory cascade**: ≥3 normal lab results within 90 days
2. **Unresolved referrals**: Referral codes without resolution (NYD codes)
3. **Psychotropic medication patterns**: Continuous use ≥90 days
4. **High healthcare utilization**: >75th percentile encounters

## Statistical Analysis

### Propensity Score Methods

We estimated propensity scores using gradient boosting (XGBoost) with the following covariates:
- Age (continuous)
- Sex (binary)
- Charlson comorbidity index (0-5)
- Rural/urban status
- Province
- Baseline healthcare utilization

```python
# Propensity score estimation
from xgboost import XGBClassifier

ps_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective='binary:logistic'
)
ps_model.fit(X_confounders, treatment)
propensity_scores = ps_model.predict_proba(X_confounders)[:, 1]
```

### Inverse Probability of Treatment Weighting (IPTW)

Weights were calculated as:
- Treated: 1/PS
- Control: 1/(1-PS)

Weights were trimmed at the 1st and 99th percentiles to reduce extreme values.

### Outcome Models

For count outcomes (healthcare encounters), we used Poisson regression with robust standard errors:

```python
# Poisson regression for count outcomes
import statsmodels.api as sm

poisson_model = sm.GLM(
    y_outcome,
    X_weighted,
    family=sm.families.Poisson(),
    freq_weights=weights
)
results = poisson_model.fit(cov_type='HC0')
```

## Causal Inference Methods

### Assumptions

1. **Exchangeability**: No unmeasured confounding
2. **Positivity**: 0 < P(Treatment|Confounders) < 1
3. **Consistency**: Well-defined treatment
4. **No interference**: SUTVA holds

### Sensitivity Analyses

1. E-value calculation for unmeasured confounding
2. Multiple imputation for missing data (m=5)
3. Alternative exposure definitions (OR vs AND logic)
4. Varying follow-up periods

## Code Snippets

### Weight Diagnostics

```python
def calculate_effective_sample_size(weights):
    '''Calculate Kish's effective sample size'''
    return np.sum(weights)**2 / np.sum(weights**2)

def validate_weights(weights):
    '''Ensure weights are not extreme'''
    ess = calculate_effective_sample_size(weights)
    ess_ratio = ess / len(weights)
    
    if ess_ratio < 0.5:
        raise ValueError(f"ESS too low: {ess_ratio:.2%}")
    
    max_weight_ratio = np.max(weights) / np.median(weights)
    if max_weight_ratio > 10:
        raise ValueError(f"Extreme weights detected: {max_weight_ratio:.1f}x median")
```

### Cluster-Robust Standard Errors

```python
def cluster_robust_se(model, cluster_ids):
    '''Calculate cluster-robust standard errors'''
    from statsmodels.stats.sandwich_covariance import cov_cluster
    
    cov_matrix = cov_cluster(model, cluster_ids)
    return np.sqrt(np.diag(cov_matrix))
```

## Software

All analyses were conducted using:
- Python 3.12
- R 4.3.0
- Key packages: pandas, numpy, statsmodels, xgboost, scikit-learn

## References

1. Hernán MA, Robins JM. Causal Inference: What If. Chapman & Hall/CRC; 2020.
2. Austin PC. An introduction to propensity score methods. Multivariate Behavioral Research. 2011;46(3):399-424.
3. Cameron AC, Miller DL. A practitioner's guide to cluster-robust inference. Journal of Human Resources. 2015;50(2):317-372.

## Author Contributions

- RS: Conceptualization, methodology, software, analysis, writing
- AG: Supervision, review & editing
