# Transport Weights Documentation for SSD Mental Health Study

## Executive Summary

Transport weights are a critical component of our study's external validity strategy, designed to generalize findings from our Car4Mind mental health cohort (n=256,746) to the broader Ontario population. This document explains why transport weights matter for our thesis, how they're implemented, and what happens when ICES population data is unavailable.

## Table of Contents
1. [Project Context and Rationale](#project-context-and-rationale)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Original ICES Data Plan](#original-ices-data-plan)
4. [Implementation Details](#implementation-details)
5. [Data Format Expectations](#data-format-expectations)
6. [Graceful Degradation Strategy](#graceful-degradation-strategy)
7. [Impact on Thesis Conclusions](#impact-on-thesis-conclusions)
8. [Code References and Examples](#code-references-and-examples)

## Project Context and Rationale

### Research Question Alignment
Our primary research question asks whether SSD patterns **causally increase healthcare utilization** in mental health patients. While we can establish internal validity through propensity score matching and multiple imputation, **external validity** requires demonstrating that our findings generalize beyond the Car4Mind sample.

Transport weights address a fundamental limitation: our cohort consists of patients already engaged with mental health services, potentially representing a more severe or treatment-seeking subset of the Ontario mental health population.

### Why Transport Weights Matter for Our Hypotheses

From our methodology blueprint (`SSD THESIS final METHODOLOGIES blueprint - UPDATED 20250701.md`):
- **H1-H3**: Test causal effects of diagnostic cascades, referral loops, and medication persistence
- **H4**: Tests mediation through SSDSI  
- **H5**: Tests effect modification across subgroups
- **H6**: Proposes clinical interventions

Without transport weights, our effect estimates apply only to "patients like those in Car4Mind." With transport weights, we can claim our findings apply to "mental health patients in Ontario" - a critical distinction for policy recommendations.

## Theoretical Foundation

Transport weights implement the transportability framework from epidemiology (Westreich et al., 2017; Dahabreh & Hernán, 2019):

```
P(Y|A,S=0) = Σ P(Y|A,X,S=1) × P(X|S=0)
```

Where:
- `Y` = healthcare utilization outcome
- `A` = SSD exposure patterns
- `X` = covariates (age, sex, region, etc.)
- `S=1` = study sample (Car4Mind)
- `S=0` = target population (Ontario)

The weights rebalance our sample to match Ontario's demographic distribution, ensuring our causal estimates reflect population-level effects.

## Original ICES Data Plan

### What We Requested from ICES
We requested marginal distributions for key demographic variables from ICES (Institute for Clinical Evaluative Sciences), Ontario's administrative health data repository:

1. **Age group distributions** (18-29, 30-39, 40-49, 50-59, 60-69, 70+)
2. **Sex distributions** (Male, Female)
3. **Geographic region** (LHIN regions)
4. **Socioeconomic quintiles** (1-5, neighborhood-level)

### Why ICES Data is Gold Standard
- Covers entire Ontario population (~14 million)
- Validated linkage across health administrative databases
- Updated annually with minimal missingness
- Standard reference for Ontario health services research

### Privacy and Access Challenges
ICES data requires:
- Ethics approval (which we have)
- Data sharing agreement (pending)
- Cell size suppression (n<6 suppressed)
- Aggregate statistics only (no individual data)

## Implementation Details

### Core Function from `transport_weights.py`

```python
def calculate_transport_weights(study_data: pd.DataFrame,
                               target_marginals_path: Optional[Path] = None,
                               variables: Optional[list] = None) -> Dict[str, Any]:
    """
    Calculate transportability weights for external validity
    
    The function implements a multiplicative weighting scheme:
    1. For each demographic variable (age, sex, region)
    2. Calculate study sample proportions
    3. Load target population proportions from ICES
    4. Weight = Π(target_prop / study_prop) for each category
    """
    
    # Default path expects ICES data here
    if target_marginals_path is None:
        target_marginals_path = Path("data/external/ices_marginals.csv")
    
    # Graceful handling of missing ICES data
    if not target_marginals_path.exists():
        logger.warning(f"ICES marginals file not found: {target_marginals_path}")
        return {
            'status': 'skipped',
            'reason': 'ICES marginals file not available',
            'weights': np.ones(n),  # Uniform weights
            'effective_sample_size': n,
            'max_weight': 1.0
        }
```

### Weight Calculation Algorithm

From lines 99-124 of `transport_weights.py`:

```python
# Initialize uniform weights
weights = np.ones(len(study_data))

# Multiplicative adjustment for each variable
for var in available_vars:
    var_target = target_marginals[target_marginals['variable'] == var]
    
    if len(var_target) > 0:
        # Create mapping of categories to target proportions
        target_prop = dict(zip(var_target['category'], var_target['proportion']))
        study_prop = study_marginals[var]
        
        # Apply weight ratio to each observation
        for category in study_prop.index:
            if category in target_prop:
                weight_ratio = target_prop[category] / study_prop[category]
                mask = study_data[var] == category
                weights[mask] *= weight_ratio
```

### Effective Sample Size Calculation

Transport weights can reduce statistical power. We monitor this using Kish's effective sample size:

```python
def calculate_effective_sample_size(weights: np.ndarray) -> float:
    """
    ESS = (Σw)² / Σw²
    
    If ESS << n, weights are highly variable and inference is unstable
    """
    return (np.sum(weights)**2) / np.sum(weights**2)
```

## Data Format Expectations

### Expected ICES Marginals File Format

The system expects `data/external/ices_marginals.csv` in long format:

```csv
variable,category,proportion,n_total,source,date_extracted
age_group,18-29,0.142,1845231,ICES_OHIP,2024-01-01
age_group,30-39,0.156,2028456,ICES_OHIP,2024-01-01
age_group,40-49,0.148,1923789,ICES_OHIP,2024-01-01
age_group,50-59,0.165,2145234,ICES_OHIP,2024-01-01
age_group,60-69,0.172,2236547,ICES_OHIP,2024-01-01
age_group,70+,0.217,2820743,ICES_OHIP,2024-01-01
sex,Female,0.512,6656000,ICES_RPDB,2024-01-01
sex,Male,0.488,6344000,ICES_RPDB,2024-01-01
region,Toronto,0.205,2665000,ICES_RPDB,2024-01-01
region,Central,0.287,3731000,ICES_RPDB,2024-01-01
region,East,0.198,2574000,ICES_RPDB,2024-01-01
region,West,0.223,2899000,ICES_RPDB,2024-01-01
region,North,0.087,1131000,ICES_RPDB,2024-01-01
```

### Variable Definitions Must Match

Our study data uses these exact variable names and categories:
- `age_group`: Derived from `age` in `01_cohort_builder.py`
- `sex`: Binary M/F from demographics
- `region`: LHIN-based regions (if available)

## Graceful Degradation Strategy

### When ICES Data is Unavailable

From `transport_weights.py` lines 54-71:

```python
if not target_marginals_path.exists():
    # Return uniform weights with clear status
    return {
        'status': 'skipped',
        'reason': 'ICES marginals file not available',
        'weights': np.ones(n),  # All weights = 1
        'effective_sample_size': n,  # No reduction in ESS
        'max_weight': 1.0
    }
```

### Pipeline Integration

In `SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb`, Step 21.5:

```python
result = run_pipeline_script("transport_weights.py",
                           args="--study-data data_derived/cohort_bias_corrected.parquet " +
                                "--target-pop-stats config/ontario_population_stats.json " +
                                "--variables age,sex,comorbidity_score",
                           description="Transport Weights Analysis")

if result['status'] != 'success':
    print("WARNING: Transport weights analysis failed")
    print("Proceeding without external validity adjustment")
```

### Impact on Downstream Analyses

When transport weights are unavailable:
1. **Causal estimates remain valid** for the study sample
2. **Confidence intervals unchanged** (uniform weights)
3. **Generalizability claims must be qualified** in discussion
4. **Sensitivity analyses** should acknowledge this limitation

## Impact on Thesis Conclusions

### With Transport Weights
We can state: "Among Ontario adults with mental health conditions, SSD patterns increase healthcare utilization by X% (95% CI: Y-Z), with effects generalizable to the provincial population after accounting for demographic differences."

### Without Transport Weights  
We must state: "Among mental health patients in the Car4Mind network, SSD patterns increase healthcare utilization by X% (95% CI: Y-Z). Generalizability to the broader Ontario population requires careful consideration of potential selection effects."

### Mitigation Strategies

1. **Sensitivity Analysis**: Compare weighted vs. unweighted estimates
2. **Subgroup Consistency**: Show effects are similar across demographic strata
3. **Literature Triangulation**: Compare our estimates to other jurisdictions
4. **Clinical Face Validity**: Engage clinicians to assess reasonableness

## Code References and Examples

### Integration Points

1. **Data Preparation** (`01_cohort_builder.py`):
```python
# Creates age_group variable needed for transport weights
cohort_data['age_group'] = pd.cut(cohort_data['age'], 
                                  bins=[18, 30, 40, 50, 60, 70, 100],
                                  labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
```

2. **Weight Application** (pseudocode for `06_causal_estimators.py`):
```python
# If transport weights available, multiply with PS weights
if transport_weights is not None:
    combined_weights = ps_weights * transport_weights
    tmle_model.fit(X, y, weights=combined_weights)
else:
    tmle_model.fit(X, y, weights=ps_weights)
```

3. **Diagnostics** (`positivity_diagnostics.py`):
```python
# Check weight extremes after transport adjustment
def check_weight_distribution(transport_weights, ps_weights):
    combined = transport_weights * ps_weights
    return {
        'max_combined_weight': np.max(combined),
        'ess_reduction': calculate_ess(combined) / len(combined),
        'extreme_weights_pct': np.mean(combined > 10) * 100
    }
```

### Validation Functions

From `transport_weights.py` lines 192-220:

```python
def validate_transport_weights(weights: np.ndarray, 
                              max_weight_threshold: float = 20.0,
                              min_ess_ratio: float = 0.1):
    """
    Ensures transport weights don't compromise inference
    
    Checks:
    1. No extreme weights (>20x average)
    2. ESS doesn't drop below 10% of original
    3. Weight distribution is reasonable
    """
```

## References

1. **Westreich, D., Edwards, J. K., Lesko, C. R., Stuart, E., & Cole, S. R. (2017)**. Transportability of trial results using inverse odds of sampling weights. *American Journal of Epidemiology*, 186(8), 1010-1014.

2. **Dahabreh, I. J., & Hernán, M. A. (2019)**. Extending inferences from a randomized trial to a target population. *European Journal of Epidemiology*, 34(8), 719-722.

3. **Austin, P. C. (2011)**. An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399-424.

4. **Kish, L. (1965)**. Survey Sampling. New York: John Wiley & Sons.

## Appendix: File Locations

- Main implementation: `/src/transport_weights.py`
- Expected ICES data: `/data/external/ices_marginals.csv`
- Configuration: `/config/ontario_population_stats.json`
- Integration: `SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb` (Step 21.5)
- Related modules: 
  - `/src/positivity_diagnostics.py` (weight validation)
  - `/src/06_causal_estimators.py` (weight application)
  - `/src/rubins_pooling_engine.py` (accounts for weight uncertainty)

---
*Document created: 2025-01-02*  
*Author: Assistant to Ryhan Suny*  
*Status: Complete*