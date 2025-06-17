# STROBE-CI Checklist: Mental Health Causal Inference Study
Updated: 2025-06-17 10:34:01

## Title and Abstract
| Item | Recommendation | Location |
|------|---------------|----------|
| 1a | Indicate study design in title | Title |
| 1b | Structured abstract with causal objective | Abstract |

## Introduction  
| Item | Recommendation | Location |
|------|---------------|----------|
| 2 | Scientific background for causal question | Introduction, p.1-2 |
| 3 | Specific causal objectives and hypotheses | Introduction, p.2 |

## Methods - Study Design
| Item | Recommendation | Location |
|------|---------------|----------|
| 4 | Study design and rationale for causal inference | Methods, p.3 |

## Methods - Setting  
| Item | Recommendation | Location |
|------|---------------|----------|
| 5 | Setting, locations, dates for mental health population | Methods, p.3-4 |

## Methods - Participants
| Item | Recommendation | Location |
|------|---------------|----------|
| 6a | Mental health cohort eligibility criteria (ICD F32-F48, 296.*, 300.*) | Methods, p.4 |
| 6b | Matching criteria for propensity score analysis | Methods, p.5 |

## Methods - Variables
| Item | Recommendation | Location |
|------|---------------|----------|
| 7a | SSD exposure definition (H1: normal labs, H2: psych referrals, H3: drug persistence 180d) | Methods, p.5-6 |
| 7b | MH-specific outcomes (encounters, psychiatric ED visits) | Methods, p.6 |
| 7c | Confounders and mediators in causal pathway | Methods, p.6-7 |

## Methods - Data Sources
| Item | Recommendation | Location |
|------|---------------|----------|
| 8 | EHR data sources and validation | Methods, p.7 |

## Methods - Bias
| Item | Recommendation | Location |
|------|---------------|----------|
| 9 | Bias sources and mitigation strategies | Methods, p.7-8 |

## Methods - Study Size
| Item | Recommendation | Location |
|------|---------------|----------|
| 10 | Sample size rationale (n=256,746 target) | Methods, p.8 |

## Methods - Quantitative Variables  
| Item | Recommendation | Location |
|------|---------------|----------|
| 11 | Exposure and outcome measurement | Methods, p.8-9 |

## Methods - Statistical Methods
| Item | Recommendation | Location |
|------|---------------|----------|
| 12a | IPTW with weight diagnostics (ESS >0.5N, max_wt <10×median) | Methods, p.9 |
| 12b | Cluster-robust SE for practice sites | Methods, p.9 |  
| 12c | Count models (Poisson/NB) for utilization outcomes | Methods, p.9-10 |
| 12d | Multiple imputation (m=5) for missing data | Methods, p.10 |
| 12e | Advanced methods: Mediation (H4), Causal Forest (H5), G-computation (H6) | Methods, p.10-11 |

## Results - Participants
| Item | Recommendation | Location |
|------|---------------|----------|
| 13a | Participant flow with mental health filtering | Results, p.12; Figure 1 |
| 13b | Reasons for exclusion | Results, p.12 |
| 13c | CONSORT-style flowchart | Figure 1 |

## Results - Descriptive Data
| Item | Recommendation | Location |
|------|---------------|----------|
| 14a | Baseline characteristics by SSD exposure | Results, p.13; Table 1 |
| 14b | Missing data patterns | Results, p.13 |
| 14c | Propensity score distribution and balance | Results, p.14; Figure 2 |

## Results - Outcome Data
| Item | Recommendation | Location |
|------|---------------|----------|
| 15 | MH service utilization by exposure status | Results, p.14-15 |

## Results - Main Results  
| Item | Recommendation | Location |
|------|---------------|----------|
| 16a | H1-H3 effect estimates with 95% CI | Results, p.15-16; Table 2 |
| 16b | Forest plot of main results | Figure 3 |
| 16c | Advanced analyses results (H4-H6) | Results, p.16-17; Table 3 |

## Results - Other Analyses
| Item | Recommendation | Location |
|------|---------------|----------|
| 17a | Sensitivity analyses and E-values | Results, p.17-18; Table 4 |
| 17b | Heterogeneous effects (CATE analysis) | Results, p.18; Figure 4 |
| 17c | Mediation pathway results | Results, p.18-19; Figure 5 |

## Discussion - Key Results
| Item | Recommendation | Location |
|------|---------------|----------|
| 18 | Main findings in context of causal inference | Discussion, p.19-20 |

## Discussion - Limitations
| Item | Recommendation | Location |
|------|---------------|----------|
| 19 | Unmeasured confounding and bias sources | Discussion, p.20-21 |

## Discussion - Interpretation  
| Item | Recommendation | Location |
|------|---------------|----------|
| 20 | Causal interpretation and clinical implications | Discussion, p.21-22 |

## Discussion - Generalizability
| Item | Recommendation | Location |
|------|---------------|----------|
| 21 | External validity for mental health populations | Discussion, p.22 |

## Other Information
| Item | Recommendation | Location |
|------|---------------|----------|
| 22 | Funding and competing interests | Funding statement |

## Additional STROBE-CI Items

### Causal Diagram
| Item | Recommendation | Location |
|------|---------------|----------|
| C1 | DAG with assumed causal relationships | Figure S1 |

### Target Trial Emulation
| Item | Recommendation | Location |
|------|---------------|----------|
| C2 | Target trial specification | Methods, p.11 |

### Assumption Assessment
| Item | Recommendation | Location |
|------|---------------|----------|
| C3 | Positivity assessment | Results, p.17 |
| C4 | Exchangeability discussion | Discussion, p.20 |
| C5 | Consistency assumption | Methods, p.11 |

### Effect Modification
| Item | Recommendation | Location |
|------|---------------|----------|
| C6 | Pre-specified effect modifiers | Methods, p.11 |
| C7 | Causal forest heterogeneity analysis | Results, p.18 |

### Sensitivity Analyses
| Item | Recommendation | Location |
|------|---------------|----------|
| C8 | E-value calculations for H1-H3 | Results, p.17; Table 4 |
| C9 | Bias analysis scenarios | Supplement, Table S3 |

*Note: Page numbers and table/figure references should be updated based on final manuscript layout.*
