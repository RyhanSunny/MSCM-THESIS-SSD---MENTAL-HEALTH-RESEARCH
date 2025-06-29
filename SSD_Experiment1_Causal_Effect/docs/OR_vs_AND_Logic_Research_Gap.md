# OR vs AND Logic Research Gap in SSD Administrative Data Algorithms

**Date Created**: 2025-06-29  
**Author**: Research Pipeline Documentation  
**Status**: CRITICAL RESEARCH GAP IDENTIFIED

## Problem Statement

Direct comparison studies of inclusive (OR) versus restrictive (AND) logic for DSM-5 Somatic Symptom Disorder identification are **virtually absent** from the published literature, despite their critical importance for administrative data algorithms.

## Literature Review Findings

### Current State
- Most existing algorithms use additive scoring rather than Boolean logic
- German validation study (Toussaint et al., 2020) used combined cutoffs but did not explicitly compare OR vs AND approaches
- No published studies directly compare sensitivity/specificity of OR vs AND logic for SSD identification

### Evidence Supports Combined Approaches
Current evidence suggests combined approaches using both:
- **A-criteria**: Symptom severity measures (PHQ-15, SSS-8)
- **B-criteria**: Psychological features (SSD-12)

But the specific impact of logical operators (OR vs AND) remains **unexamined**.

## Our Implementation Context

### Current Pipeline Logic
Our SSD exposure algorithm uses **OR logic** combining:
1. Normal laboratory results (≥3 normal labs)
2. Symptom-related referrals (≥2 referrals)  
3. Persistent medication use (≥180 days)

**Configuration Location**: `config/config.yaml` lines 25-31  
**Implementation**: `src/02_exposure_flag.py --logic or`

### Research Innovation Opportunity
Our approach represents **novel algorithm development** since:
- No validated administrative data algorithms exist for DSM-5 SSD
- OR vs AND logic comparison studies are absent from literature
- We are pioneering administrative data phenotyping for SSD

## Validation Requirements

### Priority 1: Internal Validation
- Compare OR vs AND logic performance within our dataset
- Calculate sensitivity/specificity for both approaches
- Assess impact on cohort size and characteristics

### Priority 2: External Validation
- Manual chart review validation study (N≥500)
- Clinical expert assessment of algorithm performance
- Comparison against gold standard diagnostic interviews

### Priority 3: Methodological Contribution
- Publish first OR vs AND logic comparison for SSD algorithms
- Contribute to administrative data phenotyping methodology
- Address critical gap identified in literature review

## Implementation Plan

### Phase 1: Comparative Analysis
```bash
# Current OR logic
python3 src/02_exposure_flag.py --logic or

# Future AND logic comparison  
python3 src/02_exposure_flag.py --logic and

# Performance comparison analysis
python3 src/logic_comparison_analysis.py
```

### Phase 2: Validation Study Design
- Develop manual chart review protocol
- Calculate required sample size for validation
- Design gold standard comparison methodology

## Research Impact

This research addresses a **critical methodological gap** and could establish our study as a pioneering contribution to:
1. DSM-5 SSD administrative data phenotyping
2. Boolean logic validation in healthcare algorithms
3. Population-level SSD identification methodology

## Next Steps

1. Document current OR logic performance
2. Implement AND logic comparison
3. Design validation study protocol
4. Prepare manuscript for methodological contribution

---
**Reference**: Based on "Validation studies for DSM-5 somatic symptom disorder" literature review (2025-06-29)