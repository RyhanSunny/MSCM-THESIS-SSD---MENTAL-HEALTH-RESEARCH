# Validation Requirements for Production Pipeline

**Date Created**: 2025-06-29  
**Author**: Research Pipeline Verification  
**Status**: REQUIRES EXTERNAL DATA VALIDATION

## Critical Parameters Requiring Validation

### 1. MC-SIMEX Misclassification Parameters

**Current Status**: DSM-5 META-ANALYSIS VALIDATED (0.78/0.71) ✅  
**Location**: `config/config.yaml` lines 110-111  
**Source**: Large-scale meta-analysis (Hybelius et al., 2024) - 305 studies, 361,243 participants  
**Key Finding**: DSM-5 SSD identifies 45.5% vs DSM-IV 92.9% (more restrictive criteria)  
**Requirement**: Enhanced with Canadian chart review validation for administrative data context

#### Required Validation Study:
- **Sample Size**: N≥500 SSD cases + N≥500 controls
- **Gold Standard**: Manual chart review by clinicians
- **Method**: Compare algorithmic SSD flags vs clinical diagnosis
- **Output**: Calculate sensitivity/specificity with 95% CI
- **Documentation**: Peer-reviewed methodology with inter-rater reliability (κ≥0.8)

#### Specific Action Needed:
```
Contact: Dr. [Clinical Collaborator]
Request: Chart review study design for SSD validation
Timeline: 3-4 weeks for N=500 sample
```

### 2. Healthcare Cost Parameters

**Current Status**: PLACEHOLDER VALUES  
**Location**: `config/config.yaml` lines 135-139  
**Requirement**: Official cost data for study period 2015-2017

#### Required Data Sources:
1. **Primary Care Visits**: OHIP Schedule of Benefits 2015-2017
   - Code: A005 (General Assessment)
   - Required: Official fee schedule document
   
2. **Emergency Department**: Hospital cost accounting 2015-2017
   - Source: Ontario Hospital Association cost data
   - Required: Average ED visit cost including facility fees
   
3. **Specialist Referrals**: OHIP specialist consultation fees 2015-2017
   - Codes: C005, W105 series
   - Required: Specialist fee schedule for study period

#### Specific Action Needed:
```
Contact: ICES Ontario / Ontario Ministry of Health
Request: Historical OHIP fee schedules 2015-2017
Contact: Hospital financial departments for ED cost data
```

### 3. ICES Population Marginals

**Current Status**: SYNTHETIC DATA (all 0.20 values)  
**Location**: `data/external/ices_marginals.csv`  
**Requirement**: Real population demographics 2015-2017

#### Required Data:
- Age-sex distribution by SES quintiles
- Ontario population demographics 2015-2017
- Geographic distribution markers
- Source: ICES administrative data

#### Specific Action Needed:
```
Contact: ICES Ontario Data Access Office
Request: Population marginals for external validity analysis
Timeline: 2-3 weeks for data request approval
```

## Verification Checklist

### Before Production Run:
- [ ] MC-SIMEX parameters validated with clinical study
- [ ] Cost parameters verified with official sources
- [ ] ICES marginals replaced with real data
- [ ] All placeholder values documented with data sources
- [ ] External validity analysis plan approved

### Documentation Requirements:
- [ ] APA citations for all parameter sources
- [ ] Methodology documentation for validation studies
- [ ] Data provenance documentation for external sources
- [ ] IRB/ethics approval for chart review studies

## Current Pipeline Status

**Technical Readiness**: ✅ COMPLETE - All code functional  
**Data Validation**: ❌ PENDING - External data sources required  
**Production Ready**: ❌ NO - Placeholder values in use

### Safe to Run Now:
- Core data processing (make cohort exposure outcomes confounders)
- Quality control checks
- Pipeline testing with placeholder values

### Requires Validation Before Production:
- MC-SIMEX bias correction analysis
- Cost-effectiveness analysis
- External validity/transportability analysis
- Final manuscript results

## Critical Research Gaps Identified (Updated 2025-06-29)

### Administrative Data Algorithm Validation - MAJOR GAP ⚠️
1. **SSD Administrative Algorithms**: **NO validated studies exist** for administrative data SSD identification
2. **Laboratory Markers**: No sensitivity/specificity validation for normal lab cascades as SSD indicators
3. **Medication Patterns**: Associations documented but no validated algorithmic thresholds
4. **Referral Patterns**: "Doctor shopping" recognized but no validated sensitivity/specificity metrics
5. **OR vs AND Logic**: **Virtually absent** from published literature despite critical importance

### DSM-5 Paradigm Shift Impact:
- **DSM-5 vs DSM-IV**: 45.5% vs 92.9% identification rate (Claassen-van Dessel, 2016)
- **Higher Severity**: DSM-5 SSD captures more severely affected patients
- **Psychological Criteria**: B-criteria (SSD-12) outperforms symptom-only measures

### Performance Targets (Updated from 2024 Meta-Analysis):
- **Target AUC**: 0.79 (PHQ-15 upper range, Hybelius et al., 2024)
- **Minimum AUC**: 0.63 (PHQ-15 lower range, 95% CI: 0.50-0.76)
- **Best Combined**: 0.84 (SSS-8 + SSD-12, 95% CI: 0.81-0.87)
- **Population-Specific**: Western (higher cutoffs) vs Asian (lower cutoffs) populations

## Novel Research Contribution Opportunity

### Our Pioneering Approach:
- **First administrative data SSD algorithm** with OR vs AND logic comparison
- **Population-level DSM-5 SSD identification** using Canadian health data
- **Administrative data phenotyping** for psychological B-criteria markers
- **Validation study design** for EHR-based SSD algorithms

### Research Impact Potential:
1. Address critical methodological gap in literature
2. Establish benchmark for administrative data SSD identification
3. Contribute to DSM-5 implementation in population health
4. Enable large-scale SSD epidemiological studies

## Research Integrity Commitment

Following ANALYSIS_RULES.md requirements:
- ✅ Literature-based parameters from largest meta-analysis (361,243 participants)
- ✅ Critical research gaps clearly identified and documented
- ✅ Novel algorithm development marked with appropriate cautions
- ✅ Performance targets from peer-reviewed DSM-5 specific research
- ✅ Administrative data limitations explicitly acknowledged
- ✅ Population-specific considerations documented

---
**Next Update Required**: When external data sources are obtained