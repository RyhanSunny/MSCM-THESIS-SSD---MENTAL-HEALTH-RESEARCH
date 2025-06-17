# STROBE Flow Diagram for SSD Causal Effect Study

## Figure 1: Participant Flow Diagram

```mermaid
graph TB
    A[CPCSSN Primary Care Database<br/>2015-2017] 
    
    A -->|"Mental Health ICD-9 Codes<br/>290-319, 327, 331-333, 347,<br/>625, 698, 780, 786-788, 799, 995"| B[352,161 Patients with<br/>Mental Health Diagnoses]
    
    B -->|"Inclusion Criteria Applied"| C{Eligibility<br/>Assessment}
    
    C -->|"Excluded n=95,415"| D[Excluded:<br/>Breakdown by reason<br/>NOT YET CALCULATED]
    
    C -->|"Met all criteria"| E[256,746 Patients<br/>Final Analytic Cohort<br/>72.9% retention]
    
    E --> F[Cohort Characteristics:<br/>- Demographics: 100% complete<br/>- Lab data: 8,528,807 records<br/>- Normal labs detected: 45%<br/>- Medication data: 7,706,628 records<br/>- Referral data: 1,141,061 records]
    
    style A fill:#ffffff,stroke:#333,stroke-width:2px,color:#000
    style B fill:#ffffff,stroke:#333,stroke-width:2px,color:#000
    style C fill:#ffffff,stroke:#333,stroke-width:2px,color:#000
    style D fill:#f8f8f8,stroke:#333,stroke-width:2px,color:#000
    style E fill:#ffffff,stroke:#333,stroke-width:3px,color:#000
    style F fill:#ffffff,stroke:#333,stroke-width:1px,color:#000
```

## Data Notes

### Verified Numbers from Pipeline:
- **Initial cohort**: 352,161 patients (from SQL query with mental health ICD-9 codes)
- **Final cohort**: 256,746 patients (after cohort builder filters)
- **Excluded**: 95,415 patients (27.1% exclusion rate)
- **Retention**: 72.9%

### Inclusion Criteria Applied (from blueprint):
1. Age ≥ 18 years as of January 1, 2015
2. ≥ 30 consecutive months of EHR data before January 1, 2015
3. No palliative care codes (V66.7, Z51.5)
4. Charlson Comorbidity Index ≤ 5
5. Not opted out of CPCSSN

### Data Completeness (from checkpoint metadata):
- **Patient demographics**: 352,220 records
- **Encounters**: 11,577,739 records
- **Encounter diagnoses**: 12,471,764 records
- **Laboratory tests**: 8,528,807 records
- **Medications**: 7,706,628 records
- **Referrals**: 1,141,061 records

### Missing Data Summary:
- **Postal codes**: 0% available (critical for neighborhood deprivation)
- **Education**: 1.4% available
- **Occupation**: 7.6% available
- **Housing status**: 0.01% available

Note: Individual exclusion counts by criterion are NOT YET CALCULATED as the cohort builder script aggregates exclusions without itemizing by reason. 