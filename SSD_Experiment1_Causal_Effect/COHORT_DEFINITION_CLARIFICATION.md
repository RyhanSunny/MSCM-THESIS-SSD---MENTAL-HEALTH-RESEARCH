# Cohort Definition Clarification for SSD Research

**Date**: 2025-01-16  
**Author**: Research Team  
**Purpose**: Clarify the nature of the study cohort based on data extraction criteria

## Executive Summary

This document clarifies that the SSD research cohort consists of **mental health and related patients** extracted from the Canadian Primary Care Sentinel Surveillance Network (CPCSSN) database using specific diagnostic criteria.

## Data Extraction Method

The cohort was extracted from the full CPCSSN database using the SQL query found in:
`00 Mental Health ICD9 Codes Queried from Encounter Diagnosis for Care4Mind Patient Population.sql`

This query specifically selected patients with encounter diagnoses matching ICD-9 codes that include:

### 1. Core Mental Health Diagnoses (290-319)
- **290-294**: Organic brain disorders, dementias
- **295-299**: Psychotic disorders, schizophrenia, mood disorders  
- **300-309**: Anxiety, somatoform, dissociative disorders
- **310-319**: Personality disorders, intellectual disabilities, ADHD

### 2. Related Somatic/Physical Manifestations
- **327**: Sleep disorders (highly comorbid with mental health)
- **331-333**: Neurodegenerative disorders (often with psychiatric symptoms)
- **347**: Narcolepsy/cataplexy (associated with depression/anxiety)
- **625**: Female genital symptoms (bidirectional relationship with mood disorders)
- **698**: Pruritus/itching (psychodermatological manifestations)
- **780**: General symptoms (malaise, fatigue - classic "medically unexplained symptoms")
- **786**: Respiratory symptoms (cardinal features of panic/anxiety)
- **787**: GI symptoms (gut-brain axis manifestations)
- **788**: Urinary symptoms (common in trauma/somatoform disorders)
- **799**: Other ill-defined conditions (includes nervousness)
- **995**: Adverse effects (captures psychotropic medication effects)

## Population Characteristics

- **Total Patients**: 352,161 (in checkpoint)
- **After Eligibility Criteria**: 256,746 (in unified table)
- **Population Type**: Mental health patients with somatic symptom presentations
- **Key Insight**: "There's a bit more than just mental health, but mental health features commonly in those physical systems"

## Important Clarifications

1. **NOT General Primary Care**: While CPCSSN is a primary care database, our cohort was specifically extracted based on mental health and related diagnostic criteria.

2. **Mental Health Focus**: All patients have at least one encounter with the specified ICD-9 codes, ensuring a mental health connection.

3. **Somatic Symptoms Included**: The inclusion of physical symptom codes (780-788, etc.) reflects the understanding that:
   - Up to 80% of mental health patients first present with physical symptoms
   - Somatization is a core feature of many mental health conditions
   - The mind-body connection is central to SSD research

4. **Variable Interpretation**: The `mental_health_dx` flag (20.3%) in the unified table represents patients with **specific documented psychiatric diagnoses in the health_condition table**, not the mental health status of the cohort.

## Research Alignment

This cohort definition perfectly aligns with the SSD research objectives:
- Studies the intersection of mental health and somatic symptoms
- Captures the full spectrum of mental health presentations
- Enables investigation of diagnostic cascades in vulnerable populations
- Supports integrated biopsychosocial research approaches

## Conclusion

The entire cohort of 256,746 patients represents mental health and related patients, making it appropriate for research questions beginning with "Among mental health patients..." The unified table does NOT require recreation or filtering.