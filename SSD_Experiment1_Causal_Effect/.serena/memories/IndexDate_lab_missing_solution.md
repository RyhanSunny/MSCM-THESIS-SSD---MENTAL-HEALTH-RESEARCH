# Solution for 28.3% Missing IndexDate_lab in SSD Study

## Clinical Context
- 70,762 patients (28.3%) have NO lab records - not just missing dates
- DSM-5 SSD does NOT require normal labs or absence of medical conditions
- Focus is on excessive thoughts, feelings, behaviors about symptoms
- Missing labs represent a distinct clinical phenotype

## Three Evidence-Based Solutions

### 1. Phenotype-Stratified Analysis (Primary Recommendation)
**Clinical Justification**: Patients without labs represent distinct care-seeking patterns
- Healthcare avoidance due to fear of diagnosis (Psychiatry.org, 2023)
- Symptoms managed without diagnostic testing
- Different primary care approaches

**Implementation**:
```python
df['lab_utilization_phenotype'] = pd.cut(
    (~df['IndexDate_lab'].isnull()).astype(int),
    bins=[-0.5, 0.5, 1.5],
    labels=['No_Lab_Phenotype', 'Lab_Testing_Phenotype']
)
```

### 2. Alternative Index Date Strategy
**Clinical Justification**: DSM-5 requires persistence >6 months, needs temporal anchor
- First mental health encounter aligns with study population
- Maintains temporal sequence for causal inference

**Implementation**:
```python
# Use first MH encounter as index for no-lab patients
df['IndexDate_mh'] = encounter[
    encounter.DiagnosisCode_calc.str.match(r'^(29[0-9]|3[0-3][0-9])')
].groupby('Patient_ID')['EncounterDate'].min()

df['IndexDate_unified'] = df['IndexDate_lab'].fillna(df['IndexDate_mh'])
```

### 3. DSM-5 Aligned Exposure Definition
**Clinical Justification**: B-criteria (psychological response) are core to DSM-5 SSD
- Persistent medication use indicates excessive health concerns
- Referral patterns show healthcare seeking behavior

**Implementation**:
```python
# Lab-independent SSD exposure based on B-criteria proxies
df['dsm5_b_criteria_met'] = (
    (df['symptom_referral_count'] >= 2) |  # Excessive healthcare seeking
    (df['psychotropic_days'] >= 180) |     # Persistent anxiety/distress
    (df['encounter_frequency_z'] > 2)      # Excessive time/energy
)

# Combined exposure respecting DSM-5
df['ssd_exposure_dsm5'] = (
    df['dsm5_b_criteria_met'] & 
    (df['symptom_diagnosis_flag'] == 1)    # A-criteria: somatic symptoms
)
```

## References
1. DSM-5-TR (2022): "The extent to which thoughts, feelings and behaviors are excessive defines the disorder"
2. Claassen-van Dessel et al. (2016): DSM-5 captures only 45.5% vs DSM-IV's 92.9%
3. Cleveland Clinic (2023): "May avoid doctor... or seek repeated reassurance"
4. AAFP (2016): "Limited laboratory testing is recommended"

## Key Insight
The 28.3% without labs may represent the "avoidant subtype" of SSD, equally important for understanding the full spectrum of somatic symptom presentations.