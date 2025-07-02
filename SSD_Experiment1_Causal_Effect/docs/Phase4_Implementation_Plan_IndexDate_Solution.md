# Phase 4 Implementation Plan: Resolving IndexDate_lab Missing Values

**Document Version**: 1.0  
**Date**: January 3, 2025  
**Purpose**: Practical implementation guide to resolve Phase 4 imputation blocking issue

## Problem Summary

- **Issue**: 70,762 missing IndexDate_lab values (28.3%) cannot be imputed as datetime
- **Impact**: Phase 4 (07b_missing_data_master.py) reports 70,764 remaining missing values
- **Root Cause**: Patients with NO laboratory records (informative missingness)
- **Solution**: Implement validated three-pronged approach with target trial emulation

## Implementation Strategy

### Step 1: Modify 01_cohort_builder.py to Create Enhanced Index Dates

```python
# Add after line 184 in 01_cohort_builder.py
def create_hierarchical_index_dates(elig, lab, encounter, medication):
    """
    Creates hierarchical index dates following validation report methodology
    """
    log.info("Creating hierarchical index dates for all patients...")
    
    # 1. Laboratory index (existing)
    idx_lab = lab.groupby("Patient_ID")["PerformedDate"].min().rename("IndexDate_lab")
    elig = elig.merge(idx_lab, left_on="Patient_ID", right_index=True, how="left")
    
    # 2. First MH diagnosis encounter
    mh_encounters = encounter[
        encounter.DiagnosisCode_calc.str.match(r'^(29[0-9]|3[0-3][0-9])')
    ]
    idx_mh = mh_encounters.groupby("Patient_ID")["EncounterDate"].min().rename("IndexDate_mh_dx")
    elig = elig.merge(idx_mh, left_on="Patient_ID", right_index=True, how="left")
    
    # 3. First psychotropic prescription (≥180 days)
    psych_meds = medication[
        (medication['drug_class'].isin(['anxiolytic', 'antidepressant', 'hypnotic'])) &
        (medication.groupby('Patient_ID')['days_supply'].transform('sum') >= 180)
    ]
    idx_psych = psych_meds.groupby("Patient_ID")["DispenseDate"].min().rename("IndexDate_psych_rx")
    elig = elig.merge(idx_psych, left_on="Patient_ID", right_index=True, how="left")
    
    # 4. Create unified index with source tracking
    elig['IndexDate_unified'] = (elig['IndexDate_lab']
                                 .fillna(elig['IndexDate_mh_dx'])
                                 .fillna(elig['IndexDate_psych_rx']))
    
    # Track source for stratification
    elig['index_date_source'] = np.select(
        [elig['IndexDate_lab'].notna(),
         elig['IndexDate_mh_dx'].notna() & elig['IndexDate_lab'].isna(),
         elig['IndexDate_psych_rx'].notna() & elig['IndexDate_mh_dx'].isna() & elig['IndexDate_lab'].isna()],
        ['Laboratory', 'MH_Diagnosis', 'Psychotropic_Rx'],
        default='None'
    )
    
    # Create phenotype based on lab availability
    elig['lab_utilization_phenotype'] = np.where(
        elig['IndexDate_lab'].notna(), 
        'Test_Seeking',
        'Avoidant'
    )
    
    log.info(f"Index date source distribution:\n{elig['index_date_source'].value_counts()}")
    log.info(f"Lab phenotype distribution:\n{elig['lab_utilization_phenotype'].value_counts()}")
    
    return elig
```

### Step 2: Update 07b_missing_data_master.py to Exclude Datetime Columns

```python
# Modify the imputation section in 07b_missing_data_master.py (around line 280)

# Identify datetime columns
datetime_cols = master_table.select_dtypes(include=['datetime64']).columns.tolist()
log.info(f"Excluding datetime columns from imputation: {datetime_cols}")

# Separate datetime and non-datetime columns
non_datetime_cols = [col for col in master_table.columns if col not in datetime_cols]
datetime_data = master_table[datetime_cols].copy()
imputable_data = master_table[non_datetime_cols].copy()

# Perform imputation only on non-datetime columns
log.info(f"Imputing {imputable_data.shape[1]} non-datetime columns...")

# Create and fit MICE imputer
imputer = IterativeImputer(
    estimator=BayesianRidge(),
    n_nearest_features=10,
    initial_strategy='median',
    max_iter=10,
    random_state=42,
    verbose=2
)

# Impute each dataset
imputed_datasets = []
for i in range(m):
    log.info(f"Creating imputation {i+1}/{m}...")
    X_imputed = imputer.fit_transform(imputable_data)
    
    # Create dataframe with imputed values
    df_imputed = pd.DataFrame(
        X_imputed,
        columns=imputable_data.columns,
        index=imputable_data.index
    )
    
    # Rejoin datetime columns
    df_complete = pd.concat([df_imputed, datetime_data], axis=1)
    
    # Reorder columns to match original
    df_complete = df_complete[master_table.columns]
    
    imputed_datasets.append(df_complete)

# After imputation, check remaining missing
remaining_missing = sum(df.isnull().sum().sum() for df in imputed_datasets) / m
log.info(f"Average remaining missing values across imputations: {remaining_missing}")
log.info(f"These should only be in datetime columns: {datetime_cols}")
```

### Step 3: Create New Script for DSM-5 B-Criteria Exposure

Create `src/02b_exposure_dsm5_validated.py`:

```python
"""
DSM-5 validated exposure definition using B-criteria proxies
Independent of laboratory data to handle avoidant phenotype
"""
import pandas as pd
import numpy as np
from config import Config
from src.utils.logging_utils import setup_logging
from src.utils.file_utils import ensure_directory, save_with_metadata

log = setup_logging(__name__)

def create_dsm5_b_criteria_exposure(cohort_path: str) -> pd.DataFrame:
    """
    Creates DSM-5 aligned exposure using validated B-criteria thresholds
    """
    log.info("Loading cohort data...")
    cohort = pd.read_parquet(cohort_path)
    
    # Load additional data needed
    encounter = pd.read_parquet(Config.ENCOUNTER_PATH)
    medication = pd.read_parquet(Config.MEDICATION_PATH)
    referral = pd.read_parquet(Config.REFERRAL_PATH)
    
    # Calculate healthcare utilization percentile
    cohort['annual_encounters'] = encounter.groupby('Patient_ID').size() / cohort['followup_years']
    cohort['utilization_percentile'] = cohort.groupby('age_group')['annual_encounters'].rank(pct=True)
    
    # A-Criteria: Somatic symptoms (ICD-9: 780-799)
    symptom_encounters = encounter[
        encounter['DiagnosisCode_calc'].str.match(r'^78[0-9]|79[0-9]')
    ]
    cohort['dsm5_a_symptoms'] = cohort['Patient_ID'].isin(
        symptom_encounters['Patient_ID'].unique()
    )
    
    # B1: Disproportionate concern (≥95th percentile utilization)
    cohort['dsm5_b1_disproportionate'] = cohort['utilization_percentile'] >= 0.95
    
    # B2: Persistent anxiety (≥6 months psychotropic)
    psych_duration = medication[
        medication['drug_class'].isin(['anxiolytic', 'antidepressant', 'hypnotic'])
    ].groupby('Patient_ID')['days_supply'].sum()
    cohort['psychotropic_days'] = cohort['Patient_ID'].map(psych_duration).fillna(0)
    cohort['dsm5_b2_anxiety'] = cohort['psychotropic_days'] >= 180
    
    # B3: Excessive healthcare behavior (>3 specialty referrals/year)
    referral_counts = referral.groupby('Patient_ID').size() / cohort['followup_years']
    cohort['annual_referrals'] = cohort['Patient_ID'].map(referral_counts).fillna(0)
    cohort['dsm5_b3_excessive'] = cohort['annual_referrals'] > 3
    
    # B-Criteria met (any)
    cohort['dsm5_b_criteria_met'] = (
        cohort['dsm5_b1_disproportionate'] | 
        cohort['dsm5_b2_anxiety'] | 
        cohort['dsm5_b3_excessive']
    )
    
    # C-Criteria: Persistence (using symptom duration)
    cohort['dsm5_c_persistence'] = cohort['symptom_duration_months'] >= 6
    
    # Complete DSM-5 SSD diagnosis
    cohort['ssd_exposure_dsm5'] = (
        cohort['dsm5_a_symptoms'] & 
        cohort['dsm5_b_criteria_met'] & 
        cohort['dsm5_c_persistence']
    )
    
    # Create stratified exposure by phenotype
    cohort['ssd_exposure_stratified'] = cohort['ssd_exposure_dsm5'].astype(str) + '_' + cohort['lab_utilization_phenotype']
    
    log.info(f"DSM-5 SSD exposure created:")
    log.info(f"  A-criteria met: {cohort['dsm5_a_symptoms'].sum():,}")
    log.info(f"  B-criteria met: {cohort['dsm5_b_criteria_met'].sum():,}")
    log.info(f"  C-criteria met: {cohort['dsm5_c_persistence'].sum():,}")
    log.info(f"  Complete SSD: {cohort['ssd_exposure_dsm5'].sum():,}")
    log.info(f"\nBy phenotype:")
    log.info(cohort.groupby('lab_utilization_phenotype')['ssd_exposure_dsm5'].value_counts())
    
    # Save enhanced cohort
    output_path = Config.DATA_DIR / "cohort_dsm5_exposure.parquet"
    save_with_metadata(cohort, output_path, {"exposure_type": "DSM-5 validated"})
    
    return cohort

if __name__ == "__main__":
    create_dsm5_b_criteria_exposure(Config.COHORT_PATH)
```

### Step 4: Update Pipeline Configuration

Update `config.yaml`:

```yaml
# Imputation settings
imputation:
  exclude_datetime_columns: true
  datetime_columns:
    - IndexDate_lab
    - IndexDate_mh_dx  
    - IndexDate_psych_rx
    - IndexDate_unified
  m_imputations: 30

# Exposure definitions
exposure:
  primary_definition: "dsm5_validated"  # Change from "or_logic"
  use_hierarchical_index: true
  stratify_by_phenotype: true
  
# Phenotype analysis
phenotype_analysis:
  enabled: true
  phenotypes:
    - Test_Seeking
    - Avoidant
  stratified_analysis: true
  
# Validation subsample
validation:
  create_subsample: true
  n_per_stratum: 50
  validation_measures:
    - SCID-5
    - SSD-12
```

### Step 5: Create Makefile Targets

Add to `Makefile`:

```makefile
# Enhanced cohort with hierarchical index dates
cohort-enhanced:
	$(PYTHON) src/01_cohort_builder_enhanced.py $(ARGS)
	@echo "✓ Enhanced cohort with hierarchical index dates created"

# DSM-5 validated exposure
exposure-dsm5:
	$(PYTHON) src/02b_exposure_dsm5_validated.py $(ARGS)
	@echo "✓ DSM-5 validated exposure created"

# Phenotype-stratified analysis
analysis-stratified:
	$(PYTHON) src/10_stratified_analysis.py $(ARGS)
	@echo "✓ Stratified analysis by phenotype completed"

# Validation subsample
validation-sample:
	$(PYTHON) src/11_create_validation_sample.py $(ARGS)
	@echo "✓ Validation subsample created"

# Full pipeline with enhancements
pipeline-enhanced: clean cohort-enhanced exposure-dsm5 mediator outcome confounder \
                   pre-imputation master-impute ps-model causal-estimation \
                   analysis-stratified reporting
	@echo "✓ Enhanced pipeline completed successfully"
```

## Immediate Next Steps

1. **Backup current pipeline**:
   ```bash
   cp -r src/ src_backup_$(date +%Y%m%d)/
   git commit -am "Backup before IndexDate_lab enhancement"
   ```

2. **Implement Step 1** - Modify cohort builder:
   ```bash
   # Edit src/01_cohort_builder.py to add hierarchical dates
   # Test with small sample first
   python src/01_cohort_builder.py --sample_size=1000
   ```

3. **Implement Step 2** - Fix imputation:
   ```bash
   # Edit src/07b_missing_data_master.py to exclude datetime
   # Run imputation
   python src/07b_missing_data_master.py
   ```

4. **Verify fix**:
   ```python
   # Check results
   import pandas as pd
   imputed = pd.read_parquet("data_derived/master_table_imputed_001.parquet")
   print(f"Remaining missing: {imputed.isnull().sum().sum()}")
   print(f"Missing by column:\n{imputed.isnull().sum()[imputed.isnull().sum() > 0]}")
   ```

5. **Run enhanced pipeline**:
   ```bash
   make pipeline-enhanced
   ```

## Expected Outcomes

1. **Imputation will complete** with only datetime columns having missing values
2. **Three analysis streams**:
   - Primary: Test-seeking phenotype with traditional exposure (n=179,263)
   - Secondary: Full cohort with DSM-5 validated exposure (n=250,025)
   - Sensitivity: Phenotype-stratified comparisons

3. **Publication advantages**:
   - Novel identification of avoidant SSD phenotype
   - First DSM-5 aligned administrative algorithm
   - Methodological contribution to missing data handling

## Risk Mitigation

1. **Backup all data** before changes
2. **Test with small samples** first (--sample_size=1000)
3. **Validate each step** before proceeding
4. **Document all changes** in git commits
5. **Create validation plots** for phenotype distributions

## Success Criteria

- [ ] Phase 4 completes without errors
- [ ] Remaining missing values ≤ 300,000 (only datetime columns)
- [ ] All 250,025 patients have unified index dates
- [ ] Phenotype distribution matches expected (71.7% test-seeking)
- [ ] DSM-5 exposure identifies 15-25% prevalence
- [ ] Stratified results show consistent direction of effects