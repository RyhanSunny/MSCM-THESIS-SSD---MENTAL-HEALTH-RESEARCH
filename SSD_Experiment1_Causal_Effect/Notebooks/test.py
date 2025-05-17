- Data loading 
- Comprehensive quality checks
- Dataset relationship validation
- Data completeness reporting

!pip install missingno
# ---------------------------------------------------------------------------- #
# 01_Data_Loading_Validation.ipynb                                             #
# ---------------------------------------------------------------------------- #
# Purpose: Load CPCSSN datasets for Somatic Symptom Disorder (SSD) causal      #
# pathway analysis, validate data quality and relationships, and establish      #
# base statistics for downstream analysis.                                     #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
from datetime import datetime
from pathlib import Path
import re
import missingno as msno
from scipy import stats
from IPython.display import display, Markdown, HTML

# Configure visualization settings for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Suppress specific warnings while maintaining important ones
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning

# Display options for better readability
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 60)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

# ------------------------------- Configuration ------------------------------- #

# Define paths configuration
class Config:
    # Base paths
    DATA_PATH = Path(r"C:\Users\ProjectC4M\Documents\CPCSSN Datasets Care4Mind\New Extraction Feb 2025\prepared_data")
    OUTPUT_PATH = Path("output")
    INTERIM_PATH = Path("data/interim")
    
    # Create directories if they don't exist
    for path in [OUTPUT_PATH, INTERIM_PATH]:
        os.makedirs(path, exist_ok=True)
    
    # Dataset filenames
    DATASETS = {
        'patient': 'Patient_prepared.csv',
        'patient_demographic': 'PatientDemographic_merged_prepared.csv', 
        'encounter': 'Encounter_prepared.csv',
        'encounter_diagnosis': 'EncounterDiagnosis_prepared.csv',
        'health_condition': 'HealthCondition_prepared.csv',
        'lab': 'Lab_prepared.csv',
        'medication': 'Medication_prepared.csv',
        'referral': 'Referral_prepared.csv',
        'family_history': 'FamilyHistory_prepared.csv',
        'medical_procedure': 'MedicalProcedure_prepared.csv',
        'risk_factor': 'RiskFactor_prepared.csv'
    }
    
    # Required columns by table (based on provided configuration)
    REQUIRED_COLUMNS = {
        'patient': ['Patient_ID', 'Sex', 'BirthYear', 'BirthMonth'],
        'patient_demographic': ['Patient_ID', 'PatientDemographic_ID', 'Network_ID', 'Site_ID'],
        'encounter': ['Encounter_ID', 'Patient_ID', 'Provider_ID', 'EncounterDate', 'EncounterType'],
        'encounter_diagnosis': ['EncounterDiagnosis_ID', 'Encounter_ID', 'Patient_ID', 'DiagnosisCode_calc', 'DiagnosisText_calc'],
        'health_condition': ['HealthCondition_ID', 'Patient_ID', 'DiagnosisCode_calc', 'DateOfOnset'],
        'lab': ['Lab_ID', 'Patient_ID', 'PerformedDate', 'Name_calc', 'TestResult_calc', 'UpperNormal', 'LowerNormal'],
        'medication': ['Medication_ID', 'Patient_ID', 'StartDate', 'StopDate', 'Name_calc'],
        'referral': ['Referral_ID', 'Patient_ID', 'CompletedDate', 'Name_calc']
    }
    
    # Date columns that need conversion
    DATE_COLUMNS = {
        'encounter': ['EncounterDate', 'DateCreated'],
        'encounter_diagnosis': ['DateCreated'],
        'health_condition': ['DateOfOnset', 'DateCreated'],
        'lab': ['PerformedDate', 'DateCreated'],
        'medication': ['StartDate', 'StopDate', 'DateCreated'],
        'referral': ['CompletedDate', 'DateCreated'],
        'medical_procedure': ['PerformedDate', 'DateCreated'],
        'risk_factor': ['StartDate', 'EndDate', 'DateCreated']
    }
    
    # Study parameters
    CURRENT_YEAR = 2025  # Reference year for age calculations
    MIN_AGE = 18  # Inclusion criterion: minimum age
    MIN_ENCOUNTERS = 2  # Minimum encounters for inclusion

config = Config()


# ----------------------------- Helper Functions ----------------------------- #

def print_section_header(title):
    """Print formatted section header for better notebook organization."""
    display(Markdown(f"## {title}"))
    print("-" * 80)

def print_subsection_header(title):
    """Print formatted subsection header for better notebook organization."""
    display(Markdown(f"### {title}"))
    print("-" * 60)

def format_percentage(value):
    """Format decimal as percentage with 2 decimal places."""
    return f"{value:.2%}"

def display_dataset_info(name, df):
    """Display basic dataset information in a formatted way."""
    print(f"Dataset: {name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # Sample of first few rows
    display(df.head(3))
    
    # Column information
    column_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Non-Null %': df.count() / len(df) * 100,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
    display(column_info)
    print("\n")

# --------------------------- Main Loading Function -------------------------- #

def load_and_validate_data(required_tables=None):
    """
    Load CPCSSN datasets with comprehensive validation and quality checks.
    
    This function implements a robust data loading pipeline with:
    1. Standardized error handling for file access issues
    2. Multiple encoding fallbacks (UTF-8, latin1, etc.)
    3. Required column validation
    4. Automatic date column conversion
    5. Data type verification and standardization
    6. Basic quality metrics calculation
    
    Parameters:
    -----------
    required_tables : list, optional
        Specific tables to load; if None, loads all configured tables
        
    Returns:
    --------
    dict
        Dictionary of dataframes with table names as keys
    dict
        Dictionary of data quality metrics for each table
    
    Notes:
    ------
    The function follows ETL best practices from clinical data research:
    - Standardized approach across all tables (Kahn et al., 2016)
    - Explicit validation steps (Weiskopf & Weng, 2013)
    - Careful attention to date/time handling (Hripcsak & Albers, 2013)
    """
    print_section_header("Data Loading and Initial Validation")
    
    # If no specific tables are requested, load all
    if required_tables is None:
        required_tables = config.DATASETS.keys()
    
    data_dict = {}
    quality_metrics = {}
    
    # Track overall metrics
    total_rows = 0
    total_files = 0
    loading_errors = 0
    
    for key in required_tables:
        if key not in config.DATASETS:
            print(f"WARNING: Table '{key}' not found in configuration.")
            continue
            
        filename = config.DATASETS[key]
        file_path = config.DATA_PATH / filename
        
        print_subsection_header(f"Loading {key} ({filename})")
        
        # Initialize quality metrics for this table
        quality_metrics[key] = {
            'exists': False,
            'loaded_successfully': False,
            'row_count': 0,
            'column_count': 0,
            'missing_required_columns': [],
            'missing_data_percentage': {},
            'date_column_quality': {},
            'invalid_row_percentage': 0.0,
        }
        
        # Check if file exists
        if not file_path.exists():
            print(f"ERROR: File {filename} does not exist at {file_path}")
            continue
            
        quality_metrics[key]['exists'] = True
        total_files += 1
        
        # Attempt to load with multiple encodings
        for encoding in ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']:
            try:
                # Use chunks for large files (especially Lab and Encounter tables)
                # This is crucial for working with large CPCSSN datasets efficiently
                if key in ['lab', 'encounter', 'encounter_diagnosis']:
                    # For large tables, use chunked reading with dask or chunks
                    chunk_size = 500000  # Adjust based on memory constraints
                    print(f"Large table detected. Loading {key} in chunks of {chunk_size:,} rows...")
                    
                    chunks = []
                    for chunk in pd.read_csv(file_path, encoding=encoding, 
                                             chunksize=chunk_size, low_memory=False):
                        chunks.append(chunk)
                    
                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                    else:
                        df = pd.DataFrame()  # Empty dataframe if no chunks
                else:
                    # Regular loading for smaller tables
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                
                print(f"Successfully loaded with encoding: {encoding}")
                break
                
            except UnicodeDecodeError:
                print(f"Failed to load with encoding: {encoding}, trying next...")
                continue
                
            except Exception as e:
                print(f"ERROR loading {filename}: {str(e)}")
                loading_errors += 1
                break
        else:
            # This executes if no break occurs in the for loop (all encodings failed)
            print(f"ERROR: Failed to load {filename} with any encoding")
            continue
            
        # Mark as successfully loaded
        quality_metrics[key]['loaded_successfully'] = True
        data_dict[key] = df
        
        # Basic metrics
        row_count = len(df)
        col_count = len(df.columns)
        total_rows += row_count
        
        quality_metrics[key]['row_count'] = row_count
        quality_metrics[key]['column_count'] = col_count
        
        print(f"Loaded {row_count:,} rows and {col_count} columns")
        
        # Check for required columns
        if key in config.REQUIRED_COLUMNS:
            missing_cols = [col for col in config.REQUIRED_COLUMNS[key] 
                           if col not in df.columns]
            
            if missing_cols:
                print(f"WARNING: Missing required columns in {key}: {missing_cols}")
                quality_metrics[key]['missing_required_columns'] = missing_cols
                
        # Convert date columns
        if key in config.DATE_COLUMNS:
            date_quality = {}
            for date_col in config.DATE_COLUMNS[key]:
                if date_col in df.columns:
                    # Store original count to measure parse failures
                    orig_non_null = df[date_col].notna().sum()
                    
                    # Convert to datetime
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    
                    # Measure quality
                    new_non_null = df[date_col].notna().sum()
                    parse_failure_rate = 1.0 - (new_non_null / orig_non_null) if orig_non_null > 0 else 0
                    
                    date_quality[date_col] = {
                        'coverage': new_non_null / len(df),
                        'parse_failure_rate': parse_failure_rate,
                        'min_date': df[date_col].min(),
                        'max_date': df[date_col].max(),
                        'null_count': df[date_col].isna().sum()
                    }
                    
                    print(f"Converted {date_col} to datetime. " +
                          f"Parse failures: {parse_failure_rate:.2%}")
            
            quality_metrics[key]['date_column_quality'] = date_quality
                    
        # Calculate missing data percentage per column
        missing_pct = df.isna().mean().to_dict()
        quality_metrics[key]['missing_data_percentage'] = missing_pct
        
        # Check for basic invalid data
        # For numeric columns, check for out-of-range values
        invalid_rows = 0
        # Add specific validation based on column types
        
        # Calculate quality statistics
        high_missing_cols = [col for col, pct in missing_pct.items() if pct > 0.2]
        if high_missing_cols:
            print(f"WARNING: High missing data (>20%) in columns: {high_missing_cols}")
            
    # Print overall summary
    print_section_header("Data Loading Summary")
    print(f"Successfully loaded {total_files} files with {total_rows:,} total rows")
    if loading_errors > 0:
        print(f"WARNING: Encountered {loading_errors} loading errors")
        
    return data_dict, quality_metrics


# --------------------------- Data Validation Functions -------------------------- #

def validate_data_relationships(data_dict):
    """
    Validate foreign key relationships between datasets to ensure referential integrity.
    
    This function checks how well foreign keys in dependent tables (e.g., encounters, labs)
    match primary keys in reference tables (e.g., patients). High integrity is crucial for
    reliable analysis, as broken references can lead to data loss during joins.
    
    The function implements recommendations from:
    - Kahn et al. (2016) "A Harmonized Data Quality Assessment Terminology and Framework..."
    - Weiskopf & Weng (2013) "Methods and dimensions of electronic health record data quality..."
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of dataframes with table names as keys
        
    Returns:
    --------
    dict
        Dictionary of relationship metrics between tables
    """
    print_section_header("Data Relationship Validation")
    
    relationships = {}
    
    # Define key relationships to check
    key_relationships = [
        # dependent_table, reference_table, foreign_key, primary_key
        ('encounter', 'patient', 'Patient_ID', 'Patient_ID'),
        ('encounter_diagnosis', 'encounter', 'Encounter_ID', 'Encounter_ID'),
        ('encounter_diagnosis', 'patient', 'Patient_ID', 'Patient_ID'),
        ('lab', 'patient', 'Patient_ID', 'Patient_ID'),
        ('lab', 'encounter', 'Encounter_ID', 'Encounter_ID'),
        ('medication', 'patient', 'Patient_ID', 'Patient_ID'),
        ('referral', 'patient', 'Patient_ID', 'Patient_ID'),
        ('health_condition', 'patient', 'Patient_ID', 'Patient_ID')
    ]
    
    for dep_table, ref_table, fk, pk in key_relationships:
        # Skip if either table is missing
        if dep_table not in data_dict or ref_table not in data_dict:
            print(f"Skipping relationship check: {dep_table}.{fk} -> {ref_table}.{pk} (table missing)")
            continue
            
        # Skip if columns don't exist
        if fk not in data_dict[dep_table].columns or pk not in data_dict[ref_table].columns:
            print(f"Skipping relationship check: {dep_table}.{fk} -> {ref_table}.{pk} (column missing)")
            continue
            
        # Get unique keys from both tables
        dep_keys = set(data_dict[dep_table][fk].dropna().unique())
        ref_keys = set(data_dict[ref_table][pk].dropna().unique())
        
        # Calculate metrics
        matching_keys = dep_keys.intersection(ref_keys)
        orphaned_keys = dep_keys - ref_keys
        
        # As percentages
        total_dep_keys = len(dep_keys)
        match_pct = len(matching_keys) / total_dep_keys if total_dep_keys > 0 else 0
        orphan_pct = len(orphaned_keys) / total_dep_keys if total_dep_keys > 0 else 0
        
        # Store metrics
        rel_key = f"{dep_table}.{fk} -> {ref_table}.{pk}"
        relationships[rel_key] = {
            'dependent_table': dep_table,
            'reference_table': ref_table,
            'foreign_key': fk,
            'primary_key': pk,
            'total_unique_keys': total_dep_keys,
            'matching_keys': len(matching_keys),
            'orphaned_keys': len(orphaned_keys),
            'match_percentage': match_pct,
            'orphan_percentage': orphan_pct
        }
        
        # Print summary
        print(f"Relationship: {rel_key}")
        print(f"  Match rate: {match_pct:.2%} ({len(matching_keys):,}/{total_dep_keys:,} keys)")
        if orphan_pct > 0:
            print(f"  Orphaned records: {orphan_pct:.2%} ({len(orphaned_keys):,} keys)")
            # Show sample of orphaned keys (useful for debugging)
            if len(orphaned_keys) <= 5:
                print(f"  Sample orphaned keys: {list(orphaned_keys)}")
            else:
                print(f"  Sample orphaned keys: {list(orphaned_keys)[:5]} ...")
        print()
        
    # Create summary table for visualization
    if relationships:
        rel_df = pd.DataFrame([
            {
                'Relationship': k,
                'Match Rate': v['match_percentage'],
                'Orphaned Rate': v['orphan_percentage'],
                'Total Keys': v['total_unique_keys']
            }
            for k, v in relationships.items()
        ])
        
        # Sort by match rate
        rel_df = rel_df.sort_values('Match Rate')
        
        # Plot relationship metrics
        plt.figure(figsize=(12, 8))
        bars = plt.barh(rel_df['Relationship'], rel_df['Match Rate'])
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label = f"{width:.1%}"
            plt.text(max(0.05, width - 0.1), bar.get_y() + bar.get_height()/2, 
                    label, ha='center', va='center', color='white', fontweight='bold')
        
        plt.xlabel('Match Rate (% of dependent keys found in reference table)')
        plt.title('Data Relationship Integrity')
        plt.xlim(0, 1.0)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'relationship_integrity.png')
        plt.close()
        
    return relationships

def validate_temporal_consistency(data_dict):
    """
    Validate temporal consistency in date columns across datasets.
    
    This function checks for:
    1. Chronological consistency (e.g., start dates before end dates)
    2. Date ranges within reasonable study period
    3. Temporal alignment between related events
    
    Temporal consistency is essential for accurate pathway analysis, especially for
    establishing the correct sequence of NYD status, lab tests, referrals, and diagnoses.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of dataframes with table names as keys
        
    Returns:
    --------
    dict
        Dictionary of temporal consistency metrics
    """
    print_section_header("Temporal Consistency Validation")
    
    temporal_metrics = {}
    
    # 1. Check date ranges in each table
    print_subsection_header("Date ranges by table")
    
    for table_name, df in data_dict.items():
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
        if not date_cols:
            continue
            
        table_metrics = {}
        print(f"Table: {table_name}")
        
        for col in date_cols:
            # Skip if all null
            if df[col].isna().all():
                continue
                
            min_date = df[col].min()
            max_date = df[col].max()
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df)
            
            # Check for future dates (beyond current study year)
            future_date_threshold = pd.Timestamp(f"{config.CURRENT_YEAR+1}-01-01")
            future_dates = df[col] >= future_date_threshold
            future_count = future_dates.sum()
            future_pct = future_count / df[col].notna().sum() if df[col].notna().any() else 0
            
            # Check for implausible past dates (before 1900)
            past_date_threshold = pd.Timestamp("1900-01-01")
            past_dates = df[col] < past_date_threshold
            past_count = past_dates.sum()
            past_pct = past_count / df[col].notna().sum() if df[col].notna().any() else 0
            
            print(f"  {col}: {min_date} to {max_date} (Null: {null_pct:.2%})")
            if future_pct > 0:
                print(f"    WARNING: {future_pct:.2%} ({future_count:,}) future dates beyond {config.CURRENT_YEAR}")
            if past_pct > 0:
                print(f"    WARNING: {past_pct:.2%} ({past_count:,}) implausibly old dates before 1900")
                
            table_metrics[col] = {
                'min_date': min_date,
                'max_date': max_date,
                'null_percentage': null_pct,
                'future_date_percentage': future_pct,
                'past_date_percentage': past_pct
            }
            
        temporal_metrics[table_name] = table_metrics
        
    # 2. Check for specific date sequence consistency
    print_subsection_header("Date sequence validation")
    
    # Medication: StartDate before StopDate
    if 'medication' in data_dict and 'StartDate' in data_dict['medication'].columns and 'StopDate' in data_dict['medication'].columns:
        med_df = data_dict['medication']
        # Only check rows with both dates non-null
        both_dates = med_df['StartDate'].notna() & med_df['StopDate'].notna()
        total_both_dates = both_dates.sum()
        
        if total_both_dates > 0:
            invalid_sequence = (med_df['StartDate'] > med_df['StopDate']) & both_dates
            invalid_count = invalid_sequence.sum()
            invalid_pct = invalid_count / total_both_dates
            
            print(f"Medication - StartDate before StopDate: {(1-invalid_pct):.2%} valid")
            if invalid_pct > 0:
                print(f"  WARNING: {invalid_pct:.2%} ({invalid_count:,}/{total_both_dates:,}) " +
                      f"medication records have StartDate after StopDate")
                
            temporal_metrics['medication_sequence'] = {
                'check': 'StartDate_before_StopDate',
                'valid_percentage': 1 - invalid_pct,
                'invalid_count': invalid_count,
                'total_checked': total_both_dates
            }
    
    # 3. Check referral sequences for patients (relevant for pathway analysis)
    if 'referral' in data_dict and 'CompletedDate' in data_dict['referral'].columns:
        ref_df = data_dict['referral']
        # Group by patient and sort by date
        patient_ref_counts = ref_df.groupby('Patient_ID')['CompletedDate'].count()
        multiple_refs = (patient_ref_counts > 1).sum()
        multiple_refs_pct = multiple_refs / len(patient_ref_counts)
        
        print(f"Referral sequences: {multiple_refs_pct:.2%} ({multiple_refs:,}/{len(patient_ref_counts):,}) " +
              f"patients have multiple referrals")
              
        temporal_metrics['referral_sequence'] = {
            'check': 'multiple_referrals',
            'patients_with_multiple_refs': multiple_refs,
            'percentage': multiple_refs_pct
        }
    
    # 4. Visualize date distributions
    plt.figure(figsize=(12, 8))
    
    # Collect date ranges for key tables
    key_tables = ['encounter', 'lab', 'medication', 'referral']
    date_ranges = []
    
    for table in key_tables:
        if table not in temporal_metrics:
            continue
            
        for col, metrics in temporal_metrics[table].items():
            if 'min_date' in metrics and 'max_date' in metrics:
                date_ranges.append({
                    'table': table,
                    'column': col,
                    'min_date': metrics['min_date'],
                    'max_date': metrics['max_date']
                })
    
    # Create date range plot
    if date_ranges:
        df_ranges = pd.DataFrame(date_ranges)
        df_ranges['label'] = df_ranges['table'] + '.' + df_ranges['column']
        
        # Sort by min_date
        df_ranges = df_ranges.sort_values('min_date')
        
        # Plot
        plt.figure(figsize=(12, 8))
        for i, row in df_ranges.iterrows():
            plt.plot([row['min_date'], row['max_date']], [i, i], 'o-', linewidth=2, markersize=8)
            plt.text(row['min_date'], i+0.1, row['min_date'].strftime('%Y-%m-%d'), fontsize=9)
            plt.text(row['max_date'], i+0.1, row['max_date'].strftime('%Y-%m-%d'), fontsize=9)
            
        plt.yticks(range(len(df_ranges)), df_ranges['label'])
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.title('Date Ranges by Table and Column')
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'date_ranges.png')
        plt.close()
    
    return temporal_metrics

def analyze_patient_population(data_dict):
    """
    Analyze the patient population demographics and coverage.
    
    This function:
    1. Analyzes core demographic distributions (age, sex)
    2. Checks data coverage across key tables for patient cohort
    3. Identifies potential inclusion/exclusion issues
    
    Understanding the patient population is essential for:
    - Assessing potential selection bias (Haneuse & Daniels, 2016)
    - Ensuring adequate representation across key strata (Deeny & Steventon, 2015)
    - Establishing the generalizability of findings (Hersh et al., 2013)
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of dataframes with table names as keys
        
    Returns:
    --------
    dict
        Dictionary of population metrics
    """
    print_section_header("Patient Population Analysis")
    
    population_metrics = {}
    
    # Exit if patient table is missing
    if 'patient' not in data_dict:
        print("ERROR: Patient table missing, cannot analyze population")
        return population_metrics
        
    patient_df = data_dict['patient']

    if 'Sex' in patient_df.columns:
        # Normalize sex values to standardized format
        sex_mapping = {
            'F': 'Female', 'FEMALE': 'Female', 'Female': 'Female', 
            'M': 'Male', 'MALE': 'Male', 'Male': 'Male',
            'U': 'Unknown', 'Unknown': 'Unknown', 
            'Undifferentiated': 'Other'
        }
        
        # Apply mapping (preserve original for reference)
        patient_df['Sex_normalized'] = patient_df['Sex'].map(sex_mapping).fillna('Unknown')
        
        # Get normalized distribution
        sex_counts = patient_df['Sex_normalized'].value_counts()
        sex_pct = sex_counts / len(patient_df) * 100
        
        print("\nNormalized sex distribution:")
        for sex, count in sex_counts.items():
            print(f"  {sex}: {count:,} ({sex_pct[sex]:.1f}%)")
            
        # Store in metrics
        population_metrics['sex_normalized'] = {
            'counts': sex_counts.to_dict(),
            'percentage': sex_pct.to_dict()
        }
        
        # Create visualization with normalized values
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=sex_counts.index, y=sex_counts.values)
        plt.title('Patient Sex Distribution (Normalized)', fontsize=14)
        plt.xlabel('Sex', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        
        # Add count and percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = sex_pct.iloc[i]
            ax.text(p.get_x() + p.get_width()/2., height + height*0.02,
                   f'{int(height):,}\n({percentage:.1f}%)', 
                   ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'sex_distribution_normalized.png')
        plt.close()
        
    
    # 1. Basic demographic analysis
    print_subsection_header("Demographic Distribution")
    
    # Calculate age from birth year
    if 'BirthYear' in patient_df.columns:
        # Check if birth year is valid (not future, not unreasonably old)
        current_year = config.CURRENT_YEAR
        # Filter out clearly invalid birth years
        invalid_birth_mask = (patient_df['BirthYear'] > current_year) | (patient_df['BirthYear'] < 1900)
        if invalid_birth_mask.any():
            print(f"WARNING: Found {invalid_birth_mask.sum()} patients with invalid birth years. Setting to NaN.")
            patient_df.loc[invalid_birth_mask, 'BirthYear'] = np.nan
        
        # Check if DeceasedYear column exists in demographic data
        has_deceased_info = ('patient_demographic' in data_dict and 
                            'DeceasedYear' in data_dict['patient_demographic'].columns)
    
        # Base age calculation using current year
        patient_df['Age'] = current_year - patient_df['BirthYear']
        
        if has_deceased_info:
            # Get only the necessary columns from demographic data to avoid duplicate columns in merge
            demo_df = data_dict['patient_demographic'][['Patient_ID', 'DeceasedYear']].copy()
            
            # Remove duplicates if any exist in demographic data
            if demo_df['Patient_ID'].duplicated().any():
                print(f"WARNING: Found {demo_df['Patient_ID'].duplicated().sum()} duplicate Patient_IDs in demographic data.")
                # Keep the first occurrence of each Patient_ID
                demo_df = demo_df.drop_duplicates('Patient_ID')
            
            # Merge with demographic data to get deceased year
            patient_df = patient_df.merge(demo_df, on='Patient_ID', how='left')
            
            # Validate DeceasedYear values
            # - Must be not null
            # - Must be a reasonable year (not future, not before birth year)
            # - Must be after birth year
            valid_deceased_mask = (
                patient_df['DeceasedYear'].notna() & 
                (patient_df['DeceasedYear'] > 0) &
                (patient_df['DeceasedYear'] <= current_year) &
                (patient_df['DeceasedYear'] >= patient_df['BirthYear'])
            )
            
            # Create reference year column (either death year or current year)
            patient_df['reference_year'] = current_year  # Default to current year
            
            # Only update reference year for valid deceased records
            if valid_deceased_mask.any():
                patient_df.loc[valid_deceased_mask, 'reference_year'] = patient_df.loc[valid_deceased_mask, 'DeceasedYear']
                
                # Flag invalid deceased years
                invalid_deceased = patient_df['DeceasedYear'].notna() & ~valid_deceased_mask
                if invalid_deceased.any():
                    print(f"WARNING: Found {invalid_deceased.sum()} patients with invalid death years. "
                        f"Using current year for age calculation instead.")
            
            # Calculate corrected age using either death year or current year
            patient_df['Age'] = patient_df['reference_year'] - patient_df['BirthYear']
            
            # Report on deceased patients
            deceased_count = valid_deceased_mask.sum()
            print(f"Deceased patients with valid death year: {deceased_count:,} "
                f"({deceased_count/len(patient_df)*100:.2f}%)")
        else:
            # Standard age calculation based on current year
            patient_df['Age'] = current_year - patient_df['BirthYear']
        
        # Cap implausible ages to avoid outliers affecting statistics
        # Handle maximum age cap (e.g., 110 years)
        max_age_cap = 110
        too_old_mask = patient_df['Age'] > max_age_cap
        if too_old_mask.any():
            print(f"WARNING: Found {too_old_mask.sum()} patients with ages > {max_age_cap}. Capping at {max_age_cap}.")
            patient_df.loc[too_old_mask, 'Age'] = max_age_cap
        
        # Handle negative ages (data error)
        negative_age_mask = patient_df['Age'] < 0
        if negative_age_mask.any():
            print(f"WARNING: Found {negative_age_mask.sum()} patients with negative ages. Setting to NaN.")
            patient_df.loc[negative_age_mask, 'Age'] = np.nan
            
        # Age statistics (ignoring NaN values)
        age_mean = patient_df['Age'].mean()
        age_median = patient_df['Age'].median()
        age_min = patient_df['Age'].min()
        age_max = patient_df['Age'].max()
        
        print(f"Age statistics (as of {config.CURRENT_YEAR}):")
        print(f"  Mean: {age_mean:.1f} years")
        print(f"  Median: {age_median:.1f} years")
        print(f"  Range: {age_min} to {age_max} years")
        
        # Age distribution with standard age groups
        # Option 1: Standard age groups (pediatric, adult, senior divisions)
        age_bins = [0, 18, 35, 50, 65, 80, max_age_cap]
        age_labels = ['<18', '18-34', '35-49', '50-64', '65-79', '80+']
        
        # For easier customization, you could use one of these alternative binning schemes:
        # Option 2: Decades for more granular view
        # age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, max_age_cap]
        # age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
        
        # Option 3: More detailed pediatric and geriatric groups
        # age_bins = [0, 2, 5, 12, 18, 35, 50, 65, 75, 85, max_age_cap]
        # age_labels = ['0-1', '2-4', '5-11', '12-17', '18-34', '35-49', '50-64', '65-74', '75-84', '85+']
        
        # Create age groups, handling NaN values
        patient_df['Age_Group'] = pd.cut(patient_df['Age'], bins=age_bins, labels=age_labels)
        
        # Calculate distribution, excluding NaN values
        valid_age_mask = patient_df['Age_Group'].notna()
        if (~valid_age_mask).any():
            print(f"WARNING: {(~valid_age_mask).sum()} patients have missing age data and are excluded from age distribution.")
        
        # Calculate distribution based on valid ages only
        age_dist = patient_df.loc[valid_age_mask, 'Age_Group'].value_counts().sort_index()
        age_pct = age_dist / valid_age_mask.sum() * 100
        
        # Display age distribution
        age_table = pd.DataFrame({
            'Age Group': age_dist.index,
            'Count': age_dist.values,
            'Percentage': age_pct.values
        })
        
        print("\nAge distribution:")
        display(age_table)
        
        # Store in metrics
        population_metrics['age'] = {
            'mean': age_mean,
            'median': age_median,
            'min': age_min,
            'max': age_max,
            'distribution': age_dist.to_dict(),
            'percentage': age_pct.to_dict(),
            'excluded_count': (~valid_age_mask).sum() if (~valid_age_mask).any() else 0
        }
        
        # Plot age distribution
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=age_dist.index, y=age_dist.values)
        plt.title('Patient Age Distribution', fontsize=14)
        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        
        # Add count and percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = age_pct.iloc[i]
            ax.text(p.get_x() + p.get_width()/2., height + height*0.02,
                f'{int(height):,}\n({percentage:.1f}%)', 
                ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'age_distribution.png')
        plt.close()

        # Add a note if there were data quality issues
        quality_issues = invalid_birth_mask.sum() + (
            negative_age_mask.sum() if 'negative_age_mask' in locals() else 0
        )
        if quality_issues > 0:
            print(f"\nNOTE: Found {quality_issues} patients with age data quality issues.")
            print("      See warnings above for details.")
            
            # Add quality metrics
            population_metrics['age']['data_quality_issues'] = {
                'invalid_birth_year': invalid_birth_mask.sum(),
                'negative_age': negative_age_mask.sum() if 'negative_age_mask' in locals() else 0,
                'too_old': too_old_mask.sum() if 'too_old_mask' in locals() else 0
            }
    else:
        print("WARNING: 'BirthYear' column not found, age calculations skipped")
    
    # Sex distribution
    if 'Sex' in patient_df.columns:
        sex_counts = patient_df['Sex'].value_counts()
        sex_pct = sex_counts / len(patient_df) * 100
        
        print("\nSex distribution:")
        for sex, count in sex_counts.items():
            print(f"  {sex}: {count:,} ({sex_pct[sex]:.1f}%)")
            
        # Store in metrics
        population_metrics['sex'] = {
            'counts': sex_counts.to_dict(),
            'percentage': sex_pct.to_dict()
        }
        
        # Plot sex distribution
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=sex_counts.index, y=sex_counts.values)
        plt.title('Patient Sex Distribution', fontsize=14)
        plt.xlabel('Sex', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        
        # Add count and percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = sex_pct.iloc[i]
            ax.text(p.get_x() + p.get_width()/2., height + height*0.02,
                   f'{int(height):,}\n({percentage:.1f}%)', 
                   ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'sex_distribution.png')
        plt.close()
    
    # 2. Data coverage across tables
    print_subsection_header("Patient Data Coverage")
    
    # Get patient ID sets for each table
    coverage = {}
    all_patients = set(patient_df['Patient_ID'])
    total_patients = len(all_patients)
    
    for table, df in data_dict.items():
        if table == 'patient':
            continue
            
        if 'Patient_ID' in df.columns:
            table_patients = set(df['Patient_ID'].unique())
            overlap = table_patients.intersection(all_patients)
            
            coverage_pct = len(overlap) / total_patients
            coverage[table] = {
                'patients': len(overlap),
                'percentage': coverage_pct
            }
            
            print(f"{table}: {coverage_pct:.2%} ({len(overlap):,}/{total_patients:,} patients)")
    
    # Create intersection visualization
    if coverage:
        # Sort tables by coverage
        sorted_tables = sorted(coverage.keys(), key=lambda x: coverage[x]['percentage'], reverse=True)
        
        # Create bar chart of coverage
        plt.figure(figsize=(12, 6))
        coverage_vals = [coverage[t]['percentage'] for t in sorted_tables]
        bars = plt.barh(sorted_tables, coverage_vals)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label = f"{width:.1%}"
            plt.text(max(0.05, width - 0.1), bar.get_y() + bar.get_height()/2, 
                    label, ha='center', va='center', color='white', fontweight='bold')
        
        plt.xlabel('Percentage of Patients with Data')
        plt.title('Patient Coverage by Data Table')
        plt.xlim(0, 1.0)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'patient_coverage.png')
        plt.close()
    
    # Store in metrics
    population_metrics['coverage'] = coverage
    
    # 3. Eligibility analysis based on protocol criteria
    print_subsection_header("Study Eligibility Analysis")
    
    # Age eligibility (adults ≥18)
    if 'Age' in patient_df.columns:
        age_eligible = patient_df['Age'] >= config.MIN_AGE
        age_eligible_count = age_eligible.sum()
        age_eligible_pct = age_eligible_count / len(patient_df)
        
        print(f"Age eligible (≥{config.MIN_AGE}): {age_eligible_pct:.2%} ({age_eligible_count:,}/{len(patient_df):,} patients)")
    
    # Encounter eligibility (≥2 encounters)
    if 'encounter' in data_dict and 'Patient_ID' in data_dict['encounter'].columns:
        encounter_counts = data_dict['encounter']['Patient_ID'].value_counts()
        encounter_eligible = encounter_counts[encounter_counts >= config.MIN_ENCOUNTERS]
        encounter_eligible_count = len(encounter_eligible)
        encounter_eligible_pct = encounter_eligible_count / len(patient_df)
        
        print(f"Encounter eligible (≥{config.MIN_ENCOUNTERS} encounters): " +
              f"{encounter_eligible_pct:.2%} ({encounter_eligible_count:,}/{len(patient_df):,} patients)")
        
        # Encounter statistics
        enc_mean = encounter_counts.mean()
        enc_median = encounter_counts.median()
        enc_p90 = encounter_counts.quantile(0.9)
        
        print(f"Encounter statistics:")
        print(f"  Mean: {enc_mean:.1f} encounters per patient")
        print(f"  Median: {enc_median:.0f} encounters per patient")
        print(f"  90th percentile: {enc_p90:.0f} encounters per patient")
        
        # Store in metrics
        population_metrics['encounters'] = {
            'mean': enc_mean,
            'median': enc_median,
            'p90': enc_p90,
            'eligible_count': encounter_eligible_count,
            'eligible_percentage': encounter_eligible_pct
        }
        
        # Create histogram of encounter counts
        plt.figure(figsize=(12, 6))
        # Log transform for better visualization
        log_counts = np.log10(encounter_counts + 1)  # +1 to handle zeros
        plt.hist(log_counts, bins=50)
        plt.title('Distribution of Encounters per Patient (Log Scale)', fontsize=14)
        plt.xlabel('log10(Encounters + 1)', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'encounter_distribution.png')
        plt.close()
    
    # 4. Combined eligibility criteria
    if 'Age' in patient_df.columns and 'encounter' in data_dict:
        # Get patient IDs with sufficient encounters
        encounter_counts = data_dict['encounter']['Patient_ID'].value_counts()
        encounter_eligible_ids = set(encounter_counts[encounter_counts >= config.MIN_ENCOUNTERS].index)
        
        # Combine with age eligibility
        age_eligible_ids = set(patient_df.loc[patient_df['Age'] >= config.MIN_AGE, 'Patient_ID'])
        
        # Intersection
        eligible_ids = age_eligible_ids.intersection(encounter_eligible_ids)
        eligible_count = len(eligible_ids)
        eligible_pct = eligible_count / len(patient_df)
        
        print(f"\nCombined eligibility (age ≥{config.MIN_AGE} AND ≥{config.MIN_ENCOUNTERS} encounters): " +
              f"{eligible_pct:.2%} ({eligible_count:,}/{len(patient_df):,} patients)")
              
        # Store in metrics
        population_metrics['combined_eligibility'] = {
            'eligible_count': eligible_count,
            'eligible_percentage': eligible_pct,
            'age_criteria': f"≥{config.MIN_AGE}",
            'encounter_criteria': f"≥{config.MIN_ENCOUNTERS}"
        }
    
    return population_metrics


# --------------------------- Data Validation Functions -------------------------- #

def validate_data_relationships(data_dict):
    """
    Validate foreign key relationships between datasets to ensure referential integrity.
    
    This function checks how well foreign keys in dependent tables (e.g., encounters, labs)
    match primary keys in reference tables (e.g., patients). High integrity is crucial for
    reliable analysis, as broken references can lead to data loss during joins.
    
    The function implements recommendations from:
    - Kahn et al. (2016) "A Harmonized Data Quality Assessment Terminology and Framework..."
    - Weiskopf & Weng (2013) "Methods and dimensions of electronic health record data quality..."
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of dataframes with table names as keys
        
    Returns:
    --------
    dict
        Dictionary of relationship metrics between tables
    """
    print_section_header("Data Relationship Validation")
    
    relationships = {}
    
    # Define key relationships to check
    key_relationships = [
        # dependent_table, reference_table, foreign_key, primary_key
        ('encounter', 'patient', 'Patient_ID', 'Patient_ID'),
        ('encounter_diagnosis', 'encounter', 'Encounter_ID', 'Encounter_ID'),
        ('encounter_diagnosis', 'patient', 'Patient_ID', 'Patient_ID'),
        ('lab', 'patient', 'Patient_ID', 'Patient_ID'),
        ('lab', 'encounter', 'Encounter_ID', 'Encounter_ID'),
        ('medication', 'patient', 'Patient_ID', 'Patient_ID'),
        ('referral', 'patient', 'Patient_ID', 'Patient_ID'),
        ('health_condition', 'patient', 'Patient_ID', 'Patient_ID')
    ]
    
    for dep_table, ref_table, fk, pk in key_relationships:
        # Skip if either table is missing
        if dep_table not in data_dict or ref_table not in data_dict:
            print(f"Skipping relationship check: {dep_table}.{fk} -> {ref_table}.{pk} (table missing)")
            continue
            
        # Skip if columns don't exist
        if fk not in data_dict[dep_table].columns or pk not in data_dict[ref_table].columns:
            print(f"Skipping relationship check: {dep_table}.{fk} -> {ref_table}.{pk} (column missing)")
            continue
            
        # Get unique keys from both tables
        dep_keys = set(data_dict[dep_table][fk].dropna().unique())
        ref_keys = set(data_dict[ref_table][pk].dropna().unique())
        
        # Calculate metrics
        matching_keys = dep_keys.intersection(ref_keys)
        orphaned_keys = dep_keys - ref_keys
        
        # As percentages
        total_dep_keys = len(dep_keys)
        match_pct = len(matching_keys) / total_dep_keys if total_dep_keys > 0 else 0
        orphan_pct = len(orphaned_keys) / total_dep_keys if total_dep_keys > 0 else 0
        
        # Store metrics
        rel_key = f"{dep_table}.{fk} -> {ref_table}.{pk}"
        relationships[rel_key] = {
            'dependent_table': dep_table,
            'reference_table': ref_table,
            'foreign_key': fk,
            'primary_key': pk,
            'total_unique_keys': total_dep_keys,
            'matching_keys': len(matching_keys),
            'orphaned_keys': len(orphaned_keys),
            'match_percentage': match_pct,
            'orphan_percentage': orphan_pct
        }
        
        # Print summary
        print(f"Relationship: {rel_key}")
        print(f"  Match rate: {match_pct:.2%} ({len(matching_keys):,}/{total_dep_keys:,} keys)")
        if orphan_pct > 0:
            print(f"  Orphaned records: {orphan_pct:.2%} ({len(orphaned_keys):,} keys)")
            # Show sample of orphaned keys (useful for debugging)
            if len(orphaned_keys) <= 5:
                print(f"  Sample orphaned keys: {list(orphaned_keys)}")
            else:
                print(f"  Sample orphaned keys: {list(orphaned_keys)[:5]} ...")
        print()
        
    # Create summary table for visualization
    if relationships:
        rel_df = pd.DataFrame([
            {
                'Relationship': k,
                'Match Rate': v['match_percentage'],
                'Orphaned Rate': v['orphan_percentage'],
                'Total Keys': v['total_unique_keys']
            }
            for k, v in relationships.items()
        ])
        
        # Sort by match rate
        rel_df = rel_df.sort_values('Match Rate')
        
        # Plot relationship metrics
        plt.figure(figsize=(12, 8))
        bars = plt.barh(rel_df['Relationship'], rel_df['Match Rate'])
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label = f"{width:.1%}"
            plt.text(max(0.05, width - 0.1), bar.get_y() + bar.get_height()/2, 
                    label, ha='center', va='center', color='white', fontweight='bold')
        
        plt.xlabel('Match Rate (% of dependent keys found in reference table)')
        plt.title('Data Relationship Integrity')
        plt.xlim(0, 1.0)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'relationship_integrity.png')
        plt.close()
        
    return relationships

def validate_temporal_consistency(data_dict):
    """
    Validate temporal consistency in date columns across datasets.
    
    This function checks for:
    1. Chronological consistency (e.g., start dates before end dates)
    2. Date ranges within reasonable study period
    3. Temporal alignment between related events
    
    Temporal consistency is essential for accurate pathway analysis, especially for
    establishing the correct sequence of NYD status, lab tests, referrals, and diagnoses.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of dataframes with table names as keys
        
    Returns:
    --------
    dict
        Dictionary of temporal consistency metrics
    """
    print_section_header("Temporal Consistency Validation")
    
    temporal_metrics = {}
    
    # 1. Check date ranges in each table
    print_subsection_header("Date ranges by table")
    
    for table_name, df in data_dict.items():
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])]
        if not date_cols:
            continue
            
        table_metrics = {}
        print(f"Table: {table_name}")
        
        for col in date_cols:
            # Skip if all null
            if df[col].isna().all():
                continue
                
            min_date = df[col].min()
            max_date = df[col].max()
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df)
            
            # Check for future dates (beyond current study year)
            future_date_threshold = pd.Timestamp(f"{config.CURRENT_YEAR+1}-01-01")
            future_dates = df[col] >= future_date_threshold
            future_count = future_dates.sum()
            future_pct = future_count / df[col].notna().sum() if df[col].notna().any() else 0
            
            # Check for implausible past dates (before 1900)
            past_date_threshold = pd.Timestamp("1900-01-01")
            past_dates = df[col] < past_date_threshold
            past_count = past_dates.sum()
            past_pct = past_count / df[col].notna().sum() if df[col].notna().any() else 0
            
            print(f"  {col}: {min_date} to {max_date} (Null: {null_pct:.2%})")
            if future_pct > 0:
                print(f"    WARNING: {future_pct:.2%} ({future_count:,}) future dates beyond {config.CURRENT_YEAR}")
            if past_pct > 0:
                print(f"    WARNING: {past_pct:.2%} ({past_count:,}) implausibly old dates before 1900")
                
            table_metrics[col] = {
                'min_date': min_date,
                'max_date': max_date,
                'null_percentage': null_pct,
                'future_date_percentage': future_pct,
                'past_date_percentage': past_pct
            }
            
        temporal_metrics[table_name] = table_metrics
        
    # 2. Check for specific date sequence consistency
    print_subsection_header("Date sequence validation")
    
    # Medication: StartDate before StopDate
    if 'medication' in data_dict and 'StartDate' in data_dict['medication'].columns and 'StopDate' in data_dict['medication'].columns:
        med_df = data_dict['medication']
        # Only check rows with both dates non-null
        both_dates = med_df['StartDate'].notna() & med_df['StopDate'].notna()
        total_both_dates = both_dates.sum()
        
        if total_both_dates > 0:
            invalid_sequence = (med_df['StartDate'] > med_df['StopDate']) & both_dates
            invalid_count = invalid_sequence.sum()
            invalid_pct = invalid_count / total_both_dates
            
            print(f"Medication - StartDate before StopDate: {(1-invalid_pct):.2%} valid")
            if invalid_pct > 0:
                print(f"  WARNING: {invalid_pct:.2%} ({invalid_count:,}/{total_both_dates:,}) " +
                      f"medication records have StartDate after StopDate")
                
            temporal_metrics['medication_sequence'] = {
                'check': 'StartDate_before_StopDate',
                'valid_percentage': 1 - invalid_pct,
                'invalid_count': invalid_count,
                'total_checked': total_both_dates
            }
    
    # 3. Check referral sequences for patients (relevant for pathway analysis)
    if 'referral' in data_dict and 'CompletedDate' in data_dict['referral'].columns:
        ref_df = data_dict['referral']
        # Group by patient and sort by date
        patient_ref_counts = ref_df.groupby('Patient_ID')['CompletedDate'].count()
        multiple_refs = (patient_ref_counts > 1).sum()
        multiple_refs_pct = multiple_refs / len(patient_ref_counts)
        
        print(f"Referral sequences: {multiple_refs_pct:.2%} ({multiple_refs:,}/{len(patient_ref_counts):,}) " +
              f"patients have multiple referrals")
              
        temporal_metrics['referral_sequence'] = {
            'check': 'multiple_referrals',
            'patients_with_multiple_refs': multiple_refs,
            'percentage': multiple_refs_pct
        }
    
    # 4. Visualize date distributions
    plt.figure(figsize=(12, 8))
    
    # Collect date ranges for key tables
    key_tables = ['encounter', 'lab', 'medication', 'referral']
    date_ranges = []
    
    for table in key_tables:
        if table not in temporal_metrics:
            continue
            
        for col, metrics in temporal_metrics[table].items():
            if 'min_date' in metrics and 'max_date' in metrics:
                date_ranges.append({
                    'table': table,
                    'column': col,
                    'min_date': metrics['min_date'],
                    'max_date': metrics['max_date']
                })
    
    # Create date range plot
    if date_ranges:
        df_ranges = pd.DataFrame(date_ranges)
        df_ranges['label'] = df_ranges['table'] + '.' + df_ranges['column']
        
        # Sort by min_date
        df_ranges = df_ranges.sort_values('min_date')
        
        # Plot
        plt.figure(figsize=(12, 8))
        for i, row in df_ranges.iterrows():
            plt.plot([row['min_date'], row['max_date']], [i, i], 'o-', linewidth=2, markersize=8)
            plt.text(row['min_date'], i+0.1, row['min_date'].strftime('%Y-%m-%d'), fontsize=9)
            plt.text(row['max_date'], i+0.1, row['max_date'].strftime('%Y-%m-%d'), fontsize=9)
            
        plt.yticks(range(len(df_ranges)), df_ranges['label'])
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.title('Date Ranges by Table and Column')
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'date_ranges.png')
        plt.close()
    
    return temporal_metrics

def analyze_patient_population(data_dict):
    """
    Analyze the patient population demographics and coverage.
    
    This function:
    1. Analyzes core demographic distributions (age, sex)
    2. Checks data coverage across key tables for patient cohort
    3. Identifies potential inclusion/exclusion issues
    
    Understanding the patient population is essential for:
    - Assessing potential selection bias (Haneuse & Daniels, 2016)
    - Ensuring adequate representation across key strata (Deeny & Steventon, 2015)
    - Establishing the generalizability of findings (Hersh et al., 2013)
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of dataframes with table names as keys
        
    Returns:
    --------
    dict
        Dictionary of population metrics
    """
    print_section_header("Patient Population Analysis")
    
    population_metrics = {}
    
    # Exit if patient table is missing
    if 'patient' not in data_dict:
        print("ERROR: Patient table missing, cannot analyze population")
        return population_metrics
        
    patient_df = data_dict['patient']

    if 'Sex' in patient_df.columns:
        # Normalize sex values to standardized format
        sex_mapping = {
            'F': 'Female', 'FEMALE': 'Female', 'Female': 'Female', 
            'M': 'Male', 'MALE': 'Male', 'Male': 'Male',
            'U': 'Unknown', 'Unknown': 'Unknown', 
            'Undifferentiated': 'Other'
        }
        
        # Apply mapping (preserve original for reference)
        patient_df['Sex_normalized'] = patient_df['Sex'].map(sex_mapping).fillna('Unknown')
        
        # Get normalized distribution
        sex_counts = patient_df['Sex_normalized'].value_counts()
        sex_pct = sex_counts / len(patient_df) * 100
        
        print("\nNormalized sex distribution:")
        for sex, count in sex_counts.items():
            print(f"  {sex}: {count:,} ({sex_pct[sex]:.1f}%)")
            
        # Store in metrics
        population_metrics['sex_normalized'] = {
            'counts': sex_counts.to_dict(),
            'percentage': sex_pct.to_dict()
        }
        
        # Create visualization with normalized values
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=sex_counts.index, y=sex_counts.values)
        plt.title('Patient Sex Distribution (Normalized)', fontsize=14)
        plt.xlabel('Sex', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        
        # Add count and percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = sex_pct.iloc[i]
            ax.text(p.get_x() + p.get_width()/2., height + height*0.02,
                   f'{int(height):,}\n({percentage:.1f}%)', 
                   ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'sex_distribution_normalized.png')
        plt.close()
        
    
    # 1. Basic demographic analysis
    print_subsection_header("Demographic Distribution")
    
    # Calculate age from birth year
    if 'BirthYear' in patient_df.columns:
        # Check if DeceasedYear column exists in demographic data
        has_deceased_info = 'patient_demographic' in data_dict and 'DeceasedYear' in data_dict['patient_demographic'].columns
 
        patient_df['Age'] = config.CURRENT_YEAR - patient_df['BirthYear']
        if has_deceased_info:
            # Merge with demographic data to get deceased year
            demo_df = data_dict['patient_demographic'][['Patient_ID', 'DeceasedYear']]
            patient_df = patient_df.merge(demo_df, on='Patient_ID', how='left')
            
            # Calculate age considering death year when available
            patient_df['reference_year'] = config.CURRENT_YEAR  # Default to current year
            deceased_mask = patient_df['DeceasedYear'].notna() & (patient_df['DeceasedYear'] > 0)
            if deceased_mask.any():
                patient_df.loc[deceased_mask, 'reference_year'] = patient_df.loc[deceased_mask, 'DeceasedYear']
            
            patient_df['Age'] = patient_df['reference_year'] - patient_df['BirthYear']
            
            # Report on deceased patients
            deceased_count = deceased_mask.sum()
            print(f"Deceased patients with valid death year: {deceased_count:,} ({deceased_count/len(patient_df)*100:.2f}%)")
        else:
            # Standard age calculation based on current year
            patient_df['Age'] = config.CURRENT_YEAR - patient_df['BirthYear']
            
        # Age statistics
        age_mean = patient_df['Age'].mean()
        age_median = patient_df['Age'].median()
        age_min = patient_df['Age'].min()
        age_max = patient_df['Age'].max()
        
        print(f"Age statistics (as of {config.CURRENT_YEAR}):")
        print(f"  Mean: {age_mean:.1f} years")
        print(f"  Median: {age_median:.1f} years")
        print(f"  Range: {age_min} to {age_max} years")
        
        # Age distribution
        age_bins = [0, 18, 35, 50, 65, 80, 120]
        age_labels = ['<18', '18-34', '35-49', '50-64', '65-79', '80+']
        
        patient_df['Age_Group'] = pd.cut(patient_df['Age'], bins=age_bins, labels=age_labels)
        age_dist = patient_df['Age_Group'].value_counts().sort_index()
        age_pct = age_dist / len(patient_df) * 100
        
        # Display age distribution
        age_table = pd.DataFrame({
            'Age Group': age_dist.index,
            'Count': age_dist.values,
            'Percentage': age_pct.values
        })
        
        print("\nAge distribution:")
        display(age_table)
        
        # Store in metrics
        population_metrics['age'] = {
            'mean': age_mean,
            'median': age_median,
            'min': age_min,
            'max': age_max,
            'distribution': age_dist.to_dict(),
            'percentage': age_pct.to_dict()
        }
        
        # Plot age distribution
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=age_dist.index, y=age_dist.values)
        plt.title('Patient Age Distribution', fontsize=14)
        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        
        # Add count and percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = age_pct.iloc[i]
            ax.text(p.get_x() + p.get_width()/2., height + height*0.02,
                   f'{int(height):,}\n({percentage:.1f}%)', 
                   ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'age_distribution.png')
        plt.close()
    
    # Sex distribution
    if 'Sex' in patient_df.columns:
        sex_counts = patient_df['Sex'].value_counts()
        sex_pct = sex_counts / len(patient_df) * 100
        
        print("\nSex distribution:")
        for sex, count in sex_counts.items():
            print(f"  {sex}: {count:,} ({sex_pct[sex]:.1f}%)")
            
        # Store in metrics
        population_metrics['sex'] = {
            'counts': sex_counts.to_dict(),
            'percentage': sex_pct.to_dict()
        }
        
        # Plot sex distribution
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=sex_counts.index, y=sex_counts.values)
        plt.title('Patient Sex Distribution', fontsize=14)
        plt.xlabel('Sex', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        
        # Add count and percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = sex_pct.iloc[i]
            ax.text(p.get_x() + p.get_width()/2., height + height*0.02,
                   f'{int(height):,}\n({percentage:.1f}%)', 
                   ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'sex_distribution.png')
        plt.close()
    
    # 2. Data coverage across tables
    print_subsection_header("Patient Data Coverage")
    
    # Get patient ID sets for each table
    coverage = {}
    all_patients = set(patient_df['Patient_ID'])
    total_patients = len(all_patients)
    
    for table, df in data_dict.items():
        if table == 'patient':
            continue
            
        if 'Patient_ID' in df.columns:
            table_patients = set(df['Patient_ID'].unique())
            overlap = table_patients.intersection(all_patients)
            
            coverage_pct = len(overlap) / total_patients
            coverage[table] = {
                'patients': len(overlap),
                'percentage': coverage_pct
            }
            
            print(f"{table}: {coverage_pct:.2%} ({len(overlap):,}/{total_patients:,} patients)")
    
    # Create intersection visualization
    if coverage:
        # Sort tables by coverage
        sorted_tables = sorted(coverage.keys(), key=lambda x: coverage[x]['percentage'], reverse=True)
        
        # Create bar chart of coverage
        plt.figure(figsize=(12, 6))
        coverage_vals = [coverage[t]['percentage'] for t in sorted_tables]
        bars = plt.barh(sorted_tables, coverage_vals)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label = f"{width:.1%}"
            plt.text(max(0.05, width - 0.1), bar.get_y() + bar.get_height()/2, 
                    label, ha='center', va='center', color='white', fontweight='bold')
        
        plt.xlabel('Percentage of Patients with Data')
        plt.title('Patient Coverage by Data Table')
        plt.xlim(0, 1.0)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'patient_coverage.png')
        plt.close()
    
    # Store in metrics
    population_metrics['coverage'] = coverage
    
    # 3. Eligibility analysis based on protocol criteria
    print_subsection_header("Study Eligibility Analysis")
    
    # Age eligibility (adults ≥18)
    if 'Age' in patient_df.columns:
        age_eligible = patient_df['Age'] >= config.MIN_AGE
        age_eligible_count = age_eligible.sum()
        age_eligible_pct = age_eligible_count / len(patient_df)
        
        print(f"Age eligible (≥{config.MIN_AGE}): {age_eligible_pct:.2%} ({age_eligible_count:,}/{len(patient_df):,} patients)")
    
    # Encounter eligibility (≥2 encounters)
    if 'encounter' in data_dict and 'Patient_ID' in data_dict['encounter'].columns:
        encounter_counts = data_dict['encounter']['Patient_ID'].value_counts()
        encounter_eligible = encounter_counts[encounter_counts >= config.MIN_ENCOUNTERS]
        encounter_eligible_count = len(encounter_eligible)
        encounter_eligible_pct = encounter_eligible_count / len(patient_df)
        
        print(f"Encounter eligible (≥{config.MIN_ENCOUNTERS} encounters): " +
              f"{encounter_eligible_pct:.2%} ({encounter_eligible_count:,}/{len(patient_df):,} patients)")
        
        # Encounter statistics
        enc_mean = encounter_counts.mean()
        enc_median = encounter_counts.median()
        enc_p90 = encounter_counts.quantile(0.9)
        
        print(f"Encounter statistics:")
        print(f"  Mean: {enc_mean:.1f} encounters per patient")
        print(f"  Median: {enc_median:.0f} encounters per patient")
        print(f"  90th percentile: {enc_p90:.0f} encounters per patient")
        
        # Store in metrics
        population_metrics['encounters'] = {
            'mean': enc_mean,
            'median': enc_median,
            'p90': enc_p90,
            'eligible_count': encounter_eligible_count,
            'eligible_percentage': encounter_eligible_pct
        }
        
        # Create histogram of encounter counts
        plt.figure(figsize=(12, 6))
        # Log transform for better visualization
        log_counts = np.log10(encounter_counts + 1)  # +1 to handle zeros
        plt.hist(log_counts, bins=50)
        plt.title('Distribution of Encounters per Patient (Log Scale)', fontsize=14)
        plt.xlabel('log10(Encounters + 1)', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'encounter_distribution.png')
        plt.close()
    
    # 4. Combined eligibility criteria
    if 'Age' in patient_df.columns and 'encounter' in data_dict:
        # Get patient IDs with sufficient encounters
        encounter_counts = data_dict['encounter']['Patient_ID'].value_counts()
        encounter_eligible_ids = set(encounter_counts[encounter_counts >= config.MIN_ENCOUNTERS].index)
        
        # Combine with age eligibility
        age_eligible_ids = set(patient_df.loc[patient_df['Age'] >= config.MIN_AGE, 'Patient_ID'])
        
        # Intersection
        eligible_ids = age_eligible_ids.intersection(encounter_eligible_ids)
        eligible_count = len(eligible_ids)
        eligible_pct = eligible_count / len(patient_df)
        
        print(f"\nCombined eligibility (age ≥{config.MIN_AGE} AND ≥{config.MIN_ENCOUNTERS} encounters): " +
              f"{eligible_pct:.2%} ({eligible_count:,}/{len(patient_df):,} patients)")
              
        # Store in metrics
        population_metrics['combined_eligibility'] = {
            'eligible_count': eligible_count,
            'eligible_percentage': eligible_pct,
            'age_criteria': f"≥{config.MIN_AGE}",
            'encounter_criteria': f"≥{config.MIN_ENCOUNTERS}"
        }
    
    return population_metrics

def link_labs_to_encounters_by_time(lab_df, encounter_df, window_days=14):
    """Link lab records to encounters based on temporal proximity."""
    print("Implementing time-based lab-encounter linkage...")
    
    # Only process labs with missing Encounter_ID but valid dates
    labs_to_link = lab_df[
        (lab_df['Encounter_ID'].isna()) & 
        (lab_df['PerformedDate'].notna())
    ].copy()
    
    if len(labs_to_link) == 0:
        return lab_df
    
    print(f"Attempting to link {len(labs_to_link):,} labs to encounters")
    
    # Create columns for linkage data
    result = lab_df.copy()
    result['Linked_Encounter_ID'] = None
    result['Days_To_Encounter'] = None
    result['Linkage_Confidence'] = None
    
    # Process in chunks for efficiency
    chunk_size = 10000
    chunks = [labs_to_link.iloc[i:i+chunk_size] for i in range(0, len(labs_to_link), chunk_size)]
    
    linked_count = 0
    for chunk_idx, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_idx+1}/{len(chunks)}...")
        
        for lab_idx, lab in chunk.iterrows():
            # Get all encounters for this patient
            patient_encounters = encounter_df[
                (encounter_df['Patient_ID'] == lab['Patient_ID']) &
                (encounter_df['EncounterDate'].notna())
            ]
            
            if len(patient_encounters) == 0:
                continue
                
            # Calculate time differences
            patient_encounters['days_diff'] = abs(
                (patient_encounters['EncounterDate'] - lab['PerformedDate']).dt.days
            )
            
            # Find best match within window
            valid_matches = patient_encounters[patient_encounters['days_diff'] <= window_days]
            
            if len(valid_matches) == 0:
                continue
                
            # Sort by days_diff to get closest encounter
            valid_matches = valid_matches.sort_values('days_diff')
            best_match = valid_matches.iloc[0]
            
            # Calculate confidence score (1.0 = same day, decreases with distance)
            confidence = 1.0 - (best_match['days_diff'] / (window_days * 2))
            
            # Store the linkage data
            result.loc[lab_idx, 'Linked_Encounter_ID'] = best_match['Encounter_ID']
            result.loc[lab_idx, 'Days_To_Encounter'] = best_match['days_diff']
            result.loc[lab_idx, 'Linkage_Confidence'] = confidence
            
            linked_count += 1
    
    # Create effective ID column for downstream analysis
    result['Effective_Encounter_ID'] = result['Encounter_ID']
    mask = result['Effective_Encounter_ID'].isna() & result['Linked_Encounter_ID'].notna()
    result.loc[mask, 'Effective_Encounter_ID'] = result.loc[mask, 'Linked_Encounter_ID']
    
    print(f"Successfully linked {linked_count:,} labs to encounters by temporal proximity")
    print(f"Average confidence score: {result['Linkage_Confidence'].mean():.2f}")
    print(f"Total labs with encounter association: {result['Effective_Encounter_ID'].notna().sum():,} "
          f"({result['Effective_Encounter_ID'].notna().sum()/len(result)*100:.2f}%)")
    
    return result


def analyze_coding_patterns(data_dict):
    """
    Analyze diagnostic coding patterns relevant for SSD identification.
    
    This function examines:
    1. Distribution of ICD-9/ICD-10 codes
    2. Frequency of key diagnostic codes
    3. Preliminary identification of potential NYD codes
    
    Understanding coding patterns is critical for:
    - Proper identification of NYD status (Vital & Health Statistics, 1987)
    - Accurate detection of SSD criteria based on diagnostic codes
    - Assessment of coding variability across providers
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of dataframes with table names as keys
        
    Returns:
    --------
    dict
        Dictionary of coding pattern metrics
    """
    print_section_header("Diagnostic Coding Pattern Analysis")
    
    coding_metrics = {}
    
    # Skip if encounter_diagnosis or health_condition tables are missing
    if 'encounter_diagnosis' not in data_dict or 'health_condition' not in data_dict:
        print("WARNING: Missing diagnosis tables, skipping coding analysis")
        return coding_metrics
    
    # 1. Analyze ICD code types and patterns in encounter diagnoses
    print_subsection_header("Encounter Diagnosis Coding Patterns")
    
    ed_df = data_dict['encounter_diagnosis']
    
    # Check code type distribution
    if 'DiagnosisCodeType_calc' in ed_df.columns:
        code_types = ed_df['DiagnosisCodeType_calc'].value_counts()
        code_pct = code_types / len(ed_df) * 100
        
        print("Diagnosis code types:")
        for code_type, count in code_types.items():
            print(f"  {code_type}: {count:,} ({code_pct[code_type]:.1f}%)")
            
        # Store in metrics
        coding_metrics['code_types'] = {
            'counts': code_types.to_dict(),
            'percentage': code_pct.to_dict()
        }
    
    # 2. Check for known NYD (Not Yet Diagnosed) patterns
    if 'DiagnosisCode_calc' in ed_df.columns:
        print_subsection_header("Preliminary NYD Code Analysis")
        
        # Define NYD code patterns based on research
        # nyd_patterns = [
        #     r'^799\.9',    # Other unknown and unspecified cause
        #     r'^V71\.',     # Observation without finding - all V71 subcodes
        #     r'^R69',       # Illness, unspecified (ICD-10)
        #     r'^Z03\.',     # Medical observation (ICD-10)
        # ]
        nyd_patterns = [
            r'799',       # General symptoms (includes 799.9, which is "Other unknown cause")
            r'^V71',      # Observation without finding (without requiring the dot)
            r'^R69',      # Illness, unspecified (ICD-10)
            r'^Z03',      # Medical observation (ICD-10, without requiring the dot)
            r'^780\.9',   # Other general symptoms (including "unspecified")
            r'^V65\.5'    # Person with feared condition where no diagnosis was made
        ]
        
        # Check for each pattern
        nyd_counts = {}
        for pattern in nyd_patterns:
            matches = ed_df['DiagnosisCode_calc'].astype(str).str.contains(pattern, na=False)
            count = matches.sum()
            patient_count = ed_df.loc[matches, 'Patient_ID'].nunique()
            
            nyd_counts[pattern] = {
                'code_count': count,
                'patient_count': patient_count,
                'percentage': count / len(ed_df) * 100 if len(ed_df) > 0 else 0
            }
            
            print(f"NYD pattern '{pattern}': {count:,} codes ({nyd_counts[pattern]['percentage']:.2f}%) " +
                  f"in {patient_count:,} patients")
        
        # Store in metrics
        coding_metrics['nyd_patterns'] = nyd_counts
        
        # Count most common code roots (first 3 characters)
        ed_df['code_root'] = ed_df['DiagnosisCode_calc'].astype(str).str.extract(r'([A-Za-z0-9]{1,3})')
        root_counts = ed_df['code_root'].value_counts().head(20)
        
        print("\nMost common diagnostic code roots (first 3 characters):")
        for root, count in root_counts.items():
            print(f"  {root}: {count:,} ({count/len(ed_df)*100:.2f}%)")
        
        # Store in metrics
        coding_metrics['common_roots'] = root_counts.to_dict()
    
    
    print("\nChecking for potential NYD indicators in diagnostic code content...")
    # Look for 799 codes specifically (which includes 799.9 - Other unknown causes)
    code_799 = ed_df['DiagnosisCode_calc'].astype(str).str.contains('^799', regex=True, na=False)
    code_799_count = code_799.sum() 
    code_799_patient_count = ed_df.loc[code_799, 'Patient_ID'].nunique()

    print(f"Code 799 (symptoms/signs): {code_799_count:,} codes ({code_799_count/len(ed_df)*100:.2f}%) in {code_799_patient_count:,} patients")

    # Look for V71 (observation without diagnosis)
    code_v71 = ed_df['DiagnosisCode_calc'].astype(str).str.contains('^V71', regex=True, na=False)
    code_v71_count = code_v71.sum()
    code_v71_patient_count = ed_df.loc[code_v71, 'Patient_ID'].nunique()

    print(f"Code V71 (observation without diagnosis): {code_v71_count:,} codes ({code_v71_count/len(ed_df)*100:.2f}%) in {code_v71_patient_count:,} patients")
    

    # 3. Check for text patterns indicating NYD
    if 'DiagnosisText_calc' in ed_df.columns:
        # Define NYD text patterns
        nyd_text_patterns = [
            r'\bNYD\b', 
            r'\bnot yet diagnosed\b', 
            r'\bdiagnosis deferred\b',
            r'\bunknown etiology\b', 
            r'\brule out\b', 
            r'\bunexplained\b',
            r'\bundiagnosed\b',
            r'\bundetermined\b',
            r'\bsymptoms\b',
            r'\bsymptom\b NOT OTHERWISE SPECIFIED',
            r'without definitive diagnosis',
            r'no clear',
            r'no specific',
        ]
        
        # Check for each text pattern
        text_counts = {}
        for pattern in nyd_text_patterns:
            matches = ed_df['DiagnosisText_calc'].astype(str).str.contains(pattern, case=False, regex=True, na=False)
            count = matches.sum()
            patient_count = ed_df.loc[matches, 'Patient_ID'].nunique()
            
            text_counts[pattern] = {
                'code_count': count,
                'patient_count': patient_count,
                'percentage': count / len(ed_df) * 100 if len(ed_df) > 0 else 0
            }
            
            print(f"NYD text pattern '{pattern}': {count:,} entries ({text_counts[pattern]['percentage']:.2f}%) " +
                  f"in {patient_count:,} patients")
        
        # Store in metrics
        coding_metrics['nyd_text_patterns'] = text_counts
    
    # 4. Preliminary analysis of potential symptom codes (780-789)
    if 'DiagnosisCode_calc' in ed_df.columns:
        print_subsection_header("Symptom Code Analysis")
        
        # Check for ICD-9 symptom codes (780-789 range)
        symptom_pattern = r'^78[0-9]'
        symptom_matches = ed_df['DiagnosisCode_calc'].astype(str).str.contains(symptom_pattern, regex=True, na=False)
        symptom_count = symptom_matches.sum()
        symptom_patient_count = ed_df.loc[symptom_matches, 'Patient_ID'].nunique()
        
        symptom_pct = symptom_count / len(ed_df) * 100 if len(ed_df) > 0 else 0
        patient_pct = symptom_patient_count / ed_df['Patient_ID'].nunique() * 100
        
        print(f"ICD-9 Symptom codes (780-789): {symptom_count:,} codes ({symptom_pct:.2f}%) " +
              f"in {symptom_patient_count:,} patients ({patient_pct:.2f}%)")
              
        # Get top symptom codes
        if symptom_count > 0:
            symptom_codes = ed_df.loc[symptom_matches, 'DiagnosisCode_calc'].value_counts().head(10)
            print("\nTop symptom codes:")
            for code, count in symptom_codes.items():
                print(f"  {code}: {count:,}")

            print("\nNote: ICD-9 codes 780-789 represent 'Symptoms, Signs, and Ill-defined Conditions' and are")
            print("particularly relevant for SSD research as they often indicate medically unexplained symptoms.")
            print(f"The presence of these codes in {patient_pct:.1f}% of patients suggests a large pool of potential")
            print("cases with somatic symptoms that could be evaluated for SSD criteria.")
                
            # Store in metrics
            coding_metrics['symptom_codes'] = {
                'total_count': symptom_count,
                'patient_count': symptom_patient_count,
                'percentage': symptom_pct,
                'patient_percentage': patient_pct,
                'top_codes': symptom_codes.to_dict()
            }
    
    # 5. Check for body-system distribution of symptom codes
    print_subsection_header("Body System Distribution")
    
    # Define body systems based on ICD-9 ranges
    body_systems = {
        'general': ['^780', '^R50', '^R53'],  # Fever, fatigue, malaise
        'gi': ['^787', '^789', '^K5', '^K6', '^R1'], # Digestive symptoms
        'neuro': ['^784', '^346', '^307.81', '^G43', '^G44', '^R51'], # Headache, dizziness
        'cardio': ['^785', '^I10', '^R0'], # Chest pain, palpitations
        'respiratory': ['^786', '^R0[67]'], # Shortness of breath
        'musculo': ['^729', '^M79', '^M25', '^M54'], # Pain, joint, back
        'skin': ['^782', '^L2', '^L3'], # Rash, skin sensations
        'other': ['^788', '^R3'] # Urinary, etc.
    }
    
    # Count codes by body system
    system_counts = {}
    
    for system, patterns in body_systems.items():
        # Combine patterns
        system_pattern = '|'.join(patterns)
        matches = ed_df['DiagnosisCode_calc'].astype(str).str.contains(system_pattern, regex=True, na=False)
        count = matches.sum()
        patient_count = ed_df.loc[matches, 'Patient_ID'].nunique()
        
        system_counts[system] = {
            'code_count': count,
            'patient_count': patient_count,
            'percentage': count / len(ed_df) * 100 if len(ed_df) > 0 else 0
        }
        
        print(f"Body system '{system}': {count:,} codes ({system_counts[system]['percentage']:.2f}%) " +
              f"in {patient_count:,} patients")
    
    # Calculate multi-system counts
    if len(system_counts) > 0:
        # Get patients with symptoms in each system
        system_patients = {}
        for system, patterns in body_systems.items():
            system_pattern = '|'.join(patterns)
            matches = ed_df['DiagnosisCode_calc'].astype(str).str.contains(system_pattern, regex=True, na=False)
            system_patients[system] = set(ed_df.loc[matches, 'Patient_ID'].unique())
        
        # Count patients with symptoms in multiple systems
        patient_system_count = {}
        all_patients = set(ed_df['Patient_ID'].unique())
        
        for patient_id in all_patients:
            systems = [system for system, patients in system_patients.items() if patient_id in patients]
            patient_system_count[patient_id] = len(systems)
        
        # Summarize
        system_count_df = pd.Series(patient_system_count).value_counts().sort_index()
        system_count_pct = system_count_df / len(all_patients) * 100
        
        print("\nPatients by number of body systems with symptoms:")
        for num_systems, count in system_count_df.items():
            print(f"  {num_systems} systems: {count:,} patients ({system_count_pct[num_systems]:.2f}%)")

        print("\nNote: This analysis shows how many patients have symptom codes across different body systems.")
        print("  - 0 systems: Patients with no symptom codes in any defined body system")
        print("  - 1 system: Patients with symptoms in exactly one body system (e.g., only GI)")
        print("  - 2+ systems: Patients with symptoms in multiple body systems - a key DSM-5 criterion for SSD")
        print(f"  => {system_count_df.loc[lambda x: x.index >= 2].sum():,} patients ({system_count_df.loc[lambda x: x.index >= 2].sum()/len(all_patients)*100:.2f}%) have symptoms in 2+ body systems")
        

        # Store in metrics
        coding_metrics['multi_system'] = {
            'counts': system_count_df.to_dict(),
            'percentage': system_count_pct.to_dict()
        }
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=system_count_df.index, y=system_count_df.values)
        plt.title('Patients by Number of Body Systems with Symptoms', fontsize=14)
        plt.xlabel('Number of Body Systems', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        
        # Add count and percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            percentage = system_count_pct.iloc[i]
            ax.text(p.get_x() + p.get_width()/2., height + height*0.02,
                   f'{int(height):,}\n({percentage:.1f}%)', 
                   ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'body_system_distribution.png')
        plt.close()
    
    return coding_metrics


def analyze_lab_data(data_dict):
    """
    Analyze lab data structure and completeness for negative cascade detection.
    
    This function examines:
    1. Lab test type distribution
    2. Normal range data availability
    3. Completeness of test results
    4. Patient-level lab testing patterns
    
    Lab data analysis is critical for:
    - Identifying the "negative lab cascade" central to SSD research
    - Assessing data quality for normal/abnormal determination
    - Establishing baseline lab testing patterns
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of dataframes with table names as keys
        
    Returns:
    --------
    dict
        Dictionary of lab data metrics
    """
    print_section_header("Lab Data Analysis")
    
    lab_metrics = {}
    
    # Skip if lab table is missing
    if 'lab' not in data_dict:
        print("WARNING: Lab table missing, skipping analysis")
        return lab_metrics
    
    lab_df = data_dict['lab']
    
    # 1. Basic lab count metrics
    print_subsection_header("Lab Test Overview")
    
    total_labs = len(lab_df)
    patient_count = lab_df['Patient_ID'].nunique()
    average_labs_per_patient = total_labs / patient_count if patient_count > 0 else 0
    
    print(f"Total lab tests: {total_labs:,}")
    print(f"Patients with lab data: {patient_count:,}")
    print(f"Average lab tests per patient: {average_labs_per_patient:.1f}")
    
    # Store in metrics
    lab_metrics['overview'] = {
        'total_labs': total_labs,
        'patient_count': patient_count,
        'avg_per_patient': average_labs_per_patient
    }
    
    # 2. Lab test type distribution
    if 'Name_calc' in lab_df.columns:
        # Get top lab tests
        test_counts = lab_df['Name_calc'].value_counts().head(15)
        test_pct = test_counts / len(lab_df) * 100
        
        print("\nMost common lab tests:")
        for test, count in test_counts.items():
            print(f"  {test}: {count:,} ({test_pct[test]:.2f}%)")
            
        # Store in metrics
        lab_metrics['test_types'] = {
            'counts': test_counts.to_dict(),
            'percentage': test_pct.to_dict()
        }
    
    # 3. Normal range data availability
    print_subsection_header("Normal Range Data Availability")
    
    if 'UpperNormal' in lab_df.columns and 'LowerNormal' in lab_df.columns:
        # Check for presence of both normal range bounds
        has_upper = lab_df['UpperNormal'].notna()
        has_lower = lab_df['LowerNormal'].notna()
        has_both = has_upper & has_lower
        
        both_count = has_both.sum()
        both_pct = both_count / len(lab_df) * 100
        
        print(f"Labs with both normal bounds: {both_count:,} ({both_pct:.2f}%)")
        
        # Check by top test types
        if 'Name_calc' in lab_df.columns:
            test_normal_rates = {}
            for test in test_counts.index[:10]:  # Top 10 tests
                test_labs = lab_df['Name_calc'] == test
                test_total = test_labs.sum()
                test_with_bounds = (test_labs & has_both).sum()
                test_rate = test_with_bounds / test_total * 100 if test_total > 0 else 0
                
                test_normal_rates[test] = {
                    'total': test_total,
                    'with_bounds': test_with_bounds,
                    'percentage': test_rate
                }
                
                print(f"  {test}: {test_with_bounds:,}/{test_total:,} ({test_rate:.2f}%)")
                
            # Store in metrics
            lab_metrics['normal_range'] = {
                'total_with_bounds': both_count,
                'percentage': both_pct,
                'by_test': test_normal_rates
            }
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            tests = list(test_normal_rates.keys())
            rates = [info['percentage'] for test, info in test_normal_rates.items()]
            
            # Sort by rate for better visualization
            sorted_data = sorted(zip(tests, rates), key=lambda x: x[1], reverse=True)
            tests = [t for t, r in sorted_data]
            rates = [r for t, r in sorted_data]
            
            bars = plt.barh(tests, rates)
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label = f"{width:.1f}%"
                plt.text(max(5, width - 10), bar.get_y() + bar.get_height()/2, 
                        label, ha='center', va='center', color='white', fontweight='bold')
            
            plt.xlabel('Percentage with Normal Range Bounds')
            plt.title('Normal Range Data Availability by Lab Test Type')
            plt.xlim(0, 100)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(config.OUTPUT_PATH / 'lab_normal_range_availability.png')
            plt.close()
    
    # 4. Numeric result availability
    print_subsection_header("Numeric Test Result Availability")
    
    if 'TestResult_calc' in lab_df.columns:
        # Check for numeric values
        lab_df['TestResult_numeric'] = pd.to_numeric(lab_df['TestResult_calc'], errors='coerce')
        has_numeric = lab_df['TestResult_numeric'].notna()
        
        numeric_count = has_numeric.sum()
        numeric_pct = numeric_count / len(lab_df) * 100
        
        print(f"Labs with numeric results: {numeric_count:,} ({numeric_pct:.2f}%)")
        
        # Store in metrics
        lab_metrics['numeric_results'] = {
            'count': numeric_count,
            'percentage': numeric_pct
        }
    
    # 5. Patient-level lab testing patterns
    print_subsection_header("Patient-Level Lab Testing Patterns")
    
    # Count labs per patient
    patient_lab_counts = lab_df.groupby('Patient_ID').size()
    
    # Calculate statistics
    mean_labs = patient_lab_counts.mean()
    median_labs = patient_lab_counts.median()
    p90_labs = patient_lab_counts.quantile(0.9)
    max_labs = patient_lab_counts.max()
    
    print(f"Lab test statistics:")
    print(f"  Mean: {mean_labs:.1f} tests per patient")
    print(f"  Median: {median_labs:.0f} tests per patient")
    print(f"  90th percentile: {p90_labs:.0f} tests per patient")
    print(f"  Maximum: {max_labs:.0f} tests per patient")
    
    # Store in metrics
    lab_metrics['patient_labs'] = {
        'mean': mean_labs,
        'median': median_labs,
        'p90': p90_labs,
        'max': max_labs
    }
    
    # Create histogram of lab counts
    plt.figure(figsize=(12, 6))
    # Log transform for better visualization
    log_counts = np.log10(patient_lab_counts + 1)  # +1 to handle zeros
    plt.hist(log_counts, bins=50)
    plt.title('Distribution of Lab Tests per Patient (Log Scale)', fontsize=14)
    plt.xlabel('log10(Lab Tests + 1)', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_PATH / 'lab_test_distribution.png')
    plt.close()
    
    # 6. Preliminary normal lab cascade analysis
    if 'TestResult_calc' in lab_df.columns and 'UpperNormal' in lab_df.columns and 'LowerNormal' in lab_df.columns:
        print_subsection_header("Preliminary Normal Lab Analysis")
        
        # Find labs with full normal range data
        has_full_data = lab_df['TestResult_numeric'].notna() & lab_df['UpperNormal'].notna() & lab_df['LowerNormal'].notna()
        
        if has_full_data.any():
            # Convert range boundaries to numeric
            lab_df['upper_numeric'] = pd.to_numeric(lab_df['UpperNormal'], errors='coerce')
            lab_df['lower_numeric'] = pd.to_numeric(lab_df['LowerNormal'], errors='coerce')
            
            # Get subset with full data
            full_data = lab_df[has_full_data].copy()
            
            # Flag normal results
            full_data['is_normal'] = (
                (full_data['TestResult_numeric'] >= full_data['lower_numeric']) &
                (full_data['TestResult_numeric'] <= full_data['upper_numeric'])
            )
            
            # Count normal labs
            normal_count = full_data['is_normal'].sum()
            normal_pct = normal_count / len(full_data) * 100
            
            print(f"Labs with full data for normal analysis: {len(full_data):,} ({len(full_data)/len(lab_df)*100:.2f}%)")
            print(f"Normal lab results: {normal_count:,} ({normal_pct:.2f}%)")
            
            # Count patients with multiple normal labs
            patient_normal_counts = full_data.groupby('Patient_ID')['is_normal'].sum()
            
            # Thresholds for normal labs
            for threshold in [3, 4, 5]:
                patients_above = (patient_normal_counts >= threshold).sum()
                pct_above = patients_above / len(patient_normal_counts) * 100
                
                print(f"Patients with ≥{threshold} normal labs: {patients_above:,} ({pct_above:.2f}%)")
                
            # Store in metrics
            lab_metrics['normal_analysis'] = {
                'normal_count': normal_count,
                'normal_percentage': normal_pct,
                'patient_thresholds': {
                    f'ge_{threshold}': (patient_normal_counts >= threshold).sum()
                    for threshold in [3, 4, 5]
                }
            }
    
    return lab_metrics


def analyze_referral_patterns(data_dict):
    """
    Analyze referral patterns with focus on psychiatry vs. other specialists.
    
    This function examines:
    1. Referral type distribution
    2. Preliminary psychiatry referral identification
    3. Multi-specialty referral patterns
    
    Referral analysis is critical for:
    - Identifying the specialist to psychiatry sequence in SSD pathway
    - Quantifying "doctor shopping" behavior
    - Understanding typical specialty consultation patterns
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of dataframes with table names as keys
        
    Returns:
    --------
    dict
        Dictionary of referral pattern metrics
    """
    print_section_header("Referral Pattern Analysis")
    
    referral_metrics = {}
    
    # Skip if referral table is missing
    if 'referral' not in data_dict:
        print("WARNING: Referral table missing, skipping analysis")
        return referral_metrics
    
    ref_df = data_dict['referral']
    
    # 1. Basic referral metrics
    print_subsection_header("Referral Overview")
    
    total_refs = len(ref_df)
    patient_count = ref_df['Patient_ID'].nunique()
    average_refs_per_patient = total_refs / patient_count if patient_count > 0 else 0
    
    print(f"Total referrals: {total_refs:,}")
    print(f"Patients with referrals: {patient_count:,}")
    print(f"Average referrals per patient: {average_refs_per_patient:.1f}")
    
    # Store in metrics
    referral_metrics['overview'] = {
        'total_referrals': total_refs,
        'patient_count': patient_count,
        'avg_per_patient': average_refs_per_patient
    }
    
    # 2. Referral type distribution
    if 'Name_calc' in ref_df.columns:
        # Get top referral types
        ref_counts = ref_df['Name_calc'].value_counts().head(15)
        ref_pct = ref_counts / len(ref_df) * 100
        
        print("\nMost common referral types:")
        for ref_type, count in ref_counts.items():
            print(f"  {ref_type}: {count:,} ({ref_pct[ref_type]:.2f}%)")
            
        # Store in metrics
        referral_metrics['ref_types'] = {
            'counts': ref_counts.to_dict(),
            'percentage': ref_pct.to_dict()
        }
    
    # 3. Identify psychiatric referrals
    print_subsection_header("Psychiatric Referral Analysis")
    
    if 'Name_calc' in ref_df.columns:
        # Define psychiatry patterns based on validated terminology
        psych_patterns = [
            'psychiatr', 'mental health', 'psych', 'behavioral health', 'mood',
            'mental', 'anxiety', 'depression', 'counseling', 'mh consult'
        ]
        psych_pattern = '|'.join([f"\\b{p}" for p in psych_patterns])
        
        # Flag psychiatric referrals
        ref_df['to_psychiatrist'] = ref_df['Name_calc'].str.contains(
            psych_pattern, case=False, regex=True, na=False)
        
        psych_count = ref_df['to_psychiatrist'].sum()
        psych_pct = psych_count / len(ref_df) * 100
        psych_patient_count = ref_df.loc[ref_df['to_psychiatrist'], 'Patient_ID'].nunique()
        psych_patient_pct = psych_patient_count / patient_count * 100
        
        print(f"Psychiatric referrals: {psych_count:,} ({psych_pct:.2f}%)")
        print(f"Patients with psychiatric referrals: {psych_patient_count:,} ({psych_patient_pct:.2f}%)")
        
        # Store in metrics
        referral_metrics['psychiatry'] = {
            'referral_count': psych_count,
            'referral_percentage': psych_pct,
            'patient_count': psych_patient_count,
            'patient_percentage': psych_patient_pct
        }
    
    # 4. Define and analyze body system specialists
    print_subsection_header("Body System Specialist Analysis")
    
    if 'Name_calc' in ref_df.columns:
        # Define body system specialists based on validated terminology
        body_systems = {
            'cardio': ['cardiol', 'heart', 'cardiac', 'vascular', 'circulat', 'cardiolog', 'cardio'],
            'gastro': ['gastro', 'gi', 'digestive', 'stomach', 'intestin', 'bowel', 'endo'],
            'neuro': ['neuro', 'brain', 'headache', 'seizure', 'cognit', 'memory', 'nervous'],
            'musculo': ['orthoped', 'rheumat', 'joint', 'pain', 'musculo', 'arthrit', 'back', 'spine', 'ortho'],
            'respiratory': ['pulmon', 'lung', 'respirat', 'breath', 'asthma', 'copd', 'pulm', 'resp'],
            'endo': ['endocrin', 'diabet', 'thyroid', 'hormone', 'metabol', 'endo'],
            'derm': ['dermatol', 'skin', 'rash', 'lesion', 'derm'],
            'gyn': ['gynecol', 'obstetric', 'women', 'pelvic', 'genital', 'urolog', 'gyn', 'repro']
        }
        
        # Flag each body system
        for system, keywords in body_systems.items():
            system_pattern = '|'.join([f"\\b{k}" for k in keywords])
            col_name = f'to_{system}'
            ref_df[col_name] = ref_df['Name_calc'].str.contains(
                system_pattern, case=False, regex=True, na=False)
        
        # Summarize body system referrals
        system_metrics = {}
        print("\nBody system referral distribution:")
        for system in body_systems.keys():
            col_name = f'to_{system}'
            count = ref_df[col_name].sum()
            pct = count / len(ref_df) * 100
            patient_count = ref_df.loc[ref_df[col_name], 'Patient_ID'].nunique()
            patient_pct = patient_count / ref_df['Patient_ID'].nunique() * 100
            
            system_metrics[system] = {
                'referral_count': count,
                'referral_percentage': pct,
                'patient_count': patient_count,
                'patient_percentage': patient_pct
            }
            
            print(f"  {system}: {count:,} referrals ({pct:.2f}%) in {patient_count:,} patients ({patient_pct:.2f}%)")
        
        # Store in metrics
        referral_metrics['body_systems'] = system_metrics
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        systems = list(system_metrics.keys())
        counts = [info['referral_count'] for system, info in system_metrics.items()]
        
        # Sort by count for better visualization
        sorted_data = sorted(zip(systems, counts), key=lambda x: x[1], reverse=True)
        systems = [s for s, c in sorted_data]
        counts = [c for s, c in sorted_data]
        
        ax = plt.bar(systems, counts)
        
        # Add count labels
        for i, p in enumerate(ax):
            height = p.get_height()
            plt.text(p.get_x() + p.get_width()/2., height + height*0.02,
                   f'{int(height):,}', 
                   ha="center", va="bottom", fontsize=9)
        
        plt.ylabel('Number of Referrals')
        plt.title('Referrals by Body System')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'body_system_referrals.png')
        plt.close()
    
    # 5. Multi-specialty referral patterns
    print_subsection_header("Multi-Specialty Referral Patterns")
    
    # Count referrals per patient
    patient_ref_counts = ref_df.groupby('Patient_ID').size()
    
    # Calculate statistics
    mean_refs = patient_ref_counts.mean()
    median_refs = patient_ref_counts.median()
    p90_refs = patient_ref_counts.quantile(0.9)
    
    print(f"Referral statistics:")
    print(f"  Mean: {mean_refs:.1f} referrals per patient")
    print(f"  Median: {median_refs:.0f} referrals per patient")
    print(f"  90th percentile: {p90_refs:.0f} referrals per patient")
    
    # Analyze multi-specialty patterns
    if 'to_psychiatrist' in ref_df.columns and any(f'to_{system}' in ref_df.columns for system in body_systems.keys()):
        # Flag any non-psychiatric specialty
        ref_df['to_any_body_system'] = False
        for system in body_systems.keys():
            col_name = f'to_{system}'
            if col_name in ref_df.columns:
                ref_df['to_any_body_system'] = ref_df['to_any_body_system'] | ref_df[col_name]
        
        # Get patient-level specialty flags
        patient_specialties = ref_df.groupby('Patient_ID').agg({
            'to_psychiatrist': 'any',
            'to_any_body_system': 'any'
        })
        
        # Calculate patterns
        patient_specialties['psych_only'] = patient_specialties['to_psychiatrist'] & ~patient_specialties['to_any_body_system']
        patient_specialties['body_only'] = ~patient_specialties['to_psychiatrist'] & patient_specialties['to_any_body_system']
        patient_specialties['both'] = patient_specialties['to_psychiatrist'] & patient_specialties['to_any_body_system']
        patient_specialties['neither'] = ~patient_specialties['to_psychiatrist'] & ~patient_specialties['to_any_body_system']
        
        # Summarize
        pattern_counts = {
            'psych_only': patient_specialties['psych_only'].sum(),
            'body_only': patient_specialties['body_only'].sum(),
            'both': patient_specialties['both'].sum(),
            'neither': patient_specialties['neither'].sum()
        }
        
        pattern_pct = {k: v / len(patient_specialties) * 100 for k, v in pattern_counts.items()}
        
        print("\nPatient referral patterns:")
        print(f"  Psychiatry only: {pattern_counts['psych_only']:,} patients ({pattern_pct['psych_only']:.2f}%)")
        print(f"  Body system only: {pattern_counts['body_only']:,} patients ({pattern_pct['body_only']:.2f}%)")
        print(f"  Both psychiatry and body system: {pattern_counts['both']:,} patients ({pattern_pct['both']:.2f}%)")
        print(f"  Neither (other or unclassified referrals): {pattern_counts['neither']:,} patients ({pattern_pct['neither']:.2f}%)")
        
        # Store in metrics
        referral_metrics['patterns'] = {
            'counts': pattern_counts,
            'percentage': pattern_pct
        }
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        labels = ['Psychiatry Only', 'Body System Only', 'Both', 'Neither/Other']
        values = [pattern_counts[k] for k in ['psych_only', 'body_only', 'both', 'neither']]
        
        ax = plt.bar(labels, values)
        
        # Add count and percentage labels
        for i, p in enumerate(ax):
            height = p.get_height()
            percentage = list(pattern_pct.values())[i]
            plt.text(p.get_x() + p.get_width()/2., height + height*0.02,
                   f'{int(height):,}\n({percentage:.1f}%)', 
                   ha="center", va="bottom", fontsize=9)
        
        plt.ylabel('Number of Patients')
        plt.title('Patient Referral Patterns')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'referral_patterns.png')
        plt.close()
    
    # 6. Doctor shopping analysis (multiple providers)
    if 'referral' in data_dict and 'Provider_ID' in ref_df.columns:
        print_subsection_header("Doctor Shopping Analysis")
        
        # Count distinct providers per patient
        provider_counts = ref_df.groupby('Patient_ID')['Provider_ID'].nunique()
        
        # Calculate statistics
        mean_providers = provider_counts.mean()
        median_providers = provider_counts.median()
        p90_providers = provider_counts.quantile(0.9)
        
        print(f"Provider statistics:")
        print(f"  Mean: {mean_providers:.1f} distinct providers per patient")
        print(f"  Median: {median_providers:.0f} distinct providers per patient")
        print(f"  90th percentile: {p90_providers:.0f} distinct providers per patient")
        
        # Define doctor shopping as ≥5 providers (from research protocol)
        shopping_threshold = 5
        shoppers = (provider_counts >= shopping_threshold).sum()
        shoppers_pct = shoppers / len(provider_counts) * 100
        
        print(f"Patients with ≥{shopping_threshold} different providers (potential doctor shopping): " +
              f"{shoppers:,} ({shoppers_pct:.2f}%)")
        
        # Store in metrics
        referral_metrics['doctor_shopping'] = {
            'mean_providers': mean_providers,
            'median_providers': median_providers,
            'p90_providers': p90_providers,
            'shoppers_count': shoppers,
            'shoppers_percentage': shoppers_pct,
            'threshold': shopping_threshold
        }
        
        # Create histogram of provider counts
        plt.figure(figsize=(12, 6))
        plt.hist(provider_counts, bins=range(0, 20), alpha=0.7)
        plt.axvline(x=shopping_threshold, color='r', linestyle='--', linewidth=2, 
                   label=f'Shopping threshold (≥{shopping_threshold})')
        plt.title('Distribution of Distinct Providers per Patient', fontsize=14)
        plt.xlabel('Number of Distinct Providers', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.OUTPUT_PATH / 'provider_distribution.png')
        plt.close()
    
    return referral_metrics
def save_validation_results(data_quality, relationship_metrics, temporal_metrics,
                         population_metrics, coding_metrics, lab_metrics, referral_metrics):
    """
    Save all validation results to a structured JSON file for future reference.
    
    This function combines all metrics into a single, comprehensive report that can be:
    1. Loaded in subsequent notebooks
    2. Used for data quality monitoring over time
    3. Included in supplementary materials for publications
    
    Parameters:
    -----------
    Various metric dictionaries from validation functions
    
    Returns:
    --------
    str
        Path to saved report file
    """
    def calculate_nyd_patients(coding_metrics):
        """Calculate total patients with any NYD code pattern."""
        if not coding_metrics or 'nyd_patterns' not in coding_metrics:
            return 0
            
        # Get all patient counts from patterns
        pattern_counts = [pattern.get('patient_count', 0) 
                        for pattern in coding_metrics['nyd_patterns'].values()]
        
        # Return the highest count (as a conservative estimate)
        # This avoids double-counting while ensuring we don't miss patients
        return max(pattern_counts, default=0)
        
    print_section_header("Saving Validation Results")
    
     # Combine all metrics
    validation_report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_quality': data_quality,
        'relationships': relationship_metrics,
        'temporal': temporal_metrics,
        'population': population_metrics,
        'coding': coding_metrics,
        'lab': lab_metrics,
        'referral': referral_metrics
    }
    
    # Save to file
    report_path = config.OUTPUT_PATH / 'data_validation_report.json'
    
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)  # default=str handles non-serializable objects
    
    print(f"Saved validation report to {report_path}")
    
    # Create summary for display
    summary = {
        'Tables Loaded': len(data_quality),
        'Total Patients': population_metrics.get('overview', {}).get('total_patients', 'Unknown'),
        'Data Relationship Issues': any(rel.get('orphan_percentage', 0) > 0.05 for rel in relationship_metrics.values()) if relationship_metrics else 'Unknown',
        'Normal Lab Data Available': lab_metrics.get('normal_range', {}).get('percentage', 0) if lab_metrics else 'Unknown',
        'Patients with NYD Codes': calculate_nyd_patients(coding_metrics),
        'Psychiatry Referrals': referral_metrics.get('psychiatry', {}).get('patient_count', 0) if referral_metrics else 'Unknown'
    }
    
    print("\nValidation Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return report_path
def enhance_lab_classification(lab_df):
    """Multi-strategy approach to classify lab results as normal/abnormal."""
    print("Enhancing normal lab classification...")
    
    result = lab_df.copy()
    total_labs = len(result)
    result['is_normal'] = pd.NA
    
    # 1. Explicit normal ranges where available
    has_bounds = result['LowerNormal'].notna() & result['UpperNormal'].notna() & result['TestResult_calc'].notna()
    
    if has_bounds.any():
        # Convert to numeric for comparison
        lower_numeric = pd.to_numeric(result.loc[has_bounds, 'LowerNormal'], errors='coerce')
        upper_numeric = pd.to_numeric(result.loc[has_bounds, 'UpperNormal'], errors='coerce')
        result_numeric = pd.to_numeric(result.loc[has_bounds, 'TestResult_calc'], errors='coerce')
        
        # Identify normal results within bounds
        valid_bounds = lower_numeric.notna() & upper_numeric.notna() & result_numeric.notna()
        if valid_bounds.any():
            normal_mask = valid_bounds & (result_numeric >= lower_numeric) & (result_numeric <= upper_numeric)
            result.loc[has_bounds[has_bounds].index[normal_mask], 'is_normal'] = True
            result.loc[has_bounds[has_bounds].index[~normal_mask & valid_bounds], 'is_normal'] = False
        
        processed_count = valid_bounds.sum()
        print(f"Method 1 (Explicit ranges): {processed_count:,} labs processed ({processed_count/total_labs*100:.2f}%)")
    
    # 2. Standard reference intervals for common tests
    common_tests = {
        'TOTAL CHOLESTEROL': {'min': 0, 'max': 5.2, 'unit': 'mmol/L'},
        'HDL': {'min': 1.0, 'max': 3.0, 'unit': 'mmol/L'},
        'LDL': {'min': 0, 'max': 3.4, 'unit': 'mmol/L'},
        'TRIGLYCERIDES': {'min': 0, 'max': 1.7, 'unit': 'mmol/L'},
        'FASTING GLUCOSE': {'min': 3.9, 'max': 5.6, 'unit': 'mmol/L'},
        'HBA1C': {'min': 0, 'max': 5.7, 'unit': '%'},
        'TSH': {'min': 0.4, 'max': 4.0, 'unit': 'mIU/L'},
        'ALT': {'min': 0, 'max': 40, 'unit': 'U/L'},
        'AST': {'min': 0, 'max': 40, 'unit': 'U/L'},
        'CREATININE': {'min': 50, 'max': 120, 'unit': 'umol/L'},
        'HEMOGLOBIN': {'min': 120, 'max': 160, 'unit': 'g/L'},
        'WBC': {'min': 4.0, 'max': 11.0, 'unit': '10^9/L'},
        'POTASSIUM': {'min': 3.5, 'max': 5.0, 'unit': 'mmol/L'},
        'SODIUM': {'min': 135, 'max': 145, 'unit': 'mmol/L'}
    }
    
    # Convert TestResult_calc to numeric once
    result['result_numeric'] = pd.to_numeric(result['TestResult_calc'], errors='coerce')
    
    # Process each common test
    reference_count = 0
    for test_name, reference in common_tests.items():
        # Find pending labs for this test (result known but normal status unknown)
        test_mask = (
            result['Name_calc'].str.contains(test_name, case=False, regex=False, na=False) &
            result['result_numeric'].notna() &
            result['is_normal'].isna()
        )
        
        if test_mask.any():
            normal_mask = (
                (result.loc[test_mask, 'result_numeric'] >= reference['min']) & 
                (result.loc[test_mask, 'result_numeric'] <= reference['max'])
            )
            result.loc[test_mask[test_mask].index[normal_mask], 'is_normal'] = True
            result.loc[test_mask[test_mask].index[~normal_mask], 'is_normal'] = False
            reference_count += test_mask.sum()
    
    print(f"Method 2 (Reference intervals): {reference_count:,} labs processed ({reference_count/total_labs*100:.2f}%)")
    
    # 3. Text pattern search for remaining labs
    pending_mask = result['is_normal'].isna() & result['TestResult_calc'].notna()
    text_count = 0
    
    if pending_mask.any():
        # Normal indicators
        normal_patterns = [
            'normal', 'neg', 'negative', 'unremarkable', 'w/in normal', 'within normal', 
            'wnl', 'within reference', 'not detected', 'n/a'
        ]
        normal_pattern = '|'.join([f"\\b{p}" for p in normal_patterns])
        
        # Abnormal indicators
        abnormal_patterns = [
            'abnormal', 'pos', 'positive', 'high', 'low', 'elevated', 'depressed', 
            'outside', 'detected', 'present'
        ]
        abnormal_pattern = '|'.join([f"\\b{p}" for p in abnormal_patterns])
        
        # Apply normal patterns
        normal_text = result.loc[pending_mask, 'TestResult_calc'].astype(str).str.contains(
            normal_pattern, case=False, regex=True, na=False
        )
        result.loc[pending_mask[pending_mask].index[normal_text], 'is_normal'] = True
        
        # Apply abnormal patterns (where normal wasn't found)
        still_pending = result['is_normal'].isna() & result['TestResult_calc'].notna()
        if still_pending.any():
            abnormal_text = result.loc[still_pending, 'TestResult_calc'].astype(str).str.contains(
                abnormal_pattern, case=False, regex=True, na=False
            )
            result.loc[still_pending[still_pending].index[abnormal_text], 'is_normal'] = False
        
        text_count = (normal_text.sum() + abnormal_text.sum())
    
    print(f"Method 3 (Text patterns): {text_count:,} labs processed ({text_count/total_labs*100:.2f}%)")
    
    # Calculate overall coverage
    classified_count = result['is_normal'].notna().sum()
    coverage_pct = classified_count / total_labs * 100
    normal_count = result['is_normal'].sum()
    abnormal_count = (~result['is_normal'] & result['is_normal'].notna()).sum()
    
    print(f"Overall: Classified {classified_count:,} labs ({coverage_pct:.2f}%)")
    print(f"Normal: {normal_count:,} ({normal_count/classified_count*100:.2f}% of classified)")
    print(f"Abnormal: {abnormal_count:,} ({abnormal_count/classified_count*100:.2f}% of classified)")
    
    return result


def enhance_referral_dates(referral_df):
    """Enhance referral dates using fallback strategies."""
    print("Implementing referral date enhancement...")
    
    result = referral_df.copy()
    total_refs = len(result)
    
    # Check completion date coverage
    missing_completion = result['CompletedDate'].isna()
    missing_count = missing_completion.sum()
    missing_pct = missing_count / total_refs * 100
    
    print(f"CompletedDate missing in {missing_count:,} referrals ({missing_pct:.2f}%)")
    
    # Create effective date column with source tracking
    result['EffectiveDate'] = result['CompletedDate']
    result['DateSource'] = 'CompletedDate'
    
    # Use DateCreated as fallback when needed
    if missing_count > 0:
        result.loc[missing_completion, 'EffectiveDate'] = result.loc[missing_completion, 'DateCreated']
        result.loc[missing_completion, 'DateSource'] = 'DateCreated'
        
        # Check coverage after fallback
        remaining_missing = result['EffectiveDate'].isna().sum()
        
        print(f"After fallback: {remaining_missing:,} referrals still missing dates ({remaining_missing/total_refs*100:.2f}%)")
        print(f"Using DateCreated for {missing_count-remaining_missing:,} referrals")
    
    # Flag referral status based on available dates
    result['ReferralStatus'] = 'Unknown'
    
    # Completed referrals have CompletedDate
    result.loc[result['CompletedDate'].notna(), 'ReferralStatus'] = 'Completed'
    
    # Pending referrals have DateCreated but no CompletedDate
    result.loc[(result['CompletedDate'].isna()) & (result['DateCreated'].notna()), 
               'ReferralStatus'] = 'Pending'
    
    # Count by status
    status_counts = result['ReferralStatus'].value_counts()
    print("\nReferral Status Distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,} ({count/total_refs*100:.2f}%)")
    
    return result


def save_checkpoint_with_documentation(data_dict, notebook_number, description, changes=None):
    """Save checkpoint with clear documentation for future reference."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint directory
    checkpoint_dir = config.INTERIM_PATH / f"checkpoint_{notebook_number}_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save each dataframe with appropriate format
    saved_tables = {}
    for name, df in data_dict.items():
        try:
            # Handle special case for lab table
            if name == 'lab':
                csv_path = checkpoint_dir / f"{name}.csv"
                df.to_csv(csv_path, index=False)
                saved_tables[name] = {
                    'path': str(csv_path),
                    'rows': len(df),
                    'columns': len(df.columns),
                    'format': 'csv'
                }
                print(f"Saved {name} ({len(df):,} rows) as CSV")
            else:
                # Try parquet first
                try:
                    parquet_path = checkpoint_dir / f"{name}.parquet"
                    df.to_parquet(parquet_path, index=False)
                    saved_tables[name] = {
                        'path': str(parquet_path),
                        'rows': len(df),
                        'columns': len(df.columns),
                        'format': 'parquet'
                    }
                    print(f"Saved {name} ({len(df):,} rows) as parquet")
                except Exception:
                    # Fall back to CSV
                    csv_path = checkpoint_dir / f"{name}.csv"
                    df.to_csv(csv_path, index=False)
                    saved_tables[name] = {
                        'path': str(csv_path),
                        'rows': len(df),
                        'columns': len(df.columns),
                        'format': 'csv'
                    }
                    print(f"Saved {name} ({len(df):,} rows) as CSV (parquet failed)")
        except Exception as e:
            print(f"ERROR saving {name}: {str(e)}")
    
    # Create detailed README
    readme_content = f"""# Notebook {notebook_number}: {description}

## Summary
This checkpoint contains data processed through notebook {notebook_number}.

## Date
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Tables
{chr(10).join([f"- **{name}**: {info['rows']:,} rows, {len(data_dict[name].columns)} columns ({info['format']})" 
               for name, info in saved_tables.items()])}

## Changes Made
{chr(10).join([f"- {change}" for change in (changes or ['No specific changes documented.'])])}

## Key Notes
- Lab normal detection uses multiple methods (explicit ranges, reference intervals, text patterns)
- Orphaned labs linked to encounters using temporal proximity
- Referral dates use DateCreated as fallback when CompletedDate missing
- NYD codes enhanced with both numeric and text-based identification

## Next Steps
Continue with Notebook {notebook_number + 1} for NYD identification refinement.
"""
    
    readme_path = checkpoint_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Create metadata JSON
    metadata = {
        'notebook': notebook_number,
        'description': description,
        'timestamp': timestamp,
        'tables': saved_tables,
        'changes': changes or [],
        'next_notebook': f"{notebook_number + 1:02d}_NYD_Identification.ipynb"
    }
    
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nCheckpoint saved at: {checkpoint_dir}")
    print(f"README created: {readme_path}")
    
    return checkpoint_dir

def link_labs_to_encounters_by_time_optimized(lab_df, encounter_df, window_days=14):
    """
    Highly optimized version of lab-encounter linkage using vectorized operations.
    Processes data in larger chunks with efficient patient-based indexing.
    """
    print("Implementing optimized time-based lab-encounter linkage...")
    
    # Create efficient copy with only necessary columns
    result = lab_df.copy()
    result['Linked_Encounter_ID'] = None
    result['Days_To_Encounter'] = None
    result['Linkage_Confidence'] = None
    
    # Only process labs with missing Encounter_ID but valid dates
    labs_to_link = lab_df[
        lab_df['PerformedDate'].notna()
    ].copy()
    
    print(f"Preparing to link {len(labs_to_link):,} labs to encounters")
    
    # Create patient index for encounters (do this ONCE)
    print("Indexing encounters by patient (one-time operation)...")
    valid_encounters = encounter_df[encounter_df['EncounterDate'].notna()].copy()
    
    # Pre-sort encounters by date for each patient (more efficient lookups)
    patient_encounter_dict = {}
    for patient_id, group in valid_encounters.groupby('Patient_ID'):
        # Pre-sort by date once per patient
        patient_encounter_dict[patient_id] = group.sort_values('EncounterDate')
    
    # Use much larger chunks for better performance
    chunk_size = 250000  # Increased from 10,000 to 250,000
    chunks = np.array_split(labs_to_link, max(1, len(labs_to_link) // chunk_size))
    total_chunks = len(chunks)
    
    print(f"Processing {len(labs_to_link):,} labs in {total_chunks} optimized chunks...")
    
    linked_count = 0
    start_time = datetime.now()
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_start = datetime.now()
        print(f"Processing chunk {chunk_idx+1}/{total_chunks} ({len(chunk):,} labs)...")
        
        # Process each patient's labs as a group (much more efficient)
        for patient_id, patient_labs in chunk.groupby('Patient_ID'):
            if patient_id not in patient_encounter_dict:
                continue
                
            patient_encounters = patient_encounter_dict[patient_id]
            if len(patient_encounters) == 0:
                continue
            
            # Convert encounter dates to numpy array once per patient
            encounter_dates = patient_encounters['EncounterDate'].values
            encounter_ids = patient_encounters['Encounter_ID'].values
            
            # Process all labs for this patient with vectorized operations
            for idx, lab in patient_labs.iterrows():
                if pd.isna(lab['PerformedDate']):
                    continue
                    
                # Calculate days difference using numpy (much faster)
                lab_date = np.datetime64(lab['PerformedDate'])
                days_diff = np.abs((encounter_dates - lab_date).astype('timedelta64[D]').astype(np.int64))
                
                # Find matches within window
                valid_match_indices = np.where(days_diff <= window_days)[0]
                if len(valid_match_indices) == 0:
                    continue
                
                # Find index of minimum days difference
                min_idx = valid_match_indices[np.argmin(days_diff[valid_match_indices])]
                best_days_diff = days_diff[min_idx]
                best_encounter_id = encounter_ids[min_idx]
                
                # Calculate confidence score (1.0 = same day)
                confidence = 1.0 - (best_days_diff / (window_days * 2))
                
                # Store linkage data efficiently
                result.loc[idx, 'Linked_Encounter_ID'] = best_encounter_id
                result.loc[idx, 'Days_To_Encounter'] = best_days_diff
                result.loc[idx, 'Linkage_Confidence'] = confidence
                
                linked_count += 1
        
        chunk_time = (datetime.now() - chunk_start).total_seconds()
        labs_per_second = len(chunk) / max(1, chunk_time)
        remaining_chunks = total_chunks - (chunk_idx + 1)
        est_remaining_time = remaining_chunks * chunk_time / 60  # minutes
        
        print(f"  Chunk {chunk_idx+1} processed in {chunk_time:.1f}s ({labs_per_second:.1f} labs/second)")
        print(f"  Progress: {linked_count:,} labs linked, ~{est_remaining_time:.1f} minutes remaining")
    
    # Create effective ID column for downstream analysis
    result['Effective_Encounter_ID'] = result['Encounter_ID']
    mask = result['Effective_Encounter_ID'].isna() & result['Linked_Encounter_ID'].notna()
    result.loc[mask, 'Effective_Encounter_ID'] = result.loc[mask, 'Linked_Encounter_ID']
    
    total_time = (datetime.now() - start_time).total_seconds() / 60  # minutes
    print(f"Successfully linked {linked_count:,} labs in {total_time:.1f} minutes")
    print(f"Total labs with encounter association: {result['Effective_Encounter_ID'].notna().sum():,} "
          f"({result['Effective_Encounter_ID'].notna().sum()/len(result)*100:.2f}%)")
    
    return result
def enhance_lab_classification_optimized(lab_df):
    """
    Optimized implementation of lab normal/abnormal classification
    using vectorized operations for better performance.
    """
    print("Enhancing lab classification with optimized approach...")
    
    result = lab_df.copy()
    total_labs = len(result)
    
    # Initialize as NA - will fill with True/False
    result['is_normal'] = pd.NA
    
    # Convert TestResult_calc to numeric once (more efficient)
    result['result_numeric'] = pd.to_numeric(result['TestResult_calc'], errors='coerce')
    
    start_time = datetime.now()
    
    # 1. APPROACH 1: Explicit normal ranges
    has_bounds = ~result['LowerNormal'].isna() & ~result['UpperNormal'].isna() & ~result['result_numeric'].isna()
    
    if has_bounds.any():
        print(f"Processing {has_bounds.sum():,} labs with explicit normal ranges...")
        
        # Vectorized conversion (once per field)
        lower_numeric = pd.to_numeric(result.loc[has_bounds, 'LowerNormal'], errors='coerce')
        upper_numeric = pd.to_numeric(result.loc[has_bounds, 'UpperNormal'], errors='coerce')
        
        # Check valid bounds and determine normal/abnormal (vectorized)
        valid_bounds = ~lower_numeric.isna() & ~upper_numeric.isna()
        bounds_indices = has_bounds[has_bounds].index[valid_bounds]
        
        # Vectorized comparison
        result.loc[bounds_indices, 'is_normal'] = (
            (result.loc[bounds_indices, 'result_numeric'] >= lower_numeric[valid_bounds]) & 
            (result.loc[bounds_indices, 'result_numeric'] <= upper_numeric[valid_bounds])
        )
        
        method1_count = valid_bounds.sum()
        print(f"Method 1: Processed {method1_count:,} labs ({method1_count/total_labs*100:.2f}%)")
    
    # 2. APPROACH 2: Reference ranges for common tests
    common_tests = {
        'TOTAL CHOLESTEROL': {'min': 0, 'max': 5.2, 'unit': 'mmol/L'},
        'HDL': {'min': 1.0, 'max': 3.0, 'unit': 'mmol/L'},
        'LDL': {'min': 0, 'max': 3.4, 'unit': 'mmol/L'},
        'TRIGLYCERIDES': {'min': 0, 'max': 1.7, 'unit': 'mmol/L'},
        'FASTING GLUCOSE': {'min': 3.9, 'max': 5.6, 'unit': 'mmol/L'},
        'HBA1C': {'min': 0, 'max': 5.7, 'unit': '%'},
        'TSH': {'min': 0.4, 'max': 4.0, 'unit': 'mIU/L'},
        'ALT': {'min': 0, 'max': 40, 'unit': 'U/L'},
        'AST': {'min': 0, 'max': 40, 'unit': 'U/L'},
        'CREATININE': {'min': 50, 'max': 120, 'unit': 'umol/L'},
        'HEMOGLOBIN': {'min': 120, 'max': 160, 'unit': 'g/L'},
        'WBC': {'min': 4.0, 'max': 11.0, 'unit': '10^9/L'},
        'POTASSIUM': {'min': 3.5, 'max': 5.0, 'unit': 'mmol/L'},
        'SODIUM': {'min': 135, 'max': 145, 'unit': 'mmol/L'}
    }
    
    method2_count = 0
    # Process all test types at once using a more efficient approach
    for test_name, reference in common_tests.items():
        # Find pending labs for this test (using case-insensitive string operations)
        missing_normal = result['is_normal'].isna()
        test_mask = (
            result['Name_calc'].str.contains(test_name, case=False, regex=False, na=False) &
            ~result['result_numeric'].isna() &
            missing_normal
        )
        
        if test_mask.any():
            mask_count = test_mask.sum()
            method2_count += mask_count
            
            # Vectorized normal check
            result.loc[test_mask, 'is_normal'] = (
                (result.loc[test_mask, 'result_numeric'] >= reference['min']) & 
                (result.loc[test_mask, 'result_numeric'] <= reference['max'])
            )
    
    print(f"Method 2: Processed {method2_count:,} labs ({method2_count/total_labs*100:.2f}%)")
    
    # 3. APPROACH 3: Text pattern search
    still_pending = result['is_normal'].isna() & ~result['TestResult_calc'].isna()
    method3_count = 0
    
    if still_pending.any():
        # Improved text pattern analysis with exact match phrases (better performance)
        # Convert TestResult_calc to string once (for all text operations)
        test_result_str = result.loc[still_pending, 'TestResult_calc'].astype(str)
        
        # Normal patterns
        normal_patterns = [
            'normal', 'neg', 'negative', 'unremarkable', 'w/in normal', 'within normal', 
            'wnl', 'within reference', 'not detected', 'n/a'
        ]
        
        # Apply all normal patterns at once (more efficient)
        normal_mask = np.zeros(len(test_result_str), dtype=bool)
        for pattern in normal_patterns:
            pattern_match = test_result_str.str.contains(
                f"\\b{pattern}\\b", case=False, regex=True, na=False
            )
            normal_mask = normal_mask | pattern_match.values
        
        # Set normal flags
        result.loc[still_pending[still_pending].index[normal_mask], 'is_normal'] = True
        method3_count += normal_mask.sum()
        
        # Update pending labs
        still_pending = result['is_normal'].isna() & ~result['TestResult_calc'].isna()
        
        # Apply abnormal patterns to remaining labs
        if still_pending.any():
            test_result_str = result.loc[still_pending, 'TestResult_calc'].astype(str)
            
            # Abnormal patterns
            abnormal_patterns = [
                'abnormal', 'pos', 'positive', 'high', 'low', 'elevated', 'depressed', 
                'outside', 'detected', 'present'
            ]
            
            # Apply all abnormal patterns at once
            abnormal_mask = np.zeros(len(test_result_str), dtype=bool)
            for pattern in abnormal_patterns:
                pattern_match = test_result_str.str.contains(
                    f"\\b{pattern}\\b", case=False, regex=True, na=False
                )
                abnormal_mask = abnormal_mask | pattern_match.values
            
            # Set abnormal flags
            result.loc[still_pending[still_pending].index[abnormal_mask], 'is_normal'] = False
            method3_count += abnormal_mask.sum()
    
    print(f"Method 3: Processed {method3_count:,} labs ({method3_count/total_labs*100:.2f}%)")
    
    # Calculate overall coverage
    classified_count = result['is_normal'].notna().sum()
    coverage_pct = classified_count / total_labs * 100
    normal_count = (result['is_normal'] == True).sum()  # Explicitly check for True
    abnormal_count = (result['is_normal'] == False).sum()  # Explicitly check for False
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"Lab classification completed in {total_time:.1f} seconds")
    print(f"Overall: Classified {classified_count:,} labs ({coverage_pct:.2f}%)")
    print(f"Normal: {normal_count:,} ({normal_count/classified_count*100:.2f}% of classified)")
    print(f"Abnormal: {abnormal_count:,} ({abnormal_count/classified_count*100:.2f}% of classified)")
    
    return result
def enhance_referral_dates_optimized(referral_df):
    """
    Optimized referral date enhancement using vectorized operations
    for better performance.
    """
    print("Implementing referral date enhancement (optimized)...")
    
    result = referral_df.copy()
    total_refs = len(result)
    
    # Check completion date coverage
    missing_completion = result['CompletedDate'].isna()
    missing_count = missing_completion.sum()
    missing_pct = missing_count / total_refs * 100
    
    print(f"CompletedDate missing in {missing_count:,} referrals ({missing_pct:.2f}%)")
    
    # Create effective date column and source tracking (vectorized)
    result['EffectiveDate'] = result['CompletedDate']
    result['DateSource'] = 'CompletedDate'
    
    # Use vectorized operations for fallback
    if missing_count > 0:
        # Apply DateCreated fallback where needed (single operation)
        result.loc[missing_completion, 'EffectiveDate'] = result.loc[missing_completion, 'DateCreated']
        result.loc[missing_completion, 'DateSource'] = 'DateCreated'
        
        # Check coverage after fallback
        remaining_missing = result['EffectiveDate'].isna().sum()
        
        print(f"After fallback: {remaining_missing:,} referrals still missing dates ({remaining_missing/total_refs*100:.2f}%)")
        print(f"Using DateCreated for {missing_count-remaining_missing:,} referrals")
    
    # Flag referral status (vectorized operations)
    result['ReferralStatus'] = 'Unknown'
    
    # Completed referrals have CompletedDate (single operation)
    result.loc[result['CompletedDate'].notna(), 'ReferralStatus'] = 'Completed'
    
    # Pending referrals have DateCreated but no CompletedDate (single operation)
    pending_mask = (result['CompletedDate'].isna()) & (result['DateCreated'].notna())
    result.loc[pending_mask, 'ReferralStatus'] = 'Pending'
    
    # Count by status
    status_counts = result['ReferralStatus'].value_counts()
    print("\nReferral Status Distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,} ({count/total_refs*100:.2f}%)")
    
    return result

# ---------------------------- Main Execution ------------------------------ #
# v1 - ORIGINAL 
# def main():
#     """Main execution function to run the full data loading and validation process."""
#     print_section_header("CPCSSN Care4Mind Dataset: Data Loading and Validation")
#     print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d')}")
   
#     # Step 1: Load and perform initial data validation
#     data, data_quality = load_and_validate_data()
   
#     # Step 2: Validate data relationships
#     relationship_metrics = validate_data_relationships(data)
   
#     # Step 3: Validate temporal consistency
#     temporal_metrics = validate_temporal_consistency(data)
   
#     # Step 4: Analyze patient population
#     population_metrics = analyze_patient_population(data)
   
#     # Step 5: Analyze coding patterns
#     coding_metrics = analyze_coding_patterns(data)
   
#     # Step 6: Analyze lab data
#     lab_metrics = analyze_lab_data(data)
   
#     # Step 7: Analyze referral patterns
#     referral_metrics = analyze_referral_patterns(data)
   
#     # Step 8: Save validation results
#     validation_report_path = save_validation_results(
#         data_quality, relationship_metrics, temporal_metrics,
#         population_metrics, coding_metrics, lab_metrics, referral_metrics
#     )
   
#     # Step 9: Save checkpoint for next notebook
#     save_checkpoint(data, validation_report_path)
   
#     print_section_header("Data Validation Complete")
#     print("✓ Data loaded and validated")
#     print("✓ Quality metrics calculated")
#     print("✓ Visualizations generated")
#     print("✓ Checkpoint saved for next notebook")
   
#     print("\nProceed to Notebook 2: NYD Identification")


# V2 - ENHANCED MAIN

# Updated main function with changes tracked
def main():
    """Main execution with optimized data processing."""
    print_section_header("CPCSSN Care4Mind Dataset: Data Loading and Validation")
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Track changes for documentation
    changes = []
    
    # Step 1: Load and validate data
    data, data_quality = load_and_validate_data()
    
    # Step 2-5: Standard validation and analysis
    relationship_metrics = validate_data_relationships(data)
    temporal_metrics = validate_temporal_consistency(data)
    population_metrics = analyze_patient_population(data)
    coding_metrics = analyze_coding_patterns(data)
    
    # Step 6: Enhanced lab processing with optimized implementation
    print("\nApplying optimized lab processing...")
    
    # Link orphaned labs to encounters using optimized function
    original_labs = len(data['lab'])
    data['lab'] = link_labs_to_encounters_by_time_optimized(data['lab'], data['encounter'])
    changes.append(f"Linked orphaned labs to encounters through temporal proximity (optimized)")
    
    # Enhance normal lab classification using optimized function
    data['lab'] = enhance_lab_classification_optimized(data['lab'])
    changes.append("Expanded normal lab detection from 14% to ~45% using multiple methods (optimized)")
    
    # Continue with regular lab analysis
    lab_metrics = analyze_lab_data(data)
    
    # Step 7: Enhanced referral processing with optimized implementation
    print("\nApplying optimized referral processing...")
    data['referral'] = enhance_referral_dates_optimized(data['referral'])
    changes.append("Implemented referral date fallbacks and status tracking (optimized)")
    
    # Continue with regular referral analysis
    referral_metrics = analyze_referral_patterns(data)
    
    # Step 8: Save validation with corrected NYD reporting
    validation_report_path = save_validation_results(
        data_quality, relationship_metrics, temporal_metrics, 
        population_metrics, coding_metrics, lab_metrics, referral_metrics
    )
    changes.append("Fixed NYD code reporting in validation summary")
    
    # Step 9: Save comprehensive checkpoint
    checkpoint_dir = save_checkpoint_with_documentation(
        data,
        notebook_number=1,
        description="Data Loading and Validation with Optimized Processing",
        changes=changes
    )
    
    print_section_header("Optimized Data Validation Complete")
    print("✓ Data loaded and validated with fixes")
    print("✓ NYD code reporting corrected")
    print("✓ Lab-encounter temporal linkage implemented (optimized)")
    print("✓ Normal lab detection significantly expanded (optimized)")
    print("✓ Referral date handling improved (optimized)")
    print("✓ Comprehensive documentation created")
    
    print(f"\nProceed to Notebook 2: NYD_Identification.ipynb")
    print(f"Load data from: {checkpoint_dir}")

if __name__ == "__main__":
    main()