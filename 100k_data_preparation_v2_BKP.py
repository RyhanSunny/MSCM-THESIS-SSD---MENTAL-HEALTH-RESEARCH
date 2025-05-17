# data_preparation_v2.py

# The data is pre-filtered at the SQL level
# # WHERE Substring([DiagnosisCode_orig],1,3) in ('290', '291',...,'799', '995')
# SQL shows ICD-9 codes used for mental health conditions (290-319, plus some related physical conditions)

# What this Script Does:

# Reads pipe-delimited CSVs from CPCSSN dataset (clinical data)
# Cleans and standardizes each table according to config.yaml rules:

# Patient details
# Demographics
# Encounters/visits
# Diagnoses
# Labs/medications etc

# also finds the date ranges earliest date and latest

# Takes a stratified patient sample (2000 patients by default)
# Validates relationships between tables (e.g., Patient_IDs match across tables)
# Checks data quality (missing values, date ranges, etc)
# Saves cleaned CSV files to a 'prepared_data' directory

# Outputs:

# Cleaned CSV files in prepared_data/:

# *_prepared.csv for each input table
# All data linked to sampled patients
# Standardized formats (dates, codes, etc)

# Quality metrics including:

# Row/column counts
# Missing value percentages
# Date ranges
# Data relationship issues

# data_preparation_v2.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import yaml
from typing import Dict, List, Optional, Set, Tuple
import warnings
import sys
warnings.filterwarnings('ignore')

class DataPreparationPipeline:
    """
    Pipeline for preparing CPCSSN data with date-range filtering,
    stratified sampling, and relationship integrity checks.
    """

    def __init__(self, config_path: str, data_dir: str = None):
        self.setup_logging()
        self.logger.info("Initializing pipeline...")
        
        try:
            self.config = self._load_config(config_path)
            self.logger.info("Configuration loaded successfully")

            # Resolve config paths
            config_dir = Path(config_path).parent
            self.config['data_paths']['raw_data'] = (config_dir / self.config['data_paths']['raw_data']).resolve()
            self.config['data_paths']['prepared_data'] = (config_dir / self.config['data_paths']['prepared_data']).resolve()
            
            if data_dir:
                self.config['data_paths']['raw_data'] = Path(data_dir).resolve()
                self.logger.info(f"Using custom data directory: {data_dir}")
            
            # Validate raw data path
            raw_data_path = self.config['data_paths']['raw_data']
            if not raw_data_path.exists():
                raise ValueError(f"Raw data directory not found: {raw_data_path}")
            
            self.chunk_size = self.config.get("chunk_size", 0)
            self.logger.info(f"Chunk size set to: {self.chunk_size}")
            
            # Create 'outliers' directory
            outliers_dir = Path(self.config['data_paths']['prepared_data']) / 'outliers'
            outliers_dir.mkdir(parents=True, exist_ok=True)
            
            # Cache for outlier rows by table
            self.outlier_rows = {}  # {table_name: pd.DataFrame}

        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise

    def setup_logging(self):
        """Configure logging."""
        log_path = Path('logs')
        log_path.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / f'data_preparation_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            required_sections = ['data_paths', 'tables', 'quality_thresholds', 'sampling']
            missing_sections = [s for s in required_sections if s not in config]
            if missing_sections:
                raise ValueError(f"Missing required config sections: {missing_sections}")
        return config

    def _append_outliers(self, table_name: str, outlier_df: pd.DataFrame):
        """
        Collect outlier rows in a class-level dictionary for each table.
        We'll write them out once table loading is complete.
        """
        if table_name not in self.outlier_rows:
            self.outlier_rows[table_name] = []
        self.outlier_rows[table_name].append(outlier_df)

    def clean_calc_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        calc_columns = [col for col in df.columns if col.endswith('_calc')]
        for col in calc_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip().str.upper()
        return df

    def standardize_diagnosis_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'DiagnosisCode_calc' in df.columns and 'DiagnosisCodeType_calc' in df.columns:
            mask_icd = df['DiagnosisCodeType_calc'].str.contains('ICD', na=False)
            df.loc[mask_icd, 'DiagnosisCode_calc'] = (
                df.loc[mask_icd, 'DiagnosisCode_calc'].str.replace('.', '')
            )
        return df

    def clean_date_columns(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        date_columns: List[str]
    ) -> pd.DataFrame:
        """
        Convert date columns, log out-of-range, and remove them from df.
        Write out-of-range rows to an outliers buffer.
        """
        min_date = pd.Timestamp(self.config['quality_thresholds']['min_date'])
        max_date = pd.Timestamp(self.config['quality_thresholds']['max_date'])

        for col in date_columns:
            if col not in df.columns:
                continue
            # Convert to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Identify out-of-range rows
            invalid_mask = df[col].notna() & ((df[col] < min_date) | (df[col] > max_date))
            n_invalid = invalid_mask.sum()
            if n_invalid > 0:
                self.logger.warning(
                    f"Found {n_invalid} dates outside expected range "
                    f"({min_date.date()} to {max_date.date()}) in column {col} (table {table_name})"
                )
                # Collect outlier rows
                outliers = df[invalid_mask].copy()
                # (Optional) label which column triggered outlier
                outliers['__OutlierDateColumn'] = col
                self._append_outliers(table_name, outliers)
                # Remove them from main df
                df = df[~invalid_mask]
        return df

    def load_table(self, table_name: str) -> pd.DataFrame:
        """
        Load table in chunks, convert date columns,
        store outliers in self.outlier_rows, 
        and remove them from the final DataFrame.
        """
        table_config = self.config['tables'][table_name]
        file_path = (
            Path(self.config['data_paths']['raw_data']) 
            / f"{table_config['filename']}{self.config['data_paths']['file_extension']}"
        )

        self.logger.info(f"Loading table: {table_name} from {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        date_cols = table_config.get('date_columns', [])
        dfs = []
        chunk_count = 0

        if self.chunk_size > 0:
            # Read in chunks
            for chunk in pd.read_csv(
                file_path,
                sep='|',
                quotechar='"',
                chunksize=self.chunk_size,
                dtype=str,
                on_bad_lines='warn'
            ):
                chunk_count += 1
                # Basic string cleaning
                for c in chunk.select_dtypes(['object']).columns:
                    chunk[c] = chunk[c].str.strip()

                chunk = self.clean_calc_columns(chunk)
                chunk = self.clean_date_columns(chunk, table_name, date_cols)
                if table_name in ['EncounterDiagnosis', 'HealthCondition']:
                    chunk = self.standardize_diagnosis_codes(chunk)
                dfs.append(chunk)

                if chunk_count % 10 == 0:
                    self.logger.info(f"Processed {chunk_count} chunks of {table_name}")
            df = pd.concat(dfs, ignore_index=True)

        else:
            # Single shot read
            df = pd.read_csv(
                file_path, sep='|', quotechar='"', dtype=str, on_bad_lines='warn'
            )
            for c in df.select_dtypes(['object']).columns:
                df[c] = df[c].str.strip()
            df = self.clean_calc_columns(df)
            df = self.clean_date_columns(df, table_name, date_cols)
            if table_name in ['EncounterDiagnosis', 'HealthCondition']:
                df = self.standardize_diagnosis_codes(df)

        # Check required columns
        missing_cols = set(table_config['required_columns']) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in {table_name}: {missing_cols}")

        self.logger.info(f"Successfully loaded {table_name}: {len(df)} rows")
        return df

    def sample_patients(self, df_demo: pd.DataFrame, df_patient: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str]]:
        """
        1) Merge 'Sex' from Patient into Demographic
        2) Do stratified sampling by Sex, if present
        3) Return the filtered Demographic DF + set of Patient_IDs
        """
        self.logger.info(f"Sampling {self.config['sampling']['n_patients']} patients from PatientDemographic")

        # Merge sex from df_patient
        if 'Patient_ID' in df_patient.columns and 'Patient_ID' in df_demo.columns:
            if 'Sex' in df_patient.columns:
                # Merge so that df_demo has the 'Sex' column
                df_demo = pd.merge(
                    df_demo, 
                    df_patient[['Patient_ID','Sex']], 
                    on='Patient_ID', 
                    how='left',
                    suffixes=('', '_Pat')
                )
            else:
                self.logger.warning("Patient table does not have 'Sex' column - cannot stratify by Sex.")
        else:
            self.logger.warning("Either Patient or PatientDemographic does not have Patient_ID. No merging.")

        n_patients = self.config['sampling']['n_patients']
        stratify_cols = self.config['sampling'].get('stratify_columns', [])

        # Ensure all stratify columns exist in df_demo
        all_exists = all((col in df_demo.columns) for col in stratify_cols)
        if all_exists and len(stratify_cols) > 0:
            self.logger.info(f"Performing stratified sampling by {stratify_cols}")
            # Group by (Sex) or multiple columns
            grouped = df_demo.groupby(stratify_cols, group_keys=False)
            
            # Each group gets a proportion of n_patients ~ equal to group size
            def sample_group(subdf):
                # fraction of this group in the total
                fraction = len(subdf) / len(df_demo)
                desired_n = int(round(n_patients * fraction))
                if desired_n > len(subdf):
                    desired_n = len(subdf)
                return subdf.sample(desired_n, random_state=42)  # set random_state for reproducibility

            sampled_df = grouped.apply(sample_group)
        else:
            # No valid stratification columns, or they're missing
            self.logger.warning("Stratification columns not found or empty. Using simple random sample.")
            if n_patients > len(df_demo):
                n_patients = len(df_demo)
            sampled_df = df_demo.sample(n=n_patients, random_state=42)

        sampled_ids = set(sampled_df['Patient_ID'].unique())
        self.logger.info(f"Selected {len(sampled_ids)} unique patients in sample.")
        return sampled_df, sampled_ids

    def filter_related_tables(self, df: pd.DataFrame, patient_ids: Set[str]) -> pd.DataFrame:
        """Keep only rows where Patient_ID is in the sampled set."""
        if 'Patient_ID' in df.columns:
            before = len(df)
            df = df[df['Patient_ID'].isin(patient_ids)]
            self.logger.info(f"Filtered from {before} to {len(df)} rows by Patient_ID.")
        return df

    def filter_missing_encounter_ids(self, 
                                     df: pd.DataFrame, 
                                     encounter_ids: Set[str], 
                                     table_name: str) -> pd.DataFrame:
        """
        Exclude rows whose Encounter_ID is not in the main Encounter table,
        ensuring 'solid' relationships.
        """
        if 'Encounter_ID' in df.columns:
            before = len(df)
            df = df[df['Encounter_ID'].isin(encounter_ids)]
            after = len(df)
            removed = before - after
            if removed > 0:
                self.logger.info(
                    f"Removed {removed} rows in {table_name} referencing missing Encounter_ID."
                )
        return df

    def validate_relationships(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Check referential integrity and return warnings."""
        issues = {}
        # Validate that if Demo has Patient_ID, it’s in Patient
        if 'Patient' in tables and 'PatientDemographic' in tables:
            patient_ids = set(tables['Patient']['Patient_ID'])
            demo_ids = set(tables['PatientDemographic']['Patient_ID'])
            missing_ids = demo_ids - patient_ids
            if missing_ids:
                issues['PatientDemographic'] = [
                    f"{len(missing_ids)} Patient_IDs not found in Patient table"
                ]
        # Validate Encounter-based relationships
        if 'Encounter' in tables:
            enc_ids = set(tables['Encounter']['Encounter_ID'])
            for tn in ['EncounterDiagnosis','Medication','Lab']:
                if tn in tables and 'Encounter_ID' in tables[tn].columns:
                    missing = set(tables[tn]['Encounter_ID']) - enc_ids
                    if missing:
                        issues[tn] = [
                            f"{len(missing)} Encounter_IDs not found in Encounter table"
                        ]
        return issues

    def assess_data_quality(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """High-level metrics on row/column counts, missing percentages, date ranges."""
        quality_metrics = {}
        for tname, df in tables.items():
            self.logger.info(f"Assessing quality metrics for {tname}")
            qc = {
                'row_count': len(df),
                'column_count': df.shape[1],
                'missing_counts': df.isnull().sum().to_dict(),
                'missing_percentages': (df.isnull().mean() * 100).round(2).to_dict()
            }
            # Date columns ranges
            dcols = self.config['tables'][tname].get('date_columns', [])
            for col in dcols:
                if col in df.columns:
                    dd = pd.to_datetime(df[col], errors='coerce')
                    qc[f'{col}_range'] = {
                        'min': dd.min().strftime('%Y-%m-%d') if not pd.isna(dd.min()) else None,
                        'max': dd.max().strftime('%Y-%m-%d') if not pd.isna(dd.max()) else None,
                        'null_count': dd.isnull().sum()
                    }
            quality_metrics[tname] = qc
        return quality_metrics

    def save_outliers(self):
        """
        Write each table’s collected outliers to 
        prepared_data/outliers/{table}_Outside_Date_Range_2000to2025.csv
        """
        outlier_dir = Path(self.config['data_paths']['prepared_data']) / 'outliers'
        for table_name, list_of_dfs in self.outlier_rows.items():
            if not list_of_dfs:
                continue
            df_outliers = pd.concat(list_of_dfs, ignore_index=True)
            out_path = outlier_dir / f"{table_name}_Outside_Date_Range_2000to2025.csv"
            df_outliers.to_csv(out_path, index=False, encoding='utf-8')
            self.logger.info(f"Wrote {len(df_outliers)} outlier rows to {out_path}")

    def save_prepared_data(self, tables: Dict[str, pd.DataFrame]):
        """Save final CSVs to prepared_data/ directory."""
        output_dir = Path(self.config['data_paths']['prepared_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            out_file = output_dir / f"{name}_prepared.csv"
            df.to_csv(out_file, index=False, encoding='utf-8')
            self.logger.info(f"Saved {name} -> {out_file}")

def main():
    print("Starting data preparation pipeline...")
    pipeline = DataPreparationPipeline('config.yaml')
    print("Pipeline initialized successfully")

    # 1) Analyze date ranges across certain tables (same as before)
    tables = {}
    date_tables = [
        'Encounter','EncounterDiagnosis','HealthCondition','Lab',
        'MedicalProcedure','Medication','Referral','RiskFactor'
    ]
    print("\nAnalyzing date ranges across tables...")
    for tn in date_tables:
        if tn in pipeline.config['tables']:
            print(f"Loading {tn} for date analysis...")
            tables[tn] = pipeline.load_table(tn)

    # (Optional) find and display date ranges ...
    # (Same as your existing code for date analysis) 
    # Skipping here for brevity

    # Clear memory for next step
    tables.clear()

    # 2) Full data load
    print("\nProceeding with full data processing...")
    tables = {}
    for table_name in pipeline.config['tables'].keys():
        print(f"Processing table: {table_name}")
        tables[table_name] = pipeline.load_table(table_name)

    # 2b) Save outliers from all tables so far
    pipeline.save_outliers()

    # 3) Now sample patients from PatientDemographic 
    #    but we also need 'Patient' to get Sex
    if 'PatientDemographic' in tables and 'Patient' in tables:
        print("\nSampling patients...")
        sampled_demo, sampled_ids = pipeline.sample_patients(
            tables['PatientDemographic'],
            tables['Patient']
        )
        tables['PatientDemographic'] = sampled_demo
        # Filter other tables by Patient_ID
        print("\nFiltering related tables to sampled patients...")
        for tname in tables:
            if tname not in ('PatientDemographic','Patient'):  # Keep 'Patient' as is
                tables[tname] = pipeline.filter_related_tables(
                    tables[tname], sampled_ids
                )
    else:
        print("ERROR: Must have both 'PatientDemographic' and 'Patient' for sampling.")
        return

    # 4) Remove orphan Encounter_ID references from EncounterDiagnosis, Medication, Lab
    if 'Encounter' in tables:
        valid_enc_ids = set(tables['Encounter']['Encounter_ID'])
        for tname in ['EncounterDiagnosis','Medication','Lab']:
            if tname in tables:
                tables[tname] = pipeline.filter_missing_encounter_ids(
                    tables[tname], valid_enc_ids, tname
                )

    # 5) Validate relationships & log warnings
    print("\nValidating table relationships...")
    relationship_issues = pipeline.validate_relationships(tables)
    if relationship_issues:
        print("WARNING: Found data relationship issues:")
        for tbl, msgs in relationship_issues.items():
            for m in msgs:
                print(f"  {tbl}: {m}")

    # 6) Assess data quality
    print("\nAssessing data quality...")
    quality_metrics = pipeline.assess_data_quality(tables)

    # 7) Save final prepared data
    print("\nSaving prepared data...")
    pipeline.save_prepared_data(tables)

    print("\nData preparation completed successfully!")
    return tables, quality_metrics, relationship_issues

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)
