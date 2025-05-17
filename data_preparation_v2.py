# data_preparation_v2.py

# The data is pre-filtered at the SQL level
# # WHERE Substring([DiagnosisCode_orig],1,3) in ('290', '291',...,'799', '995')
# SQL shows ICD-9 codes used for mental health conditions (290-319, plus some related physical conditions)

# What this Script Does:
#
# Reads pipe-delimited CSVs from the CPCSSN dataset (clinical data)
# Cleans and standardizes each table according to config.yaml rules:
#
#   - Patient details
#   - Demographics
#   - Encounters/visits
#   - Diagnoses
#   - Labs/medications, etc.
#
# Also finds the date ranges (earliest and latest dates) for quality checks.
#
# Merges the Patient and PatientDemographic tables to create a unified demographic table,
# standardizing gender values (MALE/FEMALE/OTHER/UNKNOWN), flagging patients recorded as deceased,
# and calculating age dynamically (from BirthDate to DeathDate if deceased, or to March 2025 if alive).
#
# Instead of dropping rows with out-of-range dates, these rows are stored in outliers CSV files 
# for review. The expected date range is now 1900-01-01 to 2024-12-31.
#
# Validates relationships between tables (e.g., matching Patient_IDs across tables)
# and checks data quality (missing values, date ranges, etc).
#
# Saves cleaned CSV files to a 'prepared_data' directory.
#
# Outputs:
#
#   - Cleaned CSV files in prepared_data/:
#       *_prepared.csv for each input table (including the merged PatientDemographic table)
#       Standardized formats (dates, codes, etc)
#
#   - Outliers CSV files in prepared_data/outliers/ containing rows with dates outside the range (1900-2024)
#
#   - Quality metrics including:
#       Row/column counts, missing value percentages, date ranges, and data relationship issues.
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import yaml
from typing import Dict, List
import warnings
import sys
warnings.filterwarnings('ignore')

class DataPreparationPipeline:
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
            
            raw_data_path = self.config['data_paths']['raw_data']
            if not raw_data_path.exists():
                raise ValueError(f"Raw data directory not found: {raw_data_path}")
            
            self.chunk_size = self.config.get("chunk_size", 0)
            self.logger.info(f"Chunk size set to: {self.chunk_size}")
            
            # Create 'outliers' directory
            outliers_dir = Path(self.config['data_paths']['prepared_data']) / 'outliers'
            outliers_dir.mkdir(parents=True, exist_ok=True)
            
            # Cache for outlier rows by table
            self.outlier_rows = {}
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise

    def setup_logging(self):
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
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            required_sections = ['data_paths', 'tables', 'quality_thresholds']
            missing_sections = [s for s in required_sections if s not in config]
            if missing_sections:
                raise ValueError(f"Missing required config sections: {missing_sections}")
        return config

    def _append_outliers(self, table_name: str, outlier_df: pd.DataFrame):
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

    def clean_date_columns(self, df: pd.DataFrame, table_name: str, date_columns: List[str]) -> pd.DataFrame:
        min_date = pd.Timestamp(self.config['quality_thresholds']['min_date'])
        max_date = pd.Timestamp(self.config['quality_thresholds']['max_date'])
        for col in date_columns:
            if col not in df.columns:
                continue
            df[col] = pd.to_datetime(df[col], errors='coerce')
            invalid_mask = df[col].notna() & ((df[col] < min_date) | (df[col] > max_date))
            n_invalid = invalid_mask.sum()
            if n_invalid > 0:
                self.logger.warning(
                    f"Found {n_invalid} dates outside expected range "
                    f"({min_date.date()} to {max_date.date()}) in column {col} (table {table_name})"
                )
                outliers = df[invalid_mask].copy()
                outliers['__OutlierDateColumn'] = col
                self._append_outliers(table_name, outliers)
                # Do not drop these rows to reduce data loss.
        return df

    def load_table(self, table_name: str) -> pd.DataFrame:
        table_config = self.config['tables'][table_name]
        file_path = (Path(self.config['data_paths']['raw_data']) / f"{table_config['filename']}{self.config['data_paths']['file_extension']}")
        self.logger.info(f"Loading table: {table_name} from {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        date_cols = table_config.get('date_columns', [])
        dfs = []
        chunk_count = 0
        if self.chunk_size > 0:
            for chunk in pd.read_csv(
                file_path,
                sep='|',
                quotechar='"',
                chunksize=self.chunk_size,
                dtype=str,
                on_bad_lines='warn'
            ):
                chunk_count += 1
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
            df = pd.read_csv(file_path, sep='|', quotechar='"', dtype=str, on_bad_lines='warn')
            for c in df.select_dtypes(['object']).columns:
                df[c] = df[c].str.strip()
            df = self.clean_calc_columns(df)
            df = self.clean_date_columns(df, table_name, date_cols)
            if table_name in ['EncounterDiagnosis', 'HealthCondition']:
                df = self.standardize_diagnosis_codes(df)
        missing_cols = set(table_config['required_columns']) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in {table_name}: {missing_cols}")
        self.logger.info(f"Successfully loaded {table_name}: {len(df)} rows")
        return df

    def filter_missing_encounter_ids(self, df: pd.DataFrame, encounter_ids: set, table_name: str) -> pd.DataFrame:
        """
        Instead of removing rows with missing Encounter_IDs,
        flag them by adding a new column 'orphan_encounter'.
        """
        if 'Encounter_ID' in df.columns:
            df['orphan_encounter'] = ~df['Encounter_ID'].isin(encounter_ids)
            orphan_count = df['orphan_encounter'].sum()
            if orphan_count > 0:
                self.logger.info(f"Flagged {orphan_count} rows in {table_name} as orphan (missing Encounter_ID).")
        return df

    def validate_relationships(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        issues = {}
        if 'Patient' in tables and 'PatientDemographic' in tables:
            patient_ids = set(tables['Patient']['Patient_ID'])
            demo_ids = set(tables['PatientDemographic']['Patient_ID'])
            missing_ids = demo_ids - patient_ids
            if missing_ids:
                issues['PatientDemographic'] = [f"{len(missing_ids)} Patient_IDs not found in Patient table"]
        if 'Encounter' in tables:
            enc_ids = set(tables['Encounter']['Encounter_ID'])
            for tn in ['EncounterDiagnosis', 'Medication', 'Lab']:
                if tn in tables and 'Encounter_ID' in tables[tn].columns:
                    missing = set(tables[tn]['Encounter_ID']) - enc_ids
                    if missing:
                        issues[tn] = [f"{len(missing)} Encounter_IDs not found in Encounter table"]
        return issues

    def assess_data_quality(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Compute high-level metrics on row/column counts, missing percentages,
        and date ranges. For tables not defined in config, skip date range checks.
        """
        quality_metrics = {}
        for tname, df in tables.items():
            self.logger.info(f"Assessing quality metrics for {tname}")
            # Use default empty config if table not in YAML
            table_config = self.config['tables'].get(tname, {})
            dcols = table_config.get('date_columns', [])
            qc = {
                'row_count': len(df),
                'column_count': df.shape[1],
                'missing_counts': df.isnull().sum().to_dict(),
                'missing_percentages': (df.isnull().mean() * 100).round(2).to_dict()
            }
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
        outlier_dir = Path(self.config['data_paths']['prepared_data']) / 'outliers'
        min_date = pd.Timestamp(self.config['quality_thresholds']['min_date'])
        max_date = pd.Timestamp(self.config['quality_thresholds']['max_date'])
        date_range_str = f"{min_date.year}to{max_date.year}"
        for table_name, list_of_dfs in self.outlier_rows.items():
            if not list_of_dfs:
                continue
            df_outliers = pd.concat(list_of_dfs, ignore_index=True)
            out_path = outlier_dir / f"{table_name}_Outside_Date_Range_{date_range_str}.csv"
            df_outliers.to_csv(out_path, index=False, encoding='utf-8')
            self.logger.info(f"Wrote {len(df_outliers)} outlier rows to {out_path}")

    def save_prepared_data(self, tables: Dict[str, pd.DataFrame]):
        output_dir = Path(self.config['data_paths']['prepared_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            out_file = output_dir / f"{name}_prepared.csv"
            df.to_csv(out_file, index=False, encoding='utf-8')
            self.logger.info(f"Saved {name} -> {out_file}")

    def standardize_gender(self, gender_val):
        if pd.isnull(gender_val):
            return "UNKNOWN"
        g = str(gender_val).strip().upper()
        if g in ["MALE", "M"]:
            return "MALE"
        elif g in ["FEMALE", "F"]:
            return "FEMALE"
        elif g in ["OTHER", "NON-BINARY", "NB"]:
            return g
        else:
            return "UNKNOWN"

    def merge_patient_demographics(self, df_demo: pd.DataFrame, df_patient: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(df_demo, df_patient, on="Patient_ID", how="left", suffixes=('_demo', '_pat'))
        if 'Sex' in merged.columns:
            merged['Sex'] = merged['Sex'].apply(self.standardize_gender)
        elif 'Sex_demo' in merged.columns:
            merged['Sex'] = merged['Sex_demo'].apply(self.standardize_gender)
        elif 'Sex_pat' in merged.columns:
            merged['Sex'] = merged['Sex_pat'].apply(self.standardize_gender)
        else:
            merged['Sex'] = "UNKNOWN"
        
        death_date_col = None
        if 'DeathDate' in merged.columns:
            death_date_col = 'DeathDate'
            merged[death_date_col] = pd.to_datetime(merged[death_date_col], errors='coerce')
        elif 'DeceasedDate' in merged.columns:
            death_date_col = 'DeceasedDate'
            merged[death_date_col] = pd.to_datetime(merged[death_date_col], errors='coerce')
        
        if 'BirthDate' in merged.columns:
            merged['BirthDate'] = pd.to_datetime(merged['BirthDate'], errors='coerce')
        
        if death_date_col:
            merged['recorded_deceased'] = merged[death_date_col].notnull().apply(lambda x: "YES" if x else "NO")
        else:
            merged['recorded_deceased'] = "NO"
        
        reference_date = pd.Timestamp("2025-03-01")
        def compute_age(row):
            if pd.isnull(row.get('BirthDate')):
                return None
            if death_date_col and pd.notnull(row.get(death_date_col)):
                end_date = row.get(death_date_col)
            else:
                end_date = reference_date
            age = end_date.year - row['BirthDate'].year - ((end_date.month, end_date.day) < (row['BirthDate'].month, row['BirthDate'].day))
            return age
        
        merged['age'] = merged.apply(compute_age, axis=1)
        return merged

def main():
    print("Starting data preparation pipeline...")
    pipeline = DataPreparationPipeline('config.yaml')
    print("Pipeline initialized successfully")

    # 1) Analyze date ranges across selected tables (for quality checking)
    tables = {}
    date_tables = [
        'Encounter', 'EncounterDiagnosis', 'HealthCondition', 'Lab',
        'MedicalProcedure', 'Medication', 'Referral', 'RiskFactor'
    ]
    print("\nAnalyzing date ranges across tables...")
    for tn in date_tables:
        if tn in pipeline.config['tables']:
            print(f"Loading {tn} for date analysis...")
            tables[tn] = pipeline.load_table(tn)
    tables.clear()

    # 2) Full data load for all tables
    print("\nProceeding with full data processing...")
    tables = {}
    for table_name in pipeline.config['tables'].keys():
        print(f"Processing table: {table_name}")
        tables[table_name] = pipeline.load_table(table_name)

    # 2b) Save outliers identified so far
    pipeline.save_outliers()

    # 3) Merge Patient and PatientDemographic tables (using the full dataset)
    if 'PatientDemographic' in tables and 'Patient' in tables:
        print("\nMerging Patient and PatientDemographic tables...")
        merged_demo = pipeline.merge_patient_demographics(tables['PatientDemographic'], tables['Patient'])
        tables['PatientDemographic_Merged'] = merged_demo
        print("Created merged PatientDemographic_Merged table with 'recorded_deceased' and 'age' columns.")
    else:
        print("ERROR: Must have both 'PatientDemographic' and 'Patient' for merging.")
        return

    # 4) Flag orphan Encounter_ID references in related tables
    if 'Encounter' in tables:
        valid_enc_ids = set(tables['Encounter']['Encounter_ID'])
        for tname in ['EncounterDiagnosis', 'Medication', 'Lab']:
            if tname in tables:
                tables[tname] = pipeline.filter_missing_encounter_ids(tables[tname], valid_enc_ids, tname)

    # 5) Validate relationships and log any warnings
    print("\nValidating table relationships...")
    relationship_issues = pipeline.validate_relationships(tables)
    if relationship_issues:
        print("WARNING: Found data relationship issues:")
        for tbl, msgs in relationship_issues.items():
            for m in msgs:
                print(f"  {tbl}: {m}")

    # 6) Assess overall data quality
    print("\nAssessing data quality...")
    quality_metrics = pipeline.assess_data_quality(tables)

    # 7) Save final cleaned and prepared data
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
