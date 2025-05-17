# data_preparation.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import yaml
from typing import Dict, List, Optional, Set, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataPreparationPipeline:
    """
    Pipeline for preparing Care4Mind EHR data:
      - Reads pipe-delimited CSV with quote="\""
      - Allows chunked reading for large files
      - Optional GPU usage (via cudf) if use_gpu = True in config
    """

    def __init__(self, config_path: str, data_dir: str = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config_path: Path to YAML configuration file
            data_dir: Optional override for data directory path
        """
        self.config = self._load_config(config_path)
        if data_dir:
            self.config['data_paths']['raw_data'] = data_dir
        self.use_gpu = self.config.get("use_gpu", False)
        self.chunk_size = self.config.get("chunk_size", 0)
        self.setup_logging()

    def setup_logging(self):
        """Configure logging with timestamp and level."""
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
            return yaml.safe_load(file)

    def load_table(self, table_name: str) -> pd.DataFrame:
        """
        Load a single table from pipe-delimited CSV, with chunking if specified,
        and optional GPU acceleration if 'use_gpu' is true.

        Args:
            table_name: Name of the table to load

        Returns:
            pd.DataFrame: Loaded and cleaned dataframe
        """
        table_config = self.config['tables'][table_name]
        file_path = Path(self.config['data_paths']['raw_data']) / f"{table_config['filename']}{self.config['data_paths']['file_extension']}"
        self.logger.info(f"Loading table: {table_name} from {file_path}")

        try:
            # Attempt GPU read if use_gpu = True
            if self.use_gpu:
                try:
                    import cudf
                    self.logger.info("Using cudf for GPU-accelerated read_csv")
                    df_cu = cudf.read_csv(
                        file_path,
                        sep='|',
                        quotechar='"'
                    )
                    df = df_cu.to_pandas()
                except ImportError:
                    self.logger.warning("cudf not available; falling back to pandas.")
                    df = self._read_with_pandas(file_path)
            else:
                # Normal pandas read (with or without chunking)
                df = self._read_with_pandas(file_path)

            # Basic cleaning
            df = self._basic_cleaning(df, table_config)

            # Validate required columns
            missing_cols = set(table_config['required_columns']) - set(df.columns)
            if missing_cols:
                self.logger.error(f"Missing required columns in {table_name}: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")

            self.logger.info(f"Successfully loaded {table_name}: {len(df)} rows")
            return df

        except Exception as e:
            self.logger.error(f"Error loading {table_name}: {str(e)}")
            raise

    def _read_with_pandas(self, file_path: Path) -> pd.DataFrame:
        """
        Helper to read with pandas, optionally chunked, with extra tolerance for malformed lines.
        """
        # You can switch 'warn' -> 'skip' to skip malformed lines silently
        on_bad = 'warn'
        self.logger.info(f"Reading {file_path} with on_bad_lines='{on_bad}'")

        if self.chunk_size and self.chunk_size > 0:
            # Chunked reading
            self.logger.info(f"Reading {file_path} in chunks of size={self.chunk_size}")
            chunk_list = []
            for chunk in pd.read_csv(
                file_path,
                sep='|',
                quotechar='"',
                doublequote=True,
                engine='python',
                dtype=str,
                chunksize=self.chunk_size,
                on_bad_lines=on_bad
            ):
                chunk_list.append(chunk)
            df = pd.concat(chunk_list, ignore_index=True)
        else:
            # Single shot read
            df = pd.read_csv(
                file_path,
                sep='|',
                quotechar='"',
                doublequote=True,
                engine='python',
                dtype=str,
                on_bad_lines=on_bad
            )

        return df

    def _basic_cleaning(self, df: pd.DataFrame, table_config: dict) -> pd.DataFrame:
        """
        Perform basic cleaning operations:
          - Strips whitespace
          - Converts date columns
          - Convert certain code columns to categorical
        """
        # Strip whitespace from string columns
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            df[col] = df[col].str.strip()

        # Convert date columns
        date_columns = table_config.get('date_columns', [])
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to datetime: {e}")

        # Example: treat columns ending in '_calc' or containing 'Type' as categorical
        for col in df.columns:
            if col.endswith('_calc') or 'Type' in col:
                df[col] = df[col].astype('category')

        return df

    def sample_patients(self, demographic_df: pd.DataFrame, n_patients: int = 2000) -> Tuple[pd.DataFrame, Set[str]]:
        """
        Take a random sample of patients from the PatientDemographic table.

        Args:
            demographic_df: PatientDemographic dataframe
            n_patients: Number of patients to sample

        Returns:
            (sampled_df, set_of_patient_ids)
        """
        self.logger.info(f"Sampling {n_patients} patients from PatientDemographic")
        stratify_columns = self.config['sampling'].get('stratify_columns', [])

        # If all stratify columns exist, do stratified sampling
        if all(col in demographic_df.columns for col in stratify_columns):
            sampled_df = (
                demographic_df
                .groupby(stratify_columns, group_keys=False)
                .apply(
                    lambda x: x.sample(min(len(x), 
                                           int(n_patients * len(x) / len(demographic_df))))
                )
            )
        else:
            # Simple random sample
            sampled_df = demographic_df.sample(n=n_patients)

        sampled_ids = set(sampled_df['Patient_ID'])
        self.logger.info(f"Selected {len(sampled_ids)} unique patients")
        return sampled_df, sampled_ids

    def filter_related_tables(self, df: pd.DataFrame, patient_ids: Set[str]) -> pd.DataFrame:
        """
        Filter any table (by Patient_ID) to include only the sampled patients.

        Args:
            df: input DataFrame
            patient_ids: set of patient IDs to keep

        Returns:
            filtered DataFrame
        """
        if 'Patient_ID' in df.columns:
            return df[df['Patient_ID'].isin(patient_ids)]
        return df

    def assess_data_quality(self, df: pd.DataFrame, table_name: str) -> Dict:
        """
        Assess data quality metrics:
          - Row/column counts
          - Missing percentages
          - Unique counts
          - Date ranges
          - Category distributions
        """
        metrics = {
            'table_name': table_name,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'missing_percentages': df.isnull().mean().to_dict(),
            'unique_counts': df.nunique().to_dict()
        }

        # Date range metrics
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            metrics[f'{col}_range'] = {
                'min': str(df[col].min()),
                'max': str(df[col].max())
            }

        # RefSet/categorical columns
        cat_columns = df.select_dtypes(include=['category']).columns
        for col in cat_columns:
            metrics[f'{col}_distribution'] = df[col].value_counts(dropna=False).to_dict()

        return metrics

    def validate_relationships(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Basic foreign key checks (e.g., that all Patient_IDs in Demographic are in Patient, etc.)
        """
        issues = {}

        # Example: check that all PatientDemographic Patient_IDs appear in Patient
        if 'Patient' in tables and 'PatientDemographic' in tables:
            patient_ids = set(tables['Patient']['Patient_ID'])
            demo_ids = set(tables['PatientDemographic']['Patient_ID'])
            missing_in_patient = demo_ids - patient_ids
            if missing_in_patient:
                issues['PatientDemographic'] = [f"{len(missing_in_patient)} IDs not in Patient table"]

        # Check Encounter vs EncounterDiagnosis
        if 'Encounter' in tables and 'EncounterDiagnosis' in tables:
            encounter_ids = set(tables['Encounter']['Encounter_ID'])
            diag_enc_ids = set(tables['EncounterDiagnosis']['Encounter_ID'])
            missing_diag = diag_enc_ids - encounter_ids
            if missing_diag:
                issues['EncounterDiagnosis'] = [f"{len(missing_diag)} Encounter_IDs not in Encounter table"]

        return issues

    def save_prepared_data(self, dfs: Dict[str, pd.DataFrame], output_dir: str):
        """
        Save dataframes to CSV with UTF-8 encoding.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for name, df in dfs.items():
            file_path = out_path / f"{name}_prepared.csv"
            df.to_csv(file_path, index=False, encoding='utf-8')
            self.logger.info(f"Saved {name} -> {file_path}")

def main():
    """Example main execution."""
    pipeline = DataPreparationPipeline('config.yaml')
    tables = {}

    # Load all tables
    for table_name in pipeline.config['tables'].keys():
        tables[table_name] = pipeline.load_table(table_name)

    # Sample from PatientDemographic
    patdemo_df = tables['PatientDemographic']
    patdemo_df_sample, sampled_ids = pipeline.sample_patients(
        patdemo_df,
        n_patients=pipeline.config['sampling']['n_patients']
    )

    tables['PatientDemographic'] = patdemo_df_sample

    # Filter other tables to the sampled patients
    for tname in tables:
        if tname != 'PatientDemographic':
            tables[tname] = pipeline.filter_related_tables(tables[tname], sampled_ids)

    # Validate relationships
    relationship_issues = pipeline.validate_relationships(tables)
    if relationship_issues:
        pipeline.logger.warning(f"Relationship issues found: {relationship_issues}")

    # Assess data quality
    quality_reports = {}
    for tname, df in tables.items():
        quality_reports[tname] = pipeline.assess_data_quality(df, tname)

    # Save
    pipeline.save_prepared_data(tables, pipeline.config['data_paths']['prepared_data'])

    return tables, quality_reports, relationship_issues


if __name__ == "__main__":
    main()
