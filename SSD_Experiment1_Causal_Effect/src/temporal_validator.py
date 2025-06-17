#!/usr/bin/env python3
"""
Temporal Ordering Validator Module

Validates temporal ordering for causal inference following Hill's temporality criterion.
Essential for establishing that exposure precedes outcome in observational studies.

Key Features:
- Exposure-outcome temporal sequence validation
- SSD-specific criteria temporal checks
- Cross-dataset temporal consistency validation
- Causal inference temporal requirements
- Comprehensive reporting and error handling

Author: Ryhan Suny
Date: 2025-06-17
Version: 1.0.0

References:
- Hill, A.B. (1965). The environment and disease: association or causation? 
  Proceedings of the Royal Society of Medicine, 58(5), 295-300.
- Hernán, M.A. & Robins, J.M. (2020). Causal Inference: What If. Chapter 7.
- Rothman, K.J. (2012). Epidemiology: An Introduction. Chapter 2.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Union, Optional, List, Tuple
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalValidationError(Exception):
    """Custom exception for temporal validation failures"""
    pass


def create_temporal_windows(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create temporal analysis windows from configuration.
    
    Args:
        config: Configuration dictionary with temporal parameters
        
    Returns:
        Dictionary with parsed temporal windows
    """
    windows = {}
    
    # Parse dates
    for key in ['exposure_window_start', 'exposure_window_end', 
                'outcome_window_start', 'outcome_window_end']:
        if key in config:
            windows[key.replace('_window', '')] = pd.Timestamp(config[key])
    
    # Parse durations
    if 'washout_period_days' in config:
        windows['washout_days'] = config['washout_period_days']
    
    # Validate temporal logic
    if all(k in windows for k in ['exposure_start', 'exposure_end', 'outcome_start', 'outcome_end']):
        if windows['exposure_end'] >= windows['outcome_start']:
            logger.warning("Exposure and outcome windows overlap - may lead to immortal time bias")
    
    return windows


class TemporalOrderingValidator:
    """
    Comprehensive temporal ordering validation for causal inference.
    
    Implements Hill's temporality criterion and best practices for
    observational study temporal validation.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize temporal validator.
        
        Args:
            strict_mode: If True, raises exceptions for violations
        """
        self.strict_mode = strict_mode
        self.validation_results = {}
    
    def validate_exposure_outcome_sequence(
        self, 
        df: pd.DataFrame, 
        exposure_date_col: str, 
        outcome_date_col: str,
        patient_id_col: str = 'Patient_ID'
    ) -> Dict[str, Any]:
        """
        Validate that exposure precedes outcome for all patients.
        
        Args:
            df: DataFrame with patient data
            exposure_date_col: Column name for exposure date
            outcome_date_col: Column name for outcome date
            patient_id_col: Column name for patient identifier
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating temporal sequence: {exposure_date_col} → {outcome_date_col}")
        
        # Validate input columns
        required_cols = [patient_id_col, exposure_date_col, outcome_date_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise TemporalValidationError(f"Missing required columns: {missing_cols}")
        
        # Convert to datetime if not already
        try:
            exposure_dates = pd.to_datetime(df[exposure_date_col])
            outcome_dates = pd.to_datetime(df[outcome_date_col])
        except Exception as e:
            raise TemporalValidationError(f"Could not parse dates: {e}")
        
        # Check for missing dates
        missing_exposure = exposure_dates.isna().sum()
        missing_outcome = outcome_dates.isna().sum()
        
        # Calculate temporal differences (outcome - exposure)
        temporal_diff = outcome_dates - exposure_dates
        
        # Find violations (outcome before exposure)
        violations = temporal_diff < pd.Timedelta(0)
        n_violations = violations.sum()
        
        # Calculate statistics
        n_total = len(df)
        n_valid = n_total - missing_exposure - missing_outcome
        violation_rate = n_violations / n_valid if n_valid > 0 else 0
        
        # Summary statistics for valid cases
        valid_diffs = temporal_diff[~temporal_diff.isna() & ~violations]
        if len(valid_diffs) > 0:
            median_gap_days = valid_diffs.median().days
            min_gap_days = valid_diffs.min().days
            max_gap_days = valid_diffs.max().days
        else:
            median_gap_days = min_gap_days = max_gap_days = None
        
        results = {
            'temporal_ordering_valid': bool(n_violations == 0),
            'n_total': int(n_total),
            'n_valid_pairs': int(n_valid),
            'n_violations': int(n_violations),
            'violation_rate': float(violation_rate),
            'missing_exposure_dates': int(missing_exposure),
            'missing_outcome_dates': int(missing_outcome),
            'median_gap_days': median_gap_days,
            'min_gap_days': min_gap_days,
            'max_gap_days': max_gap_days,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Add violation details if any
        if n_violations > 0:
            violation_patients = df.loc[violations, patient_id_col].tolist()
            results['violation_patient_ids'] = violation_patients[:10]  # First 10 for debugging
            
            logger.warning(f"Found {n_violations} temporal violations ({violation_rate:.1%})")
        else:
            logger.info("✓ All patients have proper temporal ordering")
        
        # Handle strict mode
        if self.strict_mode and n_violations > 0:
            raise TemporalValidationError(
                f"Temporal ordering violations detected: {n_violations} patients "
                f"have outcome before exposure ({violation_rate:.1%})"
            )
        
        return results
    
    def validate_ssd_exposure_sequence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate temporal sequence for SSD-specific exposure criteria.
        
        Tests H1, H2, H3 exposure criteria against outcome windows.
        
        Args:
            df: DataFrame with SSD exposure and outcome date columns
            
        Returns:
            Dictionary with SSD-specific temporal validation results
        """
        logger.info("Validating SSD-specific temporal sequences")
        
        results = {}
        
        # H1: Normal lab cascade temporal validation
        if all(col in df.columns for col in ['last_normal_lab_date', 'outcome_window_start']):
            h1_result = self.validate_exposure_outcome_sequence(
                df, 'last_normal_lab_date', 'outcome_window_start'
            )
            results['h1_temporal_valid'] = h1_result['temporal_ordering_valid']
            results['h1_violations'] = h1_result['n_violations']
            results['h1_median_gap_days'] = h1_result['median_gap_days']
        
        # H2: Referral loop temporal validation
        if all(col in df.columns for col in ['last_referral_date', 'outcome_window_start']):
            h2_result = self.validate_exposure_outcome_sequence(
                df, 'last_referral_date', 'outcome_window_start'
            )
            results['h2_temporal_valid'] = h2_result['temporal_ordering_valid']
            results['h2_violations'] = h2_result['n_violations']
            results['h2_median_gap_days'] = h2_result['median_gap_days']
        
        # H3: Medication persistence temporal validation
        if all(col in df.columns for col in ['medication_end_date', 'outcome_window_start']):
            h3_result = self.validate_exposure_outcome_sequence(
                df, 'medication_end_date', 'outcome_window_start'
            )
            results['h3_temporal_valid'] = h3_result['temporal_ordering_valid']
            results['h3_violations'] = h3_result['n_violations']
            results['h3_median_gap_days'] = h3_result['median_gap_days']
        
        return results
    
    def validate_all_ssd_criteria(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation of all SSD temporal criteria.
        
        Args:
            df: DataFrame with all SSD exposure and outcome data
            
        Returns:
            Comprehensive temporal validation results
        """
        logger.info("Running comprehensive SSD temporal validation")
        
        # Individual hypothesis validations
        ssd_results = self.validate_ssd_exposure_sequence(df)
        
        # Overall temporal validity
        individual_validities = [
            ssd_results.get('h1_temporal_valid', True),
            ssd_results.get('h2_temporal_valid', True), 
            ssd_results.get('h3_temporal_valid', True)
        ]
        
        overall_valid = all(individual_validities)
        
        # Total violations across all hypotheses
        total_violations = sum([
            ssd_results.get('h1_violations', 0),
            ssd_results.get('h2_violations', 0),
            ssd_results.get('h3_violations', 0)
        ])
        
        # Add overall results
        ssd_results.update({
            'overall_temporal_valid': overall_valid,
            'total_violations_all_hypotheses': total_violations,
            'hypotheses_validated': len([h for h in ['h1', 'h2', 'h3'] 
                                        if f'{h}_temporal_valid' in ssd_results])
        })
        
        logger.info(f"Overall SSD temporal validation: {'✓ PASS' if overall_valid else '✗ FAIL'}")
        
        return ssd_results
    
    def check_cross_dataset_consistency(
        self,
        patients_df: pd.DataFrame,
        encounters_df: pd.DataFrame,
        labs_df: pd.DataFrame,
        patient_id_col: str = 'Patient_ID'
    ) -> Dict[str, Any]:
        """
        Check temporal consistency across multiple related datasets.
        
        Args:
            patients_df: Patient demographic data with study dates
            encounters_df: Encounter data with dates
            labs_df: Laboratory data with dates
            patient_id_col: Patient identifier column name
            
        Returns:
            Cross-dataset temporal consistency results
        """
        logger.info("Checking cross-dataset temporal consistency")
        
        results = {}
        
        # Merge datasets to check consistency
        if 'study_entry_date' in patients_df.columns and 'encounter_date' in encounters_df.columns:
            merged = encounters_df.merge(
                patients_df[[patient_id_col, 'study_entry_date', 'last_observation_date']], 
                on=patient_id_col, how='left'
            )
            
            # Check encounters are within study period
            encounter_violations = (
                (merged['encounter_date'] < merged['study_entry_date']) |
                (merged['encounter_date'] > merged['last_observation_date'])
            ).sum()
            
            results['encounter_date_violations'] = int(encounter_violations)
        
        # Similar check for lab dates
        if 'study_entry_date' in patients_df.columns and 'lab_date' in labs_df.columns:
            merged_labs = labs_df.merge(
                patients_df[[patient_id_col, 'study_entry_date', 'last_observation_date']], 
                on=patient_id_col, how='left'
            )
            
            lab_violations = (
                (merged_labs['lab_date'] < merged_labs['study_entry_date']) |
                (merged_labs['lab_date'] > merged_labs['last_observation_date'])
            ).sum()
            
            results['lab_date_violations'] = int(lab_violations)
        
        # Overall consistency
        total_violations = sum([v for k, v in results.items() if 'violations' in k])
        results['consistency_check_passed'] = total_violations == 0
        results['total_cross_dataset_violations'] = total_violations
        
        return results
    
    def validate_causal_inference_sequence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate temporal requirements specific to causal inference.
        
        Implements best practices for temporal validation in causal inference studies.
        
        Args:
            df: DataFrame with exposure and outcome data
            
        Returns:
            Causal inference temporal validation results
        """
        logger.info("Validating causal inference temporal requirements")
        
        results = {}
        
        # Basic exposure → outcome sequence
        if all(col in df.columns for col in ['ssd_exposure_end', 'outcome_measurement_start']):
            basic_result = self.validate_exposure_outcome_sequence(
                df, 'ssd_exposure_end', 'outcome_measurement_start'
            )
            
            results['causal_temporality_valid'] = basic_result['temporal_ordering_valid']
            results['exposure_outcome_gap_days'] = basic_result['median_gap_days']
        
        # Check for sufficient follow-up period
        if all(col in df.columns for col in ['outcome_measurement_start', 'outcome_measurement_end']):
            followup_duration = (
                pd.to_datetime(df['outcome_measurement_end']) - 
                pd.to_datetime(df['outcome_measurement_start'])
            ).dt.days.median()
            
            results['median_followup_days'] = followup_duration
            results['sufficient_followup'] = followup_duration >= 365  # At least 1 year
        
        # Check for immortal time bias potential
        if all(col in df.columns for col in ['ssd_exposure_start', 'outcome_measurement_start']):
            exposure_outcome_overlap = (
                pd.to_datetime(df['ssd_exposure_start']) >= 
                pd.to_datetime(df['outcome_measurement_start'])
            ).any()
            
            results['no_immortal_time_bias'] = not exposure_outcome_overlap
        
        # Overall causal inference temporal validity
        causal_checks = [
            results.get('causal_temporality_valid', True),
            results.get('sufficient_followup', True),
            results.get('no_immortal_time_bias', True)
        ]
        
        results['causal_inference_temporal_valid'] = all(causal_checks)
        
        return results
    
    def validate_master_table_consistency(self, master_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate temporal consistency in the patient master table.
        
        Args:
            master_df: Patient master table with all temporal variables
            
        Returns:
            Master table temporal consistency results
        """
        logger.info("Validating master table temporal consistency")
        
        results = {}
        
        # Check study timeline consistency
        timeline_cols = ['study_entry_date', 'exposure_window_start', 'exposure_window_end',
                        'outcome_window_start', 'outcome_window_end', 'study_exit_date']
        
        available_cols = [col for col in timeline_cols if col in master_df.columns]
        
        if len(available_cols) >= 4:  # Need at least exposure and outcome windows
            # Convert to datetime
            date_cols = {}
            for col in available_cols:
                if master_df[col].dtype == 'object':
                    date_cols[col] = pd.to_datetime(master_df[col])
                else:
                    date_cols[col] = master_df[col]
            
            # Check logical ordering
            timeline_consistent = True
            
            # Exposure window should be before outcome window
            if all(col in date_cols for col in ['exposure_window_end', 'outcome_window_start']):
                exposure_outcome_gap = (
                    date_cols['outcome_window_start'] - date_cols['exposure_window_end']
                ).dt.days.min()
                
                results['exposure_outcome_separation'] = int(exposure_outcome_gap)
                timeline_consistent &= exposure_outcome_gap >= 0
            
            results['study_timeline_consistent'] = timeline_consistent
        
        # Check for patients with complete temporal data
        temporal_completeness = {}
        for col in available_cols:
            missing_count = master_df[col].isna().sum()
            temporal_completeness[f'{col}_missing'] = int(missing_count)
        
        results['temporal_completeness'] = temporal_completeness
        results['master_table_temporal_valid'] = results.get('study_timeline_consistent', True)
        
        return results
    
    def generate_temporal_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive temporal validation report.
        
        Args:
            validation_results: Results from temporal validation
            
        Returns:
            Formatted report string
        """
        report = "Temporal Validation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall status
        overall_valid = validation_results.get('overall_temporal_valid', 
                                             validation_results.get('temporal_ordering_valid', True))
        
        status = "✓ PASS" if overall_valid else "✗ FAIL"
        report += f"Overall Status: {status}\n\n"
        
        # Detailed results
        if 'n_violations' in validation_results:
            report += f"Temporal Violations: {validation_results['n_violations']}\n"
            report += f"Violation Rate: {validation_results.get('violation_rate', 0):.1%}\n"
        
        if 'median_gap_days' in validation_results:
            report += f"Median Exposure-Outcome Gap: {validation_results['median_gap_days']} days\n"
        
        # SSD-specific results
        for hypothesis in ['h1', 'h2', 'h3']:
            valid_key = f'{hypothesis}_temporal_valid'
            if valid_key in validation_results:
                h_status = "✓" if validation_results[valid_key] else "✗"
                violations = validation_results.get(f'{hypothesis}_violations', 0)
                report += f"H{hypothesis[-1]} Temporal Validation: {h_status} ({violations} violations)\n"
        
        # Recommendations
        report += "\nRecommendations:\n"
        if not overall_valid:
            report += "- Review and correct temporal ordering violations\n"
            report += "- Ensure exposure criteria are met before outcome measurement\n"
            report += "- Consider excluding patients with temporal violations\n"
        else:
            report += "- Temporal validation passed - proceed with causal analysis\n"
        
        return report


def validate_exposure_outcome_sequence(
    df: pd.DataFrame, 
    exposure_col: str, 
    outcome_col: str,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for basic temporal validation.
    
    Args:
        df: DataFrame with temporal data
        exposure_col: Exposure date column
        outcome_col: Outcome date column
        strict: Whether to raise exceptions for violations
        
    Returns:
        Validation results dictionary
    """
    validator = TemporalOrderingValidator(strict_mode=strict)
    return validator.validate_exposure_outcome_sequence(df, exposure_col, outcome_col)


def check_temporal_consistency(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Check temporal consistency across multiple datasets.
    
    Args:
        data_dict: Dictionary of dataset_name -> DataFrame
        
    Returns:
        Cross-dataset temporal consistency results
    """
    validator = TemporalOrderingValidator()
    
    # Extract main datasets
    patients = data_dict.get('patients')
    encounters = data_dict.get('encounters') 
    labs = data_dict.get('labs')
    
    if all(df is not None for df in [patients, encounters, labs]):
        return validator.check_cross_dataset_consistency(patients, encounters, labs)
    else:
        logger.warning("Missing required datasets for cross-consistency check")
        return {'consistency_check_passed': None, 'reason': 'insufficient_data'}


def main():
    """CLI interface for temporal validation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Temporal ordering validation for causal inference"
    )
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with simulated SSD data')
    
    args = parser.parse_args()
    
    if args.demo:
        print("Generating SSD temporal validation demonstration...")
        
        np.random.seed(42)
        n = 1000
        
        # Generate SSD analysis timeline
        df = pd.DataFrame({
            'Patient_ID': range(n),
            
            # Study timeline
            'study_entry_date': pd.date_range('2014-06-01', '2014-12-31', periods=n),
            'exposure_window_start': pd.Timestamp('2015-01-01'),
            'exposure_window_end': pd.Timestamp('2015-12-31'),
            'outcome_window_start': pd.Timestamp('2016-01-01'),
            'outcome_window_end': pd.Timestamp('2017-12-31'),
            
            # H1: Normal lab exposure
            'first_normal_lab_date': pd.date_range('2015-02-01', '2015-08-31', periods=n),
            'last_normal_lab_date': pd.date_range('2015-06-01', '2015-11-30', periods=n),
            'h1_normal_lab_flag': np.random.binomial(1, 0.4, n),
            
            # Outcomes
            'total_encounters': np.random.poisson(8, n),
            'site_id': np.random.randint(1, 21, n)
        })
        
        print(f"Generated {n} patients for temporal validation")
        
        # Run comprehensive validation
        validator = TemporalOrderingValidator(strict_mode=False)
        
        print("\n1. Basic exposure-outcome validation...")
        basic_result = validator.validate_exposure_outcome_sequence(
            df, 'last_normal_lab_date', 'outcome_window_start'
        )
        print(f"   Result: {'✓ PASS' if basic_result['temporal_ordering_valid'] else '✗ FAIL'}")
        print(f"   Violations: {basic_result['n_violations']}")
        print(f"   Median gap: {basic_result['median_gap_days']} days")
        
        print("\n2. SSD-specific validation...")
        ssd_result = validator.validate_ssd_exposure_sequence(df)
        print(f"   H1 temporal validity: {'✓' if ssd_result.get('h1_temporal_valid', True) else '✗'}")
        
        print("\n3. Master table consistency...")
        master_result = validator.validate_master_table_consistency(df)
        print(f"   Master table valid: {'✓' if master_result['master_table_temporal_valid'] else '✗'}")
        
        # Generate report
        print("\n" + "="*50)
        print(validator.generate_temporal_report(basic_result))
        
        print("\n✓ Temporal validation demonstration completed successfully")


if __name__ == "__main__":
    main()