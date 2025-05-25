#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lab_utils.py - Laboratory result utilities

Provides functions for determining if lab results are within normal ranges.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Optional
import re


def is_normal_lab(row: pd.Series) -> bool:
    """
    Check if lab result is within normal range
    
    Args:
        row: Pandas Series containing lab test data with columns:
             - TestResult_calc: The numeric test result
             - LowerNormal: Lower bound of normal range
             - UpperNormal: Upper bound of normal range
             - TestName_calc: Name of the test (optional, for assay-specific logic)
             
    Returns:
        bool: True if result is within normal range, False otherwise
    """
    try:
        # Try to convert result to float
        result = float(row.get('TestResult_calc', np.nan))
        
        # Check if we have explicit normal ranges
        if pd.notna(row.get('LowerNormal')) and pd.notna(row.get('UpperNormal')):
            lower = float(row['LowerNormal'])
            upper = float(row['UpperNormal'])
            return lower <= result <= upper
        
        # Assay-specific logic for common tests without explicit ranges
        test_name = str(row.get('TestName_calc', '')).upper()
        
        # Add common test-specific normal ranges
        if 'GLUCOSE' in test_name and 'FASTING' in test_name:
            return 3.9 <= result <= 5.6  # mmol/L
        elif 'HEMOGLOBIN' in test_name or 'HGB' in test_name:
            # Assuming adult ranges
            if 'sex' in row and row['sex'] == 'M':
                return 130 <= result <= 170  # g/L
            else:
                return 120 <= result <= 150  # g/L
        elif 'TSH' in test_name:
            return 0.4 <= result <= 4.0  # mIU/L
        elif 'CHOLESTEROL' in test_name and 'TOTAL' in test_name:
            return result < 5.2  # mmol/L
        elif 'HDL' in test_name:
            return result > 1.0  # mmol/L
        elif 'LDL' in test_name:
            return result < 3.4  # mmol/L
        elif 'TRIGLYCERIDE' in test_name:
            return result < 1.7  # mmol/L
        elif 'CREATININE' in test_name:
            if 'sex' in row and row['sex'] == 'M':
                return 60 <= result <= 110  # umol/L
            else:
                return 45 <= result <= 90  # umol/L
        elif 'EGFR' in test_name or 'GFR' in test_name:
            return result >= 60  # mL/min/1.73m²
        elif 'ALT' in test_name:
            return result <= 40  # U/L
        elif 'AST' in test_name:
            return result <= 35  # U/L
        elif 'BILIRUBIN' in test_name and 'TOTAL' in test_name:
            return result <= 21  # umol/L
        elif 'ALBUMIN' in test_name:
            return 35 <= result <= 50  # g/L
        elif 'SODIUM' in test_name or 'NA' == test_name:
            return 135 <= result <= 145  # mmol/L
        elif 'POTASSIUM' in test_name or 'K' == test_name:
            return 3.5 <= result <= 5.0  # mmol/L
        elif 'CHLORIDE' in test_name or 'CL' == test_name:
            return 98 <= result <= 107  # mmol/L
        elif 'CALCIUM' in test_name or 'CA' == test_name:
            return 2.15 <= result <= 2.55  # mmol/L
        elif 'MAGNESIUM' in test_name or 'MG' == test_name:
            return 0.70 <= result <= 1.00  # mmol/L
        elif 'PHOSPHATE' in test_name:
            return 0.80 <= result <= 1.45  # mmol/L
        elif 'WBC' in test_name or 'WHITE' in test_name:
            return 4.0 <= result <= 11.0  # x10^9/L
        elif 'PLATELET' in test_name or 'PLT' in test_name:
            return 150 <= result <= 400  # x10^9/L
        elif 'HBA1C' in test_name or 'A1C' in test_name:
            return result <= 6.0  # %
        elif 'INR' in test_name:
            return 0.8 <= result <= 1.2
        elif 'PTT' in test_name or 'APTT' in test_name:
            return 25 <= result <= 35  # seconds
        elif 'FERRITIN' in test_name:
            if 'sex' in row and row['sex'] == 'M':
                return 30 <= result <= 400  # ug/L
            else:
                return 15 <= result <= 200  # ug/L
        elif 'B12' in test_name or 'COBALAMIN' in test_name:
            return 150 <= result <= 700  # pmol/L
        elif 'FOLATE' in test_name:
            return result >= 7  # nmol/L
        elif 'VITAMIN D' in test_name or '25-OH' in test_name:
            return result >= 75  # nmol/L
        elif 'CRP' in test_name:
            return result < 10  # mg/L
        elif 'ESR' in test_name:
            # Age-dependent, using simplified criteria
            return result < 20  # mm/hr
            
        # If no specific logic applies, we can't determine normality
        return False
        
    except (ValueError, TypeError):
        # If we can't parse the result as a number, it's not normal
        return False


def categorize_lab_result(row: pd.Series) -> str:
    """
    Categorize lab result as low, normal, or high
    
    Args:
        row: Pandas Series containing lab test data
        
    Returns:
        str: 'low', 'normal', 'high', or 'unknown'
    """
    try:
        result = float(row.get('TestResult_calc', np.nan))
        
        if pd.notna(row.get('LowerNormal')) and pd.notna(row.get('UpperNormal')):
            lower = float(row['LowerNormal'])
            upper = float(row['UpperNormal'])
            
            if result < lower:
                return 'low'
            elif result > upper:
                return 'high'
            else:
                return 'normal'
        
        # Check if it's normal using our function
        if is_normal_lab(row):
            return 'normal'
        
        # Can't determine
        return 'unknown'
        
    except (ValueError, TypeError):
        return 'unknown'


def extract_numeric_result(result_text: str) -> Optional[float]:
    """
    Extract numeric value from lab result text
    
    Args:
        result_text: Raw lab result text (may contain units, qualifiers, etc.)
        
    Returns:
        float or None: Extracted numeric value
    """
    if pd.isna(result_text):
        return None
    
    # Convert to string
    result_text = str(result_text)
    
    # Remove common qualifiers
    result_text = re.sub(r'[<>≤≥]', '', result_text)
    
    # Try to find first number (including decimals)
    match = re.search(r'[-+]?\d*\.?\d+', result_text)
    
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    
    return None


def standardize_units(value: float, unit: str, test_name: str) -> tuple[float, str]:
    """
    Standardize lab values to common units
    
    Args:
        value: Numeric lab value
        unit: Current unit
        test_name: Name of the test
        
    Returns:
        tuple: (converted_value, standard_unit)
    """
    unit = str(unit).lower().strip()
    test_name = str(test_name).upper()
    
    # Glucose conversions
    if 'GLUCOSE' in test_name:
        if unit in ['mg/dl', 'mg/l']:
            # Convert to mmol/L
            return value * 0.0555, 'mmol/L'
    
    # Creatinine conversions
    elif 'CREATININE' in test_name:
        if unit in ['mg/dl', 'mg/l']:
            # Convert to umol/L
            return value * 88.4, 'umol/L'
    
    # Cholesterol conversions
    elif any(x in test_name for x in ['CHOLESTEROL', 'HDL', 'LDL']):
        if unit in ['mg/dl', 'mg/l']:
            # Convert to mmol/L
            return value * 0.0259, 'mmol/L'
    
    # Triglyceride conversions
    elif 'TRIGLYCERIDE' in test_name:
        if unit in ['mg/dl', 'mg/l']:
            # Convert to mmol/L
            return value * 0.0113, 'mmol/L'
    
    # No conversion needed
    return value, unit


# Batch processing function for DataFrames
def add_normal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normal/abnormal flags to a lab results DataFrame
    
    Args:
        df: DataFrame with lab results
        
    Returns:
        DataFrame with added 'is_normal' and 'result_category' columns
    """
    df = df.copy()
    
    # Extract numeric results if needed
    if 'TestResult_numeric' not in df.columns:
        df['TestResult_numeric'] = df['TestResult_calc'].apply(extract_numeric_result)
    
    # Add normal flag
    df['is_normal'] = df.apply(is_normal_lab, axis=1)
    
    # Add category
    df['result_category'] = df.apply(categorize_lab_result, axis=1)
    
    return df


if __name__ == "__main__":
    # Test the functions
    test_data = pd.DataFrame({
        'TestResult_calc': [5.0, 12.0, 140, 4.5],
        'LowerNormal': [3.9, 10.0, 135, None],
        'UpperNormal': [5.6, 15.0, 145, None],
        'TestName_calc': ['Glucose Fasting', 'Hemoglobin', 'Sodium', 'TSH']
    })
    
    test_data['is_normal'] = test_data.apply(is_normal_lab, axis=1)
    test_data['category'] = test_data.apply(categorize_lab_result, axis=1)
    
    print("Lab utility test results:")
    print(test_data)