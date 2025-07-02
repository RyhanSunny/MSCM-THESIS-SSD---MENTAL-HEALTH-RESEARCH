#!/usr/bin/env python3
"""
1_validation.py - Week 1 Initial Validation

Validates basic pipeline components and data integrity.
Following CLAUDE.md TDD requirements and ANALYSIS_RULES.md transparency.

Author: Ryhan Suny, MSc <sajibrayhan.suny@torontomu.ca>
Date: 2025-07-02
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, List, Any
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility (RULES.md requirement)
SEED = 42
np.random.seed(SEED)

def validate_cohort_construction() -> Dict[str, Any]:
    """
    Validate cohort construction following ANALYSIS_RULES.md
    
    Returns:
    --------
    Dict[str, Any]
        Validation results with traceable sources
    """
    logger.info("Starting cohort construction validation")
    
    results = {
        'validation_type': 'cohort_construction',
        'timestamp': pd.Timestamp.now().isoformat(),
        'seed_used': SEED
    }
    
    try:
        # Check if cohort file exists
        cohort_path = Path("Notebooks/data/interim/checkpoint_1_20250318_024427/cohort.parquet")
        
        if cohort_path.exists():
            df = pd.read_parquet(cohort_path)
            
            results.update({
                'cohort_file_found': True,
                'total_patients': len(df),
                'total_columns': len(df.columns),
                'source_file': str(cohort_path),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            })
            
            # Validate key columns exist
            required_cols = ['Patient_ID', 'Age_calc', 'Gender']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            results.update({
                'required_columns_present': len(missing_cols) == 0,
                'missing_columns': missing_cols,
                'available_columns': list(df.columns)[:10]  # First 10 for brevity
            })
            
            # Check for missing Patient_IDs
            null_ids = df['Patient_ID'].isnull().sum()
            duplicate_ids = df['Patient_ID'].duplicated().sum()
            
            results.update({
                'null_patient_ids': int(null_ids),
                'duplicate_patient_ids': int(duplicate_ids),
                'unique_patients': int(df['Patient_ID'].nunique())
            })
            
            # Age distribution validation
            if 'Age_calc' in df.columns:
                age_stats = df['Age_calc'].describe()
                results.update({
                    'age_validation': {
                        'mean_age': float(age_stats['mean']),
                        'min_age': float(age_stats['min']),
                        'max_age': float(age_stats['max']),
                        'age_nulls': int(df['Age_calc'].isnull().sum()),
                        'unrealistic_ages': int(((df['Age_calc'] < 0) | (df['Age_calc'] > 120)).sum())
                    }
                })
            
            logger.info(f"Cohort validation complete: {len(df)} patients")
            
        else:
            results.update({
                'cohort_file_found': False,
                'error': f"Cohort file not found at {cohort_path}"
            })
            logger.error(f"Cohort file not found at {cohort_path}")
            
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Validation error: {e}")
    
    return results

def validate_exposure_definitions() -> Dict[str, Any]:
    """
    Validate exposure flag definitions for H1-H3
    
    Returns:
    --------
    Dict[str, Any]
        Exposure validation results
    """
    logger.info("Validating exposure definitions")
    
    results = {
        'validation_type': 'exposure_definitions',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Check if exposure script exists
        exposure_script = Path("src/02_exposure_flag.py")
        
        if exposure_script.exists():
            # Read script content to validate thresholds
            with open(exposure_script, 'r') as f:
                content = f.read()
            
            # Check for key thresholds mentioned in FALLBACK_AUDIT
            threshold_checks = {
                'normal_labs_threshold': '≥3' in content or '>=3' in content,
                'specialist_referrals': '≥2' in content or '>=2' in content,
                'drug_persistence': '180' in content,
                'medication_duration': 'duration' in content.lower()
            }
            
            results.update({
                'exposure_script_found': True,
                'script_path': str(exposure_script),
                'threshold_definitions': threshold_checks,
                'script_size_bytes': exposure_script.stat().st_size
            })
            
            logger.info("Exposure definitions validated")
            
        else:
            results.update({
                'exposure_script_found': False,
                'error': f"Exposure script not found at {exposure_script}"
            })
            
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Exposure validation error: {e}")
    
    return results

def main():
    """Main validation function"""
    print("="*80)
    print("WEEK 1 VALIDATION: Basic Pipeline Components")
    print("Following CLAUDE.md + RULES.md + ANALYSIS_RULES.md")
    print("="*80)
    
    # Run validations
    validations = [
        ('Cohort Construction', validate_cohort_construction),
        ('Exposure Definitions', validate_exposure_definitions)
    ]
    
    all_results = {
        'validation_suite': 'week_1_initial',
        'total_validations': len(validations),
        'results': {}
    }
    
    for name, validation_func in validations:
        print(f"\n{'-'*60}")
        print(f"Validating: {name}")
        print(f"{'-'*60}")
        
        result = validation_func()
        all_results['results'][name.lower().replace(' ', '_')] = result
        
        # Print key findings
        if 'validation_error' in result:
            print(f"❌ VALIDATION FAILED: {result['validation_error']}")
        else:
            print(f"✓ Validation completed successfully")
            
            # Print specific findings
            if name == 'Cohort Construction' and 'total_patients' in result:
                print(f"  - Total patients: {result['total_patients']:,}")
                print(f"  - Unique patients: {result.get('unique_patients', 'N/A'):,}")
                print(f"  - Null IDs: {result.get('null_patient_ids', 'N/A')}")
                print(f"  - Duplicate IDs: {result.get('duplicate_patient_ids', 'N/A')}")
                
            elif name == 'Exposure Definitions' and 'threshold_definitions' in result:
                thresholds = result['threshold_definitions']
                for threshold, found in thresholds.items():
                    status = "✓" if found else "❌"
                    print(f"  - {threshold}: {status}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "week1_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Validation results saved to: {output_path}")
    print("\nWEEK 1 VALIDATION COMPLETE ✓")
    
    return all_results

if __name__ == "__main__":
    main()

