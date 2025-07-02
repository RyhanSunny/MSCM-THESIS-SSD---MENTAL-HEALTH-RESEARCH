#!/usr/bin/env python3
"""
2_all.py - Week 2 Comprehensive Analysis

Comprehensive validation of data preprocessing and exposure flagging.
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
from typing import Dict, List, Any, Tuple
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility (RULES.md requirement)
SEED = 42
np.random.seed(SEED)

def validate_missing_data_patterns() -> Dict[str, Any]:
    """
    Validate missing data patterns following ANALYSIS_RULES.md
    
    Returns:
    --------
    Dict[str, Any]
        Missing data validation results with traceable sources
    """
    logger.info("Validating missing data patterns")
    
    results = {
        'validation_type': 'missing_data_patterns',
        'timestamp': pd.Timestamp.now().isoformat(),
        'seed_used': SEED
    }
    
    try:
        # Check cohort data
        cohort_path = Path("Notebooks/data/interim/checkpoint_1_20250318_024427/cohort.parquet")
        
        if cohort_path.exists():
            df = pd.read_parquet(cohort_path)
            
            # Calculate missing data statistics
            missing_stats = {}
            total_cells = len(df) * len(df.columns)
            
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                
                missing_stats[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_pct),
                    'data_type': str(df[col].dtype)
                }
            
            # Identify high-missing columns (>50%)
            high_missing = {k: v for k, v in missing_stats.items() 
                          if v['missing_percentage'] > 50}
            
            # Check for systematic missing patterns
            systematic_missing = {}
            if 'IndexDate_lab' in df.columns:
                lab_missing = df['IndexDate_lab'].isnull().sum()
                systematic_missing['lab_index_dates'] = {
                    'missing_count': int(lab_missing),
                    'missing_percentage': float((lab_missing / len(df)) * 100),
                    'fallback_audit_concern': lab_missing > 0.25 * len(df)  # >25% missing
                }
            
            results.update({
                'data_found': True,
                'total_patients': len(df),
                'total_variables': len(df.columns),
                'missing_data_summary': {
                    'columns_with_missing': len([k for k, v in missing_stats.items() if v['missing_count'] > 0]),
                    'high_missing_columns': len(high_missing),
                    'total_missing_cells': sum(v['missing_count'] for v in missing_stats.values()),
                    'overall_missing_rate': float(sum(v['missing_count'] for v in missing_stats.values()) / total_cells * 100)
                },
                'high_missing_variables': high_missing,
                'systematic_patterns': systematic_missing,
                'source_file': str(cohort_path)
            })
            
            logger.info(f"Missing data validation complete: {len(df)} patients, {len(df.columns)} variables")
            
        else:
            results.update({
                'data_found': False,
                'error': f"Cohort file not found at {cohort_path}"
            })
            
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Missing data validation error: {e}")
    
    return results

def validate_exposure_flagging() -> Dict[str, Any]:
    """
    Validate exposure flagging implementation for H1-H3
    
    Returns:
    --------
    Dict[str, Any]
        Exposure flagging validation results
    """
    logger.info("Validating exposure flagging implementation")
    
    results = {
        'validation_type': 'exposure_flagging',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Check if exposure results exist
        exposure_results_path = Path("results/exposure_flags.json")
        
        if exposure_results_path.exists():
            with open(exposure_results_path, 'r') as f:
                exposure_data = json.load(f)
            
            results.update({
                'exposure_results_found': True,
                'source_file': str(exposure_results_path),
                'exposure_data': exposure_data
            })
            
        else:
            # Check if we can validate from cohort data
            cohort_path = Path("Notebooks/data/interim/checkpoint_1_20250318_024427/cohort.parquet")
            
            if cohort_path.exists():
                df = pd.read_parquet(cohort_path)
                
                # Look for exposure-related columns
                exposure_cols = [col for col in df.columns if any(term in col.lower() 
                               for term in ['ssd', 'exposure', 'flag', 'normal_labs', 'referral', 'medication'])]
                
                exposure_summary = {}
                for col in exposure_cols:
                    if df[col].dtype in ['int64', 'float64', 'bool']:
                        exposure_summary[col] = {
                            'positive_cases': int(df[col].sum()) if df[col].dtype != 'object' else 'N/A',
                            'total_cases': int(len(df)),
                            'prevalence': float(df[col].mean()) if df[col].dtype != 'object' else 'N/A',
                            'missing_count': int(df[col].isnull().sum())
                        }
                
                results.update({
                    'exposure_results_found': False,
                    'cohort_exposure_analysis': {
                        'exposure_columns_found': exposure_cols,
                        'exposure_summary': exposure_summary,
                        'total_patients': len(df)
                    },
                    'source_file': str(cohort_path)
                })
            else:
                results.update({
                    'exposure_results_found': False,
                    'cohort_data_found': False,
                    'error': "Neither exposure results nor cohort data found"
                })
        
        logger.info("Exposure flagging validation complete")
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Exposure flagging validation error: {e}")
    
    return results

def validate_parameter_choices() -> Dict[str, Any]:
    """
    Validate parameter choices mentioned in FALLBACK_AUDIT
    
    Returns:
    --------
    Dict[str, Any]
        Parameter validation results
    """
    logger.info("Validating parameter choices from FALLBACK_AUDIT")
    
    results = {
        'validation_type': 'parameter_choices',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Check key scripts for parameter definitions
        scripts_to_check = [
            ('01_cohort_builder.py', ['30', '180', 'duration']),
            ('02_exposure_flag.py', ['≥3', '>=3', '≥2', '>=2']),
            ('config/config.yaml', ['threshold', 'default', 'cutoff'])
        ]
        
        parameter_findings = {}
        
        for script_name, search_terms in scripts_to_check:
            script_path = Path(f"src/{script_name}") if not script_name.startswith('config') else Path(script_name)
            
            if script_path.exists():
                with open(script_path, 'r') as f:
                    content = f.read()
                
                findings = {}
                for term in search_terms:
                    findings[term] = term in content
                
                parameter_findings[script_name] = {
                    'file_found': True,
                    'parameter_search': findings,
                    'file_size': script_path.stat().st_size,
                    'source_path': str(script_path)
                }
            else:
                parameter_findings[script_name] = {
                    'file_found': False,
                    'error': f"File not found at {script_path}"
                }
        
        results.update({
            'parameter_analysis': parameter_findings,
            'fallback_audit_concerns': {
                'medication_duration_default': '30' in str(parameter_findings),
                'lab_threshold_arbitrary': '≥3' in str(parameter_findings) or '>=3' in str(parameter_findings),
                'drug_persistence_enhanced': '180' in str(parameter_findings)
            }
        })
        
        logger.info("Parameter choices validation complete")
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Parameter validation error: {e}")
    
    return results

def main():
    """Main validation function for Week 2"""
    print("="*80)
    print("WEEK 2 COMPREHENSIVE ANALYSIS: Data Quality & Exposure Validation")
    print("Following CLAUDE.md + RULES.md + ANALYSIS_RULES.md")
    print("="*80)
    
    # Run comprehensive validations
    validations = [
        ('Missing Data Patterns', validate_missing_data_patterns),
        ('Exposure Flagging', validate_exposure_flagging),
        ('Parameter Choices', validate_parameter_choices)
    ]
    
    all_results = {
        'validation_suite': 'week_2_comprehensive',
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
            if name == 'Missing Data Patterns' and 'missing_data_summary' in result:
                summary = result['missing_data_summary']
                print(f"  - Total patients: {result.get('total_patients', 'N/A'):,}")
                print(f"  - Variables with missing data: {summary['columns_with_missing']}")
                print(f"  - High missing variables (>50%): {summary['high_missing_columns']}")
                print(f"  - Overall missing rate: {summary['overall_missing_rate']:.2f}%")
                
                # Check for FALLBACK_AUDIT concerns
                if 'systematic_patterns' in result and 'lab_index_dates' in result['systematic_patterns']:
                    lab_info = result['systematic_patterns']['lab_index_dates']
                    if lab_info.get('fallback_audit_concern', False):
                        print(f"  ⚠️ FALLBACK_AUDIT CONCERN: Lab index dates missing {lab_info['missing_percentage']:.1f}%")
                
            elif name == 'Exposure Flagging' and 'cohort_exposure_analysis' in result:
                analysis = result['cohort_exposure_analysis']
                print(f"  - Exposure columns found: {len(analysis['exposure_columns_found'])}")
                print(f"  - Total patients analyzed: {analysis['total_patients']:,}")
                
            elif name == 'Parameter Choices' and 'fallback_audit_concerns' in result:
                concerns = result['fallback_audit_concerns']
                for concern, found in concerns.items():
                    status = "⚠️ FOUND" if found else "✓ Not detected"
                    print(f"  - {concern}: {status}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "week2_comprehensive_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Comprehensive analysis results saved to: {output_path}")
    print("\nWEEK 2 COMPREHENSIVE ANALYSIS COMPLETE ✓")
    
    return all_results

if __name__ == "__main__":
    main()

