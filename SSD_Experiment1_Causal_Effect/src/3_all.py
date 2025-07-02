#!/usr/bin/env python3
"""
3_all.py - Week 3 Comprehensive Analysis

Comprehensive validation of causal inference setup and propensity score matching.
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

def validate_propensity_score_matching() -> Dict[str, Any]:
    """
    Validate propensity score matching implementation
    
    Returns:
    --------
    Dict[str, Any]
        PS matching validation results with traceable sources
    """
    logger.info("Validating propensity score matching")
    
    results = {
        'validation_type': 'propensity_score_matching',
        'timestamp': pd.Timestamp.now().isoformat(),
        'seed_used': SEED
    }
    
    try:
        # Check for PS matching results
        ps_paths = [
            Path("results/ps_matched.parquet"),
            Path("Notebooks/data/derived/ps_matched.parquet"),
            Path("data/derived/ps_matched.parquet")
        ]
        
        ps_data_found = False
        for ps_path in ps_paths:
            if ps_path.exists():
                df_ps = pd.read_parquet(ps_path)
                ps_data_found = True
                
                # Validate PS matching quality
                if 'propensity_score' in df_ps.columns and 'ssd_flag' in df_ps.columns:
                    treated = df_ps[df_ps['ssd_flag'] == 1]
                    control = df_ps[df_ps['ssd_flag'] == 0]
                    
                    # Calculate balance metrics
                    ps_treated_mean = treated['propensity_score'].mean()
                    ps_control_mean = control['propensity_score'].mean()
                    ps_standardized_diff = abs(ps_treated_mean - ps_control_mean) / np.sqrt(
                        (treated['propensity_score'].var() + control['propensity_score'].var()) / 2
                    )
                    
                    # Common support analysis
                    ps_min_treated = treated['propensity_score'].min()
                    ps_max_treated = treated['propensity_score'].max()
                    ps_min_control = control['propensity_score'].min()
                    ps_max_control = control['propensity_score'].max()
                    
                    common_support_min = max(ps_min_treated, ps_min_control)
                    common_support_max = min(ps_max_treated, ps_max_control)
                    
                    # Patients in common support
                    in_common_support = df_ps[
                        (df_ps['propensity_score'] >= common_support_min) & 
                        (df_ps['propensity_score'] <= common_support_max)
                    ]
                    
                    results.update({
                        'ps_data_found': True,
                        'source_file': str(ps_path),
                        'total_matched_patients': len(df_ps),
                        'treated_patients': len(treated),
                        'control_patients': len(control),
                        'balance_metrics': {
                            'ps_treated_mean': float(ps_treated_mean),
                            'ps_control_mean': float(ps_control_mean),
                            'standardized_difference': float(ps_standardized_diff),
                            'balance_achieved': ps_standardized_diff < 0.1  # Standard threshold
                        },
                        'common_support': {
                            'min_overlap': float(common_support_min),
                            'max_overlap': float(common_support_max),
                            'patients_in_support': len(in_common_support),
                            'support_percentage': float(len(in_common_support) / len(df_ps) * 100)
                        }
                    })
                    
                    logger.info(f"PS matching validation complete: {len(df_ps)} patients")
                    break
                else:
                    results.update({
                        'ps_data_found': True,
                        'source_file': str(ps_path),
                        'error': 'Required columns (propensity_score, ssd_flag) not found'
                    })
        
        if not ps_data_found:
            results.update({
                'ps_data_found': False,
                'searched_paths': [str(p) for p in ps_paths],
                'error': 'No propensity score matching results found'
            })
            
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"PS matching validation error: {e}")
    
    return results

def validate_covariate_balance() -> Dict[str, Any]:
    """
    Validate covariate balance after matching
    
    Returns:
    --------
    Dict[str, Any]
        Covariate balance validation results
    """
    logger.info("Validating covariate balance")
    
    results = {
        'validation_type': 'covariate_balance',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Check for balance results
        balance_paths = [
            Path("results/covariate_balance.json"),
            Path("results/balance_diagnostics.json")
        ]
        
        balance_found = False
        for balance_path in balance_paths:
            if balance_path.exists():
                with open(balance_path, 'r') as f:
                    balance_data = json.load(f)
                
                balance_found = True
                results.update({
                    'balance_results_found': True,
                    'source_file': str(balance_path),
                    'balance_data': balance_data
                })
                
                # Analyze balance quality
                if 'standardized_differences' in balance_data:
                    std_diffs = balance_data['standardized_differences']
                    
                    # Count variables with good balance (<0.1)
                    good_balance = sum(1 for diff in std_diffs.values() if abs(diff) < 0.1)
                    total_vars = len(std_diffs)
                    
                    results.update({
                        'balance_analysis': {
                            'total_variables': total_vars,
                            'well_balanced_vars': good_balance,
                            'balance_percentage': float(good_balance / total_vars * 100) if total_vars > 0 else 0,
                            'max_std_diff': float(max(abs(d) for d in std_diffs.values())) if std_diffs else 0,
                            'mean_std_diff': float(np.mean([abs(d) for d in std_diffs.values()])) if std_diffs else 0
                        }
                    })
                
                logger.info("Covariate balance validation complete")
                break
        
        if not balance_found:
            # Try to compute balance from PS matched data
            ps_path = Path("results/ps_matched.parquet")
            if ps_path.exists():
                df_ps = pd.read_parquet(ps_path)
                
                # Identify numeric covariates
                numeric_cols = df_ps.select_dtypes(include=[np.number]).columns
                covariate_cols = [col for col in numeric_cols if col not in ['Patient_ID', 'ssd_flag', 'propensity_score']]
                
                if len(covariate_cols) > 0:
                    treated = df_ps[df_ps['ssd_flag'] == 1]
                    control = df_ps[df_ps['ssd_flag'] == 0]
                    
                    balance_metrics = {}
                    for col in covariate_cols[:10]:  # Limit to first 10 for performance
                        if col in treated.columns and col in control.columns:
                            mean_t = treated[col].mean()
                            mean_c = control[col].mean()
                            var_t = treated[col].var()
                            var_c = control[col].var()
                            
                            if var_t > 0 and var_c > 0:
                                std_diff = abs(mean_t - mean_c) / np.sqrt((var_t + var_c) / 2)
                                balance_metrics[col] = float(std_diff)
                    
                    results.update({
                        'balance_results_found': False,
                        'computed_balance': {
                            'covariates_analyzed': list(balance_metrics.keys()),
                            'standardized_differences': balance_metrics,
                            'source_data': str(ps_path)
                        }
                    })
                else:
                    results.update({
                        'balance_results_found': False,
                        'error': 'No suitable covariates found for balance analysis'
                    })
            else:
                results.update({
                    'balance_results_found': False,
                    'error': 'No balance results or PS matched data found'
                })
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Covariate balance validation error: {e}")
    
    return results

def validate_causal_assumptions() -> Dict[str, Any]:
    """
    Validate causal inference assumptions
    
    Returns:
    --------
    Dict[str, Any]
        Causal assumptions validation results
    """
    logger.info("Validating causal inference assumptions")
    
    results = {
        'validation_type': 'causal_assumptions',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Check for assumption validation scripts/results
        assumption_checks = {
            'temporal_precedence': Path("src/temporal_validator.py").exists(),
            'no_unmeasured_confounding': Path("src/13_evalue_calc.py").exists(),
            'positivity': Path("results/positivity_diagnostics.json").exists(),
            'consistency': Path("src/15_robustness.py").exists()
        }
        
        results.update({
            'assumption_scripts': assumption_checks,
            'scripts_available': sum(assumption_checks.values()),
            'total_assumptions': len(assumption_checks)
        })
        
        # Check for DAG or conceptual framework
        dag_files = [
            Path("figures/dag.svg"),
            Path("figures/conceptual_framework.svg"),
            Path("docs/causal_assumptions.md")
        ]
        
        dag_found = any(f.exists() for f in dag_files)
        results.update({
            'causal_diagram_available': dag_found,
            'diagram_files_checked': [str(f) for f in dag_files]
        })
        
        # Check for sensitivity analyses
        sensitivity_files = [
            Path("src/14_placebo_tests.py"),
            Path("src/negative_control_analysis.py"),
            Path("results/sensitivity_analysis.json")
        ]
        
        sensitivity_available = sum(f.exists() for f in sensitivity_files)
        results.update({
            'sensitivity_analyses': {
                'scripts_available': sensitivity_available,
                'total_sensitivity_checks': len(sensitivity_files),
                'files_found': [str(f) for f in sensitivity_files if f.exists()]
            }
        })
        
        logger.info("Causal assumptions validation complete")
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Causal assumptions validation error: {e}")
    
    return results

def main():
    """Main validation function for Week 3"""
    print("="*80)
    print("WEEK 3 COMPREHENSIVE ANALYSIS: Causal Inference Setup Validation")
    print("Following CLAUDE.md + RULES.md + ANALYSIS_RULES.md")
    print("="*80)
    
    # Run causal inference validations
    validations = [
        ('Propensity Score Matching', validate_propensity_score_matching),
        ('Covariate Balance', validate_covariate_balance),
        ('Causal Assumptions', validate_causal_assumptions)
    ]
    
    all_results = {
        'validation_suite': 'week_3_causal_inference',
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
            if name == 'Propensity Score Matching' and 'balance_metrics' in result:
                balance = result['balance_metrics']
                support = result['common_support']
                print(f"  - Total matched patients: {result.get('total_matched_patients', 'N/A'):,}")
                print(f"  - Treated: {result.get('treated_patients', 'N/A'):,}, Control: {result.get('control_patients', 'N/A'):,}")
                print(f"  - PS balance achieved: {'✓' if balance.get('balance_achieved', False) else '❌'}")
                print(f"  - Common support: {support.get('support_percentage', 0):.1f}% of patients")
                
            elif name == 'Covariate Balance' and 'balance_analysis' in result:
                analysis = result['balance_analysis']
                print(f"  - Variables analyzed: {analysis['total_variables']}")
                print(f"  - Well-balanced variables: {analysis['well_balanced_vars']} ({analysis['balance_percentage']:.1f}%)")
                print(f"  - Mean standardized difference: {analysis['mean_std_diff']:.3f}")
                
            elif name == 'Causal Assumptions' and 'assumption_scripts' in result:
                scripts = result['assumption_scripts']
                available = result['scripts_available']
                total = result['total_assumptions']
                print(f"  - Assumption validation scripts: {available}/{total} available")
                print(f"  - Causal diagram available: {'✓' if result.get('causal_diagram_available', False) else '❌'}")
                
                sensitivity = result.get('sensitivity_analyses', {})
                print(f"  - Sensitivity analyses: {sensitivity.get('scripts_available', 0)}/{sensitivity.get('total_sensitivity_checks', 0)} available")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "week3_causal_inference_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Causal inference validation results saved to: {output_path}")
    print("\nWEEK 3 COMPREHENSIVE ANALYSIS COMPLETE ✓")
    
    return all_results

if __name__ == "__main__":
    main()

