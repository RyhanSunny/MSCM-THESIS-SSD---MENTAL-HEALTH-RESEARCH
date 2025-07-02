#!/usr/bin/env python3
"""
4_all.py - Week 4 Comprehensive Analysis

Comprehensive validation of causal estimation results and statistical refinements.
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

def validate_causal_estimation_results() -> Dict[str, Any]:
    """
    Validate causal estimation results across all methods
    
    Returns:
    --------
    Dict[str, Any]
        Causal estimation validation results with traceable sources
    """
    logger.info("Validating causal estimation results")
    
    results = {
        'validation_type': 'causal_estimation_results',
        'timestamp': pd.Timestamp.now().isoformat(),
        'seed_used': SEED
    }
    
    try:
        # Check for causal estimation results
        causal_paths = [
            Path("results/imputed_causal_results"),
            Path("results/pooled_causal_estimates.json"),
            Path("results/causal_estimates.json")
        ]
        
        estimation_results = {}
        
        # Check imputed results directory
        imputed_dir = Path("results/imputed_causal_results")
        if imputed_dir.exists():
            result_files = list(imputed_dir.glob("causal_results_imp*.json"))
            error_files = list(imputed_dir.glob("causal_error_imp*.txt"))
            
            estimation_results['imputed_results'] = {
                'directory_found': True,
                'successful_imputations': len(result_files),
                'failed_imputations': len(error_files),
                'total_expected': 30,  # Standard number of imputations
                'success_rate': float(len(result_files) / 30 * 100) if len(result_files) > 0 else 0,
                'source_directory': str(imputed_dir)
            }
            
            # Analyze first successful result for methods
            if result_files:
                with open(result_files[0], 'r') as f:
                    first_result = json.load(f)
                
                methods_found = []
                if 'estimates' in first_result:
                    methods_found = [est.get('method', 'Unknown') for est in first_result['estimates']]
                
                estimation_results['imputed_results']['methods_available'] = methods_found
                estimation_results['imputed_results']['sample_file'] = str(result_files[0])
        else:
            estimation_results['imputed_results'] = {
                'directory_found': False,
                'error': f"Imputed results directory not found at {imputed_dir}"
            }
        
        # Check pooled results
        pooled_path = Path("results/pooled_causal_estimates.json")
        if pooled_path.exists():
            with open(pooled_path, 'r') as f:
                pooled_data = json.load(f)
            
            # Analyze pooled results structure
            hypotheses_found = list(pooled_data.keys()) if isinstance(pooled_data, dict) else []
            
            estimation_results['pooled_results'] = {
                'file_found': True,
                'hypotheses_analyzed': hypotheses_found,
                'total_hypotheses': len(hypotheses_found),
                'rubin_pooling_applied': 'df_barnard_rubin' in str(pooled_data),
                'fdr_correction_applied': pooled_data.get('fdr_correction_applied', False),
                'source_file': str(pooled_path)
            }
            
            # Check for key statistical components
            statistical_components = {
                'confidence_intervals': any('ci_lower' in str(h) for h in pooled_data.values()),
                'effect_estimates': any('ate' in str(h) for h in pooled_data.values()),
                'standard_errors': any('se' in str(h) for h in pooled_data.values()),
                'degrees_freedom': any('df' in str(h) for h in pooled_data.values())
            }
            
            estimation_results['pooled_results']['statistical_components'] = statistical_components
        else:
            estimation_results['pooled_results'] = {
                'file_found': False,
                'error': f"Pooled results not found at {pooled_path}"
            }
        
        results.update({'causal_estimation_analysis': estimation_results})
        logger.info("Causal estimation validation complete")
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Causal estimation validation error: {e}")
    
    return results

def validate_statistical_refinements() -> Dict[str, Any]:
    """
    Validate statistical refinements implementation
    
    Returns:
    --------
    Dict[str, Any]
        Statistical refinements validation results
    """
    logger.info("Validating statistical refinements")
    
    results = {
        'validation_type': 'statistical_refinements',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Check for statistical refinement results
        refinement_paths = [
            Path("results/fdr_adjusted_results.json"),
            Path("results/week4_statistical_results.json"),
            Path("results/evalue_results.json")
        ]
        
        refinement_analysis = {}
        
        # Check FDR correction results
        fdr_path = Path("results/fdr_adjusted_results.json")
        if fdr_path.exists():
            with open(fdr_path, 'r') as f:
                fdr_data = json.load(f)
            
            refinement_analysis['fdr_correction'] = {
                'results_found': True,
                'source_file': str(fdr_path),
                'benjamini_hochberg_applied': 'fdr_adjusted' in fdr_data,
                'hypotheses_tested': len(fdr_data.get('fdr_adjusted', {}).get('original_pvalues', [])),
                'significant_after_fdr': fdr_data.get('fdr_adjusted', {}).get('n_significant', 0),
                'fdr_threshold': fdr_data.get('fdr_adjusted', {}).get('fdr_threshold', 'N/A')
            }
            
            # Check for E-values
            if 'evalues' in fdr_data:
                evalues = fdr_data['evalues']
                refinement_analysis['evalue_analysis'] = {
                    'evalues_computed': True,
                    'hypotheses_with_evalues': len(evalues),
                    'strong_evidence_count': sum(1 for h, data in evalues.items() if data.get('evalue', 0) >= 2.0),
                    'moderate_evidence_count': sum(1 for h, data in evalues.items() if 1.5 <= data.get('evalue', 0) < 2.0)
                }
        else:
            refinement_analysis['fdr_correction'] = {
                'results_found': False,
                'error': f"FDR results not found at {fdr_path}"
            }
        
        # Check for week4 statistical refinements script
        week4_script = Path("src/week4_statistical_refinements.py")
        if week4_script.exists():
            with open(week4_script, 'r') as f:
                script_content = f.read()
            
            # Check for key statistical methods
            methods_implemented = {
                'benjamini_hochberg': 'benjamini_hochberg' in script_content.lower(),
                'evalue_computation': 'evalue' in script_content.lower(),
                'bootstrap_methods': 'bootstrap' in script_content.lower(),
                'weight_diagnostics': 'weight' in script_content.lower()
            }
            
            refinement_analysis['week4_script'] = {
                'script_found': True,
                'source_file': str(week4_script),
                'methods_implemented': methods_implemented,
                'total_methods': sum(methods_implemented.values()),
                'script_size': week4_script.stat().st_size
            }
        else:
            refinement_analysis['week4_script'] = {
                'script_found': False,
                'error': f"Week4 script not found at {week4_script}"
            }
        
        results.update({'statistical_refinements_analysis': refinement_analysis})
        logger.info("Statistical refinements validation complete")
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Statistical refinements validation error: {e}")
    
    return results

def validate_hypothesis_testing() -> Dict[str, Any]:
    """
    Validate hypothesis testing framework for H1-H6
    
    Returns:
    --------
    Dict[str, Any]
        Hypothesis testing validation results
    """
    logger.info("Validating hypothesis testing framework")
    
    results = {
        'validation_type': 'hypothesis_testing',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Define expected hypotheses
        expected_hypotheses = [
            'H1: Normal Labs → Healthcare Utilization',
            'H2: Enhanced SSD → Healthcare Utilization', 
            'H3: Medication Persistence → ED Visits',
            'H4: SSDSI Mediation ≥55%',
            'H5: Sequential Diagnostic Pathway',
            'H6: Cost-Effectiveness'
        ]
        
        hypothesis_analysis = {
            'expected_hypotheses': expected_hypotheses,
            'total_expected': len(expected_hypotheses)
        }
        
        # Check pooled results for hypothesis coverage
        pooled_path = Path("results/pooled_causal_estimates.json")
        if pooled_path.exists():
            with open(pooled_path, 'r') as f:
                pooled_data = json.load(f)
            
            # Map found results to hypotheses
            found_hypotheses = list(pooled_data.keys()) if isinstance(pooled_data, dict) else []
            
            # Check for hypothesis-specific patterns
            h1_found = any('normal_lab' in h.lower() or 'h1' in h.lower() for h in found_hypotheses)
            h2_found = any('enhanced' in h.lower() or 'h2' in h.lower() for h in found_hypotheses)
            h3_found = any('medication' in h.lower() or 'persistence' in h.lower() or 'h3' in h.lower() for h in found_hypotheses)
            
            hypothesis_coverage = {
                'H1_normal_labs': h1_found,
                'H2_enhanced_ssd': h2_found,
                'H3_medication_persistence': h3_found,
                'total_found': len(found_hypotheses),
                'found_hypothesis_keys': found_hypotheses
            }
            
            hypothesis_analysis.update({
                'pooled_results_available': True,
                'hypothesis_coverage': hypothesis_coverage,
                'coverage_rate': sum([h1_found, h2_found, h3_found]) / 3 * 100
            })
        else:
            hypothesis_analysis.update({
                'pooled_results_available': False,
                'error': 'Pooled results not available for hypothesis validation'
            })
        
        # Check for mediation analysis (H4)
        mediation_path = Path("results/mediation_results.json")
        if mediation_path.exists():
            with open(mediation_path, 'r') as f:
                mediation_data = json.load(f)
            
            hypothesis_analysis['h4_mediation'] = {
                'mediation_results_found': True,
                'proportion_mediated': mediation_data.get('proportion_mediated', 'N/A'),
                'h4_threshold_met': mediation_data.get('proportion_mediated', 0) >= 0.55,
                'bootstrap_ci_available': 'bootstrap_ci' in mediation_data,
                'source_file': str(mediation_path)
            }
        else:
            hypothesis_analysis['h4_mediation'] = {
                'mediation_results_found': False,
                'error': f"Mediation results not found at {mediation_path}"
            }
        
        results.update({'hypothesis_testing_analysis': hypothesis_analysis})
        logger.info("Hypothesis testing validation complete")
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Hypothesis testing validation error: {e}")
    
    return results

def main():
    """Main validation function for Week 4"""
    print("="*80)
    print("WEEK 4 COMPREHENSIVE ANALYSIS: Causal Results & Statistical Refinements")
    print("Following CLAUDE.md + RULES.md + ANALYSIS_RULES.md")
    print("="*80)
    
    # Run causal results validations
    validations = [
        ('Causal Estimation Results', validate_causal_estimation_results),
        ('Statistical Refinements', validate_statistical_refinements),
        ('Hypothesis Testing Framework', validate_hypothesis_testing)
    ]
    
    all_results = {
        'validation_suite': 'week_4_causal_results',
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
            if name == 'Causal Estimation Results' and 'causal_estimation_analysis' in result:
                analysis = result['causal_estimation_analysis']
                
                if 'imputed_results' in analysis:
                    imputed = analysis['imputed_results']
                    if imputed.get('directory_found', False):
                        print(f"  - Successful imputations: {imputed['successful_imputations']}/30 ({imputed['success_rate']:.1f}%)")
                        print(f"  - Methods available: {', '.join(imputed.get('methods_available', []))}")
                    else:
                        print(f"  ❌ Imputed results directory not found")
                
                if 'pooled_results' in analysis:
                    pooled = analysis['pooled_results']
                    if pooled.get('file_found', False):
                        print(f"  - Hypotheses analyzed: {pooled['total_hypotheses']}")
                        print(f"  - Rubin pooling applied: {'✓' if pooled['rubin_pooling_applied'] else '❌'}")
                        print(f"  - FDR correction applied: {'✓' if pooled['fdr_correction_applied'] else '❌'}")
                
            elif name == 'Statistical Refinements' and 'statistical_refinements_analysis' in result:
                analysis = result['statistical_refinements_analysis']
                
                if 'fdr_correction' in analysis:
                    fdr = analysis['fdr_correction']
                    if fdr.get('results_found', False):
                        print(f"  - Hypotheses tested: {fdr['hypotheses_tested']}")
                        print(f"  - Significant after FDR: {fdr['significant_after_fdr']}")
                        print(f"  - FDR threshold: {fdr['fdr_threshold']}")
                
                if 'evalue_analysis' in analysis:
                    evalue = analysis['evalue_analysis']
                    print(f"  - E-values computed: {evalue['hypotheses_with_evalues']} hypotheses")
                    print(f"  - Strong evidence (E≥2.0): {evalue['strong_evidence_count']}")
                
            elif name == 'Hypothesis Testing Framework' and 'hypothesis_testing_analysis' in result:
                analysis = result['hypothesis_testing_analysis']
                
                if 'hypothesis_coverage' in analysis:
                    coverage = analysis['hypothesis_coverage']
                    print(f"  - Hypothesis coverage: {analysis.get('coverage_rate', 0):.1f}%")
                    print(f"  - H1 (Normal Labs): {'✓' if coverage['H1_normal_labs'] else '❌'}")
                    print(f"  - H2 (Enhanced SSD): {'✓' if coverage['H2_enhanced_ssd'] else '❌'}")
                    print(f"  - H3 (Medication): {'✓' if coverage['H3_medication_persistence'] else '❌'}")
                
                if 'h4_mediation' in analysis:
                    h4 = analysis['h4_mediation']
                    if h4.get('mediation_results_found', False):
                        prop_med = h4.get('proportion_mediated', 'N/A')
                        threshold_met = h4.get('h4_threshold_met', False)
                        print(f"  - H4 Mediation: {prop_med} ({'✓ ≥55%' if threshold_met else '❌ <55%'})")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "week4_causal_results_validation.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Causal results validation saved to: {output_path}")
    print("\nWEEK 4 COMPREHENSIVE ANALYSIS COMPLETE ✓")
    
    return all_results

if __name__ == "__main__":
    main()

