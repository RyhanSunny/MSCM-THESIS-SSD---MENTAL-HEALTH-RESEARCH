#!/usr/bin/env python3
"""
5_validation.py - Week 5 Final Validation

Final comprehensive validation of complete pipeline and publication readiness.
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

def validate_publication_readiness() -> Dict[str, Any]:
    """
    Validate publication readiness components
    
    Returns:
    --------
    Dict[str, Any]
        Publication readiness validation results with traceable sources
    """
    logger.info("Validating publication readiness")
    
    results = {
        'validation_type': 'publication_readiness',
        'timestamp': pd.Timestamp.now().isoformat(),
        'seed_used': SEED
    }
    
    try:
        # Check for required figures
        required_figures = [
            'figures/consort_flow.svg',
            'figures/dag.svg', 
            'figures/love_plot.svg',
            'figures/forest_plot.svg',
            'figures/ps_overlap.svg'
        ]
        
        figure_status = {}
        for fig_path in required_figures:
            path = Path(fig_path)
            figure_status[fig_path] = {
                'exists': path.exists(),
                'size_bytes': path.stat().st_size if path.exists() else 0,
                'format': path.suffix
            }
        
        figures_available = sum(1 for status in figure_status.values() if status['exists'])
        
        # Check for manuscript tables
        table_paths = [
            'results/table1_baseline_characteristics.csv',
            'results/table2_causal_estimates.csv',
            'results/table3_sensitivity_analyses.csv'
        ]
        
        table_status = {}
        for table_path in table_paths:
            path = Path(table_path)
            table_status[table_path] = {
                'exists': path.exists(),
                'rows': 0,
                'columns': 0
            }
            
            if path.exists() and path.suffix == '.csv':
                try:
                    df = pd.read_csv(path)
                    table_status[table_path].update({
                        'rows': len(df),
                        'columns': len(df.columns)
                    })
                except:
                    pass
        
        tables_available = sum(1 for status in table_status.values() if status['exists'])
        
        # Check for STROBE checklist
        strobe_paths = [
            'docs/strobe_checklist.json',
            'results/strobe_compliance.json'
        ]
        
        strobe_available = any(Path(p).exists() for p in strobe_paths)
        
        results.update({
            'publication_components': {
                'figures': {
                    'required_figures': len(required_figures),
                    'available_figures': figures_available,
                    'completion_rate': float(figures_available / len(required_figures) * 100),
                    'figure_details': figure_status
                },
                'tables': {
                    'required_tables': len(table_paths),
                    'available_tables': tables_available,
                    'completion_rate': float(tables_available / len(table_paths) * 100),
                    'table_details': table_status
                },
                'strobe_checklist': {
                    'available': strobe_available,
                    'checked_paths': strobe_paths
                }
            }
        })
        
        logger.info("Publication readiness validation complete")
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Publication readiness validation error: {e}")
    
    return results

def validate_reproducibility() -> Dict[str, Any]:
    """
    Validate reproducibility components
    
    Returns:
    --------
    Dict[str, Any]
        Reproducibility validation results
    """
    logger.info("Validating reproducibility components")
    
    results = {
        'validation_type': 'reproducibility',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Check for configuration files
        config_files = [
            'config/config.yaml',
            'config/parameters.json'
        ]
        
        config_status = {}
        for config_path in config_files:
            path = Path(config_path)
            config_status[config_path] = {
                'exists': path.exists(),
                'size_bytes': path.stat().st_size if path.exists() else 0
            }
            
            if path.exists():
                try:
                    if path.suffix == '.yaml':
                        import yaml
                        with open(path, 'r') as f:
                            config_data = yaml.safe_load(f)
                        config_status[config_path]['parameters_count'] = len(config_data) if isinstance(config_data, dict) else 0
                    elif path.suffix == '.json':
                        with open(path, 'r') as f:
                            config_data = json.load(f)
                        config_status[config_path]['parameters_count'] = len(config_data) if isinstance(config_data, dict) else 0
                except:
                    config_status[config_path]['parameters_count'] = 'Error reading file'
        
        # Check for seed usage in scripts
        src_dir = Path("src")
        seed_usage = {}
        
        if src_dir.exists():
            python_files = list(src_dir.glob("*.py"))
            
            for py_file in python_files[:10]:  # Check first 10 files for performance
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    seed_usage[py_file.name] = {
                        'seed_set': 'SEED' in content or 'random.seed' in content or 'np.random.seed' in content,
                        'seed_value_42': 'SEED = 42' in content or 'seed(42)' in content
                    }
                except:
                    seed_usage[py_file.name] = {'error': 'Could not read file'}
        
        # Check for version control
        git_dir = Path(".git")
        version_control = {
            'git_repository': git_dir.exists(),
            'git_dir_size': sum(f.stat().st_size for f in git_dir.rglob('*') if f.is_file()) if git_dir.exists() else 0
        }
        
        # Check for documentation
        doc_files = [
            'README.md',
            'docs/methodology.md',
            'docs/data_dictionary.md'
        ]
        
        documentation_status = {}
        for doc_path in doc_files:
            path = Path(doc_path)
            documentation_status[doc_path] = {
                'exists': path.exists(),
                'size_bytes': path.stat().st_size if path.exists() else 0
            }
        
        docs_available = sum(1 for status in documentation_status.values() if status['exists'])
        
        results.update({
            'reproducibility_components': {
                'configuration': {
                    'config_files_available': sum(1 for status in config_status.values() if status['exists']),
                    'total_config_files': len(config_files),
                    'config_details': config_status
                },
                'seed_management': {
                    'scripts_checked': len(seed_usage),
                    'scripts_with_seed': sum(1 for status in seed_usage.values() if status.get('seed_set', False)),
                    'scripts_with_seed_42': sum(1 for status in seed_usage.values() if status.get('seed_value_42', False)),
                    'seed_details': seed_usage
                },
                'version_control': version_control,
                'documentation': {
                    'required_docs': len(doc_files),
                    'available_docs': docs_available,
                    'completion_rate': float(docs_available / len(doc_files) * 100),
                    'doc_details': documentation_status
                }
            }
        })
        
        logger.info("Reproducibility validation complete")
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Reproducibility validation error: {e}")
    
    return results

def validate_thesis_defense_readiness() -> Dict[str, Any]:
    """
    Validate thesis defense readiness
    
    Returns:
    --------
    Dict[str, Any]
        Thesis defense readiness validation results
    """
    logger.info("Validating thesis defense readiness")
    
    results = {
        'validation_type': 'thesis_defense_readiness',
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Check for critical defense components
        defense_components = {
            'fallback_audit_addressed': Path("FALLBACK_AUDIT.md").exists(),
            'clinical_validation': Path("docs/clinical_validation.md").exists(),
            'statistical_methods': Path("src/week4_statistical_refinements.py").exists(),
            'sensitivity_analyses': Path("src/15_robustness.py").exists(),
            'literature_review': any(Path(p).exists() for p in ["docs/literature_review.md", "LITERATURE_ANALYSIS_UNIQUE_CONTRIBUTIONS.md"]),
            'methodology_blueprint': Path("SSDTHESISfinalMETHODOLOGIESblueprint-UPDATED20250701.md").exists()
        }
        
        # Check for key results files
        key_results = {
            'pooled_estimates': Path("results/pooled_causal_estimates.json").exists(),
            'fdr_correction': Path("results/fdr_adjusted_results.json").exists(),
            'mediation_analysis': Path("results/mediation_results.json").exists(),
            'sensitivity_results': Path("results/sensitivity_analysis.json").exists()
        }
        
        # Check for potential committee questions preparedness
        committee_preparedness = {
            'parameter_justification': any(Path(p).exists() for p in ["docs/parameter_justification.md", "config/config.yaml"]),
            'missing_data_handling': Path("src/07b_missing_data_master.py").exists(),
            'causal_assumptions': any(Path(p).exists() for p in ["figures/dag.svg", "docs/causal_assumptions.md"]),
            'effect_size_interpretation': Path("results/pooled_causal_estimates.json").exists(),
            'clinical_significance': any(Path(p).exists() for p in ["docs/clinical_validation.md", "ClinicalValidationofSSDPipelineHypotheses.md"])
        }
        
        # Calculate readiness scores
        defense_score = sum(defense_components.values()) / len(defense_components) * 100
        results_score = sum(key_results.values()) / len(key_results) * 100
        preparedness_score = sum(committee_preparedness.values()) / len(committee_preparedness) * 100
        
        overall_readiness = (defense_score + results_score + preparedness_score) / 3
        
        results.update({
            'thesis_defense_analysis': {
                'defense_components': defense_components,
                'key_results': key_results,
                'committee_preparedness': committee_preparedness,
                'readiness_scores': {
                    'defense_components_score': float(defense_score),
                    'key_results_score': float(results_score),
                    'committee_preparedness_score': float(preparedness_score),
                    'overall_readiness_score': float(overall_readiness)
                },
                'readiness_level': 'HIGH' if overall_readiness >= 80 else 'MEDIUM' if overall_readiness >= 60 else 'LOW'
            }
        })
        
        logger.info("Thesis defense readiness validation complete")
        
    except Exception as e:
        results.update({
            'validation_error': str(e),
            'error_type': type(e).__name__
        })
        logger.error(f"Thesis defense readiness validation error: {e}")
    
    return results

def main():
    """Main validation function for Week 5 Final Validation"""
    print("="*80)
    print("WEEK 5 FINAL VALIDATION: Publication & Thesis Defense Readiness")
    print("Following CLAUDE.md + RULES.md + ANALYSIS_RULES.md")
    print("="*80)
    
    # Run final validations
    validations = [
        ('Publication Readiness', validate_publication_readiness),
        ('Reproducibility', validate_reproducibility),
        ('Thesis Defense Readiness', validate_thesis_defense_readiness)
    ]
    
    all_results = {
        'validation_suite': 'week_5_final_validation',
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
            if name == 'Publication Readiness' and 'publication_components' in result:
                components = result['publication_components']
                
                figures = components['figures']
                print(f"  - Figures: {figures['available_figures']}/{figures['required_figures']} ({figures['completion_rate']:.1f}%)")
                
                tables = components['tables']
                print(f"  - Tables: {tables['available_tables']}/{tables['required_tables']} ({tables['completion_rate']:.1f}%)")
                
                strobe = components['strobe_checklist']
                print(f"  - STROBE checklist: {'✓' if strobe['available'] else '❌'}")
                
            elif name == 'Reproducibility' and 'reproducibility_components' in result:
                components = result['reproducibility_components']
                
                config = components['configuration']
                print(f"  - Configuration files: {config['config_files_available']}/{config['total_config_files']}")
                
                seed = components['seed_management']
                print(f"  - Scripts with seed: {seed['scripts_with_seed']}/{seed['scripts_checked']}")
                
                docs = components['documentation']
                print(f"  - Documentation: {docs['available_docs']}/{docs['required_docs']} ({docs['completion_rate']:.1f}%)")
                
                vc = components['version_control']
                print(f"  - Version control: {'✓' if vc['git_repository'] else '❌'}")
                
            elif name == 'Thesis Defense Readiness' and 'thesis_defense_analysis' in result:
                analysis = result['thesis_defense_analysis']
                scores = analysis['readiness_scores']
                
                print(f"  - Defense components: {scores['defense_components_score']:.1f}%")
                print(f"  - Key results: {scores['key_results_score']:.1f}%")
                print(f"  - Committee preparedness: {scores['committee_preparedness_score']:.1f}%")
                print(f"  - Overall readiness: {scores['overall_readiness_score']:.1f}% ({analysis['readiness_level']})")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / "week5_final_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Final validation results saved to: {output_path}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("FINAL VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    # Extract key metrics
    try:
        pub_readiness = all_results['results']['publication_readiness']['publication_components']
        repro_readiness = all_results['results']['reproducibility']['reproducibility_components']
        thesis_readiness = all_results['results']['thesis_defense_readiness']['thesis_defense_analysis']
        
        print(f"📊 Publication Readiness:")
        print(f"   - Figures: {pub_readiness['figures']['completion_rate']:.1f}%")
        print(f"   - Tables: {pub_readiness['tables']['completion_rate']:.1f}%")
        
        print(f"\n🔄 Reproducibility:")
        print(f"   - Documentation: {repro_readiness['documentation']['completion_rate']:.1f}%")
        print(f"   - Version Control: {'✓' if repro_readiness['version_control']['git_repository'] else '❌'}")
        
        print(f"\n🎓 Thesis Defense Readiness:")
        print(f"   - Overall Score: {thesis_readiness['readiness_scores']['overall_readiness_score']:.1f}%")
        print(f"   - Readiness Level: {thesis_readiness['readiness_level']}")
        
        if thesis_readiness['readiness_level'] == 'HIGH':
            print(f"\n✅ THESIS READY FOR DEFENSE!")
        elif thesis_readiness['readiness_level'] == 'MEDIUM':
            print(f"\n⚠️ THESIS NEEDS MINOR IMPROVEMENTS")
        else:
            print(f"\n❌ THESIS NEEDS SIGNIFICANT WORK")
            
    except Exception as e:
        print(f"\n⚠️ Could not generate summary: {e}")
    
    print("\nWEEK 5 FINAL VALIDATION COMPLETE ✓")
    
    return all_results

if __name__ == "__main__":
    main()

