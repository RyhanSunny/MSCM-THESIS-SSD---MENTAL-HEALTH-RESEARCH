#!/usr/bin/env python3
"""
Systematic analysis of pipeline completeness following CLAUDE.md guidelines
Author: Manus AI (following CLAUDE.md requirements)
Date: July 2, 2025
"""

import re
import json
from pathlib import Path

def extract_scripts_from_notebook(notebook_path):
    """Extract all script calls from the notebook"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all run_pipeline_script calls
    script_pattern = r'run_pipeline_script\(["\']([^"\']+)["\']'
    scripts_called = re.findall(script_pattern, content)
    
    # Find all direct python calls
    python_pattern = r'python\s+([^\s"\']+\.py)'
    python_calls = re.findall(python_pattern, content)
    
    return {
        'pipeline_scripts': list(set(scripts_called)),
        'direct_python': list(set(python_calls)),
        'total_unique': list(set(scripts_called + python_calls))
    }

def get_available_scripts(src_dir):
    """Get all available scripts in src directory"""
    src_path = Path(src_dir)
    scripts = [f.name for f in src_path.glob('*.py') if f.name != '__init__.py']
    return sorted(scripts)

def analyze_completeness():
    """Main analysis function"""
    notebook_path = "SSD_Complete_Pipeline_Analysis_v2_CLEAN_COMPLETE.ipynb"
    src_dir = "src"
    
    # Extract scripts from notebook
    notebook_scripts = extract_scripts_from_notebook(notebook_path)
    
    # Get available scripts
    available_scripts = get_available_scripts(src_dir)
    
    # Find gaps
    called_scripts = set(notebook_scripts['total_unique'])
    available_set = set(available_scripts)
    
    missing_from_notebook = available_set - called_scripts
    missing_from_src = called_scripts - available_set
    
    # Analysis results
    results = {
        'notebook_analysis': {
            'scripts_called_via_pipeline': notebook_scripts['pipeline_scripts'],
            'scripts_called_directly': notebook_scripts['direct_python'],
            'total_unique_calls': len(notebook_scripts['total_unique']),
            'all_called_scripts': sorted(notebook_scripts['total_unique'])
        },
        'src_analysis': {
            'total_available_scripts': len(available_scripts),
            'available_scripts': available_scripts
        },
        'gap_analysis': {
            'scripts_in_src_not_in_notebook': sorted(missing_from_notebook),
            'scripts_called_but_missing_from_src': sorted(missing_from_src),
            'count_missing_from_notebook': len(missing_from_notebook),
            'count_missing_from_src': len(missing_from_src)
        }
    }
    
    return results

if __name__ == "__main__":
    results = analyze_completeness()
    
    print("=== PIPELINE COMPLETENESS ANALYSIS ===")
    print(f"Following CLAUDE.md guidelines: systematic verification without assumptions")
    print()
    
    print("📊 NOTEBOOK ANALYSIS:")
    print(f"  - Scripts called via run_pipeline_script: {len(results['notebook_analysis']['scripts_called_via_pipeline'])}")
    print(f"  - Scripts called directly: {len(results['notebook_analysis']['scripts_called_directly'])}")
    print(f"  - Total unique script calls: {results['notebook_analysis']['total_unique_calls']}")
    print()
    
    print("📁 SRC FOLDER ANALYSIS:")
    print(f"  - Total available scripts: {results['src_analysis']['total_available_scripts']}")
    print()
    
    print("🔍 GAP ANALYSIS:")
    print(f"  - Scripts in src/ but NOT called in notebook: {results['gap_analysis']['count_missing_from_notebook']}")
    print(f"  - Scripts called but MISSING from src/: {results['gap_analysis']['count_missing_from_src']}")
    print()
    
    if results['gap_analysis']['scripts_in_src_not_in_notebook']:
        print("⚠️  SCRIPTS IN SRC BUT NOT USED IN NOTEBOOK:")
        for script in results['gap_analysis']['scripts_in_src_not_in_notebook']:
            print(f"    - {script}")
        print()
    
    if results['gap_analysis']['scripts_called_but_missing_from_src']:
        print("❌ SCRIPTS CALLED BUT MISSING FROM SRC:")
        for script in results['gap_analysis']['scripts_called_but_missing_from_src']:
            print(f"    - {script}")
        print()
    
    # Save detailed results
    with open('pipeline_completeness_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("📄 Detailed results saved to: pipeline_completeness_analysis.json")
