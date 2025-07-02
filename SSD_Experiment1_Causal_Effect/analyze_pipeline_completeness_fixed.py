#!/usr/bin/env python3
"""
Fixed systematic analysis of pipeline completeness following CLAUDE.md guidelines
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
    
    # Find all run_pipeline_script calls - more comprehensive patterns
    script_patterns = [
        r'run_pipeline_script\(["\']([^"\']+\.py)["\']',  # Basic pattern
        r'run_pipeline_script\(\s*["\']([^"\']+\.py)["\']',  # With whitespace
        r'run_pipeline_script\(["\']([^"\']+)["\']',  # Without .py extension
    ]
    
    scripts_called = []
    for pattern in script_patterns:
        matches = re.findall(pattern, content)
        scripts_called.extend(matches)
    
    # Find all direct python calls
    python_patterns = [
        r'python\s+([^\s"\']+\.py)',
        r'!python\s+([^\s"\']+\.py)',
        r'subprocess.*python.*?([^\s"\']+\.py)',
    ]
    
    python_calls = []
    for pattern in python_patterns:
        matches = re.findall(pattern, content)
        python_calls.extend(matches)
    
    # Also look for script names mentioned in the notebook
    script_mentions = re.findall(r'([0-9]+[a-z_]*\.py)', content)
    
    # Clean up script names (remove .py if not present, add if missing)
    all_scripts = scripts_called + python_calls + script_mentions
    cleaned_scripts = []
    for script in all_scripts:
        if not script.endswith('.py'):
            script += '.py'
        cleaned_scripts.append(script)
    
    return {
        'pipeline_scripts': list(set(scripts_called)),
        'direct_python': list(set(python_calls)),
        'script_mentions': list(set(script_mentions)),
        'total_unique': list(set(cleaned_scripts))
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
            'script_mentions': notebook_scripts['script_mentions'],
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
    
    print("=== PIPELINE COMPLETENESS ANALYSIS (FIXED) ===")
    print(f"Following CLAUDE.md guidelines: systematic verification without assumptions")
    print()
    
    print("📊 NOTEBOOK ANALYSIS:")
    print(f"  - Scripts called via run_pipeline_script: {len(results['notebook_analysis']['scripts_called_via_pipeline'])}")
    print(f"  - Scripts called directly: {len(results['notebook_analysis']['scripts_called_directly'])}")
    print(f"  - Script mentions found: {len(results['notebook_analysis']['script_mentions'])}")
    print(f"  - Total unique script calls: {results['notebook_analysis']['total_unique_calls']}")
    print()
    
    print("📁 SRC FOLDER ANALYSIS:")
    print(f"  - Total available scripts: {results['src_analysis']['total_available_scripts']}")
    print()
    
    print("🔍 GAP ANALYSIS:")
    print(f"  - Scripts in src/ but NOT called in notebook: {results['gap_analysis']['count_missing_from_notebook']}")
    print(f"  - Scripts called but MISSING from src/: {results['gap_analysis']['count_missing_from_src']}")
    print()
    
    print("📋 SCRIPTS FOUND IN NOTEBOOK:")
    for script in sorted(results['notebook_analysis']['all_called_scripts'])[:20]:  # Show first 20
        print(f"    - {script}")
    if len(results['notebook_analysis']['all_called_scripts']) > 20:
        print(f"    ... and {len(results['notebook_analysis']['all_called_scripts']) - 20} more")
    print()
    
    if results['gap_analysis']['scripts_in_src_not_in_notebook']:
        print("⚠️  SCRIPTS IN SRC BUT NOT USED IN NOTEBOOK (first 20):")
        for script in results['gap_analysis']['scripts_in_src_not_in_notebook'][:20]:
            print(f"    - {script}")
        if len(results['gap_analysis']['scripts_in_src_not_in_notebook']) > 20:
            print(f"    ... and {len(results['gap_analysis']['scripts_in_src_not_in_notebook']) - 20} more")
        print()
    
    if results['gap_analysis']['scripts_called_but_missing_from_src']:
        print("❌ SCRIPTS CALLED BUT MISSING FROM SRC:")
        for script in results['gap_analysis']['scripts_called_but_missing_from_src']:
            print(f"    - {script}")
        print()
    
    # Save detailed results
    with open('pipeline_completeness_analysis_fixed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("📄 Detailed results saved to: pipeline_completeness_analysis_fixed.json")
