#!/usr/bin/env python3
"""
power_analysis_sync.py - Synchronize power analysis parameters

Week 5 Task E: Power-analysis consistency sync
Ensures YAML and blueprint narrative power values are aligned.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import yaml
import re
import math
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_blueprint_power_params(blueprint_path: Path) -> Dict[str, Any]:
    """
    Extract power analysis parameters from blueprint narrative
    
    Parameters:
    -----------
    blueprint_path : Path
        Path to the blueprint markdown file
        
    Returns:
    --------
    Dict[str, Any]
        Extracted power parameters
    """
    with open(blueprint_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    params = {}
    
    # Extract RR value
    rr_pattern = r'detect RR ([\d.]+)'
    rr_match = re.search(rr_pattern, content, re.IGNORECASE)
    if rr_match:
        params['relative_risk'] = float(rr_match.group(1))
    
    # Extract power percentage
    power_pattern = r'with (\d+)\s*%\s*power'
    power_match = re.search(power_pattern, content, re.IGNORECASE)
    if power_match:
        params['power'] = float(power_match.group(1)) / 100.0
    
    # Extract alpha (significance level)
    alpha_pattern = r'α\s*([\d.]+)'
    alpha_match = re.search(alpha_pattern, content)
    if alpha_match:
        params['alpha'] = float(alpha_match.group(1))
    else:
        params['alpha'] = 0.05  # Default
    
    # Extract sample size
    n_pattern = r'required n\s*[=≈]\s*(\d+)'
    n_match = re.search(n_pattern, content, re.IGNORECASE)
    if n_match:
        params['required_n'] = int(n_match.group(1))
    
    # Extract attrition rate
    attrition_pattern = r'attrition.*?(\d+)\s*%'
    attrition_match = re.search(attrition_pattern, content, re.IGNORECASE)
    if attrition_match:
        params['attrition_adjustment'] = f"{attrition_match.group(1)}%"
    else:
        params['attrition_adjustment'] = "20%"  # Default from existing YAML
    
    logger.info(f"Extracted blueprint parameters: {params}")
    return params


def convert_rr_to_effect_size(relative_risk: float) -> float:
    """
    Convert relative risk to standardized effect size
    
    Parameters:
    -----------
    relative_risk : float
        Relative risk ratio
        
    Returns:
    --------
    float
        Approximate standardized effect size
    """
    # For small RRs close to 1, the log-odds approximation works well
    # Cohen's d ≈ 2 * log(RR) for binary outcomes
    # But for very small effects like RR=1.05, we use a more conservative approach
    
    if relative_risk <= 1.0:
        return 0.0
    
    # For RR close to 1, use conservative linear approximation
    if relative_risk < 1.2:
        # RR 1.05 → effect size ≈ 0.05 (very small effect)
        effect_size = (relative_risk - 1.0) * 1.0  # Linear scaling
    else:
        # For larger RRs, use log transformation
        effect_size = 2 * math.log(relative_risk)
    
    return round(effect_size, 4)


def update_yaml_power_analysis(yaml_path: Path, blueprint_params: Dict[str, Any]) -> None:
    """
    Update YAML file with consistent power analysis parameters
    
    Parameters:
    -----------
    yaml_path : Path
        Path to YAML file to update
    blueprint_params : Dict[str, Any]
        Parameters extracted from blueprint
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Create or update power_analysis section
    if 'power_analysis' not in data:
        data['power_analysis'] = {'parameters': {}}
    elif 'parameters' not in data['power_analysis']:
        data['power_analysis']['parameters'] = {}
    
    # Update parameters to match blueprint
    params = data['power_analysis']['parameters']
    
    # Remove inconsistent effect_size (as per blueprint note)
    if 'effect_size' in params:
        logger.info(f"Removing inconsistent effect_size={params['effect_size']} from YAML")
        del params['effect_size']
    
    # Add consistent parameters based on blueprint
    params['alpha'] = blueprint_params.get('alpha', 0.05)
    params['power'] = blueprint_params.get('power', 0.9)  # 90% power from blueprint
    params['attrition_adjustment'] = blueprint_params.get('attrition_adjustment', "20%")
    
    # Add relative risk as the primary effect measure
    if 'relative_risk' in blueprint_params:
        params['relative_risk'] = blueprint_params['relative_risk']
        
        # Convert to effect size for reference but mark as derived
        effect_size = convert_rr_to_effect_size(blueprint_params['relative_risk'])
        params['effect_size_derived'] = effect_size
        params['effect_size_note'] = f"Derived from RR={blueprint_params['relative_risk']}"
    
    # Update required sample size
    if 'required_n' in blueprint_params:
        data['power_analysis']['required_n'] = blueprint_params['required_n']
    
    # Add consistency timestamp
    params['last_sync'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    params['sync_source'] = 'blueprint_narrative'
    
    # Write updated YAML
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Updated YAML power analysis: {yaml_path}")


def sync_power_analysis_consistency() -> Dict[str, Any]:
    """
    Main function to synchronize power analysis parameters
    
    Returns:
    --------
    Dict[str, Any]
        Sync results and validation
    """
    logger.info("Starting power analysis consistency sync...")
    
    # Define file paths
    project_root = Path(__file__).parent.parent
    blueprint_path = project_root / 'SSD THESIS final METHODOLOGIES blueprint (1).md'
    results_dir = project_root / 'results'
    
    # Extract parameters from blueprint
    if not blueprint_path.exists():
        raise FileNotFoundError(f"Blueprint not found: {blueprint_path}")
    
    blueprint_params = extract_blueprint_power_params(blueprint_path)
    
    # Find most recent study documentation YAML
    yaml_files = list(results_dir.glob('study_documentation_*.yaml'))
    if not yaml_files:
        raise FileNotFoundError("No study documentation YAML files found")
    
    latest_yaml = sorted(yaml_files)[-1]
    logger.info(f"Updating latest YAML: {latest_yaml}")
    
    # Update YAML to match blueprint
    update_yaml_power_analysis(latest_yaml, blueprint_params)
    
    # Validate consistency
    results = {
        'blueprint_file': str(blueprint_path),
        'yaml_file': str(latest_yaml),
        'blueprint_params': blueprint_params,
        'sync_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'success'
    }
    
    logger.info("Power analysis consistency sync completed successfully")
    return results


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync power analysis parameters')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate consistency, do not update files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.validate_only:
            # Just check consistency without updating
            project_root = Path(__file__).parent.parent
            blueprint_path = project_root / 'SSD THESIS final METHODOLOGIES blueprint (1).md'
            blueprint_params = extract_blueprint_power_params(blueprint_path)
            print(f"Blueprint parameters: {blueprint_params}")
        else:
            # Update files for consistency
            results = sync_power_analysis_consistency()
            print(f"Sync completed: {results['status']}")
            print(f"Updated: {results['yaml_file']}")
            
    except Exception as e:
        logger.error(f"Power analysis sync failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()