#!/usr/bin/env python3
"""
test_power_analysis_consistency.py - Tests for power analysis consistency

Tests for Week 5 Task E: Power-analysis consistency sync
Ensures YAML and blueprint narrative power values align within 1e-6 tolerance.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import pytest
import yaml
import re
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestPowerAnalysisConsistency:
    """Test power analysis consistency between YAML and blueprint"""
    
    def test_power_analysis_yaml_exists(self):
        """Test that power_analysis section exists in study documentation"""
        # Find most recent study documentation YAML
        results_dir = Path(__file__).parent.parent / 'results'
        yaml_files = list(results_dir.glob('study_documentation_*.yaml'))
        
        assert len(yaml_files) > 0, "No study documentation YAML files found"
        
        # Use most recent file
        latest_yaml = sorted(yaml_files)[-1]
        
        with open(latest_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        assert 'power_analysis' in data, "power_analysis section missing from YAML"
        assert 'parameters' in data['power_analysis'], "parameters missing from power_analysis"
    
    def test_blueprint_power_narrative_exists(self):
        """Test that blueprint contains power analysis narrative"""
        blueprint_path = Path(__file__).parent.parent / 'SSD THESIS final METHODOLOGIES blueprint (1).md'
        
        assert blueprint_path.exists(), "Blueprint file not found"
        
        with open(blueprint_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for power analysis section
        power_pattern = r'detect RR [\d.]+.*power.*%'
        matches = re.search(power_pattern, content, re.IGNORECASE)
        
        assert matches is not None, "Power analysis narrative not found in blueprint"
    
    def test_rr_effect_size_consistency(self):
        """Test that RR values are consistent between YAML and blueprint"""
        # Load YAML data
        results_dir = Path(__file__).parent.parent / 'results'
        yaml_files = list(results_dir.glob('study_documentation_*.yaml'))
        latest_yaml = sorted(yaml_files)[-1]
        
        with open(latest_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Load blueprint
        blueprint_path = Path(__file__).parent.parent / 'SSD THESIS final METHODOLOGIES blueprint (1).md'
        with open(blueprint_path, 'r', encoding='utf-8') as f:
            blueprint_content = f.read()
        
        # Extract RR from blueprint
        rr_pattern = r'detect RR ([\d.]+)'
        rr_match = re.search(rr_pattern, blueprint_content, re.IGNORECASE)
        
        assert rr_match is not None, "RR value not found in blueprint"
        blueprint_rr = float(rr_match.group(1))
        
        # Convert YAML effect_size to RR if present
        if 'effect_size' in yaml_data['power_analysis']['parameters']:
            yaml_effect_size = yaml_data['power_analysis']['parameters']['effect_size']
            
            # For health outcomes, effect size 0.2 often corresponds to RR ≈ 1.2-1.25
            # But blueprint says RR 1.05, which is a much smaller effect
            # This should fail as per MAX-EVAL requirement
            
            # Convert standardized effect size to approximate RR
            # Using Cohen's conventions: small effect = 0.2 ≈ RR 1.2+
            expected_rr_from_effect_size = 1.0 + yaml_effect_size  # Rough approximation
            
            # Check if values are inconsistent (should fail per requirement)
            rr_difference = abs(blueprint_rr - expected_rr_from_effect_size)
            
            # This test should FAIL when effect_size=0.2 and RR=1.05 (difference > 1e-6)
            assert rr_difference <= 1e-6, (
                f"RR values inconsistent: Blueprint RR={blueprint_rr}, "
                f"YAML effect_size={yaml_effect_size} implies RR≈{expected_rr_from_effect_size}, "
                f"difference={rr_difference} > 1e-6 tolerance"
            )
    
    def test_power_percentage_consistency(self):
        """Test that power percentages match between YAML and blueprint"""
        # Load YAML data
        results_dir = Path(__file__).parent.parent / 'results'
        yaml_files = list(results_dir.glob('study_documentation_*.yaml'))
        latest_yaml = sorted(yaml_files)[-1]
        
        with open(latest_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Load blueprint
        blueprint_path = Path(__file__).parent.parent / 'SSD THESIS final METHODOLOGIES blueprint (1).md'
        with open(blueprint_path, 'r', encoding='utf-8') as f:
            blueprint_content = f.read()
        
        # Extract power from YAML
        yaml_power = yaml_data['power_analysis']['parameters']['power']
        
        # Extract power percentage from blueprint
        power_pattern = r'with (\d+)\s*%\s*power'
        power_match = re.search(power_pattern, blueprint_content, re.IGNORECASE)
        
        assert power_match is not None, "Power percentage not found in blueprint"
        blueprint_power_pct = float(power_match.group(1))
        blueprint_power = blueprint_power_pct / 100.0
        
        # Check consistency within tolerance
        power_difference = abs(yaml_power - blueprint_power)
        
        assert power_difference <= 1e-6, (
            f"Power values inconsistent: YAML power={yaml_power}, "
            f"Blueprint power={blueprint_power} ({blueprint_power_pct}%), "
            f"difference={power_difference} > 1e-6 tolerance"
        )
    
    def test_alpha_consistency(self):
        """Test that alpha values are consistent"""
        # Load YAML data
        results_dir = Path(__file__).parent.parent / 'results'
        yaml_files = list(results_dir.glob('study_documentation_*.yaml'))
        latest_yaml = sorted(yaml_files)[-1]
        
        with open(latest_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Load blueprint
        blueprint_path = Path(__file__).parent.parent / 'SSD THESIS final METHODOLOGIES blueprint (1).md'
        with open(blueprint_path, 'r', encoding='utf-8') as f:
            blueprint_content = f.read()
        
        # Extract alpha from YAML
        yaml_alpha = yaml_data['power_analysis']['parameters']['alpha']
        
        # Extract alpha from blueprint (look for α 0.05 pattern)
        alpha_pattern = r'α\s*([\d.]+)'
        alpha_match = re.search(alpha_pattern, blueprint_content)
        
        if alpha_match:
            blueprint_alpha = float(alpha_match.group(1))
            
            # Check consistency within tolerance
            alpha_difference = abs(yaml_alpha - blueprint_alpha)
            
            assert alpha_difference <= 1e-6, (
                f"Alpha values inconsistent: YAML alpha={yaml_alpha}, "
                f"Blueprint alpha={blueprint_alpha}, "
                f"difference={alpha_difference} > 1e-6 tolerance"
            )
    
    def test_sample_size_consistency(self):
        """Test that required sample sizes are consistent"""
        # Load YAML data
        results_dir = Path(__file__).parent.parent / 'results'
        yaml_files = list(results_dir.glob('study_documentation_*.yaml'))
        latest_yaml = sorted(yaml_files)[-1]
        
        with open(latest_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Load blueprint
        blueprint_path = Path(__file__).parent.parent / 'SSD THESIS final METHODOLOGIES blueprint (1).md'
        with open(blueprint_path, 'r', encoding='utf-8') as f:
            blueprint_content = f.read()
        
        # Extract required n from YAML
        yaml_required_n = yaml_data['power_analysis']['required_n']
        
        # Extract required n from blueprint (look for "required n = X" pattern)
        n_pattern = r'required n\s*[=≈]\s*(\d+)'
        n_match = re.search(n_pattern, blueprint_content, re.IGNORECASE)
        
        if n_match:
            blueprint_required_n = int(n_match.group(1))
            
            # Allow for small differences in sample size calculations
            n_difference = abs(yaml_required_n - blueprint_required_n)
            relative_diff = n_difference / max(yaml_required_n, blueprint_required_n)
            
            assert relative_diff <= 0.1, (  # Allow 10% difference for rounding
                f"Sample size inconsistent: YAML required_n={yaml_required_n}, "
                f"Blueprint required_n={blueprint_required_n}, "
                f"relative difference={relative_diff:.3f} > 0.1 tolerance"
            )


if __name__ == "__main__":
    pytest.main([__file__])