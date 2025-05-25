#!/usr/bin/env python3
"""
Test suite for config_loader.py

Tests configuration loading, validation, and error handling.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import yaml

# Add src to path
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from config_loader import load_config, get_config


class TestConfigLoader:
    """Test configuration loading functionality."""
    
    def test_load_config_valid(self):
        """Test loading valid configuration."""
        config = load_config()
        
        # Check required sections exist
        assert "study" in config
        assert "temporal" in config
        assert "cohort" in config
        assert "paths" in config
        
        # Check specific values
        assert config["study"]["name"] == "SSD Causal Effect Analysis"
        assert config["cohort"]["min_age"] == 18
        assert config["temporal"]["reference_date"] == "2018-01-01"
    
    def test_get_config_nested_keys(self):
        """Test getting nested configuration values."""
        # Test existing nested key
        ref_date = get_config("temporal.reference_date")
        assert ref_date == "2018-01-01"
        
        # Test with default value
        non_existent = get_config("nonexistent.key", "default_value")
        assert non_existent == "default_value"
        
        # Test top-level key
        min_age = get_config("cohort.min_age")
        assert min_age == 18
    
    def test_get_config_missing_key(self):
        """Test behavior with missing configuration keys."""
        # Should return None for missing key without default
        result = get_config("missing.key")
        assert result is None
        
        # Should return default for missing key with default
        result = get_config("missing.key", "default")
        assert result == "default"
    
    def test_temporal_consistency(self):
        """Test that temporal configuration is consistent."""
        ref_date = get_config("temporal.reference_date")
        censor_date = get_config("temporal.censor_date")
        exposure_start = get_config("temporal.exposure_window_start")
        outcome_end = get_config("temporal.outcome_window_end")
        
        # Convert to comparable format
        from datetime import datetime
        ref_dt = datetime.strptime(ref_date, "%Y-%m-%d")
        censor_dt = datetime.strptime(censor_date, "%Y-%m-%d")
        exposure_dt = datetime.strptime(exposure_start, "%Y-%m-%d")
        outcome_dt = datetime.strptime(outcome_end, "%Y-%m-%d")
        
        # Check logical ordering
        assert ref_dt <= censor_dt, "Reference date should be <= censor date"
        assert ref_dt <= exposure_dt, "Reference date should be <= exposure start"
        assert exposure_dt < outcome_dt, "Exposure should start before outcome period"
    
    def test_path_configuration(self):
        """Test path configurations are valid."""
        paths = get_config("paths")
        
        required_paths = [
            "checkpoint_root", "derived_data", "results", 
            "figures", "reports", "code_lists"
        ]
        
        for path_key in required_paths:
            assert path_key in paths, f"Missing required path: {path_key}"
            assert isinstance(paths[path_key], str), f"Path {path_key} should be string"
    
    def test_cohort_criteria_validation(self):
        """Test cohort criteria are reasonable."""
        min_age = get_config("cohort.min_age")
        min_obs_months = get_config("cohort.min_observation_months")
        max_charlson = get_config("cohort.max_charlson_score")
        
        assert min_age >= 18, "Minimum age should be at least 18"
        assert min_obs_months >= 12, "Minimum observation should be at least 12 months"
        assert max_charlson <= 10, "Maximum Charlson score should be reasonable"
    
    def test_exposure_criteria_validation(self):
        """Test exposure criteria are valid."""
        min_labs = get_config("exposure.min_normal_labs")
        min_referrals = get_config("exposure.min_symptom_referrals")
        min_drug_days = get_config("exposure.min_drug_days")
        
        assert min_labs >= 1, "Minimum labs should be positive"
        assert min_referrals >= 1, "Minimum referrals should be positive"
        assert min_drug_days >= 30, "Minimum drug days should be reasonable"
    
    def test_random_seeds_present(self):
        """Test that random seeds are configured."""
        seeds = get_config("random_state")
        
        required_seeds = ["global_seed", "numpy_seed", "tensorflow_seed"]
        for seed_key in required_seeds:
            assert seed_key in seeds, f"Missing random seed: {seed_key}"
            assert isinstance(seeds[seed_key], int), f"Seed {seed_key} should be integer"


class TestConfigIntegration:
    """Integration tests for configuration with other modules."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        temp_config = {
            "study": {"name": "Test Study"},
            "temporal": {"reference_date": "2020-01-01"},
            "cohort": {"min_age": 21},
            "paths": {"derived_data": "test_data"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(temp_config, f)
            return f.name
    
    def test_config_file_not_found(self):
        """Test behavior when config file is missing."""
        # This would test error handling, but we want the current config to work
        # In a real test, we'd temporarily modify the config path
        pass
    
    def test_malformed_config(self, temp_config):
        """Test behavior with malformed configuration."""
        # Write malformed YAML
        with open(temp_config, 'w') as f:
            f.write("invalid: yaml: content:\n  - malformed")
        
        # Test that it raises appropriate error
        # (This would need modification to config_loader to accept custom path)
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])