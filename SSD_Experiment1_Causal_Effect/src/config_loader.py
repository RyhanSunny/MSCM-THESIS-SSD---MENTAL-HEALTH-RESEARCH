#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config_loader.py - Configuration management for SSD pipeline

Provides centralized configuration loading and validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigLoader:
    """Singleton configuration loader"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration dictionary"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Optional path to config file. If not provided,
                        uses default location.
                        
        Returns:
            Dictionary containing configuration
        """
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        
        # Allow environment variable override
        env_config = os.environ.get('SSD_CONFIG_PATH')
        if env_config:
            config_path = Path(env_config)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['temporal', 'cohort', 'exposure', 'paths']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'temporal.reference_date')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload(self) -> None:
        """Force reload of configuration"""
        self._config = None


# Convenience functions
def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    if config_path:
        return loader.load_config(config_path)
    return loader.config


def get_config(key: str = None, default: Any = None) -> Any:
    """
    Get configuration value
    
    Args:
        key: Configuration key using dot notation (e.g., 'temporal.reference_date')
             If None, returns entire config
        default: Default value if key not found
        
    Returns:
        Configuration value or entire config dict
    """
    loader = ConfigLoader()
    if key is None:
        return loader.config
    return loader.get(key, default)


# Specialized getters for common values
def get_reference_date() -> str:
    """Get the reference date for the study"""
    return get_config('temporal.reference_date')


def get_data_path(subpath: str = '') -> Path:
    """Get path to derived data directory"""
    base_path = Path(get_config('paths.derived_data', 'data_derived'))
    if subpath:
        return base_path / subpath
    return base_path


def get_results_path(subpath: str = '') -> Path:
    """Get path to results directory"""
    base_path = Path(get_config('paths.results', 'results'))
    if subpath:
        return base_path / subpath
    return base_path


def get_random_state() -> int:
    """Get the global random seed"""
    return get_config('random_state.global_seed', 42)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print(f"Configuration loaded successfully")
    print(f"Reference date: {get_reference_date()}")
    print(f"Random state: {get_random_state()}")
    print(f"Data path: {get_data_path()}")