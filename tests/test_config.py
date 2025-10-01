"""Tests for configuration management module."""

import pytest
import os
import yaml
from pathlib import Path
from cosmo_analysis.config import Config, load_config, get_config


class TestConfig:
    """Test cases for Config class."""
    
    def test_default_config(self):
        """Test that default configuration is loaded correctly."""
        config = Config.load_config()
        
        assert config is not None
        assert 'paths' in config.data
        assert 'simulation_parameters' in config.data
        assert 'plotting_defaults' in config.data
        assert 'analysis_options' in config.data
    
    def test_load_from_file(self, temp_config_file):
        """Test loading configuration from file."""
        config = Config.load_config(str(temp_config_file))
        
        assert config.get('paths.output_directory') == '/tmp/test_output'
        assert config.get('simulation_parameters.particle_types.gas') == 'PartType0'
        assert config.get('plotting_defaults.save_plots') is False
    
    def test_get_with_dot_notation(self):
        """Test getting values using dot notation."""
        config = Config.load_config()
        
        # Test nested access
        gas_type = config.get('simulation_parameters.particle_types.gas')
        assert gas_type == 'PartType0'
        
        # Test default value
        nonexistent = config.get('nonexistent.key', 'default')
        assert nonexistent == 'default'
    
    def test_dictionary_access(self):
        """Test getting values using dictionary syntax."""
        config = Config.load_config()
        
        assert 'paths' in config['paths'] or config['paths'] is not None
        assert isinstance(config['simulation_parameters'], dict)
    
    def test_singleton_instance(self):
        """Test that Config uses singleton pattern."""
        config1 = Config.get_instance()
        config2 = Config.get_instance()
        
        assert config1 is config2
    
    def test_set_instance(self, temp_config_file):
        """Test setting singleton instance."""
        custom_config = Config.load_config(str(temp_config_file))
        Config.set_instance(custom_config)
        
        retrieved = Config.get_instance()
        assert retrieved is custom_config
    
    def test_convenience_functions(self, temp_config_file):
        """Test convenience functions."""
        # Test load_config function
        config = load_config(str(temp_config_file))
        assert config.get('paths.output_directory') == '/tmp/test_output'
        
        # Reset and test get_config function
        Config.set_instance(config)
        retrieved = get_config()
        assert retrieved is config


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_config_with_environment_variable(self, temp_config_file, monkeypatch):
        """Test loading configuration via environment variable."""
        monkeypatch.setenv('COSMO_CONFIG', str(temp_config_file))
        
        config = Config.load_config()
        assert config.get('paths.output_directory') == '/tmp/test_output'
    
    def test_missing_file_fallback(self):
        """Test fallback to defaults when file doesn't exist."""
        config = Config.load_config('/nonexistent/path/config.yaml')
        
        # Should fall back to defaults
        assert config.get('simulation_parameters.particle_types.gas') == 'PartType0'
    
    def test_default_config_structure(self):
        """Test that default config has all required sections."""
        config = Config.load_config()
        
        required_sections = [
            'paths',
            'simulation_parameters',
            'plotting_defaults',
            'analysis_options',
            'cosmology',
            'logging'
        ]
        
        for section in required_sections:
            assert section in config.data, f"Missing required section: {section}"
