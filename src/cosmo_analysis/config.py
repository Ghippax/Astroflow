"""Configuration management for cosmo_analysis.

This module provides functionality to load and manage configuration from YAML files.
It replaces hardcoded constants and paths with a flexible configuration system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration container for cosmo_analysis.
    
    Attributes:
        data (dict): The configuration dictionary loaded from YAML
        _instance (Config): Singleton instance
    """
    
    _instance: Optional['Config'] = None
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary loaded from YAML
        """
        self.data = config_dict
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file. If None, looks for:
                        1. Environment variable COSMO_CONFIG
                        2. config.yaml in current directory
                        3. config.yaml in package root
                        4. Falls back to default values
        
        Returns:
            Config: Configuration object
            
        Raises:
            FileNotFoundError: If specified config_path doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        # Determine config path
        if config_path is None:
            # Try environment variable
            config_path = os.environ.get('COSMO_CONFIG')
            
            # Try current directory
            if config_path is None and os.path.exists('config.yaml'):
                config_path = 'config.yaml'
            
            # Try package root
            if config_path is None:
                package_root = Path(__file__).parent.parent.parent
                candidate = package_root / 'config.yaml'
                if candidate.exists():
                    config_path = str(candidate)
        
        # Load from file if path exists
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(config_dict)
        
        # Fall back to defaults
        return cls(cls._get_default_config())
    
    @classmethod
    def get_instance(cls) -> 'Config':
        """Get singleton configuration instance.
        
        Returns:
            Config: The singleton configuration instance
        """
        if cls._instance is None:
            cls._instance = cls.load_config()
        return cls._instance
    
    @classmethod
    def set_instance(cls, config: 'Config'):
        """Set the singleton configuration instance.
        
        Args:
            config: Configuration instance to set as singleton
        """
        cls._instance = config
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration values.
        
        Returns:
            dict: Default configuration dictionary
        """
        return {
            'paths': {
                'output_directory': './output',
                'data_files': {
                    'bilcontour': 'bilcontour.txt',
                    'projection_list': 'outputlist_projection.txt'
                }
            },
            'simulation_parameters': {
                'particle_types': {
                    'gas': 'PartType0',
                    'dm': 'PartType1',
                    'star': 'PartType4'
                },
                'gadget_units': {
                    'UnitLength_in_cm': 3.08568e+21,
                    'UnitMass_in_g': 1.989e+43,
                    'UnitVelocity_in_cm_per_s': 100000
                },
                'arepo_units': {
                    'UnitLength_in_cm': 3.0868e+24,
                    'UnitMass_in_g': 1.989e+43,
                    'UnitVelocity_in_cm_per_s': 100000
                }
            },
            'plotting_defaults': {
                'figsize': [8, 8],
                'yt_figsize': 12,
                'fontsize': 12,
                'dpi': 300,
                'default_colormap': 'algae',
                'save_plots': True,
                'show_plots': False
            },
            'analysis_options': {
                'fig_width': 30,
                'buffer_size': 800,
                'young_star_age': 20,
                'low_res_mock': 750,
                'star_fig_size': 80,
                'star_buffer_size': 400,
                'z_solar': 0.0204
            },
            'cosmology': {
                'hubble_constant': 0.702,
                'omega_matter': 0.272,
                'omega_lambda': 0.728
            },
            'logging': {
                'level': 'INFO',
                'verbose_level': 15
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
                     (e.g., 'paths.output_directory')
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('simulation_parameters.particle_types.gas')
            'PartType0'
        """
        keys = key_path.split('.')
        value = self.data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If key doesn't exist
        """
        return self.data[key]


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.
    
    This is a convenience function that calls Config.load_config().
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config: Configuration object
    """
    return Config.load_config(config_path)


# Global configuration instance getter
def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Config: The singleton configuration instance
    """
    return Config.get_instance()
