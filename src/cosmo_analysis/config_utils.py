"""Utility functions for working with configuration.

This module provides helper functions to make it easier to integrate
the configuration system throughout the codebase.
"""

from typing import Optional, Any
from .config import get_config


def get_particle_type(particle_name: str) -> str:
    """Get particle type string from configuration.
    
    Args:
        particle_name: Name of particle type ('gas', 'dm', 'star')
        
    Returns:
        str: Particle type string (e.g., 'PartType0')
        
    Example:
        >>> get_particle_type('gas')
        'PartType0'
    """
    config = get_config()
    default_types = {
        'gas': 'PartType0',
        'dm': 'PartType1', 
        'star': 'PartType4'
    }
    
    config_key = f'simulation_parameters.particle_types.{particle_name}'
    return config.get(config_key, default_types.get(particle_name, 'PartType0'))


def get_output_directory() -> str:
    """Get output directory from configuration.
    
    Returns:
        str: Path to output directory
    """
    config = get_config()
    return config.get('paths.output_directory', './output')


def get_plotting_default(key: str, default: Any = None) -> Any:
    """Get plotting default value from configuration.
    
    Args:
        key: Configuration key under plotting_defaults (e.g., 'figsize', 'dpi')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Example:
        >>> get_plotting_default('dpi', 300)
        300
    """
    config = get_config()
    return config.get(f'plotting_defaults.{key}', default)


def get_analysis_option(key: str, default: Any = None) -> Any:
    """Get analysis option from configuration.
    
    Args:
        key: Configuration key under analysis_options (e.g., 'fig_width')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    config = get_config()
    return config.get(f'analysis_options.{key}', default)


def get_data_file_path(file_key: str) -> Optional[str]:
    """Get path to a data file from configuration.
    
    Args:
        file_key: Key for data file (e.g., 'bilcontour', 'projection_list')
        
    Returns:
        str or None: Path to data file, or None if not configured
    """
    config = get_config()
    return config.get(f'paths.data_files.{file_key}')


def should_save_plots() -> bool:
    """Check if plots should be saved by default.
    
    Returns:
        bool: True if plots should be saved
    """
    return get_plotting_default('save_plots', True)


def should_show_plots() -> bool:
    """Check if plots should be shown by default.
    
    Returns:
        bool: True if plots should be shown
    """
    return get_plotting_default('show_plots', False)
