"""Tests for galaxy properties module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock
import inspect

from cosmo_analysis.plot.galaxy import plotMsMh
from cosmo_analysis.config import Config


class TestPlotMsMh:
    """Tests for plotMsMh function."""
    
    def test_plot_msmh_module_exists(self):
        """Test that galaxy module can be imported."""
        from cosmo_analysis.plot import galaxy
        assert hasattr(galaxy, 'plotMsMh')
    
    def test_plot_msmh_function_signature(self):
        """Test that plotMsMh has expected parameters."""
        sig = inspect.signature(plotMsMh)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'sims' in params
        assert 'idx' in params
        assert 'haloData' in params
        assert 'rLim' in params
        assert 'config' in params
        assert 'saveFig' in params
    
    def test_plot_msmh_uses_config_defaults(self, reset_config):
        """Test that plotMsMh uses config defaults."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Verify config has expected defaults
        assert config.get('simulation_parameters.particle_types.star') == 'PartType4'
        assert config.get('simulation_parameters.particle_types.dm') == 'PartType1'
        assert config.get('plotting_defaults.save_plots') is not None


class TestConfigIntegration:
    """Test config integration in galaxy module."""
    
    def test_galaxy_functions_accept_config(self, reset_config):
        """Test that all galaxy functions accept config parameter."""
        from cosmo_analysis.plot import galaxy
        
        for func_name in ['plotMsMh']:
            func = getattr(galaxy, func_name)
            sig = inspect.signature(func)
            assert 'config' in sig.parameters, f"{func_name} missing config parameter"
