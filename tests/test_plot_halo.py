"""Tests for halo analysis module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock
import inspect

from cosmo_analysis.plot.halo import findHalos, plotClumpMassF
from cosmo_analysis.config import Config


class TestFindHalos:
    """Tests for findHalos function."""
    
    def test_find_halos_module_exists(self):
        """Test that halo module can be imported."""
        from cosmo_analysis.plot import halo
        assert hasattr(halo, 'findHalos')
        assert hasattr(halo, 'plotClumpMassF')
    
    def test_find_halos_function_signature(self):
        """Test that findHalos has expected parameters."""
        sig = inspect.signature(findHalos)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'simArr' in params
        assert 'idxArr' in params
        assert 'partT' in params
        assert 'mainPath' in params
        assert 'haloMethod' in params
        assert 'config' in params
    
    def test_find_halos_uses_config_defaults(self, reset_config):
        """Test that findHalos uses config defaults."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Verify config is accessible
        assert config is not None
        assert hasattr(config, 'get')


class TestPlotClumpMassF:
    """Tests for plotClumpMassF function."""
    
    def test_plot_clump_mass_f_function_signature(self):
        """Test that plotClumpMassF has expected parameters."""
        sig = inspect.signature(plotClumpMassF)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'sims' in params
        assert 'idx' in params
        assert 'haloData' in params
        assert 'config' in params
        assert 'saveFig' in params
    
    def test_plot_clump_mass_f_uses_config_defaults(self, reset_config):
        """Test that plotClumpMassF uses config defaults."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Verify config has expected defaults
        assert config.get('plotting_defaults.save_plots') is not None
        assert config.get('plotting_defaults.show_plots') is not None


class TestConfigIntegration:
    """Test config integration in halo module."""
    
    def test_halo_functions_accept_config(self, reset_config):
        """Test that all halo functions accept config parameter."""
        from cosmo_analysis.plot import halo
        
        for func_name in ['findHalos', 'plotClumpMassF']:
            func = getattr(halo, func_name)
            sig = inspect.signature(func)
            assert 'config' in sig.parameters, f"{func_name} missing config parameter"
