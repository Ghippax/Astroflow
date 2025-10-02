"""Tests for profile plotting module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock
import inspect

from cosmo_analysis.plot.profiles import plotBinned, plotRotDisp
from cosmo_analysis.config import Config


class TestPlotBinned:
    """Tests for plotBinned function."""
    
    def test_plot_binned_module_exists(self):
        """Test that profiles module can be imported."""
        from cosmo_analysis.plot import profiles
        assert hasattr(profiles, 'plotBinned')
    
    def test_plot_binned_function_signature(self):
        """Test that plotBinned has expected parameters."""
        sig = inspect.signature(plotBinned)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'sims' in params
        assert 'idx' in params
        assert 'binFields' in params
        assert 'config' in params
        assert 'saveFig' in params
    
    def test_plot_binned_uses_config_defaults(self, reset_config):
        """Test that plotBinned uses config defaults."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Verify config has expected defaults
        assert config.get('simulation_parameters.particle_types.gas') == 'PartType0'
        assert config.get('plotting_defaults.figsize') is not None


class TestPlotRotDisp:
    """Tests for plotRotDisp function."""
    
    def test_plot_rot_disp_module_exists(self):
        """Test that profiles module can be imported."""
        from cosmo_analysis.plot import profiles
        assert hasattr(profiles, 'plotRotDisp')
    
    def test_plot_rot_disp_function_signature(self):
        """Test that plotRotDisp has expected parameters."""
        sig = inspect.signature(plotRotDisp)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'sims' in params
        assert 'idx' in params
        assert 'nBins' in params
        assert 'rLim' in params
        assert 'part' in params
        assert 'config' in params
    
    def test_plot_rot_disp_uses_config_defaults(self, reset_config):
        """Test that plotRotDisp uses config defaults."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Verify config has expected defaults
        assert config.get('plotting_defaults.save_plots') is not None
        assert config.get('plotting_defaults.show_plots') is not None


class TestConfigIntegration:
    """Test config integration in profiles module."""
    
    def test_profiles_functions_accept_config(self, reset_config):
        """Test that all profile functions accept config parameter."""
        from cosmo_analysis.plot import profiles
        
        for func_name in ['plotBinned', 'plotRotDisp']:
            func = getattr(profiles, func_name)
            sig = inspect.signature(func)
            assert 'config' in sig.parameters, f"{func_name} missing config parameter"
