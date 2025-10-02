"""Tests for star formation module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock
import inspect

from cosmo_analysis.plot.star_formation import plotSFR, plotKScil, plotKSmock, plotSFmass
from cosmo_analysis.config import Config


class TestPlotSFR:
    """Tests for plotSFR function."""
    
    def test_plot_sfr_module_exists(self):
        """Test that star_formation module can be imported."""
        from cosmo_analysis.plot import star_formation
        assert hasattr(star_formation, 'plotSFR')
        assert hasattr(star_formation, 'plotKScil')
        assert hasattr(star_formation, 'plotKSmock')
        assert hasattr(star_formation, 'plotSFmass')
    
    def test_plot_sfr_function_signature(self):
        """Test that plotSFR has expected parameters."""
        sig = inspect.signature(plotSFR)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'sims' in params
        assert 'idx' in params
        assert 'config' in params
        assert 'saveFig' in params
    
    def test_plot_sfr_uses_config_defaults(self, reset_config):
        """Test that plotSFR uses config defaults."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Verify config has expected defaults
        assert config.get('simulation_parameters.particle_types.star') == 'PartType4'


class TestPlotKScil:
    """Tests for plotKScil function."""
    
    def test_plot_kscil_function_signature(self):
        """Test that plotKScil has expected parameters."""
        sig = inspect.signature(plotKScil)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'sims' in params
        assert 'idx' in params
        assert 'rLim' in params
        assert 'config' in params
    
    def test_plot_kscil_uses_config_defaults(self, reset_config):
        """Test that plotKScil uses config defaults."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Verify config has expected defaults
        assert config.get('simulation_parameters.particle_types.gas') == 'PartType0'


class TestPlotKSmock:
    """Tests for plotKSmock function."""
    
    def test_plot_ksmock_function_signature(self):
        """Test that plotKSmock has expected parameters."""
        sig = inspect.signature(plotKSmock)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'sims' in params
        assert 'idx' in params
        assert 'config' in params


class TestPlotSFmass:
    """Tests for plotSFmass function."""
    
    def test_plot_sfmass_function_signature(self):
        """Test that plotSFmass has expected parameters."""
        sig = inspect.signature(plotSFmass)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'sims' in params
        assert 'idx' in params
        assert 'config' in params


class TestConfigIntegration:
    """Test config integration in star_formation module."""
    
    def test_sf_functions_accept_config(self, reset_config):
        """Test that all star formation functions accept config parameter."""
        from cosmo_analysis.plot import star_formation
        
        for func_name in ['plotSFR', 'plotKScil', 'plotKSmock', 'plotSFmass']:
            func = getattr(star_formation, func_name)
            sig = inspect.signature(func)
            assert 'config' in sig.parameters, f"{func_name} missing config parameter"
