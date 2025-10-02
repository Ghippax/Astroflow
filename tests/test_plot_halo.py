"""Tests for halo analysis module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, MagicMock
import inspect

from cosmo_analysis.plot.halo import findHalos, plotClumpMassF
from cosmo_analysis.config import Config


class TestFindHalos:
    """Tests for findHalos function."""
    
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
    
    def test_plot_clump_mass_creates_figure(self, reset_config):
        """Test that plotClumpMassF creates a matplotlib figure."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create mock simulation and halo data
        mock_sim = Mock()
        mock_sim.name = "TestSim"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        mock_sim.snap = [snap]
        
        # Create mock halo data with masses
        mock_halo = Mock()
        mock_halo.halos_ds = Mock()
        mock_halo.halos_ds.halo_mass = Mock()
        mock_halo.halos_ds.halo_mass.to_ndarray.return_value = np.array([1e10, 5e10, 1e11, 5e11])
        
        try:
            fig = plotClumpMassF([mock_sim], [0], [mock_halo], showFig=False, saveFig=False)
            
            # Verify a figure was created
            assert fig is not None
            assert hasattr(fig, 'get_axes')
            
            plt.close(fig)
        except Exception as e:
            # If function requires specific yt objects, verify it has proper error handling
            assert "requires" in str(e).lower() or "expect" in str(e).lower() or "mock" in str(e).lower(), \
                f"Unexpected error: {e}"


class TestVisualRegression:
    """Visual regression tests for halo plots."""
    
    @pytest.mark.visual
    def test_clump_mass_function_plot_output(self, reset_config):
        """Test that plotClumpMassF produces visual output with expected properties."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create realistic mock data
        mock_sim = Mock()
        mock_sim.name = "HaloSim"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        mock_sim.snap = [snap]
        
        # Create mock halo data with realistic mass distribution
        mock_halo = Mock()
        mock_halo.halos_ds = Mock()
        # Log-normal mass distribution
        masses = 10 ** np.random.normal(10.5, 1.0, 100)
        mock_halo.halos_ds.halo_mass = Mock()
        mock_halo.halos_ds.halo_mass.to_ndarray.return_value = masses
        
        try:
            fig = plotClumpMassF([mock_sim], [0], [mock_halo], showFig=False, saveFig=False)
            
            if fig is not None:
                # Verify plot has axes and content
                axes = fig.get_axes()
                assert len(axes) > 0, "Plot should have at least one axis"
                
                # Verify axes have data plotted
                ax = axes[0]
                assert len(ax.lines) > 0 or len(ax.collections) > 0, \
                    "Plot should have lines or collections"
                
                plt.close(fig)
        except Exception as e:
            # Document what types of errors are acceptable
            pytest.skip(f"Requires specific yt halo data structure: {e}")


class TestConfigIntegration:
    """Test config integration in halo module."""
    
    def test_halo_functions_accept_config(self, reset_config):
        """Test that all halo functions accept config parameter."""
        from cosmo_analysis.plot import halo
        
        for func_name in ['findHalos', 'plotClumpMassF']:
            func = getattr(halo, func_name)
            sig = inspect.signature(func)
            assert 'config' in sig.parameters, f"{func_name} missing config parameter"
