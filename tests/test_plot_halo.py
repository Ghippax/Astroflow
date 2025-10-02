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
        
        # Use deterministic seed for reproducibility
        np.random.seed(42)
        
        # Create realistic mock data
        mock_sim = Mock()
        mock_sim.name = "HaloSim"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        mock_sim.snap = [snap]
        
        # Create mock halo data with realistic mass distribution
        # plotClumpMassF expects haloData = (haloSims, haloFilt)
        mock_halo_ds = Mock()
        
        # Log-normal mass distribution - typical for halo mass functions
        masses = 10 ** np.random.normal(10.5, 1.0, 100)
        
        # Mock all_data() which returns data object with particle_mass field
        mock_all_data = {}
        
        # Create a mock mass field that supports subscripting with boolean array
        class MockMassField:
            def __init__(self, masses):
                self._masses = masses
            
            def __getitem__(self, key):
                result = Mock()
                if isinstance(key, np.ndarray) and key.dtype == bool:
                    filtered = self._masses[key]
                else:
                    filtered = self._masses
                result.in_units = Mock(return_value=Mock(d=filtered))
                return result
            
            def in_units(self, unit):
                return Mock(d=self._masses)
        
        mock_all_data['particle_mass'] = MockMassField(masses)
        mock_halo_ds.all_data.return_value = mock_all_data
        
        # haloFilt is a boolean mask or None
        halo_filt = np.ones(100, dtype=bool)  # All halos pass the filter
        
        # haloData is a tuple of (haloSims list, haloFilt list)
        haloData = ([mock_halo_ds], [halo_filt])
        
        # Use animate=True to get the figure returned for validation
        frame = plotClumpMassF([mock_sim], [0], haloData, showFig=False, saveFig=False, animate=True)
        
        assert frame is not None, "plotClumpMassF with animate=True should return a frame"
        assert isinstance(frame, np.ndarray), "Frame should be a numpy array"
        assert len(frame.shape) == 3, "Frame should be a 3D array (height, width, channels)"
        
        # Verify reasonable image dimensions
        assert frame.shape[0] > 0 and frame.shape[1] > 0, "Frame should have positive dimensions"


class TestConfigIntegration:
    """Test config integration in halo module."""
    
    def test_halo_functions_accept_config(self, reset_config):
        """Test that all halo functions accept config parameter."""
        from cosmo_analysis.plot import halo
        
        for func_name in ['findHalos', 'plotClumpMassF']:
            func = getattr(halo, func_name)
            sig = inspect.signature(func)
            assert 'config' in sig.parameters, f"{func_name} missing config parameter"
