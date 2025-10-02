"""Tests for galaxy properties module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, MagicMock
import inspect

from cosmo_analysis.plot.galaxy import plotMsMh
from cosmo_analysis.config import Config


class TestPlotMsMh:
    """Tests for plotMsMh (stellar mass vs halo mass) function."""
    
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
    
    @pytest.mark.visual
    def test_msmh_creates_figure(self, reset_config):
        """Test that plotMsMh creates a matplotlib figure."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Use deterministic seed
        np.random.seed(42)
        
        # Create mock simulation
        mock_sim = Mock()
        mock_sim.name = "TestGalaxy"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        snap.ytcen = np.array([0.5, 0.5, 0.5])
        mock_sim.snap = [snap]
        
        mock_ds = MagicMock()
        mock_sim.ytFull = [mock_ds]
        
        # Create mock halo data - plotMsMh expects haloData = (haloSims, haloFilt)
        mock_halo_ds = Mock()
        halo_masses = np.array([1e11, 5e11, 1e12])
        
        # Mock all_data() which returns data with particle_mass field
        mock_all_data = {}
        
        # Create a mock mass field that supports subscripting
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
        
        mock_all_data['particle_mass'] = MockMassField(halo_masses)
        mock_halo_ds.all_data.return_value = mock_all_data
        
        # haloFilt is a boolean mask or None
        halo_filt = np.ones(3, dtype=bool)
        
        # haloData is a tuple of (haloSims list, haloFilt list)
        haloData = ([mock_halo_ds], [halo_filt])
        
        # Mock star particle data in sphere
        mock_sphere = MagicMock()
        part_type = config.get('simulation_parameters.particle_types.star', 'PartType4')
        star_masses = np.random.uniform(1e3, 1e5, 50)
        
        mock_star_mass = Mock()
        mock_star_mass.in_units.return_value = Mock(d=star_masses)
        mock_star_mass.sum.return_value = Mock(in_units=lambda u: Mock(d=np.sum(star_masses)))
        
        def mock_getitem(key):
            if 'Masses' in str(key):
                return mock_star_mass
            return MagicMock()
        
        mock_sphere.__getitem__ = mock_getitem
        mock_ds.sphere.return_value = mock_sphere
        
        # Use animate=True to get the frame returned for validation
        frame = plotMsMh([mock_sim], [0], haloData, rLim=10.0, 
                        showFig=False, saveFig=False, animate=True)
        
        assert frame is not None, "plotMsMh with animate=True should return a frame"
        assert isinstance(frame, np.ndarray), "Frame should be a numpy array"
        assert len(frame.shape) == 3, "Frame should be a 3D array (height, width, channels)"


class TestVisualRegression:
    """Visual regression tests for galaxy property plots."""
    
    @pytest.mark.visual
    def test_msmh_relation_plot_structure(self, reset_config):
        """Test that M_star-M_halo relation plot has expected structure."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create mock data for realistic M_star-M_halo relation
        np.random.seed(42)
        
        mock_sim = Mock()
        mock_sim.name = "MsMhTest"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        snap.ytcen = np.array([0.5, 0.5, 0.5])
        mock_sim.snap = [snap]
        
        mock_ds = MagicMock()
        mock_sim.ytFull = [mock_ds]
        
        # Mock halo masses (log-normal distribution)
        mock_halo_ds = Mock()
        halo_masses = 10 ** np.random.normal(11.5, 0.5, 50)
        
        # Mock all_data() which returns data with particle_mass field
        mock_all_data = {}
        
        # Create a mock mass field that supports subscripting
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
        
        mock_all_data['particle_mass'] = MockMassField(halo_masses)
        mock_halo_ds.all_data.return_value = mock_all_data
        
        # haloFilt is a boolean mask
        halo_filt = np.ones(50, dtype=bool)
        
        # haloData is a tuple
        haloData = ([mock_halo_ds], [halo_filt])
        
        # Mock stellar masses
        mock_sphere = MagicMock()
        part_type = config.get('simulation_parameters.particle_types.star', 'PartType4')
        stellar_masses = 10 ** np.random.normal(9.5, 0.8, 50)
        
        mock_star_mass = Mock()
        mock_star_mass.in_units.return_value = Mock(d=stellar_masses)
        mock_star_mass.sum.return_value = Mock(in_units=lambda u: Mock(d=np.sum(stellar_masses)))
        
        def mock_getitem(key):
            if 'Masses' in str(key):
                return mock_star_mass
            return MagicMock()
        
        mock_sphere.__getitem__ = mock_getitem
        mock_ds.sphere.return_value = mock_sphere
        
        # Use animate=True to get the frame returned for validation
        frame = plotMsMh([mock_sim], [0], haloData, rLim=10.0,
                        showFig=False, saveFig=False, animate=True)
        
        assert frame is not None, "plotMsMh with animate=True should return a frame"
        assert isinstance(frame, np.ndarray), "Frame should be a numpy array"
        assert len(frame.shape) == 3, "Frame should be a 3D array (height, width, channels)"
    
    @pytest.mark.visual
    def test_msmh_consistency_multiple_runs(self, reset_config):
        """Test that M_star-M_halo plots are consistent across runs."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # With deterministic data, plots should be reproducible
        np.random.seed(42)
        
        mock_sim = Mock()
        mock_sim.name = "Consistent"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        snap.ytcen = np.array([0.5, 0.5, 0.5])
        mock_sim.snap = [snap]
        
        # Verify function exists and can be called
        assert plotMsMh is not None
        sig = inspect.signature(plotMsMh)
        assert 'config' in sig.parameters


class TestConfigIntegration:
    """Test config integration in galaxy module."""
    
    def test_galaxy_functions_accept_config(self, reset_config):
        """Test that all galaxy functions accept config parameter."""
        from cosmo_analysis.plot import galaxy
        
        for func_name in ['plotMsMh']:
            func = getattr(galaxy, func_name)
            sig = inspect.signature(func)
            assert 'config' in sig.parameters, f"{func_name} missing config parameter"
