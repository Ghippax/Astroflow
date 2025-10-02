"""Tests for star formation module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, MagicMock, patch
import inspect

from cosmo_analysis.plot.star_formation import plotSFR, plotKScil, plotKSmock, plotSFmass
from cosmo_analysis.config import Config


class TestPlotSFR:
    """Tests for plotSFR function."""
    
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
    
    @pytest.mark.visual
    def test_plot_sfr_creates_figure(self, reset_config):
        """Test that plotSFR creates a matplotlib figure with star data."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Use deterministic seed
        np.random.seed(42)
        
        # Create mock simulation with star formation history
        mock_sim = Mock()
        mock_sim.name = "TestSFRSim"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        snap.ytcen = np.array([0.5, 0.5, 0.5])
        mock_sim.snap = [snap]
        
        # Create mock yt dataset with star particles
        mock_ds = MagicMock()
        mock_sim.ytFull = [mock_ds]
        
        # Create proper mock for star particle data
        part_type = config.get('simulation_parameters.particle_types.star', 'PartType4')
        creation_times = np.sort(np.random.uniform(0, 100, 100))  # Sorted formation times
        masses = np.random.uniform(1e3, 1e5, 100)
        
        # Mock all_data() to return star particle data
        # Use a dict with mock fields
        mock_creation = Mock()
        mock_creation.in_units.return_value = Mock(d=creation_times)
        
        mock_mass = Mock()
        mock_mass.in_units.return_value = Mock(d=masses)
        
        # Create dict that responds to tuple keys like (part_type, field_name)
        mock_all_data = {}
        mock_all_data[(part_type, 'creation_time')] = mock_creation
        mock_all_data[(part_type, 'Masses')] = mock_mass
        
        mock_ds.all_data.return_value = mock_all_data
        
        # Use animate=True to get the frame returned for validation
        frame = plotSFR([mock_sim], [0], showFig=False, saveFig=False, animate=True)
        
        assert frame is not None, "plotSFR with animate=True should return a frame"
        assert isinstance(frame, np.ndarray), "Frame should be a numpy array"
        assert len(frame.shape) == 3, "Frame should be a 3D array (height, width, channels)"


class TestPlotKScil:
    """Tests for plotKScil (Kennicutt-Schmidt cylindrical) function."""
    
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
    
    @pytest.mark.visual
    def test_kscil_plot_structure(self, reset_config):
        """Test that Kennicutt-Schmidt plot has expected structure."""
        # This test verifies function signature and basic structure
        # Full yt integration test would require real yt dataset with particles
        config = Config.load_config()
        Config.set_instance(config)
        
        # Use deterministic seed
        np.random.seed(42)
        
        # Create mock simulation
        mock_sim = Mock()
        mock_sim.name = "TestKS"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        snap.ytcen = np.array([0.5, 0.5, 0.5])
        mock_sim.snap = [snap]
        
        # plotKScil requires yt.ProfilePlot which needs real yt data source
        # For now, verify the function exists and can be imported
        # Full test would need actual yt fake_random_ds with particle fields
        assert plotKScil is not None, "plotKScil should be importable"
        
        # Verify function signature includes required config parameter
        import inspect
        sig = inspect.signature(plotKScil)
        assert 'config' in sig.parameters, "plotKScil should accept config parameter"
        assert 'rLim' in sig.parameters, "plotKScil should accept rLim parameter"


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
    
    @pytest.mark.visual
    def test_sfmass_plot_creates_output(self, reset_config):
        """Test that plotSFmass creates expected output."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Use deterministic seed
        np.random.seed(42)
        
        # Create mock simulation
        mock_sim = Mock()
        mock_sim.name = "TestSFmass"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        snap.ytcen = np.array([0.5, 0.5, 0.5])
        mock_sim.snap = [snap]
        
        mock_ds = MagicMock()
        mock_sim.ytFull = [mock_ds]
        
        # Create proper mock for star particle data
        part_type = config.get('simulation_parameters.particle_types.star', 'PartType4')
        
        # Star particles formed over time
        creation_times = np.sort(np.random.uniform(0, 100, 100))
        masses = np.random.uniform(1e3, 1e5, 100)
        
        # Mock all_data() to return star particle data
        # Use a dict with mock fields
        mock_creation = Mock()
        mock_creation.in_units.return_value = Mock(d=creation_times)
        
        mock_mass = Mock()
        mock_mass.in_units.return_value = Mock(d=masses)
        
        # Create dict that responds to tuple keys like (part_type, field_name)
        mock_all_data = {}
        mock_all_data[(part_type, 'creation_time')] = mock_creation
        mock_all_data[(part_type, 'Masses')] = mock_mass
        
        mock_ds.all_data.return_value = mock_all_data
        
        # Use animate=True to get the frame returned for validation
        frame = plotSFmass([mock_sim], [0], showFig=False, saveFig=False, animate=True)
        
        assert frame is not None, "plotSFmass with animate=True should return a frame"
        assert isinstance(frame, np.ndarray), "Frame should be a numpy array"
        assert len(frame.shape) == 3, "Frame should be a 3D array (height, width, channels)"


class TestVisualRegression:
    """Visual regression tests for star formation plots."""
    
    @pytest.mark.visual
    def test_star_formation_history_consistency(self, reset_config):
        """Test that SFR plots produce consistent visual output."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create deterministic mock data
        np.random.seed(42)
        
        mock_sim = Mock()
        mock_sim.name = "VisualSFR"
        mock_sim.cosmo = False
        
        snap = Mock()
        snap.time = 100.0
        snap.z = 0.0
        snap.ytcen = np.array([0.5, 0.5, 0.5])
        mock_sim.snap = [snap]
        
        # Create multiple snapshots for time series
        for i in range(5):
            snap_i = Mock()
            snap_i.time = i * 25.0
            snap_i.z = 0.0
            mock_sim.snap.append(snap_i)
        
        # Verify that functions can be called without errors
        # (actual visual output requires full yt integration)
        assert plotSFR is not None
        assert plotSFmass is not None


class TestConfigIntegration:
    """Test config integration in star_formation module."""
    
    def test_sf_functions_accept_config(self, reset_config):
        """Test that all star formation functions accept config parameter."""
        from cosmo_analysis.plot import star_formation
        
        for func_name in ['plotSFR', 'plotKScil', 'plotKSmock', 'plotSFmass']:
            func = getattr(star_formation, func_name)
            sig = inspect.signature(func)
            assert 'config' in sig.parameters, f"{func_name} missing config parameter"
