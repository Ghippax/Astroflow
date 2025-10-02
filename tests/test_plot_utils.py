"""Tests for plotting utility functions."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch

from cosmo_analysis.plot.utils import (
    binFunctionCilBins, binFunctionSphBins, binFunctionCilBinsSFR,
    makeZbinFun, binFunctionSphVol, aFromT, makeMovie
)
from cosmo_analysis.config import Config


class TestBinningFunctions:
    """Tests for various binning functions."""
    
    def test_bin_function_cil_bins(self, reset_config):
        """Test cylindrical bin surface density calculation."""
        cil = np.array([1.0, 2.0, 3.0])  # kpc
        bin_data = np.array([100.0, 200.0, 300.0])
        
        result = binFunctionCilBins(cil, bin_data)
        
        assert len(result) == len(cil)
        assert all(isinstance(x, (float, np.floating)) for x in result)
        # Surface density should be positive
        assert all(x > 0 for x in result)
    
    def test_bin_function_sph_bins(self, reset_config):
        """Test spherical bin volume density calculation."""
        cil = np.array([1.0, 2.0, 3.0])  # kpc
        bin_data = np.array([100.0, 200.0, 300.0])
        
        result = binFunctionSphBins(cil, bin_data)
        
        assert len(result) == len(cil)
        assert all(isinstance(x, (float, np.floating)) for x in result)
        # Volume density should be positive
        assert all(x > 0 for x in result)
    
    def test_bin_function_cil_bins_sfr(self, reset_config):
        """Test SFR surface density calculation for cylindrical bins."""
        cil = np.array([1.0, 2.0, 3.0])  # kpc
        bin_data = np.array([100.0, 200.0, 300.0])
        
        result = binFunctionCilBinsSFR(cil, bin_data, youngStarAge=20)
        
        assert len(result) == len(cil)
        assert all(isinstance(x, (float, np.floating)) for x in result)
        # SFR surface density should be positive
        assert all(x > 0 for x in result)
    
    def test_bin_function_cil_bins_sfr_uses_config(self, temp_config_file, reset_config):
        """Test that SFR binning uses config for young star age."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        cil = np.array([1.0, 2.0, 3.0])
        bin_data = np.array([100.0, 200.0, 300.0])
        
        result = binFunctionCilBinsSFR(cil, bin_data, config=config)
        
        assert len(result) == len(cil)
        assert all(x > 0 for x in result)
    
    def test_make_z_bin_fun(self, reset_config):
        """Test vertical binning function creator."""
        rlimit = 10.0  # kpc
        
        bin_func = makeZbinFun(rlimit)
        
        z_data = np.array([0.0, 1.0, 2.0])
        bin_data = np.array([100.0, 200.0, 300.0])
        
        result = bin_func(z_data, bin_data)
        
        assert len(result) == len(z_data)
        assert all(isinstance(x, (float, np.floating)) for x in result)
    
    def test_bin_function_sph_vol(self, reset_config):
        """Test spherical volume binning."""
        # Create mock bin object
        bin_obj = Mock()
        bin_obj.x = np.array([0.0, 1.0, 2.0, 3.0])  # Bin edges
        bin_obj.d = np.array([100.0, 200.0, 300.0])  # Bin data (one less than edges)
        
        result = binFunctionSphVol(None, bin_obj)
        
        assert len(result) == len(bin_obj.d)
        assert all(isinstance(x, (float, np.floating)) for x in result)


class TestCosmology:
    """Tests for cosmological calculations."""
    
    def test_a_from_t_basic(self, reset_config):
        """Test scale factor calculation from time."""
        time = 1000.0  # Myr
        
        result = aFromT(time)
        
        assert isinstance(result, (float, np.floating))
        # Scale factor should be positive and less than 1 for early times
        assert result > 0
    
    def test_a_from_t_zero_time(self, reset_config):
        """Test that very small times return 0."""
        time = 0.01  # Less than eps=0.1
        
        result = aFromT(time)
        
        assert result == 0
    
    def test_a_from_t_uses_config(self, temp_config_file, reset_config):
        """Test that aFromT uses cosmology parameters from config."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        time = 1000.0
        
        result = aFromT(time, config=config)
        
        assert isinstance(result, (float, np.floating))
        assert result > 0


class TestMakeMovie:
    """Tests for animation creation."""
    
    def test_make_movie_creates_animation(self, tmp_path, reset_config):
        """Test that makeMovie creates an animation file."""
        # Create simple dummy frames as numpy arrays
        frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)]
        
        save_path = str(tmp_path)
        message = "test_animation"
        
        with patch('matplotlib.animation.FuncAnimation.save') as mock_save:
            result = makeMovie(frames, interval=50, saveFigPath=save_path, message=message)
            
            # Animation should be created
            mock_save.assert_called_once()
        
        plt.close('all')
    
    def test_make_movie_uses_config_output_dir(self, temp_config_file, tmp_path, reset_config):
        """Test that makeMovie uses output directory from config."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        # Create minimal frames
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)]
        
        with patch('os.makedirs'), patch('matplotlib.animation.FuncAnimation.save') as mock_save:
            makeMovie(frames, message="test", config=config)
            mock_save.assert_called_once()
        
        plt.close('all')
    
    def test_make_movie_warns_no_message(self, tmp_path, reset_config):
        """Test that makeMovie warns when no message is provided."""
        from cosmo_analysis import log as cosmo_log
        
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(2)]
        save_path = str(tmp_path)
        
        with patch('matplotlib.animation.FuncAnimation.save'), \
             patch.object(cosmo_log.logger, 'warning') as mock_warning:
            makeMovie(frames, saveFigPath=save_path, message=None)
            mock_warning.assert_called()
            assert any("TITLE NOT SPECIFIED" in str(call) for call in mock_warning.call_args_list)
        
        plt.close('all')


class TestConfigIntegration:
    """Test config integration in utils module."""
    
    def test_all_functions_accept_config(self, temp_config_file, reset_config):
        """Test that config can be passed to all relevant functions."""
        config = Config.load_config(str(temp_config_file))
        
        # Test binning functions
        cil = np.array([1.0, 2.0])
        bin_data = np.array([100.0, 200.0])
        
        result = binFunctionCilBinsSFR(cil, bin_data, config=config)
        assert len(result) == len(cil)
        
        # Test cosmology
        result = aFromT(1000.0, config=config)
        assert result > 0
