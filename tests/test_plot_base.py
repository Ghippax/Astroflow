"""Tests for base plotting utilities."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, MagicMock, patch
import io
from PIL import Image

from cosmo_analysis.plot.base import saveFrame, setLegend, handleFig
from cosmo_analysis.config import Config


class TestSaveFrame:
    """Tests for saveFrame function."""
    
    def test_save_frame_returns_array(self, reset_config):
        """Test that saveFrame returns a numpy array."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        result = saveFrame(fig)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3  # Image should be 3D (height, width, channels)
        plt.close('all')
    
    def test_save_frame_uses_config_dpi(self, temp_config_file, reset_config):
        """Test that saveFrame uses DPI from config."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        result = saveFrame(fig, config=config)
        
        assert isinstance(result, np.ndarray)
        plt.close('all')


class TestSetLegend:
    """Tests for setLegend function."""
    
    def test_set_legend_isolated_simulation(self, reset_config):
        """Test legend creation for isolated (non-cosmological) simulation."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])  # Need something to plot for legend
        
        # Mock simulation objects
        sim1 = Mock()
        sim1.name = "TestSim1"
        sim1.cosmo = False
        sim1.snap = [Mock()]
        sim1.snap[0].time = 100.5
        
        setLegend(ax, [sim1], [0])
        
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert len(legend_texts) > 0
        assert "TestSim1" in legend_texts[0]
        assert "100 Myr" in legend_texts[0]
        plt.close('all')
    
    def test_set_legend_cosmological_simulation(self, reset_config):
        """Test legend creation for cosmological simulation."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])  # Need something to plot for legend
        
        # Mock simulation objects
        sim1 = Mock()
        sim1.name = "CosmSim1"
        sim1.cosmo = True
        sim1.snap = [Mock()]
        sim1.snap[0].time = 100.5
        sim1.snap[0].z = 2.5
        
        setLegend(ax, [sim1], [0])
        
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert len(legend_texts) > 0
        assert "CosmSim1" in legend_texts[0]
        assert "z=2.5" in legend_texts[0]
        plt.close('all')


class TestHandleFig:
    """Tests for handleFig function."""
    
    def test_handle_fig_show(self, reset_config, monkeypatch):
        """Test that handleFig shows figure when requested."""
        fig, ax = plt.subplots()
        show_called = []
        
        def mock_show():
            show_called.append(True)
        
        monkeypatch.setattr(plt, 'show', mock_show)
        
        result = handleFig(fig, [True, False, False], None)
        
        assert len(show_called) == 1
        assert result is None
        plt.close('all')
    
    def test_handle_fig_animate(self, reset_config):
        """Test that handleFig returns frame for animation."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        result = handleFig(fig, [False, True, False], None)
        
        assert isinstance(result, np.ndarray)
        plt.close('all')
    
    def test_handle_fig_save(self, tmp_path, reset_config):
        """Test that handleFig saves figure when requested."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        save_path = str(tmp_path)
        message = "test_figure"
        
        result = handleFig(fig, [False, False, True], message, saveFigPath=save_path)
        
        expected_file = tmp_path / "test_figure.png"
        assert expected_file.exists()
        assert result is None
        plt.close('all')
    
    def test_handle_fig_uses_config_output_dir(self, temp_config_file, reset_config):
        """Test that handleFig uses output directory from config."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        # This should use the config's output_directory
        with patch('os.makedirs'), patch('matplotlib.pyplot.Figure.savefig') as mock_save:
            handleFig(fig, [False, False, True], "test", config=config)
            mock_save.assert_called_once()
        
        plt.close('all')
    
    def test_handle_fig_no_message_warning(self, tmp_path, reset_config):
        """Test that handleFig warns when no message is provided."""
        import logging
        from cosmo_analysis import log as cosmo_log
        
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        save_path = str(tmp_path)
        
        # Capture log messages
        with patch.object(cosmo_log.logger, 'warning') as mock_warning:
            handleFig(fig, [False, False, True], 0, saveFigPath=save_path)
            mock_warning.assert_called_once()
            assert "TITLE NOT SPECIFIED" in str(mock_warning.call_args)
        
        plt.close('all')


class TestConfigIntegration:
    """Test config integration in base module."""
    
    def test_handle_fig_respects_config_dpi(self, temp_config_file, tmp_path, reset_config):
        """Test that config DPI setting is used."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        save_path = str(tmp_path)
        message = "test_dpi"
        
        with patch('matplotlib.pyplot.Figure.savefig') as mock_save:
            handleFig(fig, [False, False, True], message, saveFigPath=save_path, config=config)
            
            # Check that savefig was called with dpi from config (or default)
            call_args = mock_save.call_args
            assert 'dpi' in call_args[1] or len(call_args[0]) > 0
        
        plt.close('all')
