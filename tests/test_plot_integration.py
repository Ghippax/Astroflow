"""Integration tests for plotting functions.

These tests verify that plotting functions work together correctly
with the config system and produce expected outputs.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import os

from cosmo_analysis.plot.base import handleFig, saveFrame, setLegend
from cosmo_analysis.plot.utils import makeMovie, aFromT
from cosmo_analysis.config import Config


@pytest.mark.integration
class TestPlottingWorkflow:
    """Test complete plotting workflows."""
    
    def test_save_frame_to_movie_workflow(self, tmp_path, temp_config_file, reset_config):
        """Test creating frames and combining them into a movie."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        # Create frames
        frames = []
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot([0, 1], [i, i+1])
            ax.set_title(f"Frame {i}")
            
            frame = saveFrame(fig, config=config)
            frames.append(frame)
        
        # Create movie from frames
        save_path = str(tmp_path)
        
        with patch('matplotlib.animation.FuncAnimation.save') as mock_save:
            result = makeMovie(frames, interval=50, saveFigPath=save_path, 
                             message="test_movie", config=config)
            mock_save.assert_called_once()
        
        plt.close('all')
    
    def test_config_propagation_through_workflow(self, temp_config_file, tmp_path, reset_config):
        """Test that config settings propagate correctly through functions."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        # Verify config values are accessible
        assert config.get('plotting_defaults.save_plots') == False
        assert config.get('plotting_defaults.show_plots') == False
        assert config.get('analysis_options.fig_width') == 30
        
        # Create a figure and use handleFig with config
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        save_path = str(tmp_path)
        
        # This should respect config settings
        result = handleFig(fig, [False, False, True], "test_config", 
                          saveFigPath=save_path, config=config)
        
        # Check file was created
        expected_file = tmp_path / "test_config.png"
        assert expected_file.exists()
        
        plt.close('all')


@pytest.mark.integration
class TestSimulationPlotting:
    """Test plotting functions with mock simulation data."""
    
    def test_legend_with_multiple_simulations(self, mock_simulation, reset_config):
        """Test creating legends for multiple simulations."""
        # Create second mock simulation
        sim2 = Mock()
        sim2.name = "MockSim2"
        sim2.cosmo = True
        
        snap2 = Mock()
        snap2.time = 200.0
        snap2.z = 1.5
        
        sim2.snap = [snap2]
        
        # Test legend creation
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="data1")
        ax.plot([0, 1], [1, 2], label="data2")
        
        setLegend(ax, [mock_simulation, sim2], [0, 0])
        
        legend = ax.get_legend()
        assert legend is not None
        
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert len(legend_texts) == 2
        assert "MockSim" in legend_texts[0]
        assert "MockSim2" in legend_texts[1]
        assert "100 Myr" in legend_texts[0]
        assert "z=1.5" in legend_texts[1]
        
        plt.close('all')
    
    def test_multiple_simulation_animation_frames(self, mock_simulation, tmp_path, reset_config):
        """Test creating animation frames from multiple simulations."""
        frames = []
        
        # Create frames for different time steps
        for i in range(3):
            fig, ax = plt.subplots()
            
            # Mock updating simulation time
            mock_simulation.snap[0].time = 100.0 + i * 50
            
            ax.plot([0, 1], [i, i+1])
            ax.set_title(f"{mock_simulation.name} at {mock_simulation.snap[0].time} Myr")
            
            frame = saveFrame(fig)
            frames.append(frame)
        
        # Verify frames were created
        assert len(frames) == 3
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
        
        plt.close('all')


@pytest.mark.integration
class TestCosmologyIntegration:
    """Test cosmology-related functions in context."""
    
    def test_cosmological_time_series(self, temp_config_file, reset_config):
        """Test cosmological calculations across time series."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        # Create time series
        times = [100, 500, 1000, 2000, 5000]  # Myr
        
        # Calculate scale factors
        scale_factors = [aFromT(t, config=config) for t in times]
        
        # Verify scale factors make sense
        assert len(scale_factors) == len(times)
        for a in scale_factors:
            assert isinstance(a, (float, np.floating))
            assert a >= 0
            assert a <= 1  # Scale factor should be <= 1 for finite time
        
        # Scale factor should increase with time (universe expands)
        for i in range(len(scale_factors) - 1):
            if scale_factors[i] > 0 and scale_factors[i+1] > 0:
                # Both non-zero, should be increasing
                assert scale_factors[i+1] > scale_factors[i]


@pytest.mark.integration
class TestConfigDefaultBehavior:
    """Test that functions work with default config when none provided."""
    
    def test_functions_work_without_explicit_config(self, tmp_path, reset_config):
        """Test that functions can work with default config."""
        # Reset to ensure we're using defaults
        from cosmo_analysis.config import Config
        Config._instance = None
        
        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        # Save without explicit config (should use defaults)
        save_path = str(tmp_path)
        result = handleFig(fig, [False, False, True], "test_default", 
                          saveFigPath=save_path)
        
        # Should create file
        expected_file = tmp_path / "test_default.png"
        assert expected_file.exists()
        
        plt.close('all')
    
    def test_cosmology_with_default_config(self, reset_config):
        """Test cosmology functions with default config."""
        from cosmo_analysis.config import Config
        Config._instance = None
        
        # Should work with default cosmology parameters
        result = aFromT(1000.0)
        
        assert isinstance(result, (float, np.floating))
        assert result > 0


@pytest.mark.visual
class TestVisualOutput:
    """Tests for visual output quality and consistency."""
    
    def test_saved_figure_properties(self, tmp_path, temp_config_file, reset_config):
        """Test that saved figures have correct properties."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_title("Test Figure")
        
        save_path = str(tmp_path)
        handleFig(fig, [False, False, True], "visual_test", 
                 saveFigPath=save_path, config=config)
        
        # Check file exists and has content
        output_file = tmp_path / "visual_test.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        
        # Optionally load and check image properties
        from PIL import Image
        img = Image.open(output_file)
        assert img.mode in ['RGB', 'RGBA']
        assert img.size[0] > 0
        assert img.size[1] > 0
        
        plt.close('all')
    
    def test_frame_capture_consistency(self, reset_config):
        """Test that frame capture produces consistent results."""
        frames = []
        
        # Create identical figures
        for i in range(3):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            frame = saveFrame(fig)
            frames.append(frame)
        
        # All frames should have same shape
        shapes = [f.shape for f in frames]
        assert all(s == shapes[0] for s in shapes)
        
        # Frames should be similar (not necessarily identical due to rendering)
        for i in range(len(frames) - 1):
            assert frames[i].shape == frames[i+1].shape
        
        plt.close('all')
