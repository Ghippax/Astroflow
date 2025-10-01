"""Tests for configuration utility functions."""

import pytest
from cosmo_analysis.config import Config
from cosmo_analysis.config_utils import (
    get_particle_type,
    get_output_directory,
    get_plotting_default,
    get_analysis_option,
    get_data_file_path,
    should_save_plots,
    should_show_plots
)


class TestParticleTypes:
    """Test particle type retrieval."""
    
    def test_get_gas_particle_type(self):
        """Test getting gas particle type."""
        particle_type = get_particle_type('gas')
        assert particle_type == 'PartType0'
    
    def test_get_dm_particle_type(self):
        """Test getting dark matter particle type."""
        particle_type = get_particle_type('dm')
        assert particle_type == 'PartType1'
    
    def test_get_star_particle_type(self):
        """Test getting star particle type."""
        particle_type = get_particle_type('star')
        assert particle_type == 'PartType4'


class TestPathRetrieval:
    """Test path retrieval functions."""
    
    def test_get_output_directory(self):
        """Test getting output directory."""
        output_dir = get_output_directory()
        assert output_dir is not None
        assert isinstance(output_dir, str)
    
    def test_get_data_file_path(self):
        """Test getting data file path."""
        bilcontour_path = get_data_file_path('bilcontour')
        assert bilcontour_path is not None
        assert isinstance(bilcontour_path, str)


class TestPlottingDefaults:
    """Test plotting default retrieval."""
    
    def test_get_plotting_default_with_existing_key(self):
        """Test getting existing plotting default."""
        figsize = get_plotting_default('figsize')
        assert figsize is not None
    
    def test_get_plotting_default_with_missing_key(self):
        """Test getting non-existent plotting default with fallback."""
        value = get_plotting_default('nonexistent_key', 'fallback')
        assert value == 'fallback'
    
    def test_should_save_plots(self):
        """Test save plots flag."""
        save = should_save_plots()
        assert isinstance(save, bool)
    
    def test_should_show_plots(self):
        """Test show plots flag."""
        show = should_show_plots()
        assert isinstance(show, bool)


class TestAnalysisOptions:
    """Test analysis option retrieval."""
    
    def test_get_analysis_option(self):
        """Test getting analysis option."""
        fig_width = get_analysis_option('fig_width')
        assert fig_width is not None
        assert isinstance(fig_width, (int, float))
    
    def test_get_analysis_option_with_default(self):
        """Test getting analysis option with default."""
        value = get_analysis_option('nonexistent_option', 42)
        assert value == 42


class TestConfigIntegration:
    """Integration tests with custom configuration."""
    
    def test_custom_config_particle_types(self, temp_config_file):
        """Test particle types with custom configuration."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        gas_type = get_particle_type('gas')
        assert gas_type == 'PartType0'
    
    def test_custom_config_output_dir(self, temp_config_file):
        """Test output directory with custom configuration."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        output_dir = get_output_directory()
        assert output_dir == '/tmp/test_output'
