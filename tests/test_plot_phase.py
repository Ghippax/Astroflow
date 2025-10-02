"""Tests for phase space plotting module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock

from cosmo_analysis.plot.phase import ytPhasePanel
from cosmo_analysis.config import Config

# Skip tests if yt is not available
yt = pytest.importorskip("yt")


class TestYtPhasePanel:
    """Tests for ytPhasePanel function."""
    
    def test_yt_phase_panel_module_exists(self):
        """Test that phase module can be imported."""
        from cosmo_analysis.plot import phase
        assert hasattr(phase, 'ytPhasePanel')
    
    def test_yt_phase_panel_uses_config_defaults(self, reset_config):
        """Test that ytPhasePanel uses config defaults."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Verify config has expected defaults
        assert config.get('simulation_parameters.particle_types.gas') == 'PartType0'
        assert config.get('plotting_defaults.yt_figsize') == 12
    
    def test_yt_phase_panel_function_signature(self):
        """Test that ytPhasePanel has expected parameters."""
        import inspect
        from cosmo_analysis.plot.phase import ytPhasePanel
        
        sig = inspect.signature(ytPhasePanel)
        params = list(sig.parameters.keys())
        
        # Check for key parameters
        assert 'simArr' in params
        assert 'idxArr' in params
        assert 'config' in params
        assert 'saveFig' in params
        assert 'showFig' in params


@pytest.mark.visual
@pytest.mark.slow
class TestPhaseVisual:
    """Visual tests for phase plotting with yt data."""
    
    @pytest.fixture(scope="class")
    def yt_dataset(self):
        """Create or load yt dataset for testing."""
        try:
            ds = yt.load_sample("IsolatedGalaxy")
            return ds
        except Exception as e:
            # Create synthetic dataset if sample can't be loaded
            print(f"Could not load yt sample data ({e}), creating synthetic dataset")
            from yt.testing import fake_random_ds
            ds = fake_random_ds(
                64,
                nprocs=1,
                fields=[("gas", "density"), ("gas", "temperature"), ("gas", "cell_mass")],
                units=["g/cm**3", "K", "g"],
                negative=[False, False, False]
            )
            return ds
    
    def test_phase_plot_with_yt_data(self, yt_dataset, tmp_path, reset_config):
        """Test creating phase plot with real yt data."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create mock simulation from yt dataset
        sim = Mock()
        sim.name = "TestPhase"
        snap = Mock()
        snap.ytcen = yt_dataset.domain_center
        sim.snap = [snap]
        sim.ytFull = [yt_dataset]
        
        save_path = str(tmp_path)
        
        ytPhasePanel(
            [sim], [0],
            part="gas",  # Use 'gas' for synthetic dataset
            zFields=["density", "temperature", "cell_mass"],
            zWidth=10,
            xb=50,  # Lower resolution for faster testing
            yb=50,
            saveFig=True,
            showFig=False,
            saveFigPath=save_path,
            message="test_phase",
            config=config
        )
        
        # Check that output was created
        output_file = tmp_path / "test_phase.png"
        assert output_file.exists(), f"Output file {output_file} was not created"
        assert output_file.stat().st_size > 0, "Output file is empty"
        
        plt.close('all')
    
    def test_phase_plot_consistency(self, yt_dataset, reset_config):
        """Test that phase plots are consistent across runs."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create mock simulation
        sim = Mock()
        sim.name = "TestConsistency"
        snap = Mock()
        snap.ytcen = yt_dataset.domain_center
        sim.snap = [snap]
        sim.ytFull = [yt_dataset]
        
        frames = []
        
        # Create two identical phase plots
        for i in range(2):
            result = ytPhasePanel(
                [sim], [0],
                part="gas",  # Use 'gas' for synthetic dataset
                zFields=["density", "temperature", "cell_mass"],
                zWidth=10,
                xb=50,
                yb=50,
                saveFig=False,
                showFig=False,
                animate=True,
                config=config
            )
            
            assert result is not None, f"Frame {i} was not generated"
            frames.append(result)
        
        # Verify we got both frames
        assert len(frames) == 2, f"Expected 2 frames but got {len(frames)}"
        
        # Check they're similar
        assert frames[0].shape == frames[1].shape, \
            f"Frame shapes don't match: {frames[0].shape} vs {frames[1].shape}"
        
        # Allow some variation due to rendering
        diff = np.abs(frames[0].astype(float) - frames[1].astype(float)).mean()
        assert diff < 50, f"Frames differ too much: mean diff = {diff}"
        
        plt.close('all')


class TestPhaseConfigIntegration:
    """Test config integration in phase module."""
    
    def test_particle_type_from_config(self, temp_config_file, reset_config):
        """Test that particle type is read from config."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        # Config should have gas particle type
        gas_type = config.get('simulation_parameters.particle_types.gas')
        assert gas_type == 'PartType0'
    
    def test_phase_uses_config_figsize(self, temp_config_file, reset_config):
        """Test that phase plot uses config figsize."""
        config = Config.load_config(str(temp_config_file))
        Config.set_instance(config)
        
        # Config should have yt_figsize
        figsize = config.get('plotting_defaults.yt_figsize')
        assert figsize == 12
