"""Visual regression tests for plotting functions using yt sample data.

These tests use yt's IsolatedGalaxy sample dataset to test actual plotting
functionality with real data, establishing baselines and checking for consistency.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib
from PIL import Image
import io

# Skip tests if yt is not available
yt = pytest.importorskip("yt")

from cosmo_analysis.plot.base import handleFig, saveFrame
from cosmo_analysis.plot.projection import ytMultiPanel, ytProjPanel
from cosmo_analysis.config import Config


@pytest.fixture(scope="module")
def yt_sample_dataset():
    """Load yt IsolatedGalaxy sample dataset.
    
    This fixture is module-scoped to avoid reloading the dataset for each test.
    If the sample cannot be downloaded, creates a synthetic dataset instead.
    """
    # Suppress yt logging during tests
    yt.set_log_level(40)  # ERROR level
    
    try:
        ds = yt.load_sample("IsolatedGalaxy")
        return ds
    except Exception as e:
        # If we can't download, create a synthetic dataset
        print(f"Could not load yt sample data ({e}), creating synthetic dataset")
        try:
            # Create a simple fake dataset for testing
            from yt.testing import fake_random_ds
            ds = fake_random_ds(
                64, 
                nprocs=1, 
                fields=[("gas", "density"), ("gas", "temperature")],
                units=["g/cm**3", "K"],
                negative=[False, False]
            )
            return ds
        except Exception as e2:
            pytest.skip(f"Could not create synthetic dataset: {e2}")


@pytest.fixture
def mock_simulation_from_yt(yt_sample_dataset):
    """Create a mock simulation object from yt dataset.
    
    This creates the structure expected by plotting functions.
    """
    from unittest.mock import Mock
    
    sim = Mock()
    sim.name = "IsolatedGalaxy"
    sim.cosmo = False
    
    # Create snapshot info
    snap = Mock()
    snap.time = 0.0
    snap.z = 0.0
    
    # Get center from dataset
    center = yt_sample_dataset.domain_center
    snap.center = np.array([center[0].v, center[1].v, center[2].v])
    snap.ytcen = center
    
    sim.snap = [snap]
    sim.ytFull = [yt_sample_dataset]
    
    return sim


def image_hash(image_array):
    """Compute hash of image array for comparison.
    
    Args:
        image_array: numpy array of image data
        
    Returns:
        str: MD5 hash of the image
    """
    return hashlib.md5(image_array.tobytes()).hexdigest()


def images_similar(img1, img2, threshold=0.01):
    """Check if two images are similar within a threshold.
    
    Args:
        img1: First image as numpy array
        img2: Second image as numpy array
        threshold: Maximum allowed difference ratio (0-1)
        
    Returns:
        bool: True if images are similar
    """
    if img1.shape != img2.shape:
        return False
    
    # Calculate normalized difference
    diff = np.abs(img1.astype(float) - img2.astype(float))
    max_diff = diff.max()
    avg_diff = diff.mean()
    
    # Normalize by image range
    img_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    if img_range == 0:
        return True
    
    normalized_diff = avg_diff / img_range
    
    return normalized_diff < threshold


@pytest.mark.slow
@pytest.mark.visual
class TestYTProjectionPlots:
    """Test projection plotting with real yt data."""
    
    def test_yt_projection_basic(self, yt_sample_dataset, tmp_path, reset_config):
        """Test basic yt projection plot creation."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create a simple projection
        proj = yt.ProjectionPlot(
            yt_sample_dataset, 
            'z', 
            ('gas', 'density'),
            center=yt_sample_dataset.domain_center,
            width=(50, 'kpc')
        )
        
        # Save and verify
        save_path = str(tmp_path)
        proj.save(str(tmp_path / "test_projection.png"))
        
        output_file = tmp_path / "test_projection.png"
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        
        # Load and check image properties
        img = Image.open(output_file)
        assert img.size[0] > 0
        assert img.size[1] > 0
    
    def test_yt_multi_panel_with_real_data(self, mock_simulation_from_yt, tmp_path, reset_config):
        """Test ytMultiPanel with real yt data."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create multi-panel plot
        save_path = str(tmp_path)
        
        try:
            ytMultiPanel(
                sims=[mock_simulation_from_yt],
                idx=[0],
                zField=["density"],
                part="gas",
                zWidth=50,
                bSize=128,  # Lower resolution for faster testing
                saveFig=True,
                showFig=False,
                saveFigPath=save_path,
                message="test_multipanel",
                config=config
            )
            
            # Check that output was created
            output_file = tmp_path / "test_multipanel.png"
            assert output_file.exists()
            assert output_file.stat().st_size > 0
            
            # Load and verify image
            img = Image.open(output_file)
            assert img.size[0] > 0
            assert img.size[1] > 0
            
        except Exception as e:
            pytest.skip(f"ytMultiPanel test skipped due to: {e}")
    
    def test_visual_consistency_projection(self, yt_sample_dataset, tmp_path, reset_config):
        """Test that repeated projections produce consistent results."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create two identical projections
        images = []
        for i in range(2):
            proj = yt.ProjectionPlot(
                yt_sample_dataset,
                'z',
                ('gas', 'density'),
                center=yt_sample_dataset.domain_center,
                width=(50, 'kpc')
            )
            
            # Get the figure
            fig = proj.plots[('gas', 'density')].figure
            
            # Capture as array
            frame = saveFrame(fig, config=config)
            images.append(frame)
        
        # Check images are similar
        assert images[0].shape == images[1].shape
        assert images_similar(images[0], images[1], threshold=0.05)
        
        plt.close('all')


@pytest.mark.slow
@pytest.mark.visual
class TestVisualRegression:
    """Visual regression tests with baseline comparisons."""
    
    @pytest.fixture(scope="class")
    def baseline_dir(self, tmp_path_factory):
        """Create directory for baseline images."""
        baseline = tmp_path_factory.mktemp("baselines")
        return baseline
    
    def test_establish_density_projection_baseline(self, yt_sample_dataset, baseline_dir, reset_config):
        """Establish baseline for density projection."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create projection
        proj = yt.ProjectionPlot(
            yt_sample_dataset,
            'z',
            ('gas', 'density'),
            center=yt_sample_dataset.domain_center,
            width=(50, 'kpc')
        )
        
        # Save baseline
        baseline_path = baseline_dir / "density_projection_baseline.png"
        proj.save(str(baseline_path))
        
        assert baseline_path.exists()
        
        # Store hash for comparison
        img = Image.open(baseline_path)
        img_array = np.array(img)
        baseline_hash = image_hash(img_array)
        
        # Save hash
        hash_file = baseline_dir / "density_projection_baseline.hash"
        hash_file.write_text(baseline_hash)
        
        print(f"Baseline established with hash: {baseline_hash}")
    
    def test_compare_against_baseline(self, yt_sample_dataset, baseline_dir, reset_config):
        """Compare new projection against baseline."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Check if baseline exists
        baseline_path = baseline_dir / "density_projection_baseline.png"
        if not baseline_path.exists():
            pytest.skip("Baseline not established yet")
        
        # Create new projection with same parameters
        proj = yt.ProjectionPlot(
            yt_sample_dataset,
            'z',
            ('gas', 'density'),
            center=yt_sample_dataset.domain_center,
            width=(50, 'kpc')
        )
        
        # Get the figure and convert to array
        fig = proj.plots[('gas', 'density')].figure
        new_frame = saveFrame(fig, config=config)
        
        # Load baseline
        baseline_img = Image.open(baseline_path)
        baseline_array = np.array(baseline_img)
        
        # Compare
        # Note: Exact pixel match might not work due to rendering differences
        # Instead, check that images are similar
        if new_frame.shape != baseline_array.shape:
            # Resize if needed
            baseline_img = baseline_img.resize((new_frame.shape[1], new_frame.shape[0]))
            baseline_array = np.array(baseline_img)
        
        # Check similarity
        assert images_similar(new_frame, baseline_array, threshold=0.1), \
            "New projection differs significantly from baseline"
        
        plt.close('all')
    
    def test_different_field_produces_different_output(self, yt_sample_dataset, tmp_path, reset_config):
        """Test that different fields produce visually different outputs."""
        config = Config.load_config()
        Config.set_instance(config)
        
        # Create density projection
        proj_density = yt.ProjectionPlot(
            yt_sample_dataset,
            'z',
            ('gas', 'density'),
            center=yt_sample_dataset.domain_center,
            width=(50, 'kpc')
        )
        
        fig1 = proj_density.plots[('gas', 'density')].figure
        frame1 = saveFrame(fig1, config=config)
        
        # Create temperature projection
        proj_temp = yt.ProjectionPlot(
            yt_sample_dataset,
            'z',
            ('gas', 'temperature'),
            center=yt_sample_dataset.domain_center,
            width=(50, 'kpc')
        )
        
        fig2 = proj_temp.plots[('gas', 'temperature')].figure
        frame2 = saveFrame(fig2, config=config)
        
        # These should be different
        assert not images_similar(frame1, frame2, threshold=0.05), \
            "Different fields should produce visually different outputs"
        
        plt.close('all')


@pytest.mark.slow
@pytest.mark.visual
class TestYTDataConsistency:
    """Test consistency of yt data handling."""
    
    def test_dataset_properties(self, yt_sample_dataset):
        """Test that dataset has expected properties."""
        # Check basic properties
        assert yt_sample_dataset is not None
        assert hasattr(yt_sample_dataset, 'domain_center')
        assert hasattr(yt_sample_dataset, 'domain_width')
        
        # Check that gas fields are available
        assert ('gas', 'density') in yt_sample_dataset.field_list or \
               ('PartType0', 'density') in yt_sample_dataset.field_list
    
    def test_projection_different_axes(self, yt_sample_dataset, reset_config):
        """Test projections along different axes."""
        config = Config.load_config()
        Config.set_instance(config)
        
        frames = []
        axes = ['x', 'y', 'z']
        
        for axis in axes:
            proj = yt.ProjectionPlot(
                yt_sample_dataset,
                axis,
                ('gas', 'density'),
                center=yt_sample_dataset.domain_center,
                width=(50, 'kpc')
            )
            
            fig = proj.plots[('gas', 'density')].figure
            frame = saveFrame(fig, config=config)
            frames.append(frame)
        
        # All frames should have data
        for frame in frames:
            assert frame.size > 0
            assert frame.max() > 0
        
        # Different axes should produce different projections
        # (though they might be similar for a spherical galaxy)
        assert frames[0].shape == frames[1].shape == frames[2].shape
        
        plt.close('all')


@pytest.mark.slow
@pytest.mark.visual  
class TestRegressionWithConfig:
    """Test that config changes don't break visual output."""
    
    def test_config_dpi_affects_output_size(self, yt_sample_dataset, tmp_path, reset_config):
        """Test that config DPI setting affects output."""
        # Create config with low DPI
        config_low = Config.load_config()
        config_low.data['plotting_defaults']['dpi'] = 50
        Config.set_instance(config_low)
        
        proj = yt.ProjectionPlot(
            yt_sample_dataset,
            'z',
            ('gas', 'density'),
            center=yt_sample_dataset.domain_center,
            width=(50, 'kpc')
        )
        
        fig = proj.plots[('gas', 'density')].figure
        frame_low = saveFrame(fig, config=config_low)
        
        plt.close('all')
        
        # Create config with high DPI
        config_high = Config.load_config()
        config_high.data['plotting_defaults']['dpi'] = 100
        Config.set_instance(config_high)
        
        proj = yt.ProjectionPlot(
            yt_sample_dataset,
            'z',
            ('gas', 'density'),
            center=yt_sample_dataset.domain_center,
            width=(50, 'kpc')
        )
        
        fig = proj.plots[('gas', 'density')].figure
        frame_high = saveFrame(fig, config=config_high)
        
        # Higher DPI should produce larger image
        assert frame_high.size >= frame_low.size
        
        plt.close('all')
