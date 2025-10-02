"""Pytest configuration and fixtures for cosmo_analysis tests."""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file for testing.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Path: Path to temporary config file
    """
    config_content = """
paths:
  output_directory: "/tmp/test_output"
  data_files:
    bilcontour: "test_bilcontour.txt"
    projection_list: "test_projection.txt"

simulation_parameters:
  particle_types:
    gas: "PartType0"
    dm: "PartType1"
    star: "PartType4"

plotting_defaults:
  figsize: [8, 8]
  yt_figsize: 12
  fontsize: 12
  save_plots: false
  show_plots: false

analysis_options:
  fig_width: 30
  buffer_size: 800
"""
    
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for tests.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Path: Path to temporary output directory
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture(autouse=True)
def reset_config():
    """Reset configuration singleton between tests."""
    from cosmo_analysis.config import Config
    Config._instance = None
    yield
    Config._instance = None


@pytest.fixture
def mock_simulation():
    """Create a mock simulation object for testing plotting functions.
    
    This creates a minimal mock simulation with the structure expected by
    plotting functions, but without requiring actual simulation data.
    
    Returns:
        Mock: A mock simulation object with required attributes
    """
    from unittest.mock import Mock
    import numpy as np
    
    # Create mock simulation
    sim = Mock()
    sim.name = "MockSim"
    sim.cosmo = False
    
    # Create mock snapshot
    snap = Mock()
    snap.time = 100.0
    snap.z = 0.0
    snap.center = np.array([0.0, 0.0, 0.0])
    snap.ytcen = (0.0, 'kpc')
    
    sim.snap = [snap]
    
    # Create mock yt dataset
    yt_ds = Mock()
    yt_ds.field_info = {}
    
    sim.ytFull = [yt_ds]
    
    return sim
