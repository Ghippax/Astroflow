"""
Astroflow - A framework for analyzing astrophysical simulations.

This package provides tools for loading, analyzing, and visualizing
data from various astrophysical simulation codes through a unified interface.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .core import load
from .core.simulation import Simulation
from .analysis import derived
from .plot import data, plot, render
from . import config

# Setup variables with configuration settings
from .log import get_logger
logger = get_logger()
logger.set_level(config.get("logging/level", "INFO"))

from yt import set_log_level as yt_set_log_level
yt_set_log_level(config.get("logging/yt_level", "INFO"))

__all__ = [
    "load",
    "Simulation",
    "data",
    "render",
    "plot",
    "derived",
    "config",
    "__version__",
]