"""
Astroflow - A framework for analyzing astrophysical simulations.

This package provides tools for loading, analyzing, and visualizing
data from various astrophysical simulation codes through a unified interface.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .core import load
from .core.registries import register_derived, derived_registry
from .core.simulation import Simulation
from .analysis import derived_prop
from . import conf

# Setup variables with configuration settings
from .logging import get_logger
logger = get_logger()
logger.set_level(conf.get("logging/level", "INFO"))

from yt import set_log_level as yt_set_log_level
yt_set_log_level(conf.get("logging/yt_level", "INFO"))

__all__ = [
    "load",
    "register_derived",
    "derived_registry",
    "Simulation",
    "derived_prop",
    "conf",
    "__version__",
]