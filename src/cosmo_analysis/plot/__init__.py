"""Plotting module for cosmo_analysis.

This package provides modular plotting functions for cosmological simulations,
organized into specialized sub-modules:

- base: Core plotting utilities (figure handling, legends, frame capture)
- projection: Projection plotting functions  
- utils: Helper functions (animations, binning, cosmology)
- plots: Legacy interface that re-exports all functions for backward compatibility

For new code, import from specific modules:
    from cosmo_analysis.plot.base import handleFig
    from cosmo_analysis.plot.projection import ytMultiPanel
    from cosmo_analysis.plot.utils import makeMovie

For backward compatibility, import from plots:
    from cosmo_analysis.plot.plots import ytMultiPanel
"""

# Re-export main modules for convenience
from . import base
from . import projection
from . import utils
from . import plots

__all__ = [
    'base',
    'projection', 
    'utils',
    'plots',
]
