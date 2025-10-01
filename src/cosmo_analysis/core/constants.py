"""Constants and default configuration values for cosmo_analysis.

This module is being gradually deprecated in favor of the new config.py system.
It remains for backward compatibility but should not be used for new code.

TODO: Migrate all hardcoded constants to config.yaml
"""

import copy
import matplotlib.pyplot as plt

## OPTIONS
# Figure options
figSize   = 8  # Fig size for plt plots
ytFigSize = 12 # Window size for the yt plot
fontSize  = 12 # Fontsize in yt plots

# TODO: Fix this path - use config.yaml instead
savePath = "/sqfs/work/hp240141/z6b616/analysis"

# Modified color maps - lazily initialized to avoid import errors
_starCmap = None
_mColorMap = None
_mColorMap2 = None

def _init_colormaps():
    """Initialize colormaps lazily to avoid import-time errors."""
    global _starCmap, _mColorMap, _mColorMap2
    
    if _starCmap is None:
        _starCmap = copy.copy(plt.get_cmap('autumn'))
        _starCmap.set_bad(color='k')
        _starCmap.set_under(color='k')
    
    if _mColorMap is None:
        try:
            import yt
            _mColorMap = copy.copy(plt.get_cmap('algae'))
            _mColorMap.set_bad(_mColorMap(0.347))
            _mColorMap.set_under(_mColorMap(0))
        except (ValueError, ImportError):
            # Fall back to viridis if algae not available
            _mColorMap = copy.copy(plt.get_cmap('viridis'))
            _mColorMap.set_bad(_mColorMap(0.347))
            _mColorMap.set_under(_mColorMap(0))
    
    if _mColorMap2 is None:
        try:
            import yt
            _mColorMap2 = copy.copy(plt.get_cmap('algae'))
            _mColorMap2.set_bad(_mColorMap2(0))
            _mColorMap2.set_under(_mColorMap2(0))
        except (ValueError, ImportError):
            # Fall back to viridis if algae not available
            _mColorMap2 = copy.copy(plt.get_cmap('viridis'))
            _mColorMap2.set_bad(_mColorMap2(0))
            _mColorMap2.set_under(_mColorMap2(0))

def get_star_cmap():
    """Get star colormap, initializing if necessary."""
    _init_colormaps()
    return _starCmap

def get_metal_cmap():
    """Get metallicity colormap, initializing if necessary."""
    _init_colormaps()
    return _mColorMap

def get_metal_cmap2():
    """Get second metallicity colormap, initializing if necessary."""
    _init_colormaps()
    return _mColorMap2

# Backward compatibility - will be initialized on first use  
# Access via functions above for safety, but keep these for existing code
starCmap = None
mColorMap = None
mColorMap2 = None

# Scaffolding options (Global parameters for how the plotting functions below should operate)
verboseLevel    = 15                    # How detailed is the console log, higher is more detailed
showErrorGlobal = 0                     # If the dispersion is plotted or not
errorLimGlobal  = (-1,1)                # Y limits on the dispersion plot
showAll         = False                  # If the plots are showed in the notebook or not
saveAll         = True                 # If the plots are saved to disk or not

# Analysis options
figWidth       = 30          # Kpc (Big box)
buffSize       = 800         # N   (Default bin number for coordinate-type histograms) 
youngStarAge   = 20          # Myr (Age to be considered for SFR density calculations)
lowResMock     = 750         # pc  (Resolution of the mock observations)
starFigSize    = 80          # kpc (Size of star map)
starBufferSize = 400         # N   (Buffer size of star map)
gasPart        = "PartType0" # Field for gas SPH particles 
starPart       = "PartType4" # Field for star particles
dmPart         = "PartType1" # Field for dark matter particles
zSolar         = 0.0204      # Solar metallicity