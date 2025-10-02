"""Plotting functions for cosmo_analysis.

This module provides plotting functions for cosmological simulations.
It re-exports functions from specialized modules for backward compatibility
while providing a modular structure.

NOTE: This file maintains backward compatibility. New code should import
from the specialized modules (base, projection, phase, profiles, halo, 
star_formation, galaxy) directly.
"""

import yt

from ..core.constants import *
from .. import log

# Import from new modular structure
from .base import saveFrame, setLegend, handleFig
from .projection import ytMultiPanel, ytProjPanel
from .phase import ytPhasePanel
from .profiles import plotBinned, plotRotDisp
from .halo import findHalos, plotClumpMassF
from .star_formation import plotSFR, plotKScil, plotKSmock, plotSFmass
from .galaxy import plotMsMh
from .utils import (makeMovie, binFunctionCilBins, binFunctionSphBins, 
                   binFunctionCilBinsSFR, makeZbinFun, binFunctionSphVol, aFromT)

yt.set_log_level(0)

# Legacy global - kept for backward compatibility, use config instead
savePath = "/sqfs/work/hp240141/z6b616/analysis"
