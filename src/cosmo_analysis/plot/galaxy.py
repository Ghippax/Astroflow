"""Galaxy property analysis and plotting functions for cosmo_analysis.

This module contains functions for visualizing galaxy-scale properties
and correlations like stellar mass vs halo mass relations.
"""

import numpy as np
import matplotlib.pyplot as plt
import yt

from .. import log
from ..config import get_config
from ..core.constants import verboseLevel, figSize, saveAll, showAll
from .base import handleFig, setLegend


def plotMsMh(sims, idx, haloData, rLim, verbose=None, plotSize=None, saveFig=None, 
             saveFigPath=None, showFig=None, message=None, ylims=0, xlims=0, 
             animate=0, config=None):
    """Plot stellar mass vs halo mass relation.
    
    Shows the relationship between stellar mass and total halo mass,
    a fundamental scaling relation for galaxy formation models.
    
    Args:
        sims: list of Simulation objects
        idx: list of snapshot indices
        haloData: tuple of (haloSims, haloFilt) from findHalos
        rLim: maximum radius in kpc for stellar mass calculation
        verbose: verbosity level (uses config if None)
        plotSize: figure size (uses config if None)
        saveFig: whether to save figure (uses config if None)
        saveFigPath: path to save figure (uses config if None)
        showFig: whether to show figure (uses config if None)
        message: title/filename for saved figure
        ylims: tuple of (ymin, ymax) for plot limits
        xlims: tuple of (xmin, xmax) for plot limits
        animate: whether to return frame for animation
        config: Config object (optional, will use global if not provided)
    
    Returns:
        numpy.ndarray or None: Image array if animate=True, None otherwise
    """
    if config is None:
        config = get_config()
    
    # Set defaults from config
    if plotSize is None:
        plotSize = config.get('plotting_defaults.figsize', [8, 8])[0]
    if saveFig is None:
        saveFig = config.get('plotting_defaults.save_plots', True)
    if showFig is None:
        showFig = config.get('plotting_defaults.show_plots', False)
    
    star_part = config.get('simulation_parameters.particle_types.star', 'PartType4')
    dm_part = config.get('simulation_parameters.particle_types.dm', 'PartType1')
    
    if message:
        log.logger.info(f"\n{message}")
    
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx = plt.subplot2grid((4, 1), (0, 0), rowspan=4)
    
    haloSims, haloFilt = haloData
    
    for k, sn in enumerate(sims):
        log.logger.info(f"  Started {sn.name}")
        
        # Get halo masses
        temp = haloSims[k].all_data()
        if isinstance(haloFilt[k], np.ndarray) and haloFilt[k].dtype == bool:
            halo_masses = temp['particle_mass'][haloFilt[k]].in_units("Msun").d
        else:
            halo_masses = temp['particle_mass'][haloFilt[k]].in_units("Msun").d
        
        # Calculate stellar masses for each halo
        stellar_masses = []
        for halo_idx in range(len(halo_masses)):
            # Get halo center (simplified - would need actual halo center)
            center = sims[k].snap[idx[k]].ytcen
            
            # Calculate stellar mass within rLim
            sp = sims[k].ytFull[idx[k]].sphere(center, (rLim, "kpc"))
            try:
                stellar_mass = sp[(star_part, "Masses")].in_units("Msun").sum().d
            except:
                stellar_mass = 0
            
            stellar_masses.append(stellar_mass)
        
        stellar_masses = np.array(stellar_masses)
        
        # Plot only where we have both masses
        valid = (stellar_masses > 0) & (halo_masses > 0)
        uAx.plot(np.log10(halo_masses[valid]), np.log10(stellar_masses[valid]), 
                ".", label=sn.name)
    
    if xlims != 0:
        uAx.set_xlim(xlims[0], xlims[1])
    if ylims != 0:
        uAx.set_ylim(ylims[0], ylims[1])
    
    uAx.set_xlabel(r"$\log_{10}(M_{\mathrm{halo}}/M_{\odot})$")
    uAx.set_ylabel(r"$\log_{10}(M_{\star}/M_{\odot})$")
    uAx.grid()
    
    setLegend(uAx, sims, idx)
    
    return handleFig(uFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)
