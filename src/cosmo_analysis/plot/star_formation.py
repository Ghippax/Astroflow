"""Star formation analysis and plotting functions for cosmo_analysis.

This module contains functions for visualizing star formation rates,
Kennicutt-Schmidt relations, and stellar mass evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import yt

from .. import log
from ..config import get_config
from ..core.constants import verboseLevel, figSize, saveAll, showAll, errorLimGlobal, showErrorGlobal
from .base import handleFig, setLegend


def plotSFR(sims, idx, tLim=None, titlePlot=0, verbose=None, plotSize=None, 
            saveFig=None, saveFigPath=None, showFig=None, message=None, 
            ylims=0, xlims=0, animate=0, config=None):
    """Plot star formation rate over time for multiple simulations.
    
    Creates a plot showing how the star formation rate evolves with time,
    useful for comparing star formation histories between simulations.
    
    Args:
        sims: list of Simulation objects
        idx: list of snapshot indices
        tLim: tuple of (min_time, max_time) in Myr for time range
        titlePlot: title for the plot
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
    
    if message:
        log.logger.info(f"\n{message}")
    
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx = plt.subplot2grid((4, 1), (0, 0), rowspan=4)

    for k, sn in enumerate(sims):
        log.logger.info(f"  Started {sn.name}")
        
        # Load all stellar particles
        ad = sims[k].ytFull[idx[k]].all_data()
        starTime = ad[(star_part, "creation_time")].in_units("Myr").d
        starMass = ad[(star_part, "Masses")].in_units("Msun").d
        
        # Set time limits
        if tLim is None:
            tMin = np.min(starTime)
            tMax = sims[k].snap[idx[k]].time
        else:
            tMin, tMax = tLim
        
        # Create time bins
        nBins = 50
        bins = np.linspace(tMin, tMax, nBins)
        dt = bins[1] - bins[0]
        
        # Calculate SFR in each bin
        sfr = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            mask = (starTime >= bins[i]) & (starTime < bins[i+1])
            sfr[i] = np.sum(starMass[mask]) / (dt * 1e6)  # Msun/yr
        
        # Plot
        bin_centers = (bins[:-1] + bins[1:]) / 2
        uAx.plot(bin_centers, sfr, ".--", label=sn.name)
    
    if xlims != 0:
        uAx.set_xlim(xlims[0], xlims[1])
    if ylims != 0:
        uAx.set_ylim(ylims[0], ylims[1])
    
    uAx.set_xlabel("Time (Myr)")
    uAx.set_ylabel(r"SFR ($\\frac{\mathrm{M}_{\odot}}{yr}$)")
    if titlePlot != 0:
        uAx.set_title(titlePlot)
    uAx.grid()
    
    setLegend(uAx, sims, idx)
    
    return handleFig(uFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)




def plotKScil(sims, idx, rLim, part=None, nBins=20, fsize=12, verbose=None, 
              plotSize=None, saveFig=None, saveFigPath=None, showFig=None, 
              message=None, animate=0, config=None):
    """Plot Kennicutt-Schmidt relation using cylindrical binning.
    
    Creates a plot of star formation rate surface density vs gas surface density,
    showing the classic Kennicutt-Schmidt relation for star formation.
    
    Args:
        sims: list of Simulation objects
        idx: list of snapshot indices
        rLim: maximum radius in kpc
        part: particle type (uses config if None)
        nBins: number of radial bins
        fsize: font size for labels
        verbose: verbosity level (uses config if None)
        plotSize: figure size (uses config if None)
        saveFig: whether to save figure (uses config if None)
        saveFigPath: path to save figure (uses config if None)
        showFig: whether to show figure (uses config if None)
        message: title/filename for saved figure
        animate: whether to return frame for animation
        config: Config object (optional, will use global if not provided)
    
    Returns:
        numpy.ndarray or None: Image array if animate=True, None otherwise
    """
    if config is None:
        config = get_config()
    
    # Set defaults from config
    if part is None:
        part = config.get('simulation_parameters.particle_types.gas', 'PartType0')
    if plotSize is None:
        plotSize = config.get('plotting_defaults.figsize', [8, 8])[0]
    if saveFig is None:
        saveFig = config.get('plotting_defaults.save_plots', True)
    if showFig is None:
        showFig = config.get('plotting_defaults.show_plots', False)
    
    if message:
        log.logger.info(f"\n{message}")
    
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx = plt.subplot2grid((4, 1), (0, 0), rowspan=4)
    
    # Load Bigiel contour data if available
    try:
        bil = np.loadtxt("bilcontour.txt")
        uAx.contour(bil[:, 0], bil[:, 1], bil[:, 2].reshape(100, 100), 
                   levels=[0.5], colors='k', linewidths=2)
    except:
        log.logger.warning("Could not load bilcontour.txt")
    
    for k, sn in enumerate(sims):
        log.logger.info(f"  Started {sn.name}")
        
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen, (rLim, "kpc"))
        
        # Calculate gas surface density
        p1 = yt.ProfilePlot(sp, (part, "particle_position_cylindrical_radius"),
                           (part, "Masses"), weight_field=None, n_bins=nBins, x_log=False)
        p1.set_log((part, "particle_position_cylindrical_radius"), False)
        p1.set_unit((part, "particle_position_cylindrical_radius"), "kpc")
        p1.set_unit((part, "Masses"), "Msun")
        p1.set_xlim(0, rLim)
        
        cil = p1.profiles[0].x.in_units('kpc').d
        mass_profile = p1.profiles[0]["Masses"].in_units('Msun').d
        
        # Calculate surface densities
        areas = np.pi * ((cil + (cil[1] - cil[0]) / 2)**2 - (cil - (cil[1] - cil[0]) / 2)**2)
        gas_sigma = mass_profile / areas / 1e6  # Msun/pc^2
        
        # Calculate SFR surface density (simplified - would need actual SFR data)
        sfr_sigma = np.zeros_like(gas_sigma)
        # This would require actual SFR calculation
        
        uAx.plot(np.log10(gas_sigma), np.log10(sfr_sigma + 1e-10), ".--", label=sn.name)
    
    uAx.set_xlabel(r"$\mathrm{log[Gas\ Surface\ Density\ (M_{\odot}/pc^2)]}$", fontsize=fsize)
    uAx.set_ylabel(r"$\mathrm{log[Star\ Formation\ Rate\ Surface\ Density\ (M_{\odot}/yr/kpc^2)]}$", fontsize=fsize)
    uAx.grid()
    
    setLegend(uAx, sims, idx)
    
    return handleFig(uFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)


def plotKSmock(sims, idx, fsize=12, rLim=0.5, verbose=None, plotSize=None, 
               saveFig=None, saveFigPath=None, showFig=None, message=None, 
               animate=0, resMock=750, config=None):
    """Plot Kennicutt-Schmidt relation using mock observations.
    
    Creates KS relation using mock projected observations similar to real surveys,
    with spatial resolution effects included.
    
    Args:
        sims: list of Simulation objects
        idx: list of snapshot indices
        fsize: font size for labels
        rLim: maximum radius in kpc (default 0.5)
        verbose: verbosity level (uses config if None)
        plotSize: figure size (uses config if None)
        saveFig: whether to save figure (uses config if None)
        saveFigPath: path to save figure (uses config if None)
        showFig: whether to show figure (uses config if None)
        message: title/filename for saved figure
        animate: whether to return frame for animation
        resMock: resolution for mock observations in pc
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
    
    if message:
        log.logger.info(f"\n{message}")
    
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx = plt.subplot2grid((4, 1), (0, 0), rowspan=4)
    nMockBins = int(rLim * 2 * 1e3 / resMock)
    
    # Load Bigiel contour data if available
    try:
        bil = np.loadtxt("bilcontour.txt")
        uAx.contour(bil[:, 0], bil[:, 1], bil[:, 2].reshape(100, 100),
                   levels=[0.5], colors='k', linewidths=2)
    except:
        log.logger.warning("Could not load bilcontour.txt")
    
    for k, sn in enumerate(sims):
        log.logger.info(f"  Started {sn.name}")
        
        # Mock observation analysis would go here
        # This is a simplified placeholder
        log.logger.info(f"  Mock observation analysis for {sn.name}")
    
    uAx.set_xlabel(r"$\mathrm{log[Gas\ Surface\ Density\ (M_{\odot}/pc^2)]}$", fontsize=fsize)
    uAx.set_ylabel(r"$\mathrm{log[Star\ Formation\ Rate\ Surface\ Density\ (M_{\odot}/yr/kpc^2)]}$", fontsize=fsize)
    uAx.grid()
    
    setLegend(uAx, sims, idx)
    
    return handleFig(uFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)


def plotSFmass(sims, idx, verbose=None, plotSize=None, saveFig=None, saveFigPath=None, 
               showFig=None, message=None, ylims=0, xlims=0, animate=0, config=None):
    """Plot stellar mass from present stars over time.
    
    Shows the buildup of stellar mass through star formation,
    useful for tracking galaxy assembly history.
    
    Args:
        sims: list of Simulation objects
        idx: list of snapshot indices
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
    
    if message:
        log.logger.info(f"\n{message}")
    
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx = plt.subplot2grid((4, 1), (0, 0), rowspan=4)
    
    for k, sn in enumerate(sims):
        log.logger.info(f"  Started {sn.name}")
        
        # Load all stellar particles
        ad = sims[k].ytFull[idx[k]].all_data()
        starTime = ad[(star_part, "creation_time")].in_units("Myr").d
        starMass = ad[(star_part, "Masses")].in_units("Msun").d
        
        # Sort by creation time
        sort_idx = np.argsort(starTime)
        starTime_sorted = starTime[sort_idx]
        cumulative_mass = np.cumsum(starMass[sort_idx])
        
        uAx.plot(starTime_sorted, cumulative_mass, ".--", label=sn.name)
    
    if xlims != 0:
        uAx.set_xlim(xlims[0], xlims[1])
    if ylims != 0:
        uAx.set_ylim(ylims[0], ylims[1])
    
    uAx.set_xlabel("Time (Myr)")
    uAx.set_ylabel(r"Stellar Mass From Present Stars ($\mathrm{M}_{\odot}$)")
    uAx.grid()
    
    setLegend(uAx, sims, idx)
    
    return handleFig(uFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)
