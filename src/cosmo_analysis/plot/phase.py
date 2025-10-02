"""Phase space plotting functions for cosmo_analysis.

This module contains functions for creating phase space diagrams showing
relationships between different physical quantities (e.g., density-temperature).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import yt

from .. import log
from ..config import get_config
from .base import handleFig


def ytPhasePanel(simArr, idxArr, depositionAlg="ngp", verbose=None, plotSize=None, 
                 saveFig=None, saveFigPath=None, showFig=None, message=None, blackLine=0, 
                 panOver=0, part=None, zLog=1, zFields=None, zFieldUnits=None, cM="algae", 
                 animate=0, zFieldLim=None, zWidth=15, fsize=None, wField=0, xb=300, yb=300, 
                 grid=True, axAspect=1, config=None):
    """Create phase space panel comparing multiple simulations.
    
    Creates a grid of phase space plots (2D histograms) showing relationships
    between physical quantities like density and temperature.
    
    Args:
        simArr: list of Simulation objects
        idxArr: list of snapshot indices
        depositionAlg: deposition algorithm for particle binning ("ngp", "cic", etc.)
        verbose: verbosity level (uses config if None)
        plotSize: yt figure window size (uses config if None)
        saveFig: whether to save figure (uses config if None)
        saveFigPath: path to save figure (uses config if None)
        showFig: whether to show figure (uses config if None)
        message: title/filename for saved figure
        blackLine: whether to add average profile line
        panOver: panel layout override (rows, cols)
        part: particle type (uses config if None)
        zLog: whether to log-scale the color field
        zFields: list of [x_field, y_field, z_field] names
        zFieldUnits: list of units for each field
        cM: colormap name
        animate: whether to return frame for animation
        zFieldLim: tuple of 6 values (zmin, zmax, xmin, xmax, ymin, ymax)
        zWidth: sphere width in kpc
        fsize: font size (uses config if None)
        wField: weight field (0 for no weighting) or list per simulation
        xb: number of x bins
        yb: number of y bins
        grid: whether to show grid
        axAspect: aspect ratio for axes
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
        plotSize = config.get('plotting_defaults.yt_figsize', 12)
    if fsize is None:
        fsize = config.get('plotting_defaults.fontsize', 12)
    if saveFig is None:
        saveFig = config.get('plotting_defaults.save_plots', True)
    if showFig is None:
        showFig = config.get('plotting_defaults.show_plots', False)
    
    # Set default fields
    if zFields is None:
        zFields = ["Density", "Temperature", "Masses"]
    if zFieldUnits is None:
        zFieldUnits = ["g/cm**3", "K", "Msun"]
    if zFieldLim is None:
        zFieldLim = (1e3, 1e8, 1e-29, 1e-21, 10, 1e7)
    
    if message:
        log.logger.info(f"\n{message}")
    
    # Panel fig setup
    if not isinstance(wField, list):
        wField = [wField] * len(simArr)
    
    if panOver == 0:
        panelSize = (1, math.ceil(len(simArr)))
    else:
        panelSize = panOver
    
    panelFig = plt.figure(figsize=(1, 1))
    panelGrid = AxesGrid(
        panelFig, (0, 0, 0.4*panelSize[1], 0.4*panelSize[0]),
        aspect=False, nrows_ncols=panelSize, axes_pad=0.1,
        label_mode="1", share_all=True, cbar_location="right",
        cbar_mode="single", cbar_size="5%", cbar_pad="2%"
    )
    
    if zFieldLim == 0:
        zFieldLim = [0, 0, 0, 0, 0, 0]
    if zFieldUnits == 0:
        zFieldUnits = [0, 0, 0]

    # Loading snapshots
    snapArr = [simArr[i].ytFull[idxArr[i]] for i in range(len(simArr))]
    titleArr = [simArr[i].name for i in range(len(simArr))]

    # Getting the black line (average profile)
    goodbin = []
    goodcil = []
    if blackLine:
        log.logger.debug(f"  Calculating avg profile with {simArr[0].name}")
        sp = snapArr[0].sphere(simArr[0].snap[idxArr[0]].ytcen, (zWidth, "kpc"))

        p1 = yt.ProfilePlot(
            sp, (part, zFields[0]), (part, zFields[1]),
            weight_field=(part, zFields[2]), n_bins=30,
            x_log=False, accumulation=False
        )
        
        p1.set_log((part, zFields[0]), True)
        p1.set_log((part, zFields[1]), True)

        if zFieldUnits[0] != 0:
            p1.set_unit((part, zFields[0]), zFieldUnits[0])
        if zFieldLim[2] != 0 or zFieldLim[3] != 0:
            p1.set_xlim(zFieldLim[2], zFieldLim[3])
        if zFieldUnits[1] != 0:
            p1.set_unit((part, zFields[1]), zFieldUnits[1])

        cil = p1.profiles[0].x.in_units(zFieldUnits[0]).d
        bin_data = p1.profiles[0][zFields[1]].in_units(zFieldUnits[1]).d
        
        for i in range(len(bin_data)):
            if abs(bin_data[i]) > 1e-33:
                goodbin.append(bin_data[i]) 
                goodcil.append(cil[i])
           
    # Start of the fig making
    for i, snap in enumerate(snapArr):
        log.logger.info(f"  - Plotting {simArr[i].name}")
        sp = snap.sphere(simArr[i].snap[idxArr[i]].ytcen, (zWidth, "kpc"))
        
        # Plot phase with specified parameters
        if zLog != 1:
            snap.field_info[(part, zFields[2])].take_log = False
        
        if wField[i] != 0:
            fig1 = yt.ParticlePhasePlot(
                sp, (part, zFields[0]), (part, zFields[1]), (part, zFields[2]),
                deposition=depositionAlg, figure_size=plotSize,
                weight_field=(part, wField[i]), fontsize=fsize,
                x_bins=xb, y_bins=yb
            )
        else:
            fig1 = yt.ParticlePhasePlot(
                sp, (part, zFields[0]), (part, zFields[1]), (part, zFields[2]),
                deposition=depositionAlg, figure_size=plotSize,
                fontsize=fsize, x_bins=xb, y_bins=yb
            )
            
        # Set units
        if zFieldUnits[0] != 0:
            fig1.set_unit((part, zFields[0]), zFieldUnits[0])
        if zFieldUnits[1] != 0:
            fig1.set_unit((part, zFields[1]), zFieldUnits[1])
        if zFieldUnits[2] != 0:
            fig1.set_unit((part, zFields[2]), zFieldUnits[2])
            
        # Set limits
        if zFieldLim[0] != 0 or zFieldLim[1] != 0:
            fig1.set_zlim((part, zFields[2]), zmin=zFieldLim[0], zmax=zFieldLim[1])
        if zFieldLim[2] != 0 or zFieldLim[3] != 0:
            fig1.set_xlim(zFieldLim[2], zFieldLim[3])
        if zFieldLim[4] != 0 or zFieldLim[5] != 0:
            fig1.set_ylim(zFieldLim[4], zFieldLim[5])

        fig1.set_log((part, zFields[2]), bool(zLog))
        fig1.set_cmap(field=(part, zFields[2]), cmap=cM)

        # Transfers yt plot to plt axes and renders the figure
        log.logger.debug(f"    Rendering {simArr[i].name}")
        fullPlot = fig1.plots[part, zFields[2]]
        fullPlot.figure = panelFig
        fullPlot.axes = panelGrid[i].axes
        if i == 0:
            fullPlot.cax = panelGrid.cbar_axes[i]
        
        fig1._setup_plots()

        if blackLine:
            panelGrid.axes_all[i].plot(goodcil, goodbin, "k--")
        
        panelFig.canvas.draw()
        
        if grid:
            panelGrid.axes_all[i].grid()
        
        panelGrid.axes_all[i].set_title(titleArr[i])
        panelGrid.axes_all[i].set_box_aspect(axAspect)
        
    return handleFig(panelFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)
