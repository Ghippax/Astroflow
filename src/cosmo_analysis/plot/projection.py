"""Projection plotting functions for cosmo_analysis.

This module contains functions for creating 2D projection plots of simulation data,
including multi-panel comparisons and various field projections.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.offsetbox import AnchoredText
import yt

from .. import log
from ..config import get_config
from .base import handleFig


def ytMultiPanel(sims, idx, zField=["Density"], axisProj=0, part=None, zFieldUnit="g/cm**2", 
                 cM="algae", takeLog=1, zFieldLim=[1.5e-4, 1e-1], wField=0, zWidth=None,
                 bSize=None, flipOrder=0, verbose=None, plotSize=None, saveFig=None, 
                 saveFigPath=None, showFig=None, message=None, fsize=None, animate=0, config=None):
    """Create multi-panel projection plot for multiple simulations and fields.
    
    This function creates a grid of projection plots comparing different simulations
    and/or different fields. Layout can be flipped to have simulations or fields as rows.
    
    Args:
        sims: list of Simulation objects
        idx: list of snapshot indices for each simulation
        zField: list of field names to project (default: ["Density"])
        axisProj: projection axis (0=x, 1=y, 2=z) or list of axes per simulation
        part: particle type or list of particle types per field
        zFieldUnit: unit for field display or list per field
        cM: colormap name or list per field
        takeLog: whether to log scale (1/0) or list per field
        zFieldLim: color limits [min, max] or list per field
        wField: weight field name or list per field (0 for no weighting)
        zWidth: projection width in kpc or list per field
        bSize: buffer size (resolution) or list per field
        flipOrder: if True, simulations are rows instead of fields
        verbose: verbosity level (uses config if None)
        plotSize: yt figure window size (uses config if None)
        saveFig: whether to save figure (uses config if None)
        saveFigPath: path to save figure (uses config if None)
        showFig: whether to show figure (uses config if None)
        message: title/filename for saved figure
        fsize: font size (uses config if None)
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
    if zWidth is None:
        zWidth = config.get('analysis_options.fig_width', 30)
    if bSize is None:
        bSize = config.get('analysis_options.buffer_size', 800)
    if plotSize is None:
        plotSize = config.get('plotting_defaults.yt_figsize', 12)
    if fsize is None:
        fsize = config.get('plotting_defaults.fontsize', 12)
    if saveFig is None:
        saveFig = config.get('plotting_defaults.save_plots', True)
    if showFig is None:
        showFig = config.get('plotting_defaults.show_plots', False)
    
    if message:
        log.logger.info(f"\n{message}")
    
    # Option setup - ensure all parameters are lists matching field count
    numP = len(zField)
    numS = len(sims)
    if not isinstance(axisProj, list):
        axisProj = [axisProj] * numS
    if not isinstance(part, list):
        part = [part] * numP
    if not isinstance(zFieldUnit, list):
        zFieldUnit = [zFieldUnit] * numP
    if not isinstance(cM, list):
        cM = [cM] * numP
    if not isinstance(takeLog, list):
        takeLog = [takeLog] * numP
    if not isinstance(wField, list):
        wField = [wField] * numP
    if not isinstance(zWidth, list):
        zWidth = [zWidth] * numP
    if not isinstance(zFieldLim, list):
        zFieldLim = [zFieldLim] * numP
    if not isinstance(bSize, list):
        bSize = [bSize] * numP
    
    # Ensure zFieldLim is list of lists
    if not all(isinstance(lIdx, list) for lIdx in zFieldLim):
        zFieldLimAux = [None] * numP
        for j in range(numP):
            zFieldLimAux[j] = zFieldLim 
        zFieldLim = zFieldLimAux

    rowIter = zField
    colIter = sims
    if flipOrder:
        rowIter = sims
        colIter = zField
    
    # Panel fig setup
    panelSize = (len(rowIter), len(colIter))
    panelFig = plt.figure()
    loc = "right" if not flipOrder else "bottom"
    panelGrid = AxesGrid(panelFig, (0, 0, 1, 1), nrows_ncols=panelSize, axes_pad=0.02,
                        label_mode="1", share_all=False, cbar_location=loc,
                        cbar_mode="edge", cbar_size="5%", cbar_pad="2%")

    # Loading snapshots
    snapArr = [sims[i].ytFull[idx[i]] for i in range(numS)]
    titleArr = [sims[i].name for i in range(numS)]

    for i, snap in enumerate(snapArr):
        log.logger.info(f"  - Projecting {sims[i].name} Time {sims[i].snap[idx[i]].time:.1f} "
                       f"Redshift {sims[i].snap[idx[i]].z:.2f} Axis {axisProj[i]}")
        
        for j, pField in enumerate(zField):
            iterRow, iterCol = (j, i)
            if flipOrder:
                iterRow, iterCol = (i, j)

            log.logger.debug(f"    Projecting field {pField} Particle {part[j]} Weight {wField[j]} "
                           f"Width {zWidth[j]} Unit {zFieldUnit[j]} Lim {zFieldLim[j]}")
            
            # Setup projection of pField of snap
            if takeLog[j] == 0:
                snap.field_info[(part[j], pField)].take_log = False
            
            if wField[j] != 0:
                fig1 = yt.ProjectionPlot(snap, axisProj[i], (part[j], pField), 
                                        window_size=plotSize, weight_field=(part[j], wField[j]), 
                                        fontsize=fsize, center=sims[i].snap[idx[i]].ytcen)
            else:
                if part[j] == "PartType4":
                    fig1 = yt.ParticleProjectionPlot(snap, axisProj[i], (part[j], pField), 
                                                     window_size=plotSize, depth=(zWidth[j], "kpc"), 
                                                     fontsize=fsize, center=sims[i].snap[idx[i]].ytcen)
                else:
                    fig1 = yt.ProjectionPlot(snap, axisProj[i], (part[j], pField), 
                                            window_size=plotSize, fontsize=fsize, 
                                            center=sims[i].snap[idx[i]].ytcen)
            
            fig1.set_width(zWidth[j], "kpc")
            if zFieldUnit[j] != 0:
                fig1.set_unit((part[j], pField), zFieldUnit[j])
            if not (zFieldLim[j][0] == 0 and zFieldLim[j][1] == 0):
                fig1.set_zlim((part[j], pField), zmin=zFieldLim[j][0], zmax=zFieldLim[j][1])
            fig1.set_cmap(field=(part[j], pField), cmap=cM[j])
            fig1.set_buff_size(bSize[j])
            
            if ((not flipOrder) and iterRow == 0) or (flipOrder and iterCol == 0):
                fig1.annotate_timestamp(redshift=True)

            # Transfers yt plot to plt axes and renders the figure
            fullPlot = fig1.plots[part[j], pField]
            fullPlot.figure = panelFig
            fullPlot.axes = panelGrid[iterCol + iterRow * len(colIter)].axes
            
            fullPlot.cax = panelGrid.cbar_axes[iterRow]
            if flipOrder:
                fullPlot.cax = panelGrid.cbar_axes[iterCol]

            log.logger.debug("    Rendering")
            fig1._setup_plots()

            if ((not flipOrder) and iterRow == 0) or (flipOrder and iterCol == 0): 
                nameTag = AnchoredText(titleArr[i], loc=2, prop=dict(size=9), frameon=True)
                panelGrid[iterCol + iterRow * len(colIter)].axes.add_artist(nameTag)

    return handleFig(panelFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)


def ytProjPanel(simArr, idxArr, verbose=None, plotSize=None, saveFig=None, saveFigPath=None,
                showFig=None, message=None, twoAxis=True, axisProj=[2, 0], part=None, 
                bSize=None, zField="Density", zFieldUnit="g/cm**2", cM="algae", takeLog=1, 
                zFieldLim=(1.5e-4, 1e-1), zWidth=None, fsize=None, wField=0, ovHalo=0, 
                animate=0, config=None):
    """Create projection panel for multiple simulations.
    
    Creates a panel of projection plots, optionally showing two projection axes
    for each simulation.
    
    Args:
        simArr: list of Simulation objects
        idxArr: list of snapshot indices
        verbose: verbosity level (uses config if None)
        plotSize: yt figure window size (uses config if None)
        saveFig: whether to save figure (uses config if None)
        saveFigPath: path to save figure (uses config if None)
        showFig: whether to show figure (uses config if None)
        message: title/filename for saved figure
        twoAxis: if True, shows two projection axes per simulation
        axisProj: list of two projection axes [ax1, ax2]
        part: particle type
        bSize: buffer size (resolution)
        zField: field name to project
        zFieldUnit: unit for field display
        cM: colormap name
        takeLog: whether to log scale (1/0)
        zFieldLim: color limits (min, max)
        zWidth: projection width in kpc
        fsize: font size
        wField: weight field name (0 for no weighting)
        ovHalo: halo data for overplotting (0 to disable)
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
    if zWidth is None:
        zWidth = config.get('analysis_options.fig_width', 30)
    if bSize is None:
        bSize = config.get('analysis_options.buffer_size', 800)
    if plotSize is None:
        plotSize = config.get('plotting_defaults.yt_figsize', 12)
    if fsize is None:
        fsize = config.get('plotting_defaults.fontsize', 12)
    if saveFig is None:
        saveFig = config.get('plotting_defaults.save_plots', True)
    if showFig is None:
        showFig = config.get('plotting_defaults.show_plots', False)
    
    if message:
        log.logger.info(f"\n{message}")
    
    # Option setup
    axNum = 2 if twoAxis else 1

    # Panel fig setup
    panelSize = (axNum, len(simArr))
    panelFig = plt.figure()
    panelGrid = AxesGrid(panelFig, (0, 0, 1, 1), nrows_ncols=panelSize, axes_pad=0.1,
                        label_mode="1", share_all=True, cbar_location="right",
                        cbar_mode="single", cbar_size="5%", cbar_pad="2%")

    # Loading snapshots
    snapArr = [simArr[i].ytFull[idxArr[i]] for i in range(len(simArr))]
    titleArr = [simArr[i].name for i in range(len(simArr))]

    # Start of the fig making
    log.logger.info(f"  Setup complete - Starting fig making for {zField}")
    for i, snap in enumerate(snapArr):
        log.logger.info(f"  - Projecting {simArr[i].name} at time {simArr[i].snap[idxArr[i]].time:.1f} "
                       f"Myr Redshift {simArr[i].snap[idxArr[i]].z:.2f}")

        # Sets plotting options as detailed
        log.logger.debug(f"    Projecting in axis {axisProj[0]}")
        if takeLog == 0:
            snap.field_info[(part, zField)].take_log = False
        
        if wField != 0:
            fig1 = yt.ProjectionPlot(snap, axisProj[0], (part, zField), window_size=plotSize, 
                                    weight_field=(part, wField), fontsize=fsize, 
                                    center=simArr[i].snap[idxArr[i]].ytcen)
        else:
            if part == "PartType4":
                fig1 = yt.ParticleProjectionPlot(snap, axisProj[0], (part, zField), 
                                                window_size=plotSize, depth=(zWidth, "kpc"), 
                                                fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
            else:
                fig1 = yt.ProjectionPlot(snap, axisProj[0], (part, zField), 
                                        window_size=plotSize, fontsize=fsize, 
                                        center=simArr[i].snap[idxArr[i]].ytcen)
        
        fig1.set_width(zWidth, "kpc")
        if zFieldUnit != 0:
            fig1.set_unit((part, zField), zFieldUnit)
        if zFieldLim != 0:
            fig1.set_zlim((part, zField), zmin=zFieldLim[0], zmax=zFieldLim[1])
        fig1.set_cmap(field=(part, zField), cmap=cM)
        fig1.set_buff_size(bSize)
        fig1.annotate_timestamp(redshift=True)

        # Plots a second axis if specified
        if twoAxis:
            log.logger.debug(f"    Projecting in axis {axisProj[1]}")
            if wField != 0:
                fig2 = yt.ProjectionPlot(snap, axisProj[1], (part, zField), 
                                        window_size=plotSize, weight_field=(part, wField), 
                                        fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen) 
            else:
                if part == "PartType4":
                    fig2 = yt.ParticleProjectionPlot(snap, axisProj[1], (part, zField), 
                                                    window_size=plotSize, depth=(zWidth, "kpc"), 
                                                    fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
                else:
                    fig2 = yt.ProjectionPlot(snap, axisProj[1], (part, zField), 
                                            window_size=plotSize, fontsize=fsize, 
                                            center=simArr[i].snap[idxArr[i]].ytcen)

            fig2.set_width(zWidth, "kpc")
            if zFieldUnit != 0:
                fig2.set_unit((part, zField), zFieldUnit)
            if zFieldLim != 0:
                fig2.set_zlim((part, zField), zmin=zFieldLim[0], zmax=zFieldLim[1])
            fig2.set_cmap(field=(part, zField), cmap=cM)
            fig2.set_buff_size(bSize)
            fig2.annotate_timestamp(redshift=True)

        # Transfers yt plot to plt axes and renders the figure
        log.logger.debug(f"    Rendering {simArr[i].name}")
        fullPlot = fig1.plots[part, zField]
        fullPlot.figure = panelFig
        fullPlot.axes = panelGrid[i].axes
        fullPlot.cax = panelGrid.cbar_axes[i]

        if twoAxis:
            fullPlot2 = fig2.plots[part, zField]
            fullPlot2.figure = panelFig
            fullPlot2.axes = panelGrid[len(simArr) + i].axes
            fullPlot2.cax = panelGrid.cbar_axes[len(simArr) + i]
            fig2._setup_plots()

        fig1._setup_plots()

        # Overplot halos if prompted to and passed
        if ovHalo != 0:
            log.logger.debug("    Overplotting halos")
            haloData = ovHalo[0][i].all_data()
            haloFilt = ovHalo[1]
            
            xc = np.array(haloData['particle_position_x'][haloFilt[i]].in_units("kpc")) - \
                 simArr[i].snap[idxArr[i]].center[0] / 1e3
            yc = np.array(haloData['particle_position_y'][haloFilt[i]].in_units("kpc")) - \
                 simArr[i].snap[idxArr[i]].center[1] / 1e3
            zc = np.array(haloData['particle_position_z'][haloFilt[i]].in_units("kpc")) - \
                 simArr[i].snap[idxArr[i]].center[2] / 1e3
            rc = np.array(haloData['virial_radius'][haloFilt[i]].in_units("kpc")) * 1e3
            
            for j in range(len(xc)):
                panelGrid.axes_all[i].add_patch(plt.Circle((xc[j], yc[j]), rc[j], 
                                                          ec="r", fc="none"))
                
                if twoAxis:
                    panelGrid.axes_all[len(simArr) + i].add_patch(
                        plt.Circle((yc[j], zc[j]), rc[j], ec="r", fc="none"))

        # Sets title
        panelGrid.axes_all[i].set_title(titleArr[i])
    
    return handleFig(panelFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)
