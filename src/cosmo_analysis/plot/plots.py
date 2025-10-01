# TODO: Description of file for documentation
# TODO: Fix logging
# TODO: Description of functions for documentation


import yt
import numpy as np
import io
import os.path
import math
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from   yt.frontends.halo_catalog.data_structures import HaloDataset
from   yt.utilities.cosmology import Cosmology
from   matplotlib.offsetbox import AnchoredText
from   scipy.stats import kde
from   matplotlib  import rc_context
from   mpl_toolkits.axes_grid1 import AxesGrid
from   PIL import Image

from ..core.constants import *
from ..core.utils     import *
from ..io.load        import *
from .. import log

yt.set_log_level(0)
savePath = "/sqfs/work/hp240141/z6b616/analysis"

# PLOTTING MACRO FUNCTIONS (rerun whenever config options are changed for new defaults)
# Return the frame for an animation
def saveFrame(figure,verbose):
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    buf.seek(0)
    img = np.array(Image.open(buf))
    log.logger.debug("Figure saved to buffer")
    buf.close()
    plt.close(figure)
    return img

def setLegend(uAx,sims,idx):
    timeInfo = [None]*len(sims)
    for i in range(len(sims)):
        timeInfo[i] = " t="+str(round(sims[i].snap[idx[i]].time))+" Myr"
        if sims[i].cosmo: timeInfo[i] = " z="+str(round(sims[i].snap[idx[i]].z,2))
    uAx.legend([sims[i].name+timeInfo[i] for i in range(len(sims))])

def handleFig(figure, switches, message, saveFigPath, verbose):
    """Handle figure display, saving, or animation frame capture.
    
    Args:
        figure: matplotlib figure object
        switches: tuple of (show, animate, save) boolean flags
        message: filename or message for saved figure
        saveFigPath: directory path to save figure (overrides default)
        verbose: verbosity level for logging
    
    Returns:
        numpy array of image if animate flag is True, None otherwise
    """
    # Shows figure
    if switches[0]:
        log.logger.debug("Showing to screen")
        plt.show()

    # Return the frame for an animation
    if switches[1]:     
        return saveFrame(figure,verbose)
    
    # Saves figure with message path as title
    if switches[2]:
        # Determine base directory for saving
        if saveFigPath != 0:
            # Use provided path
            base_dir = saveFigPath
        else:
            # Try to get from config, fall back to old constant
            try:
                from ..config import get_config
                config = get_config()
                base_dir = config.get('paths.output_directory', savePath)
            except:
                base_dir = savePath
        
        # Construct full path
        if message != 0:
            fullPath = os.path.join(base_dir, message.replace(" ","_")+".png")
        else:
            fullPath = os.path.join(base_dir, "placeholder.png")
            log.logger.warning("TITLE NOT SPECIFIED FOR THIS FIGURE, PLEASE SPECIFY A TITLE")

        log.logger.info(f"  Saving figure to {fullPath}")
        
        # Get DPI from config if available
        try:
            from ..config import get_config
            config = get_config()
            dpi = config.get('plotting_defaults.dpi', 300)
        except:
            dpi = 300
            
        figure.savefig(fullPath, bbox_inches='tight', pad_inches=0.03, dpi=dpi)
        plt.close(figure)

# Plot a cool multipanel for multiple simulations and/or projections
def ytMultiPanel(sims, idx, zField = ["Density"], axisProj = 0, part = gasPart, zFieldUnit = "g/cm**2", cM = "algae", takeLog=1, zFieldLim = [1.5e-4,1e-1], wField=0, zWidth=figWidth,bSize=buffSize,
                flipOrder=0,
                verbose=verboseLevel, plotSize=ytFigSize, saveFig=saveAll, saveFigPath=0, showFig=showAll, message=0, fsize=fontSize, animate=0):
        
    if message != 0: log.logger.info(f"\n{message}")
    # Option setup
    numP = len(zField)
    numS = len(sims)
    if isinstance(axisProj  ,list) == False: axisProj   = [axisProj  ]*numS
    if isinstance(part      ,list) == False: part       = [part      ]*numP
    if isinstance(zFieldUnit,list) == False: zFieldUnit = [zFieldUnit]*numP
    if isinstance(cM        ,list) == False: cM         = [cM        ]*numP
    if isinstance(takeLog   ,list) == False: takeLog    = [takeLog   ]*numP
    if isinstance(wField    ,list) == False: wField     = [wField    ]*numP
    if isinstance(zWidth    ,list) == False: zWidth     = [zWidth    ]*numP
    if isinstance(zFieldLim ,list) == False: zFieldLim  = [zFieldLim ]*numP
    if isinstance(bSize     ,list) == False: bSize      = [bSize     ]*numP
    if not all(isinstance(lIdx, list) for lIdx in zFieldLim):
        zFieldLimAux = [None]*numP
        for j in range(numP): zFieldLimAux[j] = zFieldLim 
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
    panelGrid = AxesGrid(panelFig,(0,0,1,1),nrows_ncols=panelSize,axes_pad=0.02,label_mode="1",share_all=False,cbar_location=loc,cbar_mode="edge",cbar_size="5%",cbar_pad="2%")

    # Loading snapshots
    snapArr  = [sims[i].ytFull[idx[i]] for i in range(numS)]
    titleArr = [sims[i].name           for i in range(numS)]

    for i,snap in enumerate(snapArr):
        log.logger.info(f"  - Projecting {sims[i].name} Time {sims[i].snap[idx[i]].time:.1f} Redshift {sims[i].snap[idx[i]].z:.2f} Axis {axisProj[i]}")
        for j,pField in enumerate(zField):
            iterRow, iterCol = (j,i)
            if flipOrder: iterRow, iterCol = (i,j)

            log.logger.debug(f"    Projecting field {pField} Particle {part[j]} Weight {wField[j]} Width {zWidth[j]} Unit {zFieldUnit[j]} Lim {zFieldLim[j]}")
            # Setup projection of pField of snap
            if takeLog[j] == 0: snap.field_info[(part[j], pField)].take_log = False
            if wField[j] != 0:
                fig1 = yt.ProjectionPlot(snap, axisProj[i], (part[j], pField), window_size=plotSize, weight_field=(part[j],wField[j]), fontsize=fsize, center=sims[i].snap[idx[i]].ytcen)
            else:
                if part[j]=="PartType4":
                    fig1 = yt.ParticleProjectionPlot(snap, axisProj[i], (part[j], pField), window_size=plotSize, depth=(zWidth[j],"kpc"), fontsize=fsize, center=sims[i].snap[idx[i]].ytcen)
                else:
                    fig1 = yt.ProjectionPlot(snap, axisProj[i], (part[j], pField), window_size=plotSize, fontsize=fsize, center=sims[i].snap[idx[i]].ytcen)
            fig1.set_width(zWidth[j],"kpc")
            if zFieldUnit[j] != 0: fig1.set_unit((part[j], pField), zFieldUnit[j])
            if not(zFieldLim[j][0] == 0 and zFieldLim[j][1]  == 0): fig1.set_zlim((part[j], pField), zmin=zFieldLim[j][0], zmax=zFieldLim[j][1])
            fig1.set_cmap(field=(part[j], pField), cmap=cM[j])
            fig1.set_buff_size(bSize[j])
            if ((not flipOrder) and iterRow == 0) or (flipOrder and iterCol == 0):  fig1.annotate_timestamp(redshift=True)

            # Transfers yt plot to plt axes and renders the figure
            fullPlot        = fig1.plots[part[j], pField]
            fullPlot.figure = panelFig
            fullPlot.axes   = panelGrid[iterCol+iterRow*len(colIter)].axes
            
            fullPlot.cax = panelGrid.cbar_axes[iterRow]
            if flipOrder: fullPlot.cax = panelGrid.cbar_axes[iterCol]

            log.logger.debug("    Rendering")
            fig1._setup_plots()

            if ((not flipOrder) and iterRow == 0) or (flipOrder and iterCol == 0): 
                nameTag = AnchoredText(titleArr[i], loc=2, prop=dict(size=9), frameon=True)
                panelGrid[iterCol+iterRow*len(colIter)].axes.add_artist(nameTag)

    handleFig(panelFig,[showFig,animate,saveFig],message,saveFigPath,verbose)
    
# Plots field projections, defaults to standard density projection
def ytProjPanel(simArr, idxArr, verbose=verboseLevel, plotSize=ytFigSize, saveFig=saveAll, saveFigPath=0, showFig=showAll,
                message=0, twoAxis=True, axisProj = [2,0], part = "PartType0", bSize=buffSize, zField = "Density",
                zFieldUnit = "g/cm**2", cM = "algae",takeLog=1, zFieldLim = (1.5e-4,1e-1), zWidth=figWidth, fsize=fontSize,
                wField=0, ovHalo=0, animate=0):
        
    if message != 0: log.logger.info(f"\n{message}")
    # Option setup
    axNum = 1
    if twoAxis: axNum = 2

    # Panel fig setup
    panelSize = (axNum, len(simArr))
    panelFig = plt.figure()
    panelGrid = AxesGrid(panelFig,(0,0,1,1),nrows_ncols=panelSize,axes_pad=0.1,label_mode="1",share_all=True,cbar_location="right",cbar_mode="single",cbar_size="5%",cbar_pad="2%")

    # Loading snapshots
    snapArr  = [simArr[i].ytFull[idxArr[i]] for i in range(len(simArr))]
    titleArr = [simArr[i].name              for i in range(len(simArr))]

    # Start of the fig making
    log.logger.info(f"  Setup complete - Starting fig making for {zField}")
    for i,snap in enumerate(snapArr):
        log.logger.info(f"  - Projecting {simArr[i].name} at time {simArr[i].snap[idxArr[i]].time:.1f} Myr Redshift {simArr[i].snap[idxArr[i]].z:.2f}")

        # Sets plotting options as detailed
        log.logger.debug(f"    Projecting in axis {axisProj[0]}")
        if takeLog == 0: snap.field_info[(part, zField)].take_log = False
        if wField != 0:
            fig1 = yt.ProjectionPlot(snap, axisProj[0], (part, zField), window_size=plotSize, weight_field=(part,wField), fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
        else:
            if part=="PartType4":
                fig1 = yt.ParticleProjectionPlot(snap, axisProj[0], (part, zField), window_size=plotSize, depth=(zWidth,"kpc"), fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
            else:
                fig1 = yt.ProjectionPlot(snap, axisProj[0], (part, zField), window_size=plotSize, fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
        fig1.set_width(zWidth,"kpc")
        if zFieldUnit != 0: fig1.set_unit((part, zField), zFieldUnit)
        if zFieldLim != 0:  fig1.set_zlim((part, zField), zmin=zFieldLim[0], zmax=zFieldLim[1])
        fig1.set_cmap(field=(part, zField), cmap=cM)
        fig1.set_buff_size(bSize)
        fig1.annotate_timestamp(redshift=True)

        # Plots a second axis if specified
        if twoAxis:
            log.logger.debug(f"    Projecting in axis {axisProj[1]}")
            if wField != 0:
                fig2 = yt.ProjectionPlot(snap, axisProj[1], (part, zField), window_size=plotSize, weight_field=(part,wField), fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen) 
            else:
                if part=="PartType4":
                    fig2 = yt.ParticleProjectionPlot(snap, axisProj[1], (part, zField), window_size=plotSize, depth=(zWidth,"kpc"), fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)
                else:
                    fig2 = yt.ProjectionPlot(snap, axisProj[1], (part, zField), window_size=plotSize, fontsize=fsize, center=simArr[i].snap[idxArr[i]].ytcen)

            fig2.set_width(zWidth,"kpc")
            if zFieldUnit != 0: fig2.set_unit((part, zField), zFieldUnit)
            if zFieldLim != 0:  fig2.set_zlim((part, zField), zmin=zFieldLim[0], zmax=zFieldLim[1])
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
            fullPlot2.axes = panelGrid[len(simArr)+i].axes
            fullPlot2.cax = panelGrid.cbar_axes[len(simArr)+i]
            fig2._setup_plots()

        fig1._setup_plots()

        # Overplot halos if prompted to and passed
        if ovHalo != 0:
            log.logger.debug("    Overplotting halos")
            haloData = ovHalo[0][i].all_data()
            haloFilt = ovHalo[1]
            #log.logger.info(haloData['particle_position_x'][haloFilt[i]].in_units("kpc"))
            #log.logger.info(simArr[i].snap[idxArr[i]].center[0],simArr[i].snap[idxArr[i]].center[1],simArr[i].snap[idxArr[i]].center[2])
            xc = np.array(haloData['particle_position_x'][haloFilt[i]].in_units("kpc"))-simArr[i].snap[idxArr[i]].center[0]/1e3
            yc = np.array(haloData['particle_position_y'][haloFilt[i]].in_units("kpc"))-simArr[i].snap[idxArr[i]].center[1]/1e3
            zc = np.array(haloData['particle_position_z'][haloFilt[i]].in_units("kpc"))-simArr[i].snap[idxArr[i]].center[2]/1e3
            rc = np.array(haloData['virial_radius'][haloFilt[i]].in_units("kpc"))*1e3
            #log.logger.info(xc,yc,zc,rc)
            for j in range(len(xc)):
                panelGrid.axes_all[i].add_patch(plt.Circle((xc[j],yc[j]),rc[j],ec="r",fc="none"))
                
                if twoAxis:
                    panelGrid.axes_all[len(simArr)+i].add_patch(plt.Circle((yc[j],zc[j]),rc[j],ec="r",fc="none"))

        # Sets title
        panelGrid.axes_all[i].set_title(titleArr[i])
    
    handleFig(panelFig,[showFig,animate,saveFig],message,saveFigPath,verbose)
    
# Plots phase space 2D histograms, defaults to gas phase (Density, Temperature, Mass)
def ytPhasePanel(simArr, idxArr, depositionAlg="ngp", verbose=verboseLevel, plotSize=ytFigSize, saveFig=saveAll,
                 saveFigPath=0, showFig=showAll,message=0, blackLine=0, panOver=0, part = "PartType0", zLog=1,
                 zFields = ["Density","Temperature","Masses"], zFieldUnits = ["g/cm**3","K","Msun"], cM = "algae", animate = 0,
                 zFieldLim = (1e3,1e8,1e-29,1e-21,10,1e7), zWidth=15, fsize=12, wField=0, xb=300, yb=300, grid=True, axAspect = 1):
    if message != 0: log.logger.info(f"\n{message}")
    # Panel fig setup
    if isinstance(wField,list) == False: wField = [wField]*len(simArr)
    if panOver == 0:
        panelSize = (1, math.ceil(len(simArr)))
    else:
        panelSize = panOver
    panelFig = plt.figure(figsize=(1,1))
    panelGrid = AxesGrid(panelFig,(0,0,0.4*panelSize[1],0.4*panelSize[0]),aspect=False,nrows_ncols=panelSize,axes_pad=0.1,
                         label_mode="1",share_all=True,cbar_location="right",cbar_mode="single",cbar_size="5%",cbar_pad="2%")
    
    if zFieldLim   == 0: zFieldLim   = [0,0,0,0,0,0]
    if zFieldUnits == 0: zFieldUnits = [0,0,0]

    # Loading snapshots
    snapArr  = [simArr[i].ytFull[idxArr[i]] for i in range(len(simArr))]
    titleArr = [simArr[i].name              for i in range(len(simArr))]

    # Getting the black line
    if blackLine:
        log.logger.debug(f"  Calculating avg profile with {simArr[0].name}")
        sp = snapArr[0].sphere(simArr[0].snap[idxArr[0]].ytcen,(zWidth,"kpc"))

        p1 = yt.ProfilePlot(sp,(part,zFields[0]),(part,zFields[1]),weight_field=(part,zFields[2]), n_bins=30, x_log=False, accumulation=False)
        
        p1.set_log((part,zFields[0]),True)
        p1.set_log((part,zFields[1]),True)

        if zFieldUnits[0] != 0: p1.set_unit((part,zFields[0]), zFieldUnits[0])
        if zFieldLim[2] != 0 or zFieldLim[3] != 0: p1.set_xlim(zFieldLim[2], zFieldLim[3])
        if zFieldUnits[1] != 0: p1.set_unit((part,zFields[1]), zFieldUnits[1])

        cil = p1.profiles[0].x.in_units(zFieldUnits[0]).d
        bin = p1.profiles[0][zFields[1]].in_units(zFieldUnits[1]).d
        goodbin = []
        goodcil = []
        for i in range(len(bin)):
            if abs(bin[i]) > 1e-33:
                goodbin.append(bin[i]) 
                goodcil.append(cil[i])
           
    # Start of the fig making
    for i,snap in enumerate(snapArr):
        log.logger.info(f"  - Plotting {simArr[i].name}")
        sp = snap.sphere(simArr[i].snap[idxArr[i]].ytcen,(zWidth,"kpc"))
        # Plot phase with specified parameters
        if zLog != 1:
            snap.field_info[(part, zFields[2])].take_log = False
        
        if wField[i] != 0:
            fig1 = yt.ParticlePhasePlot(sp,  (part, zFields[0]),(part, zFields[1]),(part, zFields[2]), deposition=depositionAlg,
                                         figure_size=plotSize, weight_field=(part,wField[i]), fontsize=fsize, x_bins=xb, y_bins=yb)
        else:
            fig1 = yt.ParticlePhasePlot(sp,  (part, zFields[0]),(part, zFields[1]),(part, zFields[2]), deposition=depositionAlg,
                                         figure_size=plotSize, fontsize=fsize, x_bins=xb, y_bins=yb)
            
        
        if zFieldUnits[0] != 0: fig1.set_unit((part, zFields[0]), zFieldUnits[0])
        if zFieldUnits[1] != 0: fig1.set_unit((part, zFields[1]), zFieldUnits[1])
        if zFieldUnits[2] != 0: fig1.set_unit((part, zFields[2]), zFieldUnits[2])
            
        if zFieldLim[0] != 0 or zFieldLim[1] != 0: fig1.set_zlim((part, zFields[2]), zmin=zFieldLim[0], zmax=zFieldLim[1])
        if zFieldLim[2] != 0 or zFieldLim[3] != 0: fig1.set_xlim(zFieldLim[2], zFieldLim[3])
        if zFieldLim[4] != 0 or zFieldLim[5] != 0: fig1.set_ylim(zFieldLim[4], zFieldLim[5])

        fig1.set_log((part,zFields[2]),bool(zLog))

        fig1.set_cmap(field=(part, zFields[2]), cmap=cM)

        #fig1.annotate_text(0,0,"t="+str(round(sims[i].snap[idx[i]].time))+" Myr\n z="+str(round(sims[i].snap[idx[i]].z,2)))

        # Transfers yt plot to plt axes and renders the figure
        log.logger.debug(f"    Rendering {simArr[i].name}")
        fullPlot = fig1.plots[part, zFields[2]]
        fullPlot.figure = panelFig
        fullPlot.axes = panelGrid[i].axes
        if i == 0:
            fullPlot.cax = panelGrid.cbar_axes[i]
        
        fig1._setup_plots()

        if blackLine:
            panelGrid.axes_all[i].plot(goodcil,goodbin,"k--")
        panelFig.canvas.draw()
        if grid: panelGrid.axes_all[i].grid()
        panelGrid.axes_all[i].set_title(titleArr[i])
        panelGrid.axes_all[i].set_box_aspect(axAspect)
        #panelGrid.axes_all[i].set_aspect(axAspect)
        
    handleFig(panelFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plots a binned field
def plotBinned(sims,idx,binFields,nBins,rLim,logOverload=0,legOverload=0,diffSims=0,blLine=0,wField=0,spLim=0,binFunction=0,part=gasPart,setUnits=0,setLogs=(False,True),ylims=0,xlims=0,animate=0,
               xylabels=0,plotTitle=0,errorLim=errorLimGlobal,message=0,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,showError=showErrorGlobal,axAspect=1):
    if message != 0: log.logger.info(f"\n{message}")
    # Initialize figures
    if showError == 1:
        if isinstance(plotSize,list) == False: plotSize = [plotSize,plotSize*1.2] 
        uFig = plt.figure(figsize=(plotSize[0], plotSize[1]))
        uAx  = plt.subplot2grid((4,1),(0,0),rowspan=3)
        uAx2 = plt.subplot2grid((4,1),(3,0),rowspan=1)
    elif showError == 2:
        if isinstance(plotSize,list) == False: plotSize = [plotSize,plotSize*1.5] 
        uFig = plt.figure(figsize=(plotSize[0], plotSize[1]))
        uAx  = plt.subplot2grid((5,1),(0,0),rowspan=3)
        uAx2 = plt.subplot2grid((5,1),(3,0),rowspan=2)
    else:
        if isinstance(plotSize,list) == False: plotSize = [plotSize,plotSize] 
        uFig = plt.figure(figsize=(plotSize[0], plotSize[1]))
        uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)
 
    allYbin = [None]*len(sims)
    XGlobal = None
    for k,sn in enumerate(sims):

        # Setting up parameters
        log.logger.info(f"  Started {sn.name}")
        splimit = rLim[1]
        if spLim != 0: splimit = spLim
        weightField = None
        if wField != 0: weightField = wField

        # Actual binning proccess
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(splimit,"kpc"))
        p1 = yt.ProfilePlot(sp,(part,binFields[0]),(part,binFields[1]),weight_field=weightField, n_bins=nBins, x_log=setLogs[0], accumulation=False)
        p1.set_log((part,binFields[0]),setLogs[0])
        p1.set_log((part,binFields[1]),setLogs[1])
        if setUnits[0] != 0: p1.set_unit((part,binFields[0]), setUnits[0])
        if setUnits[1] != 0: p1.set_unit((part,binFields[1]), setUnits[1])
        p1.set_xlim(rLim[0],rLim[1])

        # Extract bins to perform further operations
        if setUnits[0] != 0: 
            xData = p1.profiles[0].x.in_units(setUnits[0]).d 
        else: 
            xData = p1.profiles[0].x.d
        if setUnits[1] != 0: 
            bin = p1.profiles[0][binFields[1]].in_units(setUnits[1]).d
        else: 
            bin = p1.profiles[0][binFields[1]].d

        # If there is a postproccessing function, do that
        if binFunction != 0: bin = binFunction(xData,bin)

        # Performs the difference between two bins of two simulation sets (of the same length!!)
        if diffSims != 0:
            sp2 = diffSims[0][k].ytFull[diffSims[1][k]].sphere(diffSims[0][k].snap[diffSims[1][k]].ytcen,(splimit,"kpc"))
            p2 = yt.ProfilePlot(sp2,(part,binFields[0]),(part,binFields[1]),weight_field=weightField, n_bins=nBins, x_log=setLogs[0], accumulation=False)
            p2.set_log((part,binFields[0]),setLogs[0])
            p2.set_log((part,binFields[1]),setLogs[1])
            if setUnits != 0: p2.set_unit((part,binFields[0]), setUnits[0])
            if setUnits != 0: p2.set_unit((part,binFields[1]), setUnits[1])
            p2.set_xlim(rLim[0],rLim[1])

            xData2 = p2.profiles[0].x.in_units(setUnits[0]).d
            bin2   = p2.profiles[0][binFields[1]].in_units(setUnits[1]).d
            if binFunction != 0: bin2 = binFunction(xData2,bin2)
            bin = np.array(bin)-np.array(bin2)

        allYbin[k] = bin
        XGlobal = xData
        uAx.plot(xData,bin,".--")

        # Setups log scaling depending on options
        setLogsPlot = setLogs
        if logOverload != 0: setLogsPlot = logOverload
        if setLogsPlot[0]: uAx.semilogx()
        if setLogsPlot[1]: uAx.semilogy()
        uAx.set_box_aspect(axAspect)
        
    # Plots dispersion of codes
    if showError == 1:
        log.logger.debug("  Plotting dispersion")
        allYbin = np.array(allYbin)
        error = [None]*nBins
        for i in range(nBins):
            average = np.mean(allYbin[:,i])
            if average != 0:
                error[i] = (allYbin[:,i]-average)/average
            else:
                error[i] = np.zeros_like(allYbin[:,i])
        error = np.array(error)
        for i in range(len(sims)):
            uAx2.plot(XGlobal,error[:,i],".")
            if setLogs[0]: uAx2.semilogx()
            
        if xlims != 0: uAx2.set_xlim(xlims[0],xlims[1])
        uAx2.set_ylim(errorLim[0],errorLim[1])
        uAx2.grid()

        if xylabels != 0: uAx2.set_ylabel("Residual ($\\frac{\sigma - \overline{\sigma}}{\overline{\sigma}}$)")
        if xylabels != 0: uAx2.set_xlabel(xylabels[0])
    elif showError == 2:
        allYbin = np.array(allYbin)
        for i in range(len(sims)):
            uAx2.plot(XGlobal,allYbin[i,:]/1e8,".")

        if xlims != 0: uAx2.set_xlim(xlims[0],xlims[1])
        uAx2.set_ylim(errorLim[0],errorLim[1])
        uAx2.grid()
        uAx2.set_xscale('log')
        uAx2.set_yscale('symlog',linthresh=0.01) 
        if xylabels != 0: uAx2.set_ylabel("Log "+xylabels[1])
        if xylabels != 0: uAx2.set_xlabel(xylabels[0])

        if blLine != 0:
            uAx2.axvline(x = blLine, color='k', linestyle ='--', linewidth=2, alpha=0.7)
    else:
        if xylabels != 0: uAx.set_xlabel(xylabels[0])
    
    # Set limits and labels
    if xylabels != 0:  uAx.set_ylabel(xylabels[1])
    if plotTitle != 0: uAx.set_title(plotTitle)
    if xlims != 0: uAx.set_xlim(xlims[0],xlims[1])
    if ylims != 0: uAx.set_ylim(ylims[0],ylims[1])
    
    # Threshold of sff
    if blLine != 0:
        uAx.axvline(x = blLine, color='k', linestyle ='--', linewidth=2, alpha=0.7)

    uAx.grid()
    
    if legOverload == 0:
        setLegend(uAx,sims,idx)
    else:
        uAx.legend(legOverload)
    
    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Calculates the velocity dispersion for a given particle type
def plotRotDisp(sims,idx,nBins,rLim,part,titlePlot=0,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,ylims=(0,170),animate=0):
    if message != 0: log.logger.info(f"\n{message}")
    uFig = plt.figure(figsize=(plotSize, plotSize*1.5))
    uAx  = plt.subplot2grid((5,1),(0,0),rowspan=3)
    uAx2 = plt.subplot2grid((5,1),(3,0),rowspan=1)
    uAx3 = plt.subplot2grid((5,1),(4,0),rowspan=1)
	
    snapArr  = [sims[i].ytFull[idx[i]] for i in range(len(sims))]
    centArr  = [sims[i].snap[idx[i]].ytcen for i in range(len(sims))]
	
    # New fields (this one is complicated, I need first to get each particle to have a field with its rotational vel from the bin it is in, and then using that as mean calculate the dispersion)
    for i,snap in enumerate(snapArr):

		# Get the velocity bins
        sp = snap.sphere(centArr[i],(rLim,"kpc"))    
        rotProf = yt.ProfilePlot(sp,(part,"particle_position_cylindrical_radius"),(part,"particle_velocity_cylindrical_theta"),weight_field=(part,"Masses"), n_bins=nBins, x_log=False)
        rotProf.set_log((part,"particle_position_cylindrical_radius"),False)
        rotProf.set_log((part,"particle_velocity_cylindrical_theta"),False)
        rotProf.set_unit((part,"particle_velocity_cylindrical_theta"), 'km/s')
        rotProf.set_unit((part,"particle_position_cylindrical_radius"), 'kpc')
        rotProf.set_xlim(0, rLim-1)
        rotProf.set_ylim((part,"particle_velocity_cylindrical_theta"), 0, 250)
        rotCilLocal = rotProf.profiles[0].x.in_units('kpc').d
        rotBinLocal = rotProf.profiles[0]["particle_velocity_cylindrical_theta"].in_units('km/s').d
	
        # Defined the functions each time, so that the rotCil and rotBin are properly set and correspond to the correct snapshot
        def make_field_func(rotCil,rotBin):
            # Get the x velocity of a particle using its binned rot vel and its angle
            def _particle_rot_vx(field, data):
                trans = np.zeros(data[(part,"particle_velocity_x")].shape)
                dr = 0.5*(rotCil[1]-rotCil[0])
                # Go through each bin
                for rad, vrot in zip(rotCil,rotBin):
                    # Select the indices of particles inside this bin
                    ind = np.where( (data[(part,"particle_position_cylindrical_radius")].in_units("kpc") >= (rad - dr)) & \
                                    (data[(part,"particle_position_cylindrical_radius")].in_units("kpc") <  (rad + dr)) )
                    # For the particles inside this bin, calculate their x velocity with the bin average 
                    trans[ind] = -np.sin(data[(part, "particle_position_cylindrical_theta")][ind]) * vrot * 1e5 
                # Return trans but with whatever code units this dataset uses
                return data.ds.arr(trans, "cm/s").in_base(data.ds.unit_system.name)
            
            # Get the y velocity of a particle using its binned rot vel and its angle
            def _particle_rot_vy(field, data):
                trans = np.zeros(data[(part,"particle_velocity_y")].shape)
                dr = 0.5*(rotCil[1]-rotCil[0])
                # Go through each bin
                for rad, vrot in zip(rotCil,rotBin):
                    # Select the indices of particles inside this bin
                    ind = np.where( (data[(part,"particle_position_cylindrical_radius")].in_units("kpc") >= (rad - dr)) & \
                                    (data[(part,"particle_position_cylindrical_radius")].in_units("kpc") <  (rad + dr)) )
                    # For the particles inside this bin, calculate their x velocity with the bin average 
                    trans[ind] = np.cos(data[(part, "particle_position_cylindrical_theta")][ind]) * vrot * 1e5 
                # Return trans but with whatever code units this dataset uses
                return data.ds.arr(trans, "cm/s").in_base(data.ds.unit_system.name)
            return _particle_rot_vx, _particle_rot_vy

        vx_func, vy_func = make_field_func(rotCilLocal, rotBinLocal)
		
        snap.add_field((part, "particle_rot_vx"), function=vx_func, take_log=False,units="cm/s",sampling_type="particle",force_override=True) 
        snap.add_field((part, "particle_rot_vy"), function=vy_func, take_log=False,units="cm/s",sampling_type="particle",force_override=True) 
		
        # Take the dispersion with respect to the velocity obtained from each bin
        def _particle_vel_disp(field, data):
            return (data[(part, "particle_velocity_x")] - data[(part, "particle_rot_vx")])**2 + \
				   (data[(part, "particle_velocity_y")] - data[(part, "particle_rot_vy")])**2 + \
				   (data[(part, "particle_velocity_z")])**2 
    
        snap.add_field((part, "particle_vel_disp"), function=_particle_vel_disp, take_log=False,units="cm**2/s**2",sampling_type="particle",force_override=True) 
        
        def _particle_velocity_z_squared(field, data):
            return (data[(part, "particle_velocity_z")])**2 
        snap.add_field((part, "particle_velocity_z_squared"), function=_particle_velocity_z_squared, take_log=False, units="cm**2/s**2", sampling_type="particle", force_override=True) 

    allYbin = [None]*len(sims)
    allYbinZ = [None]*len(sims)
    XGlobal = None

    for k,sn in enumerate(sims):

        log.logger.info(f"  Started {sn.name}")
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(rLim,"kpc"))
        
        p1 = yt.ProfilePlot(sp,(part,"particle_position_cylindrical_radius"),(part,"particle_vel_disp"),weight_field=(part,"Masses"), n_bins=nBins, x_log=False)
        p1.set_log((part,"particle_position_cylindrical_radius"), False)
        p1.set_unit((part,"particle_position_cylindrical_radius"), "kpc")
        p1.set_xlim(1e-3, rLim-1)
        
        cil = p1.profiles[0].x.in_units('kpc').d
        bins = np.sqrt(p1.profiles[0]["particle_vel_disp"]).in_units('km/s').d

        # Vertical z speed dispersion
        p2 = yt.ProfilePlot(sp, (part, "particle_position_cylindrical_radius"), (part, "particle_velocity_z_squared"),weight_field=(part,"Masses"), n_bins=nBins, x_log=False)
        p2.set_log((part,"particle_position_cylindrical_radius"), False)
        p2.set_unit((part,"particle_position_cylindrical_radius"), "kpc")
        p2.set_xlim(1e-3, rLim-1)
        allYbinZ[k] = np.sqrt(p2.profiles[0]["particle_velocity_z_squared"]).in_units('km/s').d
        
        allYbin[k] = bins
        XGlobal = cil
        uAx.plot(cil,bins,".--")
    
    allYbin = np.array(allYbin)
    error = [None]*nBins
    log.logger.debug("  Plotting dispersion")
    for i in range(nBins):
        average = np.mean(allYbin[:,i])
        if average != 0:
            error[i] = (allYbin[:,i]-average)/average
        else:
            error[i] = np.zeros_like(allYbin[:,i])
    error = np.array(error)
    for i in range(len(sims)):
        uAx2.plot(XGlobal,error[:,i],".")
    uAx2.set_xlim(0,(rLim-1))
    uAx2.set_ylim(-1,1)
    uAx2.set_ylabel("Residual ($\\frac{\sigma - \overline{\sigma}}{\overline{\sigma}}$)")
    uAx2.grid()
	
    allYbinZ = np.array(allYbinZ)
    log.logger.debug("  Plotting dispersion ratio")
    for i in range(len(sims)):
        dispRatio = allYbinZ[i,:]/allYbin[i,:]
        uAx3.plot(XGlobal,dispRatio,".")
    uAx3.set_xlim(0,(rLim-1))
    uAx3.set_ylim(0,1)
    uAx3.set_ylabel("Vertical dispersion ratio ($\\frac{\sigma_z}{\sigma}$)")
    uAx3.grid()
    uAx3.set_xlabel("Cylidrincal radius (Kpc)")

    uAx.set_xlim(0,14)
    if ylims != 0: uAx.set_ylim(ylims[0],ylims[1])

    
    uAx.set_ylabel("Velocity dispersion (km/s)")
    if titlePlot != 0: uAx.set_title(titlePlot)
    uAx.grid()

    setLegend(uAx,sims,idx)

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# CLUMP FINDING AND LOADING
def findHalos(simArr, idxArr, partT, mainPath, haloMethod = "fof", hopThresh = 4e9, fofLink = 0.0012, hardLimits = True, overWrite = True, clumpLim  = (1e6,8e8), verbose = verboseLevel):
    log.logger.info(f"\nInitiating halo finding for particle type: {partT} and Method: {haloMethod}") 
    # Initialize halo arrays
    haloSims = [None]*len(simArr)
    haloFilt = [None]*len(simArr)
    temp = None
    for i in range(len(simArr)):
        # Load parameters and paths
        log.logger.info(f"  - Loading halos for {simArr[i].name}")
        snap = simArr[i].ytFull[idxArr[i]]
        haloDirSim = os.path.join("Halos","Halo_"+haloMethod+"_"+partT+"_"+simArr[i].name.replace(" ","_"))
        haloPath = os.path.join(mainPath,haloDirSim)
        haloDirPath  = os.path.join(haloPath,snap.basename[:snap.basename.find(".")])
        haloFilePath = os.path.join(haloDirPath,snap.basename[:snap.basename.find(".")]+".0.h5")

        # Do the halo finding if no halos detected
        if os.path.exists(haloDirPath) == False or overWrite:
            # Explain what files are being modified or not
            if os.path.exists(haloDirPath) == False:
                log.logger.info(f"    No halos detected in {haloDirPath}")
            elif overWrite:
                log.logger.info(f"    Overwriting halos detected in {haloDirPath}")
            
            log.logger.info(f"    Initializing halo finding to be saved in {haloFilePath}")
            
            # Configure the halo catalog and halo finding method
            if      haloMethod == "hop":
                hopConf = hopThresh
                if isinstance(hopThresh, list): hopConf = hopThresh[i]
                hc = yt.extensions.astro_analysis.halo_analysis.HaloCatalog(data_ds=snap, data_source=snap.all_data(), finder_method="hop", output_dir=haloPath,finder_kwargs={"threshold": hopConf, "dm_only": False, "ptype": partT})
            elif    haloMethod == "fof":
                fofConf = fofLink
                if isinstance(fofLink, list): fofConf = fofLink[i]
                hc = yt.extensions.astro_analysis.halo_analysis.HaloCatalog(data_ds=snap, data_source=snap.all_data(), finder_method="fof", output_dir=haloPath,finder_kwargs={"link": fofConf, "dm_only": False, "ptype": partT})
            
            # Add filters and calculate the actual halo
            #hc.add_filter('quantity_value', 'particle_mass', '>', clumpLim[0], 'Msun') # exclude halos with less than 30 particles
            #hc.add_filter('quantity_value', 'particle_mass', '<', clumpLim[1], 'Msun') # exclude the most massive halo (threshold 1e8.4 is hand-picked, so one needs to be careful!)
            hc.create() 

        # Delete need for cosmological parameters
        def _parse_parameter_file_no_cosmo(self):
            # List of attributes expected by the halo dataset.
            for attr in [
                "cosmological_simulation",
                "cosmology",
                "current_redshift",
                "current_time",
                "dimensionality",
                "domain_dimensions",
                "domain_left_edge",
                "domain_right_edge",
                "domain_width",
                "hubble_constant",
                "omega_lambda",
                "omega_matter",
                "unique_identifier",
            ]:
                try:
                    setattr(self, attr, getattr(self.real_ds, attr))
                except AttributeError:
                    # If the attribute is missing, assign a default value or None
                    defVal = {"current_time": 0}
                    if attr in defVal:
                        setattr(self, attr, defVal[attr])
                    else:
                        setattr(self, attr, None)
        # Monkey-patch the method.
        HaloDataset._parse_parameter_file = _parse_parameter_file_no_cosmo

        # Now load the halos from disk file
        log.logger.info(f"    Loading halo from file{haloFilePath}")
        halo_ds  = yt.load(haloFilePath)
        hc = yt.extensions.astro_analysis.halo_analysis.HaloCatalog(halos_ds=halo_ds)
        hc.load()

        haloSims[i] = hc.halos_ds
        temp = haloSims[i].all_data()
        haloFilt[i] = np.ones(len(temp['particle_mass'].in_units("Msun")),dtype=bool)

        if hardLimits:    
            # Get the masses in Msun.
            mass = temp['particle_mass'][:].in_units("Msun")
            # Create a boolean mask for halos within the desired mass limits.
            keep = (mass >= clumpLim[0]) & (mass <= clumpLim[1])
            # Find the indices of the halos to keep.
            haloFilt[i] = np.where(keep)[0]
        
    log.logger.debug("  Halo loading successful!")
    return (haloSims,haloFilt)

# Plots the cumulative mass function of a collection of halos
def plotClumpMassF(sims,idx,haloData,nBins=20,mLim=(6,8.5),verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,animate=0):
    if message != 0: log.logger.info(f"\n{message}")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)

    # Calculate the cumulative mass function for each snapshot
    for k,sn in enumerate(sims):
        log.logger.info(f"  Started {sn.name}")
        temp = haloData[0][k].all_data()
        clumpMass = temp['particle_mass'][haloData[1][k]].in_units("Msun")
            
        clumpLogMass = np.log10(clumpMass)
        hist = np.histogram(clumpLogMass, bins=nBins, range=(mLim[0],mLim[1]))
        dBin = hist[1][1]-hist[1][0]
        
        uAx.plot(hist[1][:-1]+dBin, np.cumsum(hist[0][::-1])[::-1],".--")
        uAx.semilogy()
    
    # Decorate the plot
    uAx.set_xlim(mLim[0],mLim[1])
    uAx.set_ylim(0.9,50)
    uAx.set_xlabel("$\mathrm{log[Newly\ Formed\ Stellar\ Clump\ Mass\ (M_{\odot})]}$")
    uAx.set_ylabel("$\mathrm{Cumulative Stellar\ Clump\ Counts, \ \ N_{clump}(> M)}$")
    uAx.set_title("Clump Cumulative Mass Function")
    uAx.grid()

    setLegend(uAx,sims,idx)

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

def aFromT(time, eps = 0.1):
    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)
    if time < eps: return 0
    return co.a_from_t(co.quan(time,"Myr"))

# Plot the total SFR of a simulation over time
def plotSFR(sims,idx,nBins=25,tLimPreset = [0,0],verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,yLims=[0,8],animate=0,xLims=0):
    if message != 0: log.logger.info(f"\n{message}")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(figSize, figSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=3)
    uAx2 = plt.subplot2grid((4,1),(3,0),rowspan=1)

    # Bin star ages into nBins and use that to estimate total SFR
    allYbin = [None]*len(sims)
    XGlobal = None
    for k,sn in enumerate(sims):
        tLim = [tLimPreset[0],tLimPreset[1]]
        log.logger.info(f"  Started {sims[k].name} with time {sn.snap[idx[k]].time}")
        if tLimPreset[1] == 0: tLim[1] = sn.snap[idx[k]].time

        dt = (tLim[1]-tLim[0])/nBins
        timeX = np.linspace(tLim[0]+dt/2,tLim[1]-dt/2,nBins)
        starMass = [0]*nBins
        sfr = [0]*nBins

        prog = 0

        sp = sims[k].ytFull[idx[k]]
        allStarMass = np.array(sp.r["PartType4","Masses"].to("Msun"))
        
        allStarAge  = []
        binX        = None
        binLim      = None
        if sn.cosmo:
            allStarAge  = np.array(sp.r["PartType4","StellarFormationTime"])
            binX   = [aFromT(value) for value in timeX+dt/2]
            binLim = [aFromT(tLim[0]),aFromT(tLim[1])]
        else:
            allStarAge  = np.array(sp.r["PartType4","StellarFormationTime"])*1e3
            binX   = timeX+dt/2
            binLim = tLim

        for i in range(len(allStarAge)):
            if allStarAge[i] <= binLim[1] and allStarAge[i] >= binLim[0]:
                binIdx = getClosestIdx(binX,allStarAge[i])
                sfr[binIdx]      += allStarMass[i]/(dt*1e6)
                starMass[binIdx] += allStarMass[i]
            if i/len(allStarAge)*100-prog > 33:
                log.logger.debug(f"    {i/len(allStarAge)*100:.3f}%")
                prog = i/len(allStarAge)*100
        
        for i in range(nBins):
            if i == 0: continue
            starMass[i] += starMass[i-1]
            
        uAx.plot(timeX,sfr,".--")
        allYbin[k] = sfr
        XGlobal = timeX
    
    allYbin = np.array(allYbin)
    error = [None]*nBins
    for i in range(nBins):
        average = np.mean(allYbin[:,i])
        if average != 0:
            error[i] = (allYbin[:,i]-average)/average
        else:
            error[i] = np.zeros_like(allYbin[:,i])
    error = np.array(error)
    for i in range(len(sims)):
        uAx2.plot(XGlobal,error[:,i],".")
    uAx2.set_xlim(0,tLim[1])
    uAx2.set_ylim(-1,1)
    uAx2.set_ylabel("Residual ($\\frac{\sigma - \overline{\sigma}}{\overline{\sigma}}$)")
    uAx2.grid()
    uAx2.set_xlabel("Time (Myr)")

    if yLims != 0: uAx.set_ylim(yLims[0],yLims[1])
    if xLims != 0: uAx.set_xlim(xLims[0],xLims[1])
    
    uAx.set_ylabel("SFR ($\\frac{\mathrm{M}_{\odot}}{yr}$)")
    uAx.set_title("SFR Over time")
    uAx.grid()
    
    setLegend(uAx,sims,idx)

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plot the KS relation by binning cillindrically gas density and SFR
def plotKScil(sims,idx,nBins=50,rLim=0.5*figWidth,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,animate=0):
    if message != 0: log.logger.info(f"\n{message}")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)

    for k,sn in enumerate(sims):
        log.logger.info(f"  Started {sims[k].name} with time {sn.snap[idx[k]].time}")
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(rLim,"kpc"))
        
        # Calculate SFR den in cil bins
        p1 = yt.ProfilePlot(sp,(starPart,"particle_position_cylindrical_radius"),(starPart,"particle_mass_young_stars"),weight_field=None, n_bins=nBins, x_log=False)
        p1.set_log((starPart,"particle_position_cylindrical_radius"),False)
        p1.set_log((starPart,"particle_mass_young_stars"),True)
        p1.set_unit((starPart,"particle_mass_young_stars"), 'Msun')
        p1.set_unit((starPart,"particle_position_cylindrical_radius"), 'kpc')
        p1.set_xlim(1e-3, rLim)

        cil = p1.profiles[0].x.in_units('kpc').d
        binsPrev = p1.profiles[0]["particle_mass_young_stars"].in_units('Msun').d/youngStarAge/1e6

        dr = 0.5*(cil[1]-cil[0])
        SFRbins = []
        for i in range(len(cil)):
            SFRbins.append(binsPrev[i]/(np.pi * (((cil[i]+dr))**2-((cil[i]-dr))**2) ))

        # Calculate Gas den in cil bins
        p2 = yt.ProfilePlot(sp,(gasPart,"particle_position_cylindrical_radius"),(gasPart,"Masses"),weight_field=None, n_bins=nBins, x_log=False, accumulation=False)
        p2.set_log((gasPart,"Masses"),True)
        p2.set_log((gasPart,"particle_position_cylindrical_radius"),False)
        p2.set_unit((gasPart,"particle_position_cylindrical_radius"), 'kpc')
        p2.set_unit((gasPart,"Masses"), 'Msun')
        p2.set_xlim(0, rLim)

        rcil = p2.profiles[0].x.in_units('kpc').d
        massB = p2.profiles[0]["Masses"].in_units('Msun').d

        dr = 0.5*(rcil[1]-rcil[0])
        DENbins = []
        for i in range(len(rcil)):
            DENbins.append(massB[i]/(np.pi * (((rcil[i]+dr)*1e3)**2-((rcil[i]-dr)*1e3)**2) ))	

        # Calculate KS with both binned results
        # Filter low surf density bins
        ind = np.where(np.array(SFRbins) > 1e-10)

        xKS = np.log10(np.array(DENbins)[ind])
        yKS = np.log10(np.array(SFRbins)[ind])
        uAx.scatter(xKS,yKS)

        setLegend(uAx,sims,idx)

    # Obs KS line from 2008 Bigiel
    t = np.arange(-2, 5, 0.01)
    uAx.plot(t, 1.37*t - 3.78, 'k--', linewidth = 2, alpha = 0.7)

    # Obs KS contour from 2008 Bigiel
    fBigiel = open("bilcontour.txt","r+")
    dataX = []
    dataY = []
    for line in fBigiel:
        data = np.asarray(line.split(", "),dtype=float)
        dataX.append(data[0]+ np.log10(1.36))
        dataY.append(data[1])

    uAx.fill(dataX,dataY,fill=True, color='b', alpha = 0.1, hatch='\\')
    uAx.set_xlabel("$\mathrm{log[Gas\ Surface\ Density\ (M_{\odot}/pc^2)]}$")
    uAx.set_ylabel("$\mathrm{log[Star\ Formation\ Rate\ Surface\ Density\ (M_{\odot}/yr/kpc^2)]}$")
    uAx.set_xlim(0,3)
    uAx.set_ylim(-4,1)
    uAx.set_title("Kennicutt–Schmidt relation with cilindrically binned data")
    uAx.grid()

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plot the KS relation by binning gas density and SFR in squares tiling the whole galaxy (mock observations)
def plotKSmock(sims,idx,fsize=fontSize,rLim=0.5*figWidth,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,animate=0,resMock=lowResMock):
    axisProj     = 0
    zFieldLim1   = (1e0 , 1e3)
    zFieldLim2   = (3e-4, 3e-1)

    cmapDef = plt.get_cmap("tab10")
    if message != 0: log.logger.info(f"\n{message}")

    # Setup plot figures and axes
    uFig = plt.figure(figsize=(plotSize, plotSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)
    nMockBins = int(rLim*2*1e3/resMock)

    for k,sn in enumerate(sims):
        log.logger.info(f"  Started {sn.name}")

        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(rLim,"kpc"))
        
        # Calculate SFR den in mock rectangular bins
        log.logger.info(f"  Plotting {sims[k].name} in {axisProj} Time {sims[k].snap[idx[k]].time:.1f} Myr")
        fig1 = yt.ParticlePhasePlot(sp,  (starPart, "x_centered"),(starPart, "y_centered"),(starPart, "sfr_den_low_res"), weight_field=None, deposition="cic", fontsize=fsize, x_bins=nMockBins, y_bins=nMockBins)
        fig1.set_zlim((starPart, "sfr_den_low_res"), zmin=zFieldLim2[0], zmax=zFieldLim2[1])
        fig1.set_xlim(-rLim,rLim)
        fig1.set_ylim(-rLim,rLim)

        # Calculate Gas den in cil bins
        fig2 = yt.ParticlePhasePlot(sp,  (gasPart, "x_centered"),(gasPart, "y_centered"),(gasPart, "den_low_res"), weight_field=None, deposition="cic", fontsize=fsize, x_bins=nMockBins, y_bins=nMockBins)
        fig2.set_zlim((gasPart, "den_low_res"), zmin=zFieldLim1[0], zmax=zFieldLim1[1])
        fig2.set_xlim(-rLim,rLim)
        fig2.set_ylim(-rLim,rLim)

        # Calculate KS with both binned results
        SFRbins = fig1.profile[starPart,"sfr_den_low_res"].reshape(1, nMockBins**2)[0]
        DENbins = fig2.profile[gasPart,"den_low_res"].reshape(1, nMockBins**2)[0]

        # Filter low surf density bins
        ind = np.where((np.array(SFRbins) > 1e-10)&(np.array(DENbins) > 1e-10))

        xKS = np.log10(np.array(DENbins[ind]))
        yKS = np.log10(np.array(SFRbins[ind]))

        uAx.scatter(xKS,yKS,alpha=0.1)

        # Drawing contours rather than scattering all the datapoints; see http://stackoverflow.com/questions/19390320/scatterplot-contours-in-matplotlib
        if len(xKS) > 10 and len(yKS) > 10:
            Gaussian_density_estimation_nbins = 20
            kernel = kde.gaussian_kde(np.vstack([xKS, yKS])) 
            xi, yi = np.mgrid[xKS.min():xKS.max():Gaussian_density_estimation_nbins*1j, yKS.min():yKS.max():Gaussian_density_estimation_nbins*1j]
            zi = np.reshape(kernel(np.vstack([xi.flatten(), yi.flatten()])), xi.shape)
            uAx.contour(xi, yi, zi, np.array([0.2]), linewidths=1.5, colors=cmapDef(k))    # 80% percentile contour
        else: log.logger.warning(f"  Insufficent data points (xKS {len(xKS)} and yKS {len(yKS)}). Skipping contour")
        
        setLegend(uAx,sims,idx)

    # Obs KS line from 2008 Bigiel
    t = np.arange(-2, 5, 0.01)
    uAx.plot(t, 1.37*t - 3.78, 'k--', linewidth = 2, alpha = 0.7)

    # Obs KS contour from 2008 Bigiel
    fBigiel = open("bilcontour.txt","r+")
    dataX = []
    dataY = []
    for line in fBigiel:
        data = np.asarray(line.split(", "),dtype=float)
        dataX.append(data[0]+ np.log10(1.36))
        dataY.append(data[1])
    
    uAx.fill(dataX,dataY,fill=True, color='b', alpha = 0.1, hatch='\\')
    
    uAx.set_xlabel("$\mathrm{log[Gas\ Surface\ Density\ (M_{\odot}/pc^2)]}$")
    uAx.set_ylabel("$\mathrm{log[Star\ Formation\ Rate\ Surface\ Density\ (M_{\odot}/yr/kpc^2)]}$")

    uAx.set_xlim(0,3)
    uAx.set_ylim(-4,1)
    
    uAx.set_title("Kennicutt–Schmidt relation with mock observations")
    uAx.grid()
    
    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plot the total SFR of a simulation over time
def plotSFmass(sims,idx,nBins=50,zLim = [0,0],verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,yLims=[5e6,5e9],xLims=0,splimit=100,animate=0):
    if message != 0: log.logger.info(f"\n{message}")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(8/10*figSize, 6/10*figSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)

    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)

    # Bin star ages into nBins and use that to estimate total mass
    for k,sn in enumerate(sims):
        log.logger.info(f"  Started {sims[k].name} with time {sn.snap[idx[k]].time:.1f}")
        # Maybe limit to rvir?
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen,(splimit,"kpc"))
        # Gets limits from current snapshot and earliest recorded star
        if zLim[1] == 0: zLim[1] = sn.snap[idx[k]].z
        if zLim[0] == 0: zLim[0] = float(1/min(np.array(sp["PartType4","StellarFormationTime"]))-1)
        tLim = [float(co.t_from_z(zLim[0]).to("Myr")),float(co.t_from_z(zLim[1]).to("Myr"))]

        dt = (tLim[1]-tLim[0])/nBins
        timeX = np.linspace(tLim[0],tLim[1],nBins+1)
        zX    = np.array([co.z_from_t(co.quan(time, "Myr")) for time in timeX])
        dZ = (zX[-1]-zX[0])/nBins

        starMass = [0]*nBins

        allStarMass = np.array(sp["PartType4","Masses"].to("Msun"))
        allStarScale = np.array(sp["PartType4","StellarFormationTime"])
        allStarZ     = 1/allStarScale - 1
        starMass, edges = np.histogram(allStarZ,bins=zX[::-1],weights=allStarMass)
        histX = edges[0:-1]+dZ
          
        for i in range(nBins):
            if i == 0: continue
            starMass[-i-1] += starMass[-i]
        
        uAx.plot(histX,starMass,".--")
        uAx.semilogy()

    if yLims != 0: uAx.set_ylim(yLims[0],yLims[1])
    if xLims != 0: uAx.set_xlim(xLims[0],xLims[1])
    
    uAx.set_ylabel("Stellar Mass From Present Stars ($\mathrm{M}_{\odot}$)")
    uAx.set_xlabel("z")
    uAx.set_title("Stellar Mass Over time")
    uAx.grid()
    
    setLegend(uAx,sims,idx)

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)

# Plot the Ms/M200 ratio over time
def plotMsMh(sims,idx,verbose=verboseLevel,plotSize=figSize,saveFig=saveAll,saveFigPath=0,showFig=showAll,message=0,yLims=[1e-5,0.25],xLims=0,animate=0):
    if message != 0: log.logger.info(f"\n{message}")
    # Setup plot figures and axes
    uFig = plt.figure(figsize=(8/10*figSize, 6/10*figSize))
    uAx  = plt.subplot2grid((4,1),(0,0),rowspan=4)

    # Idx setup (creates len(sims) lists, each one with the snap numbers for a sim)
    if not all(isinstance(lIdx, list) for lIdx in idx):
        idxArrAux = [None]*len(sims)
        for j in range(len(sims)): idxArrAux[j] = idx 
        idx = idxArrAux

    # At z 8,7,6,5,4
    z_fix   =[8,7,6,5,4]
    rvir_fix=[5.77,7.52,8.43,11.43,25.2]

    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)

    # Bin star ages into nBins and use that to estimate total mass
    for k,sn in enumerate(sims):
        log.logger.info(f"  Started {sims[k].name}")
 
        Mstar = [0]*len(idx[k])
        Mhalo = [0]*len(idx[k])
        Mrati = [0]*len(idx[k])
        zList = [0]*len(idx[k])

        # Loops over the snapshots in a sim
        for i in range(len(idx[k])):
            curSnap   = sims[k].snap[idx[k][i]]
            curYTSnap = sims[k].ytFull[idx[k][i]]
            index     = np.argmin(np.abs(np.array(z_fix)-curSnap.z))
            # Uses the mean rvir if z is sufficiently close
            if np.abs(z_fix[index]-curSnap.z) < 0.2:
                log.logger.info("  Using the mean rvir from AGORA data, z for this snapshot is sufficiently close (dif < 0.2)")
                curRvir   = rvir_fix[index]
            else:
                log.logger.info("  Using the rvir calculated from the snapshot")
                curRvir   = sims[k].snap[idx[k][i]].rvir
            log.logger.debug(f"  - Snapshot {idx[k][i]} with t = {curSnap.time:.1f} z = {curSnap.z:.2f}")
            log.logger.debug(f"      Mapped to z = {z_fix[index]:.2f} rvir = {curRvir:.2f}")

            # Get the stellar halo and halo cutoff and calculate the total mass at this redshift
            spGal = curYTSnap.sphere(curSnap.ytcen,(0.15*curRvir, "kpc"))
            spVir = curYTSnap.sphere(curSnap.ytcen,(curRvir, "kpc"))

            zList[i] = curSnap.z
            if starPart in sims[k].snap[idx[k][i]].pType:
                Mstar[i] = spGal[(starPart,"particle_mass")].in_units("Msun").sum()
            else:
                log.logger.warning("  Star particles not in this snapshot, setting to 0")
                Mstar[i] = 0

            Mhalo[i] = getData(spVir,"particle_mass", sims[k].snap[idx[k][i]].pType, units="Msun").sum()
            #Mhalo[i] = spVir[("all","particle_mass")].in_units("Msun").sum()
            log.logger.debug(f"      Stellar Mass = {Mstar[i]:.2E} | Halo Mass = {Mhalo[i]:.2E}")
            Mrati[i] = Mstar[i]/Mhalo[i]
        
        uAx.plot(zList,Mrati,".--")
        uAx.semilogy()

    if yLims != 0: uAx.set_ylim(yLims[0],yLims[1])
    if xLims != 0: uAx.set_xlim(xLims[0],xLims[1])
    
    uAx.set_ylabel("$M_{s}/M_{h}$")
    uAx.set_xlabel("z")
    uAx.set_title("Stellar-to-Halo Mass Ratio Over Time")
    uAx.grid()
    
    uAx.legend([sims[i].name for i in range(len(sims))])

    handleFig(uFig,[showFig,animate,saveFig],message,saveFigPath,verbose)
        
# Create and save a movie from a frame list
def makeMovie(frames, interval=50, verbose=verboseLevel, saveFigPath=0, message=0):
    if message != 0: log.logger.info(f"\n{message}")
    # Create an animation figure using the first frame
    fig_anim, ax_anim = plt.subplots()
    im = ax_anim.imshow(frames[0], animated=True)
    ax_anim.axis('off') 
    fig_anim.tight_layout()
    fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    # Create animation by setting frames
    def update_frame(i):
        im.set_array(frames[i])
        return [im]
    anime = ani.FuncAnimation(fig_anim, update_frame, frames=len(frames), interval=interval, blit=True)

    # Saves figure with message path as title
    fullPath = os.path.join(savePath,"placeholder.png")
    if message != 0:
        fullPath = os.path.join(savePath,message.replace(" ","_")+".gif")
    elif saveFigPath == 0:
        log.logger.warning("TITLE NOT SPECIFIED FOR THIS FIGURE, PLEASE SPECIFY A TITLE")

    if saveFigPath != 0: fullPath = os.path.join(saveFigPath,message.replace(" ","_")+".gif")
    log.logger.info(f"  Saving animation to {fullPath}")

    with rc_context({"mathtext.fontset": "stix"}):
        anime.save(fullPath,dpi=300)
    plt.close(fig_anim)
    return anime

# Binning postproccessing functions
def binFunctionCilBins(cil,bin):
    dr = 0.5*(cil[1]-cil[0])
    newBin = []
    for i in range(len(cil)):
        newBin.append(bin[i]/(np.pi * (((cil[i]+dr)*1e3)**2-((cil[i]-dr)*1e3)**2) ))
    return newBin

def binFunctionSphBins(cil,bin):
    dr = 0.5*(cil[1]-cil[0])
    newBin = []
    for i in range(len(cil)):
        newBin.append(bin[i]/(4/3 * np.pi * (((cil[i]+dr)*1e3)**3-((cil[i]-dr)*1e3)**3) ))
    return newBin

def binFunctionCilBinsSFR(cil,bin):
    dr = 0.5*(cil[1]-cil[0])
    bin = np.array(bin)/youngStarAge
    newBin = []
    for i in range(len(cil)):
        newBin.append(bin[i]/(np.pi * (((cil[i]+dr)*1e3)**2-((cil[i]-dr)*1e3)**2) ))
    return newBin

def makeZbinFun(rlimit):
    def binFunctionZBins(zData,bin,rLim=rlimit):
        dh = (zData[1]-zData[0])
        newBin = []
        for i in range(len(zData)):
            newBin.append(bin[i]/(4*dh*1e3*rLim*1e3))
        return newBin
    return binFunctionZBins

# Not sure I'll keep this
def binFunctionSphVol(cil,bin):
    binVal  = bin.x
    vol     = (4/3)*np.pi*(binVal[1:]**3-binVal[:-1]**3)
    return vol