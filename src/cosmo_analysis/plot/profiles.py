"""Profile plotting functions for cosmo_analysis.

This module contains functions for creating radial profiles and rotation curves,
showing how physical quantities vary with distance from the center.
"""

import numpy as np
import matplotlib.pyplot as plt
import yt

from .. import log
from ..config import get_config
from .base import handleFig, setLegend


def plotBinned(sims, idx, binFields, nBins, rLim, logOverload=0, legOverload=0, diffSims=0, 
               blLine=0, wField=0, spLim=0, binFunction=0, part=None, setUnits=0, 
               setLogs=(False, True), ylims=0, xlims=0, animate=0, xylabels=0, plotTitle=0, 
               errorLim=None, message=None, verbose=None, plotSize=None, saveFig=None, 
               saveFigPath=None, showFig=None, showError=None, axAspect=1, config=None):
    """Plot binned field profiles for multiple simulations.
    
    Creates radial profiles showing how a field varies with radius,
    with optional error analysis and comparison between simulations.
    
    Args:
        sims: list of Simulation objects
        idx: list of snapshot indices
        binFields: list of [radius_field, value_field] to bin
        nBins: number of bins for binning
        rLim: tuple of (min_radius, max_radius) in kpc
        logOverload: override log scaling for plots
        legOverload: override legend entries
        diffSims: tuple of (sims_list, idx_list) to subtract from main sims
        blLine: x value for vertical reference line
        wField: weight field for binning
        spLim: override sphere limit
        binFunction: post-processing function for bins
        part: particle type (uses config if None)
        setUnits: tuple of units for (x_field, y_field)
        setLogs: tuple of (x_log, y_log) boolean flags
        ylims: tuple of (ymin, ymax) for plot limits
        xlims: tuple of (xmin, xmax) for plot limits
        animate: whether to return frame for animation
        xylabels: tuple of (xlabel, ylabel)
        plotTitle: title for the plot
        errorLim: tuple of (error_min, error_max) for residual plot
        message: title/filename for saved figure
        verbose: verbosity level (uses config if None)
        plotSize: figure size (uses config if None)
        saveFig: whether to save figure (uses config if None)
        saveFigPath: path to save figure (uses config if None)
        showFig: whether to show figure (uses config if None)
        showError: error plotting mode (0=none, 1=residuals, 2=raw values)
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
        plotSize = config.get('plotting_defaults.figsize', [8, 8])
    if saveFig is None:
        saveFig = config.get('plotting_defaults.save_plots', True)
    if showFig is None:
        showFig = config.get('plotting_defaults.show_plots', False)
    if showError is None:
        showError = config.get('analysis_options.show_error', 0)
    if errorLim is None:
        errorLim = config.get('analysis_options.error_lim', [-1, 1])
    
    if message:
        log.logger.info(f"\n{message}")
    
    # Initialize figures
    if showError == 1:
        if not isinstance(plotSize, list):
            plotSize = [plotSize, plotSize*1.2]
        uFig = plt.figure(figsize=(plotSize[0], plotSize[1]))
        uAx = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        uAx2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    elif showError == 2:
        if not isinstance(plotSize, list):
            plotSize = [plotSize, plotSize*1.5]
        uFig = plt.figure(figsize=(plotSize[0], plotSize[1]))
        uAx = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        uAx2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
    else:
        if not isinstance(plotSize, list):
            plotSize = [plotSize, plotSize]
        uFig = plt.figure(figsize=(plotSize[0], plotSize[1]))
        uAx = plt.subplot2grid((4, 1), (0, 0), rowspan=4)
 
    allYbin = [None] * len(sims)
    XGlobal = None
    
    for k, sn in enumerate(sims):
        # Setting up parameters
        log.logger.info(f"  Started {sn.name}")
        splimit = rLim[1]
        if spLim != 0:
            splimit = spLim
        weightField = None
        if wField != 0:
            weightField = wField

        # Actual binning process
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen, (splimit, "kpc"))
        p1 = yt.ProfilePlot(sp, (part, binFields[0]), (part, binFields[1]), 
                           weight_field=weightField, n_bins=nBins, x_log=setLogs[0], accumulation=False)
        p1.set_log((part, binFields[0]), setLogs[0])
        p1.set_log((part, binFields[1]), setLogs[1])
        
        if setUnits[0] != 0:
            p1.set_unit((part, binFields[0]), setUnits[0])
        if setUnits[1] != 0:
            p1.set_unit((part, binFields[1]), setUnits[1])
        p1.set_xlim(rLim[0], rLim[1])

        # Extract bins to perform further operations
        if setUnits[0] != 0:
            xData = p1.profiles[0].x.in_units(setUnits[0]).d
        else:
            xData = p1.profiles[0].x.d
        
        if setUnits[1] != 0:
            bin_data = p1.profiles[0][binFields[1]].in_units(setUnits[1]).d
        else:
            bin_data = p1.profiles[0][binFields[1]].d

        # If there is a post-processing function, do that
        if binFunction != 0:
            bin_data = binFunction(xData, bin_data)

        # Performs the difference between two bins of two simulation sets
        if diffSims != 0:
            sp2 = diffSims[0][k].ytFull[diffSims[1][k]].sphere(
                diffSims[0][k].snap[diffSims[1][k]].ytcen, (splimit, "kpc"))
            p2 = yt.ProfilePlot(sp2, (part, binFields[0]), (part, binFields[1]), 
                               weight_field=weightField, n_bins=nBins, x_log=setLogs[0], accumulation=False)
            p2.set_log((part, binFields[0]), setLogs[0])
            p2.set_log((part, binFields[1]), setLogs[1])
            
            if setUnits != 0:
                p2.set_unit((part, binFields[0]), setUnits[0])
                p2.set_unit((part, binFields[1]), setUnits[1])
            p2.set_xlim(rLim[0], rLim[1])

            xData2 = p2.profiles[0].x.in_units(setUnits[0]).d
            bin2 = p2.profiles[0][binFields[1]].in_units(setUnits[1]).d
            
            if binFunction != 0:
                bin2 = binFunction(xData2, bin2)
            bin_data = np.array(bin_data) - np.array(bin2)

        allYbin[k] = bin_data
        XGlobal = xData
        uAx.plot(xData, bin_data, ".--")

        # Setup log scaling depending on options
        setLogsPlot = setLogs
        if logOverload != 0:
            setLogsPlot = logOverload
        if setLogsPlot[0]:
            uAx.semilogx()
        if setLogsPlot[1]:
            uAx.semilogy()
        uAx.set_box_aspect(axAspect)
        
    # Plot dispersion of codes
    if showError == 1:
        log.logger.debug("  Plotting dispersion")
        allYbin = np.array(allYbin)
        error = [None] * nBins
        for i in range(nBins):
            average = np.mean(allYbin[:, i])
            if average != 0:
                error[i] = (allYbin[:, i] - average) / average
            else:
                error[i] = np.zeros_like(allYbin[:, i])
        error = np.array(error)
        
        for i in range(len(sims)):
            uAx2.plot(XGlobal, error[:, i], ".")
            if setLogs[0]:
                uAx2.semilogx()
            
        if xlims != 0:
            uAx2.set_xlim(xlims[0], xlims[1])
        uAx2.set_ylim(errorLim[0], errorLim[1])
        uAx2.grid()

        if xylabels != 0:
            uAx2.set_ylabel(r"Residual ($\frac{\sigma - \overline{\sigma}}{\overline{\sigma}}$)")
            uAx2.set_xlabel(xylabels[0])
            
    elif showError == 2:
        allYbin = np.array(allYbin)
        for i in range(len(sims)):
            uAx2.plot(XGlobal, allYbin[i, :] / 1e8, ".")

        if xlims != 0:
            uAx2.set_xlim(xlims[0], xlims[1])
        uAx2.set_ylim(errorLim[0], errorLim[1])
        uAx2.grid()
        uAx2.set_xscale('log')
        uAx2.set_yscale('symlog', linthresh=0.01)
        
        if xylabels != 0:
            uAx2.set_ylabel("Log " + xylabels[1])
            uAx2.set_xlabel(xylabels[0])

        if blLine != 0:
            uAx2.axvline(x=blLine, color='k', linestyle='--', linewidth=2, alpha=0.7)
    else:
        if xylabels != 0:
            uAx.set_xlabel(xylabels[0])
    
    # Set limits and labels
    if xylabels != 0:
        uAx.set_ylabel(xylabels[1])
    if plotTitle != 0:
        uAx.set_title(plotTitle)
    if xlims != 0:
        uAx.set_xlim(xlims[0], xlims[1])
    if ylims != 0:
        uAx.set_ylim(ylims[0], ylims[1])
    
    # Threshold line
    if blLine != 0:
        uAx.axvline(x=blLine, color='k', linestyle='--', linewidth=2, alpha=0.7)

    uAx.grid()
    
    if legOverload == 0:
        setLegend(uAx, sims, idx)
    else:
        uAx.legend(legOverload)
    
    return handleFig(uFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)


def plotRotDisp(sims, idx, nBins, rLim, part, titlePlot=0, verbose=None, plotSize=None, 
                saveFig=None, saveFigPath=None, showFig=None, message=None, ylims=(0, 170), 
                animate=0, config=None):
    """Calculate and plot velocity dispersion for a given particle type.
    
    Creates a multi-panel plot showing rotation curve, velocity dispersion,
    residuals, and vertical dispersion ratio.
    
    Args:
        sims: list of Simulation objects
        idx: list of snapshot indices
        nBins: number of bins for radial binning
        rLim: maximum radius in kpc
        part: particle type to analyze
        titlePlot: title for the plot
        verbose: verbosity level (uses config if None)
        plotSize: figure size (uses config if None)
        saveFig: whether to save figure (uses config if None)
        saveFigPath: path to save figure (uses config if None)
        showFig: whether to show figure (uses config if None)
        message: title/filename for saved figure
        ylims: tuple of (ymin, ymax) for velocity dispersion plot
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
    
    if message:
        log.logger.info(f"\n{message}")
        
    uFig = plt.figure(figsize=(plotSize, plotSize*1.5))
    uAx = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
    uAx2 = plt.subplot2grid((5, 1), (3, 0), rowspan=1)
    uAx3 = plt.subplot2grid((5, 1), (4, 0), rowspan=1)
	
    snapArr = [sims[i].ytFull[idx[i]] for i in range(len(sims))]
    centArr = [sims[i].snap[idx[i]].ytcen for i in range(len(sims))]
	
    # Define new fields for velocity dispersion calculation
    for i, snap in enumerate(snapArr):
        # Get the velocity bins
        sp = snap.sphere(centArr[i], (rLim, "kpc"))
        rotProf = yt.ProfilePlot(sp, (part, "particle_position_cylindrical_radius"),
                                (part, "particle_velocity_cylindrical_theta"),
                                weight_field=(part, "Masses"), n_bins=nBins, x_log=False)
        rotProf.set_log((part, "particle_position_cylindrical_radius"), False)
        rotProf.set_log((part, "particle_velocity_cylindrical_theta"), False)
        rotProf.set_unit((part, "particle_velocity_cylindrical_theta"), 'km/s')
        rotProf.set_unit((part, "particle_position_cylindrical_radius"), 'kpc')
        rotProf.set_xlim(0, rLim-1)
        rotProf.set_ylim((part, "particle_velocity_cylindrical_theta"), 0, 250)
        rotCilLocal = rotProf.profiles[0].x.in_units('kpc').d
        rotBinLocal = rotProf.profiles[0]["particle_velocity_cylindrical_theta"].in_units('km/s').d
	
        # Define field functions with closure over rotation data
        def make_field_func(rotCil, rotBin):
            def _particle_rot_vx(field, data):
                trans = np.zeros(data[(part, "particle_velocity_x")].shape)
                dr = 0.5 * (rotCil[1] - rotCil[0])
                for rad, vrot in zip(rotCil, rotBin):
                    ind = np.where((data[(part, "particle_position_cylindrical_radius")].in_units("kpc") >= (rad - dr)) & 
                                  (data[(part, "particle_position_cylindrical_radius")].in_units("kpc") < (rad + dr)))
                    trans[ind] = -np.sin(data[(part, "particle_position_cylindrical_theta")][ind]) * vrot * 1e5
                return data.ds.arr(trans, "cm/s").in_base(data.ds.unit_system.name)
            
            def _particle_rot_vy(field, data):
                trans = np.zeros(data[(part, "particle_velocity_y")].shape)
                dr = 0.5 * (rotCil[1] - rotCil[0])
                for rad, vrot in zip(rotCil, rotBin):
                    ind = np.where((data[(part, "particle_position_cylindrical_radius")].in_units("kpc") >= (rad - dr)) & 
                                  (data[(part, "particle_position_cylindrical_radius")].in_units("kpc") < (rad + dr)))
                    trans[ind] = np.cos(data[(part, "particle_position_cylindrical_theta")][ind]) * vrot * 1e5
                return data.ds.arr(trans, "cm/s").in_base(data.ds.unit_system.name)
            
            return _particle_rot_vx, _particle_rot_vy

        vx_func, vy_func = make_field_func(rotCilLocal, rotBinLocal)
		
        snap.add_field((part, "particle_rot_vx"), function=vx_func, take_log=False,
                      units="cm/s", sampling_type="particle", force_override=True)
        snap.add_field((part, "particle_rot_vy"), function=vy_func, take_log=False,
                      units="cm/s", sampling_type="particle", force_override=True)
		
        # Velocity dispersion field
        def _particle_vel_disp(field, data):
            return ((data[(part, "particle_velocity_x")] - data[(part, "particle_rot_vx")])**2 + 
                   (data[(part, "particle_velocity_y")] - data[(part, "particle_rot_vy")])**2 + 
                   (data[(part, "particle_velocity_z")])**2)
    
        snap.add_field((part, "particle_vel_disp"), function=_particle_vel_disp, take_log=False,
                      units="cm**2/s**2", sampling_type="particle", force_override=True)
        
        def _particle_velocity_z_squared(field, data):
            return (data[(part, "particle_velocity_z")])**2
            
        snap.add_field((part, "particle_velocity_z_squared"), function=_particle_velocity_z_squared,
                      take_log=False, units="cm**2/s**2", sampling_type="particle", force_override=True)

    allYbin = [None] * len(sims)
    allYbinZ = [None] * len(sims)
    XGlobal = None

    for k, sn in enumerate(sims):
        log.logger.info(f"  Started {sn.name}")
        sp = sims[k].ytFull[idx[k]].sphere(sims[k].snap[idx[k]].ytcen, (rLim, "kpc"))
        
        p1 = yt.ProfilePlot(sp, (part, "particle_position_cylindrical_radius"),
                           (part, "particle_vel_disp"), weight_field=(part, "Masses"),
                           n_bins=nBins, x_log=False)
        p1.set_log((part, "particle_position_cylindrical_radius"), False)
        p1.set_unit((part, "particle_position_cylindrical_radius"), "kpc")
        p1.set_xlim(1e-3, rLim-1)
        
        cil = p1.profiles[0].x.in_units('kpc').d
        bins = np.sqrt(p1.profiles[0]["particle_vel_disp"]).in_units('km/s').d

        # Vertical z speed dispersion
        p2 = yt.ProfilePlot(sp, (part, "particle_position_cylindrical_radius"),
                           (part, "particle_velocity_z_squared"), weight_field=(part, "Masses"),
                           n_bins=nBins, x_log=False)
        p2.set_log((part, "particle_position_cylindrical_radius"), False)
        p2.set_unit((part, "particle_position_cylindrical_radius"), "kpc")
        p2.set_xlim(1e-3, rLim-1)
        allYbinZ[k] = np.sqrt(p2.profiles[0]["particle_velocity_z_squared"]).in_units('km/s').d
        
        allYbin[k] = bins
        XGlobal = cil
        uAx.plot(cil, bins, ".--")
    
    # Calculate and plot residuals
    allYbin = np.array(allYbin)
    error = [None] * nBins
    log.logger.debug("  Plotting dispersion")
    
    for i in range(nBins):
        average = np.mean(allYbin[:, i])
        if average != 0:
            error[i] = (allYbin[:, i] - average) / average
        else:
            error[i] = np.zeros_like(allYbin[:, i])
    error = np.array(error)
    
    for i in range(len(sims)):
        uAx2.plot(XGlobal, error[:, i], ".")
    uAx2.set_xlim(0, (rLim-1))
    uAx2.set_ylim(-1, 1)
    uAx2.set_ylabel(r"Residual ($\frac{\sigma - \overline{\sigma}}{\overline{\sigma}}$)")
    uAx2.grid()
	
    # Plot vertical dispersion ratio
    allYbinZ = np.array(allYbinZ)
    log.logger.debug("  Plotting dispersion ratio")
    
    for i in range(len(sims)):
        dispRatio = allYbinZ[i, :] / allYbin[i, :]
        uAx3.plot(XGlobal, dispRatio, ".")
    uAx3.set_xlim(0, (rLim-1))
    uAx3.set_ylim(0, 1)
    uAx3.set_ylabel(r"Vertical dispersion ratio ($\frac{\sigma_z}{\sigma}$)")
    uAx3.grid()
    uAx3.set_xlabel("Cylindrical radius (Kpc)")

    uAx.set_xlim(0, 14)
    if ylims != 0:
        uAx.set_ylim(ylims[0], ylims[1])
    
    uAx.set_ylabel("Velocity dispersion (km/s)")
    if titlePlot != 0:
        uAx.set_title(titlePlot)
    uAx.grid()

    setLegend(uAx, sims, idx)

    return handleFig(uFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)
