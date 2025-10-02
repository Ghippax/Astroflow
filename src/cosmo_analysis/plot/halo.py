"""Halo analysis and plotting functions for cosmo_analysis.

This module contains functions for halo finding, catalog loading, and
visualizing halo properties like clump mass functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import yt
import os.path
from yt.frontends.halo_catalog.data_structures import HaloDataset

from .. import log
from ..config import get_config
from ..core.constants import verboseLevel
from .base import handleFig, setLegend


def findHalos(simArr, idxArr, partT, mainPath, haloMethod="fof", hopThresh=4e9, 
              fofLink=0.0012, hardLimits=True, overWrite=True, clumpLim=(1e6, 8e8), 
              verbose=None, config=None):
    """Find halos in simulations using FOF or HOP methods.
    
    Performs halo finding on simulations and saves/loads halo catalogs.
    Uses yt's halo analysis module to identify bound structures.
    
    Args:
        simArr: list of Simulation objects
        idxArr: list of snapshot indices
        partT: particle type for halo finding (e.g., 'PartType1' for DM)
        mainPath: base path for saving/loading halo catalogs
        haloMethod: halo finding method ('fof' or 'hop')
        hopThresh: threshold for HOP algorithm (density threshold)
        fofLink: linking length for FOF algorithm
        hardLimits: whether to apply mass limits to filter halos
        overWrite: whether to overwrite existing halo catalogs
        clumpLim: tuple of (min_mass, max_mass) in solar masses
        verbose: verbosity level (uses config if None)
        config: Config object (optional, will use global if not provided)
    
    Returns:
        tuple: (haloSims, haloFilt) where haloSims contains halo datasets
               and haloFilt contains boolean masks for mass filtering
    """
    if config is None:
        config = get_config()
    
    if verbose is None:
        verbose = config.get('analysis_options.verbose_level', 1)
    
    log.logger.info(f"\nInitiating halo finding for particle type: {partT} and Method: {haloMethod}")
    
    # Initialize halo arrays
    haloSims = [None] * len(simArr)
    haloFilt = [None] * len(simArr)
    temp = None
    
    for i in range(len(simArr)):
        # Load parameters and paths
        log.logger.info(f"  - Loading halos for {simArr[i].name}")
        snap = simArr[i].ytFull[idxArr[i]]
        haloDirSim = os.path.join("Halos", "Halo_" + haloMethod + "_" + partT + "_" + simArr[i].name.replace(" ", "_"))
        haloPath = os.path.join(mainPath, haloDirSim)
        haloDirPath = os.path.join(haloPath, snap.basename[:snap.basename.find(".")])
        haloFilePath = os.path.join(haloDirPath, snap.basename[:snap.basename.find(".")] + ".0.h5")

        # Do the halo finding if no halos detected
        if not os.path.exists(haloDirPath) or overWrite:
            # Explain what files are being modified or not
            if not os.path.exists(haloDirPath):
                log.logger.info(f"    No halos detected in {haloDirPath}")
            elif overWrite:
                log.logger.info(f"    Overwriting halos detected in {haloDirPath}")
            
            log.logger.info(f"    Initializing halo finding to be saved in {haloFilePath}")
            
            # Configure the halo catalog and halo finding method
            if haloMethod == "hop":
                hopConf = hopThresh
                if isinstance(hopThresh, list):
                    hopConf = hopThresh[i]
                hc = yt.extensions.astro_analysis.halo_analysis.HaloCatalog(
                    data_ds=snap, data_source=snap.all_data(), finder_method="hop",
                    output_dir=haloPath, finder_kwargs={"threshold": hopConf, "dm_only": False, "ptype": partT})
            elif haloMethod == "fof":
                fofConf = fofLink
                if isinstance(fofLink, list):
                    fofConf = fofLink[i]
                hc = yt.extensions.astro_analysis.halo_analysis.HaloCatalog(
                    data_ds=snap, data_source=snap.all_data(), finder_method="fof",
                    output_dir=haloPath, finder_kwargs={"link": fofConf, "dm_only": False, "ptype": partT})
            
            # Calculate the actual halo
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
        log.logger.info(f"    Loading halo from file {haloFilePath}")
        halo_ds = yt.load(haloFilePath)
        hc = yt.extensions.astro_analysis.halo_analysis.HaloCatalog(halos_ds=halo_ds)
        hc.load()

        haloSims[i] = hc.halos_ds
        temp = haloSims[i].all_data()
        haloFilt[i] = np.ones(len(temp['particle_mass'].in_units("Msun")), dtype=bool)

        if hardLimits:
            # Get the masses in Msun.
            mass = temp['particle_mass'][:].in_units("Msun")
            # Create a boolean mask for halos within the desired mass limits.
            keep = (mass >= clumpLim[0]) & (mass <= clumpLim[1])
            # Find the indices of the halos to keep.
            haloFilt[i] = np.where(keep)[0]
        
    log.logger.debug("  Halo loading successful!")
    return (haloSims, haloFilt)


def plotClumpMassF(sims, idx, haloData, nBins=20, mLim=(6, 8.5), verbose=None, 
                   plotSize=None, saveFig=None, saveFigPath=None, showFig=None, 
                   message=None, animate=0, config=None):
    """Plot the cumulative mass function of halos/clumps.
    
    Creates a plot showing the number of clumps above a given mass,
    useful for comparing halo mass distributions between simulations.
    
    Args:
        sims: list of Simulation objects
        idx: list of snapshot indices
        haloData: tuple of (haloSims, haloFilt) from findHalos
        nBins: number of bins for mass histogram
        mLim: tuple of (min_log_mass, max_log_mass) in log10(Msun)
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
    
    haloSims, haloFilt = haloData
    
    for k in range(len(sims)):
        log.logger.info(f"  Started {sims[k].name}")
        temp = haloSims[k].all_data()
        
        # Get masses
        if isinstance(haloFilt[k], np.ndarray) and haloFilt[k].dtype == bool:
            mass = temp['particle_mass'][haloFilt[k]].in_units("Msun").d
        else:
            mass = temp['particle_mass'][haloFilt[k]].in_units("Msun").d
        
        # Calculate cumulative distribution
        bins = np.logspace(mLim[0], mLim[1], nBins)
        cumulative_counts = np.array([np.sum(mass >= m) for m in bins])
        
        uAx.plot(bins, cumulative_counts, ".--", label=sims[k].name)
    
    uAx.set_xscale('log')
    uAx.set_yscale('log')
    uAx.set_xlabel(r"$\mathrm{log[Newly\ Formed\ Stellar\ Clump\ Mass\ (M_{\odot})]}$")
    uAx.set_ylabel(r"$\mathrm{Cumulative Stellar\ Clump\ Counts, \ \ N_{clump}(> M)}$")
    uAx.grid()
    
    setLegend(uAx, sims, idx)
    
    return handleFig(uFig, [showFig, animate, saveFig], message, saveFigPath, verbose, config)
