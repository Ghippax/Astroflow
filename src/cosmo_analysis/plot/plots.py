"""Plotting functions for cosmo_analysis.

This module provides plotting functions for cosmological simulations.
It re-exports functions from specialized modules for backward compatibility
while providing a modular structure.

NOTE: This file maintains backward compatibility. New code should import
from the specialized modules (base, projection, phase, etc.) directly.
"""

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

# Import from new modular structure
from .base import saveFrame, setLegend, handleFig
from .projection import ytMultiPanel, ytProjPanel
from .phase import ytPhasePanel
from .profiles import plotBinned, plotRotDisp
from .utils import (makeMovie, binFunctionCilBins, binFunctionSphBins, 
                   binFunctionCilBinsSFR, makeZbinFun, binFunctionSphVol, aFromT)

yt.set_log_level(0)

# Legacy global - kept for backward compatibility, use config instead
savePath = "/sqfs/work/hp240141/z6b616/analysis"

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