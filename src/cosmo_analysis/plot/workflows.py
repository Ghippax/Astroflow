from ..plot.plots import *
from ..core.constants import *
from .. import log

# Define all the possible analysis
def NSFFanalysis(simsNS,idxNS,saveFigPath,extraText=""):
    widthIsoAnalysis = 30 # Corresponds to two times the approximate disk size
    ### Bins
    # Surface Density Binned
    plotBinned(simsNS,idxNS,("particle_position_cylindrical_radius","Masses"),50,(1e-3,widthIsoAnalysis/2-1),binFunction=binFunctionCilBins,
            setUnits=("kpc","Msun"),xlims=(0,widthIsoAnalysis/2-1),ylims=(1e-1,2*1e3),plotTitle="Cylindrically binned surface density",saveFigPath=saveFigPath,
            xylabels=("Cylidrincal radius (Kpc)","Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"),message="Surface Density"+extraText)
    # Vertical Surface Density
    plotBinned(simsNS,idxNS,("z_abs","Masses"),10,(1e-3,1.4),spLim=widthIsoAnalysis/2,message="Vertical Surface Density"+extraText,saveFigPath=saveFigPath,
            binFunction=makeZbinFun(widthIsoAnalysis/2),setUnits=("kpc","Msun"),xlims=(1e-3,1.4),ylims=(1e-1,3e3),plotTitle="Vertically binned surface density",
            xylabels=("Vertical height (Kpc)","Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"))
    # Average Height
    plotBinned(simsNS,idxNS,("particle_position_cylindrical_radius","z_abs"),50,(0,widthIsoAnalysis/2-1),message="Average Height"+extraText,spLim=widthIsoAnalysis/2,
            wField=(gasPart,"Masses"),setLogs=(False,False),setUnits=("kpc","kpc"),xlims=(0,widthIsoAnalysis/2-1),ylims=(0,0.45),saveFigPath=saveFigPath,
            plotTitle="Average cylindrical vertical height",xylabels=("Cylidrincal radius (Kpc)","Average vertical height (Kpc)"))
    # Velocity Profile
    plotBinned(simsNS,idxNS,("particle_position_cylindrical_radius","particle_velocity_cylindrical_theta"),50,(0,widthIsoAnalysis/2-1),saveFigPath=saveFigPath,
            message="Velocity Profile"+extraText,spLim=widthIsoAnalysis,wField=(gasPart,"Masses"),setLogs=(False,False),setUnits=("kpc","km/s"),
            xlims=(0,widthIsoAnalysis/2-1),ylims=(0,250),plotTitle="Velocity profile",xylabels=("Cylidrincal radius (Kpc)","Rotational velocity (km/s)"))
    # Velocity Dispersion 
    plotRotDisp(simsNS,idxNS,50,widthIsoAnalysis/2,part=gasPart,titlePlot="Velocity dispersion profile",message="Velocity Dispersion"+extraText,saveFigPath=saveFigPath)
    # Gas Density PDF
    plotBinned(simsNS,idxNS,("Density","Masses"),50,(1e-29,1e-21),message="Gas Density PDF"+extraText,spLim=widthIsoAnalysis/2,setLogs=(True,True),
            setUnits=("g/cm**3","Msun"),xlims=(1e-28,1e-21),ylims=(1e4,1e9),plotTitle="Gas Density PDF",saveFigPath=saveFigPath,
            xylabels=("$\mathrm{Density\ (g/cm^3)}$","$\mathrm{Mass,}\/\mathrm{d}M\mathrm{/dlog}\/\mathrm{\\rho}\/\mathrm{(M_{\odot})}$"))
    # Temp Density PDF
    plotBinned(simsNS,idxNS,("TemperatureG3log","Masses"),50,(1e1,1e7),message="Gas Temperature PDF"+extraText,spLim=widthIsoAnalysis/2,setLogs=(True,True),
            setUnits=("K","Msun"),xlims=(1e1,1e7),ylims=0,plotTitle="Gas Temperature PDF",saveFigPath=saveFigPath,
            xylabels=("$\mathrm{Temperature\ (K)}$","$\mathrm{Mass,}\/\mathrm{d}M\mathrm{/dlog}\/\mathrm{\\rho}\/\mathrm{(M_{\odot})}$"))
    
    # The following plots are only generated if SmoothingLength is available in all simulations
    all_have_smoothing_length = all((gasPart, "SmoothingLength") in sim.ytFull[idxNS[i]].field_list for i, sim in enumerate(simsNS))

    if all_have_smoothing_length:
        # Cylindrically binned Smoothing Length
        plotBinned(simsNS,idxNS,("particle_position_cylindrical_radius","SmoothingLength"),50,(0,widthIsoAnalysis/2-1),saveFigPath=saveFigPath,
                wField=(gasPart,"Masses"),setLogs=(False,True),setUnits=("kpc","pc"),xlims=(0,widthIsoAnalysis/2-1), message="Cylindrically binned Smoothing Length"+extraText,
                ylims=0,plotTitle="Mass weighted Smoothing Length",xylabels=("Cylidrincal radius (Kpc)","Smoothing length (pc)"))
        # Density binned Smoothing Length
        plotBinned(simsNS,idxNS,("Density","SmoothingLength"),50,(1e-29,1e-20),saveFigPath=saveFigPath,
                wField=(gasPart,"ones"),setLogs=(True,True),setUnits=("g/cm**3","pc"),xlims=0, message="Density Binned Smoothing Length"+extraText,
                ylims=0,plotTitle="Mass weighted Smoothing Length in Density Bins",xylabels=("$\mathrm{Density\ (g/cm^3)}$","Smoothing length (pc)"))
        # Smoothing Length binned Mass
        plotBinned(simsNS,idxNS,("SmoothingLength","Masses"),50,(1e1,1e5),saveFigPath=saveFigPath,
                wField=(gasPart,"Masses"),setLogs=(True,False),setUnits=("pc","Msun"),xlims=0, message="Smoothing Length binned Mass"+extraText,
                ylims=0,plotTitle="Smoothing Length PDF",xylabels=("Smoothing length (pc)","Mass (Msun)"))
    else:
        log.logger.info("Skipping SmoothingLength plots because the field is not available in all simulations.")

    ### Phase spaces
    # Gas Phase
    ytPhasePanel(simsNS,idxNS,blackLine=1,zFields=["Density", "TemperatureG3log", "Masses"],zFieldLim = (1e3,1e8,1e-29,1e-21,10,1e7),
                message="Gas Phase"+extraText,saveFigPath=saveFigPath,zWidth=widthIsoAnalysis/2)
    if all_have_smoothing_length:
        # Smoothing Length Phase
        ytPhasePanel(simsNS,idxNS,zLog=1,zFields = ["Density","TemperatureG3log","SmoothingLength"],wField=0, message="Smoothing Length Phase"+extraText,
                    zFieldUnits = ["g/cm**3","K","pc"],zFieldLim = (1e1,1e5,1e-29,1e-21,10,1e7),saveFigPath=saveFigPath,zWidth=widthIsoAnalysis/2)
    # Radius-Temp Phase
    ytPhasePanel(simsNS,idxNS,blackLine=1,zFields=["particle_position_cylindrical_radius", "TemperatureG3log", "Masses"],saveFigPath=saveFigPath,
                zFieldLim = (1e3,1e8,1e-3,1e2,10,1e7),message="Gas Radius Temp Phase"+extraText,zFieldUnits = ["kpc","K","Msun"],zWidth=widthIsoAnalysis/2)
    # Radius-Density Phase
    ytPhasePanel(simsNS,idxNS,blackLine=1,zFields=["particle_position_cylindrical_radius", "Density", "Masses"],saveFigPath=saveFigPath,
                zFieldLim = (1e3,1e8,1e-3,1e2,1e-28,1e-18),message="Gas Radius Density Phase"+extraText,zFieldUnits = ["kpc","g/cm**3","Msun"],zWidth=widthIsoAnalysis/2)

    ### Projections
    # Density projection
    ytProjPanel(simsNS,idxNS,twoAxis=True,message="Density Proj"+extraText,zFieldLim=(0.00001, 0.1),zWidth=widthIsoAnalysis,saveFigPath=saveFigPath)
    # Temperature Proj
    ytProjPanel(simsNS,idxNS,zField="TemperatureG3log",twoAxis=True,wField="density_squared",zWidth=widthIsoAnalysis,zFieldUnit="K",
                zFieldLim=(1e1,1e6),message="Temperature Proj"+extraText,saveFigPath=saveFigPath)
    # Elevation Map
    ytProjPanel(simsNS,idxNS,twoAxis=True,zField="elevation",wField="density",zFieldUnit="kpc",zWidth=widthIsoAnalysis,zFieldLim=(-1,1),takeLog=0,
                message="Elevation Map"+extraText,saveFigPath=saveFigPath)
    # Resolution Map
    ytProjPanel(simsNS,idxNS,twoAxis=True,zField="resolution",wField="inv_volume_sq",zFieldUnit="pc",zWidth=widthIsoAnalysis,zFieldLim=(10,1e3),takeLog=1,
                message="Resolution Map"+extraText,saveFigPath=saveFigPath)
    
def SFFanalysis(sims2,idx2,saveFigPath,extraText=""):
    widthIsoAnalysis = 30 # Corresponds to two times the approximate disk size
    ### Bins
    # Clump Mass Cumulative
    #sffHalos = findHalos(sims2,idx2,starPart,hopThresh=4e9,overWrite=False)
    #plotClumpMassF(sims2,idx2,sffHalos,saveFigPath=saveFigPath,message="Clump Mass Cumulative")
    # Star Surface Density
    plotBinned(sims2,idx2,("particle_position_cylindrical_radius","Masses"),50,(0,widthIsoAnalysis/2-1),message="Star Surface Density",part=starPart,saveFigPath=saveFigPath,
            binFunction=binFunctionCilBins,setUnits=("kpc","Msun"),xlims=(0,widthIsoAnalysis/2-1),ylims=0,plotTitle="Cylindrically binned stellar surface density",
            xylabels=("Cylidrincal radius (Kpc)","Newly Formed Stars Surface density ($\\frac{\mathrm{M}_{\odot}}{pc^2}$)"))
    # Star Velocity Profile
    plotBinned(sims2,idx2,("particle_position_cylindrical_radius","particle_velocity_cylindrical_theta"),50,(0,widthIsoAnalysis/2-1),message="Star Velocity Profile",
            part=starPart,spLim=widthIsoAnalysis/2,wField=(starPart,"Masses"),setLogs=(False,False),setUnits=("kpc","km/s"),xlims=(0,widthIsoAnalysis/2-1),ylims=0,saveFigPath=saveFigPath,
            plotTitle="Newly Formed Stars Velocity Profile",xylabels=("Cylidrincal radius (Kpc)","Rotational velocity (km/s)"))
    # Star Velocity Dispersion
    plotRotDisp(sims2,idx2,50,widthIsoAnalysis/2,ylims=0,part=starPart,titlePlot="Newly Formed Stars Velocity Dispersion Profile",saveFigPath=saveFigPath,message="Star Velocity Dispersion")
    # Surface Density
    plotBinned(sims2,idx2,("particle_position_cylindrical_radius","particle_mass_young_stars"),50,(1e-3,widthIsoAnalysis/2),message="Surface Density",saveFigPath=saveFigPath,
            part=starPart,binFunction=binFunctionCilBinsSFR,setUnits=("kpc","Msun"),xlims=(0,widthIsoAnalysis/2-1),ylims=0,plotTitle="Surface Density",
            xylabels=("Cylidrincal radius (Kpc)","$\mathrm{Star\ Formation\ Rate\ Surface\ Density\ (M_{\odot}/yr/kpc^2)}$"))
    # Total SFR
    plotSFR(sims2,idx2,saveFigPath=saveFigPath,message="Total SFR",yLims=0)
    # KS Cil Binned
    plotKScil(sims2,idx2,saveFigPath=saveFigPath,message="KS Cil Binned",rLim=widthIsoAnalysis/2)
    # KS Mock Obs
    plotKSmock(sims2,idx2,saveFigPath=saveFigPath,message="KS Mock Obs")

    ### Phase spaces
    # Metal Gas Phase
    ytPhasePanel(sims2,idx2,blackLine=0,zLog=0,zFields = ["Density","TemperatureG3log","Metallicity"],wField="Masses",saveFigPath=saveFigPath,
                zFieldUnits = ["g/cm**3","K","1"],zFieldLim = (1e-2,4e-2,1e-29,1e-21,10,1e7),message="Metal Gas Phase",zWidth=widthIsoAnalysis/2)
    # Gas Obs
    nMockBins = int(widthIsoAnalysis*1e3/lowResMock)
    ytPhasePanel(sims2,idx2, cM=mColorMap2,zFields = ["x_centered","y_centered","den_low_res"], grid=False,wField=0,saveFigPath=saveFigPath,
                zFieldUnits=0, zFieldLim = (1e0,1e3,-widthIsoAnalysis/2,widthIsoAnalysis/2,-widthIsoAnalysis/2,widthIsoAnalysis/2),xb=nMockBins,yb=nMockBins,
                message="Gas Obs", depositionAlg="cic")
    # SFR Obs
    ytPhasePanel(sims2,idx2, cM=mColorMap2,part=starPart,zFields = ["x_centered","y_centered","sfr_den_low_res"], grid=False,wField=0,saveFigPath=saveFigPath,
                zFieldUnits=0, zFieldLim = (3e-4,3e-1,-widthIsoAnalysis/2,widthIsoAnalysis/2,-widthIsoAnalysis/2,widthIsoAnalysis/2),xb=nMockBins,yb=nMockBins,
                message="SFR Obs", depositionAlg="cic")

    ### Projections
    # Star Density Proj
    ytProjPanel(sims2,idx2,bSize=400,part="PartType4",zField="particle_mass",zFieldUnit="Msun",zFieldLim=0,saveFigPath=saveFigPath, #ovHalo=sffHalos,
                message="Star Density Proj",zWidth=widthIsoAnalysis)
    # Metallicity Proj
    ytProjPanel(sims2,idx2,part="PartType0",zField="Metallicity",wField="density_squared",zFieldUnit="1",takeLog=0,zFieldLim=0,saveFigPath=saveFigPath,
                cM=mColorMap,message="Metallicity Proj",zWidth=widthIsoAnalysis)