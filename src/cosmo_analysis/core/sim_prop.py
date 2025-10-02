"""Simulation property calculations.

This module provides functions for calculating physical properties of simulations,
including various centering algorithms, virial radius calculations, and other
snapshot-level properties.
"""

from .. import log
from ..core import utils
from ..io   import load
import numpy as np
import unyt
import yt

# Calculate the actual center of a simulation snapshot
def findCenter(sim, snapshotN, lim=20):
    """Calculate center by finding maximum density particle.
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
        lim: Radius limit in kpc for initial sphere (default: 20)
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    snap   = sim.ytFull[snapshotN]
    cutOff = snap.sphere("center",(lim,"kpc"))
    den    = np.array(cutOff["PartType0", "Density"].to("Msun/pc**3"))
    x      = np.array(cutOff["PartType0", "x"].to("pc"))
    y      = np.array(cutOff["PartType0", "y"].to("pc"))
    z      = np.array(cutOff["PartType0", "z"].to("pc"))
    cenIdx = utils.maxIdx(den)
    cen    = np.array([x[cenIdx],y[cenIdx],z[cenIdx]])
    return (cen,cen/1e3*unyt.kpc)

# Centering calc2 (center of mass)
def findCenter2(sim, snapshotN, lim=20):
    """Calculate center using gas center of mass.
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
        lim: Radius limit in kpc for initial sphere (default: 20)
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    snap   = sim.ytFull[snapshotN]
    cutOff = snap.sphere("center",(lim,"kpc"))
    cen    = cutOff.quantities.center_of_mass(use_gas=True, use_particles=False).in_units("pc")
    return (cen.d,cen.d/1e3*unyt.kpc)

# Centering calc3 (like AGORA Paper)
def findCenter3(sim, snapshotN):
    """Calculate center using AGORA method for isolated simulations.
    
    Finds maximum density, refines with center of mass, then finds max density again.
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    snap      = sim.ytFull[snapshotN]

    v, cen1   = snap.find_max(("gas", "density"))
    bigCutOff = snap.sphere(cen1, (30.0, "kpc"))
    cen2      = bigCutOff.quantities.center_of_mass(use_gas=True, use_particles=False).in_units("kpc")
    cutOff    = snap.sphere(cen2,(1.0,"kpc"))
    cen       = cutOff.quantities.max_location(("gas", "density"))
    center    = np.array([cen[1].d,cen[2].d,cen[3].d])*1e3

    return (center,center/1e3*unyt.kpc)

# Centering calc4 (like AGORA Paper, for CosmoRun)
def findCenter4(sim, idx, projPath=None):
    """Calculate center using AGORA method for cosmological runs.
    
    Args:
        sim: Simulation object
        idx: Snapshot index
        projPath: Path to projection file. If None, uses config value.
    
    Returns:
        tuple: (center_array, center_unyt_array)
    """
    from ..config import get_config
    
    snap = sim.ytFull[idx]
    
    # Get projection path from config if not provided
    if projPath is None:
        config = get_config()
        projPath = config.get('paths.data_files.projection_list', 'outputlist_projection.txt')
    
    f        = np.loadtxt(projPath,skiprows=4)
    idx0     = utils.getClosestIdx(f[:,0],0.99999) # Finds end of first projection
    tIdx     = utils.getClosestIdx(f[0:idx0,1],sim.snap[idx].a)

    cen1     = np.array([f[tIdx,3],f[tIdx,4],f[tIdx,5]]) # In Mpc
    center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')
    sp       = snap.sphere(center, (2,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.5,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.25,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    center   = center.to("code_length")

    cen1     = np.array([center[0].d,center[1].d,center[2].d]) 

    return   (cen1,center)

# Centering calc5 (AGORA code shortcut, for isolated)
def findCenter5(sim, snapshotN):
    """Calculate center using hardcoded AGORA coordinates.
    
    Uses a specific center position from AGORA simulations.
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    snap   = sim.ytFull[snapshotN]
    cen    = snap.arr([6.184520935812296e+21, 4.972678132728082e+21, 6.559067311284074e+21], 'cm')
    cen    = cen.to("pc")
    cent   = np.array([cen[0].d,cen[1].d,cen[2].d])
    return (cen,cen/1e3*unyt.kpc)

# Centering calc6 (just 0 0 0)
def findCenter6(sim, snapshotN):
    """Calculate center at origin (0,0,0).
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    cen    = np.array([0,0,0])
    return (cen,cen/1e3*unyt.kpc)

# Centering calc7 (like 4, but expanded in scope)
def findCenter7(sim, idx, projPath=None):
    """Calculate center using expanded AGORA method.
    
    Args:
        sim: Simulation object
        idx: Snapshot index
        projPath: Path to projection file. If None, uses config value.
    
    Returns:
        tuple: (center_array, center_unyt_array)
    """
    from ..config import get_config
    
    snap = sim.ytFull[idx]
    
    # Get projection path from config if not provided
    if projPath is None:
        config = get_config()
        projPath = config.get('paths.data_files.projection_list', 'outputlist_projection.txt')
    
    log.logger.debug("Loading projection file from: %s", projPath)

    f        = np.loadtxt(projPath,skiprows=4)
    idx0     = utils.getClosestIdx(f[:,0],0.99999) # Finds end of first projection
    tIdx     = utils.getClosestIdx(f[0:idx0,1],sim.snap[idx].a)

    cen1     = np.array([f[tIdx,3],f[tIdx,4],f[tIdx,5]]) # In Mpc
    center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')
    sp       = snap.sphere(center, (40,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (10,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.5,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.25,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    center   = center.to("code_length")

    cen1     = np.array([center[0].d,center[1].d,center[2].d]) 

    return   (cen1,center)

# Centering calc8 (like 4, for any box)
def findCenter8(sim, idx):
    """Calculate center using iterative shrinking sphere method.
    
    Starts from box center and iteratively refines using particle center of mass
    with progressively smaller spheres.
    
    Args:
        sim: Simulation object
        idx: Snapshot index
    
    Returns:
        tuple: (center_array in code units, center_unyt_array in code_length)
    """
    snap     = sim.ytFull[idx]

    boxSize = snap.parameters['BoxSize']
    
    cen1     = np.array([boxSize/2,boxSize/2,boxSize/2]) # In code units
    center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')

    itNumber   = 8
    sizeSphere = np.logspace(np.log10(boxSize),np.log10(boxSize*0.0001),itNumber)

    for i in range(itNumber):
        sp = snap.sphere(center, (sizeSphere[i],'code_length'))
        center   = sp.quantities.center_of_mass(use_gas=False,use_particles=True)

    center   = center.to("code_length")
    cen1     = np.array([center[0].d,center[1].d,center[2].d])

    return   (cen1,center)

# Calculates the critical density based on cosmology and redshift
def getCritRho(co,z):
    return co.critical_density(z).to("Msun/kpc**3")
def getCritRho200(co,z):
    return getCritRho(co,z)*200
# Calculates the mean density based on cosmology and redshift
def getMeanRho(co,z):
    rho_crit0 = getCritRho(co,0)
    rho_mean0 = co.omega_matter * rho_crit0
    return rho_mean0 * (1 + z)**3
def getMeanRho200(co,z):
    return getMeanRho(co,z)*200
# Calculates the density at virial radius according to Paper IV (based on the factor by Bryan & Norman (1998))
def getVirRho(co,z):
    Hz = co.hubble_parameter(z)
    H0 = co.hubble_parameter(0.0)
    Ez = (Hz / H0)
    # Matter fraction at z
    Omega_z = (co.omega_matter * (1 + z)**3 / Ez**2)
    x = Omega_z - 1.0
    # Δc fits from the literature:
    #   • Flat‐curvature (ΩR = 0):   Δc = 18π² + 82x – 39x²
    #   • No‐Λ universe  (ΩΛ = 0):   Δc = 18π² + 60x – 32x²
    Delta_c     = 18*np.pi**2 + 82*x - 39*x**2
    #Delta_c     = 18*np.pi**2 + 60*x - 32*x**2
    return Delta_c*getCritRho(co,z)

# Calculates the virial radius at present redshift via multiple methods
def getRvir(sim,idx,method="Vir",rvirlim=500):
    log.logger.info(f"    Initiating virial radius finder for snapshot at {idx} using method {method} and limited by {rvirlim}")
    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)
    methodDict = {"Crit":getCritRho200,"Mean":getMeanRho200,"Vir":getVirRho}
    snapshot   = sim.ytFull[idx]
    targetDen  = float(methodDict[method](co,sim.snap[idx].z))
    sp         = snapshot.sphere(sim.snap[idx].ytcen,(500,"kpc"))
    allMass    = load.getData(sp, "particle_mass", sim.snap[idx].pType, units = "Msun")
    allR       = load.getData(sp, "particle_position_spherical_radius", sim.snap[idx].pType, units = "kpc")
    #allMass    = sp[("all","particle_mass")].in_units("Msun")
    #allR       = sp[("all","particle_position_spherical_radius")].in_units("kpc")
    log.logger.info(f"    Sorting particles by radius...")
    idx   = np.argsort(allR)
    mSort = np.array(allMass)[idx]
    rSort = np.array(allR)[idx]
    cumM  = np.cumsum(mSort)
    denR  = cumM/(4/3*np.pi*rSort**3) 

    idxAtVir = np.argmin(np.abs(denR-targetDen))
    log.logger.info(f"    Found rvir: {rSort[idxAtVir]:.3f} enclosing {cumM[idxAtVir]:.3E} Msun, with predicted {targetDen*(4/3*np.pi*rSort[idxAtVir]**3):.3E}")
    return rSort[idxAtVir]

# TODO: Experimental, need to check if faces detected are good
# TODO: Do full halo recognisition, requiring integration with Rockstar or Haskap Pie
# Calculates edge_on and face_on vectors from the total angular momentum
def getAxes(sim,idx,rvirRatio=0.15):
    log.logger.info(f"    Calculating face-on axis for snapshot at {idx} limited by sphere of radius {rvirRatio*sim.snap[idx].rvir} kpc")
    # Selects a well centered sphere upto rvirRatio*rvir (defaults to the 0.15 used in AGORA Paper VIII)
    sp = sim.ytFull[idx].sphere(sim.snap[idx].ytcen,(sim.snap[idx].rvir*rvirRatio,"kpc"))
    # Gets the total angular momentum from the gas or star particles
    part = "PartType0" if not "PartType4" in sim.snap[idx].pType else "PartType4"
    lx = sp[(part,"particle_angular_momentum_x")].sum()
    ly = sp[(part,"particle_angular_momentum_y")].sum()
    lz = sp[(part,"particle_angular_momentum_z")].sum()
    lMom = np.array([lx,ly,lz])
    face_on = lMom/np.linalg.norm(lMom)
    # Pick an arbitrary vector for edge_on calc
    z0 = np.array([0,0,1.0])
    if abs(np.dot(face_on, z0)) > 0.9:
        z0 = np.array([1.0,0,0])
    # Calculate cross to get edge_on vector
    edge_on = np.cross(face_on, z0)
    edge_on /= np.linalg.norm(edge_on)
    log.logger.info(f"    Calculated face-on axis {face_on} and edge-on axis {edge_on}")

    return (face_on,edge_on)